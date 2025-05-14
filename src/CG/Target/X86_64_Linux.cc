#include <srcc/CG/CodeGen.hh>

using namespace srcc;
using namespace srcc::cg;

namespace {
/// Implements platform-agnostic call lowering for the Source calling convention.
struct SourceLowering final : CallLowering {
    DenseMap<Type, StructLayout*> layouts;
    StructLayout* slice_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, Type::IntTy});
    StructLayout* closure_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, CG.tu.I8PtrTy});

    explicit SourceLowering(CodeGen& CG) : CallLowering(CG) {}

    auto adjust_procedure_type(ProcDecl* decl, ProcType* ty) -> std::pair<ProcType*, ir::Proc::ParamAttrMap> override;
    bool has_indirect_return(ProcType* ty) override;

    /// Compute a StructLayout for a type as it is split across registers; i.e.
    /// multiple fields in the actual type may correspond to a single field in
    /// the resulting layout and vice versa.
    auto get_register_layout(Type ty) -> StructLayout*;

    /// Determine whether this is an integer type that needs to be split across registers.
    bool is_splittable_int(Type ty) {
        if (not ty->is_integer()) return false;
        auto sz = ty->size(CG.tu);
        return sz.bits() > 64 and sz.bits() <= 128;
    }

    auto lower_call(
        ProcType* ty,
        ir::Aggregate* callee,
        ir::Value* mrvalue_slot,
        ArrayRef<ir::Value*> args
    ) -> ir::Value* override;
    void lower_params(ir::Proc* proc) override;
};

struct ARValueLayoutBuilder {
    static constexpr Size Word = Size::Bits(64);
    CodeGen& CG;
    SourceLowering& lowering;
    SmallVector<Type> types;
    Size rest;

    ARValueLayoutBuilder(CodeGen& CG, SourceLowering& lowering)
        : CG(CG), lowering(lowering) {}

    [[nodiscard]] auto build(Type ty) -> StructLayout*;

private:
    void add(Size sz);
    void process(Type ty);
};
} // namespace

auto CodeGen::CreateNativeCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering> {
    // TODO: Actual native lowering.
    return CreateSourceCallLowering_X86_64_Linux();
}

auto CodeGen::CreateSourceCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering> {
    return std::make_unique<SourceLowering>(*this);
}

void ARValueLayoutBuilder::add(Size sz) {
    Assert(sz <= Word, "Unsplit value");

    // Round up to a power of 2.
    sz = Size::Bits(std::bit_ceil(sz.bits()));

    // We have a rest. Append to it if we can.
    if (rest != Size()) {
        if (sz + rest <= Word) {
            rest += sz;
            return;
        }

        // Eject the rest as a single word.
        types.push_back(Type::IntTy);
        rest = Size();
    }

    // Add the new word.
    types.push_back(Type::IntTy);
}

auto ARValueLayoutBuilder::build(Type ty) -> StructLayout* {
    Assert(ty->size(CG.tu).bits() <= 128, "Too large to be passed in registers");
    process(ty);
    if (rest != Size()) types.push_back(IntType::Get(CG.tu, rest));
    auto layout = StructLayout::Create(CG.tu, types);
    Assert(layout->size().bits() <= 128, "Layout too large");
    return layout;
}

void ARValueLayoutBuilder::process(Type ty) {
    auto sz = ty->size(CG.tu);

    // i65–i128s are split into two registers.
    if (lowering.is_splittable_int(ty)) {
        add(Word);
        add(Word);
        return;
    }

    // Split ARValues into their constituent fields.
    //
    // We need to do this from scratch for every APValue as nesting may
    // cause them to be grouped differently (i.e. a struct may be split
    // differently just because it is inside another struct).
    if (ty->is_arvalue()) {
        // A struct that contains a closure or slice is larger than 128
        // bytes, but we can still get here because of zero-sized fields;
        //
        // TODO: However, in that case it’s guaranteed that there can be
        //   nothing else of interest in the struct, so just return the
        //   corresponding layout.
        if (isa<ProcType, SliceType>(ty)) {
            add(Word);
            add(Word);
            return;
        }

        if (auto r = dyn_cast<RangeType>(ty)) {
            process(r->elem());
            process(r->elem());
            return;
        }

        auto s = cast<StructType>(ty);
        for (auto f : s->layout()->fields()) process(f.ty);
        return;
    }

    // At this point, we have an SRValue; >i128 bit integers would make
    // this too large, so the only possibility is that this is <=64 bits.
    Assert(ty->is_srvalue());
    add(sz);
}

auto SourceLowering::adjust_procedure_type(
    ProcDecl* decl,
    ProcType* ty
) -> std::pair<ProcType*, ir::Proc::ParamAttrMap> {
    SmallVector<ParamTypeData> params;
    ir::Proc::ParamAttrMap attrs;
    Type ret = ty->ret();

    // Zero-sized return types must be converted to void; keep noreturn as
    // that needs to be lowered to an LLVM attribute.
    if (ret != Type::NoReturnTy and CG.IsZeroSizedType(ret))
        ret = Type::VoidTy;

    // The implicit return pointer is the first argument.
    if (has_indirect_return(ty)) {
        params.emplace_back(Intent::Copy, CG.tu.I8PtrTy);
        attrs[0].ty = ret;
        attrs[0].ll_sret = true;
        ret = Type::VoidTy;
    }

    // Adjust the parameters.
    for (auto [ty, intent] : ty->params()) {
        // Skip zero-sized parameters entirely.
        if (CG.IsZeroSizedType(ty)) continue;

        // Convert pass-by-reference parameters to pointers.
        if (ty->pass_by_reference(intent)) {
            params.emplace_back(Intent::Copy, PtrType::Get(CG.tu, ty));
            continue;
        }

        // Split aggregates.
        if (ty->is_arvalue()) {
            for (auto el : get_register_layout(ty)->fields())
                params.emplace_back(Intent::Copy, el.ty);
            continue;
        }

        // Zero-extend 'bool'.
        u32 idx = u32(params.size());
        if (ty == Type::BoolTy) attrs[idx].ll_zeroext = true;

        // Handle integers. This is a bit of a mess:
        //  - Integers < 32 bit are extended to 32 bit *by the backend*;
        //    add a 'signext' attribute at the LLVM IR level but keep the
        //    type the same.
        //  - 32 and 64 bit integers are unchanged.
        //  - Integers < 64 need to be extended to 64 *by us*.
        //  - Integers <=128 bit are split into two 64-bit registers.
        //  - Finally, anything larger than that is extended to a multiple
        //    of 64 bit and passed in memory.
        else if (ty->is_integer()) {
            auto sz = ty->size(CG.tu);
            if (sz.bits() < 32) attrs[idx].ll_signext = true;
            else if (sz.bits() == 32 or sz.bits() == 64) { /** Nothing. **/ } else if (sz.bits() < 64) {
                ty = CG.tu.I64Ty;
            } else if (sz.bits() <= 128) {
                params.emplace_back(Intent::Copy, CG.tu.I64Ty);
                params.emplace_back(Intent::Copy, CG.tu.I64Ty);
                continue;
            } else {
                auto byval_ty = IntType::Get(CG.tu, sz.align(Align(8)));
                ty = PtrType::Get(CG.tu, byval_ty);
                attrs[idx].ty  = byval_ty;
                attrs[idx].ll_byval = true;
            }
        }

        // Everything else can stay the same.
        params.emplace_back(Intent::Copy, ty);
    }

    return {ProcType::Get(CG.tu, ret, params, ty->cconv(), ty->variadic()), std::move(attrs)};
}

auto SourceLowering::get_register_layout(Type ty) -> StructLayout* {
    Assert(ty->is_arvalue() or is_splittable_int(ty));
    if (isa<ProcType>(ty)) return closure_layout;
    if (isa<SliceType>(ty)) return slice_layout;
    auto& layout = layouts[ty];
    if (not layout) layout = ARValueLayoutBuilder(CG, *this).build(ty);
    return layout;
}

void SourceLowering::lower_params(ir::Proc* proc) {
    SmallVector<ir::Value*> temp_vec;
    auto args = proc->args();
    usz i = proc->has_indirect_return() ? 1 : 0;
    for (auto p : proc->decl()->params()) {
        // Skip zero-sized types.
        if (CG.IsZeroSizedType(p->type)) continue;

        // This is a by-reference parameter; use it directly.
        if (p->type->pass_by_reference(p->intent())) {
            CG.locals[p] = args[i++];
            continue;
        }

        // Otherwise, we have an ARValue or SRValue.
        auto rvalue = [&] -> ir::Value* {
            // If this is <=64 bits, it must be an SRValue; just get it
            // directly (we may have to truncate it to its actual size
            // though).
            auto sz = p->type->size(CG.tu);
            if (sz.bits() <= 64) {
                auto arg = args[i++];
                if (sz == arg->type()->size(CG.tu)) return arg;
                return CG.CreateSICast(arg, p->type);
            }

            // This is an in-memory SRValue integer.
            if (sz.bits() > 128) {
                Assert(p->type->is_integer());
                return CG.CreateLoad(p->type, args[i++]);
            }

            // If this is an ARValue or <=128 bit int; create an aggregate from
            // the registers this was split across; store it to memory, and then
            // load it back as the original type.
            Assert(p->type->is_arvalue() or is_splittable_int(p->type));
            auto layout = get_register_layout(p->type);
            auto temp = CG.CreateAlloca(proc, layout->size(), std::max(layout->align(), p->type->align(CG.tu)));
            temp_vec.clear();
            append_range(temp_vec, args.drop_front(i).take_front(layout->fields().size()));
            CG.StoreAggregate(temp, layout, temp_vec);
            i += layout->fields().size();

            // We should be able to avoid a memcpy() into a second temporary
            // in all cases, but check just in case.
            Assert(sz <= layout->size());
            return CG.CreateLoad(p->type, temp);
        }();

        // If this is an 'in' parameter, just use the value directly.
        if (p->intent() == Intent::In) {
            CG.locals[p] = rvalue;
            continue;
        }

        // Otherwise, create a local variable for it and initialise it with the rvalue.
        auto a = CG.locals[p] = CG.CreateAlloca(proc, p->type);
        CG.CreateStore(p->type, rvalue, a);
    }

    // We should have nothing left at this point.
    Assert(i == args.size(), "All IR parameters should have been processed");
}

bool SourceLowering::has_indirect_return(ProcType* ty) {
    if (CG.IsZeroSizedType(ty)) return false;
    return ty->ret()->rvalue_category() == ValueCategory::MRValue;
}

auto SourceLowering::lower_call(
    ProcType* proc_type,
    ir::Aggregate* callee,
    ir::Value* mrvalue_slot,
    ArrayRef<ir::Value*> args
) -> ir::Value* {
    SmallVector<ir::Value*> lowered;

    // Add the pointer for the return value if there is one.
    DebugAssert((mrvalue_slot != nullptr) == has_indirect_return(proc_type));
    if (mrvalue_slot) lowered.push_back(mrvalue_slot);

    // TODO: Once we have actual closures, the closure pointer needs to come after that.

    // This loop essentially either does the opposite of or complements
    // lower_params() above, except that we no longer have to deal with
    // intents or lvalues here since those are just pointers here.
    for (auto [arg, p] : zip(args, proc_type->params())) {
        auto ty = p.type;

        // Skip zero-sized args.
        if (CG.IsZeroSizedType(ty)) continue;

        // Handle weird integer cases.
        if (ty->is_integer()) {
            // If this is a 33-64 bit integer, extend it to 64 bits.
            auto sz = ty->size(CG.tu);
            if (ty->is_srvalue() and sz.bits() > 32 and sz.bits() < 64) {
                lowered.push_back(CG.CreateSICast(arg, CG.tu.I64Ty));
                continue;
            }

            // If this is an integer that must be passed in memory, write a
            // copy to the stack and pass a pointer to it.
            if (sz.bits() > 128) {
                auto byval = CG.CreateAlloca(CG.curr_proc, ty);
                CG.CreateStore(ty, arg, byval);
                lowered.push_back(byval);
                continue;
            }
        }

        // If this is an ARValue or <=128 bit int, split it by storing
        // it to memory and loading an Aggregate* back, except that the
        // latter is split properly across registers.
        if (ty->is_arvalue() or is_splittable_int(ty)) {
            auto layout = get_register_layout(ty);
            auto temp = CG.CreateAlloca(CG.curr_proc, layout->size(), std::max(ty->align(CG.tu), layout->align()));
            CG.CreateStore(ty, temp, arg);
            for (auto v : cast<ir::Aggregate>(CG.LoadAggregate(layout, temp))->fields()) lowered.push_back(v);
            continue;
        }

        // Finally, if we get here, just pass along the argument as-is.
        lowered.push_back(arg);
    }

    return CG.CreateCall(callee->field(0), lowered);
}
