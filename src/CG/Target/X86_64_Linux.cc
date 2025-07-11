#include <srcc/CG/CodeGen.hh>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOpsDialect.h.inc>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#if 0
using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;
namespace srcc::cg::x64_64_linux_abi {}
using namespace srcc::cg::x64_64_linux_abi;

namespace srcc::cg::x64_64_linux_abi {
struct LoweringPass :
    mlir::PassWrapper<LoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override;
};

struct ProcOpLowering : mlir::RewritePattern {
    ProcOpLowering(mlir::MLIRContext *context)
        :  RewritePattern(ProcOp::getOperationName(), 1, context) {}

    auto matchAndRewrite(
        Operation* op,
        mlir::PatternRewriter& rewriter
    ) const -> llvm::LogicalResult override;
};

/// How an argument is passed to a function; used by codegen and below.
enum class ArgumentPassingMode : u8 {
    /// This parameter should be dropped entirely.
    Drop,

    /// Pass the value as-is in a single GPR.
    Register,

    /// Pass the value split across 2 registers if we have enough GPRs left.
    RegisterPair,

    /// Pass the value in memory.
    Memory,

    /// Pass the value in a register, and let the backend zero-extend it.
    ZeroExt,

    /// Pass the value in a register, and let the backend sign-extend it.
    SignExt,

    /// Pass the value in a register, but sign extend it manually.
    SignExtManual,
};

class GPRUsageTracker {
    static constexpr u32 MaxGPRs = 6;
    u32 gprs_in_use = 0;

public:
    void add_indirect_return_pointer() { gprs_in_use++; }
    void add(ArgumentPassingMode m) {
        switch (m) {
            case ArgumentPassingMode::Register:
            case ArgumentPassingMode::ZeroExt:
            case ArgumentPassingMode::SignExt:
            case ArgumentPassingMode::SignExtManual:
                gprs_in_use++;

            case ArgumentPassingMode::RegisterPair:
                gprs_in_use += 2;

            case ArgumentPassingMode::Drop:
            case ArgumentPassingMode::Memory:
                break;
        }
    }

    bool can_pass_register_pair() {
        return gprs_in_use + 2 <= MaxGPRs;
    }
};

struct Lowering {
    TranslationUnit& tu;
    auto ClassifyParameter(mlir::Type ty, GPRUsageTracker* gprs) const -> ArgumentPassingMode;
    auto GetValueTypeSize(mlir::Type ty) const -> Size;
    bool HasIndirectReturn(ir::ProcType ty) const;
    void LowerProcOp(ProcOp op) const;
    bool NeedsLowering(ProcOp op) const;
};
}

auto Lowering::ClassifyParameter(mlir::Type ty, GPRUsageTracker* gprs) const -> ArgumentPassingMode {
    auto sz = GetValueTypeSize(ty);
    if (sz == Size()) return ArgumentPassingMode::Drop;

    // Anything larger than 128 bits is passed in memory, i.e. no register
    // is allocated for them. Integer types larger than 128 bits are aligned
    // to a multiple of 64 bits.
    if (sz > Size::Bits(128)) return ArgumentPassingMode::Memory;

    // If we get here, we have something that is at most 128 bits, and
    // we also have enough GPRs left to pass it in registers.
    //
    // This is because the only types that can have class MEMORY are
    // arrays and structs >128 bits, in which case we never get here.
    // In other words, we already know that this type has class INTEGER,
    // the only question is whether we need to split it into one or two
    // registers.
    //
    // TODO: Once we support floating point types, we’ll have to deal
    // with classes other than INTEGER and MEMORY.
    if (sz > Size::Bits(64)) {
        if (gprs and gprs->can_pass_register_pair()) return ArgumentPassingMode::Memory;
        return ArgumentPassingMode::RegisterPair;
    }

    // Zero-extend 'bool'.
    if (sz.bits() == 1 and cast<mlir::IntegerType>(ty).isUnsigned())
        return ArgumentPassingMode::ZeroExt;

    // Integers are a bit of a mess:
    //  - Integers < 32 bit are extended to 32 bit *by the backend*;
    //    add a 'signext' attribute at the LLVM IR level but keep the
    //    type the same.
    //  - 32 and 64 bit integers are unchanged.
    //  - Integers < 64 need to be extended to 64 *by us*.
    if (isa<mlir::IntegerType>(ty)) {
        if (sz.bits() < 32) return ArgumentPassingMode::SignExt;
        if (sz.bits() != 32 and sz.bits() != 64) return ArgumentPassingMode::Register;
        return ArgumentPassingMode::SignExtManual;
    }

    // Any other type (e.g. pointers) is integer-sized.
    return ArgumentPassingMode::Register;
}

auto Lowering::GetValueTypeSize(mlir::Type ty) const -> Size {
    if (isa<mlir::LLVM::LLVMPointerType>(ty)) return Size::Bytes(8);
    if (isa<mlir::NoneType>(ty)) return Size();
    if (auto i = dyn_cast<mlir::IntegerType>(ty)) return Size::Bits(i.getWidth());
    if (auto a = dyn_cast<mlir::LLVM::LLVMArrayType>(ty)) {
        Assert(a.getElementType().isInteger(8));
        return Size::Bytes(a.getNumElements());
    }

    // Tuples are only used to represent builtin types, so if we get here,
    // we have a slice/range/closure. Both closures and slices are 128 bits.
    auto t = cast<mlir::TupleType>(ty);
    if (isa<mlir::LLVM::LLVMPointerType>(t.getType(0))) return Size::Bits(128);

    // If the first parameter is not a pointer, then we have a range. If the
    // integer type is >64 bits, align it.
    auto i = cast<mlir::IntegerType>(t.getType(0));
    auto align = tu.target().int_align(Size::Bits(i.getWidth()));
    auto size = tu.target().int_size(Size::Bits(i.getWidth()));
    return size.align(align) + size;
}

bool Lowering::HasIndirectReturn(ir::ProcType ty) const {
    GPRUsageTracker gprs;
    return ty.getIndirectReturn() or
           ClassifyParameter(ty.getReturnType(), &gprs) == ArgumentPassingMode::Memory;
}


void Lowering::LowerProcOp(ProcOp proc) const {
    SmallVector<mlir::Type> params;
    SmallVector<Value> param_vals;
    GPRUsageTracker gprs;
    std::unique_ptr<Block> new_entry;
    auto proc_ty = proc.getProcType();
    auto ret = proc_ty.getReturnType();
    auto ctx = proc.getContext();
    bool has_indirect_return = HasIndirectReturn(proc_ty);

    // If this procedure has a body, build a new entry block.
    if (not proc.getBody().empty()) {
        new_entry.reset(new Block);
        auto old_entry = &proc.getBody().front();

        // Move all allocas into the new entry block.
        while (not old_entry->empty() and isa<AllocaOp>(old_entry->front()))
            old_entry->front().moveBefore(new_entry.get(), new_entry->end());
    }



    // TODO: Lower parameters: insert code for that into the new block, branch
    // to the old entry block, and then merge the two blocks.





    // The implicit return pointer is the first argument.
    if (has_indirect_return) {
        params.push_back(mlir::LLVM::LLVMPointerType::get(proc_ty.getContext()));
        proc.setArgAttr(0, mlir::LLVM::LLVMDialect::getStructRetAttrName(), mlir::TypeAttr::get(ret));
        ret = mlir::NoneType::get(ctx);
        gprs.add_indirect_return_pointer();
    }

    // Adjust the parameters.
    for (auto ty : proc_ty.getParams()) {
        auto m = ClassifyParameter(ty, &gprs);
        gprs.add(m);
        switch (m) {
            case ArgumentPassingMode::Drop:
                break;

            case ArgumentPassingMode::Register:
                params.push_back(ty);
                break;

            case ArgumentPassingMode::SignExtManual:
                params.push_back(mlir::IntegerType::get(ctx, 64));
                break;

            case ArgumentPassingMode::RegisterPair:
                params.push_back(mlir::IntegerType::get(ctx, 64));
                params.push_back(mlir::IntegerType::get(ctx, 64));
                break;

            case ArgumentPassingMode::Memory:
                ty = ty.isInteger()
                    ? mlir::IntegerType::get(ctx, Size::Bits(ty.getIntOrFloatBitWidth()).align(Align(8)).bits())
                    : ty;

                proc.setArgAttr(u32(params.size()), mlir::LLVM::LLVMDialect::getByValAttrName(), mlir::TypeAttr::get(ty));
                params.push_back(mlir::LLVM::LLVMPointerType::get(proc_ty.getContext()));
                break;

            case ArgumentPassingMode::ZeroExt:
                proc.setArgAttr(u32(params.size()), mlir::LLVM::LLVMDialect::getZExtAttrName(), mlir::UnitAttr::get(ctx));
                params.push_back(ty);
                break;

            case ArgumentPassingMode::SignExt:
                proc.setArgAttr(u32(params.size()), mlir::LLVM::LLVMDialect::getSExtAttrName(), mlir::UnitAttr::get(ctx));
                params.push_back(ty);
                break;
        }
    }


    // Overwrite the procedure type.
    auto adjusted_type = ir::ProcType::get(
        proc_ty.getCc(),
        ret,
        params,
        proc_ty.getVariadic(),
        has_indirect_return
    );

    proc.setProcType(adjusted_type);

    // Adjust all
}

bool Lowering::NeedsLowering(ProcOp op) const {
    GPRUsageTracker gprs;
    auto ty = op.getProcType();
    if (HasIndirectReturn(ty)) return true;
    for (auto p : ty.getParams()) {
        auto m = ClassifyParameter(p, &gprs);
        gprs.add(m);
        switch (m) {
            case ArgumentPassingMode::Drop:
            case ArgumentPassingMode::Register:
                break;
            case ArgumentPassingMode::RegisterPair:
            case ArgumentPassingMode::Memory:
            case ArgumentPassingMode::ZeroExt:
            case ArgumentPassingMode::SignExt:
            case ArgumentPassingMode::SignExtManual:
                return true;
        }
    }
    return false;
}


auto ProcOpLowering::matchAndRewrite(
    Operation* op_ptr,
    mlir::PatternRewriter& rewriter
) const -> llvm::LogicalResult {
    auto proc = cast<ProcOp>(op_ptr);
}


void LoweringPass::runOnOperation() {
    mlir::ConversionTarget tgt{getContext()};
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<ProcOpLowering>(getContext());
}

#endif


/*using namespace srcc;
using namespace srcc::cg;

namespace {
/// Implements platform-agnostic call lowering for the Source calling convention.
struct SourceLowering final : CallLowering {
    DenseMap<Type, StructLayout*> layouts;
    StructLayout* slice_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, Type::IntTy});
    StructLayout* closure_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, CG.tu.I8PtrTy});

    explicit SourceLowering(CodeGen& CG) : CallLowering(CG) {}

    auto adjust_procedure_type(ProcType* ty) -> std::pair<ProcType*, ir::Proc::ParamAttrMap> override;
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
        for (auto f : s->layout()->fields()) process(f);
        return;
    }

    // At this point, we have an SRValue; >i128 bit integers would make
    // this too large, so the only possibility is that this is <=64 bits.
    Assert(ty->is_srvalue());
    add(sz);
}

auto SourceLowering::adjust_procedure_type(ProcType* ty) -> std::pair<ProcType*, ir::Proc::ParamAttrMap> {
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
                params.emplace_back(Intent::Copy, el);
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
            else if (sz.bits() == 32 or sz.bits() == 64) { /** Nothing. *#1# } else if (sz.bits() < 64) {
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

            // We should be able to avoid a memcpy() into a second temporary
            // in all cases, but check just in case.
            Assert(sz <= layout->size());
            temp_vec.clear();
            append_range(temp_vec, args.drop_front(i).take_front(layout->fields().size()));
            i += layout->fields().size();
            return AssembleTypeFromRegisters(p->type, layout, temp_vec);
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
    bool indirect_return = has_indirect_return(proc_type);
    DebugAssert((mrvalue_slot != nullptr) == indirect_return);
    if (mrvalue_slot) lowered.push_back(mrvalue_slot);

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
            // For things that are basically two pointers, just forward them.
            if (isa<ProcType, SliceType>(ty)) {
                auto a = cast<ir::Aggregate>(arg);
                lowered.push_back(a->field(0));
                lowered.push_back(a->field(1));
                continue;
            }

            // Otherwise, convert the object in memory.
            auto layout = get_register_layout(ty);
            auto temp = CG.CreateAlloca(CG.curr_proc, layout->size(), std::max(ty->align(CG.tu), layout->align()));
            CG.CreateStore(ty, arg, temp);
            for (auto v : cast<ir::Aggregate>(CG.LoadAggregate(layout, temp))->fields()) lowered.push_back(v);
            continue;
        }

        // Finally, if we get here, just pass along the argument as-is.
        lowered.push_back(arg);
    }

    // If this is an indirect call, lower the procedure type.
    auto direct = dyn_cast<ir::Proc>(callee->field(0));
    auto adjusted = direct ? direct->proc_type() : adjust_procedure_type(proc_type).first;
    auto proc = callee->field(0);
    auto ret = proc_type->ret();

    // Simple value.
    if (not ret->is_arvalue()) return CG.CreateCall(
        adjusted,
        proc,
        lowered,
        ret,
        callee->field(1)
    );

    // Aggregate value.
    auto ret_layout = get_register_layout(ret);
    auto call = CG.CreateCall(
        adjusted,
        proc,
        lowered,
        ret_layout->fields(),
        callee->field(1)
    );

    return AssembleTypeFromRegisters(
        ret,
        ret_layout,
        cast<ir::Aggregate>(call)->fields()
    );
}*/


/*
#include <srcc/CG/CodeGen.hh>

using namespace srcc;
using namespace srcc::cg;

using ir::Ty;

namespace {

class GPRUsageTracker {
    static constexpr u32 MaxGPRs = 6;
    u32 gprs_in_use = 0;

public:
    void add_indirect_return_pointer() { gprs_in_use++; }
    void add(ArgumentPassingMode m) {
        switch (m) {
            case ArgumentPassingMode::Register:
            case ArgumentPassingMode::ZeroExt:
            case ArgumentPassingMode::SignExt:
            case ArgumentPassingMode::SignExtManual:
            case ArgumentPassingMode::Pointer:
                gprs_in_use++;

            case ArgumentPassingMode::RegisterPair:
                gprs_in_use += 2;

            case ArgumentPassingMode::Memory:
                break;
        }
    }

    bool can_pass_register_pair() {
        return gprs_in_use + 2 <= MaxGPRs;
    }
};

/// Implements platform-agnostic call lowering for the Source calling convention.
struct SourceLowering final : CallLowering {
    DenseMap<Type, StructLayout*> layouts;
    StructLayout* slice_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, Type::IntTy});
    StructLayout* closure_layout = StructLayout::Create(CG.tu, {CG.tu.I8PtrTy, CG.tu.I8PtrTy});

    explicit SourceLowering(CodeGen& CG) : CallLowering(CG) {}

    /// Compute a StructLayout for a type as it is split across registers; i.e.
    /// multiple fields in the actual type may correspond to a single field in
    /// the resulting layout and vice versa.
    auto get_register_layout(Type ty) -> StructLayout*;

    auto lower_call(
        ProcType* ty,
        ir::Value* callee,
        Ptr<ir::Value> env,
        ir::Value* mrvalue_slot,
        ArrayRef<ir::Value*> args
    ) -> Value override;

    void lower_params(ir::Proc* proc) override;
    auto lower_proc_type(ProcType* ty) -> ir::ProcTy* override;

    /// Create an instance of a type from a set of values as passed or returned in
    /// registers and a layout representing them.
    [[nodiscard]] auto AssembleTypeFromRegisters(
        Type ty,
        StructLayout* register_layout,
        ArrayRef<ir::Value*> register_vals
    ) -> ir::Value*;

    /// Classify a parameter to determine how it should be passed.
    auto ClassifyParameter(Type ty, Intent i, GPRUsageTracker* gprs) -> ArgumentPassingMode;
};
} // namespace

auto CodeGen::CreateNativeCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering> {
    // TODO: Actual native lowering.
    return CreateSourceCallLowering_X86_64_Linux();
}

auto CodeGen::CreateSourceCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering> {
    return std::make_unique<SourceLowering>(*this);
}

auto SourceLowering::lower_proc_type(ProcType* ty) -> ir::ProcTy* {
    Ty* ret = CG.Convert(ty->ret());
    SmallVector<Ty*> params;
    ir::ProcTy::ParamAttrMap attrs;
    GPRUsageTracker gprs;

    // The implicit return pointer is the first argument.
    if (ClassifyParameter(ty, Intent::Copy, &gprs) == ArgumentPassingMode::Memory) {
        params.push_back(CG.ptr_ty);
        attrs[0].ty = ret;
        attrs[0].ll_sret = true;
        ret = CG.void_ty;
        gprs.add_indirect_return_pointer();
    }

    // Adjust the parameters.
    for (auto [ty, intent] : ty->params()) {
        if (CG.IsZeroSizedType(ty)) continue;
        auto m = ClassifyParameter(ty, intent, &gprs);
        gprs.add(m);
        switch (m) {
            case ArgumentPassingMode::Pointer:
                params.push_back(CG.ptr_ty);
                break;

            case ArgumentPassingMode::Register:
            case ArgumentPassingMode::SignExtManual:
                params.push_back(CG.Convert(ty));
                break;

            case ArgumentPassingMode::RegisterPair: {
                params.push_back(CG.i64_ty);
                params.push_back(CG.i64_ty);
            } break;

            case ArgumentPassingMode::Memory: {
                u32 idx = u32(params.size());
                attrs[idx].ty = ty->is_integer() ? ir::IntTy::Get(CG, ty->size(CG.tu).align(Align(8))) : CG.Convert(ty);
                attrs[idx].ll_byval = true;
                params.push_back(CG.ptr_ty);
            } break;

            case ArgumentPassingMode::ZeroExt: {
                attrs[params.size()].ll_zeroext = true;
                params.push_back(CG.Convert(ty));
            } break;

            case ArgumentPassingMode::SignExt: {
                attrs[params.size()].ll_signext = true;
                params.push_back(CG.Convert(ty));
            } break;
        }
    }

    return ir::ProcTy::Get(CG, ret, params, std::move(attrs));
}

void SourceLowering::lower_params(ir::Proc* proc) {
    SmallVector<ir::Value*> temp_vec;
    GPRUsageTracker gprs;
    auto args = proc->args();
    usz i = proc->has_indirect_return() ? 1 : 0;
    if (proc->has_indirect_return()) gprs.add_indirect_return_pointer();
    for (auto p : proc->decl()->params()) {
        // Skip zero-sized types.
        if (CG.IsZeroSizedType(p->type)) continue;

        // This is a by-reference parameter; use it directly.
        auto cls = ClassifyParameter(p->type, p->intent(), &gprs);
        if (cls == ArgumentPassingMode::Pointer) {
            CG.locals[p] = args[i++];
            continue;
        }

        // Otherwise, we have an ARValue or SRValue.
        Assert(p->type->is_srvalue());
        auto rvalue = [&] -> Value {
            switch (cls) {
                case ArgumentPassingMode::Pointer: Unreachable();
                case ArgumentPassingMode::Register:
                    return args[i++];

                case ArgumentPassingMode::RegisterPair: {
                    ir::Value* v1 = args[i];
                    ir::Value* v2 = args[i + 1];

                    // If we’re passing a register pair in a range, we may have to sign
                    // extend the elements manually.
                    if (
                        auto rng = dyn_cast<RangeType>(p->type);
                        rng and ClassifyParameter(rng->elem(), Intent::Copy, nullptr) == ArgumentPassingMode::SignExtManual
                    ) {
                        auto ty = CG.Convert(rng->elem());
                        v1 = CG.CreateSICast(v1, ty);
                        v2 = CG.CreateSICast(v1, ty);
                    }

                    i += 2;
                    return {v1, v2};
                }

                // This should only happen for RegisterPairs if we run out
                // of GPRs. Actual MRValues are never passed ‘by value’ at
                // the *AST* level.
                case ArgumentPassingMode::Memory: {
                    if (not CG.IsARValue(p->type)) return CG.CreateLoad(
                        CG.Convert(p->type),
                        args[i++],
                        p->type->align(CG.tu)
                    );

                    auto [first, second] = CG.GetARValueTypes(p->type);
                    auto v1 = CG.CreateLoad(CG.Convert(first), args[i], first->align(CG.tu));
                    auto v2 = CG.CreateLoad(CG.Convert(second), CG.CreatePtrAdd(args[i], first->size(CG.tu)), second->align(CG.tu));
                    i += 2;
                    return {v1, v2};
                }

                case ArgumentPassingMode::ZeroExt:
                case ArgumentPassingMode::SignExt:
                    return {args[i++]};

                case ArgumentPassingMode::SignExtManual:
                    return CG.CreateSICast(args[i++], CG.Convert(p->type));
            }

            Unreachable();
        }();

        // If this is an 'in' parameter, just use the value directly.
        if (p->intent() == Intent::In) {
            CG.locals[p] = rvalue;
            continue;
        }








        // LEFT OFF HERE.
        //
        // TODO: Reintroduce `ir::Aggregate`, but as an *instruction* with
        //       multiple result values, as well as `ir::ExtractValueInst`;
        //       then, lower both to scalar operations in a separate pass
        //       after CodeGen (and *before* constant evaluation).










        // Otherwise, create a local variable for it and initialise it with the rvalue.
        auto a = CG.locals[p] = CG.CreateAlloca(proc, CG.Convert(p->type), p->type->align(CG.tu));
        CG.EmitInitialiser()
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
    ir::Value* callee,
    Ptr<ir::Value> env,
    ir::Value* mrvalue_slot,
    ArrayRef<ir::Value*> args
) -> Value {
    SmallVector<ir::Value*> lowered;

    // Add the pointer for the return value if there is one.
    bool indirect_return = has_indirect_return(proc_type);
    DebugAssert((mrvalue_slot != nullptr) == indirect_return);
    if (mrvalue_slot) lowered.push_back(mrvalue_slot);

    // This loop essentially either does the opposite of or complements
    // lower_params() above, except that we no longer have to deal with
    // intents or lvalues here since those are just pointers here.
    auto it = args.begin();
    for (const auto& p : proc_type->params()) {
        auto ty = p.type;

        // Skip zero-sized args.
        if (CG.IsZeroSizedType(ty)) continue;

        // Handle weird integer cases.
        if (ty->is_integer()) {
            // If this is a 33-64 bit integer, extend it to 64 bits.
            auto sz = ty->size(CG.tu);
            if (ty->is_srvalue() and sz.bits() > 32 and sz.bits() < 64) {
                lowered.push_back(CG.CreateSICast(*it++, CG.tu.I64Ty));
                continue;
            }

            // If this is an integer that must be passed in memory, write a
            // copy to the stack and pass a pointer to it.
            if (sz.bits() > 128) {
                auto byval = CG.CreateAlloca(CG.curr_proc, ty);
                CG.CreateStore(ty, *it++, byval);
                lowered.push_back(byval);
                continue;
            }
        }

        // If this is an ARValue or <=128 bit int, split it by storing
        // it to memory and loading an Aggregate* back, except that the
        // latter is split properly across registers.
        if (ty->is_arvalue() or is_splittable_int(ty)) {
            // For things that are basically two pointers, just forward them.
            if (isa<ProcType, SliceType>(ty)) {
                lowered.push_back(*it++);
                lowered.push_back(*it++);
                continue;
            }

            // Otherwise, convert the object in memory.
            auto layout = get_register_layout(ty);
            auto temp = CG.CreateAlloca(CG.curr_proc, layout->size(), std::max(ty->align(CG.tu), layout->align()));
            CG.CreateStore(ty, *it++, temp);
            lowered.push_back(CG.CreateLoad(layout->field(0), temp));
            lowered.push_back(CG.CreateLoad(layout->field(0), CG.CreatePtrAdd(temp, Size::Bytes(8))));
            continue;
        }

        // Finally, if we get here, just pass along the argument as-is.
        lowered.push_back(*it++);
    }

    // If this is an indirect call, lower the procedure type.
    auto direct = dyn_cast<ir::Proc>(callee);
    auto adjusted = direct ? direct->proc_type() : adjust_procedure_type(proc_type).first;
    auto ret = adjusted->ret();

    // Simple value.
    if (not ret->is_arvalue()) {
        auto call = CG.CreateCallImpl(adjusted, callee, lowered, ret, env);
        if (CG.IsZeroSizedType(ret)) return {};
        return new (CG) ir::InstValue(call, ret, 0);
    }

    // Aggregate value. We need to reassemble it.
    auto ret_layout = get_register_layout(ret);
    auto call = CG.CreateCallImpl(adjusted, callee, lowered, ret_layout->fields(), env);
    auto v1 = new (CG) ir::InstValue(call, ret, 0);
    auto v2 = new (CG) ir::InstValue(call, ret, 1);
}*/
