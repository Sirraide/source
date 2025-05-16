#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>
#include <srcc/Core/Constants.hh>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

#define TRY_FOLD_INT(...)                                                                \
    do {                                                                                 \
        if (auto v = Fold(a, b, [&](u64 lhs, u64 rhs) { return __VA_ARGS__; })) return v; \
    } while (false)

auto Managed::operator new(usz sz, Builder& b) -> void* {
    return b.Allocate(sz, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

Builder::Builder(TranslationUnit& tu) : tu{tu} {
    closure_layout = StructLayout::Create(tu, {tu.I8PtrTy, tu.I8PtrTy});
}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateImpl(Args&&... args) -> InstTy* {
    Assert(insert_point, "No insert point");
    auto i = new (*this) InstTy(*this, LIBBASE_FWD(args)...);
    insert_point->insts.push_back(i);
    return i;
}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateSpecialGetVal(Type val_ty, Args&&... args) -> Value* {
    return new (*this) InstValue(CreateImpl<InstTy>(LIBBASE_FWD(args)...), val_ty, 0);
}

auto Builder::Create(Op op, ArrayRef<Value*> vals) -> Inst* {
    return CreateImpl<Inst>(op, vals);
}

auto Builder::CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> InstValue* {
    return new (*this) InstValue(Create(op, vals), ty, 0);
}

void Builder::CreateAbort(AbortReason reason, Location loc, Aggregate* msg1, Aggregate* msg2) {
    CreateImpl<AbortInst>(reason, loc, ArrayRef{msg1->field(0), msg1->field(1), msg2->field(0), msg2->field(1)});
}

auto Builder::CreateAdd(Value* a, Value* b, bool nowrap) -> Value* {
    TRY_FOLD_INT(lhs + rhs);
    auto add = CreateAndGetVal(Op::Add, a->type(), {a, b});
    add->inst()->nowrap = nowrap;
    return add;
}

auto Builder::CreateAggregate(Type ty, StructLayout* layout, ArrayRef<Value*> vals) -> Aggregate* {
    Assert(layout->fields().size() == vals.size(), "Invalid layout for aggregate");
    if (ty == Type()) ty = IRAggregateType::Get(tu, layout);
    auto size = Aggregate::totalSizeToAlloc<Value*>(vals.size());
    auto mem = Allocate(size, alignof(Aggregate));
    auto a = ::new (mem) Aggregate(ty, layout);
    std::uninitialized_copy(vals.begin(), vals.end(), a->getTrailingObjects());
    return a;
}

auto Builder::CreateAlloca(Proc* parent, Type ty, Align align) -> Value* {
    Assert(parent, "Alloca without parent procedure?");
    if (align == Align()) align = ty->align(tu);
    auto a = new (*this) FrameSlot(tu, parent, ty, align);
    parent->frame_info.push_back(a);
    return a;
}

auto Builder::CreateAlloca(Proc* parent, Size sz, Align align) -> Value* {
    return CreateAlloca(parent, ArrayType::Get(tu, tu.I8Ty, i64(sz.bytes())), align);
}

auto Builder::CreateAnd(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::And, a->type(), {a, b});
}

auto Builder::CreateAShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::AShr, a->type(), {a, b});
}

auto Builder::CreateBlock(ArrayRef<Value*> args) -> std::unique_ptr<Block> {
    SmallVector<Type, 3> types;
    for (auto arg : args) types.push_back(arg->type());
    return CreateBlock(types);
}

auto Builder::CreateBlock(ArrayRef<Type> args) -> std::unique_ptr<Block> {
    u32 i = 0;
    auto b = std::unique_ptr<Block>(new Block);
    for (auto ty : args) {
        if (not ty->is_arvalue()) {
            b->split_args.push_back(new (*this) Argument(b.get(), u32(i++), ty));
            b->logical_args.push_back(b->split_args.back());
            continue;
        }

        SmallVector<Value*> vals;
        auto layout = GetAggregateLayout(ty);
        for (auto f : layout->fields()) {
            b->split_args.push_back(new (*this) Argument(b.get(), u32(i++), f));
            vals.push_back(b->split_args.back());
        }

        b->logical_args.push_back(CreateAggregate(ty, layout, vals));
    }

    return b;
}

auto Builder::CreateBlock(Proc* proc, ArrayRef<Type> args) -> Block* {
    return proc->add(CreateBlock(args));
}

auto Builder::CreateBool(bool value) -> Value* {
    return &(value ? true_val : false_val);
}

void Builder::CreateBr(Block* dest, ArrayRef<Value*> args) {
    BranchTarget tgt{dest, args};
    CreateBranchInst(nullptr, tgt, BranchTarget{});
}

void Builder::CreateBranchInst(Value* cond, BranchTarget then, BranchTarget else_) {
    SmallVector<Value*, 16> args{};
    auto Append = [&](Value* v) {
        if (auto a = dyn_cast<Aggregate>(v)) append_range(args, a->fields());
        else args.push_back(v);
    };

    // Add the condition; this is always a single value.
    if (cond) args.push_back(cond);

    // Add the then block args.
    for (auto v : then.args) Append(v);
    u32 then_args = u32(args.size() - (cond ? 1 : 0));

    // Add the else block args.
    for (auto v : else_.args) Append(v);

    // Build the branch.
    CreateImpl<BranchInst>(then_args, args, then.dest, else_.dest);
}

auto Builder::CreateCall(Proc* callee, ArrayRef<Value*> args) -> Value* {
    return CreateCall(
        callee->proc_type(),
        callee,
        args,
        callee->proc_type()->ret(),
        nullptr
    );
}

auto Builder::CreateCall(
    ProcType* proc,
    Value* callee,
    ArrayRef<Value*> args,
    ArrayRef<Type> results,
    Ptr<Value> env
) -> Value* {
    Assert(
        results.size() != 0,
        "Call that returns nothing should return void instead"
    );

    SmallVector operands{callee};
    append_range(operands, args);
    auto Create = [&](Type ret) {
        return CreateImpl<CallInst>(proc, ret, operands, env);
    };

    // Drop the environment if we know for sure that it’s nil.
    if (auto e = env.get_or_null(); e and e->is_nil()) env = nullptr;

    // This call returns a single value.
    if (results.size() == 1) {
        auto ty = results.front();
        if (ty == Type::VoidTy or ty == Type::NoReturnTy) {
            Create(Type());
            return nullptr;
        }

        return new (*this) InstValue(Create(ty), ty, 0);
    }

    // This call returns an aggregate.
    auto ty = IRAggregateType::Get(tu, StructLayout::Create(tu, results));
    auto call = Create(ty);
    operands.clear();
    for (auto [i, v] : enumerate(ty->layout()->fields()))
        operands.push_back(new (*this) InstValue(call, v, u32(i)));
    return CreateAggregate(ty, ty->layout(), operands);
}

auto Builder::CreateClosure(Proc* proc, Value* env) -> Aggregate* {
    if (not env) env = CreateNil(tu.I8PtrTy);
    auto ty = proc->proc_type();
    return CreateAggregate(ty, GetAggregateLayout(ty), {proc, env});
}

void Builder::CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block) {
    Assert(cond->type() == Type::BoolTy, "Branch condition must be a bool");
    CreateBranchInst(cond, then_block, else_block);
}

auto Builder::CreateInt(APInt val, Type type) -> Value* {
    if (type->size(tu) <= Size::Bits(64)) return CreateInt(val.getZExtValue(), type);
    auto m = std::unique_ptr<LargeInt>(new LargeInt(std::move(val), type));
    large_ints.push_back(std::move(m));
    return large_ints.back().get();
}

auto Builder::CreateInt(u64 val, Type type) -> Value* {
    return new (*this) SmallInt(val, type);
}

auto Builder::CreateInvalidLocalReference(LocalRefExpr* ref) -> Value* {
    // The actual type of this is a reference type since local refs are lvalues.
    auto ref_ty = PtrType::Get(tu, ref->type);
    return new (*this) InvalidLocalReference(ref, ref_ty);
}

auto Builder::CreateIMul(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::IMul, a->type(), {a, b});
    i->inst()->nowrap = nowrap;
    return i;
}

auto Builder::CreateICmpEq(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpEq, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpNe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpNe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpULt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpULe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpUGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpUGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSLt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSLe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGe, Type::BoolTy, {a, b});
}

auto Builder::CreateLoad(Type ty, Value* ptr) -> Value* {
    if (ty->is_arvalue()) return LoadAggregate(GetAggregateLayout(ty), ptr);
    return CreateSpecialGetVal<MemInst>(ty, Op::Load, ty, ty->align(tu), ptr);
}

auto Builder::CreateLShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::LShr, a->type(), {a, b});
}

void Builder::CreateMemCopy(Value* to, Value* from, Value* bytes, Align align) {
    Create(Op::MemCopy, {to, from, bytes})->alignment = align;
}

void Builder::CreateMemZero(Value* addr, Value* bytes, Align align) {
    Create(Op::MemZero, {addr, bytes})->alignment = align;
}

auto Builder::CreateNil(Type ty) -> Value* {
    if (ty->is_arvalue()) {
        SmallVector<Value*> values;
        auto layout = GetAggregateLayout(ty);
        for (auto f : layout->fields()) values.emplace_back(CreateNil(f));
        return CreateAggregate(ty, layout, values);
    }

    return new (*this) BuiltinConstant(BuiltinConstantKind::Nil, ty);
}

auto Builder::CreateOr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Or, a->type(), {a, b});
}

auto Builder::CreatePoison(Type ty) -> Value* {
    if (ty->is_arvalue()) {
        SmallVector<Value*> values;
        auto layout = GetAggregateLayout(ty);
        for (auto f : layout->fields()) values.emplace_back(CreatePoison(f));
        return CreateAggregate(ty, layout, values);
    }

    return new (*this) BuiltinConstant(BuiltinConstantKind::Poison, ty);
}

auto Builder::CreatePtrAdd(Value* ptr, Value* offs, bool inbounds) -> Value* {
    Assert(isa<PtrType>(ptr->type()), "First argument to ptradd must be a pointer");

    // This is really common in struct initialisers, so optimise for it.
    if (auto lit = dyn_cast<SmallInt>(offs); lit and lit->value() == 0)
        return ptr;

    auto i = CreateAndGetVal(Op::PtrAdd, ptr->type(), {ptr, offs});
    i->inst()->inbounds = inbounds;
    return i;
}

auto Builder::CreatePtrAdd(Value* ptr, Size offs, bool inbounds) -> Value* {
    if (offs == Size()) return ptr;
    return CreatePtrAdd(ptr, CreateInt(offs.bytes()),  inbounds);
}

auto Builder::CreateRange(Value* start, Value* end) -> Aggregate* {
    auto ty = RangeType::Get(tu, start->type());
    auto layout = GetAggregateLayout(ty);
    return CreateAggregate(ty, layout, {start, end});
}

void Builder::CreateReturn(Value* val) {
    ArrayRef<Value*> args;
    if (val) {
        if (auto a = dyn_cast<Aggregate>(val)) args = a->fields();
        else args = val;
    }
    Create(Op::Ret, args);
}

auto Builder::CreateSAddOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SAddOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
}

auto Builder::CreateSDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SDiv, a->type(), {a, b});
}

auto Builder::CreateSelect(Value* cond, Value* then_val, Value* else_val) -> Value* {
    return CreateAndGetVal(Op::Select, then_val->type(), {cond, then_val, else_val});
}

auto Builder::CreateShl(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Shl, a->type(), {a, b});
}

auto Builder::CreateSICast(Value* i, Type to_type) -> Value* {
    auto from = i->type()->size(tu).bits();
    auto to = to_type->size(tu).bits();
    if (i->type() == to_type) return i;

    // Fold the cast if possible; casts of literals are rather common.
    if (auto v = dyn_cast<SmallInt>(i)) return CreateInt(
        APInt(unsigned(from), v->value()).sextOrTrunc(unsigned(to)).getZExtValue(),
        to_type
    );

    // Create a cast operation.
    if (from > to) return CreateSpecialGetVal<ICast>(to_type, Op::Trunc, to_type, i);
    return CreateSpecialGetVal<ICast>(to_type, Op::SExt, to_type, i);
}

auto Builder::CreateSlice(Value* data, Value* size) -> Aggregate* {
    auto ty = SliceType::Get(tu, cast<PtrType>(data->type())->elem());
    return CreateAggregate(ty, GetAggregateLayout(ty), {data, size});
}

auto Builder::CreateSMulOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SMulOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
}

auto Builder::CreateSRem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SRem, a->type(), {a, b});
}

auto Builder::CreateSSubOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SSubOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
}

void Builder::CreateStore(Type ty, Value* val, Value* ptr) {
    // It’s easy to swap the parameters, so check for that.
    Assert(isa<PtrType>(ptr->type()), "Cannot store to non-pointer");

    // Storing nil is a memset to 0.
    if (auto b = dyn_cast<BuiltinConstant>(val); b and b->id == BuiltinConstantKind::Nil) {
        CreateMemZero(ptr, CreateInt(ty->size(tu).bytes()), ty->align(tu));
        return;
    }

    // ARValues should always be 'Aggregate*'s.
    if (auto a = dyn_cast<Aggregate>(val)) {
        StoreAggregate(ptr, a->layout(), a->fields());
        return;
    }

    // Convert stores of the nil value to memsets.
    Assert(ty, "Storing a non-aggregate requires a type");
    Assert(ty->is_srvalue(), "Cannot store non-srvalues");
    CreateImpl<MemInst>(Op::Store, ty, ty->align(tu), ptr, val);
}

auto Builder::CreateGlobalConstantPtr(Type ty, String s) -> GlobalConstant* {
    auto g = new (*this) GlobalConstant(s, PtrType::Get(tu, ty), ty->align(tu));
    global_consts.push_back(g);
    return g;
}

auto Builder::CreateGlobalStringSlice(String s) -> Aggregate* {
    auto data = CreateGlobalConstantPtr(tu.I8Ty, s);
    auto size = CreateInt(s.size(), Type::IntTy);
    data->string = true;
    return CreateAggregate(tu.StrLitTy, GetAggregateLayout(tu.StrLitTy), {data, size});
}

auto Builder::CreateProc(String s, Linkage link, ProcType* ty) -> Proc* {
    auto [it, inserted] = procs.try_emplace(s, std::unique_ptr<Proc>(new Proc(tu.I8PtrTy, s, ty, link)));
    Assert(inserted, "Procedure with name '{}' already exists", s);
    auto proc = it->second.get();
    for (auto [i, p] : enumerate(ty->params())) {
        Assert(p.intent == Intent::Copy, "Intents should have been lowered by codegen");
        proc->arguments.push_back(new (*this) Argument(proc, u32(i), p.type));
    }
    return proc;
}

auto Builder::CreateSub(Value* a, Value* b, bool nowrap) -> Value* {
    TRY_FOLD_INT(lhs - rhs);
    auto i = CreateAndGetVal(Op::Sub, a->type(), {a, b});
    i->inst()->nowrap = nowrap;
    return i;
}

auto Builder::CreateUDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::UDiv, a->type(), {a, b});
}

void Builder::CreateUnreachable() {
    CreateAndGetVal(Op::Unreachable, Type::VoidTy, {});
}

auto Builder::CreateURem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::URem, a->type(), {a, b});
}

auto Builder::CreateXor(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Xor, a->type(), {a, b});
}

auto Builder::CreateZExt(Value* i, Type to_type) -> Value* {
    auto from = i->type()->size(tu).bits();
    auto to = to_type->size(tu).bits();
    if (i->type() == to_type) return i;

    // Fold the cast if possible; casts of literals are rather common.
    if (auto v = dyn_cast<SmallInt>(i)) return CreateInt(
        APInt(unsigned(from), v->value()).zextOrTrunc(unsigned(to)).getZExtValue(),
        to_type
    );

    // Create a cast operation.
    if (from > to) return CreateSpecialGetVal<ICast>(to_type, Op::Trunc, to_type, i);
    return CreateSpecialGetVal<ICast>(to_type, Op::ZExt, to_type, i);
}

auto Builder::Fold(
    Value* a,
    Value* b,
    llvm::function_ref<u64(u64, u64)> op
) -> Value* {
    auto lhs = dyn_cast<SmallInt>(a);
    auto rhs = dyn_cast<SmallInt>(b);
    if (lhs and rhs) return CreateInt(op(lhs->value(), rhs->value()), a->type());
    return nullptr;
}


auto Builder::GetAggregateLayout(Type ty) -> StructLayout* {
    Assert(ty->is_arvalue());

#   define Compute(...) ComputeLayout([&]{ return std::array<Type, 2>{__VA_ARGS__}; })
    auto ComputeLayout = [&](llvm::function_ref<std::array<Type, 2>()> get_els) -> StructLayout* {
        auto layout = arvalue_layouts[ty];
        if (not layout) layout = StructLayout::Create(tu, get_els());
        return layout;
    };

    return ty->visit(utils::Overloaded{
        [](auto*) -> StructLayout* { Unreachable(); },
        [](StructType* s) { return s->layout(); },
        [&](SliceType* s) { return Compute(PtrType::Get(tu, s->elem()), Type::IntTy); },
        [&](ProcType*) { return closure_layout; },
        [&](RangeType* r) { return Compute(r->elem(), r->elem()); },
    });

#   undef Compute
}

auto Builder::GetOrCreateProc(String s, Linkage link, ProcType* ty) -> Proc* {
    auto it = procs.find(s.value());
    if (it != procs.end()) return it->second.get();
    return CreateProc(s, link, ty);
}

auto Builder::GetExistingProc(StringRef name) -> Ptr<Proc> {
    auto proc = procs.find(name);
    return proc == procs.end() ? nullptr : proc->second.get();
}

auto Builder::LoadAggregate(StructLayout* layout, Value* ptr) -> Aggregate* {
    SmallVector<Value*> fields;
    for (auto [ty, offs] : zip(layout->fields(), layout->offsets())) {
        ptr = CreatePtrAdd(ptr, offs);
        fields.push_back(CreateLoad(ty, ptr));
    }
    return CreateAggregate(Type(), layout, fields);
}

auto Builder::Result(Inst* i, Type ty, u32 idx) -> InstValue* {
    return new (*this) InstValue(i, ty, idx);
}

void Builder::StoreAggregate(Value* ptr, StructLayout* layout, ArrayRef<Value*> fields) {
    for (auto [offs, val] : zip(layout->offsets(), fields)) {
        ptr = CreatePtrAdd(ptr, offs);
        CreateStore(val->type(), val, ptr);
    }
}

auto AbortInst::handler_name() const -> String {
    switch (abort_reason()) {
        case AbortReason::AssertionFailed: return constants::AssertFailureHandlerName;
        case AbortReason::ArithmeticError: return constants::ArithmeticFailureHandlerName;
    }
    Unreachable();
}

bool Block::closed() const {
    if (insts.empty()) return false;
    switch (insts.back()->opcode()) {
        case Op::Abort:
        case Op::Br:
        case Op::Ret:
        case Op::Unreachable:
            return true;

        default:
            return false;
    }
}

bool Block::is_entry() const {
    return p and p->entry() == this;
}

Inst::Inst(Builder& b, Op op, ArrayRef<Value*> args, Align align)
    : arguments{args.copy(b.tu.allocator())}, op{op}, alignment{align} {
    DebugAssert(
        rgs::none_of(args, &Aggregate::classof),
        "Unsplit aggregate passed to instruction"
    );
}

bool Inst::has_multiple_results() const {
    return result_types().size() > 1;
}

auto Inst::result_types() const -> SmallVector<Type, 2> {
    using V = SmallVector<Type, 2>;
    switch (op) {
        case Op::Abort:
        case Op::Br:
        case Op::MemCopy:
        case Op::MemZero:
        case Op::Ret:
        case Op::Store:
        case Op::Unreachable:
            return {};

        case Op::Call: {
            auto c = cast<CallInst>(this);
            if (not c->res) return V{};
            if (auto a = dyn_cast<IRAggregateType>(c->res)) return V{a->layout()->fields()};
            return V{c->res};
        }

        case Op::Load:
            return {cast<MemInst>(this)->memory_type()};

        case Op::Select:
            return {arguments[1]->type()};

        case Op::SExt:
        case Op::Trunc:
        case Op::ZExt:
            return {cast<ICast>(this)->cast_result_type()};

        case Op::SAddOv:
        case Op::SMulOv:
        case Op::SSubOv:
            return {arguments[0]->type(), Type::BoolTy};

        case Op::Add:
        case Op::And:
        case Op::AShr:
        case Op::IMul:
        case Op::LShr:
        case Op::Or:
        case Op::PtrAdd:
        case Op::SDiv:
        case Op::Shl:
        case Op::SRem:
        case Op::Sub:
        case Op::UDiv:
        case Op::URem:
        case Op::Xor:
            return {arguments[0]->type()};

        case Op::ICmpEq:
        case Op::ICmpNe:
        case Op::ICmpSGe:
        case Op::ICmpSGt:
        case Op::ICmpSLe:
        case Op::ICmpSLt:
        case Op::ICmpUGe:
        case Op::ICmpUGt:
        case Op::ICmpULe:
        case Op::ICmpULt:
            return {Type::BoolTy};
    }

    Unreachable();
}

auto Proc::add(std::unique_ptr<Block> b) -> Block* {
    b->p = this;
    body.push_back(std::move(b));
    return body.back().get();
}

auto Value::as_int(TranslationUnit& tu) -> std::optional<APInt> {
    if (not isa<SmallInt, LargeInt>(this)) return std::nullopt;
    auto wd = u32(type()->size(tu).bits());
    APInt val{wd, 0};
    if (auto s = dyn_cast<SmallInt>(this)) val = s->value();
    else val = cast<LargeInt>(this)->value();
    return val;
}

bool Value::is_nil() const {
    auto b = dyn_cast<BuiltinConstant>(this);
    return b and b->id == BuiltinConstantKind::Nil;
}

/// ====================================================================
///  Printing
/// ====================================================================
namespace {
class Printer {
public:
    TranslationUnit& tu;
    SmallUnrenderedString out;
    DenseMap<GlobalConstant*, i64> global_ids;
    DenseMap<Argument*, i64> arg_ids;
    DenseMap<Block*, i64> block_ids;
    DenseMap<Inst*, i64> inst_ids;
    DenseMap<FrameSlot*, i64> frame_ids;
    Proc* curr_proc{};

    Printer(TranslationUnit& tu) : tu{tu} {}
    void Dump(Builder& b);
    void DumpInst(Inst* i);
    void DumpProc(Proc* p);
    [[nodiscard]] auto DumpValue(Value* b) -> SmallUnrenderedString;
    bool ReturnsValue(Inst* i);

    auto Id(auto& map, auto* ptr) -> i64 {
        auto it = map.find(ptr);
        return it == map.end() ? -1 : it->second;
    }
};
} // namespace

auto Builder::Dump() -> SmallUnrenderedString {
    Printer p{tu};
    p.Dump(*this);
    return std::move(p.out);
}

void Printer::Dump(Builder& b) {
    for (auto [i, g] : enumerate(b.global_constants())) {
        global_ids[g] = i64(i);
        out += std::format("%3(@{}%) %1(=%) ", i);
        if (g->is_string()) out += std::format("%3(\"{}\"%)", utils::Escape(g->value(), true, true));
        else {
            out += "%1({%) %5(";
            out += utils::join(g->value(), "%1(,%) ", "{:02X}");
            out += "%) %1(}, align %)";
            out += std::format("%5({}%)", g->align());
        }
        out += "\n";
    }

    for (auto* proc : b.procedures())
        DumpProc(proc);
}

void Printer::DumpInst(Inst* i) {
    out += "    %1(";
    if (ReturnsValue(i)) out += std::format("%8(%%{}%) = ", Id(inst_ids, i));
    defer {
        if (i->align() != Align()) out += std::format(", align %5({}%)", i->align());
        out += "%)\n";
    };

    auto Target = [&](Block* dest, ArrayRef<Value*> args) {
        out += std::format("%3(bb{}%)", Id(block_ids, dest));
        if (not args.empty()) {
            out += std::format(
                "({})",
                utils::join_as(args, [&](Value* v) { return DumpValue(v); })
            );
        }
    };

    auto IntCast = [&](StringRef name) {
        auto c = cast<ICast>(i);
        out += std::format(
            "%1({}%) {} to {}",
            name,
            DumpValue(c->args()[0]),
            c->cast_result_type()
        );
    };

    auto Simple = [&](StringRef name) {
        out += std::format(
            "%1({}%){}{}",
            name,
            i->args().empty() ? ""sv : " "sv,
            utils::join_as(i->args(), [&](Value* v) { return DumpValue(v); })
        );
    };

    switch (i->opcode()) {
        // Special instructions.
        case Op::Abort: {
            auto a = cast<AbortInst>(i);
            auto [file, line, col] = a->location().info_or_builtin(tu.context());
            out += std::format(
                "%1(abort%) at %6(<{}:{}:{}>%) %2({}%)({})",
                file,
                line,
                col,
                a->handler_name(),
                utils::join_as(a->args(), [&](Value* v) { return DumpValue(v); })
            );
        } break;

        case Op::Br: {
            auto b = cast<BranchInst>(i);
            if (b->is_conditional()) {
                out += std::format("%1(br%) {} to ", DumpValue(b->cond()));
                Target(b->then(), b->then_args());
                out += " else ";
                Target(b->else_(), b->else_args());
            } else {
                out += std::format("%1(br%) ");
                Target(b->then(), b->then_args());
            }
        } break;

        case Op::Call: {
            auto c = cast<CallInst>(i);
            auto proc = i->args().front();
            auto args = i->args().drop_front(1);
            auto ret = i->result_types();
            out += "%1(call%) ";

            if (ret.size() == 0) out += "%1(void%) ";
            else if (ret.size() == 1) out += std::format("{} ", ret.front());
            else out += std::format("%1(({})%) ", utils::join(ret));

            out += DumpValue(proc);
            if (not args.empty()) {
                out += std::format(
                    "({})",
                    utils::join_as(args, [&](Value* v) { return DumpValue(v); })
                );
            }

            if (auto env = c->environment().get_or_null())
                out += std::format(", %1(env%) {}", DumpValue(env));
        } break;

        case Op::Load: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(load%) {}, {}",
                m->memory_type(),
                DumpValue(m->args()[0])
            );
        } break;

        case Op::Store: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(store%) {} to {}, {}",
                m->memory_type(),
                DumpValue(m->args()[0]),
                DumpValue(m->args()[1])
            );
        } break;

        // Integer conversions.
        case Op::SExt: IntCast("sext"); break;
        case Op::Trunc: IntCast("trunc"); break;
        case Op::ZExt: IntCast("zext"); break;

        // Generic instructions.
        case Op::Add: Simple("add"); break;
        case Op::And: Simple("and"); break;
        case Op::AShr: Simple("ashr"); break;
        case Op::ICmpEq: Simple("icmp eq"); break;
        case Op::ICmpNe: Simple("icmp ne"); break;
        case Op::ICmpULt: Simple("icmp ult"); break;
        case Op::ICmpULe: Simple("icmp ule"); break;
        case Op::ICmpUGe: Simple("icmp uge"); break;
        case Op::ICmpUGt: Simple("icmp ugt"); break;
        case Op::ICmpSLt: Simple("icmp slt"); break;
        case Op::ICmpSLe: Simple("icmp sle"); break;
        case Op::ICmpSGe: Simple("icmp sge"); break;
        case Op::ICmpSGt: Simple("icmp sgt"); break;
        case Op::IMul: Simple("imul"); break;
        case Op::LShr: Simple("lshr"); break;
        case Op::MemCopy: Simple("copy"); break;
        case Op::MemZero: Simple("zero"); break;
        case Op::Or: Simple("or"); break;
        case Op::PtrAdd: Simple("ptradd"); break;
        case Op::Ret: Simple("ret"); break;
        case Op::SAddOv: Simple("sadd ov"); break;
        case Op::SDiv: Simple("sdiv"); break;
        case Op::Select: Simple("select"); break;
        case Op::Shl: Simple("shl"); break;
        case Op::SMulOv: Simple("smul ov"); break;
        case Op::SRem: Simple("srem"); break;
        case Op::Sub: Simple("sub"); break;
        case Op::SSubOv: Simple("ssub ov"); break;
        case Op::UDiv: Simple("udiv"); break;
        case Op::Unreachable: Simple("unreachable"); break;
        case Op::URem: Simple("urem"); break;
        case Op::Xor: Simple("xor"); break;
    }
}

void Printer::DumpProc(Proc* proc) {
    curr_proc = proc;
    if (not out.empty()) out += "\n";
    out += proc->proc_type()->print(proc->name(), true);

    // Stop if there is no body.
    if (proc->empty()) {
        out += "%1(;%)\n";
        return;
    }

    // Number all blocks and instructions.
    i64 temp = 0;
    arg_ids.clear();
    block_ids.clear();
    inst_ids.clear();

    for (auto* arg : proc->args()) arg_ids[arg] = temp++;
    for (auto* arg : proc->frame()) frame_ids[arg] = temp++;
    for (const auto& [id, b] : vws::enumerate(proc->blocks())) {
        block_ids[b] = id;
        for (auto* arg : b->split_arguments())
            arg_ids[arg] = temp++;
        for (auto* i : b->instructions())
            if (ReturnsValue(i))
                inst_ids[i] = temp++;
    }

    // Print the procedure body.
    out += " %1({%)\n";

    // Print frame allocations.
    for (auto* f : proc->frame())
        out += std::format("    %8(%%{}%) %1(=%) {}, %1(align%) {}\n", Id(frame_ids, f), f->allocated_type(), f->align());
    if (not proc->frame().empty())
        out += "\n";

    // Print blocks and instructions.
    for (const auto& [i, b] : enumerate(proc->blocks())) {
        out += i == 0 ? "%3(entry%)%1(" : std::format("\n%3(bb{}%)%1(", i);
        if (not b->arguments().empty()) {
            out += std::format("({})", utils::join_as(b->split_arguments(), [&](Argument* arg) {
                return std::format("{} %3(%%{}%)", arg->type(), arg_ids.at(arg));
            }));
        }
        out += ":%)\n";
        for (auto* inst : b->instructions())
            DumpInst(inst);
    }
    out += "%1(}%)\n";
}

bool Printer::ReturnsValue(Inst* i) {
    return not i->result_types().empty();
}

auto Printer::DumpValue(Value* v) -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (v->kind()) {
        case Value::Kind::Aggregate: {
            auto a = cast<Aggregate>(v);
            out += v->type()->print();
            out += " %1((%)";
            bool first = true;
            for (auto f : a->fields()) {
                if (first) first = false;
                else out += "%1(,%) ";
                out += DumpValue(f);
            }
            out += " %1()%)";
        } break;

        case Value::Kind::Argument: {
            auto arg = cast<Argument>(v);
            out += std::format("%{}(%%{}%)", isa<Proc>(arg->parent()) ? '4' : '3', Id(arg_ids, arg));
        } break;

        case Value::Kind::Block:
            out += std::format("%3(bb{}%)", Id(block_ids, cast<Block>(v)));
            break;

        case Value::Kind::BuiltinConstant: {
            auto c = cast<BuiltinConstant>(v);
            switch (c->id) {
                case BuiltinConstantKind::True: out += "true"; break;
                case BuiltinConstantKind::False: out += "false"; break;
                case BuiltinConstantKind::Nil: out += "nil"; break;
                case BuiltinConstantKind::Poison: out += "poison"; break;
            }
        } break;

        case Value::Kind::FrameSlot: {
            auto f = cast<FrameSlot>(v);
            out += std::format("%8(%%{}%)", Id(frame_ids, f));
        } break;

        case Value::Kind::GlobalConstant: {
            auto c = cast<GlobalConstant>(v);
            out += std::format("%3(@{}%)", Id(global_ids, c));
        } break;

        case Value::Kind::InstValue: {
            auto i = cast<InstValue>(v);
            if (i->inst()->has_multiple_results()) out += std::format("%8(%%{}:{}%)", Id(inst_ids, i->inst()), i->index());
            else out += std::format("%8(%%{}%)", Id(inst_ids, i->inst()));
        } break;

        case Value::Kind::InvalidLocalReference: {
            auto i = cast<InvalidLocalReference>(v);
            auto d = i->referenced_local()->decl;
            out += std::format(
                "<invalid access to %2({}%)::%8({}%)>",
                d->parent->name,
                d->name
            );
        } break;

        case Value::Kind::LargeInt: {
            auto l = cast<LargeInt>(v);
            out += std::format("{} %5({}%)", l->type(), l->value());
        } break;

        case Value::Kind::Proc: {
            auto p = cast<Proc>(v);
            out += std::format("%2({}%)", p->name());
        } break;

        case Value::Kind::SmallInt: {
            auto s = cast<SmallInt>(v);
            out += std::format(
                "{} %5({}%)",
                v->type(),
                APInt(u32(v->type()->size(tu).bits()), u64(s->value()))
            );
        } break;
    }
    return out;
}

void Inst::dump(TranslationUnit& tu) {
    Printer p{tu};
    p.DumpInst(this);
    p.out += "\n";
    std::print(stderr, "{}", text::RenderColours(true, p.out.str()));
}

auto Proc::attributes(u32 param_idx) const -> ParamAttrs {
    auto it = param_attrs.find(param_idx);
    if (it == param_attrs.end()) return ParamAttrs();
    return it->second;
}

void Proc::dump(TranslationUnit& tu) {
    Printer p{tu};
    p.DumpProc(this);
    p.out += "\n";
    std::print(stderr, "{}", text::RenderColours(true, p.out.str()));
}

void Value::dump(TranslationUnit& tu) const {
    Printer p{tu};
    p.out += type()->print();
    p.out += " ";
    p.out += p.DumpValue(const_cast<Value*>(this));
    p.out += "\n";
    std::print(stderr, "{}", text::RenderColours(true, p.out.str()));
}
