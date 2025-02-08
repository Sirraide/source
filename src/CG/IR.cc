#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

auto ManagedValue::operator new(usz sz, Builder& b) -> void* {
    return b.Allocate(sz, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

Builder::Builder(TranslationUnit& tu) : tu{tu} {}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateImpl(Args&&... args) -> Inst* {
    Assert(insert_point, "No insert point");
    auto i = new (*this) InstTy(*this, LIBBASE_FWD(args)...);
    insert_point->instructions.push_back(i);
    return i;
}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateSpecialGetVal(Type val_ty, Args&&... args) -> Value* {
    return new (*this) InstValue(CreateImpl<InstTy>(LIBBASE_FWD(args)...), val_ty, 0);
}

auto Builder::Create(Op op, ArrayRef<Value*> vals) -> Inst* {
    return CreateImpl<Inst>(op, vals);
}

auto Builder::CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> Value* {
    return new (*this) InstValue(Create(op, vals), ty, 0);
}

auto Builder::CreateAdd(Value* a, Value* b, bool nowrap) -> Value* {
    auto add = CreateAndGetVal(Op::Add, a->type(), {a, b});
    add->nowrap = nowrap;
    return add;
}

auto Builder::CreateAlloca(Type ty) -> Value* {
    return CreateSpecialGetVal<Alloca>(ReferenceType::Get(tu, ty), ty);
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
    auto b = std::make_unique<Block>();
    append_range(b->arg_types, args);
    for (auto [i, ty] : enumerate(b->arg_types))
        b->args.push_back(new (*this) Argument(b.get(), u32(i), ty));
    return b;
}

auto Builder::CreateBlock(Proc* proc, ArrayRef<Type> args) -> Block* {
    return proc->add(CreateBlock(args));
}

auto Builder::CreateBool(bool value) -> Value* {
    return &(value ? true_val : false_val);
}

void Builder::CreateBr(Block* dest, ArrayRef<Value*> args) {
    CreateImpl<BranchInst>(BranchTarget{dest, args});
}

auto Builder::CreateCall(Value* callee, ArrayRef<Value*> args) -> Value* {
    SmallVector operands{callee};
    append_range(operands, args);
    return CreateAndGetVal(Op::Call, callee->type(), operands);
}

void Builder::CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block) {
    CreateImpl<BranchInst>(cond, then_block, else_block);
}

auto Builder::CreateExtractValue(Value* aggregate, u32 idx) -> Value* {
    if (auto s = dyn_cast<Slice>(aggregate)) {
        if (idx == 0) return s->data;
        if (idx == 1) return s->size;
        Unreachable("Invalid index for slice");
    }

    if (auto s = dyn_cast<SliceType>(aggregate->type().ptr())) {
        if (idx == 0) return new (*this) Extract(aggregate, 0, ReferenceType::Get(tu, s->elem()));
        if (idx == 1) return new (*this) Extract(aggregate, 1, Types::IntTy);
        Unreachable("Invalid index for slice type");
    }

    Todo("Extract a value from this type");
}

auto Builder::CreateInt(APInt val) -> Value* {
    auto m = std::make_unique<LargeInt>(std::move(val), IntType::Get(tu, Size::Bits(val.getBitWidth())));
    large_ints.push_back(std::move(m));
    return large_ints.back().get();
}

auto Builder::CreateInt(i64 val, Type type) -> Value* {
    auto& i = small_ints[val];
    if (not i) i = new (*this) SmallInt(val, type);
    return i;
}

auto Builder::CreateIMul(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::IMul, a->type(), {a, b});
    i->nowrap = nowrap;
    return i;
}

auto Builder::CreateICmpEq(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpEq, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpNe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpNe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpULt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpULe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpUGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpUGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSLt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSLe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGe, Types::BoolTy, {a, b});
}

auto Builder::CreateLoad(Type ty, Value* ptr) -> Value* {
    return CreateSpecialGetVal<MemInst>(ty, Op::Load, ty, ty->align(tu), ptr);
}

auto Builder::CreateLShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::LShr, a->type(), {a, b});
}

void Builder::CreateMemZero(Value* addr, Value* bytes) {
    Create(Op::MemZero, {addr, bytes});
}

auto Builder::CreateNil(Type ty) -> Value* {
    return new (*this) BuiltinConstant(BuiltinConstantKind::Nil, ty);
}

auto Builder::CreateOr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Or, a->type(), {a, b});
}

auto Builder::CreatePoison(Type ty) -> Value* {
    return new (*this) BuiltinConstant(BuiltinConstantKind::Poison, ty);
}

auto Builder::CreatePtrAdd(Value* ptr, Value* offs, bool inbounds) -> Value* {
    Assert(isa<ReferenceType>(ptr->type()), "First argument to ptradd must be a pointer");
    auto i = CreateAndGetVal(Op::PtrAdd, ptr->type(), {ptr, offs});
    i->inbounds = inbounds;
    return i;
}

void Builder::CreateReturn(Value* val) {
    Create(Op::Ret, val ? ArrayRef{val} : ArrayRef<Value*>{});
}

auto Builder::CreateSAddOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SAddOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
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
    if (from == to) return i;
    if (from > to) return CreateSpecialGetVal<ICast>(to_type, Op::Trunc, to_type, i);
    return CreateSpecialGetVal<ICast>(to_type, Op::ZExt, to_type, i);
}

auto Builder::CreateSMulOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SMulOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
}

auto Builder::CreateSRem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SRem, a->type(), {a, b});
}

auto Builder::CreateSSubOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SSubOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
}

void Builder::CreateStore(Value* val, Value* ptr) {
    CreateImpl<MemInst>(Op::Store, val->type(), val->type()->align(tu), ptr, val);
}

auto Builder::CreateString(String s) -> Slice* {
    auto data = new (*this) StringData(s, ReferenceType::Get(tu, tu.StrLitTy->elem()));
    auto size = CreateInt(s.size(), Types::IntTy);
    return new (*this) Slice(tu.StrLitTy, data, size);
}

auto Builder::CreateSub(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::Sub, a->type(), {a, b});
    i->nowrap = nowrap;
    return i;
}

auto Builder::CreateUDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::UDiv, a->type(), {a, b});
}

void Builder::CreateUnreachable() {
    CreateAndGetVal(Op::Unreachable, Types::VoidTy, {});
}

auto Builder::CreateURem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::URem, a->type(), {a, b});
}

auto Builder::CreateXor(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Xor, a->type(), {a, b});
}

auto Builder::GetOrCreateProc(String s, Linkage link, ProcType* ty) -> Proc* {
    auto& proc = procs[s.value()];
    if (proc) return proc.get();
    proc = std::make_unique<Proc>(s, ty, link);
    return proc.get();
}

auto Builder::Result(Inst* i, Type ty, u32 idx) -> InstValue* {
    return new (*this) InstValue(i, ty, idx);
}

bool Block::closed() const {
    if (instructions.empty()) return false;
    switch (instructions.back()->opcode) {
        case Op::Br:
        case Op::CondBr:
        case Op::Ret:
        case Op::Unreachable:
            return true;

        default:
            return false;
    }
}

Inst::Inst(Builder& b, Op op, ArrayRef<Value*> args) : op{op}, arguments{args.copy(b.alloc)} {}

auto Proc::add(std::unique_ptr<Block> b) -> Block* {
    b->p = this;
    blocks.push_back(std::move(b));
    return blocks.back().get();
}

auto Proc::args(Builder& b) -> ArrayRef<Argument*> {
    if (not arguments.empty() or ty->params().empty()) return arguments;
    for (auto [i, p] : enumerate(ty->params())) {
        Type param_ty = p.type->pass_by_lvalue(ty->cconv(), p.intent)
                          ? ReferenceType::Get(b.tu, p.type)
                          : p.type;
        arguments.push_back(new (b) Argument(this, u32(i), param_ty));
    }
    return arguments;
}
