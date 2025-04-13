#ifndef SRCC_CG_IR_HH
#define SRCC_CG_IR_HH

#include <srcc/AST/Type.hh>
#include <srcc/Core/Token.hh>

#define SRCC_IR_VALUE_KINDS(X) \
    X(Argument)                \
    X(Block)                   \
    X(BuiltinConstant)         \
    X(Extract)                 \
    X(InstValue)               \
    X(InvalidLocalReference)   \
    X(LargeInt)                \
    X(Proc)                    \
    X(Slice)                   \
    X(SmallInt)                \
    X(StringData) \

/// IR used during codegen before LLVM IR or VM bytecode is emitted.
namespace srcc::cg::ir {
class Value;
class Block;
class Proc;
class SmallInt;
class Builder;

enum class AbortReason : u8 {
    AssertionFailed,
    ArithmeticError,
};

enum class Op : u8 {
    Abort,
    Add,
    Alloca,
    And,
    AShr,
    Br,
    Call,
    ICmpEq,
    ICmpNe,
    ICmpSGe,
    ICmpSGt,
    ICmpSLe,
    ICmpSLt,
    ICmpUGe,
    ICmpUGt,
    ICmpULe,
    ICmpULt,
    IMul,
    Load,
    LShr,
    MemZero,
    Or,
    PtrAdd,
    Ret,
    SAddOv,
    SDiv,
    Select,
    SExt,
    Shl,
    SMulOv,
    SRem,
    SSubOv,
    Store,
    Sub,
    Trunc,
    UDiv,
    Unreachable,
    URem,
    Xor,
    ZExt,
};

enum class BuiltinConstantKind : u8 {
    True,
    False,
    Nil,
    Poison,
};

class Value {
public:
    enum class Kind : u8 {
#define X(k) k,
        SRCC_IR_VALUE_KINDS(X)
#undef X
    };

private:
    Kind k;
    Type ty;

protected:
    Value(Kind k, Type ty) : k{k}, ty{ty} {}

public:
    [[nodiscard]] auto as_int(TranslationUnit& tu) -> std::optional<APInt>;
    [[nodiscard]] auto kind() const { return k; }
    [[nodiscard]] auto type() const { return ty; }

    template <typename Visitor>
    auto visit(Visitor&& visitor) const -> decltype(auto);
};

struct Managed {
    void* operator new(usz) = delete;
    void* operator new(usz, Builder&);
};

class ManagedValue : public Value, public Managed {
protected:
    ManagedValue(Kind k, Type ty) : Value{k, ty} {}
};

class BuiltinConstant : public ManagedValue {
    friend Builder;

public:
    const BuiltinConstantKind id;

private:
    BuiltinConstant(BuiltinConstantKind val, Type ty) : ManagedValue{Kind::BuiltinConstant, ty}, id{val} {}

public:
    static bool classof(const Value* v) { return v->kind() == Kind::BuiltinConstant; }
};

class SmallInt : public ManagedValue {
    friend Builder;

    u64 val;

    SmallInt(u64 val, Type ty = Type::IntTy) : ManagedValue{Kind::SmallInt, ty}, val{val} {}

public:
    auto value() const -> u64 { return val; }

    static bool classof(const Value* v) { return v->kind() == Kind::SmallInt; }
};

class LargeInt : public Value {
    friend Builder;

    APInt val;

    LargeInt(APInt val, Type ty = Type::IntTy) : Value{Kind::LargeInt, ty}, val{std::move(val)} {}

public:
    auto value() const -> const APInt& { return val; }

    static bool classof(const Value* v) { return v->kind() == Kind::LargeInt; }
};

class Slice : public ManagedValue {
    friend Builder;


    Slice(Type ty, Value* data, Value* size) : ManagedValue{Kind::Slice, ty}, data{data}, size{size} {}

public:
    Value* const data;
    Value* const size;

    static bool classof(const Value* v) { return v->kind() == Kind::Slice; }
};

class StringData : public ManagedValue {
    friend Builder;

    String val;

    StringData(String val, Type ty) : ManagedValue{Kind::StringData, ty}, val{val} {}

public:
    auto value() const -> String { return val; }

    static bool classof(const Value* v) { return v->kind() == Kind::StringData; }
};

class Argument : public ManagedValue {
    friend Builder;
    friend Proc;
    friend Block;

    Value* p; /// Either a proc or block.
    u32 idx;

    Argument(Value* p, u32 idx, Type ty) : ManagedValue{Kind::Argument, ty}, p{p}, idx{idx} {}

public:
    [[nodiscard]] auto parent() const -> Value* { return p; }
    [[nodiscard]] auto index() const -> u32 { return idx; }

    static bool classof(const Value* v) { return v->kind() == Kind::Argument; }
};

class Extract : public ManagedValue {
    friend Builder;

    Value* arg;
    u32 idx;

    Extract(Value* agg, u32 idx, Type ty) : ManagedValue{Kind::Extract, ty}, arg{agg}, idx{idx} {}

public:
    auto aggregate() const -> Value* { return arg; }
    auto index() const -> u32 { return idx; }

    static bool classof(const Value* v) { return v->kind() == Kind::Extract; }
};

/// Instructions don’t have types because they don’t correspond to values; rather
/// an instruction can have multiple result values.
class Inst : public Managed {
    friend Builder;

    ArrayRef<Value*> arguments;
    const Op op;

protected:
    Inst(Builder& b, Op op, ArrayRef<Value*> args);

public:
    // Not all of these are meaningful for every instruction.
    bool nowrap : 1 = false;
    bool inbounds : 1 = false;

    void dump(TranslationUnit& tu);
    [[nodiscard]] auto args() const -> ArrayRef<Value*> { return arguments; }
    [[nodiscard]] bool has_multiple_results() const;
    [[nodiscard]] auto opcode() const { return op; }
    [[nodiscard]] auto result_types() const -> SmallVector<Type, 2>;
    [[nodiscard]] auto operator[](usz idx) { return arguments[idx]; }
};

/// Allocas are separate instructions because their operand is a type.
class Alloca : public Inst {
    friend Builder;

    Type ty;

    Alloca(Builder& b, Type ty) : Inst{b, Op::Alloca, {}}, ty{ty} {}

public:
    [[nodiscard]] auto allocated_type() const { return ty; }

    static bool classof(const Inst* v) { return v->opcode() == Op::Alloca; }
};

/// Integral cast instructions also take a type.
class ICast : public Inst {
    friend Builder;

    Type to_type;

    ICast(Builder& b, Op op, Type to_type, Value* val) : Inst{b, op, {val}}, to_type{to_type} {}

public:
    [[nodiscard]] auto cast_result_type() const { return to_type; }

    static bool classof(const Inst* v) {
        return v->opcode() == Op::SExt or v->opcode() == Op::ZExt or v->opcode() == Op::Trunc;
    }
};

/// Memory instructions take a type and alignment.
class MemInst : public Inst {
    friend Builder;

    Type mem_type;
    Align alignment;

    MemInst(Builder& b, Op op, Type mem_type, Align alignment, Value* ptr, Value* val = nullptr)
        : Inst{b, op, val ? ArrayRef{ptr, val} : ArrayRef{ptr}}, mem_type{mem_type}, alignment{alignment} {}

public:
    [[nodiscard]] auto align() const { return alignment; }
    [[nodiscard]] auto memory_type() const { return mem_type; }
    [[nodiscard]] auto ptr() const { return args()[0]; }
    [[nodiscard]] auto value() const { return args()[1]; }

    static bool classof(const Inst* v) {
        return v->opcode() == Op::Load or v->opcode() == Op::Store;
    }
};

/// An 'abort' instruction is used to call a failure handler, such as for
/// assertion failure and integer overflow. This is a separate instruction
/// to simplify the implementation of such handlers at compile time.
class AbortInst : public Inst {
    friend Builder;

    AbortReason reason;
    Location loc;

    AbortInst(Builder& b, AbortReason reason, Location loc, ArrayRef<Value*> args)
        : Inst{b, Op::Abort, args}, reason{reason}, loc{loc} {}

public:
    [[nodiscard]] auto abort_reason() const { return reason; }
    [[nodiscard]] auto handler_name() const -> String;
    [[nodiscard]] auto location() const { return loc; }

    static bool classof(const Inst* v) { return v->opcode() == Op::Abort; }
};

struct BranchTarget {
    Block* dest{};
    ArrayRef<Value*> args{};

    BranchTarget() = default;
    BranchTarget(Block* dest, ArrayRef<Value*> args = {}) : dest{dest}, args{args} {}
    BranchTarget(std::unique_ptr<Block>& dest, ArrayRef<Value*> args = {}) : BranchTarget(dest.get(), args) {}
};

class BranchInst : public Inst {
    friend Builder;

    // Arguments are stored as: [cond, then_args..., else_args...] or [then_args...]
    u32 then_args_num;
    Block* then_block;
    Block* else_block;

    BranchInst(Builder& b, Value* cond, BranchTarget then, BranchTarget else_);

public:
    auto cond() const -> Value* { return is_conditional() ? args().front() : nullptr; }
    auto else_() const -> Block* { return else_block; }
    auto else_args() const -> ArrayRef<Value*> { return is_conditional() ? BlockArgs().drop_front(then_args_num) : ArrayRef<Value*>{}; }
    bool is_conditional() const { return else_block != nullptr; }
    auto then() const -> Block* { return then_block; }
    auto then_args() const -> ArrayRef<Value*> { return BlockArgs().take_front(then_args_num); }

    static bool classof(const Inst* v) { return v->opcode() == Op::Br; }

private:
    auto BlockArgs() const -> ArrayRef<Value*> { return args().drop_front(is_conditional() ? 1 : 0); }
};

class InstValue : public ManagedValue {
    friend Builder;

    Inst* i;
    const u32 idx;

    InstValue(Inst* i, Type ty, u32 index) : ManagedValue{Kind::InstValue, ty}, i{i}, idx{index} {}

public:
    [[nodiscard]] auto index() const { return idx; }
    [[nodiscard]] auto inst() -> Inst* { return i; }

    static bool classof(const Value* v) { return v->kind() == Kind::InstValue; }
};

/// Used in constant evaluation to mark references to local variables
/// that are declared outside the current constant evaluation context;
/// such references emit an error if execution actually reaches them.
class InvalidLocalReference : public ManagedValue {
    friend Builder;

    LocalRefExpr* reference;

    InvalidLocalReference(LocalRefExpr* ref, Type ref_ty)
        : ManagedValue{Kind::InvalidLocalReference, ref_ty}, reference{ref} {}

public:
    [[nodiscard]] auto referenced_local() const -> LocalRefExpr* { return reference; }

    static bool classof(const Value* v) { return v->kind() == Kind::InvalidLocalReference; }
};

class Block : public Value {
    friend Builder;
    friend Proc;

    Proc* p = nullptr;
    SmallVector<Inst*> insts;
    SmallVector<Type, 6> arg_types;
    SmallVector<Argument*> arg_vals;

    Block() : Value(Kind::Block, Type::VoidTy) {}

public:
    auto arg(u32 idx) -> Argument* { return arg_vals[idx]; }
    auto arguments() -> ArrayRef<Argument*> { return arg_vals; }
    auto argument_types() -> ArrayRef<Type> { return arg_types; }
    bool closed() const;
    auto front() -> Inst* { return insts.front(); }
    auto instructions() const -> ArrayRef<Inst*> { return insts; }
    auto parent() -> Proc* { return p; }

    static bool classof(const Value* v) { return v->kind() == Kind::Block; }
};

class Proc : public Value {
    friend Builder;

    String mangled_name;
    ProcType* ty;
    Linkage link;
    SmallVector<std::unique_ptr<Block>> body;
    SmallVector<Argument*> arguments;
    ProcDecl* associated_decl = nullptr;

    Proc(String mangled_name, ProcType* ty, Linkage link)
        : Value{Kind::Proc, ty}, mangled_name{mangled_name}, ty{ty}, link{link} {}

public:
    auto add(std::unique_ptr<Block> b) -> Block*;
    auto args() const -> ArrayRef<Argument*> { return arguments; }
    auto blocks() const {  return vws::all(body) | vws::transform([](auto& b) { return b.get(); }); }
    auto decl() const -> ProcDecl* { return associated_decl; }
    void dump(TranslationUnit& tu);
    auto empty() const -> bool { return body.empty(); }
    auto entry() const -> Block* { return body.empty() ? nullptr : body.front().get(); }
    auto linkage() const -> Linkage { return link; }
    auto name() const -> String { return mangled_name; }
    auto type() const -> ProcType* { return ty; }

    static bool classof(const Value* v) { return v->kind() == Kind::Proc; }
};

struct OverflowResult {
    Value* value;
    Value* overflow;
};

template <typename Visitor>
auto Value::visit(Visitor&& visitor) const -> decltype(auto) {
#define X(k) if (auto v = dyn_cast<k>(this)) return visitor(v);
    SRCC_IR_VALUE_KINDS(X)
#undef X
    Unreachable();
}

class Builder {
    friend Inst;

public:
    class InsertPointGuard {
        LIBBASE_IMMOVABLE(InsertPointGuard);
        Builder& b;
        Block* saved_insert_point;
    public:
        InsertPointGuard(Builder& b) : b{b} { saved_insert_point = b.insert_point; }
        ~InsertPointGuard() { b.insert_point = saved_insert_point; }
    };


    TranslationUnit& tu;

private:
    llvm::MapVector<StringRef, std::unique_ptr<Proc>> procs;
    SmallVector<std::unique_ptr<LargeInt>> large_ints;
    BuiltinConstant true_val{BuiltinConstantKind::True, Type::BoolTy};
    BuiltinConstant false_val{BuiltinConstantKind::False, Type::BoolTy};

public:
    Block* insert_point = nullptr;

    Builder(TranslationUnit& tu);

    auto procedures() const {
        return vws::all(procs) | vws::transform([](auto& e) -> Proc* { return e.second.get(); });
    }

    auto Allocate(usz sz, usz align) -> void* { return tu.allocate(sz, align); }
    auto Dump() -> SmallUnrenderedString;
    auto GetExistingProc(StringRef name) -> Ptr<Proc>;
    auto GetOrCreateProc(String s, Linkage link, ProcType* ty) -> Proc*;
    auto GetOrCreateProc(ProcDecl* proc, String mangled_name) -> Proc*;

    auto CreateAShr(Value* a, Value* b) -> Value*;
    auto CreateAbort(AbortReason reason, Location loc, Value* msg1, Value* msg2) -> void;
    auto CreateAdd(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateAlloca(ir::Proc* parent, Type ty) -> Value*;
    auto CreateAnd(Value* a, Value* b) -> Value*;
    auto CreateBlock() -> std::unique_ptr<Block> { return CreateBlock(ArrayRef<Type>{}); }
    auto CreateBlock(ArrayRef<Type> args) -> std::unique_ptr<Block>;
    auto CreateBlock(ArrayRef<Value*> args) -> std::unique_ptr<Block>;
    auto CreateBlock(Proc* proc, ArrayRef<Type> args) -> Block*;
    auto CreateBool(bool value) -> Value*;
    auto CreateBr(Block* dest, ArrayRef<Value*> args = {}) -> void;
    auto CreateCall(Value* callee, ArrayRef<Value*> args) -> Value*;
    auto CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block) -> void;
    auto CreateExtractValue(Value* aggregate, u32 idx) -> Value*;
    auto CreateICmpEq(Value* a, Value* b) -> Value*;
    auto CreateICmpNe(Value* a, Value* b) -> Value*;
    auto CreateICmpSGe(Value* a, Value* b) -> Value*;
    auto CreateICmpSGt(Value* a, Value* b) -> Value*;
    auto CreateICmpSLe(Value* a, Value* b) -> Value*;
    auto CreateICmpSLt(Value* a, Value* b) -> Value*;
    auto CreateICmpUGe(Value* a, Value* b) -> Value*;
    auto CreateICmpUGt(Value* a, Value* b) -> Value*;
    auto CreateICmpULe(Value* a, Value* b) -> Value*;
    auto CreateICmpULt(Value* a, Value* b) -> Value*;
    auto CreateIMul(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateInt(APInt val, Type type) -> Value*;
    auto CreateInt(u64 val, Type type = Type::IntTy) -> Value*;
    auto CreateInvalidLocalReference(LocalRefExpr* ref) -> Value*;
    auto CreateLShr(Value* a, Value* b) -> Value*;
    auto CreateLoad(Type ty, Value* ptr) -> Value*;
    auto CreateMemZero(Value* addr, Value* bytes) -> void;
    auto CreateNil(Type ty) -> Value*;
    auto CreateOr(Value* a, Value* b) -> Value*;
    auto CreatePoison(Type ty) -> Value*;
    auto CreatePtrAdd(Value* ptr, Value* offs, bool inbounds) -> Value*;
    auto CreateReturn(Value* val = nullptr) -> void;
    auto CreateSAddOverflow(Value* a, Value* b) -> OverflowResult;
    auto CreateSDiv(Value* a, Value* b) -> Value*;
    auto CreateSICast(Value* i, Type to_type) -> Value*;
    auto CreateSMulOverflow(Value* a, Value* b) -> OverflowResult;
    auto CreateSRem(Value* a, Value* b) -> Value*;
    auto CreateSSubOverflow(Value* a, Value* b) -> OverflowResult;
    auto CreateSelect(Value* cond, Value* then_val, Value* else_val) -> Value*;
    auto CreateShl(Value* a, Value* b) -> Value*;
    auto CreateSlice(Value* data, Value* size) -> Slice*;
    auto CreateStore(Value* val, Value* ptr) -> void;
    auto CreateString(String s) -> Slice*;
    auto CreateSub(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateUDiv(Value* a, Value* b) -> Value*;
    auto CreateURem(Value* a, Value* b) -> Value*;
    auto CreateUnreachable() -> void;
    auto CreateXor(Value* a, Value* b) -> Value*;

private:
    template <std::derived_from<Inst> InstTy, typename ...Args>
    auto CreateImpl(Args&&... args) -> Inst*;

    template <std::derived_from<Inst> InstTy, typename ...Args>
    auto CreateSpecialGetVal(Type val_ty, Args&&... args) -> Value*;

    auto Create(Op op, ArrayRef<Value*> vals) -> Inst*;
    auto CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> InstValue*;
    auto Result(Inst* i, Type ty, u32 idx) -> InstValue*;
};
}

#endif //SRCC_CG_IR_HH
