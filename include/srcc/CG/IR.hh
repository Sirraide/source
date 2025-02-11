#ifndef SRCC_CG_IR_HH
#define SRCC_CG_IR_HH

#include <srcc/AST/Type.hh>
#include <srcc/Core/Token.hh>

/// IR used during codegen before LLVM IR or VM bytecode is emitted.
namespace srcc::cg::ir {
class Value;
class Block;
class Proc;
class SmallInt;
class Builder;

enum class Op : u8 {
    Add,
    Alloca,
    And,
    AShr,
    Br,
    Call,
    CondBr,
    ICmpEq,
    ICmpNe,
    ICmpULt,
    ICmpULe,
    ICmpUGe,
    ICmpUGt,
    ICmpSLt,
    ICmpSLe,
    ICmpSGe,
    ICmpSGt,
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
    Store,
    Sub,
    SSubOv,
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
        Alloca,
        Argument,
        Extract,
        SmallInt,
        LargeInt,
        InstValue,
        BuiltinConstant,
        Block,
        Slice,
        StringData,
        Proc,
    };

private:
    Kind k;

public:
    // Not all of these are meaningful for every value.
    bool nowrap : 1 = false;
    bool inbounds : 1 = false;

private:
    Type ty;

protected:
    Value(Kind k, Type ty) : k{k}, ty{ty} {}

public:
    [[nodiscard]] auto kind() const { return k; }
    [[nodiscard]] auto type() const { return ty; }
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

    BuiltinConstantKind value;

    BuiltinConstant(BuiltinConstantKind val, Type ty) : ManagedValue{Kind::BuiltinConstant, ty}, value{val} {}
};

class SmallInt : public ManagedValue {
    friend Builder;

    i64 value;

    SmallInt(i64 val, Type ty = Types::IntTy) : ManagedValue{Kind::SmallInt, ty}, value{val} {}

public:
    static bool classof(const Value* v) { return v->kind() == Kind::SmallInt; }
};

class LargeInt : public Value {
    friend Builder;

    APInt value;

    LargeInt(APInt val, Type ty = Types::IntTy) : Value{Kind::LargeInt, ty}, value{std::move(val)} {}

public:
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

    String value;

    StringData(String val, Type ty) : ManagedValue{Kind::StringData, ty}, value{val} {}

public:
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

    Value* aggregate;
    u32 idx;

    Extract(Value* agg, u32 idx, Type ty) : ManagedValue{Kind::Extract, ty}, aggregate{agg}, idx{idx} {}

public:
    static bool classof(const Value* v) { return v->kind() == Kind::Extract; }
};

/// Instructions don’t have types because they don’t correspond to values; rather
/// an instruction can have multiple result values.
class Inst : public Managed {
    friend Builder;

    const Op op;
    ArrayRef<Value*> arguments;

protected:
    Inst(Builder& b, Op op, ArrayRef<Value*> args);

public:
    [[nodiscard]] auto args() const -> ArrayRef<Value*> { return arguments; }
    [[nodiscard]] auto opcode() const { return op; }
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

struct BranchTarget {
    Block* dest{};
    ArrayRef<Value*> args{};

    BranchTarget() = default;
    BranchTarget(Block* dest, ArrayRef<Value*> args = {}) : dest{dest}, args{args} {}
    BranchTarget(std::unique_ptr<Block>& dest, ArrayRef<Value*> args = {}) : BranchTarget(dest.get(), args) {}
};

class BranchInst : public Inst {
    friend Builder;

    BranchTarget then_block{}, else_block{};

    BranchInst(Builder& b, BranchTarget dest) : Inst{b, Op::Br, {}}, then_block(dest) {}
    BranchInst(Builder& b, Value* cond, BranchTarget then_block, BranchTarget else_block)
        : Inst{b, Op::CondBr, {cond}}, then_block{then_block}, else_block{else_block} {}

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

class Block : public Value {
    friend Builder;
    friend Proc;

    Proc* p = nullptr;
    SmallVector<Inst*> instructions;
    SmallVector<Type, 6> arg_types;
    SmallVector<Argument*> args;

    Block() : Value(Kind::Block, Types::VoidTy) {}

public:
    auto arg(u32 idx) -> Argument* { return args[idx]; }
    bool closed() const;
    auto parent() -> Proc* { return p; }
    static bool classof(const Value* v) { return v->kind() == Kind::Block; }
};

class Proc : public Value {
    friend Builder;

    String name;
    ProcType* ty;
    Linkage link;
    SmallVector<std::unique_ptr<Block>> blocks;
    SmallVector<Argument*> arguments;

    Proc(String name, ProcType* ty, Linkage link) : Value{Kind::Proc, ty}, name{name}, ty{ty}, link{link} {}

public:
    auto add(std::unique_ptr<Block> b) -> Block*;
    auto args(Builder& b) -> ArrayRef<Argument*>;
    auto empty() const -> bool { return blocks.empty(); }
    auto entry() const -> Block* { return blocks.empty() ? nullptr : blocks.front().get(); }

    static bool classof(const Value* v) { return v->kind() == Kind::Proc; }
};

struct OverflowResult {
    Value* value;
    Value* overflow;
};

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
    llvm::BumpPtrAllocator alloc;
    StringMap<std::unique_ptr<Proc>> procs;
    DenseMap<i64, SmallInt*> small_ints;
    SmallVector<std::unique_ptr<LargeInt>> large_ints;
    BuiltinConstant true_val{BuiltinConstantKind::True, Types::BoolTy};
    BuiltinConstant false_val{BuiltinConstantKind::False, Types::BoolTy};

public:
    Block* insert_point = nullptr;

    Builder(TranslationUnit& tu);

    auto Allocate(usz sz, usz align) -> void* { return alloc.Allocate(sz, align); }
    auto GetExistingProc(StringRef name) -> Ptr<Proc>;
    auto GetOrCreateProc(String s, Linkage link, ProcType* ty) -> Proc*;

    auto CreateAdd(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateAlloca(Type ty) -> Value*;
    auto CreateAnd(Value* a, Value* b) -> Value*;
    auto CreateAShr(Value* a, Value* b) -> Value*;
    auto CreateBlock() -> std::unique_ptr<Block> { return CreateBlock(ArrayRef<Type>{}); }
    auto CreateBlock(ArrayRef<Value*> args) -> std::unique_ptr<Block>;
    auto CreateBlock(ArrayRef<Type> args) -> std::unique_ptr<Block>;
    auto CreateBlock(Proc* proc, ArrayRef<Type> args) -> Block*;
    auto CreateBool(bool value) -> Value*;
    void CreateBr(Block* dest, ArrayRef<Value*> args);
    auto CreateCall(Value* callee, ArrayRef<Value*> args) -> Value*;
    void CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block);
    auto CreateExtractValue(Value* aggregate, u32 idx) -> Value*;
    auto CreateInt(APInt val) -> Value*;
    auto CreateInt(i64 val, Type type = Types::IntTy) -> Value*;
    auto CreateIMul(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateICmpEq(Value* a, Value* b) -> Value*;
    auto CreateICmpNe(Value* a, Value* b) -> Value*;
    auto CreateICmpULt(Value* a, Value* b) -> Value*;
    auto CreateICmpULe(Value* a, Value* b) -> Value*;
    auto CreateICmpUGe(Value* a, Value* b) -> Value*;
    auto CreateICmpUGt(Value* a, Value* b) -> Value*;
    auto CreateICmpSLt(Value* a, Value* b) -> Value*;
    auto CreateICmpSLe(Value* a, Value* b) -> Value*;
    auto CreateICmpSGe(Value* a, Value* b) -> Value*;
    auto CreateICmpSGt(Value* a, Value* b) -> Value*;
    auto CreateLoad(Type ty, Value* ptr) -> Value*;
    auto CreateLShr(Value* a, Value* b) -> Value*;
    void CreateMemZero(Value* addr, Value* bytes);
    auto CreateNil(Type ty) -> Value*;
    auto CreateOr(Value* a, Value* b) -> Value*;
    auto CreatePoison(Type ty) -> Value*;
    auto CreatePtrAdd(Value* ptr, Value* offs, bool inbounds) -> Value*;
    void CreateReturn(Value* val = nullptr);
    auto CreateSAddOverflow(Value* a, Value* b) -> OverflowResult;
    auto CreateSDiv(Value* a, Value* b) -> Value*;
    auto CreateSelect(Value* cond, Value* then_val, Value* else_val) -> Value*;
    auto CreateShl(Value* a, Value* b) -> Value*;
    auto CreateSICast(Value* i, Type to_type) -> Value*;
    auto CreateSlice(Value* data, Value* size) -> Slice*;
    auto CreateSMulOverflow(Value* a, Value* b) -> OverflowResult;
    auto CreateSRem(Value* a, Value* b) -> Value*;
    auto CreateSSubOverflow(Value* a, Value* b) -> OverflowResult;
    void CreateStore(Value* val, Value* ptr);
    auto CreateString(String s) -> Slice*;
    auto CreateSub(Value* a, Value* b, bool nowrap = false) -> Value*;
    auto CreateUDiv(Value* a, Value* b) -> Value*;
    void CreateUnreachable();
    auto CreateURem(Value* a, Value* b) -> Value*;
    auto CreateXor(Value* a, Value* b) -> Value*;

private:
    template <std::derived_from<Inst> InstTy, typename ...Args>
    auto CreateImpl(Args&&... args) -> Inst*;

    template <std::derived_from<Inst> InstTy, typename ...Args>
    auto CreateSpecialGetVal(Type val_ty, Args&&... args) -> Value*;

    auto Create(Op op, ArrayRef<Value*> vals) -> Inst*;
    auto CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> Value*;
    auto Result(Inst* i, Type ty, u32 idx) -> InstValue*;
};
}

#endif //SRCC_CG_IR_HH
