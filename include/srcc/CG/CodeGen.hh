#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <base/Assert.hh>

namespace srcc::cg {
class CodeGen;
class LLVMCodeGen;
class VMCodeGen;
class ArgumentMapping;

namespace detail {
class CodeGenBase {
protected:
    mlir::MLIRContext mlir;
};
}
}

class srcc::cg::ArgumentMapping {
    LIBBASE_IMMOVABLE(ArgumentMapping);

    llvm::SmallDenseMap<u32, u32> indices;

public:
    ArgumentMapping() = default;

    /// Map an index to another index
    auto map(u32 a, u32 b) { indices[a] = b; }

    /// Get the IR arg index that a AST arg maps to; if nullopt is
    /// returned, this argument should simply be dropped.
    auto map(u32 idx) const -> std::optional<u32> {
        auto it = indices.find(idx);
        if (it == indices.end()) return std::nullopt;
        return it->second;
    }
};

class srcc::cg::CodeGen : DiagsProducer<std::nullptr_t>, detail::CodeGenBase, mlir::OpBuilder {
    LIBBASE_IMMOVABLE(CodeGen);
    struct Printer;
    struct Mangler;
    friend DiagsProducer;
    friend LLVMCodeGen;

    TranslationUnit& tu;
    Opt<ir::ProcOp> printf;
    DenseMap<LocalDecl*, Value> locals;
    DenseMap<ProcDecl*, ir::ProcOp> declared_procs;
    DenseMap<ProcDecl*, String> mangled_names;
    DenseMap<ProcDecl*, std::unique_ptr<const ArgumentMapping>> argument_mapping;
    mlir::ModuleOp mlir_module;
    ir::ProcOp vm_entry_point;
    ir::ProcOp curr_proc;
    Size word_size;
    LangOpts lang_opts;
    mlir::Type ptr_ty;
    mlir::Type int_ty;
    mlir::Type closure_ty;
    mlir::Type slice_ty;
    mlir::Type bool_ty;
    usz strings = 0;

public:
    CodeGen(TranslationUnit& tu, LangOpts lang_opts, Size word_size);

    /// Get the diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine& { return tu.context().diags(); }

    /// Dump the IR module.
    [[nodiscard]] auto dump(bool verbose = false, bool generic = false) -> SmallUnrenderedString;

    /// Emit a procedure.
    void emit(ProcDecl* proc) { EmitProcedure(proc); }

    /// Emit a statement into a separate procedure.
    ///
    /// This is used exclusively to prepare a statement for
    /// constant evaluation.
    [[nodiscard]] auto emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::ProcOp;

    /// Emit LLVM IR.
    [[nodiscard]] auto emit_llvm(llvm::TargetMachine& target) -> std::unique_ptr<llvm::Module>;

    /// Finalise IR.
    [[nodiscard]] bool finalise();

    /// Optimise a module.
    void optimise(llvm::TargetMachine& target, TranslationUnit& tu, llvm::Module& module);

    /// Write the module to a file.
    int write_to_file(
        llvm::TargetMachine& machine,
        TranslationUnit& tu,
        llvm::Module& m,
        ArrayRef<std::string> additional_objects,
        StringRef program_file_name_override
    );

private:
    class EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);

        CodeGen& CG;
        ir::ProcOp old_func;
        InsertionGuard guard;

    public:
        EnterProcedure(CodeGen& CG, ir::ProcOp func);
        ~EnterProcedure() { CG.curr_proc = old_func; }
    };

    struct IRProcType {
        mlir::FunctionType type;
        bool has_indirect_return = false;
    };

    // AST -> IR converters
    auto C(CallingConvention l) -> mlir::LLVM::CConv;
    auto C(Linkage l) -> mlir::LLVM::Linkage;
    auto C(Location l) -> mlir::Location;
    auto C(Type ty) -> mlir::Type;

    auto ConvertProcType(ProcType* ty) -> IRProcType;
    auto CreateAggregate(Location loc, Value a, Value b) -> Value;
    auto CreateAlloca(Location loc, Type ty) -> Value;

    void CreateArithFailure(
        Value failure_cond,
        Tk op,
        Location loc,
        String name = "integer overflow"
    );

    template <typename Unchecked, typename Checked>
    auto CreateBinop(
        Type ty,
        Value lhs,
        Value rhs,
        Location loc,
        Tk op
    ) -> Value;

    auto CreateBlock(ArrayRef<mlir::Type> args = {}) -> std::unique_ptr<Block>;
    auto CreateBool(Location loc, bool b) -> Value;
    auto CreateGlobalStringPtr(Align align, String data, bool null_terminated) -> Value;
    auto CreateGlobalStringPtr(String data) -> Value;
    auto CreateGlobalStringSlice(Location loc, String data) -> Value;
    auto CreateICmp(mlir::Location loc, mlir::LLVM::ICmpPredicate pred, Value lhs, Value rhs) -> Value;
    auto CreateInt(const APInt& value, Type ty) -> Value;
    auto CreateInt(i64 value, Type ty = Type::IntTy) -> Value;
    auto CreateInt(i64 value, mlir::Type ty) -> Value;
    auto CreateNil(Location loc, Type ty) -> Value;
    auto CreateNullPointer(Location loc) -> Value;
    auto CreatePtrAdd(mlir::Location loc, Value addr, Value offs) -> Value;
    auto CreatePtrAdd(mlir::Location loc, Value addr, Size offs) -> Value;
    auto CreateSICast(mlir::Location loc, Value val, Type from, Type to) -> Value;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        tu.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclarePrintf() -> ir::ProcOp;
    auto DeclareProcedure(ProcDecl* proc) -> ir::ProcOp;

    void Emit(ArrayRef<ProcDecl*> procs);
    auto Emit(Stmt* stmt) -> Value;
    void EmitArrayBroadcast(Type elem_ty, Value addr, u64 elements, Expr* initialiser, Location loc);
    void EmitArrayBroadcastExpr(ArrayBroadcastExpr* e, Value mrvalue_slot);
    void EmitArrayInitExpr(ArrayInitExpr* e, Value mrvalue_slot);
    auto EmitCallExpr(CallExpr* call, Value mrvalue_slot) -> Value;
    auto EmitBlockExpr(BlockExpr* expr, Value mrvalue_slot) -> Value;
    auto EmitIfExpr(IfExpr* expr, Value mrvalue_slot) -> Value;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit##Class(Class* stmt)->Value;
#include "srcc/AST.inc"

    auto EmitArithmeticOrComparisonOperator(Tk op, Type ty, Value lhs, Value rhs, Location loc) -> Value;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(Location loc, const eval::RValue& val) -> Value;

    /// Emit any (lvalue, srvalue, mrvalue) initialiser into a memory location.
    void EmitInitialiser(Value addr, Expr* init);
    void EmitLocal(LocalDecl* decl);

    /// Emit an mrvalue into a memory location.
    void EmitMRValue(Value addr, Expr* init);

    auto EnterBlock(std::unique_ptr<Block> bb, mlir::ValueRange args = {}) -> Block*;
    auto EnterBlock(Block* bb, mlir::ValueRange args = {}) -> Block*;

    /// Get a procedure, declaring it if it doesn’t exist yet.
    auto GetOrCreateProc(Location loc, String name, Linkage linkage, ProcType* ty) -> ir::ProcOp;

    /// Check whether a procedure has an indirect return type.
    bool HasIndirectReturn(ProcType* type);

    /// Create a conditional branch and join block.
    ///
    /// This creates two blocks: a 'then' and a 'join' block. Control
    /// flow branches to 'then' if the condition is true, where the
    /// callback is used to emit its body, after which we branch to
    /// 'join'. If the condition is false, we branch to 'join' directly.
    ///
    /// The builder is positioned at the end of the join block after this
    /// returns.
    ///
    /// \return The join block.
    auto If(
        Value cond,
        mlir::ValueRange args,
        llvm::function_ref<void()> emit_body
    ) -> Block*;

    auto If(
        Value cond,
        llvm::function_ref<void()> emit_body
    ) { return If(cond, {}, emit_body); }

    /// Create a branch that can return a value.
    ///
    /// If 'emit_else' is not null, this creates three blocks: 'then',
    /// 'else', and 'join'. The condition is emitted, and if it is true,
    /// we branch to 'then', otherwise, we branch to 'else'. After 'then'
    /// and 'else' are emitted, we branch to 'join', if the blocks are
    /// not already terminated. If both branches return non-null values
    /// then a phi is created at the join block and returned.
    ///
    /// Otherwise, no 'else' block is created, and the false branch of
    /// the conditional branch is the join block.
    ///
    /// The builder is positioned at the end of the join block after this
    /// returns.
    ///
    /// \return The join block.
    auto If(
        Value cond,
        llvm::function_ref<Value()> emit_then,
        llvm::function_ref<Value()> emit_else
    ) -> Block*;

    /// Check if the size of a type is zero; this also means that every
    /// instance of this type is (or would be) identical.
    bool IsZeroSizedType(Type ty);

    /// Check if a local variable has a stack slot.
    bool LocalNeedsAlloca(LocalDecl* local);

    /// Create an infinite loop.
    ///
    /// The arguments, as well as the values returned from the callback,
    /// are passed to the condition block of the loop. The callback may
    /// return an empty vector if the loop is infinite or has no arguments.
    void Loop(llvm::function_ref<void()> emit_body);

    auto MangledName(ProcDecl* proc) -> String;

    /// Opposite of If().
    void Unless(
        Value cond,
        llvm::function_ref<void()> emit_else
    );

    /// Create a while loop.
    void While(
        llvm::function_ref<Value()> emit_cond,
        llvm::function_ref<void()> emit_body
    );
};

#endif // SRCC_CG_HH
