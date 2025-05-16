#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/CG/IR.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <base/Assert.hh>

namespace srcc::cg {
class CallLowering;
class CodeGen;
class LLVMCodeGen;
class VMCodeGen;
}

class srcc::cg::CallLowering {
    LIBBASE_IMMOVABLE(CallLowering);

protected:
    CodeGen& CG;
    CallLowering(CodeGen& cg) : CG(cg) {}

public:
    virtual ~CallLowering() = default;

    /// Adjust a procedure type and convert it to something that conforms with
    /// the target ABI.
    [[nodiscard]] virtual auto adjust_procedure_type(
        ProcType* ty
    ) -> std::pair<ProcType*, ir::Proc::ParamAttrMap> = 0;

    /// Whether this type must be returned indirectly by passing a pointer
    /// to a preallocated memory buffer to the function.
    [[nodiscard]] virtual bool has_indirect_return(ProcType* ty) = 0;

    /// Emit a call to a source-level procedure.
    ///
    /// Perform any necessary processing required to convert the logical arguments
    /// to a function into something that conforms with the calling convention.
    ///
    /// \param ty The source-level type of the procedure being called.
    /// \param callee The procedure being called, possibly indirectly.
    /// \param mrvalue_slot Slot for the return value; must be non-null iff
    ///    has_indirect_return() returns true for \p ty.
    /// \param args Source-level call arguments.
    /// \return The result of the call.
    [[nodiscard]] virtual auto lower_call(
        ProcType* ty,
        ir::Aggregate* callee,
        ir::Value* mrvalue_slot,
        ArrayRef<ir::Value*> args
    ) -> ir::Value* = 0;

    /// Emit logical parameters.
    virtual void lower_params(ir::Proc* proc) = 0;

protected:
    /// Create an instance of a type from a set of values as passed or returned in
    /// registers and a layout representing them.
    [[nodiscard]] auto AssembleTypeFromRegisters(
        Type ty,
        StructLayout* register_layout,
        ArrayRef<ir::Value*> register_vals
    ) -> ir::Value*;
};

class srcc::cg::CodeGen : DiagsProducer<std::nullptr_t>
    , public ir::Builder {
    LIBBASE_IMMOVABLE(CodeGen);
    struct Mangler;
    friend DiagsProducer;
    friend LLVMCodeGen;
    friend CallLowering;

public:
    Opt<ir::Proc*> printf;
    DenseMap<LocalDecl*, ir::Value*> locals;
    DenseMap<ProcDecl*, ir::Proc*> declared_procs;
    DenseMap<ProcDecl*, String> mangled_names;
    ir::Proc* curr_proc = nullptr;
    Size word_size;
    LangOpts lang_opts;

    // Call lowering implementation. One of these exists for every calling convention.
    DenseMap<CallingConvention, std::unique_ptr<CallLowering>> call_lowering;

    CodeGen(TranslationUnit& tu, LangOpts lang_opts, Size word_size);

    /// Get the diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine& { return tu.context().diags(); }

    /// Dump the IR module.
    [[nodiscard]] auto dump() -> SmallUnrenderedString { return Dump(); }

    /// Emit a procedure.
    void emit(ProcDecl* proc) { EmitProcedure(proc); }

    /// Emit a statement into a separate procedure.
    ///
    /// This is used exclusively to prepare a statement for
    /// constant evaluation.
    [[nodiscard]] auto emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::Proc*;

    /// Emit LLVM IR.
    [[nodiscard]] auto emit_llvm(llvm::TargetMachine& target) -> std::unique_ptr<llvm::Module>;

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

    class EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);

        CodeGen& CG;
        ir::Proc* old_func;
        InsertPointGuard guard;

    public:
        EnterProcedure(CodeGen& CG, ir::Proc* func);
        ~EnterProcedure() { CG.curr_proc = old_func; }
    };

    void CreateArithFailure(
        ir::Value* failure_cond,
        Tk op,
        Location loc,
        String name = "integer overflow"
    );

    auto CreateBinop(
        ir::Value* lhs,
        ir::Value* rhs,
        Location loc,
        Tk op,
        auto (Builder::*build_unchecked)(ir::Value*, ir::Value*, bool)->ir::Value*,
        auto (Builder::*build_overflow)(ir::Value*, ir::Value*)->ir::OverflowResult
    );

    auto CreateNativeCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering>;
    auto CreateSourceCallLowering_X86_64_Linux() -> std::unique_ptr<CallLowering>;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        tu.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclareAssertFailureHandler() -> ir::Value*;
    auto DeclareArithmeticFailureHandler() -> ir::Value*;
    auto DeclarePrintf() -> ir::Proc*;
    auto DeclareProcedure(ProcDecl* proc) -> ir::Proc*;

    void Emit(ArrayRef<ProcDecl*> procs);
    auto Emit(Stmt* stmt) -> ir::Value*;
    auto EmitCallExpr(CallExpr* call, ir::Value* mrvalue_slot) -> ir::Value*;
    auto EmitBlockExpr(BlockExpr* expr, ir::Value* mrvalue_slot) -> ir::Value*;
    auto EmitIfExpr(IfExpr* expr, ir::Value* mrvalue_slot) -> ir::Value*;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit##Class(Class* stmt)->ir::Value*;
#include "srcc/AST.inc"

    auto EmitArithmeticOrComparisonOperator(Tk op, ir::Value* lhs, ir::Value* rhs, Location loc) -> ir::Value*;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(const eval::RValue& val) -> ir::Value*;

    /// Emit any (lvalue, srvalue, mrvalue) initialiser into a memory location.
    void EmitInitialiser(ir::Value* addr, Expr* init);
    void EmitLocal(LocalDecl* decl);

    /// Emit an mrvalue into a memory location.
    void EmitMRValue(ir::Value* addr, Expr* init);

    auto EnterBlock(std::unique_ptr<ir::Block> bb, ArrayRef<ir::Value*> args = {}) -> ir::Block*;
    auto EnterBlock(ir::Block* bb, ArrayRef<ir::Value*> args = {}) -> ir::Block*;

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
        ir::Value* cond,
        ArrayRef<ir::Value*> args,
        llvm::function_ref<void()> emit_body
    ) -> ir::Block*;

    auto If(
        ir::Value* cond,
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
    /// \return The argument values of the join block, which correspond to
    /// the values returned from emit_then and emit_else.
    auto If(
        ir::Value* cond,
        llvm::function_ref<ir::Value*()> emit_then,
        llvm::function_ref<ir::Value*()> emit_else
    ) -> ArrayRef<ir::Value*>;

    /// Check if the size of a type is zero; this also means that every
    /// instance of this type is (or would be) identical.
    bool IsZeroSizedType(Type ty);

    /// Create an infinite loop.
    ///
    /// The arguments, as well as the values returned from the callback,
    /// are passed to the condition block of the loop. The callback may
    /// return an empty vector if the loop is infinite or has no arguments.
    void Loop(
        ArrayRef<ir::Value*> block_args,
        llvm::function_ref<SmallVector<ir::Value*>()> emit_body
    );

    auto MangledName(ProcDecl* proc) -> String;

    /// Opposite of If().
    void Unless(
        ir::Value* cond,
        llvm::function_ref<void()> emit_else
    );

    /// Create a while loop.
    void While(
        llvm::function_ref<ir::Value*()> emit_cond,
        llvm::function_ref<void()> emit_body
    );
};

#endif // SRCC_CG_HH
