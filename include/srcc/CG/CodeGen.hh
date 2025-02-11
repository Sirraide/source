#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

/*
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>*/

#include <base/Assert.hh>

namespace srcc::cg {
class CodeGen;
class LLVMCodeGen;
class VMCodeGen;
}

class srcc::cg::CodeGen : DiagsProducer<std::nullptr_t>, ir::Builder {
    LIBBASE_IMMOVABLE(CodeGen);
    struct Mangler;
    friend DiagsProducer;
    friend LLVMCodeGen;

    Size word_size;
    Opt<ir::Proc*> assert_handler;
    Opt<ir::Proc*> overflow_handler;
    Opt<ir::Proc*> printf;
    DenseMap<LocalDecl*, ir::Value*> locals;
    DenseMap<ProcDecl*, String> mangled_names;
    ir::Proc* curr_proc = nullptr;

    struct InvertTy {} static constexpr Invert;

public:
    CodeGen(TranslationUnit& tu, Size word_size) : Builder{tu}, word_size{word_size} {}

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& { return tu.context().diags(); }

    /// Dump the IR module.
    auto dump() -> SmallUnrenderedString { return Dump(); }

    /// Emit a procedure.
    void emit(ProcDecl* proc) { EmitProcedure(proc); }

    /// Emit LLVM IR.
    auto emit_llvm(llvm::TargetMachine& target) -> std::unique_ptr<llvm::Module>;

private:
    class EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);

        CodeGen& CG;
        ir::Proc* old_func;
        InsertPointGuard guard;

    public:
        EnterProcedure(CodeGen& CG, ir::Proc* func);
        ~EnterProcedure() { CG.curr_proc = old_func; }
    };

    void CreateArithFailure(ir::Value* failure_cond, Tk op, Location loc, String name = "integer overflow");

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        tu.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclareAssertFailureHandler() -> ir::Value*;
    auto DeclareArithmeticFailureHandler() -> ir::Value*;
    auto DeclarePrintf() -> ir::Value*;
    auto DeclareProcedure(ProcDecl* proc) -> ir::Proc*;
    auto DefineExp(Type ty) -> ir::Proc*;

    void Emit(ArrayRef<ProcDecl*> procs);
    auto Emit(Stmt* stmt) -> ir::Value*;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit## Class(Class* stmt)->ir::Value*;
#include "srcc/AST.inc"


    auto EmitArithmeticOrComparisonOperator(Tk op, ir::Value* lhs, ir::Value* rhs, Location loc) -> ir::Value*;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(const eval::Value& val) -> ir::Value*;

    void EmitLocal(LocalDecl* decl);

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
    ) -> ArrayRef<ir::Argument*>;

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

    /// Initialise a variable or memory location.
    void PerformVariableInitialisation(ir::Value* addr, Expr* init);

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


/*class srcc::cg::CGLLVM : public CodeGen {
    llvm::TargetMachine& machine;
    std::unique_ptr<llvm::Module> llvm;
    StringMap<llvm::Constant*> strings;
    llvm::IRBuilder<> builder;

    llvm::IntegerType* const IntTy;
    llvm::IntegerType* const I1Ty;
    llvm::IntegerType* const I8Ty;
    llvm::PointerType* const PtrTy;
    llvm::IntegerType* const FFIIntTy;
    llvm::StructType* const SliceTy;
    llvm::StructType* const ClosureTy;
    llvm::Type* const VoidTy;

    bool finalised = false;

public:
    CGLLVM(TranslationUnit& tu, llvm::TargetMachine& machine);

private:
    /// Perform any finalisation steps that need to be run after all code has been emitted.
    void finalise();

    /// Functions that create instructions.
    auto CreateBool(bool value) -> ir::Value* override;
    auto CreateCall(ir::Value* callee, ArrayRef<ir::Value*> args) -> ir::Value* override;
    auto CreateEmptySlice() -> ir::Value* override;
    auto CreateICmp(Tk op, ir::Value* a, ir::Value* b) -> ir::Value* override;
    auto CreateInt(const APInt& val) -> ir::Value* override;
    auto CreateInt(i64 val, Type type) -> ir::Value* override;
    auto CreateIMul(ir::Value* a, ir::Value* b) -> ir::Value* override;
    auto CreatePtrAdd(ir::Value* ptr, ir::Value* offs, bool inbounds) -> ir::Value* override;
    auto CreateStringSlice(StringRef s) -> ir::Value* override;
    void CreateUnreachable() override;

    static void OptimiseModule(llvm::TargetMachine& machine, TranslationUnit& tu, llvm::Module& compiled);
    static int EmitModuleOrProgram(
        llvm::TargetMachine& machine,
        TranslationUnit& tu,
        llvm::Module& compiled,
        ArrayRef<std::string> additional_objects,
        StringRef program_file_name_override = ""
    );

    auto GetStringPtr(StringRef s) -> llvm::Constant*;

    auto ConvertCC(CallingConvention cc) -> llvm::CallingConv::ID;
    auto ConvertLinkage(Linkage lnk) -> llvm::GlobalValue::LinkageTypes;

    template <typename Ty = llvm::Type>
    auto ConvertType(Type ty, bool array_elem = false) -> Ty* { return cast<Ty>(ConvertTypeImpl(ty, array_elem)); }
    auto ConvertTypeImpl(Type ty, bool array_elem) -> llvm::Type*;
};*/

#endif // SRCC_CG_HH
