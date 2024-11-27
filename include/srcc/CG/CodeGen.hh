#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include <base/Assert.hh>

namespace srcc {
class CodeGen;
}

class srcc::CodeGen : DiagsProducer<std::nullptr_t> {
    struct Mangler;
    friend DiagsProducer;

    llvm::TargetMachine& machine;
    TranslationUnit& M;
    std::unique_ptr<llvm::Module> llvm;
    StringMap<llvm::Constant*> strings;
    DenseMap<LocalDecl*, llvm::Value*> locals;
    DenseMap<ProcDecl*, std::string> mangled_names;
    DenseMap<llvm::Type*, llvm::FunctionCallee> exp_funcs;
    llvm::IRBuilder<> builder;
    llvm::Function* curr_func{};

    llvm::IntegerType* const IntTy;
    llvm::IntegerType* const I1Ty;
    llvm::IntegerType* const I8Ty;
    llvm::PointerType* const PtrTy;
    llvm::IntegerType* const FFIIntTy;
    llvm::StructType* const SliceTy;
    llvm::StructType* const ClosureTy;
    llvm::Type* const VoidTy;

    Opt<llvm::FunctionCallee> assert_failure_handler;
    Opt<llvm::FunctionCallee> overflow_handler;

    CodeGen(llvm::TargetMachine& machine, TranslationUnit& M);

public:
    static auto Emit(llvm::TargetMachine& machine, TranslationUnit& M) -> std::unique_ptr<llvm::Module> {
        CodeGen CG{machine, M};
        CG.Emit();
        return std::move(CG.llvm);
    }

    static int EmitModuleOrProgram(
        llvm::TargetMachine& machine,
        TranslationUnit& tu,
        llvm::Module& compiled,
        ArrayRef<std::string> additional_objects,
        StringRef program_file_name_override = ""
    );

    static void OptimiseModule(llvm::TargetMachine& machine, TranslationUnit& tu, llvm::Module& compiled);

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& { return M.context().diags(); }

private:
    class EnterFunction {
        SRCC_IMMOVABLE(EnterFunction);

        CodeGen& CG;
        llvm::Function* old_func;
        llvm::IRBuilder<>::InsertPointGuard guard;

    public:
        EnterFunction(CodeGen& CG, llvm::Function* func);
        ~EnterFunction() { CG.curr_func = old_func; }
    };

    auto ConvertCC(CallingConvention cc) -> llvm::CallingConv::ID;
    auto ConvertLinkage(Linkage lnk) -> llvm::GlobalValue::LinkageTypes;

    template <typename Ty = llvm::Type>
    auto ConvertType(Type ty, bool array_elem = false) -> Ty* { return cast<Ty>(ConvertTypeImpl(ty, array_elem)); }
    auto ConvertTypeImpl(Type ty, bool array_elem) -> llvm::Type*;
    auto ConvertTypeForMem(Type ty) -> llvm::Type*;
    auto ConvertProcType(ProcType* ty) -> llvm::FunctionType*;

    void CreateArithFailure(llvm::Value* cond, Tk op, Location loc, StringRef name = "integer overflow");

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        M.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclareAssertFailureHandler() -> llvm::FunctionCallee;
    auto DeclareArithmeticFailureHandler() -> llvm::FunctionCallee;
    auto DeclareProcedure(ProcDecl* proc) -> llvm::FunctionCallee;
    auto DefineExp(llvm::Type* ty) -> llvm::FunctionCallee;

    void Emit();
    auto Emit(Stmt* stmt) -> llvm::Value*;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit## Class(Class* stmt)->llvm::Value*;
#include "srcc/AST.inc"


    auto EmitArithmeticOrComparisonOperator(Tk op, llvm::Value* lhs, llvm::Value* rhs, Location loc) -> llvm::Value*;
    auto EmitClosure(ProcDecl* proc) -> llvm::Constant*;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(const eval::Value& val) -> llvm::Constant*;

    void EmitLocal(LocalDecl* decl);

    auto EnterBlock(llvm::BasicBlock* bb) -> llvm::BasicBlock*;

    /// Same as CreateGlobalStringPtr(), but is interned.
    auto GetStringPtr(StringRef s) -> llvm::Constant*;

    /// Create a constant slice for a string.
    auto GetStringSlice(StringRef s) -> llvm::Constant*;

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
        llvm::Value* cond,
        llvm::function_ref<void()> emit_body
    ) -> llvm::BasicBlock*;

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
    /// \return A PHI node that contains the values of both branches, or
    /// null if there is no else branch or if either branch returns null.
    auto If(
        llvm::Value* cond,
        llvm::function_ref<llvm::Value*()> emit_then,
        llvm::function_ref<llvm::Value*()> emit_else
    ) -> llvm::PHINode*;

    /// Create an infinite loop.
    ///
    /// The block before the condition block from which the loop is
    /// first entered is passed as a parameter to the callback.
    void Loop(llvm::function_ref<void(llvm::BasicBlock*)> emit_body);

    auto MakeInt(const APInt& val) -> llvm::ConstantInt*;
    auto MakeInt(u64 integer) -> llvm::ConstantInt*;
    auto MangledName(ProcDecl* proc) -> StringRef;

    /// Initialise a variable or memory location.
    void PerformVariableInitialisation(llvm::Value* addr, Expr* init);

    /// Create a while loop.
    void While(
        llvm::function_ref<llvm::Value*()> emit_cond,
        llvm::function_ref<void()> emit_body
    );
};

#endif // SRCC_CG_HH
