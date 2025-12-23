#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <srcc/CG/ABI.hh>
#include <srcc/CG/Target/Target.hh>
#include <base/Assert.hh>
#include <base/FixedVector.hh>
#include <mlir/Pass/PassManager.h>

namespace srcc::cg {
class CodeGen;
class LLVMCodeGen;
class VMCodeGen;
class ArgumentMapping;
class IRValue;
class ProcData;
enum class EvalMode : u8;

namespace detail {
class CodeGenBase {
protected:
    mlir::MLIRContext mlir;
};
}
} // namespace srcc::cg

namespace mlir {
namespace arith {
enum class CmpIPredicate : uint64_t;
}
}

/// This is either a single SSA value or a pair of SSA values; the
/// latter is used for slices, ranges, and closures.
class srcc::cg::IRValue {
    Value vals[2]{};

public:
    IRValue() = default;
    IRValue(std::nullptr_t) = delete("Use '{}' instead to create an empty SRValue or RType");
    IRValue(Value first_val) : vals{first_val, nullptr} {}
    IRValue(Value first_val, Value second_val) : vals{first_val, second_val} {
        Assert(vals[0] and vals[1]);
    }

    template <utils::ConvertibleRange<Value> T>
    IRValue(T r) {
        Assert(r.size() <= 2);
        if (r.size() >= 1) vals[0] = r[0];
        if (r.size() >= 2) vals[1] = r[1];
    }

    IRValue(mlir::ValueRange r) {
        Assert(r.size() <= 2);
        if (r.size() >= 1) vals[0] = r[0];
        if (r.size() >= 2) vals[1] = r[1];
    }

    template <std::derived_from<mlir::OpState> Op>
    requires (Op::template hasTrait<mlir::OpTrait::OneResult>())
    IRValue(Op op) {
        vals[0] = op->getResult(0);
    }

    template <std::derived_from<mlir::OpState> Op>
    requires (Op::template hasTrait<::mlir::OpTrait::NResults<2>::Impl>())
    IRValue(Op op) {
        vals[0] = op->getResult(0);
        vals[1] = op->getResult(1);
    }

    /// Allow treating this as a range.
    [[nodiscard]] auto begin() const { return operator mlir::ValueRange().begin(); }
    [[nodiscard]] auto end() const { return operator mlir::ValueRange().end(); }

    /// Run a function on each element of this range.
    auto each(auto cb) {
        using Res = std::invoke_result_t<decltype(cb), Value>;
        if constexpr (std::is_same_v<Res, void>) {
            if (vals[0]) std::invoke(cb, vals[0]);
            if (vals[1]) std::invoke(cb, vals[1]);
        } else {
            SmallVector<Res, 2> results;
            if (vals[0]) results.push_back(std::invoke(cb, vals[0]));
            if (vals[1]) results.push_back(std::invoke(cb, vals[1]));
            return results;
        }
    }

    /// Add an additional value.
    auto extend(Value e) {
        if (vals[0] == nullptr) vals[0] = e;
        else if (vals[1] == nullptr) vals[1] = e;
        else Unreachable();
    }

    /// Get the first value of an aggregate.
    [[nodiscard]] auto first() const -> Value {
        Assert(is_aggregate());
        return vals[0];
    }

    /// Copy the types into a vector.
    void into(SmallVectorImpl<Value>& container) const {
        if (vals[0]) container.push_back(vals[0]);
        if (vals[1]) container.push_back(vals[1]);
    }

    /// Check if this is an aggregate.
    [[nodiscard]] bool is_aggregate() const { return vals[1] != nullptr; }

    /// Check if this is an empty value.
    [[nodiscard]] auto is_null() const { return vals[0] == nullptr; }

    /// Check if this is a scalar value.
    [[nodiscard]] bool is_scalar() const {
        return not is_null() and not is_aggregate();
    }

    /// Get the second value of an aggregate.
    [[nodiscard]] auto second() const -> Value {
        Assert(is_aggregate());
        return vals[1];
    }

    /// Get this as a single scalar value.
    [[nodiscard]] auto scalar() const -> Value {
        Assert(is_scalar());
        return vals[0];
    }

    /// Get the first value if this is an aggregate or scalar.
    [[nodiscard]] auto scalar_or_first() const -> Value {
        Assert(not is_null());
        return vals[0];
    }

    /// Get the n-th element.
    [[nodiscard]] auto operator[](unsigned i) const -> Value {
        Assert(is_aggregate());
        Assert(i <= 2);
        return vals[i];
    }

    /// Convert this to a range of types.
    operator mlir::ValueRange() const {
        if (is_null()) return {};
        if (is_scalar()) return ArrayRef{vals, 1};
        return ArrayRef{vals, 2};
    }

    /// Check if this is empty.
    explicit operator bool() const { return not is_null(); }

    /// Get a location describing this value.
    [[nodiscard]] auto loc() const -> mlir::Location {
        Assert(not is_null());
        return scalar_or_first().getLoc();
    }
};

/// Value category of an rvalue used for evaluation; this is strictly
/// type-dependent.
///
/// This used to be expressed by splitting RValue into SRValue and
/// MRValue, but this distinction has proven unuseful outside codegen.
enum class srcc::cg::EvalMode : base::u8 {
    Scalar,
    Memory,
};

/// State used when emitting a procedure.
class srcc::cg::ProcData {
    LIBBASE_MOVE_ONLY(ProcData);

public:
    using Cleanup = std::move_only_function<void() const>;
    struct CleanupScope {
        CleanupScope* parent = nullptr;
        SmallVector<Cleanup> cleanups;
    };

    struct Loop {
        ProcData::CleanupScope* cleanup;
        mlir::Block* continue_block;
        mlir::Block* break_block;
        Loop(ProcData::CleanupScope* cleanup, mlir::Block* continue_block, mlir::Block* break_block)
            : cleanup{cleanup}, continue_block{continue_block}, break_block{break_block} {
            Assert(cleanup);
            Assert(continue_block);
            Assert(break_block);
        }
    };

    DenseMap<LocalDecl*, Value> locals;
    SmallVector<Loop> loop_stack;
    Value environment_for_nested_procs;
    Value abort_info_slot;
    ir::ProcOp proc;
    CleanupScope root_cleanup_scope{};
    CleanupScope* current_cleanup_scope = nullptr;
    ProcDecl* decl = nullptr;

    ProcData() = default;
    ProcData(ir::ProcOp proc, ProcDecl* decl) : proc{proc}, decl{decl} {}
};

class srcc::cg::CodeGen : public DiagsProducer
    , detail::CodeGenBase
    , public mlir::OpBuilder {
    LIBBASE_IMMOVABLE(CodeGen);
    struct Printer;
    struct Mangler;
    friend DiagsProducer;
    friend LLVMCodeGen;

    TranslationUnit& tu;
    ProcData curr;
    Opt<ir::ProcOp> printf;
    DenseMap<ProcDecl*, ir::ProcOp> declared_procs;
    DenseMap<ir::ProcOp, ProcDecl*> proc_reverse_lookup;
    DenseMap<ProcDecl*, String> mangled_names;
    StringMap<mlir::LLVM::GlobalOp> interned_strings;
    DenseMap<GlobalDecl*, mlir::LLVM::GlobalOp> global_vars;
    mlir::ModuleOp mlir_module;
    ir::ProcOp vm_entry_point;
    LangOpts lang_opts;
    mlir::Type ptr_ty;
    mlir::Type int_ty;
    mlir::Type i128_ty;
    mlir::Type bool_ty;
    mlir::Type type_ty;
    mlir::Type tree_ty;
    usz strings = 0;

public:
    CodeGen(TranslationUnit& tu, LangOpts lang_opts);

    /// Get the current ABI.
    [[nodiscard]] auto abi() const -> const abi::ABI& { return tu.target().abi(); }

    /// Get the context.
    [[nodiscard]] auto context() const -> Context& { return tu.context(); }

    /// Get the diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine& { return tu.context().diags(); }

    /// Dump the IR module.
    [[nodiscard]] auto dump(bool verbose = false, bool generic = false) -> SmallUnrenderedString;

    /// Emit a procedure.
    void emit(ProcDecl* proc) { EmitProcedure(proc); }

    /// Emit a list of procedures; any procedures that are not referenced are omitted.
    void emit_as_needed(ArrayRef<ProcDecl*> procs) { Emit(procs); }

    /// Emit a statement into a separate procedure.
    ///
    /// This is used exclusively to prepare a statement for
    /// constant evaluation.
    [[nodiscard]] auto emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::ProcOp;

    /// Emit LLVM IR.
    [[nodiscard]] auto emit_llvm(llvm::TargetMachine& target) -> std::unique_ptr<llvm::Module>;

    /// Finalise IR.
    [[nodiscard]] bool finalise(bool verify);

    /// Finalise a single procedure.
    [[nodiscard]] bool finalise_for_constant_evaluation(ir::ProcOp proc);

    /// Get the MLIR pointer type.
    [[nodiscard]] auto get_ptr_ty() -> mlir::Type { return ptr_ty; }

    /// Given an IR procedure, attempt to find the Source procedure it corresponds to.
    [[nodiscard]] auto lookup(ir::ProcOp op) -> Ptr<ProcDecl>;

    /// Get the MLIR context.
    [[nodiscard]] auto mlir_context() -> mlir::MLIRContext* { return &mlir; }

    /// Optimise a module.
    void optimise(llvm::TargetMachine& target, TranslationUnit& tu, llvm::Module& module);

    /// Get the translation unit we’re compiling.
    [[nodiscard]] auto translation_unit() -> TranslationUnit& { return tu; }

    /// Write the module to a file.
    int write_to_file(
        llvm::TargetMachine& machine,
        TranslationUnit& tu,
        llvm::Module& m,
        ArrayRef<std::string> lib_paths,
        ArrayRef<std::string> link_libs,
        ArrayRef<std::string> additional_objects,
        StringRef program_file_name_override
    );

    class EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);

        CodeGen& CG;
        ProcData old;
        InsertionGuard guard;

    public:
        EnterProcedure(CodeGen& CG, ir::ProcOp func, ProcDecl* old_proc);
        ~EnterProcedure();
    };

    class EnterCleanupScope {
        SRCC_IMMOVABLE(EnterCleanupScope);
        CodeGen& cg;
        ProcData::CleanupScope sc;

    public:
        EnterCleanupScope(CodeGen& CG);
        ~EnterCleanupScope();
        [[nodiscard]] auto scope() -> ProcData::CleanupScope* { return &sc; }
    };

    class EnterLoop {
        SRCC_IMMOVABLE(EnterLoop);
        CodeGen& cg;

    public:
        EnterLoop(CodeGen& CG, ProcData::Loop loop);
        ~EnterLoop();
    };

    struct RecordInitHelper {
        CodeGen& CG;
        RecordType* ty;
        Value base;
        usz i = 0;

        RecordInitHelper(CodeGen& CG, RecordType* ty, Value base) : CG{CG}, ty{ty}, base{base} {}
        void emit_next_field(Value v);
        void emit_next_field(IRValue v);
    };

    /// Add code to be run at end of scope.
    void AddCleanup(ProcData::Cleanup cleanup);

    /// Append a block to the current procedure.
    auto AppendBlock(std::unique_ptr<Block> bb) -> Block*;

    /// AST -> IR converters
    auto C(CallingConvention l) -> mlir::LLVM::CConv;
    auto C(Linkage l) -> mlir::LLVM::Linkage;
    auto C(SLoc l) -> mlir::Location;

    /// Convert a type to an IR type; does not support aggregates.
    auto C(Type ty, ValueCategory vc = Expr::RValue) -> mlir::Type;

    /// Attempt to lower a type to an MLIR type.
    auto TryConvertToMLIRType(Type ty) -> std::optional<mlir::Type>;

    /// Convert a type to an array of bytes whose dimension is the type size.
    auto ConvertToByteArrayType(Type ty) -> mlir::Type;

    auto CreateAlloca(mlir::Location loc, Type ty) -> Value;
    auto CreateAlloca(mlir::Location loc, Size sz, Align a) -> Value;
    void CreateAbort(mlir::Location loc, ir::AbortReason reason, IRValue msg1, IRValue msg2, IRValue stringifier);

    void CreateArithFailure(
        Value failure_cond,
        Tk op,
        mlir::Location loc,
        String name = "integer overflow"
    );

    template <typename Unchecked, typename Checked>
    auto CreateBinop(
        Type ty,
        Value lhs,
        Value rhs,
        mlir::Location loc,
        Tk op
    ) -> Value;

    auto CreateBlock(ArrayRef<mlir::Type> args = {}) -> std::unique_ptr<Block>;
    auto CreateBool(mlir::Location loc, bool b) -> Value;
    void CreateBuiltinAggregateStore(mlir::Location loc, Value addr, Type ty, IRValue aggregate);
    auto CreateEmptySlice(mlir::Location loc) -> IRValue;
    auto CreateGlobalStringPtr(Align align, String data, bool null_terminated) -> Value;
    auto CreateGlobalStringPtr(String data) -> Value;
    auto CreateGlobalStringSlice(mlir::Location loc, String data) -> IRValue;
    auto CreateICmp(mlir::Location loc, mlir::arith::CmpIPredicate pred, Value lhs, Value rhs) -> Value;
    auto CreateInt(mlir::Location loc, const APInt& value, Type ty) -> Value;
    auto CreateInt(mlir::Location loc, i64 value, Type ty = Type::IntTy) -> Value;
    auto CreateInt(mlir::Location loc, i64 value, mlir::Type ty) -> Value;
    auto CreateLoad(mlir::Location loc, Value addr, Type ty, Size offset = {}) -> IRValue;
    auto CreateLoad(mlir::Location loc, Value addr, mlir::Type ty, Align align, Size offset = {}) -> Value;
    void CreateMemCpy(mlir::Location loc, Value to, Value from, Type ty);
    auto CreateNullClosure(mlir::Location loc) -> IRValue;
    auto CreateNullPointer(mlir::Location loc) -> Value;
    void CreateReturn(mlir::Location loc, mlir::ValueRange values);
    auto CreatePtrAdd(mlir::Location loc, Value addr, Value offs) -> Value;
    auto CreatePtrAdd(mlir::Location loc, Value addr, Size offs) -> Value;
    auto CreateSICast(mlir::Location loc, Value val, Type from, Type to) -> Value;
    void CreateStore(mlir::Location loc, Value addr, Value val, Align align, Size offset = {});

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, SLoc where, std::format_string<Args...> fmt, Args&&... args) {
        tu.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclarePrintf() -> ir::ProcOp;
    auto DeclareProcedure(ProcDecl* proc) -> ir::ProcOp;

    void Emit(ArrayRef<ProcDecl*> procs);
    auto Emit(Stmt* stmt) -> IRValue;
    auto EmitWithCleanup(Stmt* stmt) -> IRValue;
    auto EmitScalar(Stmt* stmt) -> Value;
    void EmitArrayBroadcast(Type elem_ty, Value addr, u64 elements, Expr* initialiser, SLoc loc);
    void EmitArrayBroadcastExpr(ArrayBroadcastExpr* e, Value mrvalue_slot);
    void EmitArrayInitExpr(ArrayInitExpr* e, Value mrvalue_slot);
    auto EmitCallExpr(CallExpr* call, Value mrvalue_slot) -> IRValue;
    auto EmitBlockExpr(BlockExpr* expr, Value mrvalue_slot) -> IRValue;
    auto EmitIfExpr(IfExpr* expr, Value mrvalue_slot) -> IRValue;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit##Class(Class* stmt)->IRValue;
#include "srcc/AST.inc"

    auto EmitArithmeticOrComparisonOperator(Tk op, Type ty, Value lhs, Value rhs, mlir::Location loc) -> Value;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(mlir::Location loc, const eval::RValue& val) -> IRValue;

    /// Emit all cleanups starting at the current scope up to and including 'target_scope'.
    void EmitCleanups(const ProcData::CleanupScope& target_scope);
    void EmitCleanups();

    /// Emit a closure.
    auto EmitClosure(ProcDecl* decl, mlir::Location loc) -> IRValue;

    /// Emit the default initialiser for a type.
    auto EmitDefaultInit(Type ty, mlir::Location loc) -> IRValue;

    /// Allocate stack space for a local variable, if need be.
    void EmitAllocaForLocal(LocalDecl* decl);

    /// Emit the initialiser of a local variable, allocating stack space if needed.
    void EmitLocal(LocalDecl* decl);

    /// Emit an lvalue to rvalue conversion.
    auto EmitLValueToRValueConversion(CastExpr* expr) -> std::pair<IRValue, Value>;

    /// Create a temporary value to hold an mrvalue. Returns the address of
    /// the temporary.
    auto EmitToMemory(mlir::Location l, Expr* init) -> Value;

    /// Emit an mrvalue into a memory location.
    void EmitRValue(Value addr, Expr* init);
    void EmitEvaluatedRValue(mlir::Location loc, Value addr, const eval::RValue& rv);
    void EmitScalarRValueImpl(mlir::Location loc, Type type, Value addr, IRValue init_val);

    /// Emit a compile-time value.
    auto EmitTreeConstant(TreeValue* tree, mlir::Location loc) -> Value;
    auto EmitTypeConstant(Type, mlir::Location loc) -> Value;

    auto EnterBlock(std::unique_ptr<Block> bb, mlir::ValueRange args = {}) -> Block*;
    auto EnterBlock(Block* bb, mlir::ValueRange args = {}) -> Block*;

    /// Zero-fill a region of memory.
    void FillWithZeroes(mlir::Location loc, Value addr, Size bytes);

    /// Get an integer type.
    auto IntTy(Size wd) -> mlir::Type;

    /// Get the address of a local variable.
    auto GetAddressOfLocal(LocalDecl* decl, mlir::Location loc) -> Value;

    /// Get the current procedure’s environment pointer.
    auto GetEnvPtr() -> Value;

    /// Get the struct type equivalent to a builtin aggregate type.
    auto GetEquivalentRecordTypeForAggregate(Type ty) -> RecordType*;

    /// Determine the evaluation mode for a type.
    auto GetEvalMode(Type ty) -> EvalMode;

    /// Get a procedure, declaring it if it doesn’t exist yet.
    auto GetOrCreateProc(
        SLoc loc,
        String name,
        Linkage linkage,
        ProcType* ty,
        bool needs_environment
    ) -> ir::ProcOp;

    /// For an integer type, get what type we prefer to treat this as. This converts
    /// ‘weird’ types like 'i17' to something more sensible like 'i32'.
    auto GetPreferredIntType(mlir::Type ty) -> mlir::Type;

    /// Retrieve the static chain pointer of a procedure relative to
    /// the current procedure. This is the environment pointer that
    /// is passed to any procedures defined within that procedure.
    auto GetStaticChainPointer(ProcDecl* proc, mlir::Location location) -> Value;

    /// Handle a backend diagnostic.
    void HandleMLIRDiagnostic(mlir::Diagnostic& diag);

    /// Mark an optional as engaged or disengaged after initialisation.
    void HandleOptionalInitialised(mlir::Value addr, Expr* init, mlir::Value init_from_addr);

    /// Check whether the current insertion point has a terminator.
    bool HasTerminator();

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
        mlir::Location loc,
        Value cond,
        mlir::ValueRange args,
        llvm::function_ref<void()> emit_body
    ) -> Block*;

    auto If(
        mlir::Location loc,
        Value cond,
        llvm::function_ref<void()> emit_body
    ) { return If(loc, cond, {}, emit_body); }

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
        mlir::Location loc,
        Value cond,
        bool has_result,
        llvm::function_ref<IRValue()> emit_then,
        llvm::function_ref<IRValue()> emit_else
    ) -> Block*;

    /// Check if the size of a type is zero; this also means that every
    /// instance of this type is (or would be) identical.
    bool IsZeroSizedType(Type ty);

    /// Check if a local variable has a stack slot.
    bool LocalNeedsAlloca(LocalDecl* local);

    /// Get the mangled name of a procedure.
    auto MangledName(ProcDecl* proc) -> String;

    /// Get the mangled name of a global variable.
    auto MangledName(GlobalDecl* proc) -> String;

    /// Determine whether this parameter type is passed by reference under
    /// the given intent.
    ///
    /// No calling convention is passed to this since parameters to native
    /// procedures should always have the 'copy' intent, which by definition
    /// always passes by value.
    bool PassByReference(Type ty, Intent i);

    /// Set the insert point to the start of a block, but skip over all
    /// operations of a certain type.
    template <typename Op>
    void SetInsertPointAfterLastOpOfTypeIn(Block* bb);

    /// Opposite of If().
    void Unless(
        Value cond,
        llvm::function_ref<void()> emit_else
    );
};

#endif // SRCC_CG_HH
