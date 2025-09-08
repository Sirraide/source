#ifndef SRCC_CG_HH
#define SRCC_CG_HH

#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <base/Assert.hh>
#include <mlir/Pass/PassManager.h>

namespace srcc::cg {
class CodeGen;
class LLVMCodeGen;
class VMCodeGen;
class ArgumentMapping;
class SRValue;
class SType;

namespace detail {
template <typename Elem, typename Range>
class ValueOrTypePair;

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

template <typename Elem, typename Range>
class srcc::cg::detail::ValueOrTypePair {
protected:
    Elem vals[2]{};

public:
    ValueOrTypePair() = default;
    ValueOrTypePair(std::nullptr_t) = delete("Use '{}' instead to create an empty SRValue or RType");
    ValueOrTypePair(Elem first_val) : vals{first_val, nullptr} { Assert(first_val); }
    ValueOrTypePair(Elem first_val, Elem second_val) : vals{first_val, second_val} {
        Assert(vals[0] and vals[1]);
    }

    template <utils::ConvertibleRange<Elem> T>
    ValueOrTypePair(T r) {
        Assert(r.size() <= 2);
        if (r.size() >= 1) vals[0] = r[0];
        if (r.size() >= 2) vals[1] = r[1];
    }

    ValueOrTypePair(Range r) {
        Assert(r.size() <= 2);
        if (r.size() >= 1) vals[0] = r[0];
        if (r.size() >= 2) vals[1] = r[1];
    }

    /// Allow treating this as a range.
    [[nodiscard]] auto begin() const { return operator Range().begin(); }
    [[nodiscard]] auto end() const { return operator Range().end(); }

    /// Run a function on each element of this range.
    auto each(auto cb) {
        using Res = std::invoke_result_t<decltype(cb), Elem>;
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

    /// Get the first value of an aggregate.
    [[nodiscard]] auto first() const -> Elem {
        Assert(is_aggregate());
        return vals[0];
    }

    /// Copy the types into a vector.
    void into(SmallVectorImpl<Elem>& container) const {
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
    [[nodiscard]] auto second() const -> Elem {
        Assert(is_aggregate());
        return vals[1];
    }

    /// Get this as a single scalar value.
    [[nodiscard]] auto scalar() const -> Elem {
        Assert(is_scalar());
        return vals[0];
    }

    /// Get the first value if this is an aggregate or scalar.
    [[nodiscard]] auto scalar_or_first() const -> Elem {
        Assert(not is_null());
        return vals[0];
    }

    /// Convert this to a range of types.
    operator Range() const {
        if (is_null()) return {};
        if (is_scalar()) return ArrayRef{vals, 1};
        return ArrayRef{vals, 2};
    }

    /// Check if this is empty.
    explicit operator bool() const { return not is_null(); }
};

/// Type equivalent of 'SRValue'
class srcc::cg::SType : public detail::ValueOrTypePair<mlir::Type, mlir::TypeRange> {
    friend SRValue;

public:
    using ValueOrTypePair::ValueOrTypePair;
};

/// This is either a single SSA value or a pair of SSA values; the
/// latter is used for slices, ranges, and closures.
class srcc::cg::SRValue : public detail::ValueOrTypePair<Value, mlir::ValueRange> {
public:
    using ValueOrTypePair::ValueOrTypePair;

    /// Get a location describing this value.
    [[nodiscard]] auto loc() const -> mlir::Location {
        Assert(not is_null());
        return scalar_or_first().getLoc();
    }

    /// Get the type(s) of this value.
    [[nodiscard]] auto type() const -> SType {
        SType ty;
        if (vals[0]) ty.vals[0] = vals[0].getType();
        if (vals[1]) ty.vals[1] = vals[1].getType();
        return ty;
    }
};

class srcc::cg::CodeGen : DiagsProducer
    , detail::CodeGenBase
    , public mlir::OpBuilder {
    LIBBASE_IMMOVABLE(CodeGen);
    struct Printer;
    struct Mangler;
    friend DiagsProducer;
    friend LLVMCodeGen;

    TranslationUnit& tu;
    Opt<ir::ProcOp> printf;
    DenseMap<LocalDecl*, SRValue> locals;
    DenseMap<ProcDecl*, ir::ProcOp> declared_procs;
    DenseMap<ir::ProcOp, ProcDecl*> proc_reverse_lookup;
    DenseMap<ProcDecl*, String> mangled_names;
    StringMap<mlir::LLVM::GlobalOp> interned_strings;
    Value abort_info_slot;
    mlir::ModuleOp mlir_module;
    ir::ProcOp vm_entry_point;
    ir::ProcOp curr_proc;
    Size word_size;
    LangOpts lang_opts;
    mlir::Type ptr_ty;
    mlir::Type int_ty;
    mlir::Type bool_ty;
    SType closure_ty;
    SType slice_ty;
    usz strings = 0;

public:
    CodeGen(TranslationUnit& tu, LangOpts lang_opts, Size word_size);

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
    [[nodiscard]] bool finalise();

    /// Finalise a single procedure.
    [[nodiscard]] bool finalise(ir::ProcOp proc);

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
        ArrayRef<std::string> additional_objects,
        StringRef program_file_name_override
    );

    class EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);

        CodeGen& CG;
        ir::ProcOp old_func;
        InsertionGuard guard;

    public:
        EnterProcedure(CodeGen& CG, ir::ProcOp func);
        ~EnterProcedure() { CG.curr_proc = old_func; }
    };

    struct ABICallInfo {
        SmallVector<mlir::Type> result_types;
        SmallVector<mlir::Type> arg_types;
        SmallVector<mlir::Attribute> result_attrs;
        SmallVector<mlir::Attribute> arg_attrs;
        SmallVector<Value> args;
        mlir::FunctionType func;
    };

    struct StructInitHelper {
        CodeGen& CG;
        StructType* ty;
        Value base;
        usz i = 0;

        StructInitHelper(CodeGen& CG, StructType* ty, Value base) : CG{CG}, ty{ty}, base{base} {}
        void emit_next_field(SRValue v);
    };

    // AST -> IR converters
    auto C(CallingConvention l) -> mlir::LLVM::CConv;
    auto C(Linkage l) -> mlir::LLVM::Linkage;
    auto C(Location l) -> mlir::Location;
    auto C(Type ty) -> SType;

    auto ConvertProcType(ProcType* ty) -> ABICallInfo;
    auto CreateAlloca(mlir::Location loc, Type ty) -> Value;
    auto CreateAlloca(mlir::Location loc, Size sz, Align a) -> Value;
    void CreateAbort(mlir::Location loc, ir::AbortReason reason, SRValue msg1, SRValue msg2);

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
    auto CreateGlobalStringPtr(Align align, String data, bool null_terminated) -> Value;
    auto CreateGlobalStringPtr(String data) -> Value;
    auto CreateGlobalStringSlice(mlir::Location loc, String data) -> SRValue;
    auto CreateICmp(mlir::Location loc, mlir::arith::CmpIPredicate pred, Value lhs, Value rhs) -> Value;
    auto CreateInt(mlir::Location loc, const APInt& value, Type ty) -> Value;
    auto CreateInt(mlir::Location loc, i64 value, Type ty = Type::IntTy) -> Value;
    auto CreateInt(mlir::Location loc, i64 value, mlir::Type ty) -> Value;
    auto CreateLoad(mlir::Location loc, Value addr, SType ty, Align align) -> SRValue;
    auto CreateNil(mlir::Location loc, SType ty) -> SRValue;
    auto CreateNullPointer(mlir::Location loc) -> Value;
    auto CreatePoison(mlir::Location loc, SType ty) -> SRValue;
    auto CreatePtrAdd(mlir::Location loc, Value addr, Value offs) -> Value;
    auto CreatePtrAdd(mlir::Location loc, Value addr, Size offs) -> Value;
    auto CreateSICast(mlir::Location loc, Value val, Type from, Type to) -> Value;
    void CreateStore(mlir::Location loc, Value addr, SRValue val, Align align);

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        tu.context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    auto DeclarePrintf() -> ir::ProcOp;
    auto DeclareProcedure(ProcDecl* proc) -> ir::ProcOp;

    void Emit(ArrayRef<ProcDecl*> procs);
    auto Emit(Stmt* stmt) -> SRValue;
    void EmitArrayBroadcast(Type elem_ty, Value addr, u64 elements, Expr* initialiser, Location loc);
    void EmitArrayBroadcastExpr(ArrayBroadcastExpr* e, Value mrvalue_slot);
    void EmitArrayInitExpr(ArrayInitExpr* e, Value mrvalue_slot);
    auto EmitCallExpr(CallExpr* call, Value mrvalue_slot) -> SRValue;
    auto EmitCastExpr(CastExpr* cast, Value mrvalue_slot) -> SRValue;
    auto EmitBlockExpr(BlockExpr* expr, Value mrvalue_slot) -> SRValue;
    auto EmitIfExpr(IfExpr* expr, Value mrvalue_slot) -> SRValue;
#define AST_DECL_LEAF(Class)
#define AST_STMT_LEAF(Class) auto Emit##Class(Class* stmt)->SRValue;
#include "srcc/AST.inc"

    auto EmitArithmeticOrComparisonOperator(Tk op, Type ty, Value lhs, Value rhs, mlir::Location loc) -> Value;
    void EmitProcedure(ProcDecl* proc);
    auto EmitValue(Location loc, const eval::RValue& val) -> SRValue;

    /// Emit any (lvalue, srvalue, mrvalue) initialiser into a memory location.
    void EmitInitialiser(Value addr, Expr* init);
    void EmitLocal(LocalDecl* decl);

    /// Emit an mrvalue into a memory location.
    void EmitMRValue(Value addr, Expr* init);

    auto EnterBlock(std::unique_ptr<Block> bb, mlir::ValueRange args = {}) -> Block*;
    auto EnterBlock(Block* bb, mlir::ValueRange args = {}) -> Block*;

    /// Get an integer type.
    auto IntTy(Size wd) -> mlir::Type;

    /// Get a procedure, declaring it if it doesn’t exist yet.
    auto GetOrCreateProc(
        Location loc,
        String name,
        Linkage linkage,
        ProcType* ty
    ) -> ir::ProcOp;

    /// Offset a pointer to load the second element of an aggregate value.
    auto GetPtrToSecondAggregateElem(
        mlir::Location loc,
        Value addr,
        SType aggregate
    ) -> std::pair<Value, Align>;

    /// Handle a backend diagnostic.
    void HandleMLIRDiagnostic(mlir::Diagnostic& diag);

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
        llvm::function_ref<SRValue()> emit_then,
        llvm::function_ref<SRValue()> emit_else
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

    /// Perform ABI lowering for a call or argument list.
    auto LowerProcedureSignature(
        mlir::Location l,
        ProcType* proc,
        Value indirect_ptr,
        ArrayRef<Expr*> args
    ) -> ABICallInfo;

    /// Get the mangled name of a procedure.
    auto MangledName(ProcDecl* proc) -> String;

    /// Create a temporary value to hold an mrvalue. Returns the address of
    /// the temporary.
    auto MakeTemporary(mlir::Location l, Expr* init) -> Value;

    /// Whether a value of this type needs to be returned indirectly.
    bool NeedsIndirectReturn(Type ty);

    /// Determine whether this parameter type is passed by reference under
    /// the given intent.
    ///
    /// No calling convention is passed to this since parameters to native
    /// procedures should always have the 'copy' intent, which by definition
    /// always passes by value.
    bool PassByReference(Type ty, Intent i);

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

private:
    void AssertTriple();
};

#endif // SRCC_CG_HH
