module;

#include <optional>
#include <srcc/Macros.hh>

module srcc.eval;
using namespace srcc::eval;

// ============================================================================
//  Value
// ============================================================================
Value::Value(StrLitExpr* str)
    : value(LValue{str->value.data(), isz(str->value.size())}),
      ty(str->type) {}

Value::Value(ProcDecl* proc)
    : value(proc),
      ty(proc->type) {}

Value::Value(LValue lval, Type ty)
    : value(lval),
      ty(ty) {}

Value::Value(Reference ref, Type ty)
    : value(ref),
      ty(ty) {}

// ============================================================================
//  Evaluator
// ============================================================================
class srcc::eval::EvaluationContext : DiagsProducer<bool> {
    /*class Executor {
        std::unique_ptr<llvm::orc::ThreadSafeContext> ctx;
        std::unique_ptr<llvm::orc::LLJIT> jit;

        Executor(
            std::unique_ptr<llvm::orc::ThreadSafeContext> ctx,
            std::unique_ptr<llvm::orc::LLJIT> jit
        ) : ctx{std::move(ctx)}, jit{std::move(jit)} {}

    public:
        static auto Create() -> std::unique_ptr<Executor>;
    };*/

    TranslationUnit& tu;
    bool complain;
    /*std::unique_ptr<Executor> cached_executor;*/

public:
    EvaluationContext(TranslationUnit& tu, bool complain) : tu{tu}, complain{complain} {}

    template <typename... Args>
    void Diag(Diagnostic::Level level, Location where, fmt::format_string<Args...> fmt, Args&&... args) {
        if (complain) tu.context().diags().diag(level, where, fmt, std::forward<Args>(args)...);
    }

    /// Create an empty value.
    auto Empty() -> Value;

    /// \return True on success, false on failure.
    bool Eval(Value& out, Stmt* stmt);
    bool EvalBlockExpr(Value& out, BlockExpr* block);
    bool EvalBuiltinCallExpr(Value& out, BuiltinCallExpr* builtin);
    bool EvalCallExpr(Value& out, CallExpr* call);
    bool EvalProcDecl(Value& out, ProcDecl* proc);
    bool EvalProcRefExpr(Value& out, ProcRefExpr* proc_ref);
    bool EvalSliceDataExpr(Value& out, SliceDataExpr* slice_data);
    bool EvalStrLitExpr(Value& out, StrLitExpr* str_lit);

    /// Get the JIT executor.
    /*auto Executor() -> Executor&;*/
};

// ============================================================================
//  Helpers
// ============================================================================
auto EvaluationContext::Empty() -> Value {
    return Value{tu};
}

bool EvaluationContext::Eval(Value& out, Stmt* stmt) {
    Assert(not stmt->dependent(), "Cannot evaluate dependent statement");
    switch (stmt->kind()) {
        using K = Stmt::Kind;
#define AST_STMT_LEAF(node) \
    case K::node: return SRCC_CAT(Eval, node)(out, cast<node>(stmt));
#include "srcc/AST.inc"
    }
    Unreachable("Invalid statement kind");
}

/*auto EvaluationContext::Executor() -> class Executor& {
    if (not cached_executor.has_value()) cached_executor.emplace();
    return *cached_executor;
}*/

// ============================================================================
//  Executor
// ============================================================================
/*auto EvaluationContext::Executor::Create() -> std::unique_ptr<Executor> {
    auto ctx = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());

    // Get the target machine.
    auto tm_builder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (auto e = tm_builder.takeError()) Unreachable("Failed to get native target: {}", e);

    // Create the jit.
    auto jit_builder = std::make_unique<llvm::orc::LLJITBuilder>();
    jit_builder->setJITTargetMachineBuilder(std::move(*tm_builder));

    // Try to enable debugging support. Ignore any failures here.
    jit_builder->setPrePlatformSetup([](llvm::orc::LLJIT& j) {
        llvm::consumeError(llvm::orc::enableDebuggerSupport(j));
        return llvm::Error::success();
    });

    // Dew it.
    auto jit = jit_builder->create();
    if (auto e = jit.takeError()) Unreachable("Failed to create native executor: {}", e);

    // Load runtime.
    //
    // FIXME: Replace this with our own runtime once we have one.
    // FIXME: Have some way of importing the C runtime portably.
    auto rt = llvm::orc::DynamicLibrarySearchGenerator::Load(
        "/usr/lib/libc.so.6",
        jit->get()->getDataLayout().getGlobalPrefix()
    );

    if (auto e = rt.takeError()) Unreachable("Failed to load runtime: {}", e);
    jit->get()->getMainJITDylib().addGenerator(std::move(*rt));

    // Create the executor.
    return std::unique_ptr<Executor>(new Executor(std::move(ctx), std::move(*jit)));
}*/

// ============================================================================
//  Evaluation
// ============================================================================
bool EvaluationContext::EvalBlockExpr(Value& out, BlockExpr* block) {
    out = Empty();
    for (auto s : block->stmts()) {
        Value val;
        if (isa<Decl>(s)) continue;
        if (not Eval(val, s)) return false;
        if (s == block->return_expr()) out = std::exchange(val, Value{tu});
    }
    return true;
}

bool EvaluationContext::EvalBuiltinCallExpr(Value& out, BuiltinCallExpr* builtin) {
    switch (builtin->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            Assert(builtin->args().size() == 1);
            if (not Eval(out, builtin->args().front())) return false;
            auto lv = out.cast<Value::LValue>();
            fmt::print("{}", std::string_view{static_cast<const char*>(lv.pointer), lv.size.bytes()});
            out = Empty();
            return true;
        }
    }

    Unreachable("Invalid builtin kind");
}


bool EvaluationContext::EvalCallExpr(Value& out, CallExpr* call) {
    if (not Eval(out, call->callee)) return false;
    auto proc = out.cast<ProcDecl*>();

    // FIXME: We can only call functions defined in this module; calling
    // external functions would only be possible if we had the module providing
    // them present as a shared library (explore generating a shared library
    // and throwing it in /tmp)

    // If we have a body, just evaluate it.
    if (proc->body) return Eval(out, proc->body);

    // TODO: Otherwise, try to perform an external call.
    return Error(call->location(), "Sorry, can’t call external functions at compile-time yet");
}

bool EvaluationContext::EvalProcDecl(Value& out, ProcDecl* proc) {
    out = proc;
    return true;
}

bool EvaluationContext::EvalProcRefExpr(Value& out, ProcRefExpr* proc_ref) {
    return EvalProcDecl(out, proc_ref->decl);
}

bool EvaluationContext::EvalSliceDataExpr(Value& out, SliceDataExpr* slice_data) {
    if (not Eval(out, slice_data->slice)) return false;
    auto ref = Value::Reference{out.cast<Value::LValue>(), Size::Bytes(0)};
    out = {ref, slice_data->type};
    return true;
}

bool EvaluationContext::EvalStrLitExpr(Value& out, StrLitExpr* str_lit) {
    out = str_lit;
    return true;
}

// ============================================================================
//  API
// ============================================================================
auto srcc::eval::Evaluate(
    TranslationUnit& tu,
    Stmt* stmt,
    bool complain
) -> std::optional<Value> {
    EvaluationContext C{tu, complain};
    Value val = C.Empty();
    if (not C.Eval(val, stmt)) return std::nullopt;
    return std::move(val);
}
