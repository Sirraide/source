module;

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Allocator.h>
#include <optional>
#include <print>
#include <srcc/Macros.hh>

module srcc.ast;
using namespace srcc;
using namespace srcc::eval;

// ============================================================================
//  Value
// ============================================================================
auto LValue::base_type(TranslationUnit& tu) const -> Type {
    utils::Overloaded V{
        [&](String) -> Type { return tu.I8Ty; },
        [&](Memory* m) -> Type { return m->type(); }
    };

    return std::visit(V, base);
}

Slice::Slice(Value data, Value size)
    : data(std::make_unique<Value>(std::move(data))),
      size(std::make_unique<Value>(std::move(size))) {}

Slice::Slice(Slice&& other)
    : data(std::move(other.data)),
      size(std::move(other.size)) {}

Slice::Slice(const Slice& other)
    : data(std::make_unique<Value>(*other.data)),
      size(std::make_unique<Value>(*other.size)) {}

Slice& Slice::operator=(Slice&& other) {
    data = std::move(other.data);
    size = std::move(other.size);
    return *this;
}

Slice& Slice::operator=(const Slice& other) {
    data = std::make_unique<Value>(*other.data);
    size = std::make_unique<Value>(*other.size);
    return *this;
}

Value::Value(ProcDecl* proc)
    : value(proc),
      ty(proc->type) {}

Value::Value(Slice slice, Type ty)
    : value(std::move(slice)),
      ty(ty) {}

void Value::dump(bool use_colour) const {
    using enum utils::Colour;
    utils::Colours C{use_colour};
    utils::Overloaded V{// clang-format off
        [&](std::monostate) { },
        [&](ProcDecl* proc) { std::print("{}{}", C(Green), proc->name); },
        [&](const LValue& lval) { lval.dump(use_colour); },
        [&](const APInt& value) { std::print("{}{}", C(Magenta), llvm::toString(value, 10, true)); },
        [&](const Slice&) { std::print("<slice>"); },
        [&](this auto& Self, const Reference& ref) {
            Self(ref.lvalue);
            std::print("{}@{}{}", C(Red), C(Magenta), llvm::toString(ref.offset, 10, true));
        }
    }; // clang-format on
    visit(V);
}

auto Value::value_category() const -> ValueCategory {
    return value.visit(utils::Overloaded{
        [](std::monostate) { return Expr::SRValue; },
        [](ProcDecl*) { return Expr::SRValue; },
        [](Slice) { return Expr::SRValue; },
        [](const APInt&) { return Expr::SRValue; },
        [](const LValue&) { return Expr::LValue; },
        [](const Reference&) { return Expr::LValue; }
    });
}

// ============================================================================
//  Evaluation State
// ============================================================================
struct StackFrame {
    explicit StackFrame() = default;

    /// The local variables and parameters allocated in this frame.
    DenseMap<LocalDecl*, LValue> locals;

    /// The return value, if any.
    Value return_value;
};

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
    /*std::unique_ptr<Executor> cached_executor;*/

public:
    TranslationUnit& tu;
    bool complain;
    SmallVector<StackFrame, 16> stack;
    llvm::BumpPtrAllocator memory;

    class PushStackFrame {
        SRCC_IMMOVABLE(PushStackFrame);
        EvaluationContext& ctx;

    public:
        PushStackFrame(EvaluationContext& ctx) : ctx{ctx} { ctx.stack.emplace_back(); }
        ~PushStackFrame() { ctx.stack.pop_back(); }
    };

    EvaluationContext(TranslationUnit& tu, bool complain) : tu{tu}, complain{complain} {}

    auto AllocateMemory(Type ty, Location loc) -> Memory*;

    template <typename... Args>
    void Diag(Diagnostic::Level level, Location where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) tu.context().diags().diag(level, where, fmt, std::forward<Args>(args)...);
    }

    /// Get the current stack frame.
    [[nodiscard]] auto CurrFrame() -> StackFrame& {
        Assert(not stack.empty(), "No stack frame");
        return stack.back();
    }

    /// Read from memory.
    [[nodiscard]] bool LoadMemory(const Memory* mem, void* into, Size size, Location loc);

    /// Read from memory.
    template <typename Ty>
    requires std::is_trivially_copyable_v<Ty>
    [[nodiscard]] bool LoadMemory(const Memory* mem, Ty& into, Location loc) {
        return LoadMemory(mem, &into, Size::Of<Ty>(), loc);
    }

    /// Write into memory.
    [[nodiscard]] bool StoreMemory(Memory* mem, const void* from, Size size, Location loc);

    /// Write into memory.
    template <typename Ty>
    requires std::is_trivially_copyable_v<Ty>
    [[nodiscard]] bool StoreMemory(Memory* mem, const Ty& from, Location loc) {
        return StoreMemory(mem, &from, Size::Of<Ty>(), loc);
    }

    /// \return True on success, false on failure.
    [[nodiscard]] bool Eval(Value& out, Stmt* stmt);
    [[nodiscard]] bool EvalBlockExpr(Value& out, BlockExpr* block);
    [[nodiscard]] bool EvalBuiltinCallExpr(Value& out, BuiltinCallExpr* builtin);
    [[nodiscard]] bool EvalCallExpr(Value& out, CallExpr* call);
    [[nodiscard]] bool EvalConstExpr(Value& out, ConstExpr* constant);
    [[nodiscard]] bool EvalEvalExpr(Value& out, EvalExpr* eval);
    [[nodiscard]] bool EvalIntLitExpr(Value& out, IntLitExpr* int_lit);
    [[nodiscard]] bool EvalProcRefExpr(Value& out, ProcRefExpr* proc_ref);
    [[nodiscard]] bool EvalSliceDataExpr(Value& out, SliceDataExpr* slice_data);
    [[nodiscard]] bool EvalStrLitExpr(Value& out, StrLitExpr* str_lit);

    [[nodiscard]] bool EvalLocalDecl(Value& out, LocalDecl* decl);
    [[nodiscard]] bool EvalParamDecl(Value& out, LocalDecl* decl);
    [[nodiscard]] bool EvalProcDecl(Value& out, ProcDecl* proc);

    /// Check if we’re in a function.
    [[nodiscard]] bool InFunction() { return not stack.empty(); }

    /// Create an integer.
    [[nodiscard]] auto IntValue(std::integral auto val) -> Value;

    /// Initialise a variable.
    [[nodiscard]] bool PerformVariableInitialisation(LValue& addr, Expr* init);

    /// Report an error that involves accessing memory.
    template <typename... Args>
    bool ReportMemoryError(
        const Memory* mem,
        Location access_loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        Error(access_loc, fmt, std::forward<Args>(args)...);
        Note(mem->loc, "Of variable declared here");
        return false;
    }

    /// Get the JIT executor.
    /*auto Executor() -> Executor&;*/
};

// ============================================================================
//  LValue/Memory
// ============================================================================
void LValue::dump(bool use_colour) const {
    using enum utils::Colour;
    utils::Colours C{use_colour};
    utils::Overloaded V{// clang-format off
        [&](String s) { std::print("{}\"{}\"", C(Yellow), s); },
        [&](const Memory*) { std::print("<memory location>"); }
    }; // clang-format on
    base.visit(V);
}

auto EvaluationContext::AllocateMemory(Type ty, Location loc) -> Memory* {
    // Possibly the most cursed code in this entire codebase;
    // we effectively create and allocate the following:
    //
    // struct {
    //     Memory mem;
    //     [:ty:] type;
    // }
    //
    // That is, both the compile-time metadata for this allocation,
    // as well as the Source object that is the memory location are
    // store as a single compile-time object.
    auto ty_sz = ty->size(tu);
    auto ty_align = ty->align(tu);
    auto data_offs = Size::Bytes(sizeof(Memory)).align(ty_align);
    auto total_size = data_offs + ty_sz;
    auto total_align = std::max(Align(alignof(Memory)), ty_align);
    auto data = memory.Allocate(total_size.bytes(), total_align);
    return ::new (data) Memory{ty, loc, static_cast<char*>(data) + data_offs.bytes()};
}

bool EvaluationContext::LoadMemory(
    const Memory* mem,
    void* into,
    Size size_to_load,
    Location loc
) {
    if (mem->dead()) return ReportMemoryError(
        mem,
        loc,
        "Accessing memory outside of its lifetime"
    );

    if (size_to_load.bytes() > mem->ty->size(tu).bytes()) return ReportMemoryError(
        mem,
        loc,
        "Out-of-bounds read of size {} (total size: {})",
        mem->ty->size(tu) - size_to_load,
        size_to_load
    );

    std::memcpy(into, mem->data(), size_to_load.bytes());
    return true;
}

bool EvaluationContext::StoreMemory(
    Memory* mem,
    const void* from,
    Size size_to_write,
    Location loc
) {
    if (mem->dead()) return ReportMemoryError(
        mem,
        loc,
        "Accessing memory outside of its lifetime"
    );

    if (size_to_write.bytes() > mem->ty->size(tu).bytes()) return ReportMemoryError(
        mem,
        loc,
        "Out-of-bounds write of size {} (total size: {})",
        mem->ty->size(tu) - size_to_write,
        size_to_write
    );

    std::memcpy(mem->data(), from, size_to_write.bytes());
    return true;
}

void Memory::destroy() {
    data_and_state.setInt(LifetimeState::Uninitialised);
}

void Memory::init(TranslationUnit& tu) {
    data_and_state.setInt(LifetimeState::Initialised);

    // Clear in any case since this starts the lifetime of this thing.
    std::memset(data(), 0, ty->size(tu).bytes());
}

// ============================================================================
//  Helpers
// ============================================================================
bool EvaluationContext::Eval(Value& out, Stmt* stmt) {
    Assert(not stmt->dependent(), "Cannot evaluate dependent statement");
    switch (stmt->kind()) {
        using K = Stmt::Kind;
#define AST_STMT_LEAF(node) \
    case K::node: return SRCC_CAT(Eval, node)(out, cast<node>(stmt));
#include "../../include/srcc/AST.inc"
    }
    Unreachable("Invalid statement kind");
}

auto EvaluationContext::IntValue(std::integral auto val) -> Value {
    // FIXME: Change to 'int'.
    return Value{APInt{64, u64(val)}, tu.I64Ty};
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
    out = {};
    for (auto s : block->stmts()) {
        Value val;
        if (isa<Decl>(s)) continue;
        if (not Eval(val, s)) return false;
        if (s == block->return_expr()) out = std::exchange(val, {});
    }
    return true;
}

bool EvaluationContext::EvalBuiltinCallExpr(Value& out, BuiltinCallExpr* builtin) {
    switch (builtin->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            Assert(builtin->args().size() == 1);
            if (not Eval(out, builtin->args().front())) return false;
            std::print("{}", out.cast<Slice>().data->cast<Reference>().lvalue.base.get<String>());
            out = {};
            return true;
        }
    }

    Unreachable("Invalid builtin kind");
}

bool EvaluationContext::EvalCallExpr(Value& out, CallExpr* call) {
    if (not Eval(out, call->callee)) return false;
    auto proc = out.cast<ProcDecl*>();
    auto args = call->args();

    // If we have a body, just evaluate it.
    if (proc->body) {
        // Set up stack.
        PushStackFrame _{*this};
        auto& frame = CurrFrame();

        // Allocate and initialise local variables and initialise them.
        for (auto [i, l] : enumerate(proc->locals)) {
            auto& addr = frame.locals[l] = LValue{AllocateMemory(l->type, l->location())};
            if (not PerformVariableInitialisation(addr, args[i]))
                return false;
        }

        // Dew it.
        return Eval(out, proc->body);
    }

    // FIXME: We can only call functions defined in this module; calling
    // external functions would only be possible if we had the module providing
    // them present as a shared library (explore generating a shared library
    // and throwing it in /tmp)
    //
    // TODO: Otherwise, try to perform an external call.
    return Error(call->location(), "Sorry, can’t call external functions at compile-time yet");
}

bool EvaluationContext::EvalConstExpr(Value& out, ConstExpr* constant) {
    out = *constant->value;
    return true;
}

bool EvaluationContext::EvalEvalExpr(Value& out, EvalExpr* eval) {
    EvaluationContext C{tu, complain};
    return C.Eval(out, eval->stmt);
}

bool EvaluationContext::EvalIntLitExpr(Value& out, IntLitExpr* int_lit) {
    out = {int_lit->storage.value(), int_lit->type};
    return true;
}

bool EvaluationContext::EvalLocalDecl(Value& out, LocalDecl* decl) {
    Assert(InFunction(), "Evaluating local outside of function call?");
    Assert(CurrFrame().locals.contains(decl), "Local variable not initialised?");
    out = {CurrFrame().locals.at(decl), decl->type};
    return true;
}

bool EvaluationContext::EvalParamDecl(Value& out, LocalDecl* decl) {
    return EvalLocalDecl(out, decl);
}

bool EvaluationContext::EvalProcDecl(Value&, ProcDecl*) {
    return true; // Proc decls are never ‘evaluated’.
}

bool EvaluationContext::EvalProcRefExpr(Value& out, ProcRefExpr* proc_ref) {
    out = proc_ref->decl;
    return true;
}

bool EvaluationContext::EvalSliceDataExpr(Value& out, SliceDataExpr* slice_data) {
    if (not Eval(out, slice_data->slice)) return false;
    auto data = std::move(*out.get<Slice>()->data);
    out = std::move(data);
    return true;
}

bool EvaluationContext::EvalStrLitExpr(Value& out, StrLitExpr* str_lit) {
    out = Value{
        Slice{
            Value{
                Reference{LValue{str_lit->value, false}, APInt::getZero(64)},
                ArrayType::Get(tu, tu.I8Ty, i64(str_lit->value.size())),
            },
            IntValue(str_lit->value.size()),
        },
        tu.StrLitTy,
    };
    return true;
}

bool EvaluationContext::PerformVariableInitialisation(LValue& addr, Expr* init) {
    auto mem = addr.base.get<Memory*>();

    // For builtin types, Sema will have ensured that the RHS is
    // an srvalue of the same type.
    if (mem->type() == Types::IntTy) {
        Assert(init->value_category == Expr::SRValue);
        Value int_val;
        if (not Eval(int_val, init))
            return false;

        mem->init(tu);
        return StoreMemory(mem, int_val.cast<APInt>().getZExtValue(), init->location());
    }

    return Error(
        init->location(),
        "Unsupported variable type in constant evaluation: {}",
        mem->type().print(true)
    );
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
    Value val = {};
    if (not C.Eval(val, stmt)) return std::nullopt;
    return std::move(val);
}
