module;

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Allocator.h>
#include <optional>
#include <print>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.ast;
import srcc.token;
using namespace srcc;
using namespace srcc::eval;

struct Closure {
    ProcDecl* decl;
    void* env = nullptr;
};

#define TRY(x)                    \
    do {                          \
        if (not(x)) return false; \
    } while (0)
#define TryEval(...) TRY(Eval(__VA_ARGS__))

// ============================================================================
//  Value
// ============================================================================
Value::Value(ProcDecl* proc)
    : value(proc),
      ty(proc->type) {}

Value::Value(Slice slice, Type ty)
    : value(std::move(slice)),
      ty(ty) {}

void Value::dump(bool use_colour) const {
    std::print("{}", text::RenderColours(use_colour, print().str()));
}

auto Value::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{// clang-format off
        [&](bool) { out += std::format("%1({})", value.get<bool>()); },
        [&](std::monostate) { },
        [&](ProcDecl* proc) { out += std::format("%2({})", proc->name); },
        [&](TypeTag) { out += ty->print(); },
        [&](const LValue& lval) { out += lval.print(); },
        [&](const APInt& value) { out += std::format("%5({})", toString(value, 10, true)); },
        [&](const Slice&) { out += "<slice>"; },
        [&](this auto& Self, const Reference& ref) {
            Self(ref.lvalue);
            out += std::format("%1(@)%5({})", toString(ref.offset, 10, true));
        }
    }; // clang-format on

    visit(V);
    return out;
}

auto Value::value_category() const -> ValueCategory {
    return value.visit(utils::Overloaded{
        [](std::monostate) { return Expr::SRValue; },
        [](bool) { return Expr::SRValue; },
        [](ProcDecl*) { return Expr::SRValue; },
        [](Slice) { return Expr::SRValue; },
        [](TypeTag) { return Expr::SRValue; },
        [](const APInt&) { return Expr::SRValue; },
        [](const LValue&) { return Expr::LValue; },
        [](const Reference&) { return Expr::LValue; },
    });
}

// ============================================================================
//  Evaluation State
// ============================================================================
struct StackFrame {
    /// This is *not* a map to lvalues because some locals
    /// (e.g. small 'in' parameters, are *not* lvalues!)
    using LocalsMap = DenseMap<LocalDecl*, Value>;

    StackFrame(const StackFrame&) = delete;
    StackFrame& operator=(const StackFrame&) = delete;
    StackFrame(StackFrame&&) = default;
    StackFrame& operator=(StackFrame&&) = default;

    StackFrame() = default;
    StackFrame(Ptr<ProcDecl> proc, LocalsMap&& locals, Location loc)
        : proc{proc}, call_loc{loc}, locals{std::move(locals)} {}

    /// The procedure that this frame belongs to.
    Ptr<ProcDecl> proc;

    /// Location of the call that pushed this frame.
    Location call_loc;

    /// The local variables and parameters allocated in this frame.
    LocalsMap locals;

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
        PushStackFrame(
            EvaluationContext& ctx,
            ProcDecl* proc,
            StackFrame::LocalsMap&& locals,
            Location loc
        ) : ctx{ctx} { ctx.stack.emplace_back(proc, std::move(locals), loc); }
        ~PushStackFrame() { ctx.stack.pop_back(); }
    };

    EvaluationContext(TranslationUnit& tu, bool complain) : tu{tu}, complain{complain} {
        // Push a fake stack frame in case we need to deal with
        // local variables at the top level.
        stack.emplace_back();
    }

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& { return tu.context().diags(); }

    auto AllocateVar(StackFrame::LocalsMap& locals, LocalDecl* l) -> LValue&;
    auto AllocateMemory(Type ty) -> Memory*;
    auto AllocateMemory(Size size, Align align) -> Memory*;

    template <typename... Args>
    void Diag(Diagnostic::Level level, Location where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) {
            tu.context().diags().diag(level, where, fmt, std::forward<Args>(args)...);

            // Print call stack, but take care not to recurse infinitely here.
            if (level != Diagnostic::Level::Note) {
                for (auto& frame : ref(stack).drop_front() | vws::reverse) Note(
                    frame.call_loc,
                    "In call to '{}' here",
                    frame.proc.get()->name
                );
            }
        }
    }

    /// Get the current stack frame.
    [[nodiscard]] auto CurrFrame() -> StackFrame& {
        Assert(not stack.empty(), "No stack frame");
        return stack.back();
    }

    /// Check and diagnose for invalid memory accesses.
    [[nodiscard]] bool CheckMemoryAccess(const LValue& lval, Size size, bool write, Location access_loc);

    /// Print the contents of a stack frame for debugging.
    void DumpFrame(StackFrame& frame);

    /// Get a slice as a raw memory buffer.
    [[nodiscard]] auto GetMemoryBuffer(const Slice& slice, Location loc) -> std::optional<ArrayRef<char>>;

    /// Read from memory.
    [[nodiscard]] bool LoadMemory(const LValue& lval, void* into, Size size, Location load_loc);

    /// Read from memory.
    template <typename Ty>
    requires std::is_trivially_copyable_v<Ty>
    [[nodiscard]] bool LoadMemory(const LValue& lval, Ty& into, Location load_loc) {
        return LoadMemory(lval, &into, Size::Of<Ty>(), load_loc);
    }

    /// Write into memory.
    [[nodiscard]] bool StoreMemory(const LValue& lval, const void* from, Size size, Location store_loc);

    /// Write into memory.
    template <typename Ty>
    requires std::is_trivially_copyable_v<Ty>
    [[nodiscard]] bool StoreMemory(const LValue& lval, const Ty& from, Location store_loc) {
        return StoreMemory(lval, &from, Size::Of<Ty>(), store_loc);
    }

    /// \return True on success, false on failure.
    [[nodiscard]] bool Eval(Value& out, Stmt* stmt);
#define AST_STMT_LEAF(Class) [[nodiscard]] bool Eval##Class(Value& out, Class* expr);
#include "srcc/AST.inc"

    /// Check if we’re in a function.
    [[nodiscard]] bool InFunction() { return not stack.empty(); }

    /// Create an integer.
    [[nodiscard]] auto IntValue(std::integral auto val) -> Value;

    /// Perform an assignment to an already live variable.
    [[nodiscard]] bool PerformAssign(LValue& addr, Ptr<Expr> init, Location loc);

    /// Initialise a variable.
    [[nodiscard]] bool PerformVarInit(LValue& addr, Ptr<Expr> init, Location loc);

    /// Report an error that involves accessing memory.
    template <typename... Args>
    bool ReportMemoryError(
        const LValue& lval,
        Location access_loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        Error(access_loc, fmt, std::forward<Args>(args)...);
        Note(lval.loc, "Of variable declared here");
        return false;
    }

    /// Get the JIT executor.
    /*auto Executor() -> Executor&;*/
};

// ============================================================================
//  LValue/Memory
// ============================================================================
void LValue::dump(bool use_colour) const {
    std::print("{}", text::RenderColours(use_colour, print().str()));
}

auto LValue::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{// clang-format off
        [&](String s) { out += std::format("%3(\"{}\")", s); },
        [&](const Memory* mem) {
            out += std::format(
                "<memory:%4({}):%3({})",
                mem->size(),
                mem->alive() ? "alive"sv : "dead"sv
            );

            if (mem->alive() and type == Types::IntTy) {
                i64 value;
                std::memcpy(&value, mem->data(), sizeof(i64));
                out += std::format(" value:%5({})", value);
            }

            out += ">";
        }
    }; // clang-format on
    base.visit(V);
    return out;
}

auto EvaluationContext::AllocateVar(StackFrame::LocalsMap& locals, LocalDecl* l) -> LValue& {
    // For large integers, over-allocate so we can store multiples of words.
    Memory* mem;
    if (isa<IntType>(l->type)) {
        mem = AllocateMemory(
            l->type->size(tu).aligned(Align::Of<u64>()),
            Align::Of<u64>()
        );
    } else {
        mem = AllocateMemory(l->type);
    }

    return locals.try_emplace(l, LValue{mem, l->type, l->location()}).first->second.cast<LValue>();
}

auto EvaluationContext::AllocateMemory(Type ty) -> Memory* {
    auto ty_sz = ty->size(tu);
    auto ty_align = ty->align(tu);
    return AllocateMemory(ty_sz, ty_align);
}

auto EvaluationContext::AllocateMemory(Size size, Align align) -> Memory* {
    auto data = memory.Allocate(size.bytes(), align);
    auto mem = memory.Allocate<Memory>();
    return ::new (mem) Memory{size, data};
}

bool EvaluationContext::CheckMemoryAccess(
    const LValue& lval,
    Size size,
    bool write,
    Location loc
) {
    auto mem = lval.base.get<Memory*>();
    if (mem->dead()) return ReportMemoryError(
        lval,
        loc,
        "Accessing memory outside of its lifetime"
    );

    if (size.bytes() > mem->size().bytes()) return ReportMemoryError(
        lval,
        loc,
        "Out-of-bounds {} of size {} (total size: {})",
        write ? "write" : "read",
        mem->size() - size,
        size
    );

    return true;
}

auto EvaluationContext::GetMemoryBuffer(
    const Slice& slice,
    Location loc
) -> std::optional<ArrayRef<char>> {
    using Ret = std::optional<ArrayRef<char>>;
    auto size = slice.size.getZExtValue();
    auto offs = slice.data.offset.getZExtValue();
    auto V = utils::Overloaded{
        // clang-format off
        [&](String s) -> Ret {
            if (s.size() < size + offs) {
                // TODO: improve error.
                Error(loc, "Out-of-bounds access to string literal");
                return std::nullopt;
            }

            return ArrayRef{s.data() + offs, size};
        },

        [&](const Memory* mem) -> Ret {
            if (not CheckMemoryAccess(slice.data.lvalue, Size::Bytes(offs + size), false, loc)) return std::nullopt;
            return ArrayRef{static_cast<const char*>(mem->data()) + offs, size};
        },
    }; // clang-format on
    return slice.data.lvalue.base.visit(V);
}

bool EvaluationContext::LoadMemory(
    const LValue& lval,
    void* into,
    Size size_to_load,
    Location loc
) {
    TRY(CheckMemoryAccess(lval, size_to_load, false, loc));
    std::memcpy(into, lval.base.get<Memory*>()->data(), size_to_load.bytes());
    return true;
}

bool EvaluationContext::StoreMemory(
    const LValue& lv,
    const void* from,
    Size size_to_write,
    Location loc
) {
    auto mem = lv.base.get<Memory*>();

    // Check that we can store into this.
    TRY(CheckMemoryAccess(lv, size_to_write, true, loc));
    if (not lv.modifiable)
        return ReportMemoryError(lv, loc, "Attempting to write into read-only memory.");

    // Dew it.
    std::memcpy(mem->data(), from, size_to_write.bytes());
    return true;
}

void Memory::destroy() {
    data_and_state.setInt(LifetimeState::Uninitialised);
}

void Memory::init() {
    data_and_state.setInt(LifetimeState::Initialised);

    // Clear in any case since this starts the lifetime of this thing.
    zero();
}

void Memory::zero() {
    std::memset(data(), 0, size().bytes());
}

// ============================================================================
//  Helpers
// ============================================================================
void EvaluationContext::DumpFrame(StackFrame& frame) {
    if (not frame.proc) return;
    std::println("In procedure '{}'", frame.proc.get()->name);
    for (auto& [decl, lval] : frame.locals) {
        std::print("  {} -> ", decl->name);
        lval.dump(true);
        std::print("\n");
    }
    std::print("\n");
}

bool EvaluationContext::Eval(Value& out, Stmt* stmt) {
    if (stmt->dependent()) {
        ICE(stmt->location(), "Cannot evaluate dependent statement");
        return false;
    }

    // TODO: Add a max steps variable to prevent infinite loops.

    switch (stmt->kind()) {
        using K = Stmt::Kind;
#define AST_STMT_LEAF(node) \
    case K::node: return SRCC_CAT(Eval, node)(out, cast<node>(stmt));
#include "../../include/srcc/AST.inc"
    }
    Unreachable("Invalid statement kind");
}

auto EvaluationContext::IntValue(std::integral auto val) -> Value {
    return Value{APInt{64, u64(val)}, Types::IntTy};
}

bool EvaluationContext::PerformAssign(LValue& addr, Ptr<Expr> init, Location loc) {
    auto mem = addr.base.get<Memory*>();

    // For builtin types, Sema will have ensured that the RHS is
    // an srvalue of the same type.
    switch (addr.type->value_category()) {
        case ValueCategory::MRValue: Todo("Initialise mrvalue");
        case ValueCategory::LValue: Todo("Initialise lvalue");
        case ValueCategory::DValue:
            ICE(addr.loc, "Dependent value in constant evaluation");
            return false;

        case ValueCategory::SRValue: {
            auto InitBuiltin = [&]<typename T>(T get_value(Value&), T default_value) {
                if (auto i = init.get_or_null()) {
                    Assert(i->value_category == Expr::SRValue);
                    Value val;
                    TryEval(val, i);
                    return StoreMemory(addr, get_value(val), i->location());
                }

                // No initialiser. Initialise it to 0.
                return StoreMemory(addr, default_value, loc);
            };

            if (addr.type == Types::IntTy) return InitBuiltin(
                +[](Value& v) { return v.cast<APInt>().getZExtValue(); },
                u64(0)
            );

            if (addr.type == Types::BoolTy) return InitBuiltin(
                +[](Value& v) { return v.cast<bool>(); },
                false
            );

            if (isa<IntType>(addr.type.ptr())) {
                if (auto i = init.get_or_null()) {
                    Value val;
                    TryEval(val, i);
                    auto& ai = val.cast<APInt>();
                    auto data = ai.getRawData();
                    auto size = ai.getNumWords() * Size::Of<u64>();
                    return StoreMemory(addr, data, size, loc);
                }

                mem->zero();
                return true;
            }

            if (isa<ProcType>(addr.type)) {
                if (auto i = init.get_or_null()) {
                    Assert(i->value_category == Expr::SRValue);
                    Value closure;
                    TryEval(closure, i);
                    return StoreMemory(addr, Closure{closure.cast<ProcDecl*>()}, i->location());
                }

                ICE(loc, "Uninitialised closure in constant evaluator");
                return false;
            }

            return Error(
                loc,
                "Unsupported variable type in constant evaluation: '{}'",
                addr.type
            );
        }
    }

    Unreachable();
}

bool EvaluationContext::PerformVarInit(LValue& addr, Ptr<Expr> init, Location loc) {
    auto* mem = addr.base.get<Memory*>();
    Assert(mem->dead(), "Already initialised?");

    // TODO: Handle intents.

    mem->init();
    if (not PerformAssign(addr, init, loc)) {
        mem->destroy();
        return false;
    }

    return true;
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
bool EvaluationContext::EvalAssertExpr(Value& out, AssertExpr* expr) {
    TryEval(out, expr->cond);

    // If the assertion fails, print the message if there is one and
    // if it is a constant expression.
    if (not out.cast<bool>()) {
        if (auto m = expr->message.get_or_null(); m and Eval(out, m)) {
            auto sl = out.cast<Slice>();
            auto mem = GetMemoryBuffer(sl, m->location());
            if (mem == std::nullopt) return false;
            Error(
                expr->location(),
                "Assertion failed:\f'{}': {}",
                expr->cond->location().text(tu.context()),
                StringRef{mem->data(), mem->size()}
            );
        } else {
            Error(
                expr->location(),
                "Assertion failed:\f'{}'",
                expr->cond->location().text(tu.context())
            );
        }

        // Try to decompose the message to print a more helpful error.
        if (
            auto bin = dyn_cast<BinaryExpr>(expr->cond->strip_parens());
            bin and
            (bin->op == Tk::EqEq or bin->op == Tk::Neq)
        ) {
            Value left, right;
            Assert(Eval(left, bin->lhs));
            Assert(Eval(right, bin->rhs));
            Remark(
                "\rHelp: Comparison evaluates to '{} {} {}'",
                left,
                Spelling(bin->op),
                right
            );
        }

        return false;
    }

    // Assertion returns void.
    out = {};
    return true;
}

bool EvaluationContext::EvalBinaryExpr(Value& out, BinaryExpr* expr) {
    using enum OverflowBehaviour;

    // Some operators need special handling because not every
    // operand may be evaluated.
    switch (expr->op) {
        default: break;
        case Tk::And:
        case Tk::Or:
            return ICE(expr->location(), "TODO: Handle 'and' and 'or' properly");

        // This, as ever, is just variable initialisation.
        case Tk::Assign: {
            TryEval(out, expr->lhs);
            return PerformAssign(out.cast<LValue>(), expr->rhs, expr->location());
        }
    }

    // Any other operators require us to evaluate both sides.
    Value lhs, rhs;
    TryEval(lhs, expr->lhs);
    TryEval(rhs, expr->rhs);

    auto EvalAndCheckOverflow = [&]( // clang-format off
        APInt (APInt::* op)(const APInt&, bool&) const,
        OverflowBehaviour ob
    ) { // clang-format on
        auto& left = lhs.cast<APInt>();
        auto& right = rhs.cast<APInt>();

        bool overflow = false;
        out = {(left.*op)(right, overflow), lhs.type()};

        // Handle overflow.
        if (not overflow) return true;
        switch (ob) {
            case Wrap: return true;
            case Trap: {
                return Error(
                    expr->location(),
                    "Integer overflow in calculation\f'{} {} {}'",
                    toString(left, 10, true),
                    expr->op,
                    toString(right, 10, true)
                );
            }
        }

        Unreachable();
    };

    auto EvalUnchecked = [&](APInt (APInt::*op)(const APInt&) const) {
        out = {(lhs.cast<APInt>().*op)(rhs.cast<APInt>()), lhs.type()};
        return true;
    };

    auto EvalCompare = [&](bool (APInt::*op)(const APInt&) const) {
        out = (lhs.cast<APInt>().*op)(rhs.cast<APInt>());
        return true;
    };

    auto EvalInPlace = [&]<typename T = void>(T (APInt::*op)(const APInt&)) {
        (lhs.cast<APInt>().*op)(rhs.cast<APInt>());
        out = std::move(lhs);
        return true;
    };

    switch (expr->op) {
        default: Unreachable("Invalid operator: {}", expr->op);

        // Array or slice subscript.
        case Tk::LBrack: return ICE(expr->location(), "Array/slice subscript not supported yet");
        case Tk::In: return ICE(expr->location(), "Operator 'in' not yet implemented");

        // APInt doesn’t support this, so we have to do it ourselves.
        case Tk::StarStar: {
            auto base = lhs.cast<APInt>();
            auto exp = rhs.cast<APInt>();
            auto One = [&] { return APInt::getOneBitSet(base.getBitWidth(), 0); };
            auto Zero = [&] { return APInt::getZero(base.getBitWidth()); };
            auto MinusOne = [&] { return APInt::getAllOnes(base.getBitWidth()); };

            // Anything to the power of 0, including 0 itself, is 1,
            // *by definition*.
            if (exp.isZero()) {
                out = {One(), lhs.type()};
                return true;
            }

            // 0 to the power of a positive number is 0, and 0 to the
            // power of a negative number is undefined.
            if (base.isZero()) {
                if (exp.isNegative()) {
                    Error(
                        expr->location(),
                        "Undefined operation: 0 ** {}",
                        toString(exp, 10, true)
                    );

                    Remark(
                        "\r\vTaking 0 to the power of a negative number would "
                        "require division by zero, which is not defined for "
                        "integers."
                    );

                    return false;
                }

                out = {Zero(), lhs.type()};
                return true;
            }

            // If the exponent is negative, the result will be a fraction,
            // which collapses to 0 because this is integer arithmetic,
            // unless the base is 1, in which case the result is always 1,
            // or if the base is -1, in which case the result is -1 instead
            // if the exponent is odd.
            //
            // Check for this first so we avoid having to negate 'base' in
            // this case if it happens to be INT_MIN.
            if (exp.isNegative()) {
                if (base.isOne()) out = {One(), lhs.type()};
                else if (base.isAllOnes()) out = {exp[0] ? MinusOne() : One(), lhs.type()};
                else out = {Zero(), lhs.type()};
                return true;
            }

            // If the base is negative, the result will be negative if the
            // exponent is odd, and positive otherwise.
            // FIXME: I don’t think we need this negation dance here?
            bool negate = base.isNegative();
            if (negate) {
                // If base is INT_MIN, then its negation is not representable;
                // furthermore, we only get here if the exponent is positive,
                // which means that exponentiation would result in an even
                // larger value, which means that this overflows either way.
                if (base.isMinSignedValue()) Error(
                    expr->location(),
                    "Integer overflow in calculation\f'{} ** {}'",
                    toString(base, 10, true),
                    toString(exp, 10, true)
                );

                // Otherwise, the negation is fine.
                base.negate();
            }

            // Finally, both the base and exponent are positive here; we
            // can now perform the calculation.
            out = {One(), lhs.type()};
            auto& res = out.cast<APInt>();
            for (auto i = exp; i != 0; --i) {
                bool overflow = false;
                res = res.smul_ov(base, overflow);
                if (overflow) {
                    if (negate) base.negate();
                    return Error(
                        expr->location(),
                        "Integer overflow in calculation\f'{} ** {}'",
                        toString(base, 10, true),
                        toString(exp, 10, true)
                    );
                }
            }

            // Negate if necessary.
            //
            // This is always fine since the smallest negative number is
            // always greater in magnitude than the largest positive number
            // because of how two’s complement works.
            if (negate and exp[0]) res.negate();
            return true;
        }

        // These operations can overflow.
        case Tk::Star: return EvalAndCheckOverflow(&APInt::smul_ov, Trap);
        case Tk::Slash: return EvalAndCheckOverflow(&APInt::sdiv_ov, Trap);
        case Tk::StarTilde: return EvalAndCheckOverflow(&APInt::smul_ov, Wrap);
        case Tk::Plus: return EvalAndCheckOverflow(&APInt::sadd_ov, Trap);
        case Tk::PlusTilde: return EvalAndCheckOverflow(&APInt::sadd_ov, Wrap);
        case Tk::Minus: return EvalAndCheckOverflow(&APInt::ssub_ov, Trap);
        case Tk::MinusTilde: return EvalAndCheckOverflow(&APInt::ssub_ov, Wrap);
        case Tk::ShiftLeft: return EvalAndCheckOverflow(&APInt::sshl_ov, Trap);
        case Tk::ShiftLeftLogical: return EvalAndCheckOverflow(&APInt::ushl_ov, Trap);

        // These physically can’t.
        case Tk::ColonSlash: return EvalUnchecked(&APInt::udiv);
        case Tk::ColonPercent: return EvalUnchecked(&APInt::urem);
        case Tk::Percent: return EvalUnchecked(&APInt::srem);

        // For these, we can avoid allocating a new APInt.
        case Tk::ShiftRight: return EvalInPlace(&APInt::ashrInPlace);
        case Tk::ShiftRightLogical: return EvalInPlace(&APInt::lshrInPlace);
        case Tk::Ampersand: return EvalInPlace(&APInt::operator&=);
        case Tk::VBar: return EvalInPlace(&APInt::operator|=);

        // Comparison operators.
        case Tk::ULt: return EvalCompare(&APInt::ult);
        case Tk::UGt: return EvalCompare(&APInt::ugt);
        case Tk::ULe: return EvalCompare(&APInt::ule);
        case Tk::UGe: return EvalCompare(&APInt::uge);
        case Tk::SLt: return EvalCompare(&APInt::slt);
        case Tk::SGt: return EvalCompare(&APInt::sgt);
        case Tk::SLe: return EvalCompare(&APInt::sle);
        case Tk::SGe: return EvalCompare(&APInt::sge);
        case Tk::EqEq: return EvalCompare(&APInt::operator==);
        case Tk::Neq: return EvalCompare(&APInt::operator!=);

        // Exclusive or.
        // This needs special handling because it’s supported for both bools and ints.
        case Tk::Xor:
            return ICE(expr->location(), "Unsupported operator in constant evaluation: 'xor'");

        // Assignment operators.
        case Tk::PlusEq:
        case Tk::PlusTildeEq:
        case Tk::MinusEq:
        case Tk::MinusTildeEq:
        case Tk::StarEq:
        case Tk::StarTildeEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftLeftLogicalEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq: {
            return ICE(expr->location(), "Compound assignment operators not supported yet");
        }
    }
}

bool EvaluationContext::EvalBlockExpr(Value& out, BlockExpr* block) {
    // FIXME: Once we have destructors, this information should
    // just be available in the BlockExpr.
    struct VariableRAII {
        SRCC_IMMOVABLE(VariableRAII);
        SmallVector<Memory*> locals_to_destroy;
        VariableRAII() = default;
        ~VariableRAII() {
            for (auto v : locals_to_destroy)
                v->destroy();
        }
    } initialised_vars;

    out = {};
    for (auto s : block->stmts()) {
        Value val;

        // Variables need to be initialised.
        if (auto l = dyn_cast<LocalDecl>(s)) {
            auto& loc = AllocateVar(CurrFrame().locals, l);
            if (not PerformVarInit(loc, l->init, l->location()))
                return false;
            initialised_vars.locals_to_destroy.push_back(loc.base.get<Memory*>());
        }

        // Any other decls can be skipped.
        if (isa<Decl>(s)) continue;

        // Anything else needs to be evaluated.
        TryEval(val, s);
        if (s == block->return_expr()) out = std::exchange(val, {});
    }
    return true;
}

bool EvaluationContext::EvalBoolLitExpr(Value& out, BoolLitExpr* expr) {
    out = expr->value;
    return true;
}

bool EvaluationContext::EvalBuiltinCallExpr(Value& out, BuiltinCallExpr* builtin) {
    switch (builtin->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            for (auto arg : builtin->args()) {
                TryEval(out, arg);

                // String.
                if (auto slice = out.dyn_cast<Slice>()) {
                    auto mem = GetMemoryBuffer(*slice, arg->location());
                    if (mem == std::nullopt) return false;
                    std::print("{}", StringRef{mem->data(), mem->size()});
                    continue;
                }

                // Integer.
                if (auto int_val = out.dyn_cast<APInt>()) {
                    std::print("{}", toString(*int_val, 10, true));
                    continue;
                }

                // Bool.
                if (auto bool_val = out.dyn_cast<bool>()) {
                    std::print("{}", *bool_val);
                    continue;
                }

                Unreachable("Invalid value in __srcc_print call");
            }

            return true;
        }
    }

    Unreachable("Invalid builtin kind");
}

bool EvaluationContext::EvalCallExpr(Value& out, CallExpr* call) {
    TryEval(out, call->callee);
    auto proc = out.cast<ProcDecl*>();
    auto args = call->args();

    // If we have a body, just evaluate it.
    if (auto body = proc->body().get_or_null()) {
        StackFrame::LocalsMap locals;

        // Allocate and initialise local variables and initialise them. Do
        // this before we set up the stack frame, otherwise, we’ll try to
        // access the locals in the new frame before we’re done setting them
        // up.
        //
        // For small 'in' parameters that are passed by value, just save the
        // actual value in the parameter slot.
        for (auto [i, p] : enumerate(proc->params())) {
            auto InitVarFromRValue = [&] {
                auto& addr = AllocateVar(locals, p);
                return PerformVarInit(addr, args[i], p->location());
            };

            auto UseValueAsVar = [&](bool lvalue = true) {
                Value v;
                TryEval(v, args[i]);

                // Verify that this is an lvalue and adjust the location.
                if (lvalue) {
                    Assert(v.isa<LValue>(), "{} arg must be an lvalue", p->intent());
                    auto lval = v.cast<LValue>();
                    lval.loc = p->location();
                    locals.try_emplace(p, lval);
                    return true;
                }

                // If it is an rvalue, just bind the parameter to it.
                locals.try_emplace(p, std::move(v));
                return true;
            };

            switch (p->intent()) {
                // These are lvalues.
                case Intent::Out:
                case Intent::Inout:
                    TRY(UseValueAsVar());
                    break;

                // Copy always passes by rvalue and creates a variable
                // in the callee.
                case Intent::Copy:
                    TRY(InitVarFromRValue());
                    break;

                // Move may pass by rvalue or lvalue; if lvalue, that lvalue
                // becomes the variable; if rvalue, a variable is created in
                // the callee.
                case Intent::Move:
                    if (p->type->pass_by_rvalue(proc->cconv(), p->intent())) TRY(InitVarFromRValue());
                    else TRY(UseValueAsVar());
                    break;

                // 'in' is similar, except that no variable is created in the
                // callee either way.
                case Intent::In: {
                    TRY(UseValueAsVar(not p->type->pass_by_rvalue(proc->cconv(), p->intent())));

                    // If this was an lvalue, make it readonly, and remember to reset
                    // it when we return from this if we actually made it readonly.
                    if (auto lv = locals[p].dyn_cast<LValue>()) lv->make_readonly();
                } break;
            }
        }

        // Set up stack.
        PushStackFrame _{*this, proc, std::move(locals), call->location()};

        // Dew it.
        TryEval(out, body);
        out = CurrFrame().return_value;
        return true;
    }

    // FIXME: We can only call functions defined in this module; calling
    // external functions would only be possible if we had the module providing
    // them present as a shared library (explore generating a shared library
    // and throwing it in /tmp)
    //
    // TODO: Otherwise, try to perform an external call.
    return Error(call->location(), "Sorry, can’t call external functions at compile-time yet");
}

bool EvaluationContext::EvalCastExpr(Value& out, CastExpr* cast) {
    TryEval(out, cast->arg);
    switch (cast->kind) {
        case CastExpr::LValueToSRValue: {
            auto lvalue = out.cast<LValue>();

            // Builtin int.
            if (lvalue.type == Types::IntTy) {
                u64 value;
                TRY(LoadMemory(lvalue, value, cast->location()));
                out = IntValue(value);
                return true;
            }

            // (Potentially) large integer.
            if (auto i = dyn_cast<IntType>(lvalue.type.ptr())) {
                SmallVector<u64> words;
                words.resize(lvalue.type->size(tu).aligned(Align::Of<u64>()) / Size::Of<u64>());
                if (not LoadMemory(lvalue, words.data(), lvalue.type->size(tu), cast->location()))
                    return false;
                out = {APInt{unsigned(i->bit_width().bits()), unsigned(words.size()), words.data()}, lvalue.type};
                return true;
            }

            // Bool.
            if (lvalue.type == Types::BoolTy) {
                bool value;
                TRY(LoadMemory(lvalue, value, cast->location()));
                out = value;
                return true;
            }

            // Closures.
            if (isa<ProcType>(lvalue.type)) {
                Closure cl;
                TRY(LoadMemory(lvalue, cl, cast->location()));
                out = cl.decl;
                return true;
            }

            ICE(
                cast->location(),
                "Sorry, we don’t support lvalue->srvalue conversion of '{}' yet",
                lvalue.type
            );

            return false;
        }

        case CastExpr::Integral: {
            auto adjusted = out.cast<APInt>().sextOrTrunc(unsigned(cast->type->size(tu).bits()));
            out = {adjusted, cast->type};
            return true;
        }
    }

    Unreachable("Invalid cast");
}

bool EvaluationContext::EvalConstExpr(Value& out, ConstExpr* constant) {
    out = *constant->value;
    return true;
}

bool EvaluationContext::EvalEvalExpr(Value& out, EvalExpr* eval) {
    EvaluationContext C{tu, complain};
    return C.Eval(out, eval->stmt);
}

bool EvaluationContext::EvalIfExpr(Value& out, IfExpr* expr) {
    TryEval(out, expr->cond);

    // Always reset to an empty value if this isn’t supposed
    // to yield anything.
    defer {
        if (not expr->has_yield()) out = {};
    };

    if (out.cast<bool>()) return Eval(out, expr->then);
    if (auto e = expr->else_.get_or_null()) return Eval(out, e);
    return true;
}

bool EvaluationContext::EvalIntLitExpr(Value& out, IntLitExpr* int_lit) {
    out = {int_lit->storage.value(), int_lit->type};
    return true;
}

bool EvaluationContext::EvalLocalDecl(Value&, LocalDecl* ld) {
    Error(ld->location(), "Local variable is not a constant expression");
    return false;
}

bool EvaluationContext::EvalLocalRefExpr(Value& out, LocalRefExpr* local) {
    // Walk up the stack until we find an entry for this variable; note
    // that the same function may be present in multiple frames, so just
    // find the nearest one.
    for (auto& frame : vws::reverse(stack)) {
        auto it = frame.locals.find(local->decl);
        if (it != frame.locals.end()) {
            out = it->second;
            return true;
        }
    }

    // This should never happen since we always create all
    // locals when we set up the stack frame.
    Unreachable("Local variable not found: {}", local->decl->name);
}

bool EvaluationContext::EvalOverloadSetExpr(Value&, OverloadSetExpr*) {
    // FIXME: A function that returns an overload set should be fine
    // so long as it is only called at compile-time.
    Unreachable("Evaluating unresolved overload set?");
}

bool EvaluationContext::EvalParenExpr(Value& out, ParenExpr* expr) {
    return Eval(out, expr->expr);
}

bool EvaluationContext::EvalParamDecl(Value& out, ParamDecl* ld) {
    return EvalLocalDecl(out, ld);
}

bool EvaluationContext::EvalProcDecl(Value&, ProcDecl*) {
    Unreachable("Never evaluated");
}

bool EvaluationContext::EvalTemplateTypeDecl(Value&, TemplateTypeDecl*) {
    Unreachable("Syntactically impossible to get here");
}

bool EvaluationContext::EvalProcRefExpr(Value& out, ProcRefExpr* proc_ref) {
    out = proc_ref->decl;
    return true;
}

bool EvaluationContext::EvalSliceDataExpr(Value& out, SliceDataExpr* slice_data) {
    TryEval(out, slice_data->slice);
    auto data = std::move(out.dyn_cast<Slice>()->data);
    out = {std::move(data), slice_data->type};
    return true;
}

bool EvaluationContext::EvalStrLitExpr(Value& out, StrLitExpr* str_lit) {
    out = Value{
        Slice{
            Reference{LValue{str_lit->value, str_lit->type, str_lit->location()}, APInt::getZero(64)},
            APInt(u32(Types::IntTy->size(tu).bits()), str_lit->value.size(), false),
        },
        tu.StrLitTy,
    };
    return true;
}

bool EvaluationContext::EvalReturnExpr(Value& out, ReturnExpr* expr) {
    Assert(CurrFrame().return_value.isa<std::monostate>(), "Return value already set!");
    out = {};
    if (auto val = expr->value.get_or_null()) return Eval(CurrFrame().return_value, val);
    return true;
}

bool EvaluationContext::EvalTypeExpr(Value& out, TypeExpr* expr) {
    out = expr->value;
    return true;
}

bool EvaluationContext::EvalUnaryExpr(Value& out, UnaryExpr* expr) {
    TryEval(out, expr->arg);
    if (expr->postfix) return ICE(expr->location(), "Postfix unary operators not supported yet");
    switch (expr->op) {
        default: Unreachable("Invalid prefix operator: {}", expr->op);

        // Boolean negation.
        case Tk::Not: {
            out = not out.cast<bool>();
            return true;
        }

        // Arithmetic operators.
        case Tk::Minus: {
            auto& i = out.cast<APInt>();
            // This can overflow if this is INT_MIN.
            if (i.isMinSignedValue()) {
                Error(
                    expr->location(),
                    "Integer overflow in calculation\f'- {}'",
                    toString(i, 10, true)
                );

                Remark(
                    "\vThe {}-bit value '{}', the smallest representable negative value, "
                    "has no positive {}-bit counterpart because of two’s complement.",
                    i.getBitWidth(),
                    toString(i, 10, true),
                    i.getBitWidth()
                );

                return false;
            }

            i.negate();
            return true;
        }

        case Tk::Plus: return true;
        case Tk::Tilde: {
            out.cast<APInt>().flipAllBits();
            return true;
        }

        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            return ICE(expr->location(), "Increment/decrement operators not supported yet");
        }
    }
}

bool EvaluationContext::EvalWhileStmt(Value& out, WhileStmt* expr) {
    defer { out = {}; };
    for (;;) {
        if (not Eval(out, expr->cond)) return false;
        if (not out.cast<bool>()) return true;
        if (not Eval(out, expr->body)) return false;
    }
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
