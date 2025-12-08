#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/MLIRFormatters.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/Allocator.h>
#include <llvm/TargetParser/Host.h>

#include <base/Colours.hh>
#include <base/Formatters.hh>
#include <base/Macros.hh>
#include <base/StringUtils.hh>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <ffi.h>
#include <memory>
#include <optional>
#include <print>
#include <srcc/CG/Target/Target.hh>

using namespace srcc;
using namespace srcc::eval;
namespace ir = cg::ir;
using mlir::Block;
using mlir::Value;

static auto LoadRangeFromMemory(
    const void* ptr,
    TranslationUnit& tu,
    RangeType* r
) -> Range {
    auto ptr_u = static_cast<const u8*>(ptr);
    auto elem_a = r->elem()->align(tu);
    auto bytes = unsigned(r->elem()->memory_size(tu).bytes());
    auto bits = unsigned(r->elem()->bit_width(tu).bits());
    APInt start{bits, 0}, end{bits, 0};
    llvm::LoadIntFromMemory(start, ptr_u, bytes);
    llvm::LoadIntFromMemory(end, elem_a.align(ptr_u + bytes), bytes);
    return {start.trunc(bits), end.trunc(bits)};
}

TreeValue::TreeValue(QuoteExpr* pattern, ArrayRef<TreeValue*> unquotes) : q{pattern} {
    Assert(unquotes.size() == pattern->unquotes().size());
    std::uninitialized_copy_n(unquotes.begin(), unquotes.size(), getTrailingObjects());
}

auto TreeValue::dump(const Context* ctx) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    Format(out, "%6({}%)@", enchantum::to_string(q->quoted->kind()));
    auto lc = ctx ? q->location().seek_line_column(*ctx) : std::nullopt;
    if (lc) Format(out, "%5(<{}:{}>%)", lc->line, lc->col);
    else Format(out, "%5(<{}>%)", q->location().encode());
    return out;
}

auto TreeValue::unquotes() const -> ArrayRef<TreeValue*> {
    return getTrailingObjects(q->unquotes().size());
}

void RValue::dump() const {
    std::println("{}", text::RenderColours(false, print().str()));
}

auto RValue::print(const Context* ctx) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{
        // clang-format off
        [&](std::monostate) {},
        [&](Type ty) { out += ty->print(); },
        [&](TreeValue* tree) { out += tree->dump(ctx); },
        [&](const RawByteBuffer& b) {
            if (b.is_string()) out += utils::Escape(b.str(), true, true);
            out += "<aggregate value>";
        },
        [&](EvaluatedPointer p) {
            switch (p.kind()) {
                case EvaluatedPointer::Null: out += "%1(nil%)"; return;
                case EvaluatedPointer::Procedure: Format(out, "%2({}%)", p.proc()->name); return;
                case EvaluatedPointer::InvalidStack: out += "<vm stack pointer>"; return;
                case EvaluatedPointer::Unknown: out += "<unknown pointer>"; return;
            }

            Unreachable();
        },
        [&](this auto& self, const Range& r) {
            self(r.start);
            out += "%1(..<%)";
            self(r.end);
        },
        [&](this auto& self, const Slice& s) {
            out += "%1((%)";
            self(s.pointer);
            Format(out, "%1(, %){}%1()%)", s.size);
        },
        [&](const APInt& value) {
            if (type() == Type::BoolTy) out += value.getBoolValue() ? "%1(true%)"sv : "%1(false%)"sv;
            else Format(out, "%5({}%)", toString(value, 10, true));
        },
        [&](this auto& self, const Closure& c) {
            if (c.env.is_null()) {
                self(c.proc);
                return;
            }

            out += "<closure";
            self(c.proc);
            out += ", env: ";
            self(c.env);
            out += ">";
        },
        [&](this auto& self, const Record& r) {
            if (not llvm::isa<TupleType>(type())) out += type()->print();
            out += "%1((%)";
            bool first = true;
            for (auto* f : r.fields) {
                if (first) first = false;
                else out += "%1(, %)";
                out += f->print(ctx);
            }
            out += "%1()%)";
        }
    }; // clang-format on
    visit(V);
    return out;
}

#define TRY(x)                 \
    do {                       \
        if (not(x)) return {}; \
    } while (false)

// ============================================================================
//  SRValue Representation
// ============================================================================
namespace {
/// A pointer value; this is either a literal pointer or a 1-based index into
/// a list of procedures; the latter is used to represent pointers to interpreted
/// procedures.
///
/// \see VirtualMemoryMap
class Pointer {
    friend VirtualMemoryMap; // Only this is allowed to construct and unwrap pointers.
    uptr value{};
    Pointer(uptr v) : value{v} {}

public:
    Pointer() = default;

    /// Align this pointer.
    [[nodiscard]] auto align(Align a) const -> Pointer {
        return Pointer(uptr(Size::Bytes(value).align(a).bytes()));
    }

    /// Check if this is the null pointer.
    [[nodiscard]] bool is_null() const { return value == 0; }

    /// Offset this pointer.
    [[nodiscard]] auto offset(Size sz) const { return Pointer(value + sz.bytes()); }

    /// Get a string representation of this pointer.
    [[nodiscard]] auto str() const -> SmallString<64> {
        return Format("{}", reinterpret_cast<void*>(value));
    }

    /// Get the raw value suitable for storing to memory.
    [[nodiscard]] auto raw_value() const -> uptr { return value; }

    [[nodiscard]] friend bool operator==(Pointer, Pointer) = default;
    [[nodiscard]] static auto Null() -> Pointer { return Pointer(); }
    [[nodiscard]] explicit operator bool() const { return not is_null(); }
};

/// A compile-time srvalue.
///
/// This is essentially a value that can be stored in a VM ‘virtual register’.
class SRValue {
    Variant<APInt, Type, TreeValue*, Pointer, std::monostate> value{std::monostate{}};

public:
    SRValue() = default;
    explicit SRValue(std::same_as<bool> auto b) : value{APInt{1, u64(b)}} {}
    explicit SRValue(Type ty) : value{ty} {}
    explicit SRValue(TreeValue* tree) : value{tree} {}
    explicit SRValue(Pointer p) : value{p} {}
    explicit SRValue(APInt val) : value(std::move(val)) {}
    explicit SRValue(std::same_as<i64> auto val) : value{APInt{64, u64(val)}} {}

    [[nodiscard]] bool operator==(const SRValue& other) const;

    /// cast<>() the contained value.
    template <typename Ty>
    [[nodiscard]] auto cast() & -> Ty& { return std::get<Ty>(value); }

    template <typename Ty>
    [[nodiscard]] auto cast() && -> Ty&& { return std::move(std::get<Ty>(value)); }

    template <typename Ty>
    [[nodiscard]] auto cast() const -> const Ty& { return std::get<Ty>(value); }

    /// dyn_cast<>() the contained value.
    template <typename Ty>
    [[nodiscard]] auto dyn_cast() const -> const Ty* {
        return std::holds_alternative<Ty>(value) ? &std::get<Ty>(value) : nullptr;
    }

    /// Print this value.
    void dump(bool use_colour = true) const;
    void dump_colour() const { dump(true); }

    /// Get the index of the contained value.
    [[nodiscard]] auto index() const -> usz { return value.index(); }

    /// isa<>() on the contained value.
    template <typename Ty>
    [[nodiscard]] auto isa() const -> bool { return std::holds_alternative<Ty>(value); }

    /// Print the value to a string.
    [[nodiscard]] auto print(const Context* ctx = nullptr) const -> SmallUnrenderedString;

    /// Run a visitor over this value.
    template <typename Self, typename Visitor>
    [[nodiscard]] auto visit(this Self&& self, Visitor&& visitor) -> decltype(auto) {
        return std::visit(
            std::forward<Visitor>(visitor),
            std::forward_like<Self>(self.value)
        );
    }
};
} // namespace

template <>
struct std::formatter<SRValue> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const SRValue& val, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{val.print().str()}, ctx);
    }
};

LIBBASE_DEBUG(__attribute__((used))) void SRValue::dump(bool use_colour) const {
    std::println("{}", text::RenderColours(use_colour, print().str()));
}

auto SRValue::print(const Context* ctx) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{
        // clang-format off
        [&](std::monostate) {},
        [&](ir::ProcOp proc) { Format(out, "%2({}%)", proc.getName()); },
        [&](TreeValue* tree) { out += tree->dump(ctx); },
        [&](Type ty) { out += ty->print(); },
        [&](const APInt& value) { Format(out, "%5({}%)", toString(value, 10, true)); },
        [&](Pointer ptr) { Format(out, "%4({}%)", ptr.str()); }
    }; // clang-format on

    visit(V);
    return out;
}

bool SRValue::operator==(const SRValue& other) const {
    if (index() != other.index()) return false;
    return visit(utils::Overloaded{[&]<typename T>(const T& t) { return other.cast<T>() == t; }, [&](const APInt& a) {
        const auto& b = other.cast<APInt>();
        if (a.getBitWidth() != b.getBitWidth()) return false;
        return a == b;
    }});
}

// ============================================================================
//  Virtual Address Map
// ============================================================================
/// This class implements a virtual memory map for interpreted procedures.
///
/// In the evaluator, we need to distinguish between two kinds of pointers:
///
///   1. Pointers to host memory; these are used for allocated stack and heap
///      memory, as well as for pointers to native (compiled) functions.
///
///   2. Pointers to IR procedures that only exist in the interpreter.
///
/// These two are fundamentally different: native function pointers can be
/// called directly, while IR procedure pointers must be interpreted; we need
/// to be able to distinguish the two, but at the same time, the evaluator must
/// be able to treat all of them as pointers with the same memory representation.
///
/// Our solution to this is to reserve a region of heap memory (~1 million bytes),
/// each byte of which is actually an index into a vector of IR procedures; this
/// allows us to distinguish these two kinds of pointers by checking whether they
/// happen to fall into said reserved address range; for this to work, it actually
/// has to be reserved, i.e. we can’t use it for anything.
///
/// The zero value is used to represent the null pointer for both address spaces.
class eval::VirtualMemoryMap {
    static constexpr usz MapSize = 1 << 20;
    std::unique_ptr<std::byte[]> address_range = std::make_unique<std::byte[]>(MapSize);
    SmallVector<ir::ProcOp> procedures;
    DenseMap<ir::ProcOp, Pointer> lookup;

    /// Size of the stack.
    static constexpr Size max_stack_size = Size::Bytes(10 * 1024 * 1024);

    /// Stack memory for the evaluator.
    // TODO: Make this value configurable (via --feval-stack-size or sth.).
    std::unique_ptr<std::byte[]> stack = std::make_unique<std::byte[]>(max_stack_size.bytes());

public:
    VirtualMemoryMap() = default;

    /// Map a pointer to the procedure it references.
    [[nodiscard]] auto get_procedure(Pointer p) -> ir::ProcOp;

    /// Map a pointer to a pointer to host memory.
    template <typename T = void>
    [[nodiscard]] auto get_host_pointer(Pointer p) -> T*;

    /// Get the start of stack memory.
    [[nodiscard]] auto get_stack_bottom() -> Pointer;

    /// Check if a pointer is a host memory pointer.
    [[nodiscard]] bool is_host_pointer(Pointer p);

    /// Check if this is a pointer to stack memory.
    [[nodiscard]] bool is_stack_pointer(Pointer p);

    /// Check if a pointer is a virtual procedure pointer.
    [[nodiscard]] bool is_virtual_proc_ptr(Pointer p);

    /// Create a pointer to host memory.
    [[nodiscard]] auto make_host_pointer(uptr v) -> Pointer;

    /// Add a procedure to the table if it isn't already registered and return a VM
    /// pointer to it.
    [[nodiscard]] auto make_proc_ptr(ir::ProcOp proc) -> Pointer;

    /// Get the maximum stack size.
    [[nodiscard]] auto stack_size() -> Size { return max_stack_size; }

private:
    bool IsVirtualProcPtr(uptr p);
    auto MakeVirtualPointer(ir::ProcOp proc) -> Pointer;
    auto UnwrapVirtualPointer(Pointer ptr) -> ir::ProcOp;
};

bool VirtualMemoryMap::IsVirtualProcPtr(uptr p) {
    uptr start = uptr(address_range.get());
    uptr end = start + MapSize;
    return p >= start and p < end;
}

auto VirtualMemoryMap::MakeVirtualPointer(ir::ProcOp proc) -> Pointer {
    // Get the index AFTER insertion because it’s 1-based.
    procedures.push_back(proc);
    return Pointer(uptr(address_range.get() + procedures.size()));
}

auto VirtualMemoryMap::UnwrapVirtualPointer(Pointer ptr) -> ir::ProcOp {
    Assert(is_virtual_proc_ptr(ptr));
    auto idx = usz(ptr.value - uptr(address_range.get()) - 1);
    if (idx >= procedures.size()) return {};
    return procedures[idx];
}

template <typename T>
auto VirtualMemoryMap::get_host_pointer(Pointer p) -> T* {
    Assert(is_host_pointer(p));
    return reinterpret_cast<T*>(static_cast<char*>(reinterpret_cast<void*>(p.value)));
}

auto VirtualMemoryMap::get_procedure(Pointer p) -> ir::ProcOp {
    return UnwrapVirtualPointer(p);
}

auto VirtualMemoryMap::get_stack_bottom() -> Pointer {
    return Pointer(uptr(stack.get()));
}

bool VirtualMemoryMap::is_host_pointer(Pointer p) {
    return not is_virtual_proc_ptr(p);
}

bool VirtualMemoryMap::is_stack_pointer(Pointer p) {
    uptr start = uptr(stack.get());
    uptr end = start + max_stack_size.bytes();
    return p.value >= start and p.value < end;
}

bool VirtualMemoryMap::is_virtual_proc_ptr(Pointer p) {
    uptr start = uptr(address_range.get());
    uptr end = start + MapSize;
    return p.value >= start and p.value < end;
}

auto VirtualMemoryMap::make_host_pointer(uptr v) -> Pointer {
    // It is technically valid for the program to just randomly guess what the
    // reserved memory region is and build a pointer to it; we don’t want to crash
    // or error on that; we *will* error when trying to store to or load from it
    // anyway.
    return Pointer(v);
}

auto VirtualMemoryMap::make_proc_ptr(ir::ProcOp proc) -> Pointer {
    Assert(proc);
    auto& lookup_val = lookup[proc];
    if (not lookup_val) lookup_val = MakeVirtualPointer(proc);
    return Pointer(lookup_val);
}

// ============================================================================
//  Helpers
// ============================================================================
namespace {
/// Encoded temporary value.
///
/// This represents an encoded form of a temporary value that is guaranteed
/// to be unique *per procedure*.
enum struct Temporary : u64;
auto Encode(Value v) -> Temporary { return Temporary(u64(v.getAsOpaquePointer())); }
} // namespace

// ============================================================================
//  Evaluator
// ============================================================================
class eval::Eval : DiagsProducer {
public:
    friend DiagsProducer;

    /// A procedure on the stack.
    struct StackFrame {
        using RetVals = SmallVector<SRValue*, 2>;

        /// Location of the call.
        mlir::Location call_loc;

        /// Procedure to which this frame belongs.
        ir::ProcOp proc{};

        /// Instruction pointer for this procedure.
        Block::iterator ip{};

        /// Temporary values for each instruction.
        DenseMap<Temporary, SRValue> temporaries{};

        /// Other materialised temporaries (literal integers etc.)
        StableVector<SRValue> materialised_values{};

        /// Stack size at the start of this procedure.
        Pointer stack_base{};

        /// Return value slots.
        RetVals ret_vals{};

        /// Whether to abort constant evaluation after exiting this frame.
        bool is_assert_stringifier_frame = false;
    };

    VM& vm;
    cg::CodeGen cg;
    SmallVector<StackFrame, 4> call_stack;
    const SRValue true_val{true};
    const SRValue false_val{false};
    Pointer stack_top{};
    SLoc entry;
    bool complain;

    Eval(VM& vm, bool complain);

    [[nodiscard]] auto eval(Stmt* s) -> std::optional<RValue>;

private:
    auto diags() const -> DiagnosticsEngine& { return vm.owner().context().diags(); }
    auto frame() -> StackFrame& { return call_stack.back(); }

    template <typename... Args>
    bool Error(SLoc where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) diags().diag(Diagnostic::Level::Error, where, fmt, std::forward<Args>(args)...);
        return false;
    }

    template <typename... Args>
    void Remark(std::format_string<Args...> fmt, Args&&... args) {
        if (complain) diags().add_remark(std::format(fmt, std::forward<Args>(args)...));
    }

    [[nodiscard]] auto AdjustLangOpts(LangOpts l) -> LangOpts;
    [[nodiscard]] auto AllocateStackMemory(mlir::Location loc, Size sz, Align alignment) -> std::optional<Pointer>;
    [[nodiscard]] bool EvalLoop();
    [[nodiscard]] auto FFICall(ir::ProcOp proc, ir::CallOp call) -> std::optional<SRValue>;
    [[nodiscard]] auto FFIType(mlir::Type ty) -> ffi_type*;
    [[nodiscard]] auto GetHostMemoryPointer(Value v) -> void*;
    [[nodiscard]] auto LoadSRValue(const void* mem, mlir::Type ty) -> SRValue;
    [[nodiscard]] auto LoadInt(const void* mem, Size width) -> SRValue;
    [[nodiscard]] auto LoadPointer(const void* mem) -> SRValue;
    [[nodiscard]] auto LoadType(const void* mem) -> Type;
    [[nodiscard]] auto LoadTree(const void* mem) -> TreeValue*;
    [[nodiscard]] auto Persist(const void* mem, SLoc loc, Type ty) -> RValue;
    [[nodiscard]] auto Temp(Value v) -> SRValue&;
    [[nodiscard]] auto Val(Value v) -> const SRValue&;

    void BranchTo(Block* block, mlir::ValueRange args);
    void BranchTo(Block* block, MutableArrayRef<SRValue> args);
    void PushFrame(
        mlir::Location call_loc,
        ir::ProcOp proc,
        MutableArrayRef<SRValue> args,
        StackFrame::RetVals ret_vals
    );

    void StoreSRValue(void* mem, const SRValue& val);
};

Eval::Eval(VM& vm, bool complain)
    : vm{vm},
      cg{vm.owner(), AdjustLangOpts(vm.owner().lang_opts())},
      stack_top{vm.memory->get_stack_bottom()},
      complain{complain} {}

auto Eval::AdjustLangOpts(LangOpts l) -> LangOpts {
    l.constant_eval = true;
    l.overflow_checking = true;
    return l;
}

auto Eval::AllocateStackMemory(mlir::Location loc, Size sz, Align alignment) -> std::optional<Pointer> {
    auto ptr = stack_top.align(alignment);
    stack_top = ptr.offset(sz);
    if (vm.memory->is_stack_pointer(stack_top)) return ptr;
    Error(SLoc::Decode(loc), "Stack overflow");
    Remark(
        "This may have been caused by infinite recursion. If you don’t think that "
        "that’s the case, you can increase the maximum eval stack size by passing "
        "--feval-stack-size (current value: {:y})",
        vm.memory->stack_size()
    );
    return std::nullopt;
}


void Eval::BranchTo(Block* block, mlir::ValueRange args) {
    // Copy the argument values out of their slots in case we’re doing
    // something horrible like
    //
    //   bb1(int %1, int %2):
    //     ...
    //   br bb1(%2, %1)
    //
    // in which case simply assigning %2 to %1 would override the old value
    // of the latter before it can be used.
    SmallVector<SRValue, 6> copy;
    for (auto arg : args) copy.push_back(Val(arg));
    return BranchTo(block, copy);
}

void Eval::BranchTo(Block* block, MutableArrayRef<SRValue> args) {
    Assert(
        not block->empty(), "Malformed block in '{}'",
        cast<ir::ProcOp>(block->getParentOp()).getName()
    );

    frame().ip = block->begin();
    for (auto [slot, arg] : zip(block->getArguments(), args))
        Temp(slot) = std::move(arg);
}


bool Eval::EvalLoop() {
    auto CastOp = [&] [[nodiscard]] (
        mlir::Operation* i,
        mlir::Type ty,
        APInt (APInt::*op)(unsigned width) const
    ) -> bool {
        auto& val = Val(i->getOperand(0));
        auto to_wd = Size::Bits(cast<mlir::IntegerType>(ty).getWidth());
        auto result = (val.cast<APInt>().*op)(u32(to_wd.bits()));
        Temp(i->getResult(0)) = SRValue{std::move(result)};
        return true;
    };

#define CMP_OP(op)                                              \
    case op: {                                                  \
        const APInt& lhs = Val(i->getOperand(0)).cast<APInt>(); \
        const APInt& rhs = Val(i->getOperand(1)).cast<APInt>(); \
        Temp(i->getResult(0)) = SRValue{lhs.op(rhs)};           \
        continue;                                               \
    }

    // DO NOT use do {} while (false) here since we need to be able to
    // continue the outer loop...
#define INT_OP(op, expr)                                        \
    if (isa<mlir::arith::op>(i)) {                              \
        const APInt& lhs = Val(i->getOperand(0)).cast<APInt>(); \
        const APInt& rhs = Val(i->getOperand(1)).cast<APInt>(); \
        Temp(i->getResult(0)) = SRValue{APInt(expr)};           \
        continue;                                               \
    }

    auto OvOp = [&] [[nodiscard]] (
        mlir::Operation * i,
        APInt(APInt::* op)(const APInt& RHS, bool& Overflow) const
    ) -> bool {
        auto& lhs = Val(i->getOperand(0));
        auto& rhs = Val(i->getOperand(1));
        bool overflow = false;
        auto result = (lhs.cast<APInt>().*op)(rhs.cast<APInt>(), overflow);
        Temp(i->getResult(0)) = SRValue{std::move(result)};
        Temp(i->getResult(1)) = SRValue{overflow};
        return true;
    };

    static constexpr isz BufSize = 10000;
    struct AssertMessageBuffer {
        LIBBASE_IMMOVABLE(AssertMessageBuffer);
        char* data = new char[BufSize];
        isz size = 0;
        isz capacity = BufSize;
        AssertMessageBuffer() = default;
        ~AssertMessageBuffer() { delete[] data; }
    };

    std::optional<AssertMessageBuffer> assert_buffer;
    const u64 max_steps = vm.owner().context().eval_steps ?: std::numeric_limits<u64>::max();
    for (u64 steps = 0; steps < max_steps; steps++) {
        auto i = &*frame().ip++;
        if (auto a = dyn_cast<ir::AbortOp>(i)) {
            if (not complain) return false;

            // FIXME: We should not be relying on the native struct layout to be compatible
            // with what Source uses.
            struct AbortInfo {
                struct Slice {
                    const char* data;
                    isz size;
                    auto sv() const -> std::string_view { return {data, usz(size)}; }
                };

                struct Closure {
                    Pointer ptr;
                    Pointer env;
                };

                Slice filename;
                isz line;
                isz col;
                Slice msg1;
                Slice msg2;
                Closure closure;
            };

            auto info = vm.memory->get_host_pointer<AbortInfo>(Val(a.getAbortInfo()).cast<Pointer>());
            auto reason_str = [&] -> std::string_view {
                switch (a.getReason()) {
                    case ir::AbortReason::AssertionFailed: return "Assertion failed";
                    case ir::AbortReason::ArithmeticError: return "Arithmetic error";
                    case ir::AbortReason::InvalidLocalRef: return "Cannot access variable declared outside the current evaluation context";
                }
                Unreachable();
            }();

            SmallString<256> msg{reason_str};
            if (not info->msg1.sv().empty()) Format(msg, ": '{}'", info->msg1.sv());
            if (not info->msg2.sv().empty()) Format(msg, ": {}", info->msg2.sv());
            Error(SLoc::Decode(a.getLoc()), "{}", msg);

            // Attempt to run the stringifier if there is one.
            if (info->closure.ptr.is_null()) return false;

            // Give up immediately if we’re asserting recursively.
            if (assert_buffer.has_value()) {
                Warn(SLoc::Decode(a.getLoc()), "Assert stringifier failed constant evaluation");
                return false;
            }

            // Compile the procedure now if we haven’t done that yet.
            auto callee = vm.memory->get_procedure(info->closure.ptr);
            Assert(callee, "Address is not callable");
            if (callee.empty()) {
                auto decl = cg.lookup(callee);
                Assert(decl);
                cg.emit(decl.get());

                // TODO: This can fail if one of our lifetime analyses reports an error.
                if (not cg.finalise(callee)) Todo("Handle finalisation failure in evaluator");
            }

            SmallVector<SRValue> args;
            args.emplace_back(vm.memory->make_host_pointer(uptr(&assert_buffer.emplace())));
            args.emplace_back(info->closure.env);
            PushFrame(a.getLoc(), callee, args, {});
            call_stack.back().is_assert_stringifier_frame = true;
            continue;
        }

        if (auto a = dyn_cast<mlir::arith::ConstantOp>(i)) {
            Temp(a) = SRValue(cast<mlir::IntegerAttr>(a.getValue()).getValue());
            continue;
        }

        if (auto b = dyn_cast<mlir::cf::BranchOp>(i)) {
            BranchTo(b.getDest(), b.getDestOperands());
            continue;
        }

        if (auto b = dyn_cast<mlir::cf::CondBranchOp>(i)) {
            auto& cond = Val(b.getCondition());
            if (cond.cast<APInt>().getBoolValue()) BranchTo(b.getTrueDest(), b.getTrueDestOperands());
            else BranchTo(b.getFalseDest(), b.getFalseDestOperands());
            continue;
        }

        // This is currently only used for string data.
        if (auto a = dyn_cast<mlir::LLVM::AddressOfOp>(i)) {
            auto g = cast<mlir::LLVM::GlobalOp>(i->getParentOfType<mlir::ModuleOp>().lookupSymbol(a.getGlobalName()));
            Assert(g, "Invalid reference to global");
            Assert(g.getConstant(), "TODO: Global variables in the constant evaluator");
            auto v = cast<mlir::StringAttr>(g.getValue().value());
            Temp(a) = SRValue(vm.memory->make_host_pointer(uptr(v.data())));
            continue;
        }

        if (auto c = dyn_cast<ir::CallOp>(i)) {
            auto ptr = Val(c.getAddr()).cast<Pointer>();
            if (not ptr) return Error(SLoc::Decode(i->getLoc()), "Attempted to call nil");

            // Check that this is a valid call target.
            Assert(vm.memory->is_virtual_proc_ptr(ptr), "TODO: indirect host pointer call");
            auto callee = vm.memory->get_procedure(ptr);
            if (not callee) return Error(SLoc::Decode(i->getLoc()), "Address is not callable");

            // Compile the procedure now if we haven’t done that yet.
            if (callee.empty()) {
                // This is an external procedure.
                auto decl = cg.lookup(callee);
                if (not decl or not decl.get()->body()) {
                    auto res = FFICall(callee, c);
                    if (not res) return false;
                    if (i->getNumResults() != 0) {
                        Assert(i->getNumResults() == 1);
                        Temp(i->getResult(0)) = std::move(res.value());
                    }
                    continue;
                }

                // This is a procedure that hasn’t been compiled yet.
                cg.emit(decl.get());
                if (not cg.finalise(callee)) return false;
            }

            // Get the return value slots *before* pushing a new frame.
            StackFrame::RetVals ret_vals;
            for (auto res : c.getResults()) ret_vals.push_back(&Temp(res));

            // Enter the stack frame.
            SmallVector<SRValue, 6> args;
            for (auto a : c.getArgs()) args.push_back(Val(a));
            PushFrame(c.getLoc(), callee, args, std::move(ret_vals));
            continue;
        }

        // These are only used by lifetime analysis and don’t have any
        // runtime semantics.
        if (isa<ir::DisengageOp, ir::EngageOp, ir::EngageCopyOp, ir::UnwrapOp>(i))
            continue;

        if (auto l = dyn_cast<ir::LoadOp>(i)) {
            auto ptr = GetHostMemoryPointer(l.getAddr());
            if (not ptr) return false;
            Temp(l) = LoadSRValue(ptr, l.getType());
            continue;
        }

        if (auto m = dyn_cast<mlir::LLVM::MemcpyOp>(i)) {
            auto dest = GetHostMemoryPointer(m.getDst());
            auto src = GetHostMemoryPointer(m.getSrc());
            if (not dest or not src) return false;
            auto bytes = Val(m.getLen()).cast<APInt>().getZExtValue();
            std::memcpy(dest, src, bytes);
            continue;
        }

        if (auto m = dyn_cast<mlir::LLVM::MemsetOp>(i)) {
            auto dest = GetHostMemoryPointer(m.getDst());
            if (not dest) return false;
            auto val = Val(m.getVal()).cast<APInt>().getZExtValue();
            auto bytes = Val(m.getLen()).cast<APInt>().getZExtValue();
            std::memset(dest, u8(val), bytes);
            continue;
        }

        if (isa<ir::NilOp>(i)) {
            Temp(i->getResult(0)) = SRValue(Pointer::Null());
            continue;
        }

        if (auto tree = dyn_cast<ir::QuoteOp>(i)) {
            SmallVector<TreeValue*> unquotes;
            auto quote = cast<QuoteExpr>(tree.getTree());
            for (auto u : tree.getUnquotes()) unquotes.push_back(Val(u).cast<TreeValue*>());
            Temp(i->getResult(0)) = SRValue(vm.allocate_tree_value(quote, unquotes));
            continue;
        }

        if (auto tree = dyn_cast<ir::TreeConstantOp>(i)) {
            Temp(i->getResult(0)) = SRValue(tree.getTree());
            continue;
        }

        if (auto ty = dyn_cast<ir::TypeConstantOp>(i)) {
            Temp(i->getResult(0)) = SRValue(ty.getValue());
            continue;
        }

        if (auto te = dyn_cast<ir::TypeEqOp>(i)) {
            bool equal = Val(te.getLhs()).cast<Type>() == Val(te.getRhs()).cast<Type>();
            Temp(i->getResult(0)) = SRValue(equal);
            continue;
        }

        if (auto prop = dyn_cast<ir::TypePropertyOp>(i)) {
            auto ty = Val(prop.getTypeArgument()).cast<Type>();
            auto InvalidQuery = [&] {
                return Error(
                    SLoc::Decode(i->getLoc()),
                    "Type '{}' has no member '%5({}%)'",
                    ty,
                    prop.getProperty()
                );
            };

            switch (prop.getProperty()) {
                using enum BuiltinMemberAccessExpr::AccessKind;
                case TypeAlign:
                    Temp(i->getResult(0)) = SRValue(i64(ty->align(vm.owner()).value().bytes()));
                    continue;

                case TypeArraySize:
                    Temp(i->getResult(0)) = SRValue(i64(ty->array_size(vm.owner()).bytes()));
                    continue;

                case TypeBits:
                    Temp(i->getResult(0)) = SRValue(i64(ty->bit_width(vm.owner()).bits()));
                    continue;

                case TypeName: {
                    auto s = vm.owner().save(StripColours(ty->print()));
                    Temp(i->getResult(0)) = SRValue(vm.memory->make_host_pointer(uptr(s.data())));
                    Temp(i->getResult(1)) = SRValue(i64(s.size()));
                    continue;
                }

                case TypeBytes:
                case TypeSize:
                    Temp(i->getResult(0)) = SRValue(i64(ty->memory_size(vm.owner()).bytes()));
                    continue;

                // TODO: Trying to query 'min' or 'max' of a non-integer type should be an error in the
                // Evaluator; fortunately, 'type' values can only exist at compile time, so we can just
                // emit an error about this at evaluation time.
                case TypeMaxVal: {
                    if (not ty->is_integer()) return InvalidQuery();
                    Temp(i->getResult(0)) = SRValue(APInt::getSignedMaxValue(u32(ty->bit_width(vm.owner()).bits())));
                    continue;
                }

                case TypeMinVal: {
                    if (not ty->is_integer()) return InvalidQuery();
                    Temp(i->getResult(0)) = SRValue(APInt::getSignedMinValue(u32(ty->bit_width(vm.owner()).bits())));
                    continue;
                }

                case SliceData:
                case SliceSize:
                case RangeStart:
                case RangeEnd:
                    Unreachable("Not a type property: {}", prop.getProperty());
            }

            Unreachable();
        }

        if (auto gep = dyn_cast<mlir::LLVM::GEPOp>(i)) {
            auto idx = gep.getIndices()[0];
            uptr offs;
            if (auto lit = dyn_cast<mlir::IntegerAttr>(idx)) offs = lit.getValue().getZExtValue();
            else offs = Val(cast<Value>(idx)).cast<APInt>().getZExtValue();
            Temp(gep) = SRValue(Val(gep.getBase()).cast<Pointer>().offset(Size::Bytes(offs)));
            continue;
        }

        if (auto s = dyn_cast<ir::FrameSlotOp>(i)) {
            auto ptr = AllocateStackMemory(s.getLoc(), s.size(), s.align());
            if (not ptr) return false;
            frame().temporaries[Encode(s)] = SRValue(ptr.value());
            continue;
        }

        if (auto r = dyn_cast<ir::RetOp>(i)) {
            // If we just finished stringifying an assert message, print it now.
            if (call_stack.back().is_assert_stringifier_frame) {
                Note(
                    SLoc::Decode(call_stack.back().call_loc),
                    "Expression evaluated to '{}'",
                    std::string_view{assert_buffer->data, usz(assert_buffer->size)}
                );
                return false;
            }

            // Save the return values in the return slots.
            Assert(r.getVals().size() == frame().ret_vals.size());
            for (auto [v, slot] : zip(r.getVals(), frame().ret_vals)) *slot = Val(v);

            // Clean up local variables.
            stack_top = frame().stack_base;
            call_stack.pop_back();

            // If we’re returning from the last stack frame, we’re done.
            if (call_stack.empty()) return true;
            continue;
        }

        if (auto p = dyn_cast<ir::ProcRefOp>(i)) {
            Temp(p) = SRValue(vm.memory->make_proc_ptr(p.proc()));
            continue;
        }

        if (auto s = dyn_cast<mlir::arith::SelectOp>(i)) {
            auto& cond = Val(s.getCondition());
            Temp(s) = cond.cast<APInt>().getBoolValue() ? Val(s.getTrueValue()) : Val(s.getFalseValue());
            continue;
        }

        if (auto s = dyn_cast<ir::StoreOp>(i)) {
            auto ptr = GetHostMemoryPointer(s.getAddr());
            if (not ptr) return false;
            auto& val = Val(s.getValue());
            StoreSRValue(ptr, val);
            continue;
        }

        if (isa<mlir::LLVM::UnreachableOp>(i))
            return Error(SLoc::Decode(i->getLoc()), "Unreachable code reached");

        if (auto c = dyn_cast<mlir::arith::ExtSIOp>(i)) {
            TRY(CastOp(i, c.getType(), &APInt::sext));
            continue;
        }

        if (auto c = dyn_cast<mlir::arith::ExtUIOp>(i)) {
            TRY(CastOp(i, c.getType(), &APInt::zext));
            continue;
        }

        if (auto c = dyn_cast<mlir::arith::TruncIOp>(i)) {
            TRY(CastOp(i, c.getType(), &APInt::trunc));
            continue;
        }

        if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(i)) {
            switch (cmp.getPredicate()) {
                using enum mlir::arith::CmpIPredicate;
                CMP_OP(eq); CMP_OP(ne);
                CMP_OP(slt); CMP_OP(sle); CMP_OP(sgt); CMP_OP(sge);
                CMP_OP(ult); CMP_OP(ule); CMP_OP(ugt); CMP_OP(uge);
            }
            Unreachable();
        }

        if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(i)) {
            switch (cmp.getPredicate()) {
                using enum mlir::arith::CmpIPredicate;
                CMP_OP(eq); CMP_OP(ne);
                CMP_OP(slt); CMP_OP(sle); CMP_OP(sgt); CMP_OP(sge);
                CMP_OP(ult); CMP_OP(ule); CMP_OP(ugt); CMP_OP(uge);
            }
            Unreachable();
        }

        // We currently only emit a 'ne' for 'for' loops involving arrays.
        // FIXME: Introduce our own CMP instruction that supports both integers and pointers.
        if (auto cmp = dyn_cast<mlir::LLVM::ICmpOp>(i)) {
            Assert(cmp.getPredicate() == mlir::LLVM::ICmpPredicate::ne);
            auto lhs = Val(cmp.getLhs());
            auto rhs = Val(cmp.getRhs());
            Temp(cmp) = SRValue(lhs.cast<Pointer>() != rhs.cast<Pointer>());
            continue;
        }

        INT_OP(AndIOp,lhs & rhs);
        INT_OP(OrIOp, lhs | rhs);
        INT_OP(XOrIOp,lhs ^ rhs);
        INT_OP(AddIOp, lhs + rhs);
        INT_OP(ShRSIOp, lhs.ashr(rhs));
        INT_OP(MulIOp, lhs * rhs);
        INT_OP(ShRUIOp, lhs.lshr(rhs));
        INT_OP(DivSIOp, lhs.sdiv(rhs));
        INT_OP(ShLIOp, lhs.shl(rhs));
        INT_OP(RemSIOp, lhs.srem(rhs));
        INT_OP(SubIOp, lhs - rhs);
        INT_OP(DivUIOp, lhs.udiv(rhs));
        INT_OP(RemUIOp, lhs.urem(rhs));

        if (isa<ir::SAddOvOp>(i)) {
            TRY(OvOp(i, &APInt::sadd_ov));
            continue;
        }

        if (isa<ir::SMulOvOp>(i)) {
            TRY(OvOp(i, &APInt::smul_ov));
            continue;
        }

        if (isa<ir::SSubOvOp>(i)) {
            TRY(OvOp(i, &APInt::ssub_ov));
            continue;
        }


        return ICE(SLoc::Decode(i->getLoc()), "Unsupported op in constant evaluation: '{}'", i->getName().getStringRef());
    }

    Error(entry, "Exceeded maximum compile-time evaluation steps");
    Remark("You can increase the limit by passing '--eval-steps=N';\fthe current value is {}.", max_steps);
    return false;
}

auto Eval::FFICall(ir::ProcOp proc, ir::CallOp call) -> std::optional<SRValue> {
    if (not vm.supports_ffi_calls) {
        Error(SLoc::Decode(call.getLoc()), "Compile-time FFI calls are not supported when cross-compiling");
        Remark(
            "The target triple is set to '{}', but the host triple is '{}'",
            vm.owner_tu.target().triple().str(),
            llvm::sys::getDefaultTargetTriple()
        );
        return std::nullopt;
    }

    // Determine the return type.
    auto ty = proc.getFunctionType();
    Assert(ty.getNumResults() < 2, "FFI procedure has more than 1 result?");
    auto ffi_ret = ty.getNumResults() ? FFIType(ty.getResult(0)) : &ffi_type_void;
    if (not ffi_ret) return std::nullopt;

    // Collect the arguments.
    SmallVector<SRValue, 6> args;
    for (auto a : call.getArgs()) args.push_back(Val(a));

    // Collect the argument types.
    SmallVector<ffi_type*> arg_types;
    for (auto a : call->getOperandTypes()) {
        auto arg_ty = FFIType(a);
        if (not arg_ty) return std::nullopt;
        arg_types.push_back(arg_ty);
    }

    // Prepare the call.
    ffi_cif cif{};
    ffi_status status{};
    if (proc.getVariadic()) {
        status = ffi_prep_cif_var(
            &cif,
            FFI_DEFAULT_ABI,
            unsigned(proc.getFunctionType().getNumInputs()),
            unsigned(args.size()),
            ffi_ret,
            arg_types.data()
        );
    } else {
        status = ffi_prep_cif(
            &cif,
            FFI_DEFAULT_ABI,
            unsigned(args.size()),
            ffi_ret,
            arg_types.data()
        );
    }

    if (status != 0) {
        Error(entry, "Failed to prepare FFI call");
        return std::nullopt;
    }

    // Prepare space for the return value.
    Assert(
        ffi_ret->alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
        "TODO: Handle overaligned return types"
    );

    SmallVector<std::byte, 64> ret_storage;
    ret_storage.resize(ffi_ret->size);

    // Store the arguments to memory.
    SmallVector<void*> arg_values;
    llvm::BumpPtrAllocator alloc;
    for (auto [a, t] : zip(args, arg_types)) {
        auto mem = alloc.Allocate(t->size, t->alignment);
        StoreSRValue(mem, a);
        arg_values.push_back(mem);
    }

    // Obtain the procedure address.
    auto [it, not_found] = vm.native_symbols.try_emplace(proc, nullptr);
    if (not_found) {
        auto sym = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(std::string{proc.getSymName()});
        if (not sym) {
            Error(entry, "Failed to find symbol for FFI call to '{}'", proc.getSymName());
            return std::nullopt;
        }
        it->second = sym;
    }

    // Perform the call.
    ffi_call(
        &cif,
        reinterpret_cast<void (*)()>(it->second),
        ret_storage.data(),
        arg_values.data()
    );

    // Retrieve the return value.
    if (ty.getNumResults() == 0) return SRValue();
    return LoadSRValue(ret_storage.data(), ty.getResult(0));
}

auto Eval::FFIType(mlir::Type ty) -> ffi_type* {
    if (auto i = dyn_cast<mlir::IntegerType>(ty)) {
        switch (i.getWidth()) {
            case 1: return &ffi_type_uint8;
            case 8: return &ffi_type_sint8;
            case 16: return &ffi_type_sint16;
            case 32: return &ffi_type_sint32;
            case 64: return &ffi_type_sint64;
            default:
                ICE(entry, "Unsupported integer type in FFI call: 'i{}'", i.getWidth());
                return nullptr;
        }
    }

    if (isa<mlir::LLVM::LLVMPointerType>(ty))
        return &ffi_type_pointer;

    ICE(entry, "Unsupported type in FFI call: 'i{}'", ty);
    return nullptr;
}

auto Eval::GetHostMemoryPointer(Value v) -> void* {
    auto p = Val(v).cast<Pointer>();

    // This is the null pointer in some address space.
    if (p.is_null()) {
        Error(SLoc::Decode(v.getLoc()), "Attempted to dereference 'nil'");
        return nullptr;
    }

    // This is a procedure pointer.
    if (vm.memory->is_virtual_proc_ptr(p)) {
        Error(SLoc::Decode(v.getLoc()), "Invalid memory access");
        return nullptr;
    }

    return vm.memory->get_host_pointer(p);
}

auto Eval::LoadInt(const void* mem, Size width) -> SRValue {
    if (width == Size::Bits(1)) {
        uptr b{};
        std::memcpy(&b, mem, vm.owner().target().int_size(Size::Bits(1)).bytes());
        return SRValue(b != 0);
    }

    APInt v{unsigned(width.bits()), 0};
    auto available_size = vm.owner().target().int_size(width);
    auto size_to_load = Size::Bits(v.getBitWidth()).as_bytes();
    Assert(size_to_load <= available_size);
    llvm::LoadIntFromMemory(
        v,
        static_cast<const u8*>(mem),
        unsigned(size_to_load.bytes())
    );
    return SRValue(std::move(v));
}

auto Eval::LoadPointer(const void* mem) -> SRValue {
    // Note: We currently assert in VM::VM that std::uintptr_t can store a target pointer.
    uptr p{};
    std::memcpy(&p, mem, vm.owner().target().ptr_size().bytes());
    return SRValue(vm.memory->make_host_pointer(p));
}

auto Eval::LoadSRValue(const void* mem, mlir::Type ty) -> SRValue {
    if (auto i = dyn_cast<mlir::IntegerType>(ty))
        return LoadInt(mem, Size::Bits(i.getWidth()));
    if (isa<mlir::LLVM::LLVMPointerType>(ty))
        return LoadPointer(mem);
    if (isa<ir::TreeType>(ty)) return SRValue(LoadTree(mem));
    if (isa<ir::TypeType>(ty)) return SRValue(LoadType(mem));

    Unreachable("Cannot load value of type '{}'", ty);
}

auto Eval::LoadTree(const void* mem) -> TreeValue* {
    TreeValue* ptr{};
    std::memcpy(&ptr, mem, sizeof(TreeValue*));
    return ptr;
}


auto Eval::LoadType(const void* mem) -> Type {
    TypeBase* ptr{};
    std::memcpy(&ptr, mem, sizeof(TypeBase*));
    return Type(ptr);
}

auto Eval::Persist(const void* mem, SLoc loc, Type ty) -> RValue {
    auto PersistPointer = [&](const void* mem) -> EvaluatedPointer {
        auto ptr = LoadPointer(mem).cast<Pointer>();
        if (ptr.is_null()) return EvaluatedPointer();

        // Pointers to local variables can’t be persisted since we have no
        // way of knowing whether the variable was deallocated (and potentially
        // overwritten by a different stack frame) after the pointer was created.
        if (vm.memory->is_stack_pointer(ptr))
            return EvaluatedPointer::GetInvalidStack();

        ICE(loc, "Persisting this kind of pointer is not supported yet");
        return EvaluatedPointer::GetUnknown();
    };

    auto PersistRecord = [&](const void* mem, const RecordLayout& rl) -> Record {
        SmallVector<RValue*> values;
        values.reserve(rl.fields().size());
        for (auto f : rl.fields()) {
            auto ptr = mem + f->offset;
            values.push_back(vm.owner().save(Persist(ptr, loc, f->type)));
        }
        return Record{ArrayRef(values).copy(vm.owner().allocator())};
    };

    if (isa<PtrType>(ty)) return RValue(PersistPointer(mem), ty);
    if (ty->is_integer_or_bool()) return RValue(
        LoadInt(mem, ty->bit_width(vm.owner())).cast<APInt>(),
        ty
    );

    if (auto r = dyn_cast<RangeType>(ty)) return RValue(
        LoadRangeFromMemory(mem, vm.owner(), r),
        ty
    );

    if (isa<SliceType>(ty)) {
        auto pointer = PersistPointer(mem);
        auto size_offs = vm.owner_tu.SliceEquivalentTupleTy->layout().fields()[1]->offset;
        auto size = LoadInt(mem + size_offs, Type::IntTy->bit_width(vm.owner())).cast<APInt>();
        return RValue(Slice{pointer, std::move(size)}, ty);
    }

    if (isa<ProcType>(ty)) {
        // Handle procedure pointers here rather than in PersistPointer() since they
        // should really only occur inside closures.
        auto proc_ptr = [&] -> EvaluatedPointer {
            auto p = LoadPointer(mem).cast<Pointer>();
            if (not vm.memory->is_virtual_proc_ptr(p)) return EvaluatedPointer::GetUnknown();
            auto op = vm.memory->get_procedure(p);
            auto proc = cg.lookup(op);
            if (not proc) return EvaluatedPointer::GetUnknown();
            return proc.get();
        }();

        auto env_offs = vm.owner_tu.ClosureEquivalentTupleTy->layout().fields()[1]->offset;
        auto env_ptr = PersistPointer(mem + env_offs);
        return RValue(Closure{proc_ptr, env_ptr}, ty);
    }

    if (ty == Type::TreeTy) return RValue(LoadTree(mem));
    if (ty == Type::TypeTy) return RValue(LoadType(mem));

    // If this type does not contain pointers, just persist the entire
    // thing as a byte buffer.
    if (not ty->is_or_contains_pointer()) {
        Assert((isa<RecordType, ArrayType, OptionalType>(ty)));
        auto mrv = vm.allocate_memory_value(ty);
        std::memcpy(mrv.data(), mem, mrv.size().bytes());
        return RValue(mrv, ty);
    }

    // Otherwise, persist each field individually.
    if (auto r = dyn_cast<RecordType>(ty)) {
        Assert(r->is_complete());
        return RValue(PersistRecord(mem, r->layout()), r);
    }

    if (auto o = dyn_cast<OptionalType>(ty)) {
        if (o->has_transparent_layout()) return Persist(mem, loc, o->elem());
        return RValue(PersistRecord(mem, *o->get_equivalent_record_layout()), o);
    }

    Todo();
}

void Eval::PushFrame(
    mlir::Location call_loc,
    ir::ProcOp proc,
    MutableArrayRef<SRValue> args,
    StackFrame::RetVals ret_vals
) {
    Assert(not proc.empty());

    // We allow passing 1 extra argument if it is a null pointer; this is
    // used to simplify indirect calls: we unconditionally pass the env
    // pointer to an indirect call, irrespective of whether the callee
    // actually expects one (which we don’t know because it’s an indirect
    // call).
    if (
        args.size() == proc.getNumCallArgs() + 1 and
        not proc.getVariadic() and
        args.back().isa<Pointer>() and
        args.back().cast<Pointer>().is_null()
    ) args = args.drop_back(1);

    // Set up the stack frame.
    Assert(args.size() == proc.getNumCallArgs());
    Assert(ret_vals.size() == proc.getFunctionType().getNumResults());
    StackFrame frame{call_loc, proc};
    frame.stack_base = stack_top;

    // Allocate temporaries for instructions and block arguments.
    for (auto& b : proc.getBody()) {
        for (auto a : b.getArguments())
            frame.temporaries[Encode(a)] = {};
        for (auto& i : b)
            for (auto r : i.getResults())
                frame.temporaries[Encode(r)] = {};
    }

    // Set the return value slots.
    frame.ret_vals = std::move(ret_vals);

    // Now that we’ve set up the frame, add it to the stack; we need
    // to do this *after* we initialise the call arguments above.
    call_stack.push_back(std::move(frame));

    // Branch to the entry block.
    BranchTo(&proc.front(), args);
}

void Eval::StoreSRValue(void* ptr, const SRValue& val) {
    val.visit(utils::Overloaded{
        [](std::monostate) { Unreachable("Store of empty value?"); },
        [&](const APInt& i) {
            auto available_size = vm.owner().target().int_size(Size::Bits(i.getBitWidth()));
            auto size_to_store = Size::Bits(i.getBitWidth()).as_bytes();
            Assert(size_to_store <= available_size);
            llvm::StoreIntToMemory(
                i,
                static_cast<u8*>(ptr),
                unsigned(size_to_store.bytes())
            );
        },
        [&](Pointer p) {
            uptr v = p.raw_value();
            std::memcpy(ptr, &v, vm.owner().target().ptr_size().bytes());
        },
        [&](Type t) {
            TypeBase* p = t.ptr();
            std::memcpy(ptr, &p, sizeof(TypeBase*));
        },
        [&](TreeValue* t) {
            std::memcpy(ptr, &t, sizeof(TreeValue*));
        }
    });
}

auto Eval::Temp(Value v) -> SRValue& {
    return const_cast<SRValue&>(Val(v));
}

auto Eval::Val(Value v) -> const SRValue& {
    return frame().temporaries.at(Encode(v));
}

auto Eval::eval(Stmt* s) -> std::optional<RValue> {
    // Always reset the stack pointer when we’re done.
    tempset stack_top = stack_top;

    // Compile the procedure.
    entry = s->location();
    auto proc = cg.emit_stmt_as_proc_for_vm(s);
    if (not proc) return std::nullopt;
    Assert(proc.getNumCallArgs() <= 1);
    Assert(proc.getNumResults() == 0, "Eval procedure should return indirectly");
    auto yields_value = proc.getNumCallArgs() == 1;

    // Allocate stack memory for the return value.
    SmallVector<SRValue> args;
    Pointer ret;
    auto ty = s->type_or_void();
    if (yields_value) {
        auto mem = AllocateStackMemory(proc->getLoc(), ty->memory_size(vm.owner()), ty->align(vm.owner()));

        // FIXME: If the return type is really big, we may want to put it on the
        // heap instead instead of failing evaluation—even if we *could* allocate
        // it on the stack, we might not want to to avoid overflow.
        if (not mem) return std::nullopt;
        ret = mem.value();
        args.push_back(SRValue(ret));
    }

    // Set up a stack frame for it.
    PushFrame(proc->getLoc(), proc, args, {});

    // Otherwise, just run the procedure and convert the results to an rvalue.
    TRY(EvalLoop());
    if (not yields_value) return RValue();
    Assert(ret);
    auto mem = reinterpret_cast<const void*>(ret.raw_value());
    return Persist(mem, s->location(), ty);
}

// ============================================================================
//  VM API
// ============================================================================
VM::~VM() = default;
VM::VM(TranslationUnit& owner_tu)
    : owner_tu{owner_tu},
      memory(std::make_unique<VirtualMemoryMap>()) {}

auto VM::allocate_memory_value(Type ty) -> RawByteBuffer {
    auto sz = ty->memory_size(owner());
    auto align = ty->align(owner());
    auto mem = owner().allocate(sz.bytes(), align.value().bytes());
    std::memset(mem, 0, sz.bytes());
    return RawByteBuffer(mem, sz, ty == owner().StrLitTy);
}

auto VM::allocate_tree_value(QuoteExpr* quote, ArrayRef<TreeValue*> unquotes) -> TreeValue* {
    auto size = TreeValue::totalSizeToAlloc<TreeValue*>(unquotes.size());
    auto mem = owner().allocate(size, alignof(TreeValue));
    return ::new (mem) TreeValue(quote, unquotes);
}

auto VM::eval(
    Stmt* stmt,
    bool complain,
    bool dump_ir
) -> std::optional<RValue> { // clang-format off
    using OptVal = std::optional<RValue>;
    Assert(initialised);

    // Fast paths for common values.
    if (auto e = dyn_cast<Expr>(stmt)) {
        auto val = e->visit(utils::Overloaded{
            [](auto*) -> OptVal { return std::nullopt; },
            [](IntLitExpr* i) -> OptVal { return RValue{i->storage.value(), i->type}; },
            [](BoolLitExpr* b) -> OptVal { return RValue(b->value); },
            [](TreeValue* t) -> OptVal { return RValue{t}; },
            [](TypeExpr* t) -> OptVal { return RValue{t->value}; },
            [](ConstExpr* t) -> OptVal { return *t->value; },
        });

        // If we got a value, just return it.
        if (val.has_value()) {
            if (dump_ir) std::println("(No IR emitted for this code)");
            return val;
        }
    }

    // Otherwise, we need to do this the complicated way. Evaluate the statement.
    //
    // TODO: I think it’s possible to trigger this, actually, if an evaluation
    // performs a template instantiation that contains an 'eval' statement.
    Assert(not evaluating, "We somehow triggered a nested evaluation?");
    tempset evaluating = true;
    Eval e{*this, complain};
    auto res = e.eval(stmt);
    if (dump_ir) std::println("{}", text::RenderColours(
        owner_tu.context().use_colours,
        e.cg.dump().str())
    );
    return res;
} // clang-format on

void VM::init(const Target& tgt) {
    Assert(not initialised);
    initialised = true;
    supports_ffi_calls = tgt.triple() == llvm::Triple(llvm::sys::getDefaultTargetTriple());

    // We currently assume that a host std::uintptr_t can store a target pointer value.
    Assert(
        tgt.ptr_size() <= Size::Of<uptr>(),
        "Cross-compiling to an architecture whose pointer size is larger is not supported"
    );

    // If 'bool' doesn’t fit in 'uptr', then there’s something seriously wrong with our target.
    Assert(
        tgt.int_size(Size::Bits(1)) <= Size::Of<uptr>(),
        "What kind of unholy abomination is this target???"
    );
}
