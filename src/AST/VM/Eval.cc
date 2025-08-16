#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/MLIRFormatters.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/Allocator.h>
#include <llvm/TargetParser/Host.h>

#include <base/Formatters.hh>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <ffi.h>
#include <optional>
#include <print>
#include <ranges>

using namespace srcc;
using namespace srcc::eval;
namespace ir = cg::ir;
using mlir::Block;
using mlir::Value;

auto RValue::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{
        // clang-format off
        [&](bool) { out += std::format("%1({}%)", value.get<bool>()); },
        [&](std::monostate) {},
        [&](Type ty) { out += ty->print(); },
        [&](MRValue) { out += "<aggregate value>"; },
        [&](const APInt& value) { out += std::format("%5({}%)", toString(value, 10, true)); },
        [&](this auto& self, const Range& range) {
            self(range.start);
            out += "%1(..%)";
            self(range.end);
        },
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

    /// Check if this is the null pointer.
    [[nodiscard]] bool is_null() const { return value == 0; }

    /// Offset this pointer.
    [[nodiscard]] auto offset(usz bytes) const { return Pointer(value + bytes); }

    /// Get a string representation of this pointer.
    [[nodiscard]] auto str() const -> std::string {
        return std::format("{}", reinterpret_cast<void*>(value));
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
    Variant<APInt, Type, Pointer, std::monostate> value{std::monostate{}};

public:
    SRValue() = default;
    explicit SRValue(std::same_as<bool> auto b) : value{APInt{1, u64(b)}} {}
    explicit SRValue(Type ty) : value{ty} {}
    explicit SRValue(Pointer p) : value{p} {}
    explicit SRValue(APInt val) : value(std::move(val)) {}
    explicit SRValue(std::same_as<i64> auto val) : value{APInt{64, u64(val)}} {}

    [[nodiscard]] bool operator==(const SRValue& other) const;

    /// cast<>() the contained value.
    template <typename Ty>
    [[nodiscard]] auto cast() -> Ty& { return std::get<Ty>(value); }

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
    [[nodiscard]] auto print() const -> SmallUnrenderedString;

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

auto SRValue::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{
        // clang-format off
        [&](std::monostate) {},
        [&](ir::ProcOp proc) { out += std::format("%2({}%)", proc.getName()); },
        [&](Type ty) { out += ty->print(); },
        [&](const APInt& value) { out += std::format("%5({}%)", toString(value, 10, true)); },
        [&](Pointer ptr) { out += std::format("%4({}%)", ptr.str()); }
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

public:
    VirtualMemoryMap() = default;

    /// Map a pointer to the procedure it references.
    [[nodiscard]] auto get_procedure(Pointer p) -> ir::ProcOp;

    /// Map a pointer to a pointer to host memory.
    template <typename T = void>
    [[nodiscard]] auto get_host_pointer(Pointer p) -> T*;

    /// Check if a pointer is a host memory pointer.
    [[nodiscard]] bool is_host_pointer(Pointer p);

    /// Check if a pointer is a virtual procedure pointer.
    [[nodiscard]] bool is_virtual_proc_ptr(Pointer p);

    /// Create a pointer to host memory.
    [[nodiscard]] auto make_host_pointer(uptr v) -> Pointer;

    /// Add a procedure to the table if it isn't already registered and return a VM
    /// pointer to it.
    [[nodiscard]] auto make_proc_ptr(ir::ProcOp proc) -> Pointer;

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

bool VirtualMemoryMap::is_host_pointer(Pointer p) {
    return not is_virtual_proc_ptr(p);
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
class eval::Eval : DiagsProducer<bool> {
    friend DiagsProducer;

    /// A procedure on the stack.
    struct StackFrame {
        using RetVals = SmallVector<SRValue*, 2>;

        /// Procedure to which this frame belongs.
        ir::ProcOp proc{};

        /// Instruction pointer for this procedure.
        Block::iterator ip{};

        /// Temporary values for each instruction.
        DenseMap<Temporary, SRValue> temporaries{};

        /// Other materialised temporaries (literal integers etc.)
        StableVector<SRValue> materialised_values{};

        /// Stack size at the start of this procedure.
        std::byte* stack_base{};

        /// Return value slots.
        RetVals ret_vals{};

        /// Address used for indirect returns.
        SRValue ret_ptr{};

        /// Environment pointer.
        SRValue env_ptr{};
    };

    VM& vm;
    cg::CodeGen cg;
    SmallVector<StackFrame, 4> call_stack;
    const SRValue true_val{true};
    const SRValue false_val{false};
    std::byte* stack_top{};
    Location entry;
    bool complain;

public:
    Eval(VM& vm, bool complain);

    [[nodiscard]] auto eval(Stmt* s) -> std::optional<RValue>;

private:
    auto diags() const -> DiagnosticsEngine& { return vm.owner().context().diags(); }
    auto frame() -> StackFrame& { return call_stack.back(); }

    template <typename... Args>
    bool Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
        return false;
    }

    [[nodiscard]] auto AdjustLangOpts(LangOpts l) -> LangOpts;
    [[nodiscard]] auto AllocateStackMemory(mlir::Location loc, Size sz, Align alignment) -> std::optional<Pointer>;
    [[nodiscard]] bool EvalLoop();
    [[nodiscard]] auto FFICall(ir::ProcOp proc, ir::CallOp call) -> std::optional<SRValue>;
    [[nodiscard]] auto FFIType(mlir::Type ty) -> ffi_type*;
    [[nodiscard]] auto GetHostMemoryPointer(Value v) -> void*;
    [[nodiscard]] auto LoadSRValue(const void* mem, mlir::Type ty) -> SRValue;
    [[nodiscard]] auto Temp(Value v) -> SRValue&;
    [[nodiscard]] auto Val(Value v) -> const SRValue&;

    void BranchTo(Block* block, mlir::ValueRange args);
    void BranchTo(Block* block, MutableArrayRef<SRValue> args);
    void PushFrame(
        ir::ProcOp proc,
        MutableArrayRef<SRValue> args,
        StackFrame::RetVals ret_vals,
        SRValue ret_ptr = {},
        SRValue env_ptr = {}
    );

    void StoreSRValue(void* mem, const SRValue& val);
};

Eval::Eval(VM& vm, bool complain)
    : vm{vm},
      cg{vm.owner(), AdjustLangOpts(vm.owner().lang_opts()), vm.owner().target().ptr_size()},
      stack_top{vm.stack.get()},
      complain{complain} {}

auto Eval::AdjustLangOpts(LangOpts l) -> LangOpts {
    l.constant_eval = true;
    l.overflow_checking = true;
    return l;
}

auto Eval::AllocateStackMemory(mlir::Location loc, Size sz, Align alignment) -> std::optional<Pointer> {
    auto ptr = alignment.align(stack_top);
    stack_top = ptr + sz;
    if (stack_top > vm.stack.get() + vm.max_stack_size) {
        Error(Location::Decode(loc), "Stack overflow");
        Remark(
            "This may have been caused by infinite recursion. If you don’t think that "
            "that’s the case, you can increase the maximum eval stack size by passing "
            "--feval-stack-size (current value: {:y})",
            vm.max_stack_size
        );
        return std::nullopt;
    }
    return vm.memory->make_host_pointer(uptr(ptr));
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

    const u64 max_steps = vm.owner().context().eval_steps ?: std::numeric_limits<u64>::max();
    for (u64 steps = 0; steps < max_steps; steps++) {
        auto i = &*frame().ip++;
        if (auto a = dyn_cast<ir::AbortOp>(i)) {
            if (not complain) return false;
            struct AbortInfo {
                struct Slice {
                    const char* data;
                    isz size;
                    auto sv() const -> std::string_view { return {data, usz(size)}; }
                };

                Slice filename;
                isz line;
                isz col;
                Slice msg1;
                Slice msg2;
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

            std::string msg{reason_str};
            if (not info->msg1.sv().empty()) msg += std::format(": '{}'", info->msg1.sv());
            if (not info->msg2.sv().empty()) msg += std::format(": {}", info->msg2.sv());
            return Error(Location::Decode(a.getLoc()), "{}", msg);
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
            if (not ptr) return Error(Location::Decode(i->getLoc()), "Attempted to call nil");

            // Check that this is a valid call target.
            Assert(vm.memory->is_virtual_proc_ptr(ptr), "TODO: indirect host pointer call");
            auto callee = vm.memory->get_procedure(ptr);
            if (not callee) return Error(Location::Decode(i->getLoc()), "Address is not callable");

            // Compile the procedure now if we haven’t done that yet.
            if (callee.empty()) {
                // This is an external procedure.
                auto decl = cg.lookup(callee);
                if (not decl or not decl.get()->body()) {
                    auto res = FFICall(callee, c);
                    if (not res) return false;
                    Temp(i->getResult(0)) = std::move(res.value());
                    continue;
                }

                // This is a procedure that hasn’t been compiled yet.
                cg.emit(decl.get());
            }

            // Get the return value slots *before* pushing a new frame.
            StackFrame::RetVals ret_vals;
            for (auto res : c.getResults()) ret_vals.push_back(&Temp(res));

            // Enter the stack frame.
            SmallVector<SRValue, 6> args;
            for (auto a : c.getArgs()) args.push_back(Val(a));
            SRValue ret_ptr = c.getMrvalueSlot() ? Val(c.getMrvalueSlot()) : SRValue();
            SRValue env_ptr = c.getEnv() ? Val(c.getEnv()) : SRValue();
            PushFrame(callee, args, std::move(ret_vals), ret_ptr, env_ptr);
            continue;
        }

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

        if (auto gep = dyn_cast<mlir::LLVM::GEPOp>(i)) {
            auto idx = gep.getIndices()[0];
            uptr offs;
            if (auto lit = dyn_cast<mlir::IntegerAttr>(idx)) offs = lit.getValue().getZExtValue();
            else offs = Val(cast<Value>(idx)).cast<APInt>().getZExtValue();
            Temp(gep) = SRValue(Val(gep.getBase()).cast<Pointer>().offset(offs));
            continue;
        }

        if (auto s = dyn_cast<ir::FrameSlotOp>(i)) {
            auto ptr = AllocateStackMemory(s.getLoc(), s.size(), s.align());
            if (not ptr) return false;
            frame().temporaries[Encode(s)] = SRValue(ptr.value());
            continue;
        }

        if (auto r = dyn_cast<ir::RetOp>(i)) {
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
            return Error(Location::Decode(i->getLoc()), "Unreachable code reached");

        if (isa<ir::ReturnPointerOp>(i)) {
            Temp(i->getResult(0)) = frame().ret_ptr;
            continue;
        }

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

        return ICE(Location::Decode(i->getLoc()), "Unsupported op in constant evaluation: '{}'", i->getName().getStringRef());
    }

    Error(entry, "Exceeded maximum compile-time evaluation steps");
    Remark("You can increase the limit by passing '--eval-steps=N';\fthe current value is {}.", max_steps);
    return false;
}

auto Eval::FFICall(ir::ProcOp proc, ir::CallOp call) -> std::optional<SRValue> {
    if (not vm.supports_ffi_calls) {
        Error(Location::Decode(call.getLoc()), "Compile-time FFI calls are not supported when cross-compiling");
        Remark(
            "The target triple is set to '{}', but the host triple is '{}'",
            vm.owner_tu.target().triple().str(),
            llvm::sys::getDefaultTargetTriple()
        );
        return std::nullopt;
    }

    if (call.getMrvalueSlot()) {
        ICE(
            Location::Decode(call.getLoc()),
            "Compile-time FFI calls returning a structure in memory are currently not supported"
        );
        return std::nullopt;
    }

    if (call.getEnv()) {
        ICE(
            Location::Decode(call.getLoc()),
            "Compile-time calls to compiled closures are currently not supported"
        );
        return std::nullopt;
    }

    // Determine the return type.
    Assert(proc.getNumResults() < 2, "FFI procedure has more than 1 result?");
    auto ffi_ret = proc->getNumResults() ? FFIType(proc.getResultTypes()[0]) : &ffi_type_void;
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
    if (proc.getResultTypes().empty()) return SRValue();
    return LoadSRValue(ret_storage.data(), proc.getResultTypes()[0]);
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
        Error(Location::Decode(v.getLoc()), "Attempted to dereference 'nil'");
        return nullptr;
    }

    // This is a procedure pointer.
    if (vm.memory->is_virtual_proc_ptr(p)) {
        Error(Location::Decode(v.getLoc()), "Invalid memory access");
        return nullptr;
    }

    return vm.memory->get_host_pointer(p);
}

auto Eval::LoadSRValue(const void* mem, mlir::Type ty) -> SRValue {
    if (auto i = dyn_cast<mlir::IntegerType>(ty)) {
        if (i.getWidth() == 1) {
            uptr b{};
            std::memcpy(&b, mem, vm.owner().target().int_size(Size::Bits(1)).bytes());
            return SRValue(b != 0);
        }

        APInt v{i.getWidth(), 0};
        auto available_size = vm.owner().target().int_size(Size::Bits(i.getWidth()));
        auto size_to_load = Size::Bits(v.getBitWidth()).as_bytes();
        Assert(size_to_load <= available_size);
        llvm::LoadIntFromMemory(
            v,
            static_cast<const u8*>(mem),
            unsigned(size_to_load.bytes())
        );
        return SRValue(std::move(v));
    }

    if (isa<mlir::LLVM::LLVMPointerType>(ty)) {
        // Note: We currently assert in VM::VM that std::uintptr_t can store a target pointer.
        uptr p{};
        std::memcpy(&p, mem, vm.owner().target().ptr_size().bytes());
        return SRValue(vm.memory->make_host_pointer(p));
    }

    Unreachable("Cannot load value of type '{}'", ty);
}

void Eval::PushFrame(
    ir::ProcOp proc,
    MutableArrayRef<SRValue> args,
    StackFrame::RetVals ret_vals,
    SRValue ret_ptr,
    SRValue env_ptr
) {
    Assert(not proc.empty());
    StackFrame frame{proc};
    frame.stack_base = stack_top;
    frame.ret_ptr = ret_ptr;
    frame.env_ptr = env_ptr;

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
            std::memcpy(ptr, &p, vm.owner().target().ptr_size().bytes());
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
    // Compile the procedure.
    entry = s->location();
    auto proc = cg.emit_stmt_as_proc_for_vm(s);

    // Set up a stack frame for it.
    SmallVector<SRValue> ret(proc.getNumResults());
    StackFrame::RetVals ret_val_pointers;
    for (auto& v : ret) ret_val_pointers.push_back(&v);
    PushFrame(proc, {}, std::move(ret_val_pointers));

    // If statement returns an mrvalue, allocate one and set it
    // as the first argument to the call.
    auto ty = s->type_or_void();
    if (ty->rvalue_category() == Expr::MRValue) {
        auto mrv = vm.allocate_mrvalue(ty);
        frame().ret_ptr = SRValue(vm.memory->make_host_pointer(uptr(mrv.data())));
        TRY(EvalLoop());
        return RValue(mrv, ty);
    }

    // Otherwise, just run the procedure and convert the results to an rvalue.
    TRY(EvalLoop());

    // The procedure may have 2 results if it’s a range.
    if (isa<RangeType>(ty)) {
        Assert(proc.getFunctionType().getNumResults() == 2);
        return RValue(
            RValue::Range(
                std::move(ret[0].cast<APInt>()),
                std::move(ret[1].cast<APInt>())
            ),
            ty
        );
    }

    if (isa<IntType>(ty) or ty == Type::IntTy) {
        Assert(proc.getFunctionType().getNumResults() == 1);
        return RValue(std::move(ret[0].cast<APInt>()), ty);
    }

    if (ty == Type::BoolTy) {
        Assert(proc.getFunctionType().getNumResults() == 1);
        return RValue(ret[0].cast<APInt>().getBoolValue());
    }

    if (ty == Type::TypeTy) {
        Assert(proc.getFunctionType().getNumResults() == 1);
        return RValue(
            vm.memory->get_host_pointer<TypeBase>(ret[0].cast<Pointer>())
        );
    }

    if (ty == Type::VoidTy) {
        Assert(proc.getFunctionType().getNumResults() == 0);
        return RValue();
    }

    ICE(entry, "Don’t know how to materialise an RValue for this type: '{}'", ty);
    return std::nullopt;
}

// ============================================================================
//  VM API
// ============================================================================
VM::~VM() = default;
VM::VM(TranslationUnit& owner_tu)
    : owner_tu{owner_tu},
      memory(std::make_unique<VirtualMemoryMap>()) {}

auto VM::allocate_mrvalue(Type ty) -> MRValue {
    auto sz = ty->size(owner());
    auto align = ty->align(owner());
    auto mem = owner().allocate(sz.bytes(), align.value().bytes());
    std::memset(mem, 0, sz.bytes());
    return MRValue(mem, sz);
}

auto VM::eval(
    Stmt* stmt,
    bool complain
) -> std::optional<RValue> { // clang-format off
    using OptVal = std::optional<RValue>;
    Assert(initialised);

    // Fast paths for common values.
    if (auto e = dyn_cast<Expr>(stmt)) {
        auto val = e->visit(utils::Overloaded{
            [](auto*) -> OptVal { return std::nullopt; },
            [](IntLitExpr* i) -> OptVal { return RValue{i->storage.value(), i->type}; },
            [](BoolLitExpr* b) -> OptVal { return RValue(b->value); },
            [](TypeExpr* t) -> OptVal { return RValue{t->value}; },
        });

        // If we got a value, just return it.
        if (val.has_value()) return val;
    }

    // Otherwise, we need to do this the complicated way. Evaluate the statement.
    //
    // TODO: I think it’s possible to trigger this, actually, if an evaluation
    // performs a template instantiation that contains an 'eval' statement.
    Assert(not evaluating, "We somehow triggered a nested evaluation?");
    tempset evaluating = true;
    Eval e{*this, complain};
    return e.eval(stmt);
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
