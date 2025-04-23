#include <srcc/AST/AST.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Allocator.h>

#include <ffi.h>
#include <optional>
#include <print>
#include <ranges>

using namespace srcc;
using namespace srcc::eval;
namespace ir = cg::ir;

#define Val(x) (*({auto _v = ValImpl(x); if (not _v) return {}; _v.get(); }))
#define TRY(x) do { if (not (x)) return {}; } while (false)

// ============================================================================
//  Value
// ============================================================================
auto SRValue::Empty(TranslationUnit& tu, Type ty) -> SRValue {
    switch (ty->kind()) {
        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::StructType:
            Unreachable("Not an SRValue");

        case TypeBase::Kind::ProcType:
            Todo();

        case TypeBase::Kind::SliceType:
            return SRValue(SRSlice(), ty);

        case TypeBase::Kind::PtrType:
            return SRValue(Pointer(), ty);

        case TypeBase::Kind::IntType:
            return SRValue(APInt::getZero(u32(::cast<IntType>(ty)->bit_width().bits())), ty);

        case TypeBase::Kind::BuiltinType:
            switch (::cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::NoReturn:
                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::Deduced:
                    Unreachable();

                case BuiltinKind::Bool: return SRValue(false);
                case BuiltinKind::Int: return SRValue(APInt::getZero(u32(Type::IntTy->size(tu).bits())), ty);
                case BuiltinKind::Void: return SRValue();
                case BuiltinKind::Type: return SRValue(Type::VoidTy);
            }

            Unreachable();
    }

    Unreachable();
}

void SRValue::dump(bool use_colour) const {
    std::println("{}", text::RenderColours(use_colour, print().str()));
}

auto SRValue::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{// clang-format off
        [&](bool) { out += std::format("%1({}%)", value.get<bool>()); },
        [&](std::monostate) {},
        [&](ir::Proc* proc) { out += std::format("%2({}%)", proc->name()); },
        [&](Type ty) { out += ty->print(); },
        [&](const APInt& value) { out += std::format("%5({}%)", toString(value, 10, true)); },
        [&](Pointer ptr) { out += std::format("%4({}%)", reinterpret_cast<void*>(ptr.encode())); },
        [&](this auto& self, const SRSlice& slice) {
            out += "%1((";
            self(slice.data);
            out += ", ";
            self(slice.size);
            out += ")%)";
        },
        [&](this auto& self, const SRClosure& slice) {
            out += "%1((";
            self(slice.proc);
            out += ", ";
            self(slice.env);
            out += ")%)";
        },
    }; // clang-format on

    visit(V);
    return out;
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
auto Encode(ir::Argument* a) -> Temporary { return Temporary(u64(a)); }
auto Encode(ir::FrameSlot* f) -> Temporary { return Temporary(u64(f)); }
auto Encode(ir::Inst* i, u32 val) -> Temporary { return Temporary(u64(i) << 32 | val); }
} // namespace

// ============================================================================
//  Memory
// ============================================================================
namespace {
class HostMemoryMap {
    struct Mapping {
        /// The offset in the virtual host memory address space.
        Pointer vptr;

        /// The actual compiler memory that this maps to.
        const void* host_ptr;

        /// Size of the map.
        Size size;

        /// Create a new mapping.
        Mapping(Pointer vptr, const void* host_ptr, Size size)
            : vptr(vptr), host_ptr(host_ptr), size(size) {
            Assert(size.bytes() != 0);
            Assert(vptr.is_host_ptr());
        }

        /// Check if this mapping contains the given host memory range.
        [[nodiscard]] bool contains(const void* ptr, Size req) const {
            return ptr >= host_ptr and static_cast<const char*>(ptr) + req.bytes() <= end();
        }

        /// Check if this mapping contains the given virtual memory range.
        [[nodiscard]] bool contains(Pointer ptr, Size req) const {
            return ptr >= vptr and ptr + req.bytes() <= vend();
        }

        /// Get the end of the mapping (exclusive).
        [[nodiscard]] auto end() const -> const void* {
            return static_cast<const char*>(host_ptr) + size.bytes();
        }

        /// Get the end of the virtual mapping (exclusive).
        [[nodiscard]] auto vend() const -> Pointer {
            return vptr + size.bytes();
        }

        /// Translate a pointer in host memory to a virtual pointer.
        [[nodiscard]] auto map(const void* ptr) const -> Pointer {
            Assert(contains(ptr, Size{}));
            return vptr + usz(static_cast<const char*>(ptr) - static_cast<const char*>(host_ptr));
        }

        /// Translate a virtual pointer to a pointer in host memory.
        [[nodiscard]] auto map(Pointer ptr) const -> const void* {
            Assert(contains(ptr, Size{}));
            return static_cast<const char*>(host_ptr) + usz(ptr - vptr);
        }
    };

    /// Mapped memory ranges, sorted by host memory pointer.
    SmallVector<Mapping> maps;

    /// Total mapped memory.
    Size end;

public:
    /// Create a virtual mapping for the given memory range; if the range
    /// is already mapped, return the existing mapping.
    [[nodiscard]] auto create_map(const void* data, Size size, Align a = {}) -> Pointer;

    /// Map a virtual pointer to the corresponding host pointer.
    [[nodiscard]] auto lookup(Pointer ptr, Size size) -> const void* {
        Assert(ptr.is_host_ptr() and not ptr.is_null_ptr());
        auto it = find_if(maps, [&](const Mapping& m) { return m.contains(ptr, size); });
        if (it != maps.end()) return it->map(ptr);
        return nullptr;
    }
};
} // namespace

auto HostMemoryMap::create_map(const void* data, Size size, Align a) -> Pointer {
    auto it = find_if(maps, [&](const Mapping& m) { return m.contains(data, size); });
    if (it != maps.end()) return it->map(data);

    // Allocate virtual memory.
    end = end.align(a);
    auto start = end;
    end += size;

    // Store the mapping.
    return maps.emplace_back(Pointer::Host(start.bytes()), data, size).map(data);
}

// ============================================================================
//  Evaluator
// ============================================================================
class eval::Eval : DiagsProducer<bool> {
    friend DiagsProducer;

    /// A procedure on the stack.
    struct StackFrame {
        /// Procedure to which this frame belongs.
        ir::Proc* proc{};

        /// Instruction pointer for this procedure.
        ArrayRef<ir::Inst*>::iterator ip{};

        /// Temporary values for each instruction.
        DenseMap<Temporary, SRValue> temporaries{};

        /// Other materialised temporaries (literal integers etc.)
        StableVector<SRValue> materialised_values{};

        /// Stack size at the start of this procedure.
        usz stack_base{};

        /// Return value slot.
        SRValue* ret{};
    };

    VM& vm;
    cg::CodeGen cg;
    ByteBuffer stack;
    SmallVector<StackFrame, 4> call_stack;
    SmallVector<ir::Proc*> procedure_indices;
    HostMemoryMap host_memory;
    const SRValue true_val{true};
    const SRValue false_val{false};
    Location entry;
    bool complain;

public:
    Eval(VM& vm, bool complain);

    [[nodiscard]] auto eval(Stmt* s) -> std::optional<SRValue>;

private:
    auto diags() const -> DiagnosticsEngine& { return vm.owner().context().diags(); }

    template <typename... Args>
    bool Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
        return false;
    }

    [[nodiscard]] auto AdjustLangOpts(LangOpts l) -> LangOpts;
    [[nodiscard]] bool BranchTo(ir::Block* block, ArrayRef<ir::Value*> args);
    [[nodiscard]] auto Eq(const SRValue& a, const SRValue& b) -> std::optional<bool>;
    [[nodiscard]] bool EvalLoop();
    [[nodiscard]] auto FFICall(ir::Proc* proc, ArrayRef<ir::Value*> args) -> std::optional<SRValue>;
    [[nodiscard]] auto FFILoadRes(const void* mem, Type ty) -> std::optional<SRValue>;
    [[nodiscard]] auto FFIType(Type ty) -> ffi_type*;
    [[nodiscard]] bool FFIStoreArg(void* ptr, const SRValue& val);
    [[nodiscard]] auto GetMemoryPointer(Pointer ptr, Size accessible_size, bool readonly) -> void*;
    [[nodiscard]] auto GetStringData(const SRValue& val) -> std::string;
    [[nodiscard]] auto LoadSRValue(const SRValue& ptr, Type ty) -> std::optional<SRValue>;
    [[nodiscard]] auto LoadSRValue(const void* mem, Type ty) -> std::optional<SRValue>;
    [[nodiscard]] auto MakeProcPtr(ir::Proc* proc) -> Pointer;
    [[nodiscard]] bool PushFrame(ir::Proc* proc, ArrayRef<ir::Value*> args);
    [[nodiscard]] bool StoreSRValue(const SRValue& ptr, const SRValue& val);
    [[nodiscard]] bool StoreSRValue(void* ptr, const SRValue& val);
    [[nodiscard]] auto Temp(ir::Inst* i, u32 idx = 0) -> SRValue&;
    [[nodiscard]] auto Temp(ir::Argument* i) -> SRValue&;
    [[nodiscard]] auto Temp(ir::FrameSlot* f) -> SRValue&;
    [[nodiscard]] auto ValImpl(ir::Value* v) -> Ptr<const SRValue>;
};

Eval::Eval(VM& vm, bool complain)
    : vm{vm},
      cg{vm.owner(), AdjustLangOpts(vm.owner().lang_opts()), vm.owner().target().ptr_size()},
      complain{complain} {}

auto Eval::AdjustLangOpts(LangOpts l) -> LangOpts {
    l.constant_eval = true;
    l.overflow_checking = true;
    return l;
}

bool Eval::BranchTo(ir::Block* block, ArrayRef<ir::Value*> args) {
    call_stack.back().ip = block->instructions().begin();

    // Copy out the argument values our of their slots in case we’re doing
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

    // Now, copy in the values.
    for (auto [slot, arg] : zip(block->arguments(), copy))
        Temp(slot) = std::move(arg);

    return true;
}

auto Eval::Eq(const SRValue& a, const SRValue& b) -> std::optional<bool> {
    if (a.index() != b.index()) return false;

    // If this is not a slice, simply delegate to operator==.
    auto s1 = a.dyn_cast<SRSlice>();
    if (not s1) return a.visit([&]<typename T>(const T& t) { return t == b.cast<T>(); });

    // Otherwise, check if they’re the same size.
    auto s2 = &b.cast<SRSlice>();
    if (s1->size != s2->size) return false;

    // If so, get the memory pointers and compare them.
    auto sz = Size::Bytes(s1->size.getZExtValue());
    auto m1 = GetMemoryPointer(s1->data, sz, true);
    auto m2 = GetMemoryPointer(s2->data, sz, true);
    if (not m1 or not m2) return std::nullopt;
    return std::memcmp(m1, m2, sz.bytes()) == 0;
}

bool Eval::EvalLoop() {
    auto CastOp = [&] [[nodiscard]] (
        ir::ICast* i,
        APInt (APInt::*op)(unsigned width) const
    ) -> bool {
        auto& val = Val(i->args()[0]);
        auto to_wd = i->cast_result_type()->size(vm.owner());
        auto result = (val.cast<APInt>().*op)(u32(to_wd.bits()));
        Temp(i) = SRValue{std::move(result), i->cast_result_type()};
        return true;
    };

    auto CmpOp = [&] [[nodiscard]] (
        ir::Inst* i,
        llvm::function_ref<bool(const APInt& lhs, const APInt& rhs)> op
    ) -> bool {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        auto result = op(lhs.cast<APInt>(), rhs.cast<APInt>());
        Temp(i) = SRValue{result};
        return true;
    };

    auto IntOp = [&] [[nodiscard]] (
        ir::Inst* i,
        llvm::function_ref<APInt(const APInt& lhs, const APInt& rhs)> op
    ) -> bool {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        auto result = op(lhs.cast<APInt>(), rhs.cast<APInt>());
        Temp(i) = SRValue{std::move(result), lhs.type()};
        return true;
    };

    auto IntOrBoolOp = [&] [[nodiscard]] (ir::Inst* i, auto op) -> bool {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        if (lhs.isa<bool>()) Temp(i) = SRValue{bool(op(lhs.cast<bool>(), rhs.cast<bool>()))};
        else Temp(i) = SRValue{op(lhs.cast<APInt>(), rhs.cast<APInt>()), lhs.type()};
        return true;
    };

    auto OvOp = [&] [[nodiscard]] (
        ir::Inst* i,
        APInt (APInt::*op)(const APInt& RHS, bool& Overflow) const
    ) -> bool {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        bool overflow = false;
        auto result = (lhs.cast<APInt>().*op)(rhs.cast<APInt>(), overflow);
        Temp(i, 0) = SRValue{std::move(result), lhs.type()};
        Temp(i, 1) = SRValue{overflow};
        return true;
    };

    const u64 max_steps = vm.owner().context().eval_steps ?: std::numeric_limits<u64>::max();
    for (u64 steps = 0; steps < max_steps; steps++) {
        switch (auto i = *call_stack.back().ip++; i->opcode()) {
            case ir::Op::Abort: {
                if (not complain) return false;
                auto& a = *cast<ir::AbortInst>(i);
                auto& msg1 = Val(a[0]);
                auto& msg2 = Val(a[1]);
                auto reason_str = [&] -> std::string_view {
                    switch (a.abort_reason()) {
                        case ir::AbortReason::AssertionFailed: return "Assertion failed";
                        case ir::AbortReason::ArithmeticError: return "Arithmetic error";
                    }
                    Unreachable();
                }();

                std::string msg{reason_str};
                if (not msg1.empty()) msg += std::format(": '{}'", GetStringData(msg1));
                if (not msg2.empty()) msg += std::format(": {}", GetStringData(msg2));
                return Error(a.location(), "{}", msg);
            }

            case ir::Op::Br: {
                auto b = cast<ir::BranchInst>(i);
                if (b->is_conditional()) {
                    auto& cond = Val(b->cond());
                    if (cond.cast<bool>()) TRY(BranchTo(b->then(), b->then_args()));
                    else TRY(BranchTo(b->else_(), b->else_args()));
                } else {
                    TRY(BranchTo(b->then(), b->then_args()));
                }
            } break;

            case ir::Op::Call: {
                auto& closure = Val(i->args()[0]).cast<SRClosure>();
                auto args = i->args().drop_front(1);

                // Check that this is a valid call target.
                if (closure.proc.is_null_ptr() or not closure.proc.is_proc_ptr())
                    return Error(entry, "Attempted to call non-procedure");
                if (not closure.env.is_null_ptr())
                    return ICE(entry, "TODO: Call to procedure with environment");

                // Compile the procedure now if we haven’t done that yet.
                auto callee = procedure_indices[closure.proc.value()];
                if (callee->empty()) {
                    // This is an external procedure.
                    if (not callee->decl() or callee->decl()->is_imported()) {
                        auto res = FFICall(callee, args);
                        if (not res) return false;
                        Temp(i) = std::move(res.value());
                        break;
                    }

                    // This is not imported, so it must have a body.
                    Assert(callee->decl() and callee->decl()->body());
                    cg.emit(callee->decl());
                }

                // Get the temporary for the return value *before* pushing a new frame.
                SRValue* ret = nullptr;
                if (not i->result_types().empty()) ret = &Temp(i);

                // Enter the stack frame.
                TRY(PushFrame(callee, args));

                // Set the return value slot to the call’s temporary.
                call_stack.back().ret = ret;
            } break;

            case ir::Op::Load: {
                auto l = cast<ir::MemInst>(i);
                auto v = LoadSRValue(Val(l->ptr()), l->memory_type());
                if (not v) return false;
                Temp(l) = std::move(v.value());
            } break;

            case ir::Op::MemCopy: {
                auto& dest = Val(i->args()[0]);
                auto& src = Val(i->args()[1]);
                auto size = Size::Bytes(Val(i->args()[2]).cast<APInt>().getZExtValue());
                auto dest_ptr = GetMemoryPointer(dest.cast<Pointer>(), size, false);
                auto src_ptr = GetMemoryPointer(src.cast<Pointer>(), size, true);
                if (not dest_ptr or not src_ptr) return false;
                std::memcpy(dest_ptr, src_ptr, size.bytes());
            } break;

            case ir::Op::MemZero: {
                auto& addr = Val(i->args()[0]);
                auto size = Size::Bytes(Val(i->args()[1]).cast<APInt>().getZExtValue());
                auto ptr = GetMemoryPointer(addr.cast<Pointer>(), size, false);
                if (not ptr) return false;
                std::memset(ptr, 0, size.bytes());
            } break;

            case ir::Op::PtrAdd: {
                auto& ptr = Val(i->args()[0]);
                auto& offs = Val(i->args()[1]);
                Temp(i) = SRValue(ptr.cast<Pointer>() + offs.cast<APInt>().getZExtValue(), ptr.type());
            } break;

            case ir::Op::Ret: {
                // Save the return value in the return slot.
                if (not i->args().empty()) {
                    Assert(call_stack.back().ret, "Return value slot not set?");
                    *call_stack.back().ret = Val(i->args()[0]);
                }

                // Clean up local variables.
                stack.resize(call_stack.back().stack_base);
                call_stack.pop_back();

                // If we’re returning from the last stack frame, we’re done.
                if (call_stack.empty()) return true;
            } break;

            case ir::Op::Select: {
                auto& cond = Val(i->args()[0]);
                Temp(i) = cond.cast<bool>() ? Val(i->args()[1]) : Val(i->args()[2]);
            } break;

            case ir::Op::Store: {
                auto s = cast<ir::MemInst>(i);
                auto& ptr = Val(s->ptr());
                auto& val = Val(s->value());
                if (not StoreSRValue(ptr, val)) return false;
            } break;

            case ir::Op::Unreachable:
                return Error(entry, "Unreachable code reached");

            // Integer conversions.
            case ir::Op::SExt: TRY(CastOp(cast<ir::ICast>(i), &APInt::sext)); break;
            case ir::Op::Trunc: TRY(CastOp(cast<ir::ICast>(i), &APInt::trunc)); break;
            case ir::Op::ZExt: TRY(CastOp(cast<ir::ICast>(i), &APInt::zext)); break;

            // Equality comparison operators. These are supported for ALL types.
            case ir::Op::ICmpEq:
                if (auto v = Eq(Val(i->args()[0]), Val(i->args()[1]))) Temp(i) = SRValue(v.value());
                else return false;
                break;

            case ir::Op::ICmpNe:
                if (auto v = Eq(Val(i->args()[0]), Val(i->args()[1]))) Temp(i) = SRValue(not v.value());
                else return false;
                break;

            // Comparison operations.
            case ir::Op::ICmpSGe: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sge(rhs); })); break;
            case ir::Op::ICmpSGt: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sgt(rhs); })); break;
            case ir::Op::ICmpSLe: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sle(rhs); })); break;
            case ir::Op::ICmpSLt: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.slt(rhs); })); break;
            case ir::Op::ICmpUGe: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.uge(rhs); })); break;
            case ir::Op::ICmpUGt: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ugt(rhs); })); break;
            case ir::Op::ICmpULe: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ule(rhs); })); break;
            case ir::Op::ICmpULt: TRY(CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ult(rhs); })); break;

            // Logical operations.
            case ir::Op::And: TRY(IntOrBoolOp(i, [](const auto& lhs, const auto& rhs) { return lhs & rhs; })); break;
            case ir::Op::Or: TRY(IntOrBoolOp(i, [](const auto& lhs, const auto& rhs) { return lhs | rhs; })); break;
            case ir::Op::Xor: TRY(IntOrBoolOp(i, [](const auto& lhs, const auto& rhs) { return lhs ^ rhs; })); break;

            // Arithmetic operations.
            case ir::Op::Add: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs + rhs; })); break;
            case ir::Op::AShr: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ashr(rhs); })); break;
            case ir::Op::IMul: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs * rhs; })); break;
            case ir::Op::LShr: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.lshr(rhs); })); break;
            case ir::Op::SDiv: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sdiv(rhs); })); break;
            case ir::Op::Shl: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.shl(rhs); })); break;
            case ir::Op::SRem: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.srem(rhs); })); break;
            case ir::Op::Sub: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs - rhs; })); break;
            case ir::Op::UDiv: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.udiv(rhs); })); break;
            case ir::Op::URem: TRY(IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.urem(rhs); })); break;

            // Checked arithmetic operations.
            case ir::Op::SAddOv: TRY(OvOp(i, &APInt::sadd_ov)); break;
            case ir::Op::SMulOv: TRY(OvOp(i, &APInt::smul_ov)); break;
            case ir::Op::SSubOv: TRY(OvOp(i, &APInt::ssub_ov)); break;
        }
    }

    Error(entry, "Exceeded maximum compile-time evaluation steps");
    Remark("You can increase the limit by passing '--eval-steps=N';\fthe current value is {}.", max_steps);
    return false;
}

auto Eval::FFICall(ir::Proc* proc, ArrayRef<ir::Value*> args) -> std::optional<SRValue> {
    auto ret = FFIType(proc->type()->ret());
    if (not ret) return std::nullopt;
    SmallVector<ffi_type*> arg_types;
    for (auto a : args) {
        auto arg_ty = FFIType(cast<ir::Value>(a)->type());
        if (not arg_ty) return std::nullopt;
        arg_types.push_back(arg_ty);
    }

    ffi_cif cif{};
    ffi_status status{};
    if (proc->type()->variadic()) {
        status = ffi_prep_cif_var(
            &cif,
            FFI_DEFAULT_ABI,
            unsigned(proc->args().size()),
            unsigned(args.size()),
            ret,
            arg_types.data()
        );
    } else {
        status = ffi_prep_cif(
            &cif,
            FFI_DEFAULT_ABI,
            unsigned(args.size()),
            ret,
            arg_types.data()
        );
    }

    if (status != 0) {
        Error(entry, "Failed to prepare FFI call");
        return std::nullopt;
    }

    // Prepare space for the return type.
    SmallVector<std::byte, 64> ret_storage;
    ret_storage.resize(proc->type()->ret()->size(vm.owner()).bytes());

    // Store the arguments to memory.
    SmallVector<void*> arg_values;
    llvm::BumpPtrAllocator alloc;
    for (auto a : args) {
        auto& v = Val(a);
        auto align = v.type()->align(vm.owner());
        auto sz = v.type()->size(vm.owner());
        auto mem = alloc.Allocate(sz.bytes(), align.value().bytes());
        if (not FFIStoreArg(mem, v)) return std::nullopt;
        arg_values.push_back(mem);
    }

    // Obtain the procedure address.
    auto [it, not_found] = vm.native_symbols.try_emplace(proc, nullptr);
    if (not_found) {
        auto sym = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(std::string{proc->name().sv()});
        if (not sym) {
            Error(entry, "Failed to find symbol for FFI call to '{}'", proc->name());
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
    return FFILoadRes(ret_storage.data(), proc->type()->ret());
}

auto Eval::FFILoadRes(const void* mem, Type ty) -> std::optional<SRValue> {
    // FIXME: This just doesn’t work because we have 3 address spaces; we need to
    // merge them into a single address space (split the 64-bit address space into
    // 3 big areas, that should be enough space).
    if (isa<PtrType>(ty)) Todo("Convert pointer back to a virtual pointer");

    // FIXME: This doesn’t work if the host and target are different...
    return LoadSRValue(mem, ty);
}

auto Eval::FFIType(Type ty) -> ffi_type* {
    switch (ty->kind()) {
        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::SliceType:
        case TypeBase::Kind::StructType:
        case TypeBase::Kind::ProcType:
            ICE(entry, "Cannot call native function with value of type '{}'", ty);
            return nullptr;

        case TypeBase::Kind::PtrType:
            return &ffi_type_pointer;

        case TypeBase::Kind::IntType:
            switch (cast<IntType>(ty)->bit_width().bits()) {
                case 8: return &ffi_type_sint8;
                case 16: return &ffi_type_sint16;
                case 32: return &ffi_type_sint32;
                case 64: return &ffi_type_sint64;
                default:
                    ICE(entry, "Unsupported integer type in FFI call: {}", ty);
                    return nullptr;
            }

        case TypeBase::Kind::BuiltinType:
            switch (cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::Deduced:
                case BuiltinKind::NoReturn:
                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable();

                case BuiltinKind::Void: return &ffi_type_void;
                case BuiltinKind::Bool: return &ffi_type_uint8;
                case BuiltinKind::Type: return &ffi_type_pointer;
                case BuiltinKind::Int:
                    Assert(ty->size(vm.owner()).bits() == 64); // TODO: Support non-64 bit platforms.
                    return &ffi_type_sint64;
            }

            Unreachable();
    }

    Unreachable();
}

bool Eval::FFIStoreArg(void* ptr, const SRValue& val) {
    // We need to convert virtual pointers to host pointers here.
    //
    // We also have no idea how the function we’re calling is going to
    // use the pointer we hand it, so just be permissive with the access
    // mode (i.e. pretend we require 1 byte and that it’s readonly).
    if (val.isa<Pointer>()) {
        auto native = GetMemoryPointer(val.cast<Pointer>(), Size::Bytes(1), true);
        if (not native) return false;
        std::memcpy(ptr, &native, sizeof(native));
        return true;
    }

    // TODO: This doesn’t work if the host and target are different...
    return StoreSRValue(ptr, val);
}

auto Eval::GetMemoryPointer(Pointer p, Size accessible_size, bool readonly) -> void* {
    // Requesting 0 bytes is problematic because that might cause us
    // to recognise a pointer as part of the wrong memory region if
    // 2 regions are directly adjacent. Accesses that don’t know how
    // many bytes they’re going to access must request at least 1 byte.
    //
    // Zero-sized types and accesses should have been eliminated entirely
    // during codegen; we’re not equipped to deal with them here.
    Assert(accessible_size.bytes() != 0, "Must request at least 1 byte");

    // This is the null pointer in some address space.
    if (p.is_null_ptr()) {
        Error(entry, "Attempted to dereference 'nil'");
        return nullptr;
    }

    // This is a pointer to the stack.
    if (p.is_stack_ptr()) {
        if (p.value() + accessible_size.bytes() > stack.size()) {
            Error(entry, "Out-of-bounds memory access");
            return nullptr;
        }

        return stack.data() + p.value();
    }

    // This is a readonly pointer to host memory.
    if (p.is_host_ptr()) {
        if (not readonly) {
            Error(entry, "Attempted to write to readonly memory");
            return nullptr;
        }

        auto host_ptr = host_memory.lookup(p, accessible_size);
        if (not host_ptr) {
            Error(entry, "Out-of-bounds or invalid memory read");
            return nullptr;
        }

        // This is fine because we already checked that we’re only
        // doing this for readonly access.
        return const_cast<void*>(host_ptr);
    }

    ICE(entry, "Accessing heap memory isn’t supported yet");
    return nullptr;
}

auto Eval::GetStringData(const SRValue& val) -> std::string {
    auto sl = val.dyn_cast<SRSlice>();
    if (not sl or val.type() != vm.owner().StrLitTy) return "<invalid>";

    // Suppress memory errors if the pointer is invalid.
    tempset complain = false;
    auto sz = Size::Bytes(sl->size.getZExtValue());
    if (sz.bytes() == 0) return "";
    auto mem = GetMemoryPointer(sl->data, sz, true);
    if (not mem) return "<invalid>";

    // We got a valid pointer+size; extract the string data.
    return utils::Escape(std::string_view{static_cast<const char*>(mem), sz.bytes()});
}

auto Eval::LoadSRValue(const SRValue& ptr, Type ty) -> std::optional<SRValue> {
    auto sz = ty->size(vm.owner());
    auto mem = GetMemoryPointer(ptr.cast<Pointer>(), sz, true);
    if (not mem) return std::nullopt;
    return LoadSRValue(mem, ty);
}

auto Eval::LoadSRValue(const void* mem, Type ty) -> std::optional<SRValue> {
    /// FIXME: Load and Store assume that e.g. the pointer size on the host and target are the same.
    auto Load = [&]<typename T> -> T {
        static_assert(std::is_trivially_copyable_v<T>);
        T v;
        std::memcpy(&v, mem, sizeof(v));
        return v;
    };

    switch (ty->kind()) {
        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::StructType:
            Unreachable();

        case TypeBase::Kind::ProcType:
            return SRValue(Load.operator()<SRClosure>(), ty);

        case TypeBase::Kind::PtrType:
            return SRValue(Load.operator()<Pointer>(), ty);

        case TypeBase::Kind::SliceType: {
            auto ptr = Load.operator()<Pointer>();
            auto sz = LoadSRValue(static_cast<const char*>(mem) + sizeof(Pointer), Type::IntTy)->cast<APInt>();
            return SRValue(SRSlice{ptr, std::move(sz)}, ty);
        }

        case TypeBase::Kind::IntType: {
            auto wd = u32(cast<IntType>(ty)->bit_width().bits());
            APInt i{wd, 0};
            LoadIntFromMemory(i, static_cast<const u8*>(mem), u32(ty->size(vm.owner()).bytes()));
            return SRValue(std::move(i), ty);
        }

        case TypeBase::Kind::BuiltinType:
            switch (cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::Deduced:
                case BuiltinKind::NoReturn:
                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable();

                case BuiltinKind::Void: return SRValue();
                case BuiltinKind::Bool: return SRValue(Load.operator()<bool>());
                case BuiltinKind::Type: return SRValue(Type(Load.operator()<TypeBase*>()));
                case BuiltinKind::Int:
                    Assert(ty->size(vm.owner()).bits() == 64); // TODO: Support non-64 bit platforms.
                    return SRValue(APInt(64, Load.operator()<u64>()), ty);
            }

            Unreachable();
    }

    Unreachable();
}

auto Eval::MakeProcPtr(ir::Proc* proc) -> Pointer {
    auto it = find(procedure_indices, proc);
    if (it != procedure_indices.end()) return Pointer::Proc(u32(it - procedure_indices.begin()));
    procedure_indices.push_back(proc);
    return Pointer::Proc(u32(procedure_indices.size() - 1));
}

bool Eval::PushFrame(ir::Proc* proc, ArrayRef<ir::Value*> args) {
    Assert(not proc->empty());
    StackFrame frame{proc};
    frame.stack_base = stack.size();

    // Initialise call arguments.
    for (auto [p, a] : zip(proc->args(), args))
        frame.temporaries[Encode(p)] = Val(a);

    // Allocate frame slots.
    auto sz = Size::Bytes(stack.size());
    for (auto f : proc->frame()) {
        auto ty = f->allocated_type();
        sz = sz.align(ty->align(vm.owner()));
        frame.temporaries[Encode(f)] = SRValue(Pointer::Stack(sz.bytes()), f->type());
        sz += ty->size(vm.owner());
    }
    stack.resize(sz.bytes());

    // Allocate temporaries for instructions and block arguments.
    for (auto b : proc->blocks()) {
        for (auto a : b->arguments()) frame.temporaries[Encode(a)] = {};
        for (auto i : b->instructions()) {
            for (auto [n, _] : enumerate(i->result_types())) {
                frame.temporaries[Encode(i, u32(n))] = {};
            }
        }
    }

    // Now that we’ve set up the frame, add it to the stack; we need
    // to do this *after* we initialise the call arguments above.
    call_stack.push_back(std::move(frame));

    // Branch to the entry block.
    return BranchTo(proc->entry(), {});
}

bool Eval::StoreSRValue(const SRValue& ptr, const SRValue& val) {
    auto ty = val.type();
    auto sz = ty->size(vm.owner());
    auto mem = GetMemoryPointer(ptr.cast<Pointer>(), sz, false);
    if (not mem) return false;
    return StoreSRValue(mem, val);
}

bool Eval::StoreSRValue(void* ptr, const SRValue& val) {
    auto Store = [&]<typename T>(T v) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(ptr, &v, sizeof(v));
        return true;
    };

    auto v = utils::Overloaded{
        // clang-format off
        [&](bool b) { return Store(b); },
        [&](std::monostate) { return ICE(entry, "I don’t think we can get here?"); },
        [&](ir::Proc*) { return ICE(entry, "TODO: Store closure in memory"); },
        [&](Type ty) { return Store(ty.ptr()); },
        [&](Pointer p) { return Store(p.encode()); },
        [&](const SRClosure& cl) -> bool { return Store(cl); },
        [&](const APInt& i) {
            StoreIntToMemory(i, static_cast<u8*>(ptr), u32(val.type()->size(vm.owner()).bytes()));
            return true;
        },
        [&](this auto& self, const SRSlice& sl) -> bool {
            if (not self(sl.data)) return false;
            ptr = static_cast<char*>(ptr) + sizeof(Pointer{}.encode());
            return self(sl.size);
        },
    }; // clang-format on

    return val.visit(v);
}

auto Eval::Temp(ir::Argument* i) -> SRValue& {
    return const_cast<SRValue&>(call_stack.back().temporaries.at(Encode(i)));
}

auto Eval::Temp(ir::FrameSlot* f) -> SRValue& {
    return const_cast<SRValue&>(call_stack.back().temporaries.at(Encode(f)));
}

auto Eval::Temp(ir::Inst* i, u32 idx) -> SRValue& {
    return const_cast<SRValue&>(call_stack.back().temporaries.at(Encode(i, idx)));
}

auto Eval::ValImpl(ir::Value* v) -> Ptr<const SRValue> {
    auto Materialise = [&](SRValue val) -> const SRValue* {
        return &call_stack.back().materialised_values.push_back(std::move(val));
    };

    switch (v->kind()) {
        using K = ir::Value::Kind;
        case K::Block: Unreachable("Can’t take address of basic block");
        case K::LargeInt: Todo();
        case K::Proc: {
            auto p = cast<ir::Proc>(v);
            return Materialise(SRValue{SRClosure(MakeProcPtr(p), Pointer()), v->type()});
        }

        case K::Extract: {
            auto e = cast<ir::Extract>(v);
            auto& agg = Val(e->aggregate()).cast<SRSlice>();
            auto idx = e->index();
            if (idx == 0) return Materialise(SRValue(agg.data, e->type()));
            if (idx == 1) return Materialise(SRValue(agg.size, e->type()));
            Unreachable();
        }

        case K::InvalidLocalReference: {
            auto l = cast<ir::InvalidLocalReference>(v)->referenced_local();
            Error(l->location(), "Cannot access variable declared outside the current evaluation context");
            Note(l->decl->location(), "Variable declared here");
            return nullptr;
        }

        case K::FrameSlot:
            return &Temp(cast<ir::FrameSlot>(v));

        case K::Slice: {
            auto sl = cast<ir::Slice>(v);
            auto& ptr = Val(sl->data);
            auto& sz = Val(sl->size);
            return Materialise(SRValue(SRSlice{ptr.cast<Pointer>(), sz.cast<APInt>()}, sl->type()));
        }

        case K::StringData: {
            auto s = cast<ir::StringData>(v);
            auto ptr = host_memory.create_map(s->value().data(), Size::Bytes(s->value().size()));
            return Materialise(SRValue(ptr, v->type()));
        }

        case K::Argument:
            return &Temp(cast<ir::Argument>(v));

        case K::BuiltinConstant: {
            switch (cast<ir::BuiltinConstant>(v)->id) {
                case ir::BuiltinConstantKind::True: return &true_val;
                case ir::BuiltinConstantKind::False: return &false_val;
                case ir::BuiltinConstantKind::Poison: Unreachable("Should not exist in checked mode");
                case ir::BuiltinConstantKind::Nil: return Materialise(SRValue::Empty(vm.owner(), v->type()));
            }
            Unreachable();
        }

        case K::InstValue: {
            auto ival = cast<ir::InstValue>(v);
            return &Temp(ival->inst(), ival->index());
        }

        case K::SmallInt: {
            auto i = cast<ir::SmallInt>(v);
            auto wd = u32(i->type()->size(vm.owner()).bits());
            return Materialise(SRValue{APInt(wd, u64(i->value())), i->type()});
        }
    }

    Unreachable();
}

auto Eval::eval(Stmt* s) -> std::optional<SRValue> {
    // Compile the procedure.
    entry = s->location();
    auto proc = cg.emit_stmt_as_proc_for_vm(s);

    // Set up a stack frame for it.
    SRValue res;
    TRY(PushFrame(proc, {}));
    call_stack.back().ret = &res;

    // Dew it.
    if (not EvalLoop()) return std::nullopt;
    return std::move(res);
}

// ============================================================================
//  VM API
// ============================================================================
VM::~VM() = default;
VM::VM(TranslationUnit& owner_tu) : owner_tu{owner_tu} {}

auto VM::eval(
    Stmt* stmt,
    bool complain
) -> std::optional<SRValue> {
    using OptVal = std::optional<SRValue>;

    // Fast paths for common values.
    if (auto e = dyn_cast<Expr>(stmt)) {
        auto val = e->visit(utils::Overloaded{// clang-format off
            [](auto*) -> OptVal { return std::nullopt; },
            [](IntLitExpr* i) -> OptVal { return SRValue{i->storage.value(), i->type}; },
            [](BoolLitExpr* b) -> OptVal { return SRValue(b->value); },
            [](TypeExpr* t) -> OptVal { return SRValue{t->value}; },
        }); // clang-format on

        // If we got a value, just return it.
        if (val.has_value()) return val;
    }

    // Otherwise, we need to do this the complicated way. Compile the statement.
    Eval e{*this, complain};
    return e.eval(stmt);
}
