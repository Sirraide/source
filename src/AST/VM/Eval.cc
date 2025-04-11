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

#include <optional>
#include <print>
#include <ranges>

using namespace srcc;
using namespace srcc::eval;
namespace ir = cg::ir;

// ============================================================================
//  Value
// ============================================================================
SRValue::SRValue(ir::Proc* proc)
    : value(proc),
      ty(proc->type()) {}

bool SRValue::operator==(const SRValue& other) const {
    if (value.index() != other.value.index()) return false;
    return visit(utils::Overloaded{
        [&](bool b) { return b == other.cast<bool>(); },
        [&](std::monostate) { return other.isa<std::monostate>(); },
        [&](ir::Proc* proc) { return proc == other.cast<ir::Proc*>(); },
        [&](Type ty) { return ty == other.cast<Type>(); },
        [&](const APInt& i) { return i == other.cast<APInt>(); },
        [&](Pointer ptr) { return ptr == other.cast<Pointer>(); },
    });
}

void SRValue::dump(bool use_colour) const {
    std::print("{}", text::RenderColours(use_colour, print().str()));
}

auto SRValue::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{
        [&](bool) { out += std::format("%1({})", value.get<bool>()); },
        [&](std::monostate) {},
        [&](ir::Proc* proc) { out += std::format("%2({})", proc->name()); },
        [&](Type ty) { out += ty->print(); },
        [&](const APInt& value) { out += std::format("%5({})", toString(value, 10, true)); },
        [&](Pointer ptr) { out += std::format("%4({})", ptr.encode()); },
    };

    visit(V);
    return out;
}

// ============================================================================
//  Constant Evaluator
// ============================================================================
namespace {
/// Encoded temporary value.
enum struct Temporary : u64;

auto HashTemporary(utils::is<ir::Inst, ir::Block> auto* i, u32 n) -> Temporary {
    // Only use the low 32 bits of the pointer; if you need more than that
    // then seriously what is wrong with you?
    return Temporary(uptr(i) << 32 | uptr(n));
}

auto HashTemporary(ir::Argument* a) -> Temporary {
    return HashTemporary(cast<ir::Block>(a->parent()), a->index());
}

auto HashTemporary(ir::InstValue* i) -> Temporary {
    return HashTemporary(i->inst(), i->index());
}
} // namespace

class eval::Eval : DiagsProducer<bool> {
    friend DiagsProducer;

    /// A procedure on the stack.
    struct StackFrame {
        /// Procedure to which this frame belongs.
        ir::Proc* proc{};

        /// Instruction pointer for this procedure.
        ArrayRef<ir::Inst*>::iterator ip{};

        /// Temporary values for each instruction.
        DenseMap<Temporary, SRValue> temporaries;

        /// Other materialised temporaries (literal integers etc.)
        StableVector<SRValue> materialised_values;

        /// Stack size at the start of this procedure.
        usz stack_base{};

        /// Return value slot.
        SRValue* ret{};
    };

    VM& vm;
    cg::CodeGen cg;
    ByteBuffer stack;
    SmallVector<StackFrame, 4> call_stack;
    const SRValue true_val{true};
    const SRValue false_val{false};
    Location entry;
    const bool complain;

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

    auto AdjustLangOpts(LangOpts l) -> LangOpts;
    void BranchTo(ir::Block* block, ArrayRef<ir::Value*> args);
    bool EvalLoop();
    auto GetMemoryPointer(const SRValue& ptr, Size accessible_size) -> void*;
    auto LoadSRValue(const SRValue& ptr) -> std::optional<SRValue>;
    void PushFrame(ir::Proc* proc, ArrayRef<ir::Value*> args);
    bool StoreSRValue(const SRValue& ptr, const SRValue& val);
    auto Temp(ir::Inst* i, u32 idx = 0) -> SRValue&;
    auto Temp(ir::Argument* i) -> SRValue&;
    auto Val(ir::Value* v) -> const SRValue&;
};

Eval::Eval(VM& vm, bool complain)
    : vm{vm},
      cg{vm.owner(), AdjustLangOpts(vm.owner().lang_opts()), Size::Of<void*>()},
      complain{complain} {}

auto Eval::AdjustLangOpts(LangOpts l) -> LangOpts {
    l.constant_eval = true;
    l.overflow_checking = true;
    return l;
}

void Eval::BranchTo(ir::Block* block, ArrayRef<ir::Value*> args) {
    call_stack.back().ip = block->instructions().begin();
    for (auto [slot, arg] : zip(block->arguments(), args))
        Temp(slot) = Val(arg);
}

bool Eval::EvalLoop() {
    auto CastOp = [&](ir::ICast* i, APInt (APInt::*op)(unsigned width) const) {
        auto val = Val(i->args()[0]);
        auto to_wd = i->cast_result_type()->size(vm.owner());
        auto result = (val.cast<APInt>().*op)(u32(to_wd.bits()));
        Temp(i) = SRValue{std::move(result), i->cast_result_type()};
    };

    auto CmpOp = [&](ir::Inst* i, llvm::function_ref<bool(const APInt& lhs, const APInt& rhs)> op) {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        auto result = op(lhs.cast<APInt>(), rhs.cast<APInt>());
        Temp(i) = SRValue{result};
    };

    auto IntOp = [&](ir::Inst* i, llvm::function_ref<APInt(const APInt& lhs, const APInt& rhs)> op) {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        auto result = op(lhs.cast<APInt>(), rhs.cast<APInt>());
        Temp(i) = SRValue{std::move(result), lhs.type()};
    };

    auto OvOp = [&](ir::Inst* i, APInt (APInt::*op)(const APInt& RHS, bool& Overflow) const) {
        auto& lhs = Val(i->args()[0]);
        auto& rhs = Val(i->args()[1]);
        bool overflow = false;
        auto result = (lhs.cast<APInt>().*op)(rhs.cast<APInt>(), overflow);
        Temp(i, 0) = SRValue{std::move(result), lhs.type()};
        Temp(i, 1) = SRValue{overflow};
    };

    for (;;) {
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
                if (not msg1.empty()) msg += std::format(" '{}'", msg1.print());
                if (not msg2.empty()) msg += std::format(": {}", msg2.print());
                return Error(a.location(), "{}", msg);
            }

            case ir::Op::Alloca: {
                auto a = cast<ir::Alloca>(i);
                auto sz = Size::Bytes(stack.size());
                auto ty = a->allocated_type();
                sz = sz.align(ty->align(vm.owner()));
                Temp(i) = SRValue(Pointer::Stack(PointerValue(stack.size())), a->result_types()[0]);
                sz += ty->size(vm.owner());
                stack.resize(sz.bytes());
            } break;

            case ir::Op::Br: {
                auto b = cast<ir::BranchInst>(i);
                if (b->is_conditional()) {
                    auto& cond = Val(b->cond());
                    if (cond.cast<bool>()) BranchTo(b->then(), b->then_args());
                    else BranchTo(b->else_(), b->else_args());
                } else {
                    BranchTo(b->then(), b->then_args());
                }
            } break;

            case ir::Op::Call: {
                auto callee = dyn_cast<ir::Proc>(i->args()[0]);
                if (not callee) return ICE(entry, "Indirect calls at compile time are not supported yet");

                // Compile the procedure now if we haven’t done that yet.
                if (callee->empty()) {
                    Assert(callee->decl(), "Generated procedure has no body?");
                    if (not callee->decl()->body()) return ICE(
                        entry,
                        "Calling external procedures at compile time is not supported yet"
                    );

                    cg.emit(callee->decl());
                }

                // Enter the stack frame.
                PushFrame(callee, i->args().drop_front(1));

                // Set the return value slot to the call’s temporary.
                if (not i->result_types().empty()) call_stack.back().ret = &Temp(i);
            } break;

            case ir::Op::Load: {
                auto l = cast<ir::MemInst>(i);
                auto v = LoadSRValue(Val(l->ptr()));
                if (not v) return false;
                Temp(l) = std::move(v.value());
            } break;

            case ir::Op::MemZero: {
                auto addr = Val(i->args()[0]);
                auto size = Size::Bytes(Val(i->args()[1]).cast<APInt>().getZExtValue());
                auto ptr = GetMemoryPointer(addr, size);
                if (not ptr) return false;
                std::memset(ptr, 0, size.bytes());
            } break;

            case ir::Op::PtrAdd: {
                auto ptr = Val(i->args()[0]);
                auto offs = Val(i->args()[1]);
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
                auto cond = Val(i->args()[0]);
                Temp(i) = cond.cast<bool>() ? Val(i->args()[1]) : Val(i->args()[2]);
            } break;

            case ir::Op::Store: {
                auto s = cast<ir::MemInst>(i);
                auto ptr = Val(s->ptr());
                auto val = Val(s->value());
                if (not StoreSRValue(ptr, val)) return false;
            } break;

            case ir::Op::Unreachable:
                return Error(entry, "Unreachable code reached");

            // Integer conversions.
            case ir::Op::SExt: CastOp(cast<ir::ICast>(i), &APInt::sext); break;
            case ir::Op::Trunc: CastOp(cast<ir::ICast>(i), &APInt::trunc); break;
            case ir::Op::ZExt: CastOp(cast<ir::ICast>(i), &APInt::zext); break;

            // Comparison operations.
            case ir::Op::ICmpEq: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs == rhs; }); break;
            case ir::Op::ICmpNe: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs != rhs; }); break;
            case ir::Op::ICmpSGe: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sge(rhs); }); break;
            case ir::Op::ICmpSGt: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sgt(rhs); }); break;
            case ir::Op::ICmpSLe: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sle(rhs); }); break;
            case ir::Op::ICmpSLt: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.slt(rhs); }); break;
            case ir::Op::ICmpUGe: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.uge(rhs); }); break;
            case ir::Op::ICmpUGt: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ugt(rhs); }); break;
            case ir::Op::ICmpULe: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ule(rhs); }); break;
            case ir::Op::ICmpULt: CmpOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ult(rhs); }); break;

            // Arithmetic and logical operations.
            case ir::Op::Add: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs + rhs; }); break;
            case ir::Op::And: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs & rhs; }); break;
            case ir::Op::AShr: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.ashr(rhs); }); break;
            case ir::Op::IMul: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs * rhs; }); break;
            case ir::Op::LShr: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.lshr(rhs); }); break;
            case ir::Op::Or: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs | rhs; }); break;
            case ir::Op::SDiv: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.sdiv(rhs); }); break;
            case ir::Op::Shl: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.shl(rhs); }); break;
            case ir::Op::SRem: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.srem(rhs); }); break;
            case ir::Op::Sub: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs - rhs; }); break;
            case ir::Op::UDiv: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.udiv(rhs); }); break;
            case ir::Op::URem: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs.urem(rhs); }); break;
            case ir::Op::Xor: IntOp(i, [](const APInt& lhs, const APInt& rhs) { return lhs ^ rhs; }); break;

            // Checked arithmetic operations.
            case ir::Op::SAddOv: OvOp(i, &APInt::sadd_ov); break;
            case ir::Op::SMulOv: OvOp(i, &APInt::smul_ov); break;
            case ir::Op::SSubOv: OvOp(i, &APInt::ssub_ov); break;
        }
    }
}

auto Eval::GetMemoryPointer(const SRValue& ptr, Size accessible_size) -> void* {
    auto p = ptr.cast<Pointer>();
    if (not p.is_stack_ptr()) {
        ICE(entry, "TODO: Access heap or internal pointer");
        return nullptr;
    }

    auto vm = p.vm_ptr();
    if (vm.is_null()) {
        Error(entry, "Attempted to dereference 'nil'");
        return nullptr;
    }

    if (vm.non_null_value() + accessible_size.bytes() >= stack.size()) {
        Error(entry, "Out-of-bounds stack access");
        return nullptr;
    }

    return stack.data() + vm.non_null_value();
}

auto Eval::LoadSRValue(const SRValue& ptr) -> std::optional<SRValue> {
    auto ty = cast<ReferenceType>(ptr.type())->elem();
    auto sz = ty->size(vm.owner());
    auto mem = GetMemoryPointer(ptr, sz);
    if (not mem) return std::nullopt;
    ICE(entry, "TODO: Load SRValue of type {}", ty);
    return std::nullopt;
}

void Eval::PushFrame(ir::Proc* proc, ArrayRef<ir::Value*> args) {
    Assert(not proc->empty());
    auto& frame = call_stack.emplace_back(proc);
    frame.stack_base = stack.size();
    for (auto a : proc->args()) frame.temporaries[HashTemporary(a)] = {};
    for (auto b : proc->blocks()) {
        for (auto a : b->arguments()) frame.temporaries[HashTemporary(a)] = {};
        for (auto i : b->instructions()) {
            for (auto [n, _] : enumerate(i->result_types())) {
                frame.temporaries[HashTemporary(i, u32(n))] = {};
            }
        }
    }
    BranchTo(proc->entry(), args);
}

bool Eval::StoreSRValue(const SRValue& ptr, const SRValue& val) {
    auto ty = val.type();
    auto sz = ty->size(vm.owner());
    auto mem = GetMemoryPointer(ptr, sz);
    if (not mem) return false;
    ICE(entry, "TODO: Store SRValue of type {}", ty);
    return false;
}

auto Eval::Temp(ir::Argument* i) -> SRValue& {
    return call_stack.back().temporaries[HashTemporary(i)];
}

auto Eval::Temp(ir::Inst* i, u32 idx) -> SRValue& {
    return call_stack.back().temporaries[HashTemporary(i, idx)];
}

auto Eval::Val(ir::Value* v) -> const SRValue& {
    auto Materialise = [&](SRValue val) -> const SRValue& {
        return call_stack.back().materialised_values.push_back(std::move(val));
    };

    switch (v->kind()) {
        using K = ir::Value::Kind;
        case K::Block: Unreachable("Can’t take address of basic block");
        case K::Extract: Todo();
        case K::InvalidLocalReference: Todo();
        case K::LargeInt: Todo();
        case K::Proc: Todo();
        case K::Slice: Todo();
        case K::StringData: Todo();

        case K::Argument:
            return Temp(cast<ir::Argument>(v));

        case K::BuiltinConstant: {
            switch (cast<ir::BuiltinConstant>(v)->id) {
                case ir::BuiltinConstantKind::True: return true_val;
                case ir::BuiltinConstantKind::False: return false_val;
                case ir::BuiltinConstantKind::Poison: Unreachable("Should not exist in checked mode");
                case ir::BuiltinConstantKind::Nil: Todo("Materialise nil");
            }
            Unreachable();
        }

        case K::InstValue: {
            auto ival = cast<ir::InstValue>(v);
            return Temp(ival->inst(), ival->index());
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
    PushFrame(proc, {});
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
        e = e->strip_parens();
        auto val = e->visit(utils::Overloaded{// clang-format off
            [](auto*) -> OptVal { return std::nullopt; },
            [](IntLitExpr* i) -> OptVal { return SRValue{i->storage.value(), i->type}; },
            [](BoolLitExpr* b) -> OptVal { return SRValue(b->value); },
            [](TypeExpr* t) -> OptVal { return SRValue{t->value}; },
            [&](StrLitExpr* s) -> OptVal {
                Todo();
                /*return Value{
                    SliceVal{
                        Reference{LValue{s->value, owner().StrLitTy, s->location(), false}, APInt::getZero(64)},
                        APInt(u32(Types::IntTy->size(owner()).bits()), s->value.size(), false),
                    },
                    owner().StrLitTy,
                };*/
            }
        }); // clang-format on

        // If we got a value, just return it.
        if (val.has_value()) return val;
    }

    // Otherwise, we need to do this the complicated way. Compile the statement.
    Eval e{*this, complain};
    return e.eval(stmt);
}
