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

#include <base/Serialisation.hh>

#include <optional>
#include <print>
#include <ranges>

using namespace srcc;
using namespace srcc::eval;
namespace ir = cg::ir;

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

bool Value::operator==(const Value& other) const {
    if (value.index() != other.value.index()) return false;
    return visit(utils::Overloaded{
        [&](bool b) { return b == other.cast<bool>(); },
        [&](std::monostate) { return other.isa<std::monostate>(); },
        [&](ProcDecl* proc) { return proc == other.cast<ProcDecl*>(); },
        [&](Type ty) { return ty == other.cast<Type>(); },
        [&](const LValue& lval) { return lval == other.cast<LValue>(); },
        [&](const APInt& i) { return i == other.cast<APInt>(); },
        [&](const Slice& s) { return s == other.cast<Slice>(); },
        [&](const Reference& r) { return r == other.cast<Reference>(); },
    });
}

void Value::dump(bool use_colour) const {
    std::print("{}", text::RenderColours(use_colour, print().str()));
}

auto Value::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    utils::Overloaded V{// clang-format off
        [&](bool) { out += std::format("%1({})", value.get<bool>()); },
        [&](std::monostate) { },
        [&](ProcDecl* proc) { out += std::format("%2({})", proc->name); },
        [&](Type ty) { out += ty->print(); },
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
        [](Type) { return Expr::SRValue; },
        [](const APInt&) { return Expr::SRValue; },
        [](const LValue&) { return Expr::LValue; },
        [](const Reference&) { return Expr::LValue; },
    });
}

auto LValue::print() const -> SmallUnrenderedString {
    return SmallUnrenderedString("<lvalue>");
}

namespace {
#define INTEGER_OP(name) \
    name,                \
        name##I8 = name, \
        name##I16,       \
        name##I32,       \
        name##I64,       \
        name##APInt

#define TYPED_OP(name) \
    INTEGER_OP(name),  \
        name##Bool,    \
        name##Closure, \
        name##Ptr,     \
        name##Slice,   \
        name##Type

#define INTEGER_EXT(X, i)       \
    X(I8ToI16, i##8, i##16)     \
    X(I8ToI32, i##8, i##32)     \
    X(I8ToI64, i##8, i##64)     \
    X(I8ToAPInt, i##8, APInt)   \
    X(I16ToI32, i##16, i##32)   \
    X(I16ToI64, i##16, i##64)   \
    X(I16ToAPInt, i##16, APInt) \
    X(I32ToI64, i##32, i##64)   \
    X(I32ToAPInt, i##32, APInt) \
    X(I64ToAPInt, i##64, APInt) \
    X(APIntToAPInt, APInt, APInt)

#define INTEGER_TRUNC(X)           \
    X(TruncI8ToAPInt, i8, APInt)   \
    X(TruncI16ToI8, i16, i8)       \
    X(TruncI16ToAPInt, i16, APInt) \
    X(TruncI32ToI8, i32, i8)       \
    X(TruncI32ToI16, i32, i16)     \
    X(TruncI32ToAPInt, i32, APInt) \
    X(TruncI64ToI8, i64, i8)       \
    X(TruncI64ToI16, i64, i16)     \
    X(TruncI64ToI32, i64, i32)     \
    X(TruncI64ToAPInt, i32, APInt) \
    X(TruncAPIntToI8, APInt, i8)   \
    X(TruncAPIntToI16, APInt, i16) \
    X(TruncAPIntToI32, APInt, i32) \
    X(TruncAPIntToI64, APInt, i64) \
    X(TruncAPIntToAPInt, APInt, APInt)

// ============================================================================
//  Operations
// ============================================================================
/// Bytecode operations.
///
/// Each opcode is a single byte. The function of an opcode is
/// described by a comment above it, and a list of its operands
/// is given at the end of the comment. The operands may either
/// be source-level types (e.g. u8[]), or compiler-internal types
/// (e.g. Location).
///
/// When a vm procedure is executed, it reserves space for its
/// arguments, basic block arguments, and any temporaries produced
/// by its instructions. Instead of pushing and popping values off
/// a stack, each instruction that produces a value is mapped to
/// a slot in the frame.
///
/// If a subsequent instruction then references a value produced
/// by a prior instruction, it can simply access that instruction’s
/// slot in the frame. This allows us to deal with the fact that
/// an instruction’s values may be referenced multiple times throughout
/// the function.
///
/// An ‘I’ denotes immediate operands, and an ‘F’ denotes operands
/// that reference frame slots. A declaration of the form ‘X, Y -> Z’
/// means that the instruction takes two operands of type X and Y,
/// and its result is a value of type Z.
///
/// Immediate arguments always precede frame arguments.
enum class Op : u8 {
    /// Abort evaluation due to a constraint violation (e.g. a failed
    /// assertion or integer overflow).
    ///
    /// F: u8[], u8[]
    /// I: AbortReason, Location
    Abort,

    /// Integer addition.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Add),

    /// Integer bitwise AND.
    ///
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(And),

    /// Integer arithmetic shift right.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(AShr),

    /// Unconditional branch.
    ///
    /// I: BlockAddr
    Branch,

/// Integer conversions.
///
/// F: iX -> iY
#define X(name, ...) SExt##name,
    INTEGER_EXT(X, i)
#undef X
#define X(name, ...) ZExt##name,
        INTEGER_EXT(X, i)
#undef X

    /// Integer comparison.
    ///
    /// F: iX, iX -> bool
    INTEGER_OP(CmpEq),
    INTEGER_OP(CmpNe),
    INTEGER_OP(CmpSLt),
    INTEGER_OP(CmpULt),
    INTEGER_OP(CmpSGt),
    INTEGER_OP(CmpUGt),
    INTEGER_OP(CmpSLe),
    INTEGER_OP(CmpULe),
    INTEGER_OP(CmpSGe),
    INTEGER_OP(CmpUGe),

    /// Conditional branch.
    ///
    /// F: bool
    /// I: BlockAddr, BlockAddr
    CondBranch,

    /// Direct call to a procedure.
    ///
    /// The arguments are determine by the procedure type and
    /// remain on the stack after the call.
    ///
    /// F: [any...] -> [any...]
    /// I: Pointer
    DirectCall,

    /// Load a value.
    ///
    /// F: Pointer -> any
    TYPED_OP(Load),

    /// Integer logical shift right.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(LShr),

    /// Set memory to 0.
    ///
    /// F: Pointer
    /// I: i64
    MemZero,

    /// Integer multiplication.
    INTEGER_OP(Mul),

    /// Integer bitwise OR.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Or),

    /// Pointer addition.
    ///
    /// F: Pointer, i64 -> Pointer
    PtrAdd,

    /// Push an integer.
    ///
    /// F: _ -> iX
    INTEGER_OP(Push),

    /// Return a value.
    ///
    /// F: any
    TYPED_OP(Ret),

    /// Return from a function without a return value.
    RetVoid,

    /// Integer addition with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SAddOv),

    /// Signed integer division.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(SDiv),

    /// Select between two values.
    ///
    /// F: bool, any, any -> any
    Select,

    /// Integer left shift.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Shl),

    /// Signed multiplication with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SMulOv),

    /// Signed integer remainder.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(SRem),

    /// Signed subtraction with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SSubOv),

    /// Store a value.
    ///
    /// F: Pointer, any
    TYPED_OP(Store),

    /// Integer subtraction.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Sub),

/// Integer truncation.
///
/// F: iX -> iY
#define X(name, ...) name,
    INTEGER_TRUNC(X)
#undef X

    /// Unsigned integer division.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(UDiv),

    /// Unreachable instruction.
    Unreachable,

    /// Call to a procedure that has not been compiled yet.
    ///
    /// F: [any...] -> [any...]
    /// I: String
    UnresolvedCall,

    /// Unsigned integer remainder.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(URem),

    /// Integer bitwise XOR.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Xor),
};
} // namespace

// ============================================================================
//  Compiler
// ============================================================================
/// A compile-time string constant.
///
/// The bytecode doesn’t persist across compiler invocations, so we
/// can just store strings as a pointer+size. This type is used to
/// simplify serialisation thereof.
///
/// We don’t serialise String itself like this because we don’t want
/// to accidentally serialise pointers elsewhere in the compiler.
struct CTString {
    String value;

    void deserialise(ByteReader& r) {
        auto data = reinterpret_cast<const char*>(r.read<uptr>());
        auto size = r.read<u64>();
        value = String::CreateUnsafe(StringRef{data, size});
    }

    void serialise(ByteWriter& w) const {
        w << uptr(value.data()) << u64(value.size());
    }
};

class VM::Compiler : DiagsProducer<bool> {
    friend DiagsProducer;

public:
    VM& vm;
    cg::CodeGen cg{vm.owner(), Size::Bits(64)}; // TODO: Use target word size.
    Location l;
    bool complain = true;

    Compiler(VM& vm) : vm{vm} {}
    bool compile(Stmt* stmt, bool complain);

private:
    /// Get the diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine& { return vm.owner().context().diags(); }

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        if (complain) vm.owner().context().diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    void CompileProcedure(ir::Proc* proc);
};

class eval::EmitProcedure {
    VM::Compiler& c;
    ByteBuffer temporary;
    ByteWriter code{temporary};

    /// Total size of the stack frame for this procedure.
    u32 frame_size = 0;

    /// Offset to the slot for the temporary(s) produced
    /// by an instruction.
    DenseMap<ir::Inst*, u32> frame_offsets{};

public:
    EmitProcedure(VM::Compiler& c) : c{c} {}

    void compile(ir::Inst* i);

private:
    void Emit(ir::Value* val);
    void EmitAll(ArrayRef<ir::Value*> vals);
    void EmitBlockAddress(ir::Block* b);
};

VM::~VM() = default;
VM::VM(TranslationUnit& owner_tu) : owner_tu{owner_tu}, registers(MaxRegisters) {
    compiler.reset(new Compiler(*this));
}

auto VM::eval(
    Stmt* stmt,
    bool complain
) -> std::optional<Value> {
    using OptVal = std::optional<Value>;

    // Fast paths for common values.
    if (auto e = dyn_cast<Expr>(stmt)) {
        e = e->strip_parens();
        auto val = e->visit(utils::Overloaded{// clang-format off
            [](auto*) -> OptVal { return std::nullopt; },
            [](IntLitExpr* i) -> OptVal { return Value{i->storage.value(), i->type}; },
            [](BoolLitExpr* b) -> OptVal { return Value(b->value); },
            [](TypeExpr* t) -> OptVal { return Value{t->value}; },
            [](ProcRefExpr* p) -> OptVal { return Value{p->decl}; },
            [&](StrLitExpr* s) -> OptVal {
                return Value{
                    Slice{
                        Reference{LValue{s->value, owner().StrLitTy, s->location(), false}, APInt::getZero(64)},
                        APInt(u32(Types::IntTy->size(owner()).bits()), s->value.size(), false),
                    },
                    owner().StrLitTy,
                };
            }
        }); // clang-format on

        // If we got a value, just return it.
        if (val.has_value()) return val;
    }

    // Otherwise, we need to do this the complicated way. Compile the statement.
    Compiler c{*this};
    if (not c.compile(stmt, complain)) return std::nullopt;
    std::exit(42);
}

bool VM::Compiler::compile(Stmt* stmt, bool complain) {
    tempset this->complain = complain;
    tempset l = stmt->location();
    auto proc = cg.emit_stmt_as_proc_for_vm(stmt);

    // Codegen may have emitted intrinsics, e.g. exponentiation functions;
    // we need to compile and link these in if they’re present.
    for (auto p : cg.procedures()) {
        if (
            p == proc or
            proc->empty() or
            vm.procedure_table.contains(proc->name())
        ) continue;
        Todo("Emit procedure here");
    }

    // Compile the procedure.
    CompileProcedure(proc);
    std::exit(43);
}

void VM::Compiler::CompileProcedure(ir::Proc* proc) {
    EmitProcedure ep{*this};
    for (auto* b : proc->blocks()) {
        for (auto* i : b->instructions()) {
            ep.compile(i);
        }
    }

    std::exit(44);
}

void EmitProcedure::compile(ir::Inst* i) {
    auto SizedBinOp = [&](Op BaseOp) {
        switch (i->args()[0]->type()->size(c.vm.owner())) {
            case 8: code << Op(+BaseOp + 0); break;
            case 16: code << Op(+BaseOp + 1); break;
            case 32: code << Op(+BaseOp + 2); break;
            case 64: code << Op(+BaseOp + 3); break;
            default: code << Op(+BaseOp + 4); break;
        }
        EmitAll(i->args());
    };

    auto IntOp = [&](Type ty, Op BaseOp) {
        switch (ty->size(c.vm.owner()).bits()) {
            case 8: return Op(+BaseOp + 0);
            case 16: return Op(+BaseOp + 1);
            case 32: return Op(+BaseOp + 2);
            case 64: return Op(+BaseOp + 3);
            default: Unreachable("Invalid size for type 'int'");
        }
    };

#define COMPILE_TYPED_OP(type, op)                                                     \
    do {                                                                               \
        Type ty = type;                                                                \
        switch (ty->type_kind) {                                                       \
            using K = TypeBase::Kind;                                                  \
            case K::ArrayType: Todo();                                                 \
            case K::BuiltinType:                                                       \
                switch (cast<BuiltinType>(ty)->builtin_kind()) {                       \
                    case BuiltinKind::Deduced:                                         \
                    case BuiltinKind::Dependent:                                       \
                    case BuiltinKind::ErrorDependent:                                  \
                    case BuiltinKind::NoReturn:                                        \
                    case BuiltinKind::UnresolvedOverloadSet:                           \
                    case BuiltinKind::Void:                                            \
                        Unreachable();                                                 \
                                                                                       \
                    case BuiltinKind::Bool: code << Op::op##Bool; break;               \
                    case BuiltinKind::Int: code << IntOp(Types::IntTy, Op::op); break; \
                    case BuiltinKind::Type: code << Op::op##Type; break;               \
                }                                                                      \
                break;                                                                 \
                                                                                       \
            case K::IntType: code << IntOp(ty, Op::op); break;                         \
            case K::ProcType: code << Op::op##Closure; break;                          \
            case K::ReferenceType: code << Op::op##Ptr; break;                         \
            case K::SliceType: code << Op::op##Slice; break;                           \
            case K::StructType: Todo();                                                \
            case K::TemplateType: Unreachable();                                       \
        }                                                                              \
    } while (false)

#define COMPILE_EXT(Cast)                                          \
    do {                                                           \
        auto s1 = i->args()[0]->type()->size(c.vm.owner()).bits(); \
        auto s2 = i->args()[1]->type()->size(c.vm.owner()).bits(); \
        switch (s1) {                                              \
            case 8:                                                \
                switch (s2) {                                      \
                    case 16: code << Op::Cast##I8ToI16; break;     \
                    case 32: code << Op::Cast##I8ToI32; break;     \
                    case 64: code << Op::Cast##I8ToI64; break;     \
                    default: code << Op::Cast##I8ToAPInt; break;   \
                }                                                  \
                break;                                             \
            case 16:                                               \
                switch (s2) {                                      \
                    case 32: code << Op::Cast##I16ToI32; break;    \
                    case 64: code << Op::Cast##I16ToI64; break;    \
                    default: code << Op::Cast##I16ToAPInt; break;  \
                }                                                  \
                break;                                             \
            case 32:                                               \
                switch (s2) {                                      \
                    case 64: code << Op::Cast##I32ToI64; break;    \
                    default: code << Op::Cast##I32ToAPInt; break;  \
                }                                                  \
                break;                                             \
            case 64: code << Op::Cast##I64ToAPInt; break;          \
            default: code << Op::Cast##APIntToAPInt; break;        \
        }                                                          \
    } while (false)

    switch (i->opcode()) {
        using IR = ir::Op;

        // Special instructions.
        case IR::Abort: {
            auto a = cast<ir::AbortInst>(i);
            code << Op::Abort << a->abort_reason() << a->location().encode();
            EmitAll(a->args());
        } break;

        // FIXME: Store stack slots in the Proc instead of alloca instructions.
        case IR::Alloca: {
            Todo("IR Alloca");
        } break;

        // Branch to a different position in this function.
        case IR::Br: {
            auto b = cast<ir::BranchInst>(i);
            if (not b->then_args().empty()) Todo("Then args");
            if (not b->else_args().empty()) Todo("Else args");

            // Conditional branch.
            if (b->cond()) {
                code << Op::CondBranch;
                EmitBlockAddress(b->then());
                EmitBlockAddress(b->else_());
                Emit(b->cond());
            }

            // Unconditional branch.
            else {
                code << Op::Branch;
                EmitBlockAddress(b->then());
            }
        } break;

        case IR::Call: {
            auto proc = i->args().front();
            auto args = i->args().drop_front(1);

            // Emit procedure.
            if (auto p = dyn_cast<ir::Proc>(proc)) {
                // Procedure is already linked into the VM; call it directly.
                if (auto it = c.vm.procedure_table.find(p->name()); it != c.vm.procedure_table.end()) {
                    code << Op::DirectCall << it->second;
                }

                // Procedure hasn’t been resolved yet.
                else {
                    code << Op::UnresolvedCall << CTString(p->name());
                }
            } else {
                Todo("Indirect calls");
            }

            // Emit call args.
            EmitAll(args);
        } break;

        case IR::Load: {
            auto m = cast<ir::MemInst>(i);
            COMPILE_TYPED_OP(m->memory_type(), Load);
        }

        case IR::MemZero: {
            auto addr = i->args().front();
            code << Op::MemZero << i64(cast<ir::SmallInt>(i->args().back())->value());
            Emit(addr);
        } break;

        case IR::PtrAdd: {
            code << Op::PtrAdd;
            EmitAll(i->args());
        } break;

        case IR::Ret: {
            if (i->args().empty()) code << Op::RetVoid;
            else {
                COMPILE_TYPED_OP(i->args()[0]->type(), Ret);
                Emit(i->args()[0]);
            }
        } break;

        case IR::Select: {
            code << Op::Select;
            Emit(i->args()[0]);
            Emit(i->args()[1]);
            Emit(i->args()[2]);
        } break;

        case IR::Store: {
            auto m = cast<ir::MemInst>(i);
            COMPILE_TYPED_OP(m->memory_type(), Store);
            EmitAll(i->args());
        } break;

        case IR::Trunc: {
            auto s1 = i->args()[0]->type()->size(c.vm.owner()).bits();
            auto s2 = i->args()[1]->type()->size(c.vm.owner()).bits(); // clang-format off
            switch (s1) {
                case 8: code << Op::TruncI8ToAPInt; break;
                case 16: switch (s2) {
                    case 8: code << Op::TruncI16ToI8; break;
                    default: code << Op::TruncI16ToAPInt; break;
                } break;
                case 32: switch (s2) {
                    case 8: code << Op::TruncI32ToI8; break;
                    case 16: code << Op::TruncI32ToI16; break;
                    default: code << Op::TruncI32ToAPInt; break;
                } break;
                case 64: switch (s2) {
                    case 8: code << Op::TruncI64ToI8; break;
                    case 16: code << Op::TruncI64ToI16; break;
                    case 32: code << Op::TruncI64ToI32; break;
                    default: code << Op::TruncI64ToAPInt; break;
                } break;
                default: switch (s2) {
                    case 8: code << Op::TruncAPIntToI8; break;
                    case 16: code << Op::TruncAPIntToI16; break;
                    case 32: code << Op::TruncAPIntToI32; break;
                    case 64: code << Op::TruncAPIntToI64; break;
                    default: code << Op::TruncAPIntToAPInt; break;
                } break;
            } // clang-format on
        } break;

        case IR::Unreachable:
            code << Op::Unreachable;
            break;

        // Binary operations.
        case IR::Add: SizedBinOp(Op::Add); break;
        case IR::And: SizedBinOp(Op::And); break;
        case IR::AShr: SizedBinOp(Op::AShr); break;
        case IR::ICmpEq: SizedBinOp(Op::CmpEq); break;
        case IR::ICmpNe: SizedBinOp(Op::CmpNe); break;
        case IR::ICmpSGe: SizedBinOp(Op::CmpSGe); break;
        case IR::ICmpSGt: SizedBinOp(Op::CmpSGt); break;
        case IR::ICmpSLe: SizedBinOp(Op::CmpSLe); break;
        case IR::ICmpSLt: SizedBinOp(Op::CmpSLt); break;
        case IR::ICmpUGe: SizedBinOp(Op::CmpUGe); break;
        case IR::ICmpUGt: SizedBinOp(Op::CmpUGt); break;
        case IR::ICmpULe: SizedBinOp(Op::CmpULe); break;
        case IR::ICmpULt: SizedBinOp(Op::CmpULt); break;
        case IR::IMul: SizedBinOp(Op::Mul); break;
        case IR::LShr: SizedBinOp(Op::LShr); break;
        case IR::Or: SizedBinOp(Op::Or); break;
        case IR::SAddOv: SizedBinOp(Op::SAddOv); break;
        case IR::SDiv: SizedBinOp(Op::SDiv); break;
        case IR::SExt: COMPILE_EXT(SExt); break;
        case IR::Shl: SizedBinOp(Op::Shl); break;
        case IR::SMulOv: SizedBinOp(Op::SMulOv); break;
        case IR::SRem: SizedBinOp(Op::SRem); break;
        case IR::SSubOv: SizedBinOp(Op::SSubOv); break;
        case IR::Sub: SizedBinOp(Op::Sub); break;
        case IR::UDiv: SizedBinOp(Op::UDiv); break;
        case IR::URem: SizedBinOp(Op::URem); break;
        case IR::Xor: SizedBinOp(Op::Xor); break;
        case IR::ZExt: COMPILE_EXT(ZExt); break;
    }
}
