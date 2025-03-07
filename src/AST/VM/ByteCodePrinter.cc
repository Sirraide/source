#include "VMInternal.hh"

using namespace srcc;
using namespace srcc::eval;

#define BINOP_IMPL(lcname, size) PrintSmallIntegerBinOp<i##size>(lcname "." #size);
#define UNOP_IMPL(lcname, size) PrintSmallIntegerUnOp<i##size>(lcname "." #size);

#define FOR_EACH_SMALL_INT_TYPE(X, name, lcname)     \
    case Op::name##I8: X(lcname, 8); break;    \
    case Op::name##I16: X(lcname, 16); break; \
    case Op::name##I32: X(lcname, 32); break; \
    case Op::name##I64: X(lcname, 64); break \

#define INTEGER_BIN_OP(name, lcname) FOR_EACH_SMALL_INT_TYPE(BINOP_IMPL, name, lcname)
#define INTEGER_UN_OP(name, lcname) FOR_EACH_SMALL_INT_TYPE(UNOP_IMPL, name, lcname)

namespace {
struct Printer {
    ByteReader code;
    std::string out;
    Printer(const ByteBuffer& code_buffer) : code(code_buffer) { }

    template <typename ...Args>
    void P(std::format_string<Args...> fmt, Args&& ...args) {
        out += std::format(fmt, LIBBASE_FWD(args)...);
    }

    void print();

    auto GetFrameOffset(OpValue val) -> FrameOffset;
    template <typename IntTy> void PrintSmallIntValue();
    template <typename IntTy> void PrintSmallIntegerBinOp(std::string_view name);
    template <typename IntTy> void PrintSmallIntegerUnOp(std::string_view name);
};

auto Printer::GetFrameOffset(OpValue op) -> FrameOffset {
    if (op < OpValue::LargeOffs) return FrameOffset(+op - +OpValue::InlineOffs);
    return code.read<FrameOffset>();
}

template <typename IntTy>
void Printer::PrintSmallIntValue() {
    auto op = code.read<OpValue>();
    if (op == OpValue::Lit0 or op == OpValue::Lit1 or op == OpValue::Lit2) P("%5({})", IntTy(+op));
    else if (op == OpValue::SmallInt) return P("%5({})", code.read<u64>());
    else P("%8(@{})", +GetFrameOffset(op));
}

template <typename IntTy>
void Printer::PrintSmallIntegerBinOp(std::string_view name) {
    P("%1({}) ", name);
    PrintSmallIntValue<IntTy>();
    P(", ");
    PrintSmallIntValue<IntTy>();
}

template <typename IntTy>
void Printer::PrintSmallIntegerUnOp(std::string_view name) {
    P("%1({}) ", name);
    PrintSmallIntValue<IntTy>();
}

void Printer::print() {
    while (code.size() != 0) { // TODO: Add an empty() function.
        P("    ");
        defer { P("\n"); };

        auto op = code.read<Op>();
        switch (op) {
            INTEGER_BIN_OP(Add, "add");
            INTEGER_BIN_OP(And, "and");
            INTEGER_BIN_OP(AShr, "ashr");
            INTEGER_BIN_OP(CmpEq, "cmp.eq");
            INTEGER_BIN_OP(CmpNe, "cmp.ne");
            INTEGER_BIN_OP(CmpSLt, "cmp.slt");
            INTEGER_BIN_OP(CmpULt, "cmp.ult");
            INTEGER_BIN_OP(CmpSGt, "cmp.sgt");
            INTEGER_BIN_OP(CmpUGt, "cmp.ugt");
            INTEGER_BIN_OP(CmpSLe, "cmp.sle");
            INTEGER_BIN_OP(CmpULe, "cmp.ule");
            INTEGER_BIN_OP(CmpSGe, "cmp.sge");
            INTEGER_BIN_OP(CmpUGe, "cmp.uge");
            INTEGER_BIN_OP(LShr, "lshr");
            INTEGER_BIN_OP(Mul, "mul");
            INTEGER_BIN_OP(Or, "or");
            INTEGER_BIN_OP(SAddOv, "sadd.ov");
            INTEGER_BIN_OP(SDiv, "sdiv");
            INTEGER_BIN_OP(Shl, "shl");
            INTEGER_BIN_OP(SMulOv, "smul.ov");
            INTEGER_BIN_OP(SRem, "srem");
            INTEGER_BIN_OP(SSubOv, "ssub.ov");
            INTEGER_BIN_OP(Sub, "sub");
            INTEGER_BIN_OP(UDiv, "udiv");
            INTEGER_BIN_OP(URem, "urem");
            INTEGER_BIN_OP(Xor, "xor");
            INTEGER_UN_OP(Load, "load");
            INTEGER_UN_OP(Store, "store");
            INTEGER_UN_OP(Ret, "ret");
            default: Todo("Print op: {}", +op);
        }
    }
}
}

void eval::PrintByteCode(const ByteBuffer& code_buffer) {
    Printer p{code_buffer};
    p.print();
    return std::print("{}", text::RenderColours(true, p.out));
}
