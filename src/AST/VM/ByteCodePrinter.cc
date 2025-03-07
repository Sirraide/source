#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Utils.hh>

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
    const TranslationUnit& tu;
    ByteReader code;
    std::string out;
    Printer(const TranslationUnit& tu, const ByteBuffer& code_buffer) : tu(tu), code(code_buffer) { }

    template <typename ...Args>
    void P(std::format_string<Args...> fmt, Args&& ...args) {
        out += std::format(fmt, LIBBASE_FWD(args)...);
    }

    void print();

    auto GetFrameOffset(OpValue val) -> FrameOffset;
    void PrintOperand();
    void PrintImmAbortReason();
    void PrintImmLocation();
    void PrintImmBlock();
    void PrintImmI64();
};

auto Printer::GetFrameOffset(OpValue op) -> FrameOffset {
    if (op < OpValue::LargeOffs) return FrameOffset(+op - +OpValue::InlineOffs);
    return code.read<FrameOffset>();
}

void Printer::PrintOperand() {
    auto op = code.read<OpValue>();
    if (op == OpValue::Lit0 or op == OpValue::Lit1 or op == OpValue::Lit2) P("%5({})", i64(+op));
    else if (op == OpValue::SmallInt) return P("%5({})", code.read<u64>());
    else P("%8(@{})", +GetFrameOffset(op));
}

void Printer::PrintImmAbortReason() {
    P("%2({})", constants::AbortHandlers[+code.read<cg::ir::AbortReason>()]);
}

void Printer::PrintImmLocation() {
    auto [file, line, col] = Location::Decode(code.read<Location::Encoded>()).info_or_builtin(tu.context());
    P("%5(<{}:{}:{}>)", file, line, col);
}

void Printer::PrintImmBlock() {
    P("%4({:#0X})", +code.read<CodeOffset>());
}

void Printer::PrintImmI64() {
    P("%5({})", code.read<i64>());
}

void Printer::print() {
    while (code.size() != 0) { // TODO: Add an empty() function.
        P("    ");
        defer { P("\n"); };
        switch (code.read<Op>()) {
#define SRCC_VM_OP_PRINTERS
#include "Ops.inc"
        }
    }
}
}

void eval::PrintByteCode(const TranslationUnit& tu, const ByteBuffer& code_buffer) {
    Printer p{tu, code_buffer};
    p.print();
    return std::print("{}", text::RenderColours(true, p.out));
}
