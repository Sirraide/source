#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>

using namespace llvm;
using namespace std::literals;

struct Emitter {
    raw_ostream& os;
    const RecordKeeper &records;

    struct IfDef {
        raw_ostream& os;
        std::string_view name;
        IfDef(raw_ostream& os, std::string_view name) : os(os), name(name) {
            os << "#ifdef SRCC_VM_OP_" << name << "\n";
        }

        ~IfDef() {
            os << "#undef SRCC_VM_OP_" << name << "\n";
            os << "#endif" << "\n\n";
        }
    };

    void emit_enumerators();
    void emit_op_builders();
    void emit_op_printers();
};

void Emitter::emit_enumerators() {
    IfDef _{os, "ENUMERATORS"};
    for (auto r : records.getAllDerivedDefinitions("OpBase"))
        os << r->getName() << ",\n";
}

void Emitter::emit_op_builders() {
    static constexpr std::string_view Prefix = "_EmitIntOp";

    IfDef _{os, "BUILDERS"};
    for (const auto& [name, rec] : records.getDefs()) {
        if (not name.starts_with(Prefix)) continue;
        os << std::format(R"cxx(
void Create{0}(ArrayRef<ir::Value*> args) {{
    switch (args[0]->type()->size(c.vm.owner()).bits()) {{
        case 8: code << Op::{0}I8; break;
        case 16: code << Op::{0}I16; break;
        case 32: code << Op::{0}I32; break;
        case 64: code << Op::{0}I64; break;
        default: code << Op::{0}APInt; break;
    }}
    EmitOperands(args);
}}
)cxx", name.substr(Prefix.size()));
    }
}

void Emitter::emit_op_printers() {
    IfDef _{os, "PRINTERS"};
    for (auto r : records.getAllDerivedDefinitions("OpBase")) {
        auto Sep = [i = 0] mutable { return i++ != 0 ? "    P(\", \");\n"sv : "    P(\" \");\n"sv; };
        os << "case Op::" << r->getName() << ":\n";
        os << "    P(\"%1(" << r->getValueAsString("name") << ")\");\n";
        for (auto arg : r->getValueAsListOfStrings("immargs"))
            os << Sep() << "    PrintImm" << arg << "();\n";
        for (auto _ : r->getValueAsListOfStrings("args"))
            os << Sep() << "    PrintOperand();\n";
        os << "    break;\n";
    }
}

static bool Main(raw_ostream &os, const RecordKeeper &records) {
    Emitter e(os, records);
    e.emit_enumerators();
    e.emit_op_builders();
    e.emit_op_printers();
    return false;
}

int main(int argc, char **argv) {
    sys::PrintStackTraceOnErrorSignal(argv[0]);
    PrettyStackTraceProgram X(argc, argv);
    cl::ParseCommandLineOptions(argc, argv);
    TableGenMain(argv[0], Main);
}
