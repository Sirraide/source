#include <deque>
#include <fmt/format.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <srcc/Macros.hh>

using namespace llvm;

cl::OptionCategory tblgen_category{"Tblgen options"};
cl::opt<bool> Debug{"print-records", cl::desc("Enable debug output"), cl::cat{tblgen_category}};

template <>
struct fmt::formatter<StringRef> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(StringRef s, FormatContext& ctx) {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

namespace src::tblgen {
static constexpr StringLiteral ExprClassName = "Expr";
static constexpr StringLiteral ExprParent = "parent";
static constexpr StringLiteral ExprTrailingArrays = "trailing_arrays";
static constexpr StringLiteral ExprFields = "fields";
static constexpr StringLiteral ExprExtraPrintout = "extra_printout";

namespace {
struct Generator {
    raw_ostream& OS;
    RecordKeeper& RK;

    struct Field {
        StringInit* type;
        StringInit* name;
    };

    struct Class {
        std::string name{};
        Record* rec{};
        SmallVector<Class*> children{};
        SmallVector<Field, 10> trailing_arrays;
        SmallVector<Field, 10> fields;
        Init* parent;
        std::string parent_class_name;

        bool is_class() const { return not children.empty(); }

        void finalise() {
            Assert(rec, "Calling finalise() on root class?");
            trailing_arrays = DagValues(cast<DagInit>(rec->getValue(ExprTrailingArrays)->getValue()));
            fields = DagValues(cast<DagInit>(rec->getValue(ExprFields)->getValue()));
            parent = rec->getValue(ExprParent)->getValue();
            parent_class_name = parent->isComplete() ? parent->getAsUnquotedString() : "Expr";
            for (auto c : children) c->finalise();
        }
    };

    /// Inheritance tree.
    std::deque<Class> classes{};

    /// Pending #undefs.
    std::vector<std::string> undefs{};

    template <typename... Args>
    void Write(fmt::format_string<Args...> fmt, Args&&... args) {
        OS << fmt::format(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Write(std::size_t indent, fmt::format_string<Args...> fmt, Args&&... args) {
        OS << std::string(indent, ' ');
        OS << fmt::format(fmt, std::forward<Args>(args)...);
    }

    template <typename Callback>
    void WriteGuarded(std::string_view GuardName, Callback cb) {
        Write("#ifdef {}\n", GuardName);
        cb();
        PushUndef(GuardName);
        Write("#endif // {}\n\n", GuardName);
    }

    static auto DagValues(DagInit* dag) -> SmallVector<Field, 10> {
        SmallVector<Field, 10> values;
        for (auto [arg, name] : llvm::zip_equal(dag->getArgs(), dag->getArgNames()))
            values.push_back({cast<StringInit>(arg), cast<StringInit>(name)});
        return values;
    }

    void PushUndef(StringRef undef) {
        undefs.emplace_back(undef);
    }

    void operator()() {
        Write("// =========================================================================\n");
        Write("//  This file is generated from {}.\n", RK.getInputFilename());
        Write("//\n");
        Write("//  Do not modify.\n");
        Write("// =========================================================================\n\n");

        /// Build the inheritance tree. This is needed for RTTI.
        for (auto r : RK.getAllDerivedDefinitions(ExprClassName)) {
            /// Get or add a new class.
            auto GetClass = [&](std::string name) -> Class* {
                auto it = std::ranges::find_if(classes, [&](auto& c) { return c.name == name; });
                if (it == classes.end()) return &classes.emplace_back(name);
                return &*it;
            };

            auto parent_init = r->getValue(ExprParent)->getValue();
            auto parent = parent_init->isComplete() ? parent_init->getAsUnquotedString() : "Expr";
            auto parent_class = GetClass(parent);
            auto name = r->getName();
            auto this_class = GetClass(std::string{name});
            this_class->rec = r;
            parent_class->children.push_back(this_class);
        }

        for (auto& c : classes[0].children) c->finalise();

        /// Forward-declare all classes.
        WriteGuarded("SRCC_PARSE_TREE_FWD", [&] {
            for (auto& c : classes[0].children) EmitForwardDecl(*c);
        });

        /// Emit Kind enum.
        WriteGuarded("SRCC_PARSE_TREE_ENUMERATORS", [&] {
            for (auto& c : classes[0].children) EmitKind(*c);
        });

        /// Emit all classes.
        WriteGuarded("SRCC_PARSE_TREE_CLASSES", [&] {
            for (auto& c : classes[0].children) EmitClassDef(*c);
        });

        /// Emit the implementation of all classes.
        WriteGuarded("SRCC_PARSE_TREE_IMPL", [&] {
            for (auto& c : classes[0].children) EmitClassImpl(*c);
            EmitPrint();
        });

        /// Emit all undefs.
        for (auto& undef : undefs) Write("#undef {}\n", undef);
    }

    void EmitKind(Class& c) {
        if (c.children.empty()) Write("{},\n", c.name);
        for (auto& ch : c.children) EmitKind(*ch);
    }

    void EmitCtorParams(Class& c, std::size_t indent, bool include_kind = true) {
        if (include_kind and c.is_class()) Write(indent, "Kind kind,\n");
        for (auto [arg, name] : c.fields) Write(indent, "{} {},\n", arg->getValue(), name->getValue());
        for (auto [arg, name] : c.trailing_arrays) Write(indent, "ArrayRef<{}> {},\n", arg->getValue(), name->getValue());
        Write(indent, "Location loc\n");
    }

    void EmitClassDef(Class& c) {
        auto rec = c.rec;

        if (c.is_class() and not c.trailing_arrays.empty())
            PrintFatalError(rec, "Base class may not have trailing data!");

        // Write class header.
        Write(
            "class srcc::Parsed{}{} : public Parsed{}",
            rec->getName(),
            c.is_class() ? "" : " final",
            c.parent_class_name
        );

        // Add trailing objects here.
        if (not c.trailing_arrays.empty()) {
            Write(", llvm::TrailingObjects<Parsed{}", rec->getName());
            for (auto& f : c.trailing_arrays) Write(", {}", f.type->getValue());
            Write(">");
        }

        Write(" {{\n");
        if (not c.trailing_arrays.empty()) Write(4, "friend TrailingObjects;\n\n");

        // Emit fields.
        if (not c.fields.empty()) {
            Write("public:\n");
            for (auto& f : c.fields) Write(4, "{} {};\n", f.type->getValue(), f.name->getValue());
            Write("\nprivate:\n");
        }

        // Emit trailing object helpers.
        if (not c.trailing_arrays.empty()) {
            for (auto& f : c.trailing_arrays) Write(4, "const u32 num_{};\n", f.name->getValue());
            Write("\n");
            for (auto& [arg, name] : c.trailing_arrays) Write(
                4,
                "auto numTrailingObjects(OverloadToken<{}>) -> usz {{ return num_{}; }}\n",
                arg->getValue(),
                name->getValue()
            );
            Write("\n");
        }

        // Emit constructor. The constructor of a base class should be protected.
        if (c.is_class()) Write("protected:\n");
        Write(4, "Parsed{}(\n", rec->getName());
        EmitCtorParams(c, 8);
        Write(4, ");\n\n");

        // Factory function.
        if (not c.is_class()) {
            Write("public:\n");
            Write(4, "static auto Create(\n");
            Write(8, "Parser& p,\n");
            EmitCtorParams(c, 8, false);
            Write(4, ") -> Parsed{}*;\n\n", rec->getName());

            // Trailing object accessors.
            for (auto& [arg, name] : c.trailing_arrays) Write(
                4,
                "auto {}() -> ArrayRef<{}> {{ return {{getTrailingObjects<{}>(), num_{}}}; }}\n",
                name->getValue(),
                arg->getValue(),
                arg->getValue(),
                name->getValue()
            );
            Write("\n");
        }

        // Emit classof.
        if (c.is_class()) {
            StringRef first = [&] -> StringRef {
                auto ch = c.children.front();
                while (ch->is_class()) ch = ch->children.front();
                return ch->name;
            }();

            StringRef last = [&] -> StringRef {
                auto ch = c.children.back();
                while (ch->is_class()) ch = ch->children.back();
                return ch->name;
            }();

            Write("public:\n");
            if (first == last) {
                Write(4, "static bool classof(const ParsedExpr* e) {{ return e->kind == Kind::{}; }}\n", first);
            } else {
                Write(4, "static bool classof(const ParsedExpr* e) {{\n");
                Write(8, " return e->kind >= Kind::{} and e->kind <= Kind::{};\n", first, last);
                Write(4, "}}\n");
            }
        } else {
            Write(4, "static bool classof(const ParsedExpr* e) {{ return e->kind == Kind::{}; }}\n", rec->getName());
        }
        Write("}};\n\n");

        // Emit children.
        for (auto ch : c.children) EmitClassDef(*ch);
    }

    void EmitClassImpl(Class& c) {
        // Emit ctor.
        Write("srcc::Parsed{}::Parsed{}(\n", c.name, c.name);
        EmitCtorParams(c, 4);
        Write(") : Parsed{}{{{}, loc}}", c.parent_class_name, c.is_class() ? "kind" : fmt::format("Kind::{}", c.name));
        for (auto [_, name] : c.fields) Write("\n  , {}{{{}}}", name->getValue(), name->getValue());
        for (auto [_, name] : c.trailing_arrays) Write("\n  , num_{}{{u32({}.size())}}", name->getValue(), name->getValue());

        // Emit ctor body.
        Write(" {{\n");
        Write(4, "static_assert(std::is_trivially_destructible_v<decltype(*this)>);\n");
        for (auto [type, name] : c.trailing_arrays) Write(
            4,
            "std::uninitialized_copy_n({}.begin(), {}.size(), getTrailingObjects<{}>());\n",
            name->getValue(),
            name->getValue(),
            type->getValue()
        );
        Write("}}\n\n");

        // Emit factory function.
        if (not c.is_class()) {
            Write("auto srcc::Parsed{}::Create(\n", c.name);
            Write(4, "Parser& $P,\n");
            EmitCtorParams(c, 4, false);
            Write(") -> Parsed{}* {{\n", c.name);

            // Calculate size to allocate.
            if (not c.trailing_arrays.empty()) {
                Write(4, "const auto $size = totalSizeToAlloc<");
                bool first = true;
                for (auto& f : c.trailing_arrays) {
                    if (first) first = false;
                    else Write(", ");
                    Write("{}", f.type->getValue());
                }
                Write(">(");
                first = true;
                for (auto& f : c.trailing_arrays) {
                    if (first) first = false;
                    else Write(", ");
                    Write("{}.size()", f.name->getValue());
                }
                Write(");\n");
            }

            // Allocate.
            Write(
                4,
                "auto $mem = $P.Allocate({}, alignof(Parsed{}));\n",
                c.trailing_arrays.empty() ? fmt::format("sizeof(Parsed{})", c.name) : "$size",
                c.name
            );

            // Construct.
            Write(4, "return new ($mem) Parsed{}({}", c.name, c.is_class() ? "kind, " : "");
            for (auto& f : c.fields) Write("{}, ", f.name->getValue());
            for (auto& f : c.trailing_arrays) Write("{}, ", f.name->getValue());
            Write("loc);\n");
            Write("}}\n\n");
        }

        // Emit children.
        for (auto ch : c.children) EmitClassImpl(*ch);
    }

    void EmitPrint() {
        Write("void Printer::Print(ParsedExpr* e) {{\n");
        Write(4, "switch (e->kind) {{\n");
        Write(8, "using enum utils::Colour;\n");
        Write(8, "using Kind = srcc::ParsedExpr::Kind;\n");
        for (auto c : classes[0].children) EmitPrintCase(*c);
        Write(4, "}}\n");
        Write("}}\n");
    }

    void EmitPrintCase(Class& c) {
        if (c.is_class()) {
            for (auto child : c.children) EmitPrintCase(*child);
            return;
        }

        Write("\n");
        Write(8, "case Kind::{}: {{\n", c.name);
        Write(12, "[[maybe_unused]] auto& x = static_cast<Parsed{}&>(*e);\n\n", c.name);

        // Print the node itself.
        Write(12, "fmt::print(\n");
        Write(16, "\"{{}}{} {{}}{{}} {{}}<{{}}>\",\n", c.name);
        Write(16, "C(Red),\n");
        Write(16, "C(Blue),\n");
        Write(16, "fmt::ptr(e),\n");
        Write(16, "C(Magenta),\n");
        Write(16, "e->loc.pos\n");
        Write(12, ");\n\n");

        // Add extra printout, if any.
        auto extra = c.rec->getValue(ExprExtraPrintout)->getValue();
        if (extra->isComplete()) Write(12, "fmt::print({});\n", StringRef{extra->getAsUnquotedString()}.trim());
        Write(12, "fmt::print(\"\\n{{}}\", C(Reset));\n");

        // Print its children, if any.
        SmallVector<StringRef> fields;
        SmallVector<StringRef> arrays;
        auto AddFields = [&] (SmallVector<StringRef>& out, ArrayRef<Field> fields) {
            for (const auto& f : fields) {
                auto ty = f.type->getValue();
                if (ty.starts_with("Parsed") and ty.ends_with("*") and not ty.ends_with("**"))
                    out.emplace_back(f.name->getValue());
            }
        };

        AddFields(fields, c.fields);
        AddFields(arrays, c.trailing_arrays);

        // Print all fields.
        if (not fields.empty() or not arrays.empty()) {
            Write("\n");
            Write(12, "SmallVector<ParsedExpr*, 10> fields;\n");
            for (auto f : fields) Write(12, "if (x.{}) fields.push_back(x.{});\n", f, f);
            for (auto a : arrays) Write(12, "if (auto a = x.{}(); not a.empty()) fields.append(a.begin(), a.end());\n", a);
            Write(12, "PrintChildren(fields);\n");
        }

        Write(8, "}} break;\n");
    }

    void EmitForwardDecl(Class& c) {
        Write("class Parsed{};\n", c.name);
        for (auto child : c.children) EmitForwardDecl(*child);
    }
};

bool Generate(raw_ostream& OS, RecordKeeper& RK) {
    if (Debug.getValue()) OS << RK;
    else Generator{OS, RK}();
    return false;
}
} // namespace
} // namespace src::tblgen

int main(int argc, char** argv) {
    cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], &src::tblgen::Generate);
}
