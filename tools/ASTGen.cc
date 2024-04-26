#include <clopts.hh>
#include <fmt/core.h>
#include <fmt/std.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <ranges>
#include <filesystem>
#include <srcc/Macros.hh>

import srcc;
import srcc.utils;

using namespace srcc;

namespace srcc::detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,
    option<"-o", "Output file path", std::string, true>,
    flag<"-d", "Debug mode: dump all classes to stdout instead of generating code">,
    help<>
>; // clang-format on
}

enum struct Tk {
    Invalid,
    Eof,
    Text,
    Colon,
    LBrace,
    RBrace,
    Trailing,
    DollarIdent,
    Equals,
};

template <>
struct fmt::formatter<Tk> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(Tk t, FormatContext& out) {
        auto s = [t] -> std::string_view {
            switch (t) {
                case Tk::Invalid: return "Invalid";
                case Tk::Eof: return "Eof";
                case Tk::Text: return "Text";
                case Tk::Colon: return "Colon";
                case Tk::LBrace: return "LBrace";
                case Tk::RBrace: return "RBrace";
                case Tk::Trailing: return "Trailing";
                case Tk::DollarIdent: return "DollarIdent";
                case Tk::Equals: return "Equals";
            }

            llvm_unreachable("Invalid token kind");
        }();

        return formatter<std::string_view>::format(s, out);
    }
};

struct Field {
    std::string type;
    std::string name;
    Location loc;
};

struct Class {
    Class* base; // May be null if this is the root.
    std::string name;
    std::string decorated_name;
    std::string extra_printout;
    SmallVector<Class*> children;
    SmallVector<Field, 10> trailing_arrays;
    SmallVector<Field, 10> fields;

    SRCC_IMMOVABLE(Class);
    Class(StringRef prefix, std::string name, Class* base)
        : base(base), name(std::move(name)), decorated_name(std::string(prefix) + this->name) {}

    /// Check whether this has any derived classes.
    [[nodiscard]] bool is_base() const { return not children.empty(); }
};

// ============================================================================
//  Generator
// ============================================================================
class Generator {
    Context ctx;
    const File& input;
    File::Path out_path;
    bool debug;

    const char* curr;
    const char* end;
    char lastc;

    Tk tk;
    std::string tok_text;

    std::string class_prefix;
    std::string guard_prefix;
    std::string namespace_;
    std::string context_name;
    std::vector<std::unique_ptr<Class>> classes;
    std::vector<std::string> undefs;
    std::string out;

public:
    explicit Generator(std::string_view in_path, std::string out_path, bool debug)
        : input(ctx.get_file(in_path)),
          out_path(std::move(out_path)),
          debug(debug) {
        curr = input.data();
        end = curr + input.size();
        NextChar();
        Next();
        Parse();
        Emit();
    }

private:
    template <typename... Args>
    [[noreturn]] void Error(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename... Args>
    [[noreturn]] void Error(Location loc, fmt::format_string<Args...> fmt, Args&&... args);

    template <typename... Args>
    void Inline(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename... Args>
    void W(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename... Args>
    void W(usz indent, fmt::format_string<Args...> fmt, Args&&... args);

    [[nodiscard]] bool At(std::same_as<Tk> auto... tks);
    [[nodiscard]] bool Consume(std::same_as<Tk> auto... tks);
    void DoExpect(std::same_as<Tk> auto... tks);
    void ExpectAndConsume(Tk t);
    [[nodiscard]] auto ExpectText() -> std::string;
    [[nodiscard]] auto GetClass(std::string_view name) -> Class&;
    [[nodiscard]] auto Here() -> Location;
    [[nodiscard]] bool Kw(std::string_view text);

    [[nodiscard]] auto root() -> Class& { return *classes.front(); }

    // Lexer/Parser.
    void AddConstant(Class& cls, std::string_view name, std::string_view value);
    void Debug();
    void Next();
    void NextChar();
    void Parse();
    void ParseClass();
    void ParseClassMember(Class& cls);
    void ParseProperty(std::string& prop, std::string_view name);
    void SkipLine();

    // Emitter.
    void Emit();
    void EmitClassDef(Class& cls);
    void EmitClassImpl(Class& cls);
    void EmitCtorParams(Class& cls, usz indent, bool include_kind = true);
    void EmitForwardDecl(Class& cls);
    void EmitGuarded(std::string_view GuardName, auto cb);
    void EmitKind(Class& cls);
    void EmitOwnCtorArgs(Class& cls, usz indent);
    void EmitOwnCtorParams(Class& cls, usz indent);
    void EmitPrint();
    void EmitPrintCase(Class& cls);
    void PushUndef(StringRef undef);
};

// ============================================================================
//  General Helpers
// ============================================================================
constexpr bool IsContinue(char c) {
    return std::isalnum(c) or c == '$' or c == '_';
}

static void Trim(std::string& s) {
    auto first_non_ws = s.find_first_not_of(" \t\n\r\v\f");
    if (first_non_ws == std::string::npos) {
        s.clear();
        return;
    }

    s = s.substr(first_non_ws, s.find_last_not_of(" \t\n\r\v\f") - first_non_ws + 1);
}

template <typename... Args>
void Generator::Error(fmt::format_string<Args...> fmt, Args&&... args) {
    Error(Here(), fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void Generator::Error(Location loc, fmt::format_string<Args...> fmt, Args&&... args) {
    Diag::Fatal(ctx, loc, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void Generator::Inline(fmt::format_string<Args...> fmt, Args&&... args) {
    fmt::format_to(std::back_inserter(out), fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void Generator::W(fmt::format_string<Args...> fmt, Args&&... args) {
    W(0, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void Generator::W(usz indent, fmt::format_string<Args...> fmt, Args&&... args) {
    out.append(indent, ' ');
    Inline(fmt, std::forward<Args>(args)...);
    out += '\n';
}

// ============================================================================
//  Parser Helpers
// ============================================================================
bool Generator::At(std::same_as<Tk> auto... tks) {
    static_assert(sizeof...(tks) > 0);
    return ((tk == tks) or ...);
}

bool Generator::Consume(std::same_as<Tk> auto... tks) {
    if (At(tks...)) Next();
    else return false;
    return true;
}

void Generator::DoExpect(std::same_as<Tk> auto... tks) {
    if (not At(tks...)) {
        if constexpr (sizeof...(tks) == 1) Error("Expected '{}', got '{}'", tks...[0], tk);
        else {
            std::string expected;
            for (auto tk : {tks...}) expected += fmt::format("{}, ", tk);
            expected.pop_back();
            expected.pop_back();
            Error("Expected one of [{}], got '{}'", expected, tk);
        }
    }
}

void Generator::ExpectAndConsume(Tk t) {
    DoExpect(t);
    Next();
}

auto Generator::ExpectText() -> std::string {
    DoExpect(Tk::Text);
    auto s = tok_text;
    Next();
    return s;
}

auto Generator::GetClass(std::string_view name) -> Class& {
    auto cl = std::ranges::find_if(classes, [&](const auto& c) { return c->name == name; });
    if (cl == classes.end()) Error("Unknown class '{}'", name);
    return **cl;
}

auto Generator::Here() -> Location {
    return Location{u32(curr - input.data()), 1, u16(input.file_id())};
}

bool Generator::Kw(std::string_view text) {
    if (At(Tk::Text) and tok_text == text) Next();
    else return false;
    return true;
}

// ============================================================================
//  Parser
// ============================================================================
void Generator::AddConstant(Class& cls, std::string_view name, std::string_view value) {
    if (name == "$print") {
        cls.extra_printout += value;
        return;
    }

    Error("Unknown constant field: '{}'", name);
}

void Generator::Debug() {
    auto N = [](std::string_view sv) -> std::string {
        bool needs_quotes = false;
        for (auto& c : sv) {
            if (not IsContinue(c)) {
                needs_quotes = true;
                break;
            }
        }

        if (not needs_quotes) return std::string{sv};
        return fmt::format("\"{}\"", sv);
    };

    fmt::print("root {}\n", N(root().name));
    if (not class_prefix.empty()) fmt::print("prefix {}\n", N(class_prefix));
    if (not guard_prefix.empty()) fmt::print("guard {}\n", N(guard_prefix));
    if (not namespace_.empty()) fmt::print("namespace {}\n", N(namespace_));
    for (auto& c : classes | std::views::drop(1)) {
        fmt::print("\nclass {}", N(c->name));
        if (c->base) fmt::print(" : {}", N(c->base->name));
        fmt::print(" {{\n");
        for (auto& f : c->fields) fmt::print("    {} {}\n", N(f.type), N(f.name));
        for (auto& f : c->trailing_arrays) fmt::print("    {}[] {}\n", N(f.type), N(f.name));
        if (not c->extra_printout.empty()) fmt::print("    $print = [{{ {} }}]\n", c->extra_printout);
        fmt::print("}}\n");
    }
}

void Generator::NextChar() {
    if (curr == end) {
        lastc = 0;
        return;
    }

    lastc = *curr++;
    if (lastc == 0) Error("Null character in input file");

    /// Collapse CR LF and LF CR to a single newline,
    /// but keep CR CR and LF LF as two newlines.
    if (lastc == '\r' || lastc == '\n') {
        /// Two newlines in a row.
        if (curr != end && (*curr == '\r' || *curr == '\n')) {
            bool same = lastc == *curr;
            lastc = '\n';

            /// CR CR or LF LF.
            if (same) return;

            /// CR LF or LF CR.
            curr++;
        }

        /// Either CR or LF followed by something else.
        lastc = '\n';
    }
}

void Generator::Next() {
    while (lastc != 0 and std::isspace(lastc)) NextChar();
    const char c = lastc;
    NextChar();
    switch (c) {
        case 0: tk = Tk::Eof; break;
        case '{': tk = Tk::LBrace; break;
        case '}': tk = Tk::RBrace; break;
        case ':': tk = Tk::Colon; break;
        case '=': tk = Tk::Equals; break;

        // Comment.
        case '/': {
            if (lastc != '/') Error("Unexpected '/'. Did you mean for this to be a comment?");
            NextChar();
            SkipLine();
            Next();
        } break;

        // Unquoted text.
        case '$':
        case 'a' ... 'z':
        case 'A' ... 'Z':
        case '_': {
            tk = Tk::Text;
            tok_text = c;
            while (IsContinue(lastc)) {
                tok_text += lastc;
                NextChar();
            }
        } break;

        // Quoted text.
        case '"':
        case '\'': {
            tk = Tk::Text;
            tok_text.clear();
            while (lastc != 0 and lastc != c) {
                tok_text += lastc;
                NextChar();
            }

            if (lastc == 0) Error("Unterminated string");
            NextChar();
        } break;

        // Code or '[]'.
        case '[': {
            // Code.
            if (lastc == '{') {
                tk = Tk::Text;
                tok_text.clear();
                NextChar();
                while (lastc != 0) {
                    // Read until '}'.
                    while (lastc != '}' and lastc != 0) {
                        tok_text += lastc;
                        NextChar();
                    }

                    // Yeet '}'.
                    NextChar();

                    // At ']'. Stop.
                    if (lastc == ']') {
                        NextChar();
                        break;
                    }

                    // Append discarded '}' and keep going.
                    tok_text += '}';
                }

                // Discard whitespace at start and end of code block.
                Trim(tok_text);
                break;
            }

            // Just '[]'
            if (lastc == ']') {
                tk = Tk::Trailing;
                NextChar();
                break;
            }

            Error("Unexpected '['. Did you mean '[]' or '[{{'?");
        }

        default: Error("Unexpected character \\x{:02X}: '{}'", std::size_t(c), c);
    }
}

void Generator::SkipLine() {
    while (lastc != 0 and lastc != '\n') NextChar();
    NextChar();
}

void Generator::Parse() {
    ParseProperty(class_prefix, "prefix");
    ParseProperty(guard_prefix, "guard");
    ParseProperty(namespace_, "namespace");
    ParseProperty(context_name, "context");
    if (not Kw("root")) Error("Expected 'root' directive");
    classes.push_back(std::make_unique<Class>(class_prefix, ExpectText(), nullptr));

    while (Kw("class")) ParseClass();
    if (not At(Tk::Eof)) Error("Expected 'class'");
}

void Generator::ParseClass() {
    auto name = ExpectText();
    auto& base = Consume(Tk::Colon) ? GetClass(ExpectText()) : root();
    auto& cls = *classes.emplace_back(std::make_unique<Class>(class_prefix, name, &base));
    base.children.push_back(&cls);
    ExpectAndConsume(Tk::LBrace);
    while (not At(Tk::RBrace, Tk::Eof)) ParseClassMember(cls);
    ExpectAndConsume(Tk::RBrace);
}

void Generator::ParseClassMember(Class& cls) {
    auto loc = Here();
    auto first = ExpectText();

    // Constant field.
    if (Consume(Tk::Equals)) {
        AddConstant(cls, first, ExpectText());
        return;
    }

    // Member.
    (Consume(Tk::Trailing) ? cls.trailing_arrays : cls.fields).emplace_back( //
        std::move(first),
        ExpectText(),
        loc
    );
}

void Generator::ParseProperty(std::string& prop, std::string_view name) {
    if (Kw(name)) prop = ExpectText();
}


// ============================================================================
//  Emitter.
// ============================================================================
void Generator::Emit() {
    if (debug) {
        Debug();
        return;
    }

    W("// =========================================================================");
    W("//  This file was generated from {}.", input.name());
    W("//");
    W("//  Do not modify");
    W("// =========================================================================");
    W("");

    // Forward-declare all classes.
    EmitGuarded("FWD", [&] {
        for (auto c : root().children) EmitForwardDecl(*c);
    });

    // Emit Kind enum.
    EmitGuarded("ENUMERATORS", [&] {
        for (auto& c : root().children) EmitKind(*c);
    });

    // Emit all classes.
    EmitGuarded("CLASSES", [&] {
        for (auto& c : root().children) EmitClassDef(*c);
    });

    // Emit the implementation of all classes.
    EmitGuarded("IMPL", [&] {
        for (auto& c : root().children) EmitClassImpl(*c);
        EmitPrint();
    });

    // Emit all undefs.
    for (auto& undef : undefs) W("#undef {}_{}", guard_prefix, undef);

    // If the output file is '-', write to stdout.
    if (out_path == "-") {
        fmt::print("{}", out);
        return;
    }

    // Write to disk.
    File::WriteOrDie(out.data(), out.size(), out_path);
}

void Generator::EmitClassDef(Class& cls) {
    // Trailing data is not allowed in a base class because it would collide
    // with the data of any derived class.
    if (cls.is_base() and not cls.trailing_arrays.empty())
        Error(cls.trailing_arrays.front().loc, "Base class may not have trailing data!");

    // Write class header.
    Inline(
        "class {}::{}{} : public {}",
        namespace_,
        cls.decorated_name,
        cls.is_base() ? "" : " final",
        cls.base->decorated_name
    );

    // Add trailing objects here.
    if (not cls.trailing_arrays.empty()) {
        Inline(", llvm::TrailingObjects<{}", cls.decorated_name);
        for (auto& f : cls.trailing_arrays) Inline(",\n    {}", f.type);
        Inline("\n>");
    }

    W(" {{");
    if (not cls.trailing_arrays.empty()) W(4, "friend TrailingObjects;\n");

    // Emit fields.
    if (not cls.fields.empty()) {
        W("public:");
        for (auto& f : cls.fields) W(4, "{} {};", f.type, f.name);

        // Don’t emit an extra 'private' if we have no trailing objects and we’re a base.
        if (not cls.trailing_arrays.empty() or not cls.is_base())
            W("\nprivate:");
    }

    // Emit trailing object helpers.
    if (not cls.trailing_arrays.empty()) {
        for (auto& f : cls.trailing_arrays) W(4, "const u32 num_{};", f.name);
        W("");
        for (auto& [type, name, _] : cls.trailing_arrays) W(
            4,
            "auto numTrailingObjects(OverloadToken<{}>) -> usz {{ return num_{}; }}",
            type,
            name
        );
        W("");
    }

    // Emit constructor. The constructor of a base class should be protected.
    if (cls.is_base()) W("\nprotected:");
    W(4, "{}(", cls.decorated_name);
    EmitCtorParams(cls, 8);
    W(4, ");");
    W("");

    // Factory function.
    if (not cls.is_base()) {
        W("public:");
        W(4, "static auto Create(");
        W(8, "{}& $,", context_name);
        EmitCtorParams(cls, 8, false);
        W(4, ") -> {}*;", cls.decorated_name);
        W("");

        // Trailing object accessors. These only exist if the class is not a base.
        for (auto& [type, name, _] : cls.trailing_arrays) W(
            4,
            "[[nodiscard]] auto {}() -> ArrayRef<{}> {{ return {{getTrailingObjects<{}>(), num_{}}}; }}",
            name,
            type,
            type,
            name
        );

        if (not cls.trailing_arrays.empty()) W("");
    }

    // Emit classof.
    if (cls.is_base()) {
        StringRef first = [&] -> StringRef {
            auto ch = cls.children.front();
            while (ch->is_base()) ch = ch->children.front();
            return ch->name;
        }();

        StringRef last = [&] -> StringRef {
            auto ch = cls.children.back();
            while (ch->is_base()) ch = ch->children.back();
            return ch->name;
        }();

        W("public:");
        if (first == last) {
            W(4, "static bool classof(const {}* e) {{ return e->kind == Kind::{}; }}", root().decorated_name, first);
        } else {
            W(4, "static bool classof(const {}* e) {{", root().decorated_name);
            W(8, " return e->kind >= Kind::{} and e->kind <= Kind::{};", first, last);
            W(4, "}}");
        }
    } else {
        W(4, "static bool classof(const {}* e) {{ return e->kind == Kind::{}; }}", root().decorated_name, cls.name);
    }
    W("}};");
    W("");

    // Emit children.
    for (auto ch : cls.children) EmitClassDef(*ch);
}

void Generator::EmitClassImpl(Class& cls) {
    // Emit ctor.
    W("srcc::{}::{}(", cls.decorated_name, cls.decorated_name);
    EmitCtorParams(cls, 4);
    W(") : {} {{", cls.base->decorated_name);

    // Pass along the kind we got from our child, or ours if we’re not a base.
    if (cls.is_base()) W(8, "kind,");
    else W(8, "Kind::{},", cls.name);

    // Pass along the arguments to our bases and emit our arguments.
    EmitOwnCtorArgs(*cls.base, 8);

    // Finally, add the location.
    W(8, "loc,");
    Inline("    }}");

    // Initialise our fields.
    for (auto& [_, name, _] : cls.fields) Inline("\n  , {}{{{}}}", name, name);
    for (auto& [_, name, _] : cls.trailing_arrays) Inline("\n  , num_{}{{u32({}.size())}}", name, name);

    // Emit ctor body.
    W(" {{");
    W(4, "static_assert(std::is_trivially_destructible_v<decltype(*this)>);");
    for (auto& [type, name, _] : cls.trailing_arrays) W(
        4,
        "std::uninitialized_copy_n({}.begin(), {}.size(), getTrailingObjects<{}>());",
        name,
        name,
        type
    );
    W("}}");
    W("");

    // Emit factory function.
    if (not cls.is_base()) {
        W("auto srcc::{}::Create(", cls.decorated_name);
        W(4, "{}& $,", context_name);
        EmitCtorParams(cls, 4, false);
        W(") -> {}* {{", cls.decorated_name);

        // Calculate size to allocate.
        if (not cls.trailing_arrays.empty()) {
            W(4, "const auto $size = totalSizeToAlloc<");
            bool first = true;
            for (auto& f : cls.trailing_arrays) {
                if (first) first = false;
                else Inline(",\n");
                W(8, "{}", f.type);
            }
            W("    >(");
            first = true;
            for (auto& f : cls.trailing_arrays) {
                if (first) first = false;
                else W(",\n");
                W(8, "{}.size()", f.name);
            }
            W("    );");
        }

        // Allocate.
        W(
            4,
            "auto $mem = $.Allocate({}, alignof({}));",
            cls.trailing_arrays.empty() ? fmt::format("sizeof({})", cls.decorated_name) : "$size",
            cls.decorated_name
        );

        // Construct.
        W(4, "return new ($mem) {}({}", cls.decorated_name, cls.is_base() ? "kind, " : "");
        EmitOwnCtorArgs(cls, 8);
        W(8, "loc");
        W(4, ");");
        W("}}");
        W("");
    }

    // Emit children.
    for (auto ch : cls.children) EmitClassImpl(*ch);
}

void Generator::EmitCtorParams(Class& cls, usz indent, bool include_kind) {
    // We need to accept the kind from any child classes.
    if (include_kind and cls.is_base()) W(indent, "Kind kind,");
    EmitOwnCtorParams(cls, indent);
    W(indent, "Location loc");
}

void Generator::EmitForwardDecl(Class& cls) {
    W("class {};", cls.decorated_name);
    for (auto child : cls.children) EmitForwardDecl(*child);
}

void Generator::EmitGuarded(std::string_view GuardName, auto cb) {
    W("#ifdef {}_{}", guard_prefix, GuardName);
    cb();
    PushUndef(GuardName);
    W("#endif // {}_{}", guard_prefix, GuardName);
    W("");
}
void Generator::EmitKind(Class& cls) {
    if (not cls.is_base()) W("{},", cls.name);
    for (auto& ch : cls.children) EmitKind(*ch);
}

void Generator::EmitOwnCtorArgs(Class& cls, usz indent) {
    if (cls.base) EmitOwnCtorArgs(*cls.base, indent);
    for (auto& [_, name, _] : cls.fields) W(indent, "{},", name);
    for (auto& [_, name, _] : cls.trailing_arrays) W(indent, "{},", name);
}

void Generator::EmitOwnCtorParams(Class& cls, usz indent) {
    if (cls.base) EmitOwnCtorParams(*cls.base, indent);
    for (auto [type, name, _] : cls.fields) W(indent, "{} {},", type, name);
    for (auto [type, name, _] : cls.trailing_arrays) W(indent, "ArrayRef<{}> {},", type, name);
}

void Generator::EmitPrint() {
    W("void Printer::Print({}* e) {{", root().decorated_name);
    W(4, "switch (e->kind) {{");
    W(8, "using enum utils::Colour;");
    W(8, "using Kind = {}::{}::Kind;", namespace_, root().decorated_name);
    EmitPrintCase(root());
    W(4, "}}");
    W("}}");
}

void Generator::EmitPrintCase(Class& c) {
    // Base classes cannot be intantiated, so we don’t need to print them.
    if (c.is_base()) {
        for (auto child : c.children) EmitPrintCase(*child);
        return;
    }

    W("");
    W(8, "case Kind::{}: {{", c.name);
    W(12, "[[maybe_unused]] auto& x = static_cast<{}&>(*e);", c.decorated_name);
    W("");

    // Print the node itself.
    W(12, "fmt::print(");
    W(16, "\"{{}}{} {{}}{{}} {{}}<{{}}>\",", c.name);
    W(16, "C(Red),");
    W(16, "C(Blue),");
    W(16, "fmt::ptr(e),");
    W(16, "C(Magenta),");
    W(16, "e->loc.pos");
    W(12, ");");
    W("");

    // Add extra printout, if any.
    if (not c.extra_printout.empty()) W(12, "fmt::print({});", c.extra_printout);
    W(12, "fmt::print(\"\\n{{}}\", C(Reset));");

    // Print its children, if any.
    SmallVector<StringRef> fields;
    SmallVector<StringRef> arrays;
    auto AddFields = [&] (SmallVector<StringRef>& out, ArrayRef<Field> fields) {
        for (const auto& [type, name, _] : fields) {
            if (type.starts_with(class_prefix) and type.ends_with("*") and not type.ends_with("**"))
                out.emplace_back(name);
        }
    };

    AddFields(fields, c.fields);
    AddFields(arrays, c.trailing_arrays);

    // Print all fields.
    if (not fields.empty() or not arrays.empty()) {
        W("");
        W(12, "SmallVector<{}*, 10> fields;", root().decorated_name);
        for (auto f : fields) W(12, "if (x.{}) fields.push_back(x.{});", f, f);
        for (auto a : arrays) W(12, "if (auto a = x.{}(); not a.empty()) fields.append(a.begin(), a.end());", a);
        W(12, "PrintChildren(fields);");
    }

    W(8, "}} break;");
}

void Generator::PushUndef(StringRef undef) {
    undefs.emplace_back(undef);
}

int main(int argc, char** argv) {
    auto opts = ::srcc::detail::options::parse(argc, argv);
    Generator(*opts.get<"file">(), *opts.get<"-o">(), opts.get<"-d">());
}
