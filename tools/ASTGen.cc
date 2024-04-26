#include <clopts.hh>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/std.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <ranges>
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
    std::string type;          ///< C++ type of the field.
    std::string name;          ///< C++ name of the field.
    std::string default_value; ///< Default parameter value.
    bool final;                ///< Put this at the very end in the factory function and ctor.
    bool trailing;             ///< Trailing array data using `llvm::TrailingObjects`.
    Location loc;              ///< Location in the .ast file.

    /// The type to use in declarations.
    auto decl_type() const -> std::string {
        return trailing ? fmt::format("ArrayRef<{}>", type) : type;
    }
};

struct Class {
    SRCC_IMMOVABLE(Class);

    Class* base; // May be null if this is the root.
    std::string name;
    std::string decorated_name;
    std::string extra_printout;
    SmallVector<std::string> friends;
    SmallVector<Class*> children;
    SmallVector<Field, 10> fields;
    DenseMap<const Field*, std::string> default_init_fields;

    Class(StringRef prefix, std::string name, Class* base)
        : base(base),
          name(std::move(name)),
          decorated_name(std::string(prefix) + this->name) {
        if (base) default_init_fields = base->default_init_fields;
    }

    /// Check if a field is defaulted in this class.
    bool defaulted(const Field& f) {
        return not f.default_value.empty() or default_init_fields.contains(&f);
    }

    /// Get the default value of a field in this class.
    auto default_value(const Field& f) -> std::string_view {
        auto it = default_init_fields.find(&f);
        if (it != default_init_fields.end()) return it->second;
        return f.default_value;
    }

    /// Get all final fields.
    auto final() {
        return fields | std::views::filter([](const Field& f) { return f.final; });
    }

    /// Check whether this has any derived classes.
    bool is_base() const { return not children.empty(); }

    /// Get all fields that aren’t final.
    auto non_final() {
        return fields | std::views::filter([](const Field& f) { return not f.final; });
    }

    /// Get all non-trailing fields.
    auto non_trailing() {
        return fields | std::views::filter([](const Field& f) { return not f.trailing; });
    }

    /// Get all trailing fields.
    auto trailing() {
        return fields | std::views::filter([](const Field& f) { return f.trailing; });
    }
};

enum struct KindPolicy : u8 {
    Never,
    IfBase,
    Always,
};

enum struct ParameterOrder : u8 {
    Regular,
    Defaulted,
};

struct PrintingPolicy {
    bool default_val;     ///< Emit default value.
    bool type;            ///< Emit type.
    bool trailing_comma;  ///< Emit trailing comma.
    KindPolicy kind;      ///< Emit kind.
    ParameterOrder order; ///< Order of parameters.
};

constexpr PrintingPolicy CtorParams{
    .default_val = false,
    .type = true,
    .trailing_comma = false,
    .kind = KindPolicy::IfBase,
    .order = ParameterOrder::Regular,
};

constexpr PrintingPolicy CtorInitBase{
    .default_val = false,
    .type = false,
    .trailing_comma = true,
    .kind = KindPolicy::Never,
    .order = ParameterOrder::Regular,
};

constexpr PrintingPolicy CtorCall{
    .default_val = false,
    .type = false,
    .trailing_comma = true,
    .kind = KindPolicy::IfBase,
    .order = ParameterOrder::Regular,
};

constexpr PrintingPolicy FactoryDecl{
    .default_val = true,
    .type = true,
    .trailing_comma = false,
    .kind = KindPolicy::Never,
    .order = ParameterOrder::Defaulted,
};

constexpr PrintingPolicy FactoryDef{
    .default_val = false,
    .type = true,
    .trailing_comma = false,
    .kind = KindPolicy::Never,
    .order = ParameterOrder::Defaulted,
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

    DenseMap<const Field*, std::string>* field_overrides = nullptr;

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

    bool At(std::same_as<Tk> auto... tks);
    bool Consume(std::same_as<Tk> auto... tks);
    void DoExpect(std::same_as<Tk> auto... tks);
    void ExpectAndConsume(Tk t);
    auto ExpectText() -> std::string;
    auto GetClass(std::string_view name) -> Class&;
    auto Here() -> Location;
    bool Kw(std::string_view text);
    auto Root() -> Class& { return *classes.front(); }

    // Lexer/Parser.
    void AddConstant(Class& cls, std::string_view name, std::string_view value);
    void AddDefaultInit(Class& cls, std::string name, std::string value);
    void Debug();
    void Next();
    void NextChar();
    void Parse();
    void ParseClass();
    void ParseClassBody(Class& cls);
    void ParseClassMember(Class& cls);
    void ParseProperty(std::string& prop, std::string_view name);
    void SkipLine();

    // Emitter.
    void Emit();
    void EmitClassDef(Class& cls);
    void EmitClassImpl(Class& cls);
    void EmitForwardDecl(Class& cls);
    void EmitGuarded(std::string_view GuardName, auto cb);
    void EmitEnumerator(Class& cls);
    void EmitParams(Class& cls, usz indent, PrintingPolicy pp);
    void EmitParamsImpl(Class& most_derived, Class& cls, usz indent, PrintingPolicy pp, auto filter);
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

    if (name == "$friend") {
        cls.friends.push_back(std::string{value});
        return;
    }

    Error("Unknown constant field: '{}'", name);
}

void Generator::AddDefaultInit(Class& cls, std::string name, std::string value) {
    for (Class* base = cls.base; base; base = base->base) {
        auto it = std::ranges::find(base->fields, name, &Field::name);
        if (it == base->fields.end()) continue;
        cls.default_init_fields[&*it] = std::move(value);
        return;
    }

    Error("No field named '{}' in any of '{}'’s bases!", name, cls.name);
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

    fmt::print("root {}\n", N(Root().name));
    if (not class_prefix.empty()) fmt::print("prefix {}\n", N(class_prefix));
    if (not guard_prefix.empty()) fmt::print("guard {}\n", N(guard_prefix));
    if (not namespace_.empty()) fmt::print("namespace {}\n", N(namespace_));
    for (auto& c : classes | std::views::drop(1)) {
        fmt::print("\nclass {}", N(c->name));
        if (c->base) fmt::print(" : {}", N(c->base->name));
        fmt::print(" {{\n");
        for (auto& f : c->fields) {
            fmt::print("    {}{} {}", N(f.type), f.trailing ? "[]" : "", N(f.name));
            if (not f.default_value.empty()) fmt::print(" = [{{ {} }}]", N(f.default_value));
            fmt::print("\n");
        }

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
    if (not Kw("root")) Error("Expected 'root' class");
    classes.push_back(std::make_unique<Class>(class_prefix, ExpectText(), nullptr));
    if (At(Tk::LBrace)) ParseClassBody(Root());
    while (Kw("class")) ParseClass();
    if (not At(Tk::Eof)) Error("Expected 'class'");
}

void Generator::ParseClass() {
    auto name = ExpectText();
    auto& base = Consume(Tk::Colon) ? GetClass(ExpectText()) : Root();
    auto& cls = *classes.emplace_back(std::make_unique<Class>(class_prefix, name, &base));
    base.children.push_back(&cls);
    ParseClassBody(cls);
}

void Generator::ParseClassBody(Class& cls) {
    ExpectAndConsume(Tk::LBrace);
    while (not At(Tk::RBrace, Tk::Eof)) ParseClassMember(cls);
    ExpectAndConsume(Tk::RBrace);
}

void Generator::ParseClassMember(Class& cls) {
    auto final = Kw("final");
    auto loc = Here();
    auto type_or_key = ExpectText();

    // Constant field.
    if (Consume(Tk::Colon)) {
        AddConstant(cls, type_or_key, ExpectText());
        return;
    }

    // Default initialiser.
    if (Consume(Tk::Equals)) {
        AddDefaultInit(cls, type_or_key, ExpectText());
        return;
    }

    // Regular member, optionally with a default value.
    auto trailing = Consume(Tk::Trailing);
    std::string name = ExpectText();
    std::string default_value;
    if (Consume(Tk::Equals)) default_value = ExpectText();
    cls.fields.emplace_back(
        std::move(type_or_key),
        std::move(name),
        std::move(default_value),
        final,
        trailing,
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
        for (auto c : Root().children) EmitForwardDecl(*c);
    });

    // Emit Kind enum.
    EmitGuarded("ENUMERATORS", [&] {
        for (auto& c : Root().children) EmitEnumerator(*c);
    });

    // Emit all classes.
    EmitGuarded("CLASSES", [&] {
        for (auto& c : Root().children) EmitClassDef(*c);
    });

    // Emit the implementation of all classes.
    EmitGuarded("IMPL", [&] {
        for (auto& c : Root().children) EmitClassImpl(*c);
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
    field_overrides = &cls.default_init_fields;

    // Trailing data is not allowed in a base class because it would collide
    // with the data of any derived class.
    const bool trailing = not cls.trailing().empty();
    if (cls.is_base() and trailing)
        Error(cls.trailing().begin()->loc, "Base class may not have trailing data!");

    // Write class header.
    Inline(
        "class {}::{}{} : public {}",
        namespace_,
        cls.decorated_name,
        cls.is_base() ? "" : " final",
        cls.base->decorated_name
    );

    // Add trailing objects here.
    if (trailing) {
        Inline(", llvm::TrailingObjects<{}", cls.decorated_name);
        for (auto& f : cls.trailing()) Inline(",\n    {}", f.type);
        Inline("\n>");
    }

    W(" {{");
    if (trailing or not cls.friends.empty()) {
        if (trailing) W(4, "friend TrailingObjects;");
        for (auto& f : cls.friends) W(4, "friend {};", f);
        W("");
    }

    // Emit fields.
    if (not cls.non_trailing().empty()) {
        W("public:");
        for (auto& f : cls.non_trailing()) W(4, "{} {};", f.type, f.name);

        // Don’t emit an extra 'private' if we have no trailing objects and we’re a base.
        if (trailing or not cls.is_base())
            W("\nprivate:");
    }

    // Emit trailing object helpers.
    if (trailing) {
        for (auto& f : cls.trailing()) W(4, "const u32 num_{};", f.name);
        W("");
        for (auto& f : cls.trailing()) W(
            4,
            "auto numTrailingObjects(OverloadToken<{}>) -> usz {{ return num_{}; }}",
            f.type,
            f.name
        );
        W("");
    }

    // Emit constructor. The constructor of a base class should be protected.
    if (cls.is_base()) W("\nprotected:");
    W(4, "{}(", cls.decorated_name);
    EmitParams(cls, 8, CtorParams);
    W(4, ");");
    W("");

    // Factory function.
    if (not cls.is_base()) {
        W("public:");
        W(4, "static auto Create(");
        W(8, "{}& $,", context_name);
        EmitParams(cls, 8, FactoryDecl);
        W(4, ") -> {}*;", cls.decorated_name);
        W("");

        // Trailing object accessors. These only exist if the class is not a base.
        for (auto& f : cls.trailing()) W(
            4,
            "[[nodiscard]] auto {}() -> ArrayRef<{}> {{ return {{getTrailingObjects<{}>(), num_{}}}; }}",
            f.name,
            f.type,
            f.type,
            f.name
        );

        if (not cls.trailing().empty()) W("");
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
            W(4, "static bool classof(const {}* e) {{ return e->kind() == Kind::{}; }}", Root().decorated_name, first);
        } else {
            W(4, "static bool classof(const {}* e) {{", Root().decorated_name);
            W(8, " return e->kind() >= Kind::{} and e->kind() <= Kind::{};", first, last);
            W(4, "}}");
        }
    } else {
        W(4, "static bool classof(const {}* e) {{ return e->kind() == Kind::{}; }}", Root().decorated_name, cls.name);
    }
    W("}};");
    W("");

    // Emit children.
    for (auto ch : cls.children) EmitClassDef(*ch);
}

void Generator::EmitClassImpl(Class& cls) {
    field_overrides = &cls.default_init_fields;

    // Emit ctor.
    W("srcc::{}::{}(", cls.decorated_name, cls.decorated_name);
    EmitParams(cls, 4, CtorParams);
    W(") : {} {{", cls.base->decorated_name);
    W(8, "{},", cls.is_base() ? "kind" : fmt::format("Kind::{}", cls.name));
    EmitParams(*cls.base, 8, CtorInitBase);
    Inline("    }}");

    // Initialise our fields.
    for (auto& f : cls.non_trailing()) Inline("\n  , {}{{{}}}", f.name, f.name);
    for (auto& f : cls.trailing()) Inline("\n  , num_{}{{u32({}.size())}}", f.name, f.name);

    // Emit ctor body.
    W(" {{");
    W(4, "static_assert(std::is_trivially_destructible_v<decltype(*this)>);");
    for (auto& f : cls.trailing()) W(
        4,
        "std::uninitialized_copy_n({}.begin(), {}.size(), getTrailingObjects<{}>());",
        f.name,
        f.name,
        f.type
    );
    W("}}");
    W("");

    // Emit factory function.
    if (not cls.is_base()) {
        W("auto srcc::{}::Create(", cls.decorated_name);
        W(4, "{}& $,", context_name);
        EmitParams(cls, 4, FactoryDef);
        W(") -> {}* {{", cls.decorated_name);

        // Calculate size to allocate.
        if (not cls.trailing().empty()) {
            W(4, "const auto $size = totalSizeToAlloc<");
            bool first = true;
            for (auto& f : cls.trailing()) {
                if (first) first = false;
                else Inline(",\n");
                W(8, "{}", f.type);
            }
            W("    >(");
            first = true;
            for (auto& f : cls.trailing()) {
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
            cls.trailing().empty() ? fmt::format("sizeof({})", cls.decorated_name) : "$size",
            cls.decorated_name
        );

        // Construct.
        W(4, "return new ($mem) {}({}", cls.decorated_name, cls.is_base() ? "kind, " : "");
        EmitParams(cls, 8, CtorCall);
        W(4, ");");
        W("}}");
        W("");
    }

    // Emit children.
    for (auto ch : cls.children) EmitClassImpl(*ch);
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
void Generator::EmitEnumerator(Class& cls) {
    if (not cls.is_base()) W("{},", cls.name);
    for (auto& ch : cls.children) EmitEnumerator(*ch);
}

void Generator::EmitParamsImpl(Class& most_derived, Class& cls, usz indent, PrintingPolicy pp, auto filter) {
    if (cls.base) EmitParamsImpl(most_derived, *cls.base, indent, pp, filter);
    for (Field& f : std::invoke(filter, cls)) {
        out.append(indent, ' ');
        if (pp.type) Inline("{} ", f.decl_type());
        out += f.name;
        if (pp.default_val) {
            auto d = most_derived.default_value(f);
            if (not d.empty()) Inline(" = {}", d);
        }
        out += ",\n";
    }
}

void Generator::EmitParams(Class& cls, usz indent, PrintingPolicy pp) {
    // There are 2 parameter orders: *defaulted* parameter order and
    // *regular* parameter order. Note: *declaration order* of fields
    // is different still, but it’s much simpler so we don’t have to
    // take that into account here.
    //
    // *Defaulted* parameter order goes as follows:
    //
    //   0. Kind.
    //   1. Non-defaulted non-final fields.
    //   2. Non-defaulted final fields.
    //   3. Defaulted non-final fields.
    //   4. Defaulted final fields.
    //
    // Note that, even if a parameter wasn’t defined with a default
    // value, it can acquire one by supplying a default value in a
    // derived class; such parameters must be treated as defaulted
    // in any derived class where they have a default parameter.
    //
    // By contrast, *regular* parameter order goes as follows:
    //
    //   0. Kind.
    //   1. Non-final fields.
    //   2. Final fields.
    //
    // We need to maintain the parameter order irrespective of whether
    // we’re actually printing the default parameters or not.
    using std::views::filter;
    static const auto DefaultedIn = [](Class& cls) { return [&cls](Field& f) { return cls.defaulted(f); }; };
    auto Emit = [&](auto cb) { EmitParamsImpl(cls, cls, indent, pp, cb); };

    // Kind always goes first.
    if (pp.kind == KindPolicy::Always or (pp.kind == KindPolicy::IfBase and cls.is_base())) {
        if (cls.is_base()) {
            out.append(indent, ' ');
            if (pp.type) out.append("Kind ");
            W("kind,");
        } else {
            W("Kind::{},", cls.name);
        }
    }

    // Then the parameters, in order.
    if (pp.order == ParameterOrder::Defaulted) {
        Emit([&cls](Class& c) { return c.non_final() | filter(utils::Not(DefaultedIn(cls))); });
        Emit([&cls](Class& c) { return c.final() | filter(utils::Not(DefaultedIn(cls))); });
        Emit([&cls](Class& c) { return c.non_final() | filter(DefaultedIn(cls)); });
        Emit([&cls](Class& c) { return c.final() | filter(DefaultedIn(cls)); });
    } else {
        Emit(&Class::non_final);
        Emit(&Class::final);
    }

    // Yeet the trailing comma if we don’t want it.
    if (out.ends_with(",\n") and not pp.trailing_comma) {
        out.resize(out.size() - 2);
        out += '\n';
    }
}

void Generator::EmitPrint() {
    W("void Printer::Print({}* e) {{", Root().decorated_name);
    W(4, "switch (e->kind()) {{");
    W(8, "using enum utils::Colour;");
    W(8, "using Kind = {}::{}::Kind;", namespace_, Root().decorated_name);
    EmitPrintCase(Root());
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
    auto AddFields = [&](SmallVector<StringRef>& out, auto fields) {
        for (const auto& f : fields) {
            if (f.type.starts_with(class_prefix) and f.type.ends_with("*") and not f.type.ends_with("**"))
                out.emplace_back(f.name);
        }
    };

    AddFields(fields, c.non_trailing());
    AddFields(arrays, c.trailing());

    // Print all fields.
    if (not fields.empty() or not arrays.empty()) {
        W("");
        W(12, "SmallVector<{}*, 10> fields;", Root().decorated_name);
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
