module;

#include <algorithm>
#include <llvm/ADT/SmallString.h>
#include <memory>
#include <print>
#include <ranges>
#include <srcc/Macros.hh>
#include <utility>

module srcc.frontend.parser;
import srcc.ast.printer;
using namespace srcc;

// ============================================================================
//  Parse Tree
// ============================================================================
void ParsedModule::dump() const {
    using enum utils::Colour;
    bool c = context().use_colours();
    utils::Colours C{c};

    // Print preamble.
    std::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);
    for (auto i : imports) std::print(
        "{}Import {}<{}> {}as {}{}\n",
        C(Red),
        C(Blue),
        i.linkage_name,
        C(Red),
        C(Blue),
        i.import_name
    );

    // Print content.
    for (auto s : top_level) s->dump(this, c);
}

// ============================================================================
//  Statements
// ============================================================================
void* ParsedStmt::operator new(usz size, Parser& parser) {
    return parser.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

// ============================================================================
//  Types
// ============================================================================
ParsedProcType::ParsedProcType(ParsedStmt* ret_type, ArrayRef<ParsedStmt*> params, Location loc)
    : ParsedStmt{Kind::ProcType, loc},
      num_params{u32(params.size())},
      ret_type{ret_type} {
    std::uninitialized_copy_n(params.begin(), params.size(), getTrailingObjects<ParsedStmt*>());
}

auto ParsedProcType::Create(
    Parser& parser,
    ParsedStmt* ret_type,
    ArrayRef<ParsedStmt*> params,
    Location loc
) -> ParsedProcType* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(params.size());
    auto mem = parser.allocate(size, alignof(ParsedProcType));
    return ::new (mem) ParsedProcType{ret_type, params, loc};
}

// ============================================================================
//  Expressions
// ============================================================================
ParsedBlockExpr::ParsedBlockExpr(
    ArrayRef<ParsedStmt*> stmts,
    Location location
) : ParsedStmt{Kind::BlockExpr, location},
    num_stmts{u32(stmts.size())} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<ParsedStmt*>());
}

auto ParsedBlockExpr::Create(
    Parser& parser,
    ArrayRef<ParsedStmt*> stmts,
    Location location
) -> ParsedBlockExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(stmts.size());
    auto mem = parser.allocate(size, alignof(ParsedBlockExpr));
    return ::new (mem) ParsedBlockExpr{stmts, location};
}

ParsedCallExpr::ParsedCallExpr(
    ParsedStmt* callee,
    ArrayRef<ParsedStmt*> args,
    Location location
) : ParsedStmt{Kind::CallExpr, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<ParsedStmt*>());
}

auto ParsedCallExpr::Create(
    Parser& parser,
    ParsedStmt* callee,
    ArrayRef<ParsedStmt*> args,
    Location location
) -> ParsedCallExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(args.size());
    auto mem = parser.allocate(size, alignof(ParsedCallExpr));
    return ::new (mem) ParsedCallExpr{callee, args, location};
}

ParsedDeclRefExpr::ParsedDeclRefExpr(ArrayRef<String> names, Location location)
    : ParsedStmt(Kind::DeclRefExpr, location), num_parts(u32(names.size())) {
    std::uninitialized_copy_n(names.begin(), names.size(), getTrailingObjects<String>());
}

auto ParsedDeclRefExpr::Create(
    Parser& parser,
    ArrayRef<String> names,
    Location location
) -> ParsedDeclRefExpr* {
    const auto size = totalSizeToAlloc<String>(names.size());
    auto mem = parser.allocate(size, alignof(ParsedDeclRefExpr));
    return ::new (mem) ParsedDeclRefExpr{names, location};
}

ParsedIntLitExpr::ParsedIntLitExpr(Parser& p, APInt value, Location loc)
    : ParsedStmt{Kind::IntLitExpr, loc},
      storage{p.module().integers.store_int(std::move(value))} {}

// ============================================================================
//  Declarations
// ============================================================================
ParsedProcDecl::ParsedProcDecl(
    String name,
    ParsedProcType* type,
    ArrayRef<ParsedLocalDecl*> param_decls,
    ParsedStmt* body,
    Location location
) : ParsedDecl{Kind::ProcDecl, name, location},
    body{body},
    type{type} {
    std::uninitialized_copy_n(
        param_decls.begin(),
        param_decls.size(),
        getTrailingObjects<ParsedLocalDecl*>()
    );
}

auto ParsedProcDecl::Create(
    Parser& parser,
    String name,
    ParsedProcType* type,
    ArrayRef<ParsedLocalDecl*> param_decls,
    ParsedStmt* body,
    Location location
) -> ParsedProcDecl* {
    const auto size = totalSizeToAlloc<ParsedLocalDecl*>(param_decls.size());
    auto mem = parser.allocate(size, alignof(ParsedProcDecl));
    return ::new (mem) ParsedProcDecl{name, type, param_decls, body, location};
}

// ============================================================================
//  Printer
// ============================================================================
struct ParsedStmt::Printer : PrinterBase<ParsedStmt> {
    using enum utils::Colour;
    const ParsedModule* module = nullptr;

    Printer(
        const ParsedModule* module,
        bool use_colour,
        ParsedStmt* s
    ) : PrinterBase{use_colour}, module(module) {
        Print(s);
    }

    void PrintHeader(ParsedStmt* s, StringRef name, bool full = true);
    void Print(ParsedStmt* s);
};

void ParsedStmt::Printer::PrintHeader(ParsedStmt* s, StringRef name, bool full) {
    std::print(
        "{}{} {}{} {}<{}>",
        C(Red),
        name,
        C(Blue),
        static_cast<void*>(s),
        C(Magenta),
        s->loc.pos
    );

    if (full) std::print("{}\n", C(Reset));
    else std::print(" ");
}

void ParsedStmt::Printer::Print(ParsedStmt* s) {
    switch (s->kind()) {
        case Kind::BuiltinType:
        case Kind::TemplateType:
        case Kind::ProcType:
            std::print("{}Type {}\n", C(Red), s->dump_as_type(C));
            break;

        case Kind::BlockExpr: {
            auto& b = *cast<ParsedBlockExpr>(s);
            PrintHeader(s, "BlockExpr");

            SmallVector<ParsedStmt*, 10> children;
            if (auto a = b.stmts(); not a.empty()) children.append(a.begin(), a.end());
            PrintChildren(children);
        } break;

        case Kind::BinaryExpr: {
            auto& b = *cast<ParsedBinaryExpr>(s);
            PrintHeader(s, "BinaryExpr", false);
            std::print("{}{}{}\n", C(Red), b.op, C(Reset));
            SmallVector<ParsedStmt*, 2> children{b.lhs, b.rhs};
            PrintChildren(children);
        } break;

        case Kind::CallExpr: {
            auto& c = *cast<ParsedCallExpr>(s);
            PrintHeader(s, "CallExpr");

            SmallVector<ParsedStmt*, 10> children;
            if (c.callee) children.push_back(c.callee);
            if (auto a = c.args(); not a.empty()) children.append(a.begin(), a.end());
            PrintChildren(children);
        } break;

        case Kind::DeclRefExpr: {
            auto& d = *cast<ParsedDeclRefExpr>(s);
            PrintHeader(s, "DeclRefExpr", false);
            std::print(
                "{}{}{}\n",
                C(Reset),
                utils::join(d.names(), "::"),
                C(Reset)
            );
        } break;

        case Kind::EvalExpr: {
            auto& v = *cast<ParsedEvalExpr>(s);
            PrintHeader(s, "EvalExpr");
            PrintChildren(v.expr);
        } break;

        case Kind::IntLitExpr: {
            PrintHeader(s, "IntLitExpr", false);
            auto val = cast<ParsedIntLitExpr>(s)->storage.str(false);
            std::print("{}{}{}\n", C(Magenta), val, C(Reset));
        } break;

        case Kind::MemberExpr: {
            auto& m = *cast<ParsedMemberExpr>(s);
            PrintHeader(s, "MemberExpr", false);
            std::print(
                "{}{}\n{}",
                C(Reset),
                m.member,
                C(Reset)
            );

            PrintChildren(m.base);
        } break;

        case Kind::LocalDecl: {
            auto& p = *cast<ParsedLocalDecl>(s);
            PrintHeader(s, "LocalDecl", false);
            std::print(
                "{}{}{}{}\n",
                C(Blue),
                p.name,
                p.name.empty() ? ""sv : " "sv,
                p.type->dump_as_type(C)
            );
            if (p.init) PrintChildren(p.init.get());
        } break;

        case Kind::ProcDecl: {
            auto& p = *cast<ParsedProcDecl>(s);
            PrintHeader(s, "ProcDecl", false);
            std::print(
                "{}{}{}{}\n",
                C(Green),
                p.name,
                p.name.empty() ? ""sv : " "sv,
                p.type->dump_as_type(C)
            );

            // No need to print the param decls here.
            SmallVector<ParsedStmt*, 10> children;
            if (p.body) children.push_back(p.body);
            PrintChildren(children);
        } break;

        case Kind::StrLitExpr: {
            auto& str = *cast<ParsedStrLitExpr>(s);
            PrintHeader(s, "StrLitExpr", false);
            std::print(
                "{}\"{}\"\n{}",
                C(Yellow),
                utils::Escape(str.value),
                C(Reset)
            );
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ParsedReturnExpr>(s);
            PrintHeader(s, "ReturnExpr");
            if (auto val = ret->value.get_or_null()) PrintChildren(val);
        } break;

        case Kind::UnaryExpr: {
            auto& u = *cast<ParsedUnaryExpr>(s);
            PrintHeader(s, "UnaryExpr", false);
            std::print("{}{}{}\n", C(Red), u.op, C(Reset));
            PrintChildren(u.arg);
        } break;
    }
}

void ParsedStmt::dump(bool use_colour) const {
    dump(nullptr, use_colour);
}

void ParsedStmt::dump(const ParsedModule* owner, bool use_colour) const {
    Printer(owner, use_colour, const_cast<ParsedStmt*>(this));
}

auto ParsedStmt::dump_as_type(utils::Colours C) -> std::string {
    using enum utils::Colour;
    std::string out;

    auto Append = [C, &out](this auto& Append, ParsedStmt* type) -> void {
        switch (type->kind()) {
            case Kind::BuiltinType:
                out += cast<ParsedBuiltinType>(type)->ty->print(C.use_colours);
                break;

            case Kind::ProcType: {
                auto p = cast<ParsedProcType>(type);
                out += C(Red);
                out += "proc";

                if (not p->param_types().empty()) {
                    bool first = true;
                    out += " (";

                    for (auto t : p->param_types()) {
                        if (not first) {
                            out += C(Red);
                            out += ", ";
                        }

                        first = false;
                        Append(t);
                    }

                    out += C(Red);
                    out += ")";
                }

                out += " -> ";
                Append(p->ret_type);
            } break;

            case Kind::TemplateType: {
                auto t = cast<ParsedTemplateType>(type);
                out += C(Yellow);
                out += "$";
                out += t->name;
            } break;

            default:
                out += "<invalid type>";
                break;
        }
    };

    Append(this);
    out += C(Reset);
    return out;
}

#define PARSE_TREE_NODE(node)                                                                                \
    static_assert(alignof(SRCC_CAT(Parsed, node)) < __STDCPP_DEFAULT_NEW_ALIGNMENT__, "Alignment to large"); \
    static_assert(__is_trivially_destructible(SRCC_CAT(Parsed, node)), "Parse tree nodes must be trivially destructible");

PARSE_TREE_NODE(Stmt);
#include "srcc/ParseTree.inc"

// ============================================================================
//  Parser Helpers
// ============================================================================
constexpr int BinaryOrPostfixPrecedence(Tk t) {
    switch (t) {
        case Tk::ColonColon:
            return 100'000;

        case Tk::Dot:
            return 10'000;

        case Tk::LBrack:
            return 5'000;

        case Tk::LParen:
            return 1'000;

        case Tk::As:
        case Tk::AsBang:
            return 200;

        case Tk::StarStar:
            return 100;

        case Tk::Star:
        case Tk::Slash:
        case Tk::Percent:
        case Tk::StarTilde:
        case Tk::StarVBar:
        case Tk::ColonSlash:
        case Tk::ColonPercent:
            return 95;

        case Tk::Plus:
        case Tk::PlusTilde:
        case Tk::PlusVBar:
        case Tk::Minus:
        case Tk::MinusTilde:
        case Tk::MinusVBar:
            return 90;

        // Shifts have higher precedence than logical/bitwise
        // operators so e.g. `a & 1 << 3` works properly.
        case Tk::ShiftLeft:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
            return 85;

        case Tk::Ampersand:
        case Tk::VBar:
            return 82;

        case Tk::ULt:
        case Tk::UGt:
        case Tk::ULe:
        case Tk::UGe:
        case Tk::SLt:
        case Tk::SGt:
        case Tk::SLe:
        case Tk::SGe:
            return 80;

        case Tk::EqEq:
        case Tk::Neq:
            return 75;

        case Tk::In:
            return 72;

        case Tk::And:
        case Tk::Or:
        case Tk::Xor:
            return 70;

        // Assignment has the lowest precedence.
        case Tk::Assign:
        case Tk::PlusEq:
        case Tk::PlusTildeEq:
        case Tk::PlusVBarEq:
        case Tk::MinusEq:
        case Tk::MinusTildeEq:
        case Tk::MinusVBarEq:
        case Tk::StarEq:
        case Tk::StarTildeEq:
        case Tk::StarVBarEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
            return 1;

        default:
            return -1;
    }
}

constexpr int PrefixPrecedence = 900;

constexpr bool IsRightAssociative(Tk t) {
    switch (t) {
        case Tk::StarStar:
        case Tk::Assign:
        case Tk::PlusEq:
        case Tk::MinusEq:
        case Tk::StarEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
            return true;

        default:
            return false;
    }
}

constexpr bool IsPostfix(Tk t) {
    switch (t) {
        default:
            return false;
    }
}

bool Parser::AtStartOfExpression() {
    switch (tok->type) {
        default: return false;
        case Tk::Eval:
        case Tk::Identifier:
        case Tk::Int:
        case Tk::Integer:
        case Tk::Minus:
        case Tk::MinusMinus:
        case Tk::Not:
        case Tk::Plus:
        case Tk::PlusPlus:
        case Tk::RBrace:
        case Tk::Return:
        case Tk::StringLiteral:
        case Tk::TemplateType:
        case Tk::Tilde:
        case Tk::Var:
        case Tk::Void:
            return true;
    }
}

bool Parser::Consume(Tk tk) {
    if (At(tk)) {
        ++tok;
        return true;
    }
    return false;
}

bool Parser::Consume(Location& into, Tk tk) {
    if (At(tk)) {
        into = tok->location;
        ++tok;
        return true;
    }
    return false;
}

bool Parser::ConsumeContextual(Location& into, StringRef keyword) {
    if (At(Tk::Identifier) and tok->text == keyword) {
        into = tok->location;
        ++tok;
        return true;
    }
    return false;
}

bool Parser::ConsumeOrError(Tk tk) {
    if (not Consume(tk)) {
        Error("Expected '{}'", tk);
        return false;
    }
    return true;
}

// ============================================================================
//  Parser
// ============================================================================
auto Parser::Parse(
    const File& file,
    CommentTokenCallback cb
) -> std::unique_ptr<ParsedModule> {
    Parser P{file};
    ReadTokens(P.stream, file, std::move(cb));
    P.tok = P.stream.begin();
    P.ParseFile();
    return std::move(P.mod);
}

auto Parser::ParseBlock() -> Ptr<ParsedBlockExpr> {
    auto loc = tok->location;
    ++tok;

    // Parse statements.
    SmallVector<ParsedStmt*> stmts;
    while (not At(Tk::Eof, Tk::RBrace))
        if (auto s = ParseStmt())
            stmts.push_back(s.get());

    if (not Consume(Tk::RBrace)) {
        Error("Expected '}}'");
        Note(loc, "To match this '{{'");
    }

    return ParsedBlockExpr::Create(*this, stmts, loc);
}

// <expr-decl-ref> ::= IDENTIFIER [ "::" <expr-decl-ref> ]
auto Parser::ParseDeclRefExpr() -> Ptr<ParsedDeclRefExpr> {
    auto loc = tok->location;
    SmallVector<String> strings;
    do {
        if (not At(Tk::Identifier)) {
            Error("Expected identifier after '::'");
            SkipTo(Tk::Semicolon);
            break;
        }

        strings.push_back(tok->text);
        loc = {loc, tok->location};
        ++tok;
    } while (Consume(Tk::ColonColon));
    return ParsedDeclRefExpr::Create(*this, strings, loc);
}

// This parses expressions and also some declarations (e.g.
// variable declarations)
//
// <expr> ::= <expr-block>
//          | <expr-call>
//          | <expr-decl-ref>
//          | <expr-eval>
//          | <expr-lit>
//          | <expr-member>
//          | <expr-return>
//          | <expr-prefix>
//          | <expr-binary>
//          | <expr-subscript>
auto Parser::ParseExpr(int precedence) -> Ptr<ParsedStmt> {
    Ptr<ParsedStmt> lhs;
    bool at_start = AtStartOfExpression(); // See below.
    switch (tok->type) {
        default: return Error("Expected expression");

        // <expr-prefix> ::= <prefix> <expr>
        case Tk::Minus:
        case Tk::Plus:
        case Tk::Not:
        case Tk::Tilde:
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            auto start = tok->location;
            auto op = tok->type;
            ++tok;
            auto arg = ParseExpr(PrefixPrecedence);
            if (not arg) return {};
            lhs = new (*this) ParsedUnaryExpr{op, arg.get(), false, {start, arg.get()->loc}};
        } break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-eval> ::= EVAL <expr>
        case Tk::Eval: {
            auto start = tok->location;
            ++tok;
            auto arg = ParseExpr();
            if (not arg) return {};
            lhs = new (*this) ParsedEvalExpr{arg.get(), {start, arg.get()->loc}};
        } break;

        case Tk::Identifier:
            lhs = ParseDeclRefExpr();
            break;

        // <expr-lit> ::= STRING-LITERAL | INTEGER
        case Tk::StringLiteral:
            lhs = new (*this) ParsedStrLitExpr{tok->text, tok->location};
            ++tok;
            break;

        // <expr-lit> ::= STRING-LITERAL | INTEGER
        case Tk::Integer:
            lhs = new (*this) ParsedIntLitExpr{*this, tok->integer, tok->location};
            ++tok;
            break;

        // <expr-return>   ::= RETURN [ <expr> ]
        case Tk::Return: {
            auto loc = tok->location;
            ++tok;

            Ptr<ParsedStmt> value;
            if (AtStartOfExpression()) {
                value = ParseExpr();
                if (not value) return {};
            }

            if (value.present()) loc = {loc, value.get()->loc};
            lhs = new (*this) ParsedReturnExpr{value, loc};
        } break;

        case Tk::Int:
        case Tk::Void:
        case Tk::Var:
        case Tk::TemplateType:
            lhs = ParseType();
            break;
    }

    // There was an error.
    if (lhs.invalid()) return nullptr;

    // I keep forgetting to add new tokens to AtStartOfExpression,
    // so this is here to make sure I don’t forget.
    Assert(at_start, "Forgot to add a token to AtStartOfExpression");

    // If the next token is an identifier, then this is a declaration.
    if (At(Tk::Identifier)) return ParseVarDecl(lhs.get());

    // Big operator parse loop.
    while (
        BinaryOrPostfixPrecedence(tok->type) > precedence or
        (BinaryOrPostfixPrecedence(tok->type) == precedence and IsRightAssociative(tok->type))
    ) {
        // Some 'operators' have precedence, but require additional logic.
        switch (tok->type) {
            default: break;

            // <expr-call> ::= <expr> "(" [ <call-args> ] ")"
            // <call-args> ::= <expr> { "," <expr> } [ "," ]
            case Tk::LParen: {
                ++tok;
                SmallVector<ParsedStmt*> args;
                while (not At(Tk::RParen)) {
                    if (auto arg = ParseExpr()) {
                        args.push_back(arg.get());
                        if (not Consume(Tk::Comma)) break;
                    } else {
                        SkipTo(Tk::RParen);
                        break;
                    }
                }

                auto end = tok->location;
                ConsumeOrError(Tk::RParen);
                lhs = ParsedCallExpr::Create(*this, lhs.get(), args, {lhs.get()->loc, end});
                continue;
            }

            // <expr=subscript> ::= <expr> "[" <expr> "]"
            case Tk::LBrack: {
                ++tok;
                auto index = ParseExpr();
                if (not index) return {};
                auto end = tok->location;
                ConsumeOrError(Tk::RBrack);
                lhs = new (*this) ParsedBinaryExpr{Tk::LBrack, lhs.get(), index.get(), {lhs.get()->loc, end}};
                continue;
            }

            // <expr-member> ::= <expr> "." IDENTIFIER
            case Tk::Dot: {
                ++tok;
                if (not At(Tk::Identifier)) {
                    Error("Expected identifier after '.'");
                    SkipTo(Tk::Semicolon);
                    return {};
                }

                lhs = new (*this) ParsedMemberExpr(lhs.get(), tok->text, {lhs.get()->loc, tok->location});
                ++tok;
                continue;
            }
        }

        auto op = tok->type;
        auto end = tok->location;
        ++tok;
        if (IsPostfix(op)) {
            lhs = new (*this) ParsedUnaryExpr{op, lhs.get(), true, {lhs.get()->loc, end}};
        } else {
            auto rhs = ParseExpr(BinaryOrPostfixPrecedence(op));
            if (not rhs) return {};
            lhs = new (*this) ParsedBinaryExpr{op, lhs.get(), rhs.get(), {lhs.get()->loc, rhs.get()->loc}};
        }
    }

    return lhs;
}

// <file> ::= <preamble> { <stmt> }
void Parser::ParseFile() {
    ParsePreamble();
    while (not At(Tk::Eof))
        if (auto stmt = ParseStmt())
            mod->top_level.push_back(stmt.get());
}

// <header> ::= "program" <module-name> ";"
// <module-name> ::= IDENTIFIER
void Parser::ParseHeader() {
    if (not ConsumeContextual(mod->program_or_module_loc, "program")) {
        Error("Expected 'program' directive at start of file");
        return SkipPast(Tk::Semicolon);
    }

    if (not At(Tk::Identifier)) {
        Error("Expected identifier in 'program' directive");
        return SkipPast(Tk::Semicolon);
    }

    mod->name = tok->text;
    ++tok;
    ConsumeOrError(Tk::Semicolon);
}

// <import> ::= IMPORT CXX-HEADER-NAME AS IDENTIFIER ";"
void Parser::ParseImport() {
    Location import_loc;
    Assert(Consume(import_loc, Tk::Import), "Not at 'import'?");
    if (not At(Tk::CXXHeaderName)) {
        Error("Expected C++ header name after 'import'");
        SkipPast(Tk::Semicolon);
        return;
    }

    // Save name for later.
    auto linkage_name = tok->text;
    ++tok;

    // Read import name.
    ExpectAndConsume(Tk::As, "Syntax for header imports is `import <header> as name`");
    if (not At(Tk::Identifier)) {
        Error("Expected identifier after 'as' in import directive");
        return;
    }

    // Warn about duplicate imports.
    auto FindExisting = [&](auto& i) { return i.linkage_name == linkage_name and i.import_name == tok->text; };
    Location loc = {import_loc, tok->location};
    if (
        auto it = rgs::find_if(mod->imports, FindExisting);
        it != mod->imports.end()
    ) {
        Warn(loc, "Duplicate import ignored");
        Note(it->loc, "Previous import was here");
    } else {
        mod->imports.emplace_back(linkage_name, tok->text, loc);
    }

    ++tok;
    ConsumeOrError(Tk::Semicolon);
}

// <decl-var>   ::= <type> IDENTIFIER [ "=" <expr> ] ";"
auto Parser::ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedStmt> {
    auto decl = new (*this) ParsedLocalDecl(
        tok->text,
        type,
        {type->loc, tok->location}
    );

    // Parse the optional initialiser.
    ++tok;
    if (Consume(Tk::Assign)) decl->init = ParseExpr();

    // We don’t allow declaration groups such as 'int a, b;'.
    if (At(Tk::Comma)) {
        Error("A declaration must declare a single variable");
        SkipTo(Tk::Semicolon);
    }
    return decl;
}

// <preamble> ::= <header> { <import> }
void Parser::ParsePreamble() {
    ParseHeader();
    while (At(Tk::Import)) ParseImport();
}

// <decl-proc>  ::= PROC IDENTIFIER <signature> <proc-body>
// <signature>  ::= [ <proc-args> ] [ "->" <type> ]
// <proc-args>  ::= "(" [ <param-decl> { "," <param-decl> } [ "," ] ] ")"
// <proc-body>  ::= <expr-block> | "=" <expr> ";"
// <param-decl> ::= <type> [ IDENTIFIER ]
auto Parser::ParseProcDecl() -> Ptr<ParsedProcDecl> {
    // Yeet 'proc'.
    auto loc = tok->location;
    ++tok;

    // Parse name.
    if (not At(Tk::Identifier)) return Error("Expected identifier");
    auto name = tok->text;
    ++tok;

    // Parse signature.
    SmallVector<ParsedStmt*, 10> param_types;
    SmallVector<ParsedLocalDecl*, 10> param_decls;
    if (Consume(Tk::LParen)) {
        while (not At(Tk::RParen)) {
            auto ty = ParseType();
            if (not ty) {
                SkipTo(Tk::RParen);
                break;
            }

            String name;
            Location end = ty.get()->loc;
            if (At(Tk::Identifier)) {
                name = tok->text;
                end = tok->location;
                ++tok;
            }

            param_types.push_back(ty.get());
            param_decls.push_back(new (*this) ParsedLocalDecl{
                name,
                ty.get(),
                {ty.get()->loc, end}
            });

            if (not Consume(Tk::Comma)) break;
        }

        if (not Consume(Tk::RParen)) return Error("Expected ')'");
    }

    // Return Type.
    Ptr<ParsedStmt> ret;
    if (Consume(Tk::RArrow)) ret = ParseType();

    // If we failed to parse a return type, or if there was
    // none, just default to void instead, or deduce the type
    // if this is a '= <expr>' declaration.
    if (ret.invalid()) {
        ret = new (*this) ParsedBuiltinType(
            At(Tk::Assign) ? Types::DeducedTy.ptr() : Types::VoidTy.ptr(),
            loc
        );
    }

    // Parse body.
    Ptr<ParsedStmt> body;
    if (Consume(Tk::Assign)) {
        body = ParseExpr();
        if (not Consume(Tk::Semicolon)) Error("Expected ';'");
    } else {
        body = ParseBlock();
    }

    if (not body) return {};
    return ParsedProcDecl::Create(
        *this,
        name,
        ParsedProcType::Create(*this, ret.get(), param_types, loc),
        param_decls,
        body.get(),
        loc
    );
}

// <stmt>  ::= [ <expr> ] ";"
//           | <expr-block>
//           | <decl>
//           | EVAL <stmt>
auto Parser::ParseStmt() -> Ptr<ParsedStmt> {
    auto loc = tok->location;
    switch (tok->type) {
        case Tk::Eval: {
            ++tok;
            auto arg = ParseStmt();
            if (not arg) return {};
            return new (*this) ParsedEvalExpr{arg.get(), loc};
        }

        case Tk::LBrace: return ParseBlock();
        case Tk::Proc: return ParseProcDecl();

        // Fall through to the expression parser, but remember
        // to eat a semicolon afterwards if we don’t parse a block
        // expression.
        default:
            if (auto res = ParseExpr()) {
                if (not Consume(Tk::Semicolon)) {
                    Error("Expected ';'");
                    SkipPast(Tk::Semicolon);
                }
                return res;
            }

            // Expression parser should have already complained here;
            // just yeet everything up tot he next semicolon.
            SkipPast(Tk::Semicolon);
            return {};
    }
}

// <type> ::= <type-prim> | TEMPLATE-TYPE | <expr-decl-ref>
auto Parser::ParseType() -> Ptr<ParsedStmt> {
    auto Builtin = [&](BuiltinType* ty) {
        auto loc = tok->location;
        ++tok;
        return new (*this) ParsedBuiltinType(ty, loc);
    };

    switch (tok->type) {
        default: return Error("Expected type");

        // <type-prim> ::= INT | VOID | VAR
        case Tk::Int: return Builtin(Types::IntTy.ptr());
        case Tk::Void: return Builtin(Types::VoidTy.ptr());
        case Tk::Var: return Builtin(Types::DeducedTy.ptr());

        // TEMPLATE-TYPE
        case Tk::TemplateType: {
            // Drop the '$' from the type.
            auto ty = new (*this) ParsedTemplateType(tok->text.drop(), tok->location);
            ++tok;
            return ty;
        }

        // <expr-decl-ref>
        case Tk::Identifier:
            return ParseDeclRefExpr();
    }
}

