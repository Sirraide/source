module;

#include <algorithm>
#include <llvm/ADT/SmallString.h>
#include <memory>
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
    fmt::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);
    for (auto i : imports) fmt::print(
        "{}Import {}<{}> {}as {}{}\n",
        C(Red),
        C(Blue),
        i.linkage_name,
        C(Red),
        C(Blue),
        i.import_name
    );

    // Print content.
    for (auto s : top_level) s->dump(c);
}

void* ParsedExpr::operator new(usz size, Parser& parser) {
    return parser.Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

ParsedBlockExpr::ParsedBlockExpr(
    ArrayRef<ParsedExpr*> stmts,
    Location location
) : ParsedExpr{Kind::BlockExpr, location},
    num_stmts{u32(stmts.size())} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<ParsedExpr*>());
}

auto ParsedBlockExpr::Create(
    Parser& parser,
    ArrayRef<ParsedExpr*> stmts,
    Location location
) -> ParsedBlockExpr* {
    const auto size = totalSizeToAlloc<ParsedExpr*>(stmts.size());
    auto mem = parser.Allocate(size, alignof(ParsedBlockExpr));
    return ::new (mem) ParsedBlockExpr{stmts, location};
}

ParsedCallExpr::ParsedCallExpr(
    ParsedExpr* callee,
    ArrayRef<ParsedExpr*> args,
    Location location
) : ParsedExpr{Kind::CallExpr, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<ParsedExpr*>());
}

auto ParsedCallExpr::Create(
    Parser& parser,
    ParsedExpr* callee,
    ArrayRef<ParsedExpr*> args,
    Location location
) -> ParsedCallExpr* {
    const auto size = totalSizeToAlloc<ParsedExpr*>(args.size());
    auto mem = parser.Allocate(size, alignof(ParsedCallExpr));
    return ::new (mem) ParsedCallExpr{callee, args, location};
}

ParsedDeclRefExpr::ParsedDeclRefExpr(ArrayRef<String> names, Location location)
    : ParsedExpr(Kind::DeclRefExpr, location), num_parts(u32(names.size())) {
    std::uninitialized_copy_n(names.begin(), names.size(), getTrailingObjects<String>());
}

auto ParsedDeclRefExpr::Create(
    Parser& parser,
    ArrayRef<String> names,
    Location location
) -> ParsedDeclRefExpr* {
    const auto size = totalSizeToAlloc<String>(names.size());
    auto mem = parser.Allocate(size, alignof(ParsedDeclRefExpr));
    return ::new (mem) ParsedDeclRefExpr{names, location};
}

struct ParsedExpr::Printer : PrinterBase<ParsedExpr> {
    Printer(bool use_colour, ParsedExpr* E) : PrinterBase{use_colour} { Print(E); }
    void Print(ParsedExpr* E);
};

void ParsedExpr::Printer::Print(ParsedExpr* e) {
    switch (e->kind()) {
        using enum utils::Colour;
        case Kind::BlockExpr: {
            auto& b = *cast<ParsedBlockExpr>(e);
            fmt::print(
                "{}BlockExpr {}{} {}<{}>\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset)
            );

            SmallVector<ParsedExpr*, 10> fields;
            if (auto a = b.stmts(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::CallExpr: {
            auto& c = *cast<ParsedCallExpr>(e);
            fmt::print(
                "{}CallExpr {}{} {}<{}>\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset)
            );

            SmallVector<ParsedExpr*, 10> fields;
            if (c.callee) fields.push_back(c.callee);
            if (auto a = c.args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::DeclRefExpr: {
            auto& d = *cast<ParsedDeclRefExpr>(e);
            fmt::print(
                "{}DeclRefExpr {}{} {}<{}> {}{}\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset),
                fmt::join(d.names(), "::"),
                C(Reset)
            );
        } break;

        case Kind::EvalExpr: {
            auto& v = *cast<ParsedEvalExpr>(e);
            fmt::print(
                "{}EvalExpr {}{} {}<{}>\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset)
            );

            PrintChildren(v.expr);
        } break;

        case Kind::StrLitExpr: {
            auto& s = *cast<ParsedStrLitExpr>(e);
            fmt::print(
                "{}StrLitExpr {}{} {}<{}> {}\"{}\"\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Yellow),
                utils::Escape(s.value),
                C(Reset)
            );
        } break;

        case Kind::MemberExpr: {
            auto& m = *cast<ParsedMemberExpr>(e);
            fmt::print(
                "{}MemberExpr {}{} {}<{}> {}{}\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset),
                m.member,
                C(Reset)
            );

            PrintChildren(m.base);
        } break;

        case Kind::ProcDecl: {
            auto& p = *cast<ParsedProcDecl>(e);
            fmt::print(
                "{}ProcDecl {}{} {}<{}>\n{}",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Reset)
            );

            SmallVector<ParsedExpr*, 10> fields;
            if (p.body) fields.push_back(p.body);
            PrintChildren(fields);
        } break;
    }
}

void ParsedExpr::dump(bool use_colour) const {
    Printer(use_colour, const_cast<ParsedExpr*>(this));
}

#define PARSE_TREE_NODE(node)                                                                                \
    static_assert(alignof(SRCC_CAT(Parsed, node)) < __STDCPP_DEFAULT_NEW_ALIGNMENT__, "Alignment to large"); \
    static_assert(__is_trivially_destructible(SRCC_CAT(Parsed, node)), "Parse tree nodes must be trivially destructible");

PARSE_TREE_NODE(Expr);
#include "srcc/ParseTree.inc"

// ============================================================================
//  Parser Helpers
// ============================================================================
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
auto Parser::Parse(const File& file, CommentTokenCallback cb) -> std::unique_ptr<ParsedModule> {
    Parser P{file};
    P.ReadTokens(file, std::move(cb));
    P.ParseFile();
    return std::move(P.mod);
}

auto Parser::ParseBlock() -> Ptr<ParsedBlockExpr> {
    auto loc = tok->location;
    ++tok;

    // Parse statements.
    SmallVector<ParsedExpr*> stmts;
    while (not At(Tk::Eof, Tk::RBrace))
        if (auto s = ParseStmt())
            stmts.push_back(s.get());
    ConsumeOrError(Tk::RBrace);
    return ParsedBlockExpr::Create(*this, stmts, loc);
}

// <expr> ::= <decl-proc>
//          | <expr-block>
//          | <expr-call>
//          | <expr-decl-ref>
//          | <expr-lit>
//          | <expr-member>
auto Parser::ParseExpr() -> Ptr<ParsedExpr> {
    Ptr<ParsedExpr> lhs;
    switch (tok->type) {
        default: return Error("Expected expression");

        // <decl-proc> ::= PROC IDENTIFIER <expr-block>
        case Tk::Proc: {
            auto loc = tok->location;
            ++tok;

            if (not At(Tk::Identifier)) return Error("Expected identifier");
            auto name = tok->text;
            ++tok;

            auto body = ParseBlock();
            if (not body) return {};
            lhs = new (*this) ParsedProcDecl{name, body.get(), loc};
        } break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-decl-ref> ::= IDENTIFIER [ "::" <expr-decl-ref> ]
        case Tk::Identifier: {
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
            lhs = ParsedDeclRefExpr::Create(*this, strings, loc);
        } break;

        // <expr-lit> ::= STRING-LITERAL
        case Tk::StringLiteral: {
            lhs = new (*this) ParsedStrLitExpr{tok->text, tok->location};
            ++tok;
        } break;
    }

    // Big operator parse loop.
    // TODO: precedence.
    while (At(Tk::LParen, Tk::Dot)) {
        switch (tok->type) {
            default: break;

            // <expr-call> ::= <expr> "(" [ <call-args> ] ")"
            // <call-args> ::= <expr> { "," <expr> } [ "," ]
            case Tk::LParen: {
                ++tok;
                SmallVector<ParsedExpr*> args;
                while (not At(Tk::RParen)) {
                    if (auto arg = ParseExpr()) {
                        args.push_back(arg.get());
                        if (not Consume(Tk::Comma)) break;
                    } else {
                        SkipTo(Tk::RParen);
                        break;
                    }
                }

                ConsumeOrError(Tk::RParen);
                lhs = ParsedCallExpr::Create(*this, lhs.get(), args, {lhs.get()->loc});
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

        Unreachable("Invalid operator: {}", tok->type);
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
        return SkipTo(Tk::Semicolon);
    }

    if (not At(Tk::Identifier)) {
        Error("Expected identifier in 'program' directive");
        return SkipTo(Tk::Semicolon);
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
        SkipTo(Tk::Semicolon);
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

// <preamble> ::= <header> { <import> }
void Parser::ParsePreamble() {
    ParseHeader();
    while (At(Tk::Import)) ParseImport();
}

// <stmt>  ::= [ <expr> ] ";"
//           | <expr-block>
//           | EVAL <stmt>
auto Parser::ParseStmt() -> Ptr<ParsedExpr> {
    auto loc = tok->location;
    if (Consume(Tk::Eval)) {
        auto arg = ParseStmt();
        if (not arg) return {};
        return new (*this) ParsedEvalExpr{arg.get(), loc};
    }

    if (auto res = ParseExpr()) {
        if (not isa<ParsedDecl>(res.get()) and not isa<ParsedBlockExpr>(res.get()))
            ConsumeOrError(Tk::Semicolon);
        return res;
    }

    SkipTo(Tk::Semicolon);
    return {};
}
