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
    static_assert(std::is_trivially_destructible_v<decltype(*this)>);
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
    static_assert(std::is_trivially_destructible_v<decltype(*this)>);
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

struct ParsedExpr::Printer : PrinterBase<ParsedExpr> {
    Printer(bool use_colour, ParsedExpr* E) : PrinterBase{use_colour} { Print(E); }
    void Print(ParsedExpr* E);
};

void ParsedExpr::Printer::Print(ParsedExpr* e) {
    switch (e->kind()) {
        using enum utils::Colour;
        case Kind::BlockExpr: {
            [[maybe_unused]] auto& x = *cast<ParsedBlockExpr>(e);

            fmt::print(
                "{}BlockExpr {}{} {}<{}>",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            fmt::print("\n{}", C(Reset));

            SmallVector<ParsedExpr*, 10> fields;
            if (auto a = x.stmts(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::CallExpr: {
            [[maybe_unused]] auto& x = *cast<ParsedCallExpr>(e);

            fmt::print(
                "{}CallExpr {}{} {}<{}>",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            fmt::print("\n{}", C(Reset));

            SmallVector<ParsedExpr*, 10> fields;
            if (x.callee) fields.push_back(x.callee);
            if (auto a = x.args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::DeclRefExpr: {
            [[maybe_unused]] auto& x = *cast<ParsedDeclRefExpr>(e);

            fmt::print(
                "{}DeclRefExpr {}{} {}<{}>",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            fmt::print(" {}{}", C(Reset), x.name);
            fmt::print("\n{}", C(Reset));
        } break;

        case Kind::StrLitExpr: {
            [[maybe_unused]] auto& x = *cast<ParsedStrLitExpr>(e);

            fmt::print(
                "{}StrLitExpr {}{} {}<{}>",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            fmt::print(" {}\"{}\"", C(Yellow), utils::Escape(x.value));
            fmt::print("\n{}", C(Reset));
        } break;

        case Kind::ProcDecl: {
            [[maybe_unused]] auto& x = *cast<ParsedProcDecl>(e);

            fmt::print(
                "{}ProcDecl {}{} {}<{}>",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            fmt::print("\n{}", C(Reset));

            SmallVector<ParsedExpr*, 10> fields;
            if (x.body) fields.push_back(x.body);
            PrintChildren(fields);
        } break;
    }
}

void ParsedExpr::dump(bool use_colour) const {
    Printer(use_colour, const_cast<ParsedExpr*>(this));
}

#define PARSE_TREE_NODE(node) \
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
auto Parser::Parse(const File& file) -> Result<std::unique_ptr<ParsedModule>> {
    Parser P{file};
    P.ReadTokens(file);
    P.ParseFile();
    if (P.ctx.has_error()) return Diag();
    return std::move(P.mod);
}

auto Parser::ParseBlock() -> Result<ParsedBlockExpr*> {
    auto loc = tok->location;
    ++tok;

    // Parse statements.
    SmallVector<ParsedExpr*> stmts;
    while (not At(Tk::Eof, Tk::RBrace))
        if (auto s = ParseStmt())
            stmts.push_back(*s);
    ConsumeOrError(Tk::RBrace);
    return ParsedBlockExpr::Create(*this, stmts, loc);
}

// <expr> ::= <decl-proc>
//          | <expr-block>
//          | <expr-call>
//          | <expr-decl-ref>
//          | <expr-lit>
auto Parser::ParseExpr() -> Result<ParsedExpr*> {
    Result<ParsedExpr*> lhs = Diag();
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
            if (not body) return body.error();
            lhs = new (*this) ParsedProcDecl{name, *body, loc};
        } break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-decl-ref> ::= IDENTIFIER
        case Tk::Identifier: {
            lhs = new (*this) ParsedDeclRefExpr{tok->text, tok->location};
            ++tok;
        } break;

        // <expr-lit> ::= STRING-LITERAL
        case Tk::StringLiteral: {
            lhs = new (*this) ParsedStrLitExpr{tok->text, tok->location};
            ++tok;
        } break;
    }

    // Big operator parse loop.
    // TODO: precedence.
    while (At(Tk::LParen)) {
        // <expr-call> ::= <expr> "(" [ <call-args> ] ")"
        // <call-args> ::= <expr> { "," <expr> } [ "," ]
        ++tok;
        SmallVector<ParsedExpr*> args;
        while (not At(Tk::RParen)) {
            if (auto arg = ParseExpr()) {
                args.push_back(*arg);
                if (not Consume(Tk::Comma)) break;
            } else {
                SkipTo(Tk::RParen);
                break;
            }
        }

        ConsumeOrError(Tk::RParen);
        lhs = ParsedCallExpr::Create(*this, *lhs, args, {lhs->loc});
    }

    return lhs;
}

// <file> ::= <preamble> { <stmt> }
void Parser::ParseFile() {
    ParsePreamble();
    while (not At(Tk::Eof))
        if (auto stmt = ParseStmt())
            mod->top_level.push_back(*stmt);
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
        Diag::Warning(ctx, loc, "Duplicate import ignored");
        Diag::Note(ctx, it->loc, "Previous import was here");
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

// <stmt> ::= <decl> | <expr> ";"
auto Parser::ParseStmt() -> Result<ParsedExpr*> {
    if (auto res = ParseExpr()) {
        if (not isa<ParsedDecl>(*res)) ConsumeOrError(Tk::Semicolon);
        return res;
    }

    SkipTo(Tk::Semicolon);
    return Diag();
}
