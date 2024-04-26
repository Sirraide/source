module;

#include <algorithm>
#include <llvm/ADT/SmallString.h>
#include <memory>
#include <srcc/Macros.hh>
#include <utility>

module srcc.frontend.parser;
using namespace srcc;

// ============================================================================
//  Parse Tree
// ============================================================================
struct Printer {
    SmallString<128> leading;
    utils::Colours C;

    Printer(bool use_colour, ParsedExpr* e) : C{use_colour} { Print(e); }

    void Print(ParsedExpr* e);
    void PrintChildren(ArrayRef<ParsedExpr*> children) {
        using enum utils::Colour;
        if (children.empty()) return;
        const auto size = leading.size();

        // Print all but the last.
        leading += "│ ";
        const auto current = StringRef{leading}.take_front(size);
        for (auto c : children.drop_back(1)) {
            fmt::print("{}{}├─", C(Red), current);
            Print(c);
        }

        // Print the preheader of the last.
        leading.resize(size);
        fmt::print("{}{}└─", C(Red), StringRef{leading});

        // Print the last one.
        leading += "  ";
        Print(children.back());

        // And reset the leading text.
        leading.resize(size);
    }
};

void ParsedModule::dump() const {
    using enum utils::Colour;
    bool c = context().use_colours();
    utils::Colours C{c};
    fmt::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);
    for (auto i : imports) fmt::print("{}Import {}<{}>\n", C(Red), C(Blue), i.first);
    for (auto s : top_level) s->dump(c);
}

void ParsedExpr::dump(bool use_colour) {
    Printer(use_colour, this);
}

#define SRCC_PARSE_TREE_IMPL
#include "srcc/astgen/ParseTree.inc"
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
            lhs = ParsedProcDecl::Create(*this, name, *body, loc);
        } break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-decl-ref> ::= IDENTIFIER
        case Tk::Identifier: {
            lhs = ParsedDeclRefExpr::Create(*this, tok->text, tok->location);
            ++tok;
        } break;

        // <expr-lit> ::= STRING-LITERAL
        case Tk::StringLiteral: {
            lhs = ParsedStrLitExpr::Create(*this, tok->text, tok->location);
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

// <import> ::= IMPORT CXX-HEADER-NAME ";"
void Parser::ParseImport() {
    Location import_loc;
    Assert(Consume(import_loc, Tk::Import), "Not at 'import'?");
    if (not At(Tk::CXXHeaderName)) {
        Error("Expected C++ header name after 'import'");
        SkipTo(Tk::Semicolon);
        return;
    }

    // Warn about duplicate imports.
    Location loc = {import_loc, tok->location};
    if (
        auto it = rgs::find(mod->imports, tok->text, &ParsedModule::Import::first);
        it != mod->imports.end()
    ) {
        Diag::Warning(ctx, loc, "Duplicate import ignored");
        Diag::Note(ctx, it->second, "Previous import was here");
    } else {
        mod->imports.emplace_back(tok->text, loc);
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
