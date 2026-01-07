#include <srcc/AST/Enums.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/Support/Casting.h>

#include <memory>
#include <utility>

using namespace srcc;

#define TRY(x, ...)       ({auto _x = x; if (not _x) { __VA_ARGS__ ; return {}; } *_x; })
#define TryParseExpr() TRY(ParseExpr())
#define TryParseStmt() TRY(ParseStmt())
#define TryParseType() TRY(ParseType())

// ============================================================================
//  Parser Helpers
// ============================================================================
/// Precedence used for tokens that aren’t operators.
constexpr int NotAnOperator = -1;

/// Precedence used for prefix operators.
constexpr int PrefixPrecedence = 900;

/// Get the precedence of this operator. Does not handled unary prefix operators.
constexpr int BinaryOrPostfixPrecedence(Tk t) {
    switch (t) {
        case Tk::ColonColon:
            return 100'000;

        case Tk::Dot:
            return 10'000;

        case Tk::LBrack:
        case Tk::PlusPlus:
        case Tk::MinusMinus:
        case Tk::Caret:    // This is a type operator only.
        case Tk::Question: // This is a type operator only.
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
        case Tk::ColonSlash:
        case Tk::ColonPercent:
            return 95;

        case Tk::Plus:
        case Tk::PlusTilde:
        case Tk::Minus:
        case Tk::MinusTilde:
            return 90;

        // Shifts have higher precedence than logical/bitwise
        // operators so e.g. `a & 1 << 3` works properly.
        case Tk::ShiftLeft:
        case Tk::ShiftLeftLogical:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
            return 85;

        case Tk::Ampersand:
        case Tk::VBar:
            return 82;

        case Tk::DotDotEq:
        case Tk::DotDotLess:
        case Tk::DotDot: // See ParseExpr().
            return 81;

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
        case Tk::MinusEq:
        case Tk::MinusTildeEq:
        case Tk::StarEq:
        case Tk::StarTildeEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftLeftLogicalEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
        case Tk::Swap:
            return 1;

        default:
            return NotAnOperator;
    }
}

/// The precedence used to parse template parameters; this is set to the precedence
/// of '>' + 1 so that we stop parsing when we find the '>'.
constexpr int TemplateParamPrecedence = BinaryOrPostfixPrecedence(Tk::SGt) + 1;

/// The precedence used to parse the return type of a function; this is set to the
/// precedence of '=' + 1 so that we stop parsing if we encounter an '='.
constexpr int ReturnTypePrecedence = BinaryOrPostfixPrecedence(Tk::Assign) + 1;

/// Check if this is a right-associative operator.
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

/// Check if this is a postfix operator.
constexpr bool IsPostfix(Tk t) {
    switch (t) {
        case Tk::MinusMinus:
        case Tk::PlusPlus:
            return true;

        default:
            return false;
    }
}

/// Check if this is a prefix operator.
constexpr bool IsPrefix(Tk t) {
    switch (t) {
        case Tk::Ampersand:
        case Tk::Caret:
        case Tk::Inline:
        case Tk::Minus:
        case Tk::MinusMinus:
        case Tk::Not:
        case Tk::Plus:
        case Tk::PlusPlus:
        case Tk::Tilde:
            return true;

        default:
            return false;
    }
}

//// Check whether an expression requires a semicolon after it when
//// used in a statement context.
//static bool RequireSemicolonAfterExpr(ParsedStmt* e) { // clang-format off
//    return e->visit(utils::Overloaded{
//        // If the expression ends with a brace, then we don’t require a semicolon
//        // after it and we’re done here. Don’t just check if the previous token was
//        // '}', since for e.g. 'return { ... }', we *do* want to require a semicolon.
//
//
//        // Any other expression requires a semicolon.
//        [](auto*) { return true; }
//
//    });
//    if (isa<ParsedBlockExpr, ParsedMatchExpr>(e)) return false;
//
//    // For '#quote's, require a semicolon after '#quote()' but not '#quote{}'
//    if (auto quote = dyn_cast<ParsedQuoteExpr>(e))
//        return not quote->brace_delimited;
//
//    // If the expression has a 'body', check that.
//    if (auto proc = dyn_cast<ParsedProcDecl>(e))
//        return not proc->body or RequireSemicolonAfterExpr(proc->body.get());
//    if (auto loop = dyn_cast<ParsedLoopExpr>(e))
//        return not loop->body or RequireSemicolonAfterExpr(loop->body.get());
//    if (auto eval = dyn_cast<ParsedEvalExpr>(e))
//        return RequireSemicolonAfterExpr(eval->expr);
//    if (auto var = dyn_cast<ParsedVarDecl>(e))
//        return not var->init or RequireSemicolonAfterExpr(var->init.get());
//
//    return true;
//} // clang-format on

Parser::BracketTracker::BracketTracker(Parser& p, Tk open, bool diagnose)
    : p{p}, diagnose{diagnose}, open_bracket{open} {
    switch (open) {
        case Tk::LParen: close_bracket = Tk::RParen; p.num_parens++; break;
        case Tk::LBrace: close_bracket = Tk::RBrace; p.num_braces++; break;
        case Tk::LBrack: close_bracket = Tk::RBrack; p.num_brackets++; break;
        default: Unreachable("Invalid bracket: {}", open);
    }

    left = p.tok->location;
    if (p.At(open_bracket)) p.NextTokenImpl();
    else if (diagnose) p.Error("Expected '{}'", open_bracket);
}

Parser::BracketTracker::~BracketTracker() {
    if (not consumed_close) decrement();
}

bool Parser::BracketTracker::close() {
    Assert(not consumed_close);
    consumed_close = true;
    decrement();
    right = p.tok->location;
    if (p.At(close_bracket)) {
        p.NextTokenImpl();
        return true;
    }

    if (diagnose) p.Error("Expected '{}'", close_bracket);
    return false;
}

void Parser::BracketTracker::decrement() {
    switch (close_bracket) {
        case Tk::RParen: if (p.num_parens) p.num_parens--; break;
        case Tk::RBrace: if (p.num_braces) p.num_braces--; break;
        case Tk::RBrack: if (p.num_brackets) p.num_brackets--; break;
        default: Unreachable("Invalid bracket: {}", close_bracket);
    }
}

/// Check if this token could start an expression (that doesn’t
/// imply that it actually does!).
bool Parser::AtStartOfExpression() {
    switch (tok->type) {
        default: return false;
        case Tk::Ampersand:
        case Tk::Assert:
        case Tk::Bool:
        case Tk::Break:
        case Tk::Caret:
        case Tk::Continue:
        case Tk::Dollar:
        case Tk::Eval:
        case Tk::False:
        case Tk::Hash:
        case Tk::Identifier:
        case Tk::If:
        case Tk::Inline:
        case Tk::Int:
        case Tk::IntegerType:
        case Tk::Integer:
        case Tk::Match:
        case Tk::Minus:
        case Tk::MinusMinus:
        case Tk::Nil:
        case Tk::NoReturn:
        case Tk::Not:
        case Tk::LBrace:
        case Tk::LParen:
        case Tk::Plus:
        case Tk::PlusPlus:
        case Tk::Proc:
        case Tk::Quote:
        case Tk::Range:
        case Tk::RBrace:
        case Tk::Return:
        case Tk::StringLiteral:
        case Tk::TemplateType:
        case Tk::Tilde:
        case Tk::Tree:
        case Tk::True:
        case Tk::Type:
        case Tk::Typeof:
        case Tk::Var:
        case Tk::Void:
        case Tk::With:
            return true;
    }
}

/// Check that this token can start a statement.
bool Parser::AtStartOfStatement() {
    switch (tok->type) {
        default: return AtStartOfExpression();
        case Tk::Defer:
        case Tk::Export:
        case Tk::For:
        case Tk::Static:
        case Tk::Struct:
        case Tk::While:
        case Tk::With:
            return true;
    }
}

bool Parser::Consume(Tk tk) {
    SLoc l;
    return Consume(l, tk);
}

bool Parser::Consume(SLoc& into, Tk tk) {
    if (At(tk)) {
        into = Next();
        return true;
    }
    return false;
}

bool Parser::ConsumeContextual(StringRef keyword) {
    SLoc l;
    return ConsumeContextual(l, keyword);
}

bool Parser::ConsumeContextual(SLoc& into, StringRef keyword) {
    if (At(Tk::Identifier) and tok->text == keyword) {
        into = Next();
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

auto Parser::CreateType(Signature& sig) -> ParsedProcType* {
    // If no return type was provided, default to 'void' here.
    if (sig.ret.invalid()) {
        sig.ret = new (*this) ParsedBuiltinType(
            Type::VoidTy,
            sig.proc_loc
        );
    }

    return ParsedProcType::Create(
        *this,
        sig.ret.get(),
        sig.param_types,
        sig.attrs,
        sig.proc_loc
    );
}

bool Parser::ExpectSemicolon() {
    if (Consume(Tk::Semicolon)) return true;
    if (tok == stream.begin()) Error("Expected ';'");
    else {
        auto prev = tok - 1;
        Error(prev->location.after(ctx), "Expected ';'");
    }
    return false;
}

static auto GetCorrespondingClosingBracket(Tk t) -> Tk {
    switch (t) {
        case Tk::LParen: return Tk::RParen;
        case Tk::LBrace: return Tk::RBrace;
        case Tk::LBrack: return Tk::RBrack;
        default: Unreachable();
    }
}

static bool IsOpenBracket(Tk t) {
    switch (t) {
        case Tk::LParen:
        case Tk::LBrace:
        case Tk::LBrack:
            return true;

        default:
            return false;
    }
}

static bool IsBracket(Tk t) {
    switch (t) {
        case Tk::LParen:
        case Tk::RParen:
        case Tk::LBrace:
        case Tk::RBrace:
        case Tk::LBrack:
        case Tk::RBrack:
            return true;

        default:
            return false;
    }
}

auto Parser::LookAhead(usz n) -> const Token& {
    usz curr = usz(tok - stream.begin());
    if (n + curr >= stream.size()) return stream.back();
    return stream[n + curr];
}

auto Parser::Next() -> SLoc {
    Assert(not IsBracket(tok->type), "Should not consume brackets this way");
    return NextTokenImpl();
}

auto Parser::NextTokenImpl() -> SLoc {
    auto* t = &*tok;
    if (not At(Tk::Eof)) ++tok;
    return t->location;
}

bool Parser::SkipTo(std::same_as<Tk> auto... tks) {
    // Always skip at least one token. It’s easy to fall into an infinite
    // loop otherwise.
    bool skipped = false;
    for (;;) {
        if (At(tks...)) return true;
        switch (tok->type) {
            default: Next(); break;
            case Tk::Eof: return false;

            // Skip over properly nested brackets.
            case Tk::LParen:
            case Tk::LBrace:
            case Tk::LBrack: {
                BracketTracker t{*this, tok->type, false};
                SkipTo(t.corresponding_closing_bracket());
                t.close();
            } break;

            // Discard any stray closing delimiters, but do not consume any
            // that match a preceding open bracket, e.g. we don’t want to skip
            // over the '}' of a block if its last statement is missing a semicolon.
            //
            // We don’t care if the matching open bracket isn’t the directly preceding
            // open bracket; e.g. in '({)', we don’t want to skip the ')', even though
            // the preceding open bracket is '{'.
            case Tk::RParen:
                if (num_parens and skipped) return false;
                NextTokenImpl();
                break;

            case Tk::RBrace:
                if (num_braces and skipped) return false;
                NextTokenImpl();
                break;

            case Tk::RBrack:
                if (num_brackets and skipped) return false;
                NextTokenImpl();
                break;
        }
        skipped = true;
    }
}

/// Skip up to and past a token.
bool Parser::SkipPast(std::same_as<Tk> auto... tks) {
    if (SkipTo(tks...)) {
        Next();
        return true;
    }

    return false;
}

// ============================================================================
//  Parser
// ============================================================================
Parser::Parser(ParsedModule* mod, const TokenStream& tokens, bool internal)
    : mod{mod},
      stream{tokens},
      ctx{mod->context()},
      parsing_internal_file(internal) {
    Assert(stream.size() != 0, "Cannot parse empty token stream");
    Assert(stream.back().eof(), "Token stream must end with EOF token");
    tok = stream.begin();
}

auto Parser::Parse(
    const File& file,
    CommentTokenCallback cb,
    bool is_internal_file
) -> ParsedModule::Ptr {
    // Create a module and lex it.
    auto mod = std::make_unique<ParsedModule>(file.context());
    TokenStream stream{mod->allocator()};
    ReadTokens(stream, file, std::move(cb));

    // Create the parser and parse the file.
    Parser p{mod.get(), stream, is_internal_file};
    p.ParseFile();
    return mod;
}

auto Parser::ParseFragment(Context& ctx, const TokenStream& toks) -> ParsedModule::Ptr {
    auto mod = std::make_unique<ParsedModule>(ctx);
    Parser p{mod.get(), toks, false};
    p.ParseStmts(mod->top_level);
    return mod;
}

// <expr-assert> ::= [ "#" ] ASSERT <expr> [ "," <expr> ]
auto Parser::ParseAssert(bool is_compile_time) -> Ptr<ParsedAssertExpr> {
    auto start = Next();
    auto cond = TryParseExpr();
    SRange cond_range{cond->loc, std::prev(tok)->location};

    // Message is optional.
    Ptr<ParsedStmt> message;
    if (Consume(Tk::Comma)) message = ParseExpr();
    return new (*this) ParsedAssertExpr{
        cond,
        message,
        is_compile_time,
        start,
        cond_range
    };
}

auto Parser::ParseBlock() -> Ptr<ParsedBlockExpr> {
    BracketTracker braces{*this, Tk::LBrace};
    SmallVector<ParsedStmt*> stmts;
    ParseStmts(stmts, Tk::RBrace);
    braces.close();
    return ParsedBlockExpr::Create(*this, stmts, braces.left);
}

// <expr-decl-ref> ::= IDENTIFIER [ "::" <expr-decl-ref> ]
auto Parser::ParseDeclRefExpr() -> Ptr<ParsedDeclRefExpr> {
    auto loc = tok->location;
    SmallVector<DeclName> strings;
    do {
        if (not At(Tk::Identifier)) {
            Error("Expected identifier after '::'");
            SkipTo(Tk::Semicolon);
            return {};
        }

        strings.push_back(tok->text);
        Next();
    } while (Consume(Tk::ColonColon));
    return ParsedDeclRefExpr::Create(*this, strings, loc);
}

// This parses expressions and also some declarations (e.g.
// variable declarations)
//
// <expr> ::= <expr-assert>
//          | <expr-block>
//          | <expr-binary>
//          | <expr-break>
//          | <expr-call>
//          | <expr-continue>
//          | <expr-decl-ref>
//          | <expr-eval>
//          | <expr-if>
//          | <expr-lit>
//          | <expr-loop>
//          | <expr-match>
//          | <expr-member>
//          | <expr-paren>
//          | <expr-prefix>
//          | <expr-quote>
//          | <expr-return>
//          | <expr-subscript>
//          | <expr-tuple>
//          | <expr-postfix>
//          | <expr-with>
//          | <type>
//
// <type> ::= <type-prim>
//          | TEMPLATE-TYPE
//          | <signature>
//          | <type-qualified>
//          | <type-range>
//          | <type-typeof>
//          | <expr-decl-ref> // Named type
//          | <expr-tuple>    // Tuple type
//          | <expr-binary>   // Array type (parsed as subscript expression)
//          | <expr-call>     // Function that returns a type
//          | "(" <expr> ")"  // Any expression in parens can be treated as a type
//          | <expr>          // In type-only contexts
// <type-qualified> ::= <type> { <qualifier> }
// <type-typeof> ::= TYPEOF "(" <expr> ")"
// <type-range> ::= RANGE "<" <type> ">"
// <qualifier> ::= "[" "]" | "^" | "?"
// <type-prim> ::= BOOL | INT | VOID | VAR | INTEGER_TYPE
auto Parser::ParseExpr(int precedence, bool expect_type) -> Ptr<ParsedStmt> {
    auto BuiltinType = [&](Type ty) {
        return new (*this) ParsedBuiltinType(ty, Next());
    };

    auto ParseParenthesisedExpr = [&] -> Ptr<ParsedStmt> {
        BracketTracker parens{*this, Tk::LParen};
        auto arg = ParseExpr();
        parens.close();
        return arg;
    };

    Ptr<ParsedStmt> lhs;
    bool at_start = AtStartOfExpression(); // See below.
    auto start_tok = tok->type;
    switch (tok->type) {
        case Tk::Else:
        case Tk::Elif:
        case Tk::Then:
            return Error("Unexpected '%1({}%)'", tok->type);

        case Tk::Dollar:
            return Error(lhs.get()->loc, "'%1($()%)' outside of '%1(#quote%)'");

        // Compile-time expressions.
        case Tk::Hash: {
            auto hash_loc = Next();
            switch (tok->type) {
                case Tk::If: lhs = ParseIf(true); break;
                case Tk::Assert: lhs = ParseAssert(true); break;
                case Tk::Elif:
                case Tk::Else:
                    return Error(hash_loc, "Unexpected '%1(#{}%)'", tok->type);

                default: {
                    // <expr-inject> ::= "#" INJECT "(" <expr> ")"
                    if (SLoc loc; ConsumeContextual(loc, "inject")) {
                        auto arg = TRY(ParseParenthesisedExpr());
                        lhs = new (*this) ParsedInjectExpr{arg, loc};
                        break;
                    }

                    // <expr-macro-call> ::= "#" <expr-decl-ref> "(" [ <call-args> ] ")"
                    if (not At(Tk::Identifier)) return Error("Expected macro name or keyword after '%1(#%)'");
                    auto ident = TRY(ParseDeclRefExpr());

                    // Parse arguments as '#quote'd.
                    BracketTracker parens{*this, Tk::LParen};
                    SmallVector<ParsedStmt*> args;
                    while (not At(Tk::RParen, Tk::Eof)) {
                        if (auto arg = ParseQuotedTokenSeq(tok->location, true).get_or_null()) {
                            args.push_back(arg);
                            if (not Consume(Tk::Comma)) break;
                        } else {
                            break;
                        }
                    }

                    // Create the call and wrap it with an '#inject'.
                    parens.close();
                    lhs = ParsedCallExpr::Create(*this, ident, args, ident->loc);
                    lhs = new (*this) ParsedInjectExpr(lhs.get(), hash_loc);
                } break;
            }
        } break;

        // <expr-assert> ::= ASSERT <expr> [ "," <expr> ]
        case Tk::Assert:
            lhs = ParseAssert(false);
            break;

        // <expr-break> ::= BREAK
        case Tk::Break:
            lhs = new (*this) ParsedBreakContinueExpr{false, Next()};
            break;

        // <expr-continue> ::= CONTINUE
        case Tk::Continue:
            lhs = new (*this) ParsedBreakContinueExpr{true, Next()};
            break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-eval> ::= EVAL <sub-stmt>
        case Tk::Eval: {
            auto start = Next();
            auto arg = TryParseStmt();
            lhs = new (*this) ParsedEvalExpr{arg, start};
        } break;

        // <expr-loop> ::= LOOP [ <sub-stmt> ]
        case Tk::Loop: {
            auto start = Next();
            Ptr<ParsedStmt> arg;
            if (AtStartOfStatement()) arg = TryParseStmt();
            return new (*this) ParsedLoopExpr{arg, start};
        }

        case Tk::If:
            lhs = ParseIf(false);
            break;

        case Tk::Identifier:
            lhs = ParseDeclRefExpr();
            break;

        case Tk::Match:
            lhs = ParseMatchExpr();
            break;

        case Tk::Nil:
            lhs = new (*this) ParsedNilExpr(Next());
            break;

        case Tk::Quote:
            lhs = TRY(ParseQuotedTokenSeq(Next(), false));
            break;

        // STRING-LITERAL
        case Tk::StringLiteral:
            lhs = new (*this) ParsedStrLitExpr{tok->text, tok->location};
            Next();
            break;

        // INTEGER
        case Tk::Integer:
            lhs = new (*this) ParsedIntLitExpr{*this, tok->integer, tok->location};
            Next();
            break;

        // TRUE | FALSE
        case Tk::True:
        case Tk::False:
            lhs = new (*this) ParsedBoolLitExpr{tok->type == Tk::True, tok->location};
            Next();
            break;

        // <type-typeof> ::= TYPEOF "(" <expr> ")"
        case Tk::Typeof: {
            auto loc = Next();
            BracketTracker parens{*this, Tk::LParen};
            auto arg = TryParseExpr();
            parens.close();
            lhs = new (*this) ParsedTypeofType(arg, loc);
        } break;

        // <expr-return>   ::= RETURN [ <expr> ]
        case Tk::Return: {
            auto loc = Next();

            Ptr<ParsedStmt> value;
            if (AtStartOfExpression()) value = TryParseExpr();
            lhs = new (*this) ParsedReturnExpr{value, loc};
        } break;

        // <expr-paren> ::= "(" <expr> ")"
        // <expr-tuple> ::= "(" [ <expr> { "," <expr> } [ "," ] ] ")"
        case Tk::LParen: {
            BracketTracker parens{*this, Tk::LParen};

            // '()'.
            if (At(Tk::RParen)) {
                parens.close();
                lhs = ParsedTupleExpr::Create(*this, {}, parens.left);
                break;
            }

            // Parenthesised expression.
            lhs = ParseExpr();
            if (At(Tk::RParen)) {
                parens.close();
                if (not lhs) return {};
                lhs = new (*this) ParsedParenExpr{lhs.get(), parens.left};
                break;
            }

            // Tuple.
            SmallVector<ParsedStmt*> exprs;
            if (lhs) exprs.push_back(lhs.get());
            do {
                // Error about a missing comma, and skip multiple consecutive commas.
                if (not Consume(Tk::Comma)) Error("Expected '%1(,%)' in tuple");
                while (Consume(Tk::Comma)) Error(std::prev(tok)->location, "Unexpected ','");
                if (At(Tk::RParen)) break;

                lhs = ParseExpr();
                if (lhs) exprs.push_back(lhs.get());
                else SkipTo(Tk::Comma, Tk::RParen);
            } while (not At(Tk::RParen, Tk::Eof));
            parens.close();
            lhs = ParsedTupleExpr::Create(*this, exprs, parens.left);
        } break;

        // <type-prim> ::= BOOL | INT | TREE | TYPE | VOID | VAR | NORETURN
        case Tk::Bool: lhs = BuiltinType(Type::BoolTy); break;
        case Tk::Int: lhs = BuiltinType(Type::IntTy); break;
        case Tk::NoReturn: lhs = BuiltinType(Type::NoReturnTy); break;
        case Tk::Tree: lhs = BuiltinType(Type::TreeTy); break;
        case Tk::Type: lhs = BuiltinType(Type::TypeTy); break;
        case Tk::Void: lhs = BuiltinType(Type::VoidTy); break;
        case Tk::Var: lhs = BuiltinType(Type::DeducedTy); break;

        // INTEGER_TYPE
        case Tk::IntegerType:
            if (not tok->integer.isSingleWord()) return Error("Integer type too large");
            lhs = new (*this) ParsedIntType(Size::Bits(tok->integer.getZExtValue()), Next());
            break;

        // <signature>
        case Tk::Proc: {
            Signature sig;
            if (not ParseSignature(sig, nullptr, false)) return nullptr;

            // A procedure name is not allowed here.
            if (not sig.name.empty()) Error(
                sig.tok_after_proc,
                "A name is not allowed in a procedure type"
            );

            lhs = CreateType(sig);
        } break;

        // <type-range> ::= RANGE "<" <type> ">"
        case Tk::Range: {
            SLoc loc = Next(), end{};
            if (not Consume(Tk::SLt)) Error("Expected '%1(<%)' after '%1(range%)'");
            auto ty = TRY(ParseType(TemplateParamPrecedence));
            if (not Consume(end, Tk::SGt)) {
                Error("Expected '%1(>%)'");
                SkipTo(Tk::Semicolon, Tk::SGt);
                Consume(end, Tk::SGt);
            }

            lhs = new (*this) ParsedRangeType(ty, loc);
        } break;

        // TEMPLATE-TYPE
        case Tk::TemplateType: {
            if (not current_signature) return Error("'{}' is not allowed here; try removing the '$'", tok->text);

            // Drop the '$' from the type.
            auto ty = new (*this) ParsedTemplateType(tok->text.drop(), tok->location);
            current_signature->add_deduced_template_param(ty->name);
            Next();
            lhs = ty;
        } break;

        // <expr-with> ::= WITH <expr> [ DO ] <sub-stmt>
        case Tk::With: {
            auto loc = Next();
            auto expr = TryParseExpr();
            Consume(Tk::Do);
            auto body = TryParseStmt();
            return new (*this) ParsedWithExpr{expr, body, loc};
        }

        // <expr-prefix> ::= <prefix> <expr>
        default: {
            if (IsPrefix(tok->type)) {
                auto op = tok->type;
                auto start = Next();
                auto arg = TRY(ParseExpr(PrefixPrecedence));
                lhs = new (*this) ParsedUnaryExpr{op, arg, false, start};
                break;
            }

            if (expect_type) Error("Expected type");
            else Error("Expected expression");
            return {};
        }
    }

    // There was an error.
    if (lhs.invalid()) return nullptr;

    // I keep forgetting to add new tokens to AtStartOfExpression,
    // so this is here to make sure I don’t forget.
    Assert(at_start, "Forgot to add '{}' to AtStartOfExpression", start_tok);

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
                BracketTracker parens{*this, Tk::LParen};
                SmallVector<ParsedStmt*> args;
                while (not At(Tk::RParen)) {
                    if (auto arg = ParseExpr()) {
                        args.push_back(arg.get());
                        if (not Consume(Tk::Comma)) break;
                    } else {
                        break;
                    }
                }

                parens.close();
                lhs = ParsedCallExpr::Create(*this, lhs.get(), args, lhs.get()->loc);
                continue;
            }

            // <expr=subscript> ::= <expr> "[" <expr> "]"
            case Tk::LBrack: {
                BracketTracker brackets{*this, Tk::LBrack};
                if (At(Tk::RBrack)) {
                    brackets.close();
                    lhs = new (*this) ParsedSliceType(lhs.get(), lhs.get()->loc);
                    continue;
                }

                auto index = TryParseExpr();
                brackets.close();
                lhs = new (*this) ParsedBinaryExpr{
                    Tk::LBrack,
                    lhs.get(),
                    index,
                    lhs.get()->loc
                };
                continue;
            }

            case Tk::Caret:
                lhs = new (*this) ParsedPtrType(lhs.get(), lhs.get()->loc);
                Next();
                continue;

            case Tk::Question:
                lhs = new (*this) ParsedOptionalType(lhs.get(), lhs.get()->loc);
                Next();
                continue;

            // <expr-member> ::= <expr> "." IDENTIFIER
            case Tk::Dot: {
                Next();
                if (not At(Tk::Identifier)) {
                    Error("Expected identifier after '.'");
                    SkipTo(Tk::Semicolon);
                    return {};
                }

                lhs = new (*this) ParsedMemberExpr(lhs.get(), tok->text, lhs.get()->loc);
                Next();
                continue;
            }
        }

        auto op = tok->type;
        Next();
        if (IsPostfix(op)) {
            lhs = new (*this) ParsedUnaryExpr{op, lhs.get(), true, lhs.get()->loc};
        } else {
            if (op == Tk::DotDot) Error("'%1(..%)' is not a valid operator; did you mean '%1(..=%)' or '%1(..<%)'?");
            auto rhs = ParseExpr(BinaryOrPostfixPrecedence(op));
            if (not rhs) return {};
            lhs = new (*this) ParsedBinaryExpr{op, lhs.get(), rhs.get(), lhs.get()->loc};
        }
    }

    return lhs;
}

// <stmt-for> ::= FOR [ <for-vars> IN ] <expr> { "," <expr> } [ DO ] <stmt>
// <for-vars> ::= <idents> | ENUM IDENT [ "," <idents> ]
// <idents>   ::= IDENT { "," IDENT }
auto Parser::ParseForStmt() -> Ptr<ParsedStmt> {
    SmallVector<ParsedForStmt::LoopVar> vars;
    auto for_loc = Next();
    auto ParseIdents = [&] -> bool {
        // This is only an identifier if followed by a comma or 'in' (or another
        // identifier, which likely indicates a missing comma); otherwise, we might
        // have a loop of the form 'for x { ... }', in which case we should not treat
        // 'x' as a loop variable.
        while (At(Tk::Identifier) and LookAhead().is(Tk::Identifier, Tk::Comma, Tk::In)) {
            vars.emplace_back(tok->text, tok->location);
            Next();
            if (not Consume(Tk::Comma)) {
                if (At(Tk::Identifier)) {
                    // Recover from a missing comma if we're sure that that’s what the
                    // user intended here.
                    if (LookAhead().is(Tk::In, Tk::Do)) {
                        Error("Expected '%1(,%)'");
                        continue;
                    }

                    // Only issue this error message once if we’re not sure what we want here.
                    Error("Expected '%1(,%)', '%1(in%)', or '%1(do%)'");
                    return false;
                }
                break;
            }
        }
        return true;
    };

    // Parse enumerator and identifiers.
    String enum_name;
    SLoc enum_loc;
    if (Consume(Tk::Enum)) {
        if (At(Tk::Identifier)) {
            enum_name = tok->text;
            enum_loc = Next();
        } else {
            Error("Expected identifier after '%1(for enum%)'");
        }

        if (Consume(Tk::Comma) and not ParseIdents())
            return {};
    } else if (At(Tk::Identifier) and not ParseIdents()) {
        return {};
    }

    // 'in' is required iff we have at least one identifier.
    SmallVector<ParsedStmt*> ranges;
    if (not enum_name.empty() or not vars.empty()) {
        if (not Consume(Tk::In)) {
            // This might be a missing comma. If not, it’s unlikely that trying
            // to parse the rest of this is going to yield anything but more
            // errors.
            if (At(Tk::Identifier)) Error("Expected '%1(,%)', '%1(in%)', or '%1(do%)'");
            else return Error("Expected '%1(in%)'");
        }
    } else if (Consume(Tk::In)) {
        Error(for_loc, "'%1(for in%)' is invalid");
        Remark("Valid syntaxes for 'for' loops include 'for y' and 'for x in y'.");
        return {};
    }

    // Ranges.
    do {
        ranges.push_back(TryParseExpr());
        if (not Consume(Tk::Comma)) {
            if (At(Tk::Identifier)) Error("Expected ','");
            else break;
        }
    } while (AtStartOfExpression());

    // Body.
    Consume(Tk::Do);
    auto body = TryParseStmt();
    return ParsedForStmt::Create(*this, for_loc, enum_loc, enum_name, vars, ranges, body);
}

// <file> ::= <preamble> { <stmt> }
void Parser::ParseFile() {
    ParsePreamble();
    ParseStmts(mod->top_level);
}

// <header> ::= ( "program" | "module" ) <module-name> ";"
//   [ext]    | "__srcc_internal__"
// <module-name> ::= IDENTIFIER
void Parser::ParseHeader() {
    // Parse internal files as non-modules to disallow e.g. 'export' since
    // they may end up being included in a program.
    if (ConsumeContextual(mod->program_or_module_loc, "__srcc_internal__")) {
        if (not parsing_internal_file) Error(
            mod->program_or_module_loc,
            "'%1(__srcc_internal__%)' is not permitted in user code"
        );
        return;
    }

    bool module = ConsumeContextual(mod->program_or_module_loc, "module");
    if (
        not module and
        not ConsumeContextual(mod->program_or_module_loc, "program")
    ) {
        Error("Expected '%1(program%)' or '%1(module%)' directive at start of file");
        SkipPast(Tk::Semicolon);
        return;
    }

    if (not At(Tk::Identifier)) {
        Error("Expected identifier after '%1({}%)'", module ? "module" : "program");
        SkipPast(Tk::Semicolon);
        return;
    }

    mod->name = tok->text;
    mod->is_module = module;
    Next();
    Consume(Tk::Semicolon);
}

// <expr-if> ::= [ "#" ] IF <expr> <if-expr-body> { [ "#" ] ELIF <expr> <if-expr-body> } [ [ "#" ] ELSE <if-expr-body> ]
auto Parser::ParseIf(bool is_static) -> Ptr<ParsedIfExpr> {
    // Yeet 'if'.
    auto loc = Next();

    // Condition.
    auto cond = TryParseExpr();

    // <if-expr-body> ::= [ THEN ] <sub-stmt>
    auto ParseBody = [&] -> Ptr<ParsedStmt> {
        Consume(Tk::Then);
        return ParseStmt();
    };

    // Check for unnecessary parens around the condition, which is
    // common in other languages.
    //
    // Note that, because the parser is greedy, it is impossible for
    // the meaning of the program to change if the parens are omitted;
    // even in cases that at first seem like they might be problematic,
    // e.g. 'if (x) (3)' and 'if (x) ++y', we would actually end up with
    // '(x)(3)' (a call) and '(x)++' (a unary expression) as the condition,
    // neither of which is a ParenExpr.
    if (isa<ParsedParenExpr>(cond)) Warn(
        cond->loc,
        "Unnecessary parentheses around '%1(if%)' condition"
    );

    // '#then' is nonsense. Just drop the hash sign.
    if (At(Tk::Hash) and LookAhead().is(Tk::Then)) {
        Error(tok->location, "'%1(#then%)' is invalid; write '%1(then%)' instead");
        Next();
    }

    // Parse the if body.
    auto body = TRY(ParseBody());

    // Permit extra semicolons before '(#|)(else|if)'.
    if ( // clang-format off
        At(Tk::Semicolon) and (
            LookAhead().is(Tk::Else, Tk::Elif) or
            (LookAhead().is(Tk::Hash) and LookAhead(2).is(Tk::Else, Tk::Elif))
        )
    ) Next(); // clang-format on

    // '#if' must be paired with '#else'/'#elif'
    bool hash = At(Tk::Hash);
    if (hash and LookAhead().is(Tk::Else, Tk::Elif)) {
        Next();
        if (not is_static) {
            hash = false;
            Error(
                std::prev(tok)->location,
                "'%1(if%)' must be paired with '%1({0}%)', not '%1(#{0}%)'",
                tok->type
            );
        }
    }

    // Parse an else branch if there is one.
    Ptr<ParsedStmt> else_;

    // Handle invalid uses of 'else'.
    if (Consume(Tk::Else)) {
        bool correct_to_elif = false;

        // Correct '(#)else if' to '(#)elif'.
        if (At(Tk::If)) {
            correct_to_elif = true;
            Error(
                hash ? std::prev(tok)->location : tok->location,
                "Use '%1({0}elif%)' instead of '%1({0}else if%)'",
                hash ? "#" : ""
            );
        }

        // Diagnose '(#)else #if' and '#else #if'.
        else if (At(Tk::Hash) and LookAhead().is(Tk::If)) {
            correct_to_elif = true;
            Error(
                std::prev(tok, hash ? 2 : 1)->location,
                "'%1({}else #if%)' is invalid; did you mean '%1({}elif%)'",
                hash ? "#" : "",
                is_static ? "#" : ""
            );
            Next(); // Yeet '#'.
        }

        // '#if' must be paired with '#else'.
        else if (is_static and not hash) {
            Error(
                std::prev(tok)->location,
                "'%1(#if%)' must be paired with '%1(#else%)'"
            );
        }

        // After recovering from whatever nonsense the user potentially gave us,
        // actually parse the else clause.
        else_ = correct_to_elif ? ParseIf(is_static) : ParseBody();
    }

    // For elif, just check for a missing hash.
    else if (At(Tk::Elif)) {
        if (is_static and not hash) Error(
            tok->location,
            "'%1(#if%)' must be paired with '%1(#elif%)'"
        );

        else_ = ParseIf(is_static);
    }


    // The body of an '#if' should *not* create a new scope.
    if (is_static) {
        auto AdjustScopeFlag = [](Ptr<ParsedStmt> branch ) {
            auto block = llvm::dyn_cast_if_present<ParsedBlockExpr>(branch.get_or_null());
            if (block) block->should_push_scope = false;
        };

        AdjustScopeFlag(body);
        AdjustScopeFlag(else_);
    }

    // Finally, build the expression.
    return new (*this) ParsedIfExpr{
        cond,
        body,
        else_,
        is_static,
        loc,
    };
}

// <import>      ::= IMPORT <import-name> AS ( IDENT | "* ) ";"
// <import-name> ::= IDENT | CXX-HEADER-NAME { "," CXX-HEADER-NAME } [ "," ]
void Parser::ParseImport() {
    SLoc import_loc;
    SmallVector<String> linkage_names;
    String logical_name;
    Assert(Consume(import_loc, Tk::Import), "Not at 'import'?");
    bool is_header = false;
    if (At(Tk::CXXHeaderName)) {
        is_header = true;
        while (At(Tk::CXXHeaderName)) {
            linkage_names.push_back(tok->text);
            Next();
            if (not Consume(Tk::Comma)) break;
        }

        // Header names are not valid identifiers, so they always require 'as'.
        if (not At(Tk::As)) {
            Error("Syntax for header imports is '%1(import%) %3(<header>%) %1(as%) name`");
            SkipPast(Tk::Semicolon);
            return;
        }
    } else if (At(Tk::Identifier)) {
        logical_name = tok->text;
        linkage_names.push_back(tok->text);
        Next();
    } else {
        Error("Expected module or header name after 'import'");
        SkipPast(Tk::Semicolon);
        return;
    }

    bool star = false;
    if (Consume(Tk::As)) {
        if (At(Tk::Identifier)) {
            logical_name = tok->text;
            Next();
        } else if (Consume(Tk::Star)) {
            star = true;
        } else {
            Error("Expected identifier or '%1(*%)' after '%1(as%)' in import directive");
            SkipPast(Tk::Semicolon);
            return;
        }
    }

    mod->imports.emplace_back(
        std::move(linkage_names),
        logical_name,
        import_loc,
        star,
        is_header
    );

    if (not Consume(Tk::Semicolon)) Error("Expected ';' at end of import");
}

// <intent> ::= IN | OUT | INOUT | COPY
auto Parser::ParseIntent() -> std::pair<SLoc, Intent> {
    SLoc loc;

    // 'in' is a keyword, unlike the other intents.
    if (Consume(loc, Tk::In)) {
        // Correct 'in out' to 'inout'.
        SLoc out_loc;
        if (ConsumeContextual(out_loc, "out")) {
            Error(loc, "Cannot specify more than one parameter intent");
            Remark("Did you mean to write 'inout' instead of 'in out'?");
            return {loc, Intent::Inout};
        }

        return {loc, Intent::In};
    }

    // The other intents are contextual keywords.
    if (ConsumeContextual(loc, "out")) return {loc, Intent::Out};
    if (ConsumeContextual(loc, "inout")) return {loc, Intent::Inout};
    if (ConsumeContextual(loc, "copy")) return {loc, Intent::Copy};

    // If no intent is present, just return this as a default.
    return {{}, Intent::Move};
}

// <expr-match> ::= MATCH [ <expr> ] [ "->" <type> ] "{" <match-case> [ ";" ] "}"
// <match-case> ::= <pattern> ":" <stmt>
// <pattern>    ::= <expr>
auto Parser::ParseMatchExpr() -> Ptr<ParsedMatchExpr> {
    auto match_loc = Next();
    auto control_expr = At(Tk::LBrace, Tk::RArrow) ? nullptr : ParseExpr();
    Ptr<ParsedStmt> type;
    if (Consume(Tk::RArrow)) type = ParseType();

    // Parse cases.
    BracketTracker braces{*this, Tk::LBrace};
    SmallVector<ParsedMatchCase> cases;
    while (not At(Tk::RBrace, Tk::Eof)) {
        auto pattern = ParseExpr();
        if (not Consume(Tk::Colon)) Error("Expected ':' after pattern");
        auto body = ParseStmt(); // FIXME: ParseStmt() should skip to ';' or '}' here.
        if (pattern and body) cases.emplace_back(pattern.get(), body.get());
        Consume(Tk::Semicolon); // Permit extra semicolons here.
    }

    braces.close();
    return ParsedMatchExpr::Create(*this, control_expr, type, cases, match_loc);
}

// <decl-var> ::= [ STATIC ] <type> IDENTIFIER [ "=" <expr> ] ";"
auto Parser::ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedVarDecl> {
    auto decl = new (*this) ParsedVarDecl(
        tok->text,
        type,
        type->loc
    );

    if (not Consume(Tk::Identifier))
        Error("Expected identifier in variable declaration");

    // Parse the optional initialiser.
    if (Consume(Tk::Assign)) decl->init = ParseExpr();

    // We don’t allow declaration groups such as 'int a, b;'.
    if (At(Tk::Comma)) {
        Error("A declaration must declare a single variable");
        SkipTo(Tk::Semicolon);
    }

    return decl;
}

/// Check if the current token represents an operator name that
/// can be used in a function definition to define an overloaded
/// operator function.
///
/// Note that it’s possible for this function to do nothing at all
/// if this is a signature without a name.
void Parser::ParseOverloadableOperatorName(Signature& sig) {
    // Parse 'proc ()' as the call operator. For an empty argument list,
    // people should just omit the '()' (and for anonymous functions, we
    // shouldn’t even be using 'proc' in the first place...).
    //
    // Note that '(' on its own is not an operator, but rather the start
    // of an argument list, so don’t error in that case. Also correct
    // 'proc ( (' to 'proc () ('.
    if (At(Tk::LParen)) {
        if (LookAhead().is(Tk::RParen, Tk::LParen)) {
            sig.name = Tk::LParen;
            BracketTracker parens{*this, Tk::LParen, false};
            if (not parens.close()) Error(
                parens.left,
                "To overload the call operator, write '%1(proc ()%)'"
            );
        }
    }

    // Similarly, '[]' is a valid operator.
    else if (At(Tk::LBrack)) {
        sig.name = Tk::LBrack;
        BracketTracker brackets{*this, Tk::LBrack, false};
        if (not brackets.close()) Error(
            brackets.left,
            "To overload the subscript operator, write '%1(proc []%)'"
        );
    }

    // Handle other builtin operators.
    else if (
        BinaryOrPostfixPrecedence(tok->type) != NotAnOperator or
        IsPrefix(tok->type)
    ) {
        if (At(Tk::Assign, Tk::ColonColon, Tk::Dot)) Error(
            "Operator '{}' cannot be overloaded",
            tok->type
        );

        // Continue parsing either way.
        sig.name = tok->type;
        Next();
    }
}

// <param-decl>  ::= [ <intent> ] <param-rest>
// <param-rest>  ::= <type> [ "..." ] [ IDENT ] | <signature>
bool Parser::ParseParameter(Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls) {
    Ptr<ParsedStmt> type;
    String name;
    SLoc name_loc;
    bool variadic = false;

    // Parse intent.
    auto [start_loc, intent] = ParseIntent();
    if (intent == Intent::Move) start_loc = tok->location;

    // And do it again; two intents are an error.
    else if (auto [loc, i] = ParseIntent(); i != Intent::Move) {
        Error(loc, "Cannot specify more than one parameter intent");
        return false;
    }

    // Special handling for signatures, which may have
    // a name in this position.
    if (At(Tk::Proc)) {
        Signature inner;
        if (not ParseSignature(inner, nullptr, false)) return false;
        if (inner.name.is_operator_name()) {
            Error(inner.tok_after_proc, "Invalid parameter name: '{}'", inner.name);
        } else {
            name = inner.name.str();
            name_loc = inner.tok_after_proc;
        }

        type = CreateType(inner);

        // For all template parameters that appear in the signature,
        // add the index of the parameter that is the signature.
        for (const auto& p : inner.deduction_info)
            sig.add_deduced_template_param(p.first);
    }

    // Otherwise, parse a regular type and a name if we’re
    // creating declarations.
    else {
        type = ParseType();
        if (not type) return false;
        variadic = Consume(Tk::Ellipsis);
    }

    sig.param_types.emplace_back(intent, type.get(), variadic);

    // If decls is not null, then we allow named parameters here; parse
    // the name if there is one and create the declaration.
    if (decls) {
        if (At(Tk::Identifier)) {
            if (not name.empty()) {
                Error("Parameter cannot have two names");
                Note(name_loc, "Name was already specified here");
            }

            name = tok->text;
            Next();
        } else if (not At(Tk::Comma, Tk::RParen)) {
            if (IsKeyword(tok->type)) {
                Error(
                    "'%1({})' is not a valid parameter name because it is "
                    "a reserved word.",
                    tok->text
                );
            } else {
                Error("Unexpected token in procedure argument list");
            }

            SkipTo(Tk::Comma, Tk::RParen);
        }

        decls->push_back(new (*this) ParsedVarDecl{name, type.get(), start_loc, intent});
    } else if (At(Tk::Identifier)) {
        Error("Named parameters are not allowed here");
        Next();
    }

    return true;
}

// <preamble> ::= <header> { <import> }
void Parser::ParsePreamble() {
    ParseHeader();
    while (At(Tk::Import)) ParseImport();
}

// <decl-proc> ::= <signature> <proc-body>
// <proc-body> ::= <expr-block> | "=" <expr-braces> | "=" <expr-no-braces> ";" | ";"
auto Parser::ParseProcDecl() -> Ptr<ParsedProcDecl> {
    // Parse signature.
    SmallVector<ParsedVarDecl*, 10> param_decls;
    Signature sig;
    if (not ParseSignature(sig, &param_decls, true)) return nullptr;

    // The 'proc' syntax requires a name.
    if (sig.name.empty()) {
        Error("Procedures declared with '%1(proc%)' must have a name");
        return nullptr;
    }

    // If we failed to parse a return type, or if there was
    // none, just default to void instead, or deduce the type
    // if this is a '= <expr>' declaration.
    if (sig.ret.invalid()) {
        sig.ret = new (*this) ParsedBuiltinType(
            At(Tk::Assign) ? Type::DeducedTy.ptr() : Type::VoidTy.ptr(),
            sig.proc_loc
        );
    }

    // If the next token can’t introduce a body, skip any intervening junk
    // that the user may have put here.
    bool complained_about_body = false;
    if (not At(Tk::LBrace, Tk::Semicolon, Tk::Assign)) {
        complained_about_body = true;
        Error("Expected procedure body");
        SkipTo(Tk::LBrace, Tk::Semicolon, Tk::Assign);
    }

    // Parse the body.
    Ptr<ParsedStmt> body;
    if (SLoc assign_loc; Consume(assign_loc, Tk::Assign)) {
        if (At(Tk::LBrace)) Error(assign_loc, "'%1(= {{%)' is invalid; remove the '%1(=%)'");
        body = ParseExpr();
    } else if (At(Tk::Semicolon)) {
        // Nothing.
    } else if (At(Tk::LBrace)) {
        body = ParseBlock();
    }

    // Procedures not declared 'extern' must have a body (and vice versa).
    // FIXME: Move this diagnostic to Sema instead.
    if (
        not complained_about_body and
        sig.attrs.extern_ != not bool(body)
    ) Error(
        sig.proc_loc,
        "Procedure that is{} declared '%1(extern%)' must{} have a body",
        sig.attrs.extern_ ? ""sv : " not"sv,
        sig.attrs.extern_ ? " not"sv : ""sv
    );

    auto proc = ParsedProcDecl::Create(
        *this,
        sig.name,
        CreateType(sig),
        param_decls,
        body,
        sig.where,
        sig.name.empty() ? sig.proc_loc : sig.tok_after_proc
    );

    if (not sig.deduction_info.empty())
        mod->template_deduction_infos[proc] = std::move(sig.deduction_info);

    return proc;
}

auto Parser::ParseQuotedTokenSeq(SLoc quote_loc, bool in_macro_call) -> Ptr<ParsedStmt> {
    // If this is a macro argument, we might have no tokens at all here.
    if (in_macro_call and At(Tk::Comma, Tk::RParen))
        return ParsedQuoteExpr::Create(*this, nullptr, {}, false, quote_loc);

    // Consume the open bracket.
    SmallVector<Tk> brackets;
    bool brace_delimited = false;
    if (not in_macro_call) {
        Assert(IsOpenBracket(tok->type));
        brackets.push_back(tok->type);
        brace_delimited = tok->type == Tk::LBrace;
        NextTokenImpl();

        // This might also be empty.
        if (At(GetCorrespondingClosingBracket(brackets.front()))) {
            NextTokenImpl();
            return ParsedQuoteExpr::Create(*this, nullptr, {}, false, quote_loc);
        }
    }

    // Remember the first token that is part of the quote. We know it is
    // not an open bracket, so skip it.
    SmallVector<ParsedUnquoteExpr*> unquotes;
    auto tokens = mod->quoted_tokens.emplace_back(std::make_unique<TokenStream>(mod->allocator())).get();
    auto AddToken = [&] {
        tokens->push(*tok);
        NextTokenImpl();
    };

    // Collect the tokens.
    auto KeepSkipping = [&] {
        if (At(Tk::Eof)) return false;

        // Parse a ‘balanced’ token sequence. We don’t actually require this to
        // be fully balanced; rather, we just ignore unbalanced closing delimiters,
        // so keep going until the bracket stack is empty.
        if (not brackets.empty()) return true;

        // Stop if we’re not in a macro argument.
        if (not in_macro_call) return false;

        // Inside a macro argument, we also keep going after a balanced sequence
        // until we encounter a comma ot closing parenthesis.
        return not At(Tk::Comma, Tk::RParen);
    };

    while (KeepSkipping()) {
        if (IsOpenBracket(tok->type)) {
            brackets.push_back(tok->type);
            AddToken();
        } else if (not brackets.empty() and At(GetCorrespondingClosingBracket(brackets.back()))) {
            brackets.pop_back();

            // Don’t add the top-level delimiter to the collected tokens.
            if (not brackets.empty() or in_macro_call) AddToken();
            else NextTokenImpl();
        } else if (SLoc dollar; Consume(dollar, Tk::Dollar)) {
            BracketTracker parens{*this, Tk::LParen};
            auto arg = ParseExpr();
            parens.close();
            if (arg) unquotes.push_back(new (*this) ParsedUnquoteExpr{arg.get(), parens.left});

            // Add a marker token so we can replace it with the expansion of
            // the unquote later on.
            Token t;
            t.artificial = true;
            t.type = Tk::Unquote;
            t.location = dollar;
            tokens->push(std::move(t));
        } else {
            AddToken();
        }
    }

    // Check that we have a delimiter; do not skip it if we’re in
    // a macro argument (because the caller should handle the comma
    // or closing parenthesis).
    if (not brackets.empty()) Error(quote_loc, "Token sequence delimited by end of file");
    return ParsedQuoteExpr::Create(
        *this,
        tokens,
        unquotes,
        brace_delimited,
        quote_loc
    );
}

bool Parser::ParseSignature(
    Signature& sig,
    SmallVectorImpl<ParsedVarDecl*>* decls,
    bool allow_constraint
) {
    tempset current_signature = &sig;
    return ParseSignatureImpl(decls, allow_constraint);
}

// <signature>  ::= PROC [ IDENTIFIER ] [ <proc-args> ] <proc-attrs> [ "->" <type> ] [ <proc-where> ]
// <proc-args>  ::= "(" [ <param-decl> { "," <param-decl> } [ "," ] ] ")"
// <proc-attrs> ::= { "native" | "extern" | "nomangle" | "variadic" }
// <proc-where> ::= WHERE <expr>
bool Parser::ParseSignatureImpl(SmallVectorImpl<ParsedVarDecl*>* decls, bool allow_constraint) {
    Assert(current_signature);
    auto& sig = *current_signature;

    // Yeet 'proc'.
    sig.proc_loc = Next();

    // Parse name.
    sig.tok_after_proc = tok->location;
    if (At(Tk::Identifier)) {
        sig.name = tok->text;
        Next();
    } else {
        ParseOverloadableOperatorName(sig);
    }

    // Parse params.
    if (At(Tk::LParen)) {
        BracketTracker parens{*this, Tk::LParen};
        while (not At(Tk::RParen, Tk::Eof)) {
            if (not ParseParameter(sig, decls)) SkipTo(Tk::Comma, Tk::RParen);
            if (not Consume(Tk::Comma)) break;
        }

        if (not At(Tk::RParen)) {
            Error("Unexpected token in procedure argument list");
            SkipTo(Tk::RParen);
        }

        parens.close();
    }

    // Parse attributes.
    auto ParseAttr = [&](bool& attr, StringRef value) {
        if (SLoc loc; ConsumeContextual(loc, value)) {
            if (attr) Warn(loc, "Duplicate '%1({}%)' attribute", value);
            attr = true;
            return true;
        }

        return false;
    };

    auto ParseKwAttr = [&](bool& attr, Tk tok) {
        if (SLoc loc; Consume(loc, tok)) {
            if (attr) Warn(loc, "Duplicate '%1({}%)' attribute", tok);
            attr = true;
            return true;
        }

        return false;
    };

    while (
        ParseAttr(sig.attrs.extern_, "extern") or
        ParseAttr(sig.attrs.native, "native") or
        ParseAttr(sig.attrs.nomangle, "nomangle") or
        ParseAttr(sig.attrs.c_varargs, "varargs") or
        ParseAttr(sig.attrs.builtin_operator, "__srcc_builtin_op__") or
        ParseAttr(sig.attrs.no_mangling_number, "__srcc_no_mnum__") or
        ParseKwAttr(sig.attrs.inline_, Tk::Inline)
    );

    // Parse return type.
    if (Consume(Tk::RArrow)) sig.ret = ParseType(ReturnTypePrecedence);

    // Parse constraint.
    if (SLoc where_loc; Consume(where_loc, Tk::Where)) {
        sig.where = ParseExpr();
        if (not allow_constraint) Error(
            where_loc,
            "Constraints are only allowed in procedure declarations"
        );
    }

    return true;
}

// For the '<expr-no-braces> ";"' case, the caller has to check for
// a potentially missing semicolon.
//
// <stmt> ::= <expr-braces>
//          | <expr-no-braces> ";"
//          | <decl>
//          | <stmt-defer>
//          | <stmt-while>
//          | <stmt-for>
//
// <sub-stmt> ::= <expr> | <stmt>
auto Parser::ParseStmt() -> Ptr<ParsedStmt> {
    auto loc = tok->location;
    switch (tok->type) {
        // <stmt-defer> ::= DEFER <stmt>
        case Tk::Defer: {
            Next();
            auto arg = TryParseStmt();
            return new (*this) ParsedDeferStmt{arg, loc};
        }

        // EXPORT <decl>
        case Tk::Export: {
            Next();

            // Complain if this isn’t a module, but parse it anyway.
            if (not mod->is_module) {
                Error(loc, "'%1(export%)' is only allowed in modules");
                Note(stream.begin()->location,
                     "If you meant to create a module (i.e. a static or shared "
                     "library), use '%1(module%)' instead of '%1(program%)'");
            }

            // It’s easier to parse any statement and then disallow
            // if we parsed something that doesn’t belong here.
            auto arg = TryParseStmt();
            auto decl = dyn_cast<ParsedDecl>(arg);
            if (not decl) return Error(arg->loc, "Only declarations can be exported");

            // Avoid forcing Sema to deal with unwrapping nested
            // export declarations.
            if (isa<ParsedExportDecl>(decl)) return Error(loc, "'%1(export export%)' is invalid");
            return new (*this) ParsedExportDecl{decl, loc};
        }

        // <stmt-for>
        case Tk::For: return ParseForStmt();

        // <decl-var>
        case Tk::Static: {
            Next();
            while (At(Tk::Static)) {
                Error("'%1(static static%)' is invalid");
                Next();
            }

            auto ty = TryParseExpr();
            auto var = ParseVarDecl(ty);
            if (not var) return {};
            var.get()->is_static = true;
            return var;
        }

        // <decl-struct>
        case Tk::Struct: return ParseStructDecl();

        // <stmt-while> ::= WHILE <expr> [ DO ] <stmt>
        case Tk::While: {
            Next();
            auto cond = TryParseExpr();
            Consume(Tk::Do);
            auto body = TryParseStmt();
            return new (*this) ParsedWhileStmt{cond, body, loc};
        }

        // <decl-proc>
        case Tk::Proc: return ParseProcDecl();

        // ";"
        case Tk::Semicolon: return new (*this) ParsedEmptyStmt{Next()};

        // <expr-braces> | <expr-no-braces> | <decl-var>
        default: {
            auto e = TryParseExpr();

            // Whether it makes sense to interpret an expression as a type.
            //
            // Note that we require some of the ‘weirder’ expressions to be wrapped
            // in parentheses for them to be treated as types here.
            auto CouldReasonablyBeAType = [](ParsedStmt* s) {
                if (isa<
                    ParsedType,
                    ParsedCallExpr,
                    ParsedDeclRefExpr,
                    ParsedInjectExpr,
                    ParsedParenExpr,
                    ParsedTupleExpr,
                    ParsedUnquoteExpr
                >(s)) return true;

                // Subscripts could be array types instead.
                if (auto bin = dyn_cast<ParsedBinaryExpr>(s))
                    return bin->op == Tk::LBrack;

                // Any other expression requires parentheses to become a type.
                return false;
            };

            // <decl-var>
            //
            // Types are expressions so variable declarations start with an expression.
            // TODO: Do we really need the 'type name' syntax instead of just always using
            //       'var'/'val' to declare a variable; that would simplify this part.
            //
            // If the next token is an identifier, and the expression could reaonably be a
            // type, parse a variable declaration.
            if (At(Tk::Identifier) and CouldReasonablyBeAType(e)) return ParseVarDecl(e);
            return e;
        }
    }
}

// Parse a semicolon-separated list of statements.
//
// <stmts> ::= { <stmt> } [ <expr> ]
void Parser::ParseStmts(SmallVectorImpl<ParsedStmt*>& into, Tk stop_at) {
    while (not At(Tk::Eof, stop_at)) {
        auto s = ParseStmt();
        if (not s) {
            SkipPast(Tk::Semicolon);
            continue;
        }

        into.push_back(s.get());

        // If the expression doesn’t have braces, require a semicolon, except
        // that the last expression in the list doesn’t need one. Additionally,
        // don’t require a semicolon after a semicolon since that’s just silly.
        //
        // Note that we *allow* a semicolon after the last expression and just
        // discard it; in particular, it is *not* treated as an empty statement.
        Assert(tok != stream.begin());
        if (
            not std::prev(tok)->is(Tk::Semicolon) and
            not Consume(Tk::Semicolon) and
            not std::prev(tok)->is(Tk::RBrace) and
            not At(stop_at) and
            not ExpectSemicolon()
        ) SkipPast(Tk::Semicolon);
    }
}

// <decl-struct> ::= STRUCT IDENTIFIER "{" { <type> IDENTIFIER ";" } "}"
auto Parser::ParseStructDecl() -> Ptr<ParsedStructDecl> {
    auto struct_loc = Next();

    // Name.
    String name = tok->text;
    if (not Consume(Tk::Identifier))
        Error("Expected identifier after '%1(struct%)'");

    // Body.
    SmallVector<ParsedFieldDecl*> fields;
    BracketTracker braces{*this, Tk::LBrace};
    while (not At(Tk::RBrace, Tk::Eof)) {
        auto ParseField = [&] {
            auto ty = ParseType();
            if (not ty) {
                SkipTo(Tk::Semicolon, Tk::RBrace);
                return;
            }

            String field_name = tok->text;
            if (not Consume(Tk::Identifier)) {
                Error("Expected identifier");
                SkipTo(Tk::Semicolon, Tk::RBrace);
                return;
            }

            fields.push_back(new (*this) ParsedFieldDecl{field_name, ty.get(), ty.get()->loc});
        };

        ParseField();
        if (not ExpectSemicolon()) {
            SkipTo(Tk::Semicolon, Tk::RBrace);
            Consume(Tk::Semicolon);
        }
    }

    braces.close();
    return ParsedStructDecl::Create(*this, name, fields, struct_loc);
}
