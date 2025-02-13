#include <srcc/AST/Enums.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/SmallString.h>

#include <algorithm>
#include <memory>
#include <print>
#include <ranges>
#include <utility>

using namespace srcc;

#define TRY(x, ...)       ({auto _x = x; if (not _x) { __VA_ARGS__ ; return {}; } _x.get(); })
#define TryParseExpr(...) TRY(ParseExpr() __VA_OPT__(, ) __VA_ARGS__)
#define TryParseStmt(...) TRY(ParseStmt() __VA_OPT__(, ) __VA_ARGS__)

// ============================================================================
//  Parse Tree
// ============================================================================
void ParsedModule::dump() const {
    bool c = context().use_colours;

    // Print preamble.
    utils::Print(c, "%1({}) {}\n", is_module ? "Module" : "Program", name);
    for (auto i : imports) std::print(
        "%1(Import) %4(<{}>) %1(as) %4({})\n",
        i.linkage_name,
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
ParsedProcType::ParsedProcType(
    ParsedStmt* ret_type,
    ArrayRef<ParsedParameter> params,
    ParsedProcAttrs attrs,
    Location loc
) : ParsedStmt{Kind::ProcType, loc},
    num_params{u32(params.size())},
    ret_type{ret_type},
    attrs{attrs} {
    std::uninitialized_copy_n(
        params.begin(),
        params.size(),
        getTrailingObjects<ParsedParameter>()
    );
}

auto ParsedProcType::Create(
    Parser& parser,
    ParsedStmt* ret_type,
    ArrayRef<ParsedParameter> params,
    ParsedProcAttrs attrs,
    Location loc
) -> ParsedProcType* {
    const auto size = totalSizeToAlloc<ParsedParameter>(params.size());
    auto mem = parser.allocate(size, alignof(ParsedProcType));
    return ::new (mem) ParsedProcType{ret_type, params, attrs, loc};
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
    Ptr<ParsedStmt> body,
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
    Ptr<ParsedStmt> body,
    Location location
) -> ParsedProcDecl* {
    const auto size = totalSizeToAlloc<ParsedLocalDecl*>(param_decls.size());
    auto mem = parser.allocate(size, alignof(ParsedProcDecl));
    return ::new (mem) ParsedProcDecl{name, type, param_decls, body, location};
}

ParsedStructDecl::ParsedStructDecl(
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    Location loc
) : ParsedDecl{Kind::StructDecl, name, loc}, num_fields(u32(fields.size())) {
    std::uninitialized_copy_n(
        fields.begin(),
        fields.size(),
        getTrailingObjects<ParsedFieldDecl*>()
    );
}

auto ParsedStructDecl::Create(
    Parser& parser,
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    Location loc
) -> ParsedStructDecl* {
    const auto size = totalSizeToAlloc<ParsedFieldDecl*>(fields.size());
    auto mem = parser.allocate(size, alignof(ParsedStructDecl));
    return ::new (mem) ParsedStructDecl{name, fields, loc};
}

// ============================================================================
//  Printer
// ============================================================================
struct ParsedStmt::Printer : PrinterBase<ParsedStmt> {
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
    auto lc = module ? s->loc.seek_line_column(module->context()) : std::nullopt;
    print(
        "%1({}) %4({}) %5(<{}:{}>)",
        name,
        static_cast<void*>(s),
        lc ? lc->line : 0,
        lc ? lc->col : 0
    );

    if (full) print("\n");
    else print(" ");
}

void ParsedStmt::Printer::Print(ParsedStmt* s) {
    switch (s->kind()) {
        case Kind::BuiltinType:
        case Kind::IntType:
        case Kind::ProcType:
        case Kind::SliceType:
        case Kind::TemplateType:
            print("%1(Type) {}\n", s->dump_as_type());
            break;

        case Kind::AssertExpr: {
            auto& a = *cast<ParsedAssertExpr>(s);
            PrintHeader(s, "AssertExpr");
            SmallVector<ParsedStmt*, 2> children;
            children.push_back(a.cond);
            if (auto msg = a.message.get_or_null()) children.push_back(msg);
            PrintChildren(children);
        } break;

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
            print("%1({})\n", b.op);
            SmallVector<ParsedStmt*, 2> children{b.lhs, b.rhs};
            PrintChildren(children);
        } break;

        case Kind::BoolLitExpr: {
            PrintHeader(s, "BoolLitExpr", false);
            print("%1({})\n", cast<ParsedBoolLitExpr>(s)->value);
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
            print("%8({})\n", utils::join(d.names(), "::"));
        } break;

        case Kind::EvalExpr: {
            auto& v = *cast<ParsedEvalExpr>(s);
            PrintHeader(s, "EvalExpr");
            PrintChildren(v.expr);
        } break;

        case Kind::ExportDecl: {
            auto& e = *cast<ParsedExportDecl>(s);
            PrintHeader(s, "ExportDecl");
            PrintChildren(e.decl);
        } break;

        case Kind::FieldDecl: {
            auto& f = *cast<ParsedFieldDecl>(s);
            PrintHeader(s, "FieldDecl", false);
            print("%5({}) {}\n", f.name, f.type->dump_as_type());
        } break;

        case Kind::IfExpr: {
            auto& i = *cast<ParsedIfExpr>(s);
            PrintHeader(s, "IfExpr", not i.is_static);
            if (i.is_static) print("static\n");
            SmallVector<ParsedStmt*, 3> children{i.cond, i.then};
            if (auto e = i.else_.get_or_null()) children.push_back(e);
            PrintChildren(children);
        } break;

        case Kind::IntLitExpr: {
            PrintHeader(s, "IntLitExpr", false);
            auto val = cast<ParsedIntLitExpr>(s)->storage.str(false);
            print("%5({})\n", val);
        } break;

        case Kind::MemberExpr: {
            auto& m = *cast<ParsedMemberExpr>(s);
            PrintHeader(s, "MemberExpr", false);
            print("%8({})\n", m.member);
            PrintChildren(m.base);
        } break;

        case Kind::LocalDecl: {
            auto& p = *cast<ParsedLocalDecl>(s);
            PrintHeader(s, "LocalDecl", false);
            print("%4({}){}", p.name, p.name.empty() ? "" : " ");
            if (p.intent != Intent::Move) print("%1({}) ", p.intent);
            print("{}\n", p.type->dump_as_type());
            if (p.init) PrintChildren(p.init.get());
        } break;

        case Kind::ParenExpr: {
            PrintHeader(s, "ParenExpr");
            PrintChildren(cast<ParsedParenExpr>(s)->inner);
        } break;

        case Kind::ProcDecl: {
            auto& p = *cast<ParsedProcDecl>(s);
            PrintHeader(s, "ProcDecl", false);
            print(
                "%2({}){}{}\n",
                p.name,
                p.name.empty() ? ""sv : " "sv,
                p.type->dump_as_type()
            );

            // No need to print the param decls here.
            SmallVector<ParsedStmt*, 10> children;
            if (auto b = p.body.get_or_null()) children.push_back(b);
            PrintChildren(children);
        } break;

        case Kind::StrLitExpr: {
            auto& str = *cast<ParsedStrLitExpr>(s);
            PrintHeader(s, "StrLitExpr", false);
            print("%3(\"\002{}\003\")\n", utils::Escape(str.value));
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ParsedReturnExpr>(s);
            PrintHeader(s, "ReturnExpr");
            if (auto val = ret->value.get_or_null()) PrintChildren(val);
        } break;

        case Kind::StructDecl: {
            auto& d = *cast<ParsedStructDecl>(s);
            PrintHeader(s, "StructDecl", false);
            print("%6({})\n", d.name);
            PrintChildren<ParsedFieldDecl>(d.fields());
        } break;

        case Kind::UnaryExpr: {
            auto& u = *cast<ParsedUnaryExpr>(s);
            PrintHeader(s, "UnaryExpr", false);
            print("%1({})\n", u.op);
            PrintChildren(u.arg);
        } break;

        case Kind::WhileStmt: {
            auto& w = *cast<ParsedWhileStmt>(s);
            PrintHeader(s, "WhileStmt");
            SmallVector<ParsedStmt*, 2> children{w.cond, w.body};
            PrintChildren(children);
        } break;
    }
}

void ParsedStmt::dump(bool use_colour) const {
    dump(nullptr, use_colour);
}

void ParsedStmt::dump(const ParsedModule* owner, bool use_colour) const {
    Printer(owner, use_colour, const_cast<ParsedStmt*>(this));
}

auto ParsedStmt::dump_as_type() -> SmallUnrenderedString {
    SmallUnrenderedString out;

    auto Append = [&out](this auto& Append, ParsedStmt* type) -> void {
        switch (type->kind()) {
            case Kind::BuiltinType:
                out += cast<ParsedBuiltinType>(type)->ty->print();
                break;

            case Kind::IntType:
                out += std::format("%6(i{})", cast<ParsedIntType>(type)->bit_width);
                break;

            case Kind::ProcType: {
                auto p = cast<ParsedProcType>(type);
                out += "%1(proc";

                if (not p->param_types().empty()) {
                    bool first = true;
                    out += " (";

                    for (auto param : p->param_types()) {
                        if (not first) out += ", ";
                        first = false;
                        if (param.intent != Intent::Move) out += std::format("{} ", param.intent);
                        Append(param.type);
                    }

                    out += "\033)";
                }

                if (p->attrs.native) out += " native";
                if (p->attrs.extern_) out += " extern";
                if (p->attrs.nomangle) out += " nomangle";

                out += " -> )";
                Append(p->ret_type);
            } break;

            case Kind::SliceType: {
                auto s = cast<ParsedSliceType>(type);
                Append(s->elem);
                out += "%1([])";
            } break;

            case Kind::TemplateType: {
                auto t = cast<ParsedTemplateType>(type);
                out += std::format("%3(${})", t->name);
            } break;

            case Kind::DeclRefExpr: {
                auto d = cast<ParsedDeclRefExpr>(type);
                out += std::format("%8({})", utils::join(d->names(), "::"));
            } break;

            default:
                out += "<invalid type>";
                break;
        }
    };

    Append(this);
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
        case Tk::Assert:
        case Tk::Bool:
        case Tk::Eval:
        case Tk::False:
        case Tk::Identifier:
        case Tk::If:
        case Tk::Int:
        case Tk::IntegerType:
        case Tk::Integer:
        case Tk::Minus:
        case Tk::MinusMinus:
        case Tk::Not:
        case Tk::LBrace:
        case Tk::LParen:
        case Tk::Plus:
        case Tk::PlusPlus:
        case Tk::Proc:
        case Tk::RBrace:
        case Tk::Return:
        case Tk::Static:
        case Tk::StringLiteral:
        case Tk::TemplateType:
        case Tk::Tilde:
        case Tk::True:
        case Tk::Var:
        case Tk::Void:
            return true;
    }
}

bool Parser::Consume(Tk tk) {
    Location l;
    return Consume(l, tk);
}

bool Parser::Consume(Location& into, Tk tk) {
    if (At(tk)) {
        into = Next();
        return true;
    }
    return false;
}

bool Parser::ConsumeContextual(StringRef keyword) {
    Location l;
    return ConsumeContextual(l, keyword);
}

bool Parser::ConsumeContextual(Location& into, StringRef keyword) {
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
            Types::VoidTy.ptr(),
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

auto Parser::LookAhead(usz n) -> Token& {
    usz curr = usz(tok - stream.begin());
    if (n + curr >= stream.size()) return stream.back();
    return stream[n + curr];
}

auto Parser::Next() -> Location {
    auto loc = tok->location;
    if (not At(Tk::Eof)) ++tok;
    return loc;
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
    auto loc = Next();

    // Parse statements.
    SmallVector<ParsedStmt*> stmts;
    ParseStmts(stmts);

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
        loc = {loc, Next()};
    } while (Consume(Tk::ColonColon));
    return ParsedDeclRefExpr::Create(*this, strings, loc);
}

// This parses expressions and also some declarations (e.g.
// variable declarations)
//
// <expr> ::= <expr-assert>
//          | <expr-block>
//          | <expr-binary>
//          | <expr-call>
//          | <expr-decl-ref>
//          | <expr-eval>
//          | <expr-if>
//          | <expr-lit>
//          | <expr-member>
//          | <expr-paren>
//          | <expr-prefix>
//          | <expr-return>
//          | <expr-subscript>
auto Parser::ParseExpr(int precedence) -> Ptr<ParsedStmt> {
    Ptr<ParsedStmt> lhs;
    bool at_start = AtStartOfExpression(); // See below.
    auto start_tok = tok->type;
    switch (tok->type) {
        default: return Error("Expected expression");

        case Tk::Else:
        case Tk::Elif:
        case Tk::Then:
            return Error("Unexpected '%1({})'", tok->type);

        // <expr-prefix> ::= <prefix> <expr>
        case Tk::Minus:
        case Tk::Plus:
        case Tk::Not:
        case Tk::Tilde:
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            auto op = tok->type;
            auto start = Next();
            auto arg = TRY(ParseExpr(PrefixPrecedence));
            lhs = new (*this) ParsedUnaryExpr{op, arg, false, {start, arg->loc}};
        } break;

        // Static assert, and static if.
        case Tk::Static: {
            auto static_loc = Next();
            if (At(Tk::If)) lhs = ParseIf(true);
            else return Error("Expected '%1(if)' after '%1(static)'");
            if (auto l = lhs.get()) l->loc = {static_loc, l->loc};
        } break;

        // <expr-assert> ::= ASSERT <expr> [ "," <expr> ]
        case Tk::Assert: {
            auto start = Next();
            auto cond = TryParseExpr();

            // Message is optional.
            Ptr<ParsedStmt> message;
            if (Consume(Tk::Comma)) message = ParseExpr();

            lhs = new (*this) ParsedAssertExpr{
                cond,
                message,
                {start, message ? message.get()->loc : cond->loc},
            };
        } break;

        // <expr-block> ::= "{" { <stmt> } "}"
        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        // <expr-eval> ::= EVAL <expr>
        case Tk::Eval: {
            auto start = Next();
            auto arg = TryParseExpr();
            lhs = new (*this) ParsedEvalExpr{arg, {start, arg->loc}};
        } break;

        case Tk::If:
            lhs = ParseIf(false);
            break;

        case Tk::Identifier:
            lhs = ParseDeclRefExpr();
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

        // <expr-return>   ::= RETURN [ <expr> ]
        case Tk::Return: {
            auto loc = Next();

            Ptr<ParsedStmt> value;
            if (AtStartOfExpression()) value = TryParseExpr();
            if (value.present()) loc = {loc, value.get()->loc};
            lhs = new (*this) ParsedReturnExpr{value, loc};
        } break;

        // <expr-paren> ::= "(" <expr> ")"
        case Tk::LParen: {
            Location rparen;
            auto lparen = Next();

            lhs = ParseExpr();
            if (not Consume(rparen, Tk::RParen)) {
                Error("Expected '\033)'");
                SkipTo(Tk::RParen, Tk::Semicolon);
                if (not Consume(rparen, Tk::RParen)) return {};
            }

            lhs = new (*this) ParsedParenExpr{lhs.get(), {lparen, rparen}};
        } break;

        case Tk::Int:
        case Tk::IntegerType:
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
    Assert(at_start, "Forgot to add '{}' to AtStartOfExpression", start_tok);

    // If the next token is an identifier, then this is a declaration,
    // provided that the lhs could conceivably be a type (i.e. don’t
    // parse 'true a' as a declaration).
    if (At(Tk::Identifier)) {
        if (isa< // clang-format off
            ParsedDeclRefExpr,
            ParsedIntType,
            ParsedBuiltinType,
            ParsedProcType,
            ParsedTemplateType,
            ParsedParenExpr
        >(lhs.get())) return ParseVarDecl(lhs.get()); // clang-format on
        return lhs;
    }

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
                Next();
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
                if (LookAhead().is(Tk::RBrack)) {
                    lhs = ParseTypeRest(lhs.get());
                    continue;
                }

                Next();
                auto index = TryParseExpr();
                auto end = tok->location;
                ConsumeOrError(Tk::RBrack);
                lhs = new (*this) ParsedBinaryExpr{
                    Tk::LBrack,
                    lhs.get(),
                    index,
                    {lhs.get()->loc, end}
                };
                continue;
            }

            // <expr-member> ::= <expr> "." IDENTIFIER
            case Tk::Dot: {
                Next();
                if (not At(Tk::Identifier)) {
                    Error("Expected identifier after '.'");
                    SkipTo(Tk::Semicolon);
                    return {};
                }

                lhs = new (*this) ParsedMemberExpr(lhs.get(), tok->text, {lhs.get()->loc, tok->location});
                Next();
                continue;
            }
        }

        auto op = tok->type;
        auto end = Next();
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
    ParseStmts(mod->top_level);
}

// <header> ::= ( "program" | "module" ) <module-name> ";"
// <module-name> ::= IDENTIFIER
void Parser::ParseHeader() {
    auto module = ConsumeContextual(mod->program_or_module_loc, "module");
    if (not module and not ConsumeContextual(mod->program_or_module_loc, "program")) {
        Error("Expected '%1(program)' or '%1(module)' directive at start of file");
        return SkipPast(Tk::Semicolon);
    }

    if (not At(Tk::Identifier)) {
        Error("Expected identifier after '%1({})'", module ? "module" : "program");
        return SkipPast(Tk::Semicolon);
    }

    mod->name = tok->text;
    mod->is_module = module;
    Next();
    Consume(Tk::Semicolon);
}

// <expr-if> ::= [ STATIC] IF <expr> <if-body> { ELIF <expr> <if-body> } [ ELSE <expr> ]
// <if-body> ::= [ THEN ] <expr>
auto Parser::ParseIf(bool is_static) -> Ptr<ParsedIfExpr> {
    // Yeet 'if'.
    auto loc = Next();

    // Condition.
    auto cond = TryParseExpr(SkipTo(Tk::Semicolon));

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
        "Unnecessary parentheses around '%1(if)' condition"
    );

    // 'then' is optional.
    Consume(Tk::Then);
    auto body = TryParseStmt(SkipTo(Tk::Semicolon));

    // Disallow semicolons here.
    //
    // Requiring semicolons here would make figuring out at what point
    // we should discard the semicolon more difficult (currently, this
    // only happens in one place outside of the preamble, viz. in the
    // loop in ParseStmts()).
    if (At(Tk::Semicolon) and LookAhead().is(Tk::Elif, Tk::Else)) {
        Error("Semicolon before '%1({})' is not allowed", LookAhead().type);
        Next();
    }

    // 'else if' actually parses just fine, but I like 'elif' more so
    // we just warn on that. ;Þ
    if (At(Tk::Else) and LookAhead().is(Tk::If)) Warn(
        {tok->location, LookAhead().location},
        "Use '%1(elif)' instead of '%1(else if)'"
    );

    // Same with 'else static if'.
    if (
        At(Tk::Else) and
        LookAhead(1).is(Tk::Static) and
        LookAhead(2).is(Tk::If)
    ) Warn( //
        {tok->location, LookAhead(2).location},
        "Use '%1(static elif)' instead of '%1(else static if)'"
    );

    // Recover from redundant 'static' here.
    if (At(Tk::Static) and LookAhead().is(Tk::Else)) {
        Error(
            {tok->location, LookAhead().location},
            "'%1(static else)' is invalid"
        );
        Next();
    }

    // Parse the else/elif branch.
    bool elif_is_static = At(Tk::Static) and LookAhead().is(Tk::Elif);
    if (elif_is_static) Consume(Tk::Static);
    auto else_ = At(Tk::Elif)      ? ParseIf(elif_is_static)
               : Consume(Tk::Else) ? ParseStmt()
                                   : nullptr;
    return new (*this) ParsedIfExpr{
        cond,
        body,
        else_,
        is_static,
        {loc, else_ ? else_.get()->loc : body->loc}
    };
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
    Next();

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

    Next();
    Consume(Tk::Semicolon);
}

// <intent> ::= IN | OUT | INOUT | COPY
auto Parser::ParseIntent() -> std::pair<Location, Intent> {
    Location loc;

    // 'in' is a keyword, unlike the other intents.
    if (Consume(loc, Tk::In)) {
        // Correct 'in out' to 'inout'.
        Location out_loc;
        if (ConsumeContextual(out_loc, "out")) {
            Error({loc, out_loc}, "Cannot specify more than one parameter intent");
            Remark("Did you mean to write 'inout' instead of 'in out'?");
            return {{loc, out_loc}, Intent::Inout};
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

// <decl-var>   ::= <type> IDENTIFIER [ "=" <expr> ] ";"
auto Parser::ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedStmt> {
    auto decl = new (*this) ParsedLocalDecl(
        tok->text,
        type,
        {type->loc, tok->location}
    );

    // Parse the optional initialiser.
    Next();
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

// <decl-proc> ::= <signature> <proc-body>
// <proc-body> ::= <expr-block> | "=" <expr> ";" | ";"
auto Parser::ParseProcDecl() -> Ptr<ParsedProcDecl> {
    // Parse signature.
    SmallVector<ParsedLocalDecl*, 10> param_decls;
    Signature sig;
    if (not ParseSignature(sig, &param_decls)) return nullptr;

    // If we failed to parse a return type, or if there was
    // none, just default to void instead, or deduce the type
    // if this is a '= <expr>' declaration.
    if (sig.ret.invalid()) {
        sig.ret = new (*this) ParsedBuiltinType(
            At(Tk::Assign) ? Types::DeducedTy.ptr() : Types::VoidTy.ptr(),
            sig.proc_loc
        );
    }

    // Parse body.
    Ptr<ParsedStmt> body;
    if (Consume(Tk::Assign)) {
        body = ParseExpr();
        if (not Consume(Tk::Semicolon)) Error("Expected ';'");
    } else if (not At(Tk::Semicolon)) {
        body = ParseBlock();
    }

    if (sig.attrs.extern_ != not bool(body)) Error(
        sig.proc_loc,
        "Procedure that is{} declared '%1(extern)' must{} have a body",
        sig.attrs.extern_ ? ""sv : " not"sv,
        sig.attrs.extern_ ? " not"sv : ""sv
    );

    return ParsedProcDecl::Create(
        *this,
        sig.name,
        CreateType(sig),
        param_decls,
        body,
        sig.name.empty() ? sig.proc_loc : sig.tok_after_proc
    );
}

// <signature>  ::= PROC [ IDENTIFIER ] [ <proc-args> ] <proc-attrs> [ "->" <type> ]
// <proc-args>  ::= "(" [ <param-decl> { "," <param-decl> } [ "," ] ] ")"
// <proc-attrs> ::= { "native" | "extern" | "nomangle" }
// <param-decl> ::= [ <intent> ] <type> [ IDENTIFIER ] | [ <intent> ] <signature>
bool Parser::ParseSignature(
    Signature& sig,
    SmallVectorImpl<ParsedLocalDecl*>* decls
) {
    // Yeet 'proc'.
    sig.proc_loc = Next();

    // Parse name.
    sig.tok_after_proc = tok->location;
    if (At(Tk::Identifier)) {
        sig.name = tok->text;
        Next();
    }

    // Parse params.
    if (Consume(Tk::LParen) and not Consume(Tk::RParen)) {
        do {
            Ptr<ParsedStmt> type;
            String name;
            Location name_loc;

            // Parse intent.
            auto [start_loc, intent] = ParseIntent();
            if (intent == Intent::Move) start_loc = tok->location;

            // And do it again; two intents are an error.
            else if (auto [loc, i] = ParseIntent(); i != Intent::Move) {
                Error(loc, "Cannot specify more than one parameter intent");
                SkipTo(Tk::Comma, Tk::RParen);
                continue;
            }

            // Special handling for signatures, which may have
            // a name in this position.
            if (At(Tk::Proc)) {
                Signature inner;
                if (not ParseSignature(inner, nullptr)) {
                    SkipTo(Tk::Comma, Tk::RParen);
                    continue;
                }

                type = CreateType(inner);
                name = inner.name;
                name_loc = inner.tok_after_proc;
            }

            // Otherwise, parse a regular type and a name if we’re
            // creating declarations.
            else {
                type = ParseType();
                if (not type) {
                    SkipTo(Tk::Comma, Tk::RParen);
                    continue;
                }
            }

            sig.param_types.emplace_back(intent, type.get());

            // If decls is not null, then we allow named parameters here; parse
            // the name if there is one and create the declaration.
            if (decls) {
                Location end = type.get()->loc;
                if (At(Tk::Identifier)) {
                    if (not name.empty()) {
                        Error("Parameter cannot have two names");
                        Note(name_loc, "Name was already specified here");
                    }

                    name = tok->text;
                    end = Next();
                } else if (not At(Tk::Comma, Tk::RParen)) {
                    Error("Expected parameter name, '%1(,)', or '%1(\033))'");
                    if (IsKeyword(tok->type)) Remark(
                        "'%1({})' is not a valid parameter name because it is "
                        "a reserved word.",
                        tok->text
                    );

                    SkipTo(Tk::Comma, Tk::RParen);
                }

                decls->push_back(new (*this) ParsedLocalDecl{name, type.get(), {start_loc, end}, intent});
            } else if (At(Tk::Identifier)) {
                Error("Named parameters are not allowed here");
                Next();
            }
        } while (Consume(Tk::Comma));
        if (not Consume(Tk::RParen)) Error("Expected '\033)'");
    }

    // Parse attributes.
    auto ParseAttr = [&](bool& attr, StringRef value) {
        if (Location loc; ConsumeContextual(loc, value)) {
            if (attr) Warn(loc, "Duplicate '%1({})' attribute", value);
            attr = true;
            return true;
        }

        return false;
    };

    while (
        ParseAttr(sig.attrs.extern_, "extern") or
        ParseAttr(sig.attrs.native, "native") or
        ParseAttr(sig.attrs.nomangle, "nomangle")
    );

    // Parse return type.
    if (Consume(Tk::RArrow)) sig.ret = ParseType();
    return true;
}

// <stmt>  ::= <expr>
//           | <decl>
//           | <stmt-while>
//           | EVAL <stmt>
auto Parser::ParseStmt() -> Ptr<ParsedStmt> {
    auto loc = tok->location;
    switch (tok->type) {
        case Tk::Eval: {
            Next();
            auto arg = TryParseStmt();
            return new (*this) ParsedEvalExpr{arg, loc};
        }

        case Tk::Export: {
            Next();

            // Complain if this isn’t a module, but parse it anyway.
            if (not mod->is_module) {
                Error(loc, "'%1(export)' is only allowed in modules");
                Note(stream.begin()->location,
                     "If you meant to create a module (i.e. a static or shared "
                     "library\033), use '%1(module)' instead of '%1(program)'");
            }

            // It’s easier to parse any statement and then disallow
            // if we parsed something that doesn’t belong here.
            auto arg = TryParseStmt();
            auto decl = dyn_cast<ParsedDecl>(arg);
            if (not decl) return Error(arg->loc, "Only declarations can be exported");

            // Avoid forcing Sema to deal with unwrapping nested
            // export declarations.
            if (isa<ParsedExportDecl>(decl)) return Error({loc, decl->loc}, "'%1(export export)' is invalid");

            // The decl must have a name to be exportable.
            if (decl->name.empty()) return Error(decl->loc, "Anonymous declarations cannot be exported");
            return new (*this) ParsedExportDecl{decl, loc};
        }

        // <decl-struct>
        case Tk::Struct: return ParseStructDecl();

        // <stmt-while> ::= WHILE <expr> [ DO ] <stmt>
        case Tk::While: {
            Next();
            auto cond = TryParseExpr(SkipTo(Tk::Semicolon));
            Consume(Tk::Do);
            auto body = TryParseStmt(SkipTo(Tk::Semicolon));
            return new (*this) ParsedWhileStmt{cond, body, loc};
        }

        case Tk::Proc: return ParseProcDecl();
        default: return TryParseExpr(SkipTo(Tk::Semicolon));
    }
}

// <stmts> ::= { <stmt> [ ";" ] | <decl> }
void Parser::ParseStmts(SmallVectorImpl<ParsedStmt*>& stmts) {
    while (not At(Tk::Eof, Tk::RBrace)) {
        if (auto s = ParseStmt()) stmts.push_back(s.get());
        Consume(Tk::Semicolon);
    }
}

// <decl-struct> ::= STRUCT IDENTIFIER "{" { <type> IDENTIFIER ";" } "}"
auto Parser::ParseStructDecl() -> Ptr<ParsedStructDecl> {
    auto struct_loc = Next();

    // Name.
    String name = tok->text;
    if (not Consume(Tk::Identifier)) {
        Error("Expected identifier after '%1(struct)'");
        SkipTo(Tk::Semicolon);
        return {};
    }

    // Body.
    SmallVector<ParsedFieldDecl*> fields;
    if (not ExpectAndConsume(Tk::LBrace, "Expected '{{'")) return {};
    while (not At(Tk::RBrace)) {
        auto ty = ParseType();
        if (not ty) {
            SkipTo(Tk::RBrace, Tk::Semicolon);
            break;
        }

        String field_name = tok->text;
        if (not Consume(Tk::Identifier)) {
            Error("Expected identifier");
            SkipTo(Tk::RBrace, Tk::Semicolon);
            break;
        }

        fields.push_back(new (*this) ParsedFieldDecl{field_name, ty.get(), {ty.get()->loc, tok->location}});
        if (not ExpectAndConsume(Tk::Semicolon, "Expected ';'")) SkipTo(Tk::Semicolon);
    }

    if (not Consume(Tk::RBrace)) Error("Expected '}}'");
    return ParsedStructDecl::Create(*this, name, fields, struct_loc);
}

// <type> ::= <type-prim> | TEMPLATE-TYPE | <expr-decl-ref> | <signature> | <type-qualified>
// <type-qualified> ::= <type> { <qualifier> }
// <qualifier> ::= "[" "]"
auto Parser::ParseType() -> Ptr<ParsedStmt> {
    auto ty = ParseTypeStart();
    if (not ty) return {};
    return ParseTypeRest(ty.get());
}

auto Parser::ParseTypeRest(ParsedStmt* ty) -> Ptr<ParsedStmt> {
    // FIXME: This should just get merged into the expression parser.
    switch (tok->type) {
        default: return ty;
        case Tk::LBrack: {
            Next();
            Location loc;
            if (not Consume(loc, Tk::RBrack)) {
                Error("Expected ']'");
                SkipTo(Tk::Semicolon);
                return {};
            }

            ty = new (*this) ParsedSliceType(ty, {ty->loc, loc});
            return ParseTypeRest(ty);
        }
    }
}

auto Parser::ParseTypeStart() -> Ptr<ParsedStmt> {
    auto Builtin = [&](BuiltinType* ty) {
        return new (*this) ParsedBuiltinType(ty, Next());
    };

    switch (tok->type) {
        default: return Error("Expected type");

        // <type-prim> ::= BOOL | INT | VOID | VAR
        case Tk::Bool: return Builtin(Types::BoolTy.ptr());
        case Tk::Int: return Builtin(Types::IntTy.ptr());
        case Tk::Void: return Builtin(Types::VoidTy.ptr());
        case Tk::Var: return Builtin(Types::DeducedTy.ptr());

        // INTEGER_TYPE
        case Tk::IntegerType:
            if (not tok->integer.isSingleWord()) return Error("Integer type too large");
            return new (*this) ParsedIntType(Size::Bits(tok->integer.getZExtValue()), Next());

        // TEMPLATE-TYPE
        case Tk::TemplateType: {
            // Drop the '$' from the type.
            auto ty = new (*this) ParsedTemplateType(tok->text.drop(), tok->location);
            Next();
            return ty;
        }

        // <expr-decl-ref>
        case Tk::Identifier:
            return ParseDeclRefExpr();

        // <signature>
        case Tk::Proc: {
            Signature sig;
            if (not ParseSignature(sig, nullptr)) return nullptr;

            // A procedure name is not allowed here. If you want
            // to allow a name, call ParseSignature instead.
            if (not sig.name.empty()) Error(
                sig.tok_after_proc,
                "A name is not allowed in a procedure type"
            );

            return CreateType(sig);
        }
    }
}

