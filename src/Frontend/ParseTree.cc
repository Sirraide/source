#include <srcc/AST/Enums.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/SmallString.h>

#include <memory>
#include <utility>

using namespace srcc;

// ============================================================================
//  Parse Tree
// ============================================================================
void ParsedModule::dump() const {
    bool c = context().use_colours;

    // Print preamble.
    utils::Print(c, "%1({}%) {}\n", is_module ? "Module" : "Program", name);
    for (auto i : imports) {
        utils::Print(
            c,
            "%1(Import%) %4({}%) %1(as%) %4({}%)",
            utils::join(i.linkage_names),
            i.import_name
        );

        if (i.is_header_import) utils::Print(c, " header-import");
        utils::Print(c, "\n");
    }

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
    SLoc loc
) : ParsedType{Kind::ProcType, loc},
    num_params{u32(params.size())},
    ret_type{ret_type},
    attrs{attrs} {
    std::uninitialized_copy_n(
        params.begin(),
        params.size(),
        getTrailingObjects()
    );
}

auto ParsedProcType::Create(
    Parser& parser,
    ParsedStmt* ret_type,
    ArrayRef<ParsedParameter> params,
    ParsedProcAttrs attrs,
    SLoc loc
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
    SLoc location
) : ParsedStmt{Kind::BlockExpr, location},
    num_stmts{u32(stmts.size())} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects());
}

auto ParsedBlockExpr::Create(
    Parser& parser,
    ArrayRef<ParsedStmt*> stmts,
    SLoc location
) -> ParsedBlockExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(stmts.size());
    auto mem = parser.allocate(size, alignof(ParsedBlockExpr));
    return ::new (mem) ParsedBlockExpr{stmts, location};
}

ParsedCallExpr::ParsedCallExpr(
    ParsedStmt* callee,
    ArrayRef<ParsedStmt*> args,
    SLoc location
) : ParsedStmt{Kind::CallExpr, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects());
}

auto ParsedCallExpr::Create(
    Parser& parser,
    ParsedStmt* callee,
    ArrayRef<ParsedStmt*> args,
    SLoc location
) -> ParsedCallExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(args.size());
    auto mem = parser.allocate(size, alignof(ParsedCallExpr));
    return ::new (mem) ParsedCallExpr{callee, args, location};
}

ParsedDeclRefExpr::ParsedDeclRefExpr(ArrayRef<DeclName> names, SLoc location)
    : ParsedStmt(Kind::DeclRefExpr, location), num_parts(u32(names.size())) {
    std::uninitialized_copy_n(names.begin(), names.size(), getTrailingObjects());
}

auto ParsedDeclRefExpr::Create(
    Parser& parser,
    ArrayRef<DeclName> names,
    SLoc location
) -> ParsedDeclRefExpr* {
    const auto size = totalSizeToAlloc<DeclName>(names.size());
    auto mem = parser.allocate(size, alignof(ParsedDeclRefExpr));
    return ::new (mem) ParsedDeclRefExpr{names, location};
}

ParsedMatchExpr::ParsedMatchExpr(
    Ptr<ParsedStmt> control_expr,
    Ptr<ParsedStmt> declared_type,
    ArrayRef<ParsedMatchCase> cases,
    SLoc loc
) : ParsedStmt{Kind::MatchExpr, loc},
    num_cases{u32(cases.size())},
    has_control_expr{control_expr.present()},
    has_type{declared_type.present()} {
    if (auto c = control_expr.get_or_null()) *getTrailingObjects<ParsedStmt*>() = c;
    if (auto t = declared_type.get_or_null()) *(getTrailingObjects<ParsedStmt*>() + has_control_expr) = t;
    std::uninitialized_copy_n(cases.begin(), cases.size(), getTrailingObjects<ParsedMatchCase>());
}

auto ParsedMatchExpr::Create(
    Parser& p,
    Ptr<ParsedStmt> control_expr,
    Ptr<ParsedStmt> declared_type,
    ArrayRef<ParsedMatchCase> cases,
    SLoc loc
) -> ParsedMatchExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*, ParsedMatchCase>(
        unsigned(control_expr.present()) + unsigned(declared_type.present()),
        cases.size()
    );

    auto mem = p.allocate(size, alignof(ParsedMatchExpr));
    return ::new (mem) ParsedMatchExpr(control_expr, declared_type, cases, loc);
}

ParsedQuoteExpr::ParsedQuoteExpr(
    TokenStream* tokens,
    ArrayRef<ParsedUnquoteExpr*> unquotes,
    bool brace_delimited,
    SLoc location
) : ParsedStmt{Kind::QuoteExpr, location},
    token_stream{tokens},
    num_unquotes{u32(unquotes.size())},
    brace_delimited{brace_delimited} {
    std::uninitialized_copy_n(unquotes.begin(), unquotes.size(), getTrailingObjects());
}

auto ParsedQuoteExpr::Create(
    Parser& p,
    TokenStream* tokens,
    ArrayRef<ParsedUnquoteExpr*> unquotes,
    bool brace_delimited,
    SLoc location
) -> ParsedQuoteExpr* {
    const auto size = totalSizeToAlloc<ParsedUnquoteExpr*>(unquotes.size());
    auto mem = p.allocate(size, alignof(ParsedQuoteExpr));
    return ::new (mem) ParsedQuoteExpr(tokens, unquotes, brace_delimited, location);
}

ParsedTupleExpr::ParsedTupleExpr(ArrayRef<ParsedStmt*> exprs, SLoc loc)
    : ParsedStmt{Kind::TupleExpr, loc}, num_exprs{u32(exprs.size())} {
    std::uninitialized_copy_n(exprs.begin(), exprs.size(), getTrailingObjects());
}

auto ParsedTupleExpr::Create(
    Parser& p,
    ArrayRef<ParsedStmt*> exprs,
    SLoc loc
) -> ParsedTupleExpr* {
    const auto size = totalSizeToAlloc<ParsedStmt*>(exprs.size());
    auto mem = p.allocate(size, alignof(ParsedTupleExpr));
    return ::new (mem) ParsedTupleExpr(exprs, loc);
}

ParsedForStmt::ParsedForStmt(
    SLoc for_loc,
    SLoc enum_loc,
    String enum_name,
    ArrayRef<LoopVar> vars,
    ArrayRef<ParsedStmt*> ranges,
    ParsedStmt* body
) : ParsedStmt{Kind::ForStmt, for_loc},
    num_idents{u32(vars.size())},
    num_ranges{u32(ranges.size())},
    enum_loc{enum_loc},
    enum_name{enum_name},
    body{body}{
    std::uninitialized_copy_n(ranges.begin(), num_ranges, getTrailingObjects<ParsedStmt*>());
    std::uninitialized_copy_n(vars.begin(), num_idents, getTrailingObjects<LoopVar>());
}

auto ParsedForStmt::Create(
    Parser& parser,
    SLoc for_loc,
    SLoc enum_loc,
    String enum_name,
    ArrayRef<LoopVar> vars,
    ArrayRef<ParsedStmt*> ranges,
    ParsedStmt* body
) -> ParsedForStmt* {
    const auto size = totalSizeToAlloc<LoopVar, ParsedStmt*>(vars.size(), ranges.size());
    auto mem = parser.allocate(size, alignof(ParsedForStmt));
    return ::new (mem) ParsedForStmt{for_loc, enum_loc, enum_name, vars, ranges, body};
}

ParsedIntLitExpr::ParsedIntLitExpr(Parser& p, APInt value, SLoc loc)
    : ParsedStmt{Kind::IntLitExpr, loc},
      storage{p.module().integers.store_int(std::move(value))} {}

// ============================================================================
//  Declarations
// ============================================================================
ParsedEnumDecl::ParsedEnumDecl(
    DeclName name,
    ArrayRef<ParsedEnumerator> enumerators,
    SLoc location
) : ParsedDecl{Kind::EnumDecl, name, location}, num_enumerators{u32(enumerators.size())} {
    llvm::uninitialized_copy(enumerators, getTrailingObjects());
}

auto ParsedEnumDecl::Create(
    Parser& p,
    DeclName name,
    ArrayRef<ParsedEnumerator> enumerators,
    SLoc location
) -> ParsedEnumDecl* {
    auto size = totalSizeToAlloc<ParsedEnumerator>(enumerators.size());
    auto mem = p.allocate(size, alignof(ParsedEnumDecl));
    return ::new (mem) ParsedEnumDecl(name, enumerators, location);
}

ParsedProcDecl::ParsedProcDecl(
    DeclName name,
    ParsedProcType* type,
    ArrayRef<ParsedVarDecl*> param_decls,
    Ptr<ParsedStmt> body,
    Ptr<ParsedStmt> where,
    SLoc location
) : ParsedDecl{Kind::ProcDecl, name, location},
    body{body},
    type{type},
    where{where} {
    std::uninitialized_copy_n(
        param_decls.begin(),
        param_decls.size(),
        getTrailingObjects()
    );
}

auto ParsedProcDecl::Create(
    Parser& parser,
    DeclName name,
    ParsedProcType* type,
    ArrayRef<ParsedVarDecl*> param_decls,
    Ptr<ParsedStmt> body,
    Ptr<ParsedStmt> where,
    SLoc location
) -> ParsedProcDecl* {
    const auto size = totalSizeToAlloc<ParsedVarDecl*>(param_decls.size());
    auto mem = parser.allocate(size, alignof(ParsedProcDecl));
    return ::new (mem) ParsedProcDecl{name, type, param_decls, body, where, location};
}

ParsedStructDecl::ParsedStructDecl(
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    SLoc loc
) : ParsedDecl{Kind::StructDecl, name, loc}, num_fields(u32(fields.size())) {
    std::uninitialized_copy_n(
        fields.begin(),
        fields.size(),
        getTrailingObjects()
    );
}

auto ParsedStructDecl::Create(
    Parser& parser,
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    SLoc loc
) -> ParsedStructDecl* {
    const auto size = totalSizeToAlloc<ParsedFieldDecl*>(fields.size());
    auto mem = parser.allocate(size, alignof(ParsedStructDecl));
    return ::new (mem) ParsedStructDecl{name, fields, loc};
}

// ============================================================================
//  Tree Printer
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

    using PrinterBase::Print;
    void PrintHeader(ParsedStmt* s, StringRef name, bool full = true);
    void Print(ParsedStmt* s);
};

void ParsedStmt::Printer::PrintHeader(ParsedStmt* s, StringRef name, bool full) {
    auto lc = module ? s->loc.seek_line_column(module->context()) : std::nullopt;
    print(
        "%1({}%) %4({}%) %5(<{}:{}>%)",
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
#       define PARSE_TREE_LEAF_TYPE(n) case Kind::n:
#       include "srcc/ParseTree.inc"
            print("%1(Type%) {}\n", s->dump_as_type());
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
            print("%1({}%)\n", utils::Escape(Spelling(b.op), false, true));
            SmallVector<ParsedStmt*, 2> children{b.lhs, b.rhs};
            PrintChildren(children);
        } break;

        case Kind::BoolLitExpr: {
            PrintHeader(s, "BoolLitExpr", false);
            print("%1({}%)\n", cast<ParsedBoolLitExpr>(s)->value);
        } break;

        case Kind::BreakContinueExpr: {
            PrintHeader(s, "BreakContinueExpr", false);
            print(
                "%1({}%)\n",
                cast<ParsedBreakContinueExpr>(s)->is_continue ? "continue"sv : "break"sv
            );
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
            print("%8({}%)\n", utils::join(d.names(), "::"));
        } break;

        case Kind::DeferStmt: {
            PrintHeader(s, "DeferStmt");
            PrintChildren(cast<ParsedDeferStmt>(s)->body);
        } break;

        case Kind::EmptyStmt:
            PrintHeader(s, "EmptyStmt");
            break;

        case Kind::EnumDecl: {
            auto e = cast<ParsedEnumDecl>(s);
            PrintHeader(s, "EnumDecl");
            SmallVector<Child> children;
            for (auto& enumerator : e->enumerators()) children.emplace_back([&]{
                // FIXME: Print location.
                print("%1(Enumerator%) %4({}%)\n", enumerator.name);
                if (enumerator.value) PrintChildren(enumerator.value.get());
            });
            PrintChildren<Child>(children);
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
            print("%5({}%) {}\n", f.name, f.type->dump_as_type());
        } break;

        case Kind::ForStmt: {
            auto& f = *cast<ParsedForStmt>(s);
            PrintHeader(s, "ForStmt", false);

            if (f.has_enumerator()) {
                print(" %1(enum%) %8({}%)", f.enum_name);
                if (not f.vars().empty()) print("%1(,%) ");
            }

            for (auto [i, v] : enumerate(f.vars())) {
                if (i != 0) print("%1(,%) ");
                print(" %8({}%)", f.enum_name);
            }

            print("\n");
            SmallVector<ParsedStmt*> children{f.ranges()};
            children.push_back(f.body);
            PrintChildren(children);
        } break;

        case Kind::IfExpr: {
            auto& i = *cast<ParsedIfExpr>(s);
            PrintHeader(s, "IfExpr", not i.is_static);
            if (i.is_static) print("static\n");
            SmallVector<ParsedStmt*, 3> children{i.cond, i.then};
            if (auto e = i.else_.get_or_null()) children.push_back(e);
            PrintChildren(children);
        } break;

        case Kind::InjectExpr: {
            PrintHeader(s, "InjectExpr");
            PrintChildren(cast<ParsedInjectExpr>(s)->injected);
        } break;

        case Kind::IntLitExpr: {
            PrintHeader(s, "IntLitExpr", false);
            auto val = cast<ParsedIntLitExpr>(s)->storage.str(false);
            print("%5({}%)\n", val);
        } break;

        case Kind::LoopExpr: {
            PrintHeader(s, "LoopExpr");
            if (auto b = cast<ParsedLoopExpr>(s)->body.get_or_null()) PrintChildren(b);
        } break;

        case Kind::MatchExpr: {
            auto m = cast<ParsedMatchExpr>(s);
            PrintHeader(s, "MatchExpr");
            SmallVector<ParsedStmt*, 8> children{};
            if (auto e = m->control_expr().get_or_null()) children.push_back(e);
            if (auto e = m->declared_type().get_or_null()) children.push_back(e);
            for (auto c : m->cases()) {
                children.push_back(c.cond);
                children.push_back(c.body);
            }
            PrintChildren(children);
        } break;

        case Kind::MemberExpr: {
            auto& m = *cast<ParsedMemberExpr>(s);
            PrintHeader(s, "MemberExpr", false);
            print("%8({}%)\n", m.member);
            PrintChildren(m.base);
        } break;


        case Kind::NilExpr:
            PrintHeader(s, "NilExpr");
            break;

        case Kind::VarDecl: {
            auto& p = *cast<ParsedVarDecl>(s);
            PrintHeader(s, "VarDecl", false);
            print("%4({}%){}", p.name, p.name.empty() ? "" : " ");
            if (p.intent != Intent::Move) print("%1({}%) ", p.intent);
            if (p.is_static) print("static ");
            if (p.with_loc.is_valid()) print("with ");
            print("{}\n", p.type->dump_as_type());
            if (p.init) PrintChildren(p.init.get());
        } break;

        case Kind::QuoteExpr: {
            auto q = cast<ParsedQuoteExpr>(s);
            PrintHeader(s, "QuoteExpr");
            SmallVector<Child> children;
            u32 unquote_idx = 0;
            for (const auto &t : q->tokens()) {
                if (t.is(Tk::Unquote)) children.emplace_back([this, expr = q->unquotes()[unquote_idx++]] { Print(expr); });
                else children.emplace_back([this, &t] {
                    if (module) print("%1(Token%) {}\n", t.location.text(module->context()));
                    else print("%1(Token%) {}\n", Spelling(t.type));
                });
            }
            PrintChildren<Child>(children);
        } break;

        case Kind::UnquoteExpr: {
            PrintHeader(s, "UnquoteExpr");
            PrintChildren(cast<ParsedUnquoteExpr>(s)->arg);
        } break;

        case Kind::ParenExpr: {
            PrintHeader(s, "ParenExpr");
            PrintChildren(cast<ParsedParenExpr>(s)->inner);
        } break;

        case Kind::ProcDecl: {
            auto& p = *cast<ParsedProcDecl>(s);
            PrintHeader(s, "ProcDecl", false);
            print(
                "%2({}%){}{}\n",
                utils::Escape(p.name.str(), false, true),
                p.name.empty() ? ""sv : " "sv,
                p.type->dump_as_type()
            );

            // No need to print the param decls here.
            SmallVector<ParsedStmt*, 10> children;
            if (auto b = p.body.get_or_null()) children.push_back(b);
            if (auto where = p.where.get_or_null()) children.push_back(where);
            PrintChildren(children);
        } break;

        case Kind::StrLitExpr: {
            auto& str = *cast<ParsedStrLitExpr>(s);
            PrintHeader(s, "StrLitExpr", false);
            print("%3(\"{}\"%)\n", utils::Escape(str.value, true, true));
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ParsedReturnExpr>(s);
            PrintHeader(s, "ReturnExpr");
            if (auto val = ret->value.get_or_null()) PrintChildren(val);
        } break;

        case Kind::StructDecl: {
            auto& d = *cast<ParsedStructDecl>(s);
            PrintHeader(s, "StructDecl", false);
            print("%6({}%)\n", d.name);
            PrintChildren<ParsedFieldDecl*>(d.fields());
        } break;

        case Kind::TupleExpr: {
            auto t = cast<ParsedTupleExpr>(s);
            PrintHeader(s, "TupleExpr");
            PrintChildren(t->exprs());
        } break;

        case Kind::UnaryExpr: {
            auto& u = *cast<ParsedUnaryExpr>(s);
            PrintHeader(s, "UnaryExpr", false);
            print("%1({}%)", u.op);
            if (u.postfix) print(" %3(postfix%)");
            print("\n");
            PrintChildren(u.arg);
        } break;

        case Kind::WhileStmt: {
            auto& w = *cast<ParsedWhileStmt>(s);
            PrintHeader(s, "WhileStmt");
            SmallVector<ParsedStmt*, 2> children{w.cond, w.body};
            PrintChildren(children);
        } break;

        case Kind::WithExpr: {
            auto& w = *cast<ParsedWithExpr>(s);
            PrintHeader(s, "WithExpr");
            SmallVector<ParsedStmt*, 2> children{w.expr, w.body};
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
                Format(out, "%6(i{:i}%)", cast<ParsedIntType>(type)->bit_width);
                break;

            case Kind::OptionalType: {
                auto p = cast<ParsedOptionalType>(type);
                Append(p->elem);
                out += "%1(?%)";
            } break;

            case Kind::ProcType: {
                auto p = cast<ParsedProcType>(type);
                out += "%1(proc";

                if (not p->param_types().empty()) {
                    bool first = true;
                    out += " (";

                    for (const auto& param : p->param_types()) {
                        if (not first) out += ", ";
                        first = false;
                        if (param.intent != Intent::Move) Format(out, "{} ", param.intent);
                        Append(param.type);
                        if (param.variadic) out += "...";
                    }

                    out += ")";
                }

                if (p->attrs.native) out += " native";
                if (p->attrs.extern_) out += " extern";
                if (p->attrs.nomangle) out += " nomangle";
                if (p->attrs.c_varargs) out += " varargs";

                out += " -> %)";
                Append(p->ret_type);
            } break;

            case Kind::PtrType: {
                auto p = cast<ParsedPtrType>(type);
                Append(p->elem);
                out += "%1(^%)";
            } break;

            case Kind::RangeType: {
                auto s = cast<ParsedRangeType>(type);
                out += "%6(range%)%1(<%)";
                Append(s->elem);
                out += "%1(>%)";
            } break;

            case Kind::SliceType: {
                auto s = cast<ParsedSliceType>(type);
                Append(s->elem);
                out += "%1([]%)";
            } break;

            case Kind::TemplateType: {
                auto t = cast<ParsedTemplateType>(type);
                Format(out, "%3(${}%)", t->name);
            } break;

            case Kind::TypeofType: {
                Format(out, "%1(typeof(%)<expr>%1()%)");
            } break;

            case Kind::DeclRefExpr: {
                auto d = cast<ParsedDeclRefExpr>(type);
                Format(out, "%8({}%)", utils::join(d->names(), "::"));
            } break;

            case Kind::BinaryExpr: {
                auto e = cast<ParsedBinaryExpr>(type);
                if (e->op == Tk::LBrack) {
                    Append(e->lhs);
                    out += "%1([%)<expr>%1(]%)";
                    break;
                }

                out += "<invalid type>";
                break;
            }

            case Kind::ParenExpr: {
                auto p = cast<ParsedParenExpr>(type);
                out += "%1((";
                Append(p->inner);
                out += ")%)";
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
