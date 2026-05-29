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
            i.import_name.empty() ? "*" : i.import_name
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

ParsedDeclRefExpr::ParsedDeclRefExpr(
    InitialDREScope scope,
    Ptr<ParsedStmt> root,
    ArrayRef<DeclNameLoc> names,
    SLoc location
) : ParsedStmt(Kind::DeclRefExpr, location),
    num_parts(u32(names.size())),
    root_scope{root, scope} {
    std::uninitialized_copy_n(names.begin(), names.size(), getTrailingObjects());
    Assert(not empty(), "Empty DRE!");
}

auto ParsedDeclRefExpr::Create(
    Parser& parser,
    InitialDREScope scope,
    Ptr<ParsedStmt> root,
    ArrayRef<DeclNameLoc> names,
    SLoc location
) -> ParsedDeclRefExpr* {
    const auto size = totalSizeToAlloc<DeclNameLoc>(names.size());
    auto mem = parser.allocate(size, alignof(ParsedDeclRefExpr));
    return ::new (mem) ParsedDeclRefExpr{scope, root, names, location};
}

bool ParsedDeclRefExpr::empty() const {
    return names().empty();
}

bool ParsedDeclRefExpr::is_single_ident() const {
    return num_parts == 1 and
           names().front().name.is_str() and
           initial_scope() == InitialDREScope::None;
}

auto ParsedDeclRefExpr::to_string() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (initial_scope()) {
        case srcc::InitialDREScope::None: break;
        case srcc::InitialDREScope::Global: out += "%1(::%)"; break;
        case srcc::InitialDREScope::Expr: out += "<expr>"; break;
    }

    if (names().empty())
        return out;

    for (auto n : names().drop_back()) {
        out += utils::Escape(n.name.str(), false, true);
        out += "%1(::%)";
    }

    out += utils::Escape(names().back().name.str(), false, true);
    return out;
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
    Ptr<ParsedStmt> underlying_type,
    SLoc location
) : ParsedDecl{Kind::EnumDecl, name, location},
    num_enumerators{u32(enumerators.size())},
    underlying_type{underlying_type} {
    llvm::uninitialized_copy(enumerators, getTrailingObjects());
}

auto ParsedEnumDecl::Create(
    Parser& p,
    DeclName name,
    ArrayRef<ParsedEnumerator> enumerators,
    Ptr<ParsedStmt> underlying_type,
    SLoc location
) -> ParsedEnumDecl* {
    auto size = totalSizeToAlloc<ParsedEnumerator>(enumerators.size());
    auto mem = p.allocate(size, alignof(ParsedEnumDecl));
    return ::new (mem) ParsedEnumDecl(name, enumerators, underlying_type, location);
}

ParsedProcDecl::ParsedProcDecl(
    DeclName name,
    Ptr<ParsedStmt> associated_type,
    ParsedProcType* type,
    ArrayRef<ParsedVarDecl*> param_decls,
    Ptr<ParsedStmt> body,
    Ptr<ParsedStmt> where,
    SLoc location
) : ParsedDecl{Kind::ProcDecl, name, location},
    body{body},
    type{type},
    where{where},
    associated_type{associated_type} {
    std::uninitialized_copy_n(
        param_decls.begin(),
        param_decls.size(),
        getTrailingObjects()
    );
}

auto ParsedProcDecl::Create(
    Parser& parser,
    DeclName name,
    Ptr<ParsedStmt> associated_type,
    ParsedProcType* type,
    ArrayRef<ParsedVarDecl*> param_decls,
    Ptr<ParsedStmt> body,
    Ptr<ParsedStmt> where,
    SLoc location
) -> ParsedProcDecl* {
    const auto size = totalSizeToAlloc<ParsedVarDecl*>(param_decls.size());
    auto mem = parser.allocate(size, alignof(ParsedProcDecl));
    return ::new (mem) ParsedProcDecl{
        name,
        associated_type,
        type,
        param_decls,
        body,
        where,
        location,
    };
}

ParsedStructDecl::ParsedStructDecl(
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    Ptr<ParsedStmt> deleter,
    SLoc loc
) : ParsedDecl{Kind::StructDecl, name, loc},
    num_fields(u32(fields.size())),
    has_deleter(deleter.present()) {
    std::uninitialized_copy_n(
        fields.begin(),
        fields.size(),
        getTrailingObjects<ParsedFieldDecl*>()
    );

    if (deleter.present())
        *getTrailingObjects<ParsedStmt*>() = deleter.get();
}

auto ParsedStructDecl::Create(
    Parser& parser,
    String name,
    ArrayRef<ParsedFieldDecl*> fields,
    Ptr<ParsedStmt> deleter,
    SLoc loc
) -> ParsedStructDecl* {
    const auto size = totalSizeToAlloc<ParsedFieldDecl*, ParsedStmt*>(fields.size(), deleter.present());
    auto mem = parser.allocate(size, alignof(ParsedStructDecl));
    return ::new (mem) ParsedStructDecl{name, fields, deleter, loc};
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
            return;

        default:
            PrintHeader(s, enchantum::to_string(s->kind()));
            break;

        case Kind::BinaryExpr: {
            auto& b = *cast<ParsedBinaryExpr>(s);
            PrintHeader(s, "BinaryExpr", false);
            print("%1({}%)\n", utils::Escape(Spelling(b.op), false, true));
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

        case Kind::DeclRefExpr: {
            auto& d = *cast<ParsedDeclRefExpr>(s);
            PrintHeader(s, "DeclRefExpr", false);
            print("%8({}%)\n", d.to_string());
        } break;

        case Kind::EnumDecl: {
            auto e = cast<ParsedEnumDecl>(s);
            PrintHeader(s, "EnumDecl", not e->underlying_type);
            if (e->underlying_type.present()) print("{}\n", e->underlying_type.get()->dump_as_type());
            SmallVector<Child> children;
            for (auto& enumerator : e->enumerators()) children.emplace_back([&]{
                // FIXME: Print location.
                print("%1(Enumerator%) %4({}%)\n", enumerator.name);
                if (enumerator.value) PrintChildren(enumerator.value.get());
            });
            PrintChildren<Child>(children);
            return;
        }

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
        } break;

        case Kind::IfExpr: {
            auto& i = *cast<ParsedIfExpr>(s);
            PrintHeader(s, "IfExpr", not i.is_static);
            if (i.is_static) print("static\n");
        } break;

        case Kind::IntLitExpr: {
            PrintHeader(s, "IntLitExpr", false);
            auto val = cast<ParsedIntLitExpr>(s)->storage.str(false);
            print("%5({}%)\n", val);
        } break;

        case Kind::MemberExpr: {
            auto& m = *cast<ParsedMemberExpr>(s);
            PrintHeader(s, "MemberExpr", false);
            print("%8({}%)\n", m.member);
        } break;

        case Kind::VarDecl: {
            auto& p = *cast<ParsedVarDecl>(s);
            PrintHeader(s, "VarDecl", false);
            print("%4({}%){}", p.name, p.name.empty() ? "" : " ");
            if (p.intent != Intent::Move) print("%1({}%) ", p.intent);
            if (p.is_static) print("static ");
            if (p.is_this_param) print("this_param ");
            if (p.with_loc.is_valid()) print("with ");
            print("{}\n", p.type->dump_as_type());
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
            return;
        }

        case Kind::ProcDecl: {
            auto& p = *cast<ParsedProcDecl>(s);
            PrintHeader(s, "ProcDecl", false);
            if (auto a = p.associated_type.get_or_null()) print("{}%1(::%)", a->dump_as_type());
            if (not p.name.empty()) print("%2({}%) ", utils::Escape(p.name.str(), false, true));
            print("{}\n", p.type->dump_as_type());
        } break;

        case Kind::StrLitExpr: {
            auto& str = *cast<ParsedStrLitExpr>(s);
            PrintHeader(s, "StrLitExpr", false);
            print("%3(\"{}\"%)\n", utils::Escape(str.value, true, true));
        } break;

        case Kind::StructDecl: {
            auto& d = *cast<ParsedStructDecl>(s);
            PrintHeader(s, "StructDecl", false);
            print("%6({}%)\n", d.name);
        } break;

        case Kind::ThisExpr: {
            auto t = cast<ParsedThisExpr>(s);
            if (t->is_type) print("%1(Type%) {}\n", s->dump_as_type());
            else PrintHeader(s, "ThisExpr");
        } break;

        case Kind::UnaryExpr: {
            auto& u = *cast<ParsedUnaryExpr>(s);
            PrintHeader(s, "UnaryExpr", false);
            print("%1({}%)", u.op);
            if (u.postfix) print(" %3(postfix%)");
            print("\n");
        } break;
    }

    s->children(false).range.visit([&](auto&& r) { PrintChildren(r); });
}

auto ParsedStmt::children(bool include_types) -> Children {
    auto SingleType = [&] (ParsedStmt* ty) -> Children {
        if (include_types) return ty;
        return {};
    };

    return visit(utils::Overloaded{
        // Types.
        [&](ParsedBuiltinType*) -> Children { return {}; },
        [&](ParsedIntType*) -> Children { return {}; },
        [&](ParsedOptionalType* o) -> Children { return SingleType(o->elem); },
        [&](ParsedProcType* p) -> Children {
            if (not include_types) return {};
            Children::Owning children;
            for (auto param : p->param_types())
                children.push_back(param.type);
            children.push_back(p->ret_type);
            return std::move(children);
        },

        [&](ParsedPtrType* p) -> Children { return SingleType(p->elem); },
        [&](ParsedRangeType* r) -> Children { return SingleType(r->elem); },
        [&](ParsedSliceType* s) -> Children { return SingleType(s->elem); },
        [&](ParsedTemplateType*) -> Children { return {}; },
        [&](ParsedTypeofType* t) -> Children { return SingleType(t->arg); },
        [&](ParsedValueType* v) -> Children { return SingleType(v->elem); },

        // Statements.
        [&](ParsedAssertExpr* a) -> Children {
            Children::Owning children;
            children.push_back(a->cond);
            if (auto msg = a->message.get_or_null()) children.push_back(msg);
            return std::move(children);
        },

        [&](ParsedBlockExpr* b) -> Children { return b->stmts(); },
        [&](ParsedBinaryExpr* b) -> Children { return {b->lhs, b->rhs}; },
        [&](ParsedBoolLitExpr*) -> Children { return {}; },
        [&](ParsedBreakContinueExpr*) -> Children { return {}; },
        [&](ParsedCallExpr* c) -> Children {
            Children::Owning children;
            if (c->callee) children.push_back(c->callee);
            if (auto a = c->args(); not a.empty()) children.append(a.begin(), a.end());
            return std::move(children);
        },

        [&](ParsedDeclRefExpr* d) -> Children {
            if (auto r = d->root().get_or_null()) return r;
            return {};
        },

        [&](ParsedDeferStmt* d) -> Children { return d->body; },
        [&](ParsedDeleteExpr* d) -> Children { return d->expr; },
        [&](ParsedEmptyStmt*) -> Children { return {}; },
        [&](ParsedEnumDecl* e) -> Children {
            Children::Owning children;
            if (include_types and e->underlying_type.present())
                children.push_back(e->underlying_type.get());
            for (auto& enumerator : e->enumerators())
                if (enumerator.value.present())
                    children.push_back(enumerator.value.get());
            return std::move(children);
        },

        [&](ParsedEvalExpr* e) -> Children { return e->expr; },
        [&](ParsedExportDecl* e) -> Children { return e->decl; },
        [&](ParsedFieldDecl* f) -> Children { return f->type; },
        [&](ParsedForStmt* f) -> Children {
            Children::Owning children{f->ranges()};
            children.push_back(f->body);
            return std::move(children);
        },

        [&](ParsedIfExpr* i) -> Children {
            Children::Owning children{i->cond, i->then};
            if (auto e = i->else_.get_or_null()) children.push_back(e);
            return std::move(children);
        },

        [&](ParsedInjectExpr* i) -> Children { return i->injected; },
        [&](ParsedIntLitExpr*) -> Children { return {}; },
        [&](ParsedLoopExpr* l) -> Children {
            if (auto b = l->body.get_or_null()) return b;
            return {};
        },

        [&](ParsedMatchExpr* m) -> Children {
            Children::Owning children;
            if (auto e = m->control_expr().get_or_null()) children.push_back(e);
            if (auto e = m->declared_type().get_or_null(); e and include_types) children.push_back(e);
            for (auto c : m->cases()) {
                children.push_back(c.cond);
                children.push_back(c.body);
            }
            return std::move(children);
        },

        [&](ParsedMemberExpr* m) -> Children {  return m->base; },
        [&](ParsedNilExpr*) -> Children { return {}; },
        [&](ParsedVarDecl* v) -> Children {
            Children::Owning children;
            if (include_types) children.push_back(v->type);
            if (v->init) children.push_back(v->init.get());
            return std::move(children);
        },

        [&](ParsedQuoteExpr* q) -> Children {
            return Children::NonOwning{
                reinterpret_cast<ParsedStmt* const*>(q->unquotes().data()),
                q->unquotes().size()
            };
        },
        [&](ParsedUnquoteExpr* u) -> Children { return u->arg; },
        [&](ParsedParenExpr* p) -> Children { return p->inner; },
        [&](ParsedProcDecl* p) -> Children {
            Children::Owning children;
            if (include_types) {
                children.push_back(p->type);
                if (auto a = p->associated_type.get_or_null()) children.push_back(a);
            }

            append_range(children, p->params());
            if (auto where = p->where.get_or_null()) children.push_back(where);
            if (auto b = p->body.get_or_null()) children.push_back(b);
            return std::move(children);
        },

        [&](ParsedStrLitExpr*) -> Children { return {}; },
        [&](ParsedReturnExpr* r) -> Children {
            if (auto val = r->value.get_or_null()) return val;
            return {};
        },

        [&](ParsedStructDecl* s) -> Children {
            auto del = s->deleter().get_or_null();
            if (not del) return Children::NonOwning{
                reinterpret_cast<ParsedStmt* const*>(s->fields().data()),
                s->fields().size()
            };

            Children::Owning children;
            append_range(children, s->fields());
            children.push_back(del);
            return std::move(children);
        },

        [&](ParsedThisExpr*) -> Children { return {}; },
        [&](ParsedTupleExpr* t) -> Children { return t->exprs(); },
        [&](ParsedUnaryExpr* u) -> Children { return u->arg; },
        [&](ParsedWhileStmt* w) -> Children {  return {w->cond, w->body}; },
        [&](ParsedWithExpr* w) -> Children { return {w->expr, w->body}; },
    });
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

            case Kind::ValueType: {
                auto p = cast<ParsedValueType>(type);
                Append(p->elem);
                out += " %1(val%)";
            } break;

            case Kind::DeclRefExpr: {
                auto d = cast<ParsedDeclRefExpr>(type);
                Format(out, "%8({}%)", d->to_string());
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

            case Kind::ThisExpr:
                out += "%1(This%)";
                break;

            default:
                out += "<invalid type>";
                break;
        }
    };

    Append(this);
    return out;
}

bool ParsedStmt::traverse_impl(llvm::function_ref<bool(ParsedStmt*)> visitor) {
    if (not visitor(this)) return false;
    for (auto c : children(true))
        if (not c->traverse_impl(visitor))
            return false;
    return true;
}

#define PARSE_TREE_NODE(node)                                                                                \
    static_assert(alignof(LIBBASE_CAT(Parsed, node)) < __STDCPP_DEFAULT_NEW_ALIGNMENT__, "Alignment to large"); \
    static_assert(__is_trivially_destructible(LIBBASE_CAT(Parsed, node)), "Parse tree nodes must be trivially destructible");

PARSE_TREE_NODE(Stmt);
#include "srcc/ParseTree.inc"
