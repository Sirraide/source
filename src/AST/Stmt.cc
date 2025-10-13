#include <srcc/AST/AST.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Frontend/Parser.hh>

#include <llvm/Support/Casting.h>

#include <memory>

using namespace srcc;

void* Stmt::operator new(usz size, TranslationUnit& mod) {
    return mod.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

// Sanity check.
#define AST_STMT_LEAF(node) static_assert(std::is_trivially_destructible_v<node>);
#include "srcc/AST.inc"

// ============================================================================
//  AST
// ============================================================================
ArrayInitExpr::ArrayInitExpr(
    ArrayType* type,
    ArrayRef<Expr*> elements,
    Location loc
) : Expr{Kind::ArrayInitExpr, type, RValue, loc}, num_inits{u32(elements.size())} {
    std::uninitialized_copy_n(elements.begin(), elements.size(), getTrailingObjects());
}

auto ArrayInitExpr::Create(
    TranslationUnit& tu,
    ArrayType* type,
    ArrayRef<Expr*> elements,
    Location loc
) -> ArrayInitExpr* {
    const auto size = totalSizeToAlloc<Expr*>(elements.size());
    auto mem = tu.allocate(size, alignof(ArrayInitExpr));
    return ::new (mem) ArrayInitExpr{type, elements, loc};
}

BuiltinCallExpr::BuiltinCallExpr(
    Builtin kind,
    Type return_type,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::BuiltinCallExpr, return_type, RValue, location}, builtin{kind}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects());
    // Determine value category.
    switch (builtin) {
        // RValue.
        case Builtin::Unreachable:
            break;
    }
}

auto BuiltinCallExpr::Create(
    TranslationUnit& tu,
    Builtin kind,
    Type return_type,
    ArrayRef<Expr*> args,
    Location location
) -> BuiltinCallExpr* {
    auto size = totalSizeToAlloc<Expr*>(args.size());
    auto mem = tu.allocate(size, alignof(BuiltinCallExpr));
    return ::new (mem) BuiltinCallExpr{kind, return_type, args, location};
}

CallExpr::CallExpr(
    Type type,
    ValueCategory vc,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::CallExpr, type, vc, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects());
}

auto CallExpr::Create(
    TranslationUnit& mod,
    Type type,
    ValueCategory vc,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) -> CallExpr* {
    const auto size = totalSizeToAlloc<Expr*>(args.size());
    auto mem = mod.allocate(size, alignof(CallExpr));
    return ::new (mem) CallExpr{type, vc, callee, args, location};
}

ConstExpr::ConstExpr(
    TranslationUnit& tu,
    eval::RValue value,
    Location location,
    Ptr<Stmt> stmt
) : Expr{Kind::ConstExpr, value.type(), RValue, location},
    value{tu.save(std::move(value))},
    stmt{stmt} {}

BlockExpr::BlockExpr(
    Scope* parent_scope,
    Type type,
    ArrayRef<Stmt*> stmts,
    Location location
) : Expr{Kind::BlockExpr, type, RValue, location},
    num_stmts{u32(stmts.size())},
    scope{parent_scope} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects());
    // The value category of this is that of the return expr.
    if (auto e = return_expr()) value_category = e->value_category;
}

auto BlockExpr::Create(
    TranslationUnit& mod,
    Scope* parent_scope,
    ArrayRef<Stmt*> stmts,
    Location location
) -> BlockExpr* {
    auto last = stmts.empty() ? nullptr : dyn_cast_if_present<Expr>(stmts.back());
    auto type = last ? last->type : Type::VoidTy;
    auto size = totalSizeToAlloc<Stmt*>(stmts.size());
    auto mem = mod.allocate(size, alignof(BlockExpr));
    return ::new (mem) BlockExpr{parent_scope, type, stmts, location};
}

auto BlockExpr::return_expr() -> Expr* {
    if (type->is_void()) return nullptr;
    return cast<Expr>(stmts().back());
}

auto Expr::ignore_parens() -> Expr* {
    if (auto p = dyn_cast<ParenExpr>(this)) return p->expr->ignore_parens();
    return this;
}

ForStmt::ForStmt(
    Ptr<LocalDecl> enum_var,
    ArrayRef<LocalDecl*> vars,
    ArrayRef<Expr*> ranges,
    Stmt* body,
    Location location
) : Stmt{Kind::ForStmt, location},
    num_vars{u32(vars.size())},
    num_ranges{u32(ranges.size())},
    enum_var{enum_var}, body{body} {
    std::uninitialized_copy_n(vars.begin(), vars.size(), getTrailingObjects<LocalDecl*>());
    std::uninitialized_copy_n(ranges.begin(), ranges.size(), getTrailingObjects<Expr*>());
}

auto ForStmt::Create(
    TranslationUnit& tu,
    Ptr<LocalDecl> enum_var,
    ArrayRef<LocalDecl*> vars,
    ArrayRef<Expr*> ranges,
    Stmt* body,
    Location location
) -> ForStmt* {
    const auto size = totalSizeToAlloc<LocalDecl*, Expr*>(vars.size(), ranges.size());
    auto mem = tu.allocate(size, alignof(ForStmt));
    return ::new (mem) ForStmt{enum_var, vars, ranges, body, location};
}

LocalRefExpr::LocalRefExpr(LocalDecl* decl, ValueCategory vc, Location loc)
    : Expr(Kind::LocalRefExpr, decl->type, vc, loc), decl{decl} {}

MatchExpr::MatchExpr(
    Ptr<Expr> control_expr,
    Type ty,
    ValueCategory vc,
    ArrayRef<MatchCase> cases,
    Location loc
) : Expr(Kind::MatchExpr, ty, vc, loc),
    num_cases{u32(cases.size())},
    has_control_expr(control_expr.present()) {
    if (auto c = control_expr.get_or_null()) *getTrailingObjects<Expr*>() = c;
    std::uninitialized_copy_n(cases.begin(), cases.size(), getTrailingObjects<MatchCase>());
}

auto MatchExpr::Create(
    TranslationUnit& tu,
    Ptr<Expr> control_expr,
    Type ty,
    ValueCategory vc,
    ArrayRef<MatchCase> cases,
    Location loc
) -> MatchExpr* {
    auto size = totalSizeToAlloc<Expr*, MatchCase>(control_expr.present(), cases.size());
    auto mem = tu.allocate(size, alignof(MatchExpr));
    return ::new (mem) MatchExpr{control_expr, ty, vc, cases, loc};
}

MemberAccessExpr::MemberAccessExpr(
    Expr* base,
    FieldDecl* field,
    Location location
) : Expr{Kind::MemberAccessExpr, field->type, LValue, location},
    base{base},
    field{field} {}

OverloadSetExpr::OverloadSetExpr(
    ArrayRef<Decl*> decls,
    Location location
) : Expr{Kind::OverloadSetExpr, Type::UnresolvedOverloadSetTy, RValue, location},
    num_overloads{u32(decls.size())} {
    Assert(num_overloads != 0, "Empty overload set?");
    std::uninitialized_copy(decls.begin(), decls.end(), getTrailingObjects());
}

auto OverloadSetExpr::Create(
    TranslationUnit& tu,
    ArrayRef<Decl*> decls,
    Location location
) -> OverloadSetExpr* {
    auto size = totalSizeToAlloc<Decl*>(decls.size());
    auto mem = tu.allocate(size, alignof(OverloadSetExpr));
    return ::new (mem) OverloadSetExpr{decls, location};
}

auto OverloadSetExpr::name() -> DeclName {
    return overloads().front()->name;
}

auto ParamDecl::intent() const -> Intent {
    return parent->param_types()[idx].intent;
}

ProcRefExpr::ProcRefExpr(
    ProcDecl* decl,
    Location location
) : Expr(Kind::ProcRefExpr, decl->type, RValue, location),
    decl{decl} {}

auto ProcRefExpr::return_type() const -> Type {
    return decl->return_type();
}

auto StrLitExpr::Create(
    TranslationUnit& mod,
    String value,
    Location location
) -> StrLitExpr* {
    return new (mod) StrLitExpr{mod.StrLitTy, value, location};
}

auto Stmt::type_or_void() const -> Type {
    if (auto e = dyn_cast<Expr>(this)) return e->type;
    return Type::VoidTy;
}

auto Stmt::value_category_or_rvalue() const -> ValueCategory {
    if (auto e = dyn_cast<Expr>(this)) return e->value_category;
    return Expr::RValue;
}

// ============================================================================
//  Declarations
// ============================================================================
auto DeclName::str() const -> String {
    if (is_str()) return String::CreateUnsafe(ptr, opaque_value);
    auto t = operator_name();
    if (t == Tk::LParen) return "()";
    if (t == Tk::LBrack) return "[]";
    return Spelling(t);
}

bool srcc::operator==(DeclName a, DeclName b) {
    if (a.is_str() != b.is_str()) return false;
    if (a.is_operator_name()) return a.operator_name() == b.operator_name();
    return a.str() == b.str();
}

ImportedClangModuleDecl::ImportedClangModuleDecl(
    clang::ASTUnit& clang_ast,
    String logical_name,
    ArrayRef<String> header_names,
    Location loc
) : ModuleDecl{Kind::ImportedClangModuleDecl, logical_name, loc},
    num_headers{u32(header_names.size())},
    clang_ast{clang_ast} {
    uninitialized_copy(header_names, getTrailingObjects());
}

auto ImportedClangModuleDecl::Create(
    TranslationUnit& tu,
    clang::ASTUnit& clang_ast,
    String logical_name,
    ArrayRef<String> header_names,
    Location loc
) -> ImportedClangModuleDecl* {
    auto sz = totalSizeToAlloc<String>(header_names.size());
    auto mem = tu.allocate(sz, alignof(ImportedClangModuleDecl));
    return ::new (mem) ImportedClangModuleDecl{clang_ast, logical_name, header_names, loc};
}

ProcDecl::ProcDecl(
    TranslationUnit* owner,
    ProcType* type,
    DeclName name,
    Linkage linkage,
    Mangling mangling,
    Ptr<ProcDecl> parent,
    Location location
) : ObjectDecl{Kind::ProcDecl, owner, type, name, linkage, mangling, location},
    parent{parent} {
    owner->procs.push_back(this);
}

auto ProcDecl::Create(
    TranslationUnit& tu,
    ProcType* type,
    DeclName name,
    Linkage linkage,
    Mangling mangling,
    Ptr<ProcDecl> parent,
    Location location
) -> ProcDecl* {
    return new (tu) ProcDecl{
        &tu,
        type,
        name,
        linkage,
        mangling,
        parent,
        location,
    };
}

void ProcDecl::finalise(Ptr<Stmt> body, ArrayRef<LocalDecl*> vars) {
    body_stmt = body;
    locals = vars.copy(owner->allocator());

    Assert(locals.size() >= param_count(), "Missing parameter declarations!");
    for (auto l : locals.take_front(proc_type()->params().size()))
        Assert(isa<ParamDecl>(l), "Parameters must be ParamDecls");
}

auto ProcDecl::proc_type() const -> ProcType* {
    return cast<ProcType>(type);
}

auto ProcDecl::return_type() -> Type {
    return proc_type()->ret();
}

ProcTemplateDecl::ProcTemplateDecl(
    TranslationUnit& tu,
    ParsedProcDecl* pattern,
    Ptr<ProcDecl> parent,
    bool has_variadic_param,
    Location location
) : Decl{Kind::ProcTemplateDecl, pattern->name, location},
    owner(&tu), parent{parent}, pattern{pattern},
    has_variadic_param{has_variadic_param} {}

auto ProcTemplateDecl::Create(
    TranslationUnit& tu,
    ParsedProcDecl* pattern,
    Ptr<ProcDecl> parent,
    bool has_variadic_param
) -> ProcTemplateDecl* {
    return new (tu) ProcTemplateDecl{
        tu,
        pattern,
        parent,
        has_variadic_param,
        pattern->loc,
    };
}

auto ProcTemplateDecl::instantiations() -> ArrayRef<ProcDecl*> {
    return owner->template_instantiations[this];
}

bool ProcTemplateDecl::is_builtin_operator_template() const {
    return pattern->type->attrs.builtin_operator;
}

TupleExpr::TupleExpr(
    RecordType* ty,
    ArrayRef<Expr*> fields,
    Location location
) : Expr{Kind::TupleExpr, ty, RValue, location} {
    std::uninitialized_copy_n(fields.begin(), fields.size(), getTrailingObjects());
}

auto TupleExpr::Create(
    TranslationUnit& tu,
    RecordType* type,
    ArrayRef<Expr*> fields,
    Location location
) -> TupleExpr* {
    Assert(fields.size() == type->layout().fields().size(), "Argument count mismatch");
    auto size = totalSizeToAlloc<Expr*>(fields.size());
    auto mem = tu.allocate(size, alignof(TupleExpr));
    return ::new (mem) TupleExpr{type, fields, location};
}
