#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Frontend/Parser.hh>

#include <memory>
#include <ranges>

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
BuiltinCallExpr::BuiltinCallExpr(
    Builtin kind,
    Type return_type,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::BuiltinCallExpr, return_type, SRValue, location}, builtin{kind}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<Expr*>());
    // Determine value category.
    switch (builtin) {
        // SRValue.
        case Builtin::Print:
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
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::CallExpr, type, type->rvalue_category(), location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<Expr*>());
}

auto CallExpr::Create(
    TranslationUnit& mod,
    Type type,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) -> CallExpr* {
    const auto size = totalSizeToAlloc<Expr*>(args.size());
    auto mem = mod.allocate(size, alignof(CallExpr));
    return ::new (mem) CallExpr{type, callee, args, location};
}

ConstExpr::ConstExpr(
    TranslationUnit& tu,
    eval::SRValue value,
    Location location,
    Ptr<Stmt> stmt
) : Expr{Kind::ConstExpr, value.type(), SRValue, location},
    value{tu.save(std::move(value))},
    stmt{stmt} {}

BlockExpr::BlockExpr(
    Scope* parent_scope,
    Type type,
    ArrayRef<Stmt*> stmts,
    Location location
) : Expr{Kind::BlockExpr, type, SRValue, location},
    num_stmts{u32(stmts.size())},
    scope{parent_scope} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<Stmt*>());
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

LocalRefExpr::LocalRefExpr(LocalDecl* decl, Location loc)
    : Expr(Kind::LocalRefExpr, decl->type, LValue, loc), decl{decl} {
    // If this is a parameter that is passed as an rvalue, and the intent is 'In',
    // then we only have an rvalue in the callee (other intents may be passed by
    // value as well, but still create variables in the callee).
    auto p = dyn_cast<ParamDecl>(decl);
    if (p and p->is_rvalue_in_parameter()) value_category = SRValue;
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
) : Expr{Kind::OverloadSetExpr, Type::UnresolvedOverloadSetTy, SRValue, location},
    num_overloads{u32(decls.size())} {
    std::uninitialized_copy(decls.begin(), decls.end(), getTrailingObjects<Decl*>());
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

auto ParamDecl::intent() const -> Intent {
    return parent->param_types()[idx].intent;
}

bool ParamDecl::is_rvalue_in_parameter() const {
    auto i = intent();
    return i == Intent::In and type->pass_by_rvalue(parent->proc_type()->cconv(), i);
}

ProcRefExpr::ProcRefExpr(
    ProcDecl* decl,
    Location location
) : Expr(Kind::ProcRefExpr, decl->type, SRValue, location),
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

// ============================================================================
//  Declarations
// ============================================================================
ProcDecl::ProcDecl(
    TranslationUnit* owner,
    ProcType* type,
    String name,
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
    String name,
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
    Location location
) : Decl{Kind::ProcTemplateDecl, pattern->name, location},
    owner(&tu), parent{parent}, pattern{pattern} {}

auto ProcTemplateDecl::Create(
    TranslationUnit& tu,
    ParsedProcDecl* pattern,
    Ptr<ProcDecl> parent
) -> ProcTemplateDecl* {
    return new (tu) ProcTemplateDecl{
        tu,
        pattern,
        parent,
        pattern->loc,
    };
}

auto ProcTemplateDecl::instantiations() -> ArrayRef<ProcDecl*> {
    return owner->template_instantiations[this];
}

StructInitExpr::StructInitExpr(
    StructType* ty,
    ArrayRef<Expr*> fields,
    Location location
) : Expr{Kind::StructInitExpr, ty, ty->rvalue_category(), location} {
    std::uninitialized_copy_n(fields.begin(), fields.size(), getTrailingObjects<Expr*>());
}

auto StructInitExpr::Create(
    TranslationUnit& tu,
    StructType* type,
    ArrayRef<Expr*> fields,
    Location location
) -> StructInitExpr* {
    Assert(fields.size() == type->fields().size(), "Argument count mismatch");
    auto size = totalSizeToAlloc<Expr*>(fields.size());
    auto mem = tu.allocate(size, alignof(StructInitExpr));
    return ::new (mem) StructInitExpr{type, fields, location};
}
