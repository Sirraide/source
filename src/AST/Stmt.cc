module;

#include <memory>
#include <ranges>
#include <srcc/Macros.hh>
#include <type_traits>

module srcc.ast;
import srcc;
import :stmt;

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
    ComputeDependence();

    // Determine value category.
    switch (builtin) {
        // SRValue.
        case Builtin::Print:
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
) : Expr{Kind::CallExpr, type, type->value_category(), location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<Expr*>());
    ComputeDependence();
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
    eval::Value value,
    Location location,
    Ptr<Stmt> stmt
) : Expr{Kind::ConstExpr, value.type(), value.value_category(), location},
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
    ComputeDependence();

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
    auto type = last ? last->type : Types::VoidTy;
    auto size = totalSizeToAlloc<Stmt*>(stmts.size());
    auto mem = mod.allocate(size, alignof(BlockExpr));
    return ::new (mem) BlockExpr{parent_scope, type, stmts, location};
}

auto BlockExpr::return_expr() -> Expr* {
    if (type->is_void()) return nullptr;
    return cast<Expr>(stmts().back());
}

auto Expr::strip_parens() -> Expr* {
    auto paren = dyn_cast<ParenExpr>(this);
    if (not paren) return this;
    return paren->expr->strip_parens();
}

LocalRefExpr::LocalRefExpr(LocalDecl* decl, Location loc)
    : Expr(Kind::LocalRefExpr, decl->type, LValue, loc), decl{decl} {
    ComputeDependence();

    // If this is a parameter that is passed as an rvalue, and the intent is 'In',
    // then we only have an rvalue in the callee (other intents may be passed by
    // value as well, but still create variables in the callee).
    auto p = dyn_cast<ParamDecl>(decl);
    if (p and p->is_rvalue_in_parameter()) value_category = SRValue;
}

OverloadSetExpr::OverloadSetExpr(
    ArrayRef<Decl*> decls,
    Location location
) : Expr{Kind::OverloadSetExpr, Types::UnresolvedOverloadSetTy, SRValue, location},
    num_overloads{u32(decls.size())} {
    auto proc_decls = decls | vws::transform([](auto d) { return cast<ProcDecl>(d); });
    std::uninitialized_copy(proc_decls.begin(), proc_decls.end(), getTrailingObjects<ProcDecl*>());
    ComputeDependence();
}

auto OverloadSetExpr::Create(
    TranslationUnit& tu,
    ArrayRef<Decl*> decls,
    Location location
) -> OverloadSetExpr* {
    auto size = totalSizeToAlloc<ProcDecl*>(decls.size());
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
    decl{decl} {
    ComputeDependence();
}

auto ProcRefExpr::return_type() const -> Type {
    return decl->return_type();
}

auto SliceDataExpr::Create(TranslationUnit& mod, Expr* slice, Location location) -> SliceDataExpr* {
    auto ty = ReferenceType::Get(mod, cast<SliceType>(slice->type)->elem());
    return new (mod) SliceDataExpr{ty, slice, location};
}

auto StrLitExpr::Create(
    TranslationUnit& mod,
    String value,
    Location location
) -> StrLitExpr* {
    return new (mod) StrLitExpr{mod.StrLitTy, value, location};
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
    ProcDecl* parent,
    ArrayRef<TemplateTypeDecl*> template_params,
    Location location
) : ObjectDecl{Kind::ProcDecl, owner, type, name, linkage, mangling, location},
    num_template_params{u32(template_params.size())},
    parent{parent} {
    owner->procs.push_back(this);

    std::uninitialized_copy_n(
        template_params.begin(),
        template_params.size(),
        getTrailingObjects<TemplateTypeDecl*>()
    );
}

auto ProcDecl::Create(
    TranslationUnit& tu,
    ProcType* type,
    String name,
    Linkage linkage,
    Mangling mangling,
    ProcDecl* parent,
    Location location,
    ArrayRef<TemplateTypeDecl*> template_params
) -> ProcDecl* {
    auto size = totalSizeToAlloc<TemplateTypeDecl*>(template_params.size());
    auto mem = tu.allocate(size, alignof(ProcDecl));
    return ::new (mem) ProcDecl{
        &tu,
        type,
        name,
        linkage,
        mangling,
        parent,
        template_params,
        location,
    };
}

void ProcDecl::finalise(Ptr<Stmt> body, ArrayRef<LocalDecl*> vars) {
    body_stmt = body;
    locals = vars.copy(owner->allocator());

    Assert(locals.size() >= param_count(), "Missing parameter declarations!");
    for (auto l : locals.take_front(proc_type()->params().size()))
        Assert(isa<ParamDecl>(l), "Parameters must be ParamDecls");

    ComputeDependence();
}

auto ProcDecl::proc_type() const -> ProcType* {
    return cast<ProcType>(type).ptr();
}

auto ProcDecl::return_type() -> Type {
    return proc_type()->ret();
}

TemplateTypeDecl::TemplateTypeDecl(
    String name,
    ArrayRef<u32> deduced_indices,
    Location location
) : Decl{Kind::TemplateTypeDecl, name, location},
    num_deduced_indices{u32(deduced_indices.size())} {
    std::uninitialized_copy_n(deduced_indices.begin(), deduced_indices.size(), getTrailingObjects<u32>());
    ComputeDependence();
}

auto TemplateTypeDecl::Create(
    TranslationUnit& tu,
    String name,
    ArrayRef<u32> deduced_indices,
    Location location
) -> TemplateTypeDecl* {
    auto size = totalSizeToAlloc<u32>(deduced_indices.size());
    auto mem = tu.allocate(size, alignof(TemplateTypeDecl));
    return ::new (mem) TemplateTypeDecl{name, deduced_indices, location};
}

// ============================================================================
//  Enum -> String
// ============================================================================
auto EnumToStr(BuiltinCallExpr::Builtin b) -> String {
    switch (b) {
        case BuiltinCallExpr::Builtin::Print: return "__srcc_print";
    }

    return "<invalid builtin>";
}
