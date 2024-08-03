module;

#include <memory>
#include <srcc/Macros.hh>

module srcc.ast;
import srcc;
import :stmt;

using namespace srcc;

void* Stmt::operator new(usz size, TranslationUnit& mod) {
    return mod.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

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
    u32 idx,
    Location location
) : Expr{Kind::BlockExpr, type, SRValue, location},
    num_stmts{u32(stmts.size())},
    return_expr_index{idx},
    scope{parent_scope} {
    Assert(type->is_void() or return_expr_index <= num_stmts, "Return expression index out of bounds");
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<Stmt*>());
    ComputeDependence();

    // The value category of this is that of the return expr.
    if (auto e = return_expr()) value_category = e->value_category;
}

auto BlockExpr::Create(
    TranslationUnit& mod,
    Scope* parent_scope,
    ArrayRef<Stmt*> stmts,
    u32 idx,
    Location location
) -> BlockExpr* {
    auto type = idx == NoExprIndex ? Types::VoidTy : cast<Expr>(stmts[idx])->type;
    auto size = totalSizeToAlloc<Stmt*>(stmts.size());
    auto mem = mod.allocate(size, alignof(BlockExpr));
    return ::new (mem) BlockExpr{parent_scope, type, stmts, idx, location};
}

auto BlockExpr::return_expr() -> Expr* {
    if (type->is_void()) return nullptr;
    return cast<Expr>(stmts()[return_expr_index]);
}

LocalRefExpr::LocalRefExpr(LocalDecl* decl, Location loc)
    : Expr(Kind::LocalRefExpr, decl->type, LValue, loc), decl{decl} {
    ComputeDependence();
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
    Type type,
    String name,
    Linkage linkage,
    Mangling mangling,
    ProcDecl* parent,
    Stmt* body,
    Location location
) : ObjectDecl{Kind::ProcDecl, owner, type, name, linkage, mangling, location},
    parent{parent},
    body{body} {
    owner->procs.push_back(this);
    ComputeDependence();
}

auto ProcDecl::Create(
    TranslationUnit& tu,
    Type type,
    String name,
    Linkage linkage,
    Mangling mangling,
    ProcDecl* parent,
    Stmt* body,
    Location location
) -> ProcDecl* {
    auto mem = tu.allocate(sizeof(ProcDecl), alignof(ProcDecl));
    return ::new (mem) ProcDecl{&tu, type, name, linkage, mangling, parent, body, location};
}

void ProcDecl::finalise(ArrayRef<LocalDecl*> vars) {
    locals = vars.copy(owner->allocator());
    for (auto l : locals.take_front(proc_type()->params().size()))
        Assert(isa<ParamDecl>(l), "Parameters must be ParamDecls");
}

auto ProcDecl::proc_type() const -> ProcType* {
    return cast<ProcType>(type).ptr();
}

auto ProcDecl::return_type() -> Type {
    return proc_type()->ret();
}

// ============================================================================
//  Enum -> String
// ============================================================================
auto EnumToStr(BuiltinCallExpr::Builtin b) -> String {
    switch (b) {
        case BuiltinCallExpr::Builtin::Print: return "__builtin_print";
    }

    return "<invalid builtin>";
}
