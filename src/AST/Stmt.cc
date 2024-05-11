module;

#include <memory>
#include <srcc/Macros.hh>

module srcc.ast;
import srcc;
import :stmt;

using namespace srcc;

void* Stmt::operator new(usz size, Module& mod) {
    return mod.Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

void Scope::add(Decl* d) {
    decls[d->name].push_back(d);
}


// ============================================================================
//  AST
// ============================================================================
CallExpr::CallExpr(
    Type* type,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::CallExpr, type, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<Expr*>());
    ComputeDependence();
}

auto CallExpr::Create(
    Module& mod,
    Type* type,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) -> CallExpr* {
    const auto size = totalSizeToAlloc<Expr*>(args.size());
    auto mem = mod.Allocate(size, alignof(CallExpr));
    return ::new (mem) CallExpr{type, callee, args, location};
}

BlockExpr::BlockExpr(
    Scope* parent_scope,
    Type* type,
    ArrayRef<Stmt*> stmts,
    u32 idx,
    Location location
) : Expr{Kind::BlockExpr, type, location},
    num_stmts{u32(stmts.size())},
    return_expr_index{idx},
    scope{parent_scope} {
    Assert(type->is_void() or return_expr_index <= num_stmts, "Return expression index out of bounds");
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<Stmt*>());
    ComputeDependence();
}

auto BlockExpr::Create(
    Module& mod,
    Scope* parent_scope,
    ArrayRef<Stmt*> stmts,
    u32 idx,
    Location location
) -> BlockExpr* {
    auto type = idx == NoExprIndex ? mod.VoidTy : cast<Expr>(stmts[idx])->type;
    auto size = totalSizeToAlloc<Stmt*>(stmts.size());
    auto mem = mod.Allocate(size, alignof(BlockExpr));
    return ::new (mem) BlockExpr{parent_scope, type, stmts, idx, location};
}

auto BlockExpr::return_expr() -> Expr* {
    if (type->is_void()) return nullptr;
    return cast<Expr>(stmts()[return_expr_index]);
}

ProcRefExpr::ProcRefExpr(
    ProcDecl* decl,
    Location location
) : Expr(Kind::ProcRefExpr, decl->type, location), decl{decl} { ComputeDependence(); }

auto ProcRefExpr::return_type() const -> Type* {
    return decl->return_type();
}

auto StrLitExpr::Create(
    Module& mod,
    String value,
    Location location
) -> StrLitExpr* {
    return new (mod) StrLitExpr{mod.StrLitTy, value, location};
}

auto ProcDecl::proc_type() const -> ProcType* {
    return cast<ProcType>(type);
}

auto ProcDecl::return_type() -> Type* {
    return proc_type()->ret();
}
