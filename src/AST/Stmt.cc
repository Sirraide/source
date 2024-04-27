module;

#include <memory>

module srcc.ast;
import srcc;
import :stmt;

using namespace srcc;

void* Stmt::operator new(usz size, Module& mod) {
    return mod.Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

CallExpr::CallExpr(
    Type* type,
    Expr* callee,
    ArrayRef<Expr*> args,
    Location location
) : Expr{Kind::CallExpr, type, location},
    callee{callee}, num_args{u32(args.size())} {
    std::uninitialized_copy_n(args.begin(), args.size(), getTrailingObjects<Expr*>());
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
    Type* type,
    ArrayRef<Stmt*> stmts,
    Location location
) : Expr{Kind::BlockExpr, type, location},
    num_stmts{u32(stmts.size())} {
    std::uninitialized_copy_n(stmts.begin(), stmts.size(), getTrailingObjects<Stmt*>());
}

auto BlockExpr::Create(
    Module& mod,
    Type* type,
    ArrayRef<Stmt*> stmts,
    Location location
) -> BlockExpr* {
    const auto size = totalSizeToAlloc<Stmt*>(stmts.size());
    auto mem = mod.Allocate(size, alignof(BlockExpr));
    return ::new (mem) BlockExpr{type, stmts, location};
}

ProcRefExpr::ProcRefExpr(
    ProcDecl* decl,
    Location location
) : Expr(Kind::ProcRefExpr, decl->type, location), decl{decl} {}

StrLitExpr::StrLitExpr(
    Module* mod,
    String value,
    Location location
) : Expr(Kind::StrLitExpr, mod->StrLitTy, location), value{value} {}
