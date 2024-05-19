module;

#include <srcc/Macros.hh>

module srcc.ast;
import :stmt;
import :enums;

using namespace srcc;

void ComputeDependence(BlockExpr* e) {
    auto d = Dependence::None;

    // Propagate instantiation dependence from contained statements.
    for (auto s : e->stmts()) {
        if (s->dependent()) {
            d = Dependence::Instantiation;
            break;
        }
    }

    // But type/value/error dependence only from the return expression.
    if (auto r = e->return_expr()) d |= r->dependence();
    e->set_dependence(d);
}

void ComputeDependence(BuiltinCallExpr* e) {
    auto d = Dependence::None;

    // Always propagate instantiation dependence from arguments.
    for (auto s : e->args()) {
        if (s->dependent()) {
            d = Dependence::Instantiation;
            break;
        }
    }

    // Dependence of the call overall depends on the builtin.
    [&] {
        switch (e->builtin) {
            // Never dependent.
            case BuiltinCallExpr::Builtin::Print: return;
        }

        Unreachable("Invalid builtin: {}", +e->builtin);
    }();

    e->set_dependence(d);
}

void ComputeDependence(CallExpr* e) {
    auto d = Dependence::None;
    for (auto a : e->args()) d |= a->dependence();
    d |= e->callee->dependence();
    e->set_dependence(d);
}

void ComputeDependence(ProcRefExpr* p) {
    p->set_dependence(p->decl->dependence());
}

void ComputeDependence(ProcDecl* d) {
    if (d->body) d->set_dependence(d->body->dependence());
}

void ComputeDependence(SliceDataExpr* s) {
    s->set_dependence(s->slice->dependence());
}

void ComputeDependence(StrLitExpr*) {
    // Never dependent.
}

void Stmt::ComputeDependence() {
    switch (kind()) {
#define AST_STMT_LEAF(node)                    \
    case Kind::node:                           \
        ::ComputeDependence(cast<node>(this)); \
        return;
#include "srcc/AST.inc"
    }

    Unreachable("Invalid statement kind: {}", +kind());
}
