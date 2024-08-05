module;

#include <algorithm>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.ast;
import :stmt;
import :enums;

using namespace srcc;

// ============================================================================
//  Statements.
// ============================================================================
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

void ComputeDependence(CastExpr* e) {
    e->set_dependence(e->arg->dependence());
}

void ComputeDependence(ConstExpr*) {
    // An evaluated constant expression is NEVER dependent; it may
    // contain a dependent expression that has been instantiated and
    // evaluated, but we don’t care about that anymore since we already
    // have the value.
}

void ComputeDependence(EvalExpr* e) {
    e->set_dependence(e->stmt->dependence());
}

void ComputeDependence(IntLitExpr*) {
    // Always of type 'int' and thus never dependent.
}

void ComputeDependence(LocalDecl* d) {
    if (d->type->dependent())
        d->set_dependence(Dependence::Type);
}

void ComputeDependence(LocalRefExpr* e) {
    e->set_dependence(e->decl->dependence());
}

void ComputeDependence(ParamDecl* d) {
    ComputeDependence(static_cast<LocalDecl*>(d));
}

void ComputeDependence(ProcRefExpr* p) {
    p->set_dependence(p->decl->dependence());
}

void ComputeDependence(ProcDecl* d) {
    if (d->body) d->set_dependence(d->body->dependence());
}

void ComputeDependence(ReturnExpr* e) {
    if (auto value = e->value.get_or_null())
        e->set_dependence(value->dependence());
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

// ============================================================================
//  Types.
// ============================================================================
void ComputeDependence(BuiltinType*) {}
void ComputeDependence(IntType*) {}
void ComputeDependence(ArrayType* arr) { arr->dep = arr->elem()->dep; }
void ComputeDependence(ReferenceType* ref) { ref->dep = ref->elem()->dep; }
void ComputeDependence(SliceType* slice) { slice->dep = slice->elem()->dep; }
void ComputeDependence(ProcType* proc) {
    Dependence d = proc->ret()->dep;
    for (auto param : proc->params()) d |= param->dep;
    proc->dep = d;
}

void TypeBase::ComputeDependence() {
    switch (kind()) {
#define AST_TYPE_LEAF(node)                    \
    case Kind::node:                           \
        ::ComputeDependence(cast<node>(this)); \
        return;
#include "srcc/AST.inc"
    }
}
