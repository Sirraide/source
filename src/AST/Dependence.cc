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
void Stmt::ComputeDependence() { // clang-format off
    auto d = Dependence::None;
    visit(utils::Overloaded {
    [&](AssertExpr* e) {
        d |= e->cond->dependence();
        if (auto msg = e->message.get_or_null()) d |= msg->dependence();
    },

    [&](BinaryExpr* e) {
        d |= e->lhs->dependence();
        d |= e->rhs->dependence();
    },

    [&](BlockExpr* e) {
        // Propagate instantiation dependence from contained statements.
        for (auto s : e->stmts()) {
            if (s->dependent()) {
                d = Dependence::Instantiation;
                break;
            }
        }

        // But type/value/error dependence only from the return expression.
        if (auto r = e->return_expr()) d |= r->dependence();
    },

    [&](BoolLitExpr*) {}, // Never dependent.
    [&](BuiltinCallExpr* e) {
        // Always propagate instantiation dependence from arguments.
        for (auto s : e->args()) {
            if (s->dependent()) {
                d = Dependence::Instantiation;
                break;
            }
        }

        // Dependence of the call overall depends on the builtin.
        switch (e->builtin) {
            // Never dependent.
            case BuiltinCallExpr::Builtin::Print: return;
        }

        Unreachable("Invalid builtin: {}", +e->builtin);
    },

    [&](CallExpr* e) {
        for (auto a : e->args()) d |= a->dependence();
        d |= e->callee->dependence();
    },

    [&](CastExpr* e) { d = e->arg->dependence(); },
    [&](ConstExpr*) {
        // An evaluated constant expression is NEVER dependent; it may
        // contain a dependent expression that has been instantiated and
        // evaluated, but we don’t care about that anymore since we already
        // have the value.
    },

    [&](EvalExpr* e) { d = e->stmt->dependence(); },
    [&](IntLitExpr*) { /* Always of type 'int' and thus never dependent. */ },
    [&](LocalDecl* e) {
        // This also handles ParamDecls.
        if (e->type->dependent()) d |= Dependence::Type;
        if (auto init = e->init.get_or_null()) d |= init->dependence();
    },

    [&](LocalRefExpr* e) { d = e->decl->dependence(); },
    [&](OverloadSetExpr* e) {
      for (auto o : e->overloads()) d |= o->dependence();
    },

    [&](ParenExpr* e) { d = e->expr->dependence(); },
    [&](ProcRefExpr* e) { d = e->decl->dependence(); },
    [&](ProcDecl* e) {
        if (auto body = e->body().get_or_null()) d = body->dependence();
    },

    [&](TemplateTypeDecl*) { d = Dependence::Type; },
    [&](ReturnExpr* e) {
        if (auto value = e->value.get_or_null()) d = value->dependence();
    },

    [&](SliceDataExpr* e) { d = e->slice->dependence(); },
    [&](StrLitExpr*) { /* Never dependent */ },
    [&](TypeExpr* e) { if (e->value->dependent()) d = Dependence::Type; },
    [&](UnaryExpr* e) { d = e->arg->dependence(); },
    });
    set_dependence(d);
} // clang-format on

// ============================================================================
//  Types.
// ============================================================================
void TypeBase::ComputeDependence() { // clang-format off
    dep = visit(utils::Overloaded {
        [&](BuiltinType* e) { return e->dep; },
        [&](IntType*) { return Dependence::None; },
        [&](SingleElementTypeBase* e) { return e->dependence(); },
        [&](TemplateType*) { return Dependence::Type; },
        [&](ProcType* e) {
            Dependence d = e->ret()->dep;
            for (auto param : e->params()) d |= param->dep;
            return d;
        }
    });
} // clang-format on
