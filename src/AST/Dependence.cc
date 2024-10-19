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

    [&](BuiltinMemberAccessExpr* e) { d = e->operand->dependence(); },

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

    [&](DefaultInitExpr*) {
        // These are only ever created once we know what the actual type
        // is; a ‘dependent default initialiser’ is modelled as a call
        // expression instead.
    },

    [&](EvalExpr* e) { d = e->stmt->dependence(); },
    [&](FieldDecl* e) { d = e->type->dep; },
    [&](IfExpr* e) {
        // Only propagate instantiation dependence here; whether the *type*
        // of the if is supposed to be dependent is taken care of by Sema when
        // the node is built.
        if (
            e->cond->dependent() or
            e->then->dependent() or
            (e->else_ and e->else_.get()->dependent())
        ) d = Dependence::Instantiation;
    },

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

    [&](StaticIfExpr*) { d = Dependence::ValueAndType; },
    [&](StrLitExpr*) { /* Never dependent */ },
    [&](TypeExpr* e) { if (e->value->dependent()) d = Dependence::Type; },
    [&](TypeDecl* td) { d = td->type->dep; },
    [&](UnaryExpr* e) { d = e->arg->dependence(); },
    [&](WhileStmt* e) {
        if (e->cond->dependent() or e->body->dependent())
            d = Dependence::Instantiation;
    }
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
        [&](StructType* s) {
            auto d = Dependence::None;
            for (const auto& field : s->fields()) d |= field->type->dep;
            return d;
        },
        [&](TemplateType*) { return Dependence::Type; },
        [&](ProcType* e) {
            auto d = e->ret()->dep;
            for (const auto& param : e->params()) d |= param.type->dep;
            return d;
        }
    });
} // clang-format on
