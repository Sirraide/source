#include <source/Frontend/Sema.hh>

bool src::Sema::Evaluate(Expr* e, EvalResult& out, bool must_succeed) {
    Assert(e->sema.ok, "Refusing evaluate broken or unanalysed expression");
    switch (e->kind) {
        case Expr::Kind::ArrayType:
        case Expr::Kind::BuiltinType:
        case Expr::Kind::ClosureType:
        case Expr::Kind::EnumType:
        case Expr::Kind::IntType:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::OptionalType:
        case Expr::Kind::ProcType:
        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::SliceType:
        case Expr::Kind::StructType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::TupleType:
            out = Type(e);
            return true;

        case Expr::Kind::AliasExpr:
        case Expr::Kind::ArrayLitExpr:
        case Expr::Kind::AssertExpr:
        case Expr::Kind::ConstructExpr:
        case Expr::Kind::DeclRefExpr:
        case Expr::Kind::DeferExpr:
        case Expr::Kind::EmptyExpr:
        case Expr::Kind::ExportExpr:
        case Expr::Kind::FieldDecl:
        case Expr::Kind::ForInExpr:
        case Expr::Kind::GotoExpr:
        case Expr::Kind::ImplicitThisExpr:
        case Expr::Kind::InvokeBuiltinExpr:
        case Expr::Kind::InvokeExpr:
        case Expr::Kind::LabelExpr:
        case Expr::Kind::LocalDecl:
        case Expr::Kind::LocalRefExpr:
        case Expr::Kind::LoopControlExpr:
        case Expr::Kind::MaterialiseTemporaryExpr:
        case Expr::Kind::MemberAccessExpr:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::ParamDecl:
        case Expr::Kind::ProcDecl:
        case Expr::Kind::ReturnExpr:
        case Expr::Kind::ScopeAccessExpr:
        case Expr::Kind::SubscriptExpr:
        case Expr::Kind::UnaryPrefixExpr:
        case Expr::Kind::WhileExpr:
        case Expr::Kind::WithExpr:
        not_constexpr:
            if (must_succeed) Error(e, "Not a constant expression");
            return false;

        case Expr::Kind::Nil:
            out = nullptr;
            return true;

        case Expr::Kind::StrLitExpr:
            out = cast<StrLitExpr>(e)->string;
            return true;

        case Expr::Kind::ParenExpr:
            return Evaluate(cast<ParenExpr>(e)->expr, out, must_succeed);

        case Expr::Kind::ConstExpr:
            out = cast<ConstExpr>(e)->value;
            return true;

        case Expr::Kind::OverloadSetExpr:
            out = cast<OverloadSetExpr>(e);
            return true;

        case Expr::Kind::EnumeratorDecl: {
            auto n = cast<EnumeratorDecl>(e);
            out = {cast<ConstExpr>(n)->value.as_int(), n->type};
            return true;
        }

        case Expr::Kind::BoolLitExpr: {
            auto i = cast<BoolLitExpr>(e);
            out = {APInt(1, i->value), Type::Bool};
            return true;
        }

        case Expr::Kind::IntLitExpr: {
            auto i = cast<IntLitExpr>(e);
            out = {i->value, i->stored_type};
            return true;
        }

        case Expr::Kind::TupleExpr: {
            auto t = cast<TupleExpr>(e);
            out = {EvalResult::TupleElements{}, cast<TupleType>(t->type.desugared)};
            auto& elems = out.as_tuple_elems();
            for (auto expr : t->elements)
                if (not Evaluate(expr, elems.emplace_back(), must_succeed))
                    return false;
            return true;
        }

        case Expr::Kind::TupleIndexExpr: {
            auto t = cast<TupleIndexExpr>(e);
            if (not Evaluate(t->object, out, must_succeed)) return false;

            /// TupleIndexExprs are only constructed by sema and as such are
            /// always valid. Assert that that is actually the case.
            auto& elems = out.as_tuple_elems();
            Assert(t->field->index < elems.size(), "Tuple index out of bounds");

            /// We can’t simply assign here as that would be UB due to `out`’s
            /// destructor being called before we can move the field value out.
            std::exchange(out, std::move(elems.at(t->field->index)));
            return true;
        }

        case Expr::Kind::IfExpr: {
            auto i = cast<IfExpr>(e);
            if (not Evaluate(i->cond, out, must_succeed)) return false;
            return Evaluate(out.as_int().getBoolValue() ? i->then : i->else_, out, must_succeed);
        }

        case Expr::Kind::BlockExpr: {
            auto b = cast<BlockExpr>(e);
            if (b->exprs.empty()) {
                out = {};
                return true;
            }

            for (auto expr : ArrayRef(b->exprs).drop_back(1))
                if (not Evaluate(expr, out, false))
                    return false;

            return Evaluate(b->exprs.back(), out, must_succeed);
        }

        case Expr::Kind::CastExpr: {
            auto c = cast<CastExpr>(e);
            if (not Evaluate(c->operand, out, must_succeed)) return false;
            if (c->type.is_int(true) and c->operand->type.is_int(true)) {
                if (c->type == c->operand->type) return true;

                /// Always *zero-extend* bools to avoid turning true into -1.
                const auto bits = unsigned(c->type.size(mod->context).bits());
                if (c->operand->type == Type::Bool) {
                    out.as_int() = out.as_int().zext(bits);
                    return true;
                }

                /// Casts to bool must yield 1 if the value is nonzero.
                if (c->type == Type::Bool) {
                    out.as_int() = APInt(1, out.as_int().getBoolValue());
                    return true;
                }

                out.as_int() = out.as_int().sextOrTrunc(bits);
                return true;
            }

            goto not_constexpr;
        }

        /// FIXME: Should check for overflow.
        case Expr::Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);
            EvalResult rhs_res;
            if (not Evaluate(b->lhs, out, must_succeed)) return false;
            if (not Evaluate(b->rhs, rhs_res, must_succeed)) return false;
            auto& lhs = out.as_int();
            auto& rhs = rhs_res.as_int();
            switch (b->op) {
                case Tk::Land: lhs &= rhs; return true;
                case Tk::Lor: lhs |= rhs; return true;
                case Tk::Xor: lhs ^= rhs; return true;
                case Tk::Plus: lhs += rhs; return true;
                case Tk::Minus: lhs -= rhs; return true;
                case Tk::Star: lhs *= rhs; return true;
                case Tk::ShiftLeft: lhs = lhs.shl(rhs); return true;
                case Tk::ShiftRight: lhs.ashrInPlace(rhs); return true;
                case Tk::ShiftRightLogical: lhs.lshrInPlace(rhs); return true;

                case Tk::Lt: out = {APInt(1, lhs.slt(rhs)), Type::Bool}; return true;
                case Tk::Le: out = {APInt(1, lhs.sle(rhs)), Type::Bool}; return true;
                case Tk::Gt: out = {APInt(1, lhs.sgt(rhs)), Type::Bool}; return true;
                case Tk::Ge: out = {APInt(1, lhs.sge(rhs)), Type::Bool}; return true;
                case Tk::EqEq: out = {APInt(1, lhs == rhs), Type::Bool}; return true;
                case Tk::Neq: out = {APInt(1, lhs != rhs), Type::Bool}; return true;

                case Tk::Not:
                    if (lhs.getBoolValue()) out = {APInt(1, false), Type::Bool};
                    else out = {APInt(1, true), Type::Bool};
                    return true;

                case Tk::And:
                    if (not lhs.getBoolValue()) {
                        out = {APInt(1, false), Type::Bool};
                        return true;
                    }

                    return Evaluate(b->rhs, out, must_succeed);

                case Tk::Or:
                    if (lhs.getBoolValue()) {
                        out = {APInt(1, true), Type::Bool};
                        return true;
                    }

                    return Evaluate(b->rhs, out, must_succeed);

                case Tk::Slash:
                    if (rhs.isZero()) {
                        if (must_succeed) Error(e, "Division by zero");
                        return false;
                    }

                    lhs = lhs.sdiv(rhs);
                    return true;

                case Tk::Percent:
                    if (rhs.isZero()) {
                        if (must_succeed) Error(e, "Division by zero");
                        return false;
                    }

                    lhs = lhs.srem(rhs);
                    return true;

                case Tk::StarStar: {
                    if (rhs.isZero()) {
                        out = {APInt(unsigned(b->lhs->type.size(mod->context).bits()), 1), b->lhs->type};
                        return true;
                    }

                    if (rhs.isNegative()) {
                        out = {APInt(unsigned(b->lhs->type.size(mod->context).bits()), 0), b->lhs->type};
                        return true;
                    }

                    auto base = lhs;
                    for (; rhs.sgt(1); --rhs) lhs *= base;
                    return true;
                }

                default: goto not_constexpr;
            }
        }
    }
}

bool src::Sema::EvaluateAsBoolInPlace(Expr*& e, bool must_succeed) {
    if (not EvaluateAsIntegerInPlace(e, must_succeed)) return false;
    if (auto type = cast<ConstExpr>(e)->value.type; type != Type::Bool) {
        if (must_succeed) Error(
            e,
            "Constant condition must be of type {}, but was {}",
            Type::Bool,
            type
        );
        return false;
    }
    return true;
}

bool src::Sema::EvaluateAsIntegerInPlace(Expr*& e, bool must_succeed) {
    EvalResult res;
    if (not Evaluate(e, res, must_succeed)) return false;
    if (not res.is_int()) {
        if (must_succeed) Error(e, "Constant expression is not an integer");
        return false;
    }

    e = new (mod) ConstExpr(e, std::move(res), e->location);
    return true;
}

auto src::Sema::EvaluateAsOverloadSet(Expr* e) -> OverloadSetExpr* {
    EvalResult res;
    Assert(e->type == Type::OverloadSet, "Only call this if the type is OverloadSet");
    Assert(Evaluate(e, res, true), "Overload set must be constant");
    return res.as_overload_set();
}
