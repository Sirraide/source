#include <srcc/Frontend/Sema.hh>

using namespace srcc;

#define TRY(...) ({                      \
    auto _res = (__VA_ARGS__);           \
    if (not _res) return utils::Falsy(); \
    *_res;                               \
})

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildAssertExpr(
    Expr* cond,
    Ptr<Expr> msg,
    bool is_compile_time,
    SLoc loc,
    SRange cond_range
) -> Ptr<Expr> {
    if (not MakeCondition(cond, "assert")) return {};

    // Message must be a string literal.
    // TODO: Allow other string-like expressions.
    if (auto m = msg.get_or_null(); m and not isa<StrLitExpr>(m)) return Error(
        m->location(),
        "Assertion message must be a string literal",
        m->type
    );

    Ptr<ProcDecl> stringifier;
    if (tu->lang_opts().stringify_asserts) {
        // Get a procedure from the runtime.
        auto GetRuntimeSymbol = [&](String name) {
            return BuildDeclRefExpr(
                {
                    {DeclName("__src_runtime"), loc},
                    {DeclName(name), loc}
                },
                loc
            ).get();
        };

        // Build the stringifier.
        auto BuildBody = [&](ProcDecl* proc, SmallVectorImpl<Stmt*>& stmts) {
            // Append a string to the buffer.
            auto param = curr_proc().locals[0];
            auto AppendStr = [&](StringRef s){
                StrLitExpr* str{};

                // At compile time, we can preserve colours as we’ll be printing this
                // in a diagnostic, if at all; at runtime, we need to remove them since
                // the runtime library doesn’t understand our colour formatting.
                if (is_compile_time) {
                    str = StrLitExpr::Create(*tu, tu->save(s), loc);
                } else {
                    auto stripped = text::RenderColours(false, std::string_view{s});
                    str = StrLitExpr::Create(*tu, tu->save(stripped), loc);
                }

                auto append_str = GetRuntimeSymbol("__src_assert_append_str");
                auto call = BuildCallExpr(append_str, {CreateReference(param, loc).get(), str}, loc);
                stmts.push_back(call.get());
            };

            // Attempt to decompose the expression.
            cond->visit(utils::Overloaded{
                [&](auto* s) {
                    auto k = static_cast<Stmt*>(s)->kind();
                    AppendStr(String::CreateUnsafe("%0("s + enchantum::to_string(k) + "%)"));
                },
                [&](BoolLitExpr* ile) {
                    if (ile->value) AppendStr("%1(true%)");
                    else AppendStr("%1(false%)");
                },
                [&](TypeExpr* te) {
                    auto type_name = te->value->print().str().str();
                    if (not is_compile_time) type_name = text::RenderColours(false, type_name);
                    AppendStr(tu->save(type_name));
                },
                [&](this auto&& self, BinaryExpr* b) {
                    b->lhs->visit(self);
                    AppendStr(" %1(");
                    AppendStr(Spelling(b->op));
                    AppendStr(" %)");
                    b->rhs->visit(self);
                }
            });
        };

        auto assert_buffer_type = cast<TypeExpr>(GetRuntimeSymbol("__src_assert_msg_buf"))->value;
        stringifier = BuildImplicitProcedure(
            tu->save(Format("__srcc_assert_stringifier_{}", assert_stringifiers++)),
            {Type::VoidTy, Expr::RValue},
            {{{Intent::Inout, assert_buffer_type, false}}},
            Linkage::Internal,
            Mangling::None,
            loc,
            BuildBody
        );
    }

    auto a = new (*tu) AssertExpr(cond, std::move(msg), false, loc, cond_range, stringifier);
    if (not is_compile_time) return a;
    return EvaluateIntoExpr(a, loc);
}

auto Sema::BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, SLoc loc) -> BlockExpr* {
    return BlockExpr::Create(
        *tu,
        scope,
        stmts,
        loc
    );
}

auto Sema::BuildBinaryExpr(
    Tk op,
    Expr* lhs,
    Expr* rhs,
    SLoc loc
) -> Ptr<Expr> {
    using enum ValueCategory;
    auto Build = [&](Type ty, ValueCategory cat = RValue) {
        return new (*tu) BinaryExpr(ty, cat, op, lhs, rhs, loc);
    };

    auto ErrorInvalid = [&] {
        return Error(
            loc,
            "Invalid operation: '%1({}%)' between '{}' and '{}'",
            Spelling(op),
            lhs->type,
            rhs->type
        );
    };

    auto ErrorExpI1 = [&] {
        return Error(
            loc,
            "'%1({}%)' is not supported for type '{}' (bit width must be at least 2)",
            Spelling(op),
            lhs->type
        );
    };

    auto CheckIntegral = [&] -> bool {
        // Either operand must be an integer.
        bool lhs_int = lhs->type->is_integer();
        bool rhs_int = rhs->type->is_integer();
        if (not lhs_int and not rhs_int) return ErrorInvalid();
        return true;
    };

    auto ConvertToCommonType = [&] -> bool {
        // Find the common type of the two. We need the same logic
        // during initialisation (and it actually turns out to be
        // easier to write it that way), so reuse it here.
        if (auto lhs_conv = TryBuildInitialiser(rhs->type, lhs).get_or_null()) {
            lhs = lhs_conv;
        } else if (auto rhs_conv = TryBuildInitialiser(lhs->type, rhs).get_or_null()) {
            rhs = rhs_conv;
        } else {
            return ErrorInvalid();
        }

        // Now they’re the same type, so ensure both are srvalues.
        lhs = LValueToRValue(lhs);
        rhs = LValueToRValue(rhs);
        return true;
    };

    auto BuildCall = [&](DeclName fun) -> Ptr<Expr> {
        // Perform lookup for operator names manually since 'unknown symbol' is
        // a terrible error message for this (it looks like a parse error), and
        // moreover, the regular lookup failure diagnostics code doesn't have access
        // to the parameter types.
        auto res = LookUpUnqualifiedName(DeclNameLoc{fun, loc});
        if (res.result == LookupResult::Reason::NotFound and fun.is_operator_name()) {
            return Error(
                loc,
                "Could not find overload of '%1({}%)' for '{}' and '{}'",
                fun,
                lhs->type,
                rhs->type
            );
        }

        if (not res.successful_or_ambiguous()) {
            ReportLookupFailure(std::move(res));
            return {};
        }

        auto ref = CreateReferenceOrOverloadSet(loc, res);
        if (not ref) return nullptr;
        return BuildCallExpr(ref.get(), {lhs, rhs}, loc);
    };

    auto BuildArithmeticOrComparisonOperator = [&](bool comparison) -> Ptr<Expr> {
        if (not ConvertToCommonType()) return nullptr;
        if (isa<ArrayType>(lhs->type)) return BuildCall(DeclName(op));
        if (not CheckIntegral()) return nullptr;
        return Build(comparison ? Type::BoolTy : lhs->type);
    };

    auto CheckLValue = [&](Expr* e) -> bool {
        // Prohibit assignment to 'in' parameters.
        if (auto ref = dyn_cast<LocalRefExpr>(e)) {
            if (
                auto param = dyn_cast<ParamDecl>(ref->decl);
                param and param->intent() == Intent::In
            ) return Error(e->location(), "Cannot assign to '%1(in%)' parameter");
        }

        // LHS must be an lvalue.
        if (not e->is_mutable_lvalue()) {
            if (e->is_lvalue()) return Error(e->location(), "Cannot assign to immutable value");
            return Error(e->location(), "Invalid target for assignment");
        }

        return true;
    };

    if (IsUserDefinedOverloadedOperator(op, {lhs->type, rhs->type}))
        return BuildCall(DeclName(op));

    switch (op) {
        default: Unreachable("Invalid binary operator: '{}'", op);

        // Array or slice subscript.
        case Tk::LBrack: {
            if (auto ty = dyn_cast<TypeExpr>(lhs)) {
                auto arr = TRY(BuildArrayType({ty->value, ty->location()}, rhs));
                return BuildTypeExpr(arr, loc);
            }

            if (not isa<TupleType, SliceType, ArrayType>(lhs->type)) return Error(
                lhs->location(),
                "Cannot subscript type '{}'",
                lhs->type
            );

            // RHS must be an integer.
            if (not MakeRValue(Type::IntTy, rhs, "Index", "[]")) return {};

            // Aggregates need to be in memory before we can do anything
            // with them; slices are srvalues and should be loaded whole.
            // FIXME: Stop doing that for slices.
            //
            // Furthermote, an array/tuple subscript is immutable if the LHS
            // is immutable; a slice subscript is immutable if the slice *type*
            // is immutable.
            bool immutable = false;
            if (isa<TupleType, ArrayType>(lhs->type)) {
                immutable = lhs->is_immutable_lvalue();
                lhs = MaterialiseTemporary(lhs);
            } else {
                immutable = cast<SliceType>(lhs->type)->is_immutable();
                lhs = LValueToRValue(lhs);
            }

            // For tuples, the integer must be a compile-time constant, and
            // the result of a subscript operation is a member access.
            if (auto ty = dyn_cast<TupleType>(lhs->type)) {
                auto res = Evaluate(rhs);
                if (not res) return {};
                auto idx = i64(res->cast<APInt>().getSExtValue());
                auto tuple_elems = i64(ty->layout().fields().size());
                if (idx < 0 or idx >= tuple_elems) return Error(
                    rhs->location(),
                    "Tuple index {} is out of bounds for {}-element tuple\f{}",
                    idx,
                    tuple_elems,
                    ty
                );

                return new (*tu) MemberAccessExpr(lhs, ty->layout().fields()[usz(idx)], loc);
            }

            // A subscripting operation yields an lvalue.
            return Build(
                cast<SingleElementTypeBase>(lhs->type)->elem(),
                Expr::LValue(immutable)
            );
        }

        case Tk::As:
        case Tk::AsBang: {
            auto type = Evaluate(rhs);
            if (not type) return {};
            if (not type->isa<Type>()) return Error(rhs->location(), "Expected type");
            return BuildExplicitCast(type->cast<Type>(), lhs, loc, op == Tk::AsBang);
        }

        case Tk::In: {
            if (auto r = dyn_cast<RangeType>(rhs->type)) {
                if (not MakeRValue(r->elem(), lhs, "Left operand", Spelling(op))) return nullptr;
                rhs = LValueToRValue(rhs);
                return Build(Type::BoolTy);
            }

            // TODO: Allow for slices and arrays.
            return ICE(loc, "Operator 'in' not yet implemented");
        }

        // This is implemented as a function template.
        case Tk::StarStar: {
            if (not CheckIntegral() or not ConvertToCommonType()) return nullptr;
            if (lhs->type->bit_width(*tu) < Size::Bits(2)) return ErrorExpI1();
            return BuildCall(DeclName(Tk::StarStar));
        }

        // Range construction.
        case Tk::DotDotEq:
        case Tk::DotDotLess: {
            if (not CheckIntegral() or not ConvertToCommonType()) return nullptr;

            // Check that the RHS is not the maximum representable value.
            if (op == Tk::DotDotEq) {
                auto max = Evaluate(rhs, false);
                if (max.has_value()) {
                    auto bits = lhs->type->bit_width(*tu);

                    // FIXME: Figure out SOME way to make it so BOTH 'a..=b' and 'a..<b' are ALWAYS valid.
                    if (max->cast<APInt>() == APInt::getSignedMaxValue(unsigned(bits.bits()))) {
                        Error(
                            rhs->location(),
                            "End of inclusive range of '{}' cannot be '{}.%5(max%)'",
                            rhs->type,
                            rhs->type
                        );

                        // This may cause issues if we try to evaluate it somehow, so give up.
                        return nullptr;
                    }
                }
            }

            return Build(RangeType::Get(*tu, lhs->type));
        }

        // Arithmetic operation.
        case Tk::Star:
        case Tk::Slash:
        case Tk::Percent:
        case Tk::StarTilde:
        case Tk::ColonSlash:
        case Tk::ColonPercent:
        case Tk::Plus:
        case Tk::PlusTilde:
        case Tk::Minus:
        case Tk::MinusTilde:
        case Tk::ShiftLeft:
        case Tk::ShiftLeftLogical:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
        case Tk::Ampersand:
        case Tk::VBar:
            return BuildArithmeticOrComparisonOperator(false);

        // Comparison.
        case Tk::ULt:
        case Tk::UGt:
        case Tk::ULe:
        case Tk::UGe:
        case Tk::SLt:
        case Tk::SGt:
        case Tk::SLe:
        case Tk::SGe:
            return BuildArithmeticOrComparisonOperator(true);

        // Equality comparison. This is supported for more types.
        case Tk::EqEq:
        case Tk::Neq: {
            // Comparing an optional against 'nil' requires special handling.
            if (
                auto o = dyn_cast<OptionalType>(lhs->type);
                o and rhs->type == Type::NilTy
            ) {
                lhs = MaterialiseTemporary(lhs);
                return new (*tu) OptionalNilTestExpr(lhs, rhs, op == Tk::EqEq, loc);
            }

            // Otherwise, require a common type here.
            if (not ConvertToCommonType()) return nullptr;

            // For slices and optionals, call an overloaded operator.
            if (isa<ArrayType, OptionalType, SliceType>(lhs->type)) {
                auto call = BuildCall(DeclName(Tk::EqEq));
                return op == Tk::Neq ? BuildUnaryExpr(Tk::Not, call.get(), false, loc) : call;
            }

            return Build(Type::BoolTy);
        }

        // Logical operator.
        case Tk::And:
        case Tk::Or:
        case Tk::Xor: {
            if (not MakeCondition(lhs, Spelling(op))) return {};
            if (not MakeCondition(rhs, Spelling(op))) return {};
            return Build(Type::BoolTy);
        }

        // Assignment.
        case Tk::Assign:
        case Tk::PlusEq:
        case Tk::PlusTildeEq:
        case Tk::MinusEq:
        case Tk::MinusTildeEq:
        case Tk::StarEq:
        case Tk::StarTildeEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftLeftLogicalEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq: {
            auto DiagnoseRHS = [&] {
                Error(rhs->location(), "Cannot assign value of type '{}' to '{}'", rhs->type, lhs->type);
            };

            if (not CheckLValue(lhs)) return nullptr;

            // Regular assignment.
            if (op == Tk::Assign) {
                if (not MakeRValue(lhs->type, rhs, DiagnoseRHS)) return nullptr;
                return Build(lhs->type, MLValue);
            }

            // Compound assignment.
            if (not CheckIntegral()) return nullptr;
            if (not MakeRValue(lhs->type, rhs, DiagnoseRHS)) return nullptr;
            if (not rhs) return nullptr;
            if (op != Tk::StarStarEq) return Build(lhs->type, MLValue);

            // '**=' requires a separate function since it needs to return the lhs.
            if (lhs->type->bit_width(*tu) < Size::Bits(2)) return ErrorExpI1();
            return CastExpr::Dereference(*tu, TRY(BuildCall(DeclName(Tk::StarStarEq))));
        }

        // Swap.
        case Tk::Swap: {
            if (not CheckLValue(lhs) or not CheckLValue(rhs))
                return nullptr;

            if (lhs->type != rhs->type) return Error(
                loc,
                "Cannot swap '{}' with '{}'",
                lhs->type,
                rhs->type
            );

            return Build(Type::VoidTy);
        }
    }
}

auto Sema::BuildBuiltinCallExpr(
    BuiltinCallExpr::Builtin builtin,
    ArrayRef<Expr*> args,
    SLoc call_loc
) -> Ptr<BuiltinCallExpr> {
    auto CheckNArgs = [&](usz n) {
        if (args.size() == n) return true;
        if (n == 0) Error(call_loc, "'%2({}%)' takes no arguments", builtin);
        else Error(call_loc, "'%2({}%)' takes {} arguments, got {}", builtin, n, args.size());
        return false;
    };

    auto CheckPointer = [&](usz n) -> bool {
        if (isa<PtrType>(args[n]->type)) return true;
        return Error(
            args[n]->location(),
            "Argument #{} of '%2({}%)' must be a pointer, but was {}",
            n + 1,
            builtin,
            args[n]->type
        );
    };

    auto CheckType = [&](usz n) -> bool {
        if (args[n]->type == Type::TypeTy) return true;
        return Error(
            args[n]->location(),
            "Argument #{} of '%2({}%)' must be a type, but was {}",
            n + 1,
            builtin,
            args[n]->type
        );
    };

    auto ToInt = [&](usz n) -> Ptr<Expr> {
        return BuildInitialiser(Type::IntTy, args[n], args[n]->location());
    };

    auto Make = [&](Type ret, ArrayRef<Expr*> args = {}, ValueCategory vc = Expr::RValue) {
        return BuiltinCallExpr::Create(
            *tu,
            builtin,
            ret,
            vc,
            args,
            call_loc
        );
    };

    switch (builtin) {
        using B = BuiltinCallExpr::Builtin;
        case B::Dump: Unreachable("Handled by caller");
        case B::MakeClosure: {
            if (not CheckNArgs(3)) return nullptr;
            if (not CheckType(0) or not CheckPointer(1) or not CheckPointer(2)) return nullptr;
            auto arg0 = LValueToRValue(args[0]);
            auto ty_res = Evaluate(arg0);
            if (not ty_res) return Error(
                arg0->location(),
                "Argument #1 of '{}' is not a compile-time constant",
                builtin
            );

            auto ty = ty_res->cast<Type>();
            if (not isa<ProcType>(ty)) return Error(
                arg0->location(),
                "Argument #1 of '{}' must be a procedure type, was '{}'",
                builtin,
                ty
            );

            return Make(ty, {LValueToRValue(args[1]), LValueToRValue(args[2])});
        }

        case B::Memcpy: {
            if (not CheckNArgs(3)) return nullptr;
            if (not CheckPointer(0) or not CheckPointer(1)) return nullptr;
            auto size = TRY(ToInt(2));
            return Make(Type::VoidTy, {LValueToRValue(args[0]), LValueToRValue(args[1]), size});
        }

        case B::Ptradd: {
            if (not CheckNArgs(2)) return nullptr;
            auto ptr = dyn_cast<PtrType>(args[0]->type);
            if (not ptr or ptr->elem() != tu->I8Ty) return Error(
                args[0]->location(),
                "Argument #1 of '%2({}%)' must be a pointer to '{}', but was '{}'",
                builtin,
                tu->I8Ty,
                args[0]->type
            );

            auto size = TRY(ToInt(1));
            return Make(args[0]->type, {LValueToRValue(args[0]), size});
        }

        case B::Retain: {
            if (not CheckNArgs(1)) return nullptr;
            if (not args[0]->is_mutable_lvalue()) return Error(
                args[0]->location(),
                "Argument #1 of '%2({}%)' must be a mutable lvalue",
                builtin
            );

            return Make(args[0]->type, args[0], Expr::MLValue);
        }

        case B::SplitClosure: {
            if (not CheckNArgs(1)) return nullptr;
            if (not isa<ProcType>(args[0]->type)) return Error(
                args[0]->location(),
                "Argument #1 of '%2({}%)' must be a closure, but was '{}'",
                builtin,
                args[0]->type
            );

            return Make(tu->ClosureEquivalentTupleTy, args[0]);
        }

        case B::Unreachable: {
            if (not CheckNArgs(0)) return nullptr;
            return Make(Type::NoReturnTy);
        }
    }

    Unreachable("Invalid builtin type: {}", +builtin);
}

auto Sema::BuildBuiltinMemberAccessExpr(
    BuiltinMemberAccessExpr::AccessKind ak,
    Expr* operand,
    SLoc loc
) -> Ptr<BuiltinMemberAccessExpr> {
    auto type = [&] -> Opt<Type> {
        using enum BuiltinMemberAccessExpr::AccessKind;
        switch (ak) {
            // Type Properties.
            case TypeAlign:
            case TypeArrayElems:
            case TypeArraySize:
            case TypeBits:
            case TypeBytes:
                return Type::IntTy;

            case TypeElem:
                return Type::TypeTy;

            case TypeIsArray:
            case TypeIsOptional:
            case TypeIsSlice:
            case TypeRequiresDeletion:
                return Type::BoolTy;

            case TypeName:
                return Type(tu->StrLitTy);

            // The values of these are type-dependent, e.g. 'i8.max' yields a value of
            // type 'i8'; this means we can’t really construct such an expression if the
            // actual type is not known (e.g. because it is a parameter of type 'type').
            //
            // FIXME: Parameters and variables of type 'type' should be compile-time only
            // and thus known at compile-time; in particular, parameters of type 'type' should
            // implicitly become template parameters.
            case TypeMaxVal:
            case TypeMinVal:{
                auto te = dyn_cast<TypeExpr>(operand);
                if (not te) return ICE(
                    loc,
                    "Querying property '{}' is currently only supported for literal types",
                    ak
                );
                return te->value;
            }

            // Slice Properties.
            case SliceSize:
                return Type::IntTy;

            case SliceData:
                return cast<SliceType>(operand->type)->data_ptr_type();

            // Range Properties.
            case RangeStart:
            case RangeEnd:
                return cast<RangeType>(operand->type)->elem();
        }
        Unreachable();
    }();

    if (not type) return {};
    operand = TRY(UnwrapPointersAndOptionals(operand));
    if (BuiltinMemberAccessExpr::IsTypeProperty(ak))
        operand = LValueToRValue(operand);

    return new (*tu) BuiltinMemberAccessExpr{
        *type,
        Expr::RValue,
        operand,
        ak,
        loc
    };
}

auto Sema::BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, SLoc loc) -> Ptr<Expr> {
    auto call_arg_list = cast<TupleExpr>(TRY(BuildTuple(args, Type::CallArgListTy, {}, false, loc)));
    return BuildCallExpr(callee_expr, call_arg_list, loc, false);
}

auto Sema::BuildCallExpr(
    Expr* callee_expr,
    std::same_as<TupleExpr> auto* args,
    SLoc loc,
    bool is_associated_call
) -> Ptr<Expr> {
    auto ReportAsOverloadResolutionFailure = [&](Callee logical_callee, Candidate::Status s) {
        ReportSingleOverloadResolutionFailure(
            logical_callee,
            std::move(s),
            std::nullopt,
            args,
            loc
        );
    };

    // Check for duplicate argument names.
    if (args->is_named()) {
        llvm::StringSet<> param_names;
        for (const auto& [name, loc] : args->names()) {
            if (name.empty()) continue;
            if (not param_names.insert(name.str()).second) return Error(
                loc,
                "Duplicate {} name '{}'",
                isa<TypeExpr>(callee_expr) ? "field" : "argument",
                name
            );
        }
    }

    // If this is an overload set, perform overload resolution.
    Expr* resolved_callee = nullptr;
    SmallVector<Expr*> converted_args;
    if (auto os = dyn_cast<OverloadSetExpr>(callee_expr)) {
        ProcDecl* d{};
        std::tie(d, converted_args) = PerformOverloadResolution(os, args, is_associated_call, loc);
        if (not d) return nullptr;
        resolved_callee = CreateReference(d, loc).get();
    }

    // If the ‘callee’ is a type, then this is an initialiser call.
    else if (isa<TypeExpr>(callee_expr)) {
        auto type = Evaluate(callee_expr);
        if (not type) return ICE(
            callee_expr->location(),
            "Failed to evaluate expression designating a type"
        );

        // Only record types can use named initialisers.
        auto ty = type.value().cast<Type>();
        if (args->is_named() and not isa<RecordType>(ty)) {
            return Error(
                callee_expr->location(),
                "Cannot initialise type '{}' using named initialiser",
                ty
            );
        }

        // Strip the non-existent names and build an initialiser.
        return BuildInitialiser(ty, args, loc);
    }

    // If the type of this is a procedure, then we can skip overload
    // resolution. While this also means we have some code duplication,
    // we don’t have to build conversion sequences here and can instead
    // apply them immediately, and overload resolution also wouldn’t
    // work for indirect calls since for those we don’t have a reference
    // to the procedure declaration.
    else if (auto ty = dyn_cast<ProcType>(callee_expr->type)) {
        Callee logical_callee{ty};
        if (auto ref = dyn_cast<ProcRefExpr>(callee_expr->ignore_parens()))
            logical_callee = ref->decl;

        // The callee must be an rvalue.
        resolved_callee = LValueToRValue(callee_expr);

        // Check arg count.
        if (not logical_callee.argument_count_matches_parameters(args->num_values())) {
            ReportAsOverloadResolutionFailure(logical_callee, Candidate::ArgumentCountMismatch{});
            return {};
        }

        // Resolve named parameters.
        SmallVector<Expr*> ordered_args;
        if (args->is_named()) {
            if (logical_callee.is_indirect()) return Error(loc, "Named arguments are not allowed in indirect calls");
            auto resolved = ResolveNamedArguments(args, logical_callee);
            if (not resolved) {
                ReportAsOverloadResolutionFailure(logical_callee, std::move(resolved.error()));
                return {};
            }

            ordered_args = std::move(*resolved);
        } else {
            append_range(ordered_args, args->values());
        }

        bool ok = true;
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, SLoc loc) {
            auto p = ty->params()[param_index];
            auto arg = BuildInitialiser(
                p.type,
                args,
                loc,
                p.type->pass_by_reference(tu->target(), p.intent),
                is_associated_call and param_index == 0
            );

            // Point to the procedure if this is a direct call.
            if (not arg) {
                if (auto d = logical_callee.decl().get_or_null())
                    NoteParameter(d, u32(param_index));

                ok = false;
                return;
            }

            converted_args.push_back(arg.get());
        };

        ConvertArgumentsForCall(ordered_args, logical_callee, ConvertArg, loc);

        // We can do this before adding the varargs arguments since we only allow
        // passing trivially copyable types as varargs arguments, and those always
        // have the 'copy' intent and require no intent checking.
        if (not ok or not CheckIntents(ty, converted_args))
            return nullptr;
    }

    // Otherwise, we have no idea how to call this thing.
    else {
        return Error(
            callee_expr->location(),
            "Expression of type '{}' is not callable",
            callee_expr->type
        );
    }

    // And check C varargs arguments.
    auto ty = cast<ProcType>(resolved_callee->type);
    if (ty->has_c_varargs()) {
        for (auto a : args->values().drop_front(ty->params().size())) {
            // Codegen is not set up to handle variadic arguments that are larger
            // than a word, so reject these here. If you need one of those, then
            // seriously, wtf are you doing.
            if (
                a->type == Type::IntTy or
                a->type == Type::BoolTy or
                a->type == Type::NoReturnTy or
                (isa<IntType>(a->type) and cast<IntType>(a->type)->bit_width() <= Size::Bits(64)) or
                isa<PtrType>(a->type)
            ) {
                converted_args.push_back(LValueToRValue(a));
            } else {
                Error(
                    a->location(),
                    "Passing a value of type '{}' as a varargs argument is not supported",
                    a->type
                );
            }
        }
    }

    // Check that we can even call this at this point.
    if (ty->ret() == Type::DeducedTy) {
        Error(loc, "Cannot call procedure before its return type has been deduced");
        if (auto p = dyn_cast<ProcRefExpr>(resolved_callee)) {
            Note(p->decl->location(), "Declared here");
            Remark("\rTry specifying the return type explicitly: '%1(->%) <type>'");
        }
        return nullptr;
    }

    // Finally, create the call.
    auto proc = cast<ProcType>(resolved_callee->type);
    return CallExpr::Create(
        *tu,
        proc->ret(),
        proc->return_value_category(),
        resolved_callee,
        converted_args,
        loc
    );
}

auto Sema::BuildDeclRefExpr(
    ArrayRef<DeclNameLoc> names,
    SLoc loc,
    Opt<Type> desired_type
) -> Ptr<Expr> {
    return BuildDeclRefExpr(
        InitialDREScope::None,
        nullptr,
        names,
        loc,
        desired_type
    );
}

auto Sema::BuildDeclRefExpr(
    InitialDREScope initial_scope,
    Scope* root,
    ArrayRef<DeclNameLoc> names,
    SLoc loc,
    Opt<Type> desired_type
) -> Ptr<Expr> {
    Assert((root != nullptr) == (initial_scope == InitialDREScope::Expr));
    Assert(not names.empty());

    // Determine the initial scope.
    Ptr<Scope> scope;
    switch (initial_scope) {
        case InitialDREScope::None: break;
        case InitialDREScope::Global: scope = global_scope(); break;
        case InitialDREScope::Expr: scope = root; break;
    }

    // This does perform unqualified lookup for e.g. '::a', but that’s fine
    // since the global scope has no parents anyway.
    auto res = LookUpName(scope, names, loc, LookupHint::Any);
    if (res.successful() or res.is_overload_set())
        return CreateReferenceOrOverloadSet(loc, res);

    // If we failed to find anything, and the desired type is an enum type, try to
    // see if this is one of its enumerators.
    //
    // FIXME: This doesn't work with overload resolution (because we don't even know
    // that we want an enum when we process the arguments). Instead of doing this,
    // introduce an 'UnknownSymbolExpr' and 'UnknownSymbolType' (with the symbol
    // tracked *in the type*), and introduce a conversion from 'UnknownSymbolType<X>'
    // to any enum that has an enumerator named 'X'.
    //
    // Additionally, make sure every UnresolvedSymbolType is unique and store a pointer
    // to it in Sema at creation time (erase it if the symbol is actually resolved), and
    // diagnose it 1. if we never resolve it, or 2. if we fail to convert it to a different
    // type in a place where that is a hard error.
    //
    // An 'UnresolvedSymbolType' can contain multiple symbols, which can happen if we
    // have e.g. 'if x then A else B' where 'A' and 'B' are enumerators.
    if (
        initial_scope == InitialDREScope::None and
        names.size() == 1 and
        desired_type and
        isa<EnumType>(*desired_type) and
        res.result == LookupResult::Reason::NotFound
    ) {
        auto e = dyn_cast<EnumType>(*desired_type);
        auto res = LookUpNameInScope(e->scope(), names.front(), LookupHint::Any);
        if (res.successful()) return CreateReference(res.decls.front(), loc);
    }

    ReportLookupFailure(std::move(res));
    return {};
}

auto Sema::MaybeBuildDeleteExpr(Expr* arg, bool implicit, SLoc loc) -> Ptr<DeleteExpr> {
    // If the type does not require deletion, just throw away the argument.
    if (not arg->type->requires_deletion())
        return nullptr;

    // Only lvalues can be deleted.
    arg = MaterialiseTemporary(arg);
    if (not arg->is_mutable_lvalue()) return Error(
        loc,
        "Argument of '%1(delete%)' must be a mutable lvalue"
    );

    return new (*tu) DeleteExpr(arg, implicit, loc);
}

auto Sema::BuildEvalExpr(Stmt* arg, SLoc loc) -> Ptr<Expr> {
    // An eval expression returns an rvalue.
    if (auto e = dyn_cast<Expr>(arg)) {
        auto init = BuildInitialiser(arg->type_or_void(), e, loc);
        if (not init) return nullptr;
        arg = init.get();
    }

    return EvaluateIntoExpr(arg, loc);
}

auto Sema::BuildExplicitCast(Type to, Expr* arg, SLoc loc, bool is_hard_cast) -> Ptr<Expr> {
    auto from = arg->type;
    auto Make = [&](CastExpr::CastKind k, ValueCategory vc = Expr::RValue) {
        return new (*tu) CastExpr(to, k, arg, loc, false, vc);
    };

    auto HardCast = [&]{
        if (is_hard_cast) return;
        Error(loc, "Cast from '{}' to '{}' requires '%1(as!%)'", from, to);
    };

    auto SoftCast = [&]{
        if (not is_hard_cast) return;
        if (curr_proc().proc->is_instantiation()) return;
        Warn(loc, "Cast from '{}' to '{}' should use '%1(as%)'", from, to);
    };

    // This is a no-op if the types are the same.
    if (to == from) return arg;

    // Casting from integer/enum types to integers is always allowed.
    if ((from->is_integer() or isa<EnumType>(from)) and to->is_integer()) {
        SoftCast();
        arg = LValueToRValue(arg);
        return Make(CastExpr::Integral);
    }

    // Casting to void does nothing.
    if (to == Type::VoidTy) {
        SoftCast();
        return Make(CastExpr::ExplicitDiscard);
    }

    // Casting between pointers.
    if (isa<PtrType>(from) and isa<PtrType>(to)) {
        auto pfrom = cast<PtrType>(from);
        auto pto = cast<PtrType>(to);

        // Cast between completely different pointer types.
        if (pfrom->elem() != pto->elem()) {
            HardCast();
            arg = LValueToRValue(arg);
        }

        // Cast from an immutable to a mutable pointer or vice versa; this preserves
        // the value category, but it does require 'as!' if we’re stripping immutability.
        else {
            Assert(pfrom->is_immutable() != pto->is_immutable());
            if (pfrom->is_immutable()) HardCast();
            else SoftCast();
        }

        return Make(CastExpr::Nop, arg->value_category);
    }

    // Casting between slices (adding/removing immutability).
    if (isa<SliceType>(from) and isa<SliceType>(to)) {
        auto sfrom = cast<SliceType>(from);
        auto sto = cast<SliceType>(to);
        if (sfrom->elem() == sto->elem()) {
            Assert(sfrom->is_immutable() != sto->is_immutable());
            if (sfrom->is_immutable()) HardCast();
            else SoftCast();
            return Make(CastExpr::Nop, arg->value_category);
        }
    }

    // So is casting from integers/enums to enums.
    if ((from->is_integer_or_bool() or isa<EnumType>(from)) and isa<EnumType>(to)) {
        HardCast();
        arg = LValueToRValue(arg);
        return Make(CastExpr::Integral);
    }

    // If the source type is an optional, and the target type isn’t, unwrap it.
    if (isa<OptionalType>(from) and not isa<OptionalType>(to)) return BuildExplicitCast(
        to,
        UnwrapOptional(arg, loc),
        loc,
        is_hard_cast
    );

    // For everything else, just try to build an initialiser.
    auto init = BuildInitialiser(to, arg, loc);
    if (init) SoftCast();
    return init;
}

auto Sema::BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, SLoc loc) -> Ptr<IfExpr> {
    auto Build = [&](Type ty, ValueCategory val) {
        return new (*tu) IfExpr(
            ty,
            val,
            cond,
            then,
            else_,
            false,
            loc
        );
    };

    // Degenerate case: the condition does not return.
    if (cond->type == Type::NoReturnTy) return Build(Type::NoReturnTy, Expr::RValue);

    // Condition must be a bool.
    if (not MakeCondition(cond, "if")) return {};

    // If there is no else branch, or if either branch is not an expression,
    // the type of the 'if' is 'void'.
    if (
        not else_ or
        not isa<Expr>(then) or
        not isa<Expr>(else_.get())
    ) return Build(Type::VoidTy, Expr::RValue);

    // Otherwise, the type and value category are the common type / vc.
    Expr* exprs[2] { cast<Expr>(then), cast<Expr>(else_.get()) };
    auto tvc = ComputeCommonTypeAndValueCategory(exprs);
    return new (*tu) IfExpr(
        tvc.type(),
        tvc.value_category(),
        cond,
        exprs[0],
        exprs[1],
        false,
        loc
    );
}

auto Sema::BuildMatchExpr(
    Ptr<Expr> control_expr,
    Type ty,
    MutableArrayRef<MatchCase> cases,
    SLoc loc
) -> Ptr<Expr> {
    bool exhaustive = false;
    if (auto control = control_expr.get_or_null()) {
        control_expr = control = Save(control);
        auto ty = control->type;
        exhaustive = CheckMatchExhaustive(loc, control, ty, cases);
    } else {
        for (auto& c : cases) {
            if (c.cond.is_wildcard()) continue;
            (void) MakeCondition(c.cond.expr(), "match");
        }

        // Warn about unreachable branches.
        auto it = find_if(cases, [](auto& c) { return c.cond.is_wildcard(); });
        if (it != cases.end()) {
            exhaustive = true;
            MarkUnreachableAfter(it, cases);
        }

        if (not exhaustive and ty != Type::VoidTy and ty != Type::DeducedTy) Error(
            loc,
            "A 'match' with a fixed result type must have a wildcard arm"
        );
    }

    // If type deduction is required, compute the common type of the bodies of
    // any reachable cases, provided that all of them are expressions.
    TypeAndValueCategory tvc;
    if (ty == Type::DeducedTy) {
        if (exhaustive and llvm::all_of(cases, [](auto& c) { return c.unreachable or isa<Expr>(c.body); })) {
            SmallVector<Expr*> exprs;
            SmallVector<u32> indices;
            for (auto [i, c] : enumerate(cases)) {
                if (not c.unreachable) {
                    exprs.push_back(cast<Expr>(c.body));
                    indices.push_back(u32(i));
                }
            }

            tvc = ComputeCommonTypeAndValueCategory(exprs);

            // This process may emit lvalue-to-rvalue conversions.
            for (auto [i, e] : zip(indices, exprs)) cases[i].body = e;
        }
    }

    // Otherwise, we have a fixed target type.
    else {
        // If possible, produce an lvalue. We don’t reuse the common type
        // computation algorithm since we already know what type we want to
        // end up with here.
        if (llvm::all_of(cases, [&](auto& c) {
            return c.unreachable or (
                c.body->type_or_void() == ty and
                c.body->value_category_or_rvalue() != Expr::RValue
            );
        })) {
            // If any case yields an immutable lvalue, the entire expression is immutable.
            auto vc = Expr::MLValue;
            for (auto& c : cases) {
                if (c.body->value_category_or_rvalue() == Expr::ILValue) {
                    vc = Expr::ILValue;
                    break;
                }
            }

            tvc = {ty, vc};
        }

        // Convert each match arm. For 'void' we just skip this step because
        // that just means that this produces no value.
        else if (ty != Type::VoidTy) {
            tvc = {ty, Expr::RValue};
            for (auto& c : cases) {
                if (c.unreachable) continue;
                if (auto e = dyn_cast<Expr>(c.body); not e) {
                    Error(c.body->location(), "Expected expression");
                } else if (auto init = BuildInitialiser(ty, e, c.body->location()).get_or_null()) {
                    c.body = init;
                }
            }
        }
    }

    return MatchExpr::Create(
        *tu,
        control_expr,
        tvc.type(),
        tvc.value_category(),
        cases,
        loc
    );
}

auto Sema::BuildMemberAccessExpr(Expr* base, FieldDecl* field, SLoc loc) -> Ptr<Expr> {
    base = TRY(UnwrapPointersAndOptionals(base));
    base = MaterialiseTemporary(base);
    return new (*tu) MemberAccessExpr(base, field, loc);
}

auto Sema::BuildParamDecl(
    ProcDecl* proc,
    const ParamTypeData* param,
    u32 index,
    bool with_param,
    bool immutable,
    DeclNameLoc name
) -> ParamDecl* {
    auto decl = new (*tu) ParamDecl(
        param,
        Expr::LValue(immutable),
        name,
        proc,
        index,
        with_param
    );

    DeclareLocal(decl);
    return decl;
}

auto Sema::BuildProcDeclInitial(
    Scope* proc_scope,
    ProcType* ty,
    DeclName name,
    SLoc loc,
    ParsedProcAttrs attrs,
    InheritedProcedureProperties props,
    ProcTemplateDecl* pattern
) -> ProcDecl* {
    auto parent_scope = curr_scope() == proc_scope
        ? proc_scope->parent()
        : proc_scope;

    // Get the parent procedure, which determines whether this is a nested
    // procedure; top-level procedures (that is, procedures declared at the
    // global scope) are *not* considered nested inside the initialiser procedure
    // (since the latter is just an implementation detail).
    //
    // It *is* possible for a procedure to have the initialiser procedure
    // as its parent, however. Assuming we’re at the top-level:
    //
    //     proc x {} // Parent is nullptr.
    //     {
    //         proc x {} // Parent is initialiser procedure.
    //     }
    //
    // Note that the parent computation assumes that we’re currently inside the
    // lexical parent of this procedure; this *should* always be true since we
    // can’t e.g. refer to a nested procedure template from outside its parent,
    // but if we ever start doing weird things with templates (like returning a
    // *template* from a procedure), then this will no longer work properly and
    // we’ll require some other way of keeping track of the lexical parent.
    //
    // For templates, instead use the parent of the pattern.
    ProcDecl* parent{};
    if (pattern) parent = pattern->parent.get_or_null();
    else if (parent_scope != global_scope()) parent = curr_proc().proc;

    // Actually create the procedure.
    auto proc = ProcDecl::Create(
        *tu,
        nullptr,
        ty,
        name,
        attrs.extern_ ? Linkage::Imported : Linkage::Internal,
        attrs.nomangle ? Mangling::None : Mangling::Source,
        parent,
        props,
        loc
    );

    // Remember what template we were instantiated from.
    proc->set_instantiated_from(pattern);

    // If this is an overloaded operator, check its arity and other properties.
    if (
        proc->is_overloaded_operator() and
        proc->valid() and
        not CheckOverloadedOperator(proc, attrs.builtin_operator)
    ) proc->set_invalid();

    // Don’t e.g. diagnose calls to this if the type is invalid.
    if (not ty) proc->set_invalid();

    // Add the procedure to the module and the parent scope.
    proc->scope = proc_scope;
    return proc;
}

auto Sema::BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr> {
    // If the body is not a block, build an implicit return.
    if (not isa<BlockExpr>(body)) body = BuildReturnExpr(body, body->location(), true);

    // Make sure all paths return a value. First, if the body is
    // 'noreturn', then that means we never actually get here.
    auto body_ret = body->type;
    if (body_ret == Type::NoReturnTy) return body;

    // Next, a function marked as returning void requires no checking
    // and is allowed to not return at all; invalid return expressions
    // are checked when we first encounter them.
    //
    // We do, however, need to synthesise a return statement in that case.
    if (proc->return_type() == Type::VoidTy) {
        Assert(isa<BlockExpr>(body));
        auto ret = BuildReturnExpr(nullptr, body->location(), true);
        return BlockExpr::Create(*tu, nullptr, {body, ret}, body->location());
    }

    // In any other case, we’re missing a return statement and have
    // fallen off the end.
    return Error(
        body->location(),
        "Procedure '{}' must return a value",
        proc->name
    );
}

auto Sema::BuildReturnExpr(Ptr<Expr> value, SLoc loc, bool implicit) -> ReturnExpr* {
    // Perform return type deduction.
    auto proc = curr_proc().proc;
    if (proc->return_type() == Type::DeducedTy) {
        auto proc_type = proc->proc_type();
        Type deduced = Type::VoidTy;
        if (auto val = value.get_or_null()) deduced = val->type;
        proc->type = ProcType::AdjustRet(*tu, proc_type, deduced);
    }

    // Perform any necessary conversions.
    //
    // If the type is zero-sized, there is no need to do anything since we’ll
    // drop it anyway.
    if (auto val = value.get_or_null()) {
        value = BuildInitialiser(
            proc->return_type(),
            {val},
            val->location(),
            proc->returns_lvalue()
        );

        // Returning an immutable reference from a function that returns a
        // mutable reference is not supported.
        if (
            proc->return_value_category() == Expr::MLValue and
            value.present() and value.get()->is_immutable_lvalue()
        ) Error(loc, "Cannot return immutable value as mutable");
    }

    return new (*tu) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildStaticIfExpr(
    Expr* cond,
    ParsedStmt* then,
    Ptr<ParsedStmt> else_,
    SLoc loc
) -> Ptr<Stmt> {
    if (not MakeCondition(cond, "#if")) return {};
    auto val = Evaluate(cond);
    if (not val) return {};

    // If there is no else clause, and the condition is false, return
    // an empty RValue. This must be an expression, otherwise something
    // like 'a = #if false 1' (which is invalid), will produce a weird
    // diagnostic instead of ‘cannot assign void to int’.
    auto cond_val = val->cast<APInt>().getBoolValue();
    if (not cond_val and not else_) return MakeConstExpr(nullptr, eval::RValue(), loc);

    // Otherwise, translate the appropriate branch now, and throw
    // away the other one.
    return TranslateStmt(cond_val ? then : else_.get());
}

auto Sema::BuildTuple(
    ArrayRef<Expr*> exprs,
    Opt<Type> desired_ty,
    ArrayRef<DeclNameLoc> names,
    bool may_build_paren_expr,
    SLoc loc
) -> Ptr<Expr> {
    if (desired_ty == Type::CallArgListTy) return TupleExpr::Create(
        *tu,
        Type::CallArgListTy,
        exprs,
        names,
        loc
    );

    if (
        exprs.size() == 1 and
        (names.empty() or not names.front().name.valid()) and
        may_build_paren_expr
    ) return new (*tu) ParenExpr(exprs.front(), loc);

    // Compute the tuple type.
    auto types = llvm::to_vector(vws::transform(exprs, [&](Expr* e) { return e->type; }));
    if (not all_of(types, [&] (Type ty) { return CheckFieldType(ty, loc); })) return {};
    return TupleExpr::Create(*tu, TupleType::Get(*tu, types), exprs, names, loc);
}

auto Sema::BuildTypeExpr(Type ty, SLoc loc) -> TypeExpr* {
    return new (*tu) TypeExpr(ty, loc);
}

auto Sema::BuildUnaryExpr(Tk op, Expr* operand, bool postfix, SLoc loc) -> Ptr<Expr> {
    auto Build = [&](Type ty, ValueCategory cat) {
        return new (*tu) UnaryExpr(ty, cat, op, operand, postfix, loc);
    };

    auto BuildIntOp = [&](ValueCategory vc = Expr::RValue) -> Ptr<Expr> {
        if (not operand->type->is_integer()) return Error(
            loc,
            "Operand of '%1({}%)' must be an integer, but was '{}'",
            Spelling(op),
            operand->type
        );

        return Build(operand->type, vc);
    };

    if (postfix) {
        switch (op) {
            default: Unreachable("Invalid postfix operator: {}", op);
            case Tk::PlusPlus:
            case Tk::MinusMinus: {
                if (not operand->is_lvalue()) return Error(
                    loc,
                    "Operand of '%1({}%)' must be an lvalue",
                    Spelling(op)
                );

                return BuildIntOp();
            }
        }
    }

    // '&' cannot be overloaded.
    if (op == Tk::Ampersand) {
        // FIXME: This message needs improving; we shouldn’t expect users
        // to know what an lvalue is (or why something isn’t an lvalue in
        // the case of e.g. if/match).
        if (not operand->is_lvalue()) return Error(loc, "Cannot take address of non-lvalue");
        return Build(
            PtrType::Get(*tu, operand->type, operand->is_immutable_lvalue()),
            Expr::RValue
        );
    }

    // Handle overloaded operators.
    if (IsUserDefinedOverloadedOperator(op, operand->type)) {
        auto ref = BuildDeclRefExpr(DeclNameLoc{op, loc}, loc);
        if (not ref) return nullptr;
        return BuildCallExpr(ref.get(), operand, loc);
    }

    // Handle prefix operators.
    switch (op) {
        default: Unreachable("Invalid unary operator: {}", op);

        // Pointer -> Lvalue.
        case Tk::Caret: {
            // Check that this is an (optional) pointer.
            auto ptr = dyn_cast<PtrType>(operand->type);
            if (not ptr) {
                auto opt = dyn_cast<OptionalType>(operand->type);
                if (opt) ptr = dyn_cast<PtrType>(opt->elem());
                if (not ptr) return Error(
                    loc,
                    "Cannot dereference value of non-pointer type '{}'",
                    operand->type
                );

                // If it is an optional, unwrap it.
                operand = UnwrapOptional(operand, loc);
            }

            operand = LValueToRValue(operand);
            return Build(ptr->elem(), Expr::LValue(ptr->is_immutable()));
        }

        // Procedure call inlining.
        case Tk::Inline: {
            if (auto call = dyn_cast<CallExpr>(operand->ignore_parens())) {
                call->is_inline = true;
            } else if (isa<BuiltinCallExpr>(operand->ignore_parens())) {
                Error(operand->location(), "Call to builtin procedure cannot be '%1(inline%)'");
            } else {
                Error(operand->location(), "Argument of '%1(inline%)' must be a procedure call");
            }

            return operand;
        }

        // Boolean negation.
        case Tk::Not: {
            if (not MakeRValue(Type::BoolTy, operand, "Operand", "not")) return {};
            return Build(Type::BoolTy, Expr::RValue);
        }

        // Check for overflow if the argument is a literal.
        case Tk::Minus: {
            if (auto i = dyn_cast<IntLitExpr>(operand->ignore_parens())) {
                auto val = i->storage.value();
                val.negate();
                if (val.isStrictlyPositive()) Error(
                    loc,
                    "Type '{}' cannot represent the value '%1(-%)%5({}%)'",
                    i->type,
                    i->storage.str(false)
                );
            }

            operand = LValueToRValue(operand);
            return BuildIntOp();
        }

        // Arithmetic operators.
        case Tk::Plus:
        case Tk::Tilde:
            operand = LValueToRValue(operand);
            return BuildIntOp();

        // Increment and decrement.
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            if (not operand->is_mutable_lvalue()) return Error(
                operand->location(),
                "Invalid operand for '{}'",
                Spelling(op)
            );

            return BuildIntOp(Expr::MLValue);
        }
    }
}

// ============================================================================
//  Building Types
// ============================================================================
auto Sema::BuildArrayType(TypeLoc base, Expr* size_expr) -> Opt<Type> {
    auto size = TRY(Evaluate(size_expr));
    auto integer = size.dyn_cast<APInt>();

    // Check that the size is a 64-bit integer.
    if (not integer) return Error(
        size_expr->location(),
        "Array size must be an integer, but was '{}'",
        size.type()
    );

    if (not integer->isSingleWord()) return Error(
        size_expr->location(),
        "Array size must fit into a signed 64-bit integer"
    );

    auto v = integer->getSExtValue();
    return BuildArrayType(base, v, base.loc);
}

auto Sema::BuildArrayType(TypeLoc base, i64 size, SLoc loc) -> Opt<Type> {
    if (not CheckVariableType(base.ty, base.loc)) return {};
    if (size < 0) return Error(
        loc,
        "Array size cannot be negative (value: {})",
        size
    );

    return ArrayType::Get(*tu, base.ty, size);
}

auto Sema::BuildCompleteStructType(
    String name,
    RecordLayout* layout,
    SLoc decl_loc
) -> StructType* {
    auto scope = tu->create_scope<StructScope>(global_scope());
    for (auto f : layout->fields())
        if (f->is_valid)
            AddDeclToScope(scope, f);

    return StructType::Create(
        *tu,
        scope,
        name,
        decl_loc,
        layout
    );
}

auto Sema::BuildSliceType(Type base, bool immutable, SLoc loc) -> Opt<Type> {
    if (not CheckVariableType(base, loc)) return {};
    return SliceType::Get(*tu, base, immutable);
}

auto Sema::BuildTupleType(ArrayRef<TypeLoc> types) -> Opt<Type> {
    bool ok = true;
    for (auto [ty, loc] : types)
        if (not CheckFieldType(ty, loc))
            ok = false;

    if (not ok) return {};
    return TupleType::Get(*tu, llvm::to_vector(vws::transform(types, &TypeLoc::ty)));
}
