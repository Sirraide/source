#include <srcc/Frontend/Sema.hh>

using namespace srcc;

#define TRY(...) ({                      \
    auto _res = (__VA_ARGS__);           \
    if (not _res) return utils::Falsy(); \
    *_res;                               \
})

// ============================================================================
//  Templates.
// ============================================================================
auto Sema::DeduceType(ParsedStmt* parsed_type, Type input_type) -> Opt<Type> {
    if (isa<ParsedTemplateType>(parsed_type))
        return input_type;

    // TODO: Support more complicated deduction.
    return {};
}

auto Sema::InstantiateTemplate(
    ProcTemplateDecl* pattern,
    TemplateSubstitution& info,
    SLoc inst_loc
) -> ProcDecl* {
    if (info.instantiation) return info.instantiation;

    // Translate the declaration proper.
    info.instantiation = BuildProcDeclInitial(
        info.scope,
        info.type,
        pattern->name,
        pattern->location(),
        pattern->pattern->type->attrs,
        pattern->props,
        pattern
    );

    // Cache the instantiation.
    tu->template_instantiations[pattern].push_back(info.instantiation);

    // Translate the body and record the instantiation.
    return TranslateProc(
        info.instantiation,
        pattern->pattern->body,
        pattern->pattern->params()
    );
}

auto Sema::SubstituteTemplate(
    ProcTemplateDecl* proc_template,
    ArrayRef<TypeLoc> input_types
) -> SubstitutionResult {
    // Perform deduction.
    //
    // We need to do this *before* caching, else e.g. 'proc f($T a, T b)' will
    // be instantiated twice if called with 'f(i32, i16)' and 'f(i32, i8)', which
    // is outright wrong.
    using Deduced = std::pair<u32, TypeLoc>;
    TreeMap<String, Deduced> deduced;

    // First, handle all deduction sites.
    //
    // Note that we might not have any if this is a variadic function with
    // no actual template parameters.
    auto param_types = proc_template->pattern->type->param_types();
    auto it = template_deduction_infos.find(proc_template);
    if (it != template_deduction_infos.end()) {
        for (const auto& [name, indices] : it->second) {
            for (auto i : indices) {
                auto parsed = param_types[i];

                // There might be no corresponding argument if we didn’t
                // pass any variadic arguments to this. In that case, use
                // what ever we deduced earlier, or 'void' if this is the
                // only place where this parameter is deduced.
                if (input_types.size() == i) {
                    Assert(proc_template->has_variadic_param);
                    deduced.try_emplace(name, Deduced{u32(i), {Type::VoidTy, parsed.type->loc}});
                    continue;
                }

                // Otherwise, deduce the parameter from its corresponding argument(s).
                auto ty = DeduceType(parsed.type, input_types[i].ty);
                if (not ty) {
                    return SubstitutionResult::DeductionFailed{
                        name,
                        i
                    };
                }

                if (parsed.variadic) {
                    Assert(proc_template->has_variadic_param);

                    // We already deduced 'ty' from the first variadic argument, so
                    // start at the one after it, if there is one, and make sure we
                    // get the same type for each argument.
                    for (u32 j = u32(param_types.size()), e = u32(input_types.size()); j < e; j++) {
                        auto next = DeduceType(parsed.type, input_types[j].ty);
                        if (not next) {
                            return SubstitutionResult::DeductionFailed{
                                name,
                                j,
                            };
                        }

                        if (next != ty) return SubstitutionResult::DeductionAmbiguous{
                            name,
                            u32(i),
                            u32(j),
                            *ty,
                            *next,
                        };
                    }
                }


                // If the type has not been deduced yet, remember it.
                auto [it, inserted] = deduced.try_emplace(name, Deduced{u32(i), {*ty, parsed.type->loc}});
                if (inserted) continue;

                // Otherwise, check that the deduction result is the same.
                if (it->second.second.ty != ty) {
                    return SubstitutionResult::DeductionAmbiguous{
                        name,
                        it->second.first,
                        u32(i),
                        it->second.second.ty,
                        *ty,
                    };
                }
            }
        }
    }

    // Next, deduce any 'var' parameters.
    SmallVector<Type> deduced_var_parameters;
    for (auto [i, p] : enumerate(param_types)) {
        if (not IsBuiltinVarType(p.type)) continue;
        if (not p.variadic) deduced_var_parameters.push_back(input_types[i].ty);
        else {
            Assert(i == param_types.size() - 1);

            // Build a tuple consisting of the remaining arguments.
            // TODO: Maybe a failure to build the type here should not be a hard error.
            auto ty = TRY(BuildTupleType(input_types.drop_front(i)));
            deduced_var_parameters.push_back(ty);
        }
    }

    // Now that deduction is done, check if we’ve substituted this before.
    llvm::FoldingSetNodeID id;
    for (const auto& [_, d] : deduced) id.AddPointer(d.second.ty.ptr());
    for (auto ty : deduced_var_parameters) id.AddPointer(ty.ptr());

    // Don’t hold on to a reference to the folding set (or the insert position)
    // here as we might end up invalidating it if template instantiation occurs
    // during the translation of the procedure type below.
    void* _unused{};
    auto info = template_substitutions[proc_template].FindNodeOrInsertPos(id, _unused);
    if (info) return info;

    // Create a scope for the procedure and save the template arguments there.
    EnterScope scope{*this, EnterScope::Procedure, proc_template->props.associated_type};
    for (auto [name, d] : deduced) AddDeclToScope(
        scope.get(),
        new (*tu) TemplateTypeParamDecl(name, d.second)
    );

    // Now that that is done, we can convert the type properly.
    auto ty = TranslateProcType(
        proc_template->pattern->type,
        true,
        deduced_var_parameters
    );

    // Mark that we’re done substituting.
    for (auto d : scope.get()->decls())
        cast<TemplateTypeParamDecl>(d)->in_substitution = false;
    if (not ty) return {};

    // Check the constraint.
    if (auto where = proc_template->pattern->where.get_or_null()) {
        auto constraint = TRY(TranslateExpr(where));
        if (not MakeCondition(constraint, "where")) return {};
        auto value = Evaluate(constraint);
        if (not value.has_value()) return {};
        if (not value->cast<APInt>().getBoolValue())
            return SubstitutionResult::ConstraintNotSatisfied();
    }

    // Store the type for later if substitution succeeded.
    info = new (*tu) TemplateSubstitution(
        id.Intern(tu->allocator()),
        cast<ProcType>(*ty),
        scope.get()
    );

    template_substitutions[proc_template].InsertNode(info);
    return info;
}

// ============================================================================
//  Building Calls and Overload Resolution
// ============================================================================
template <typename Visitor>
auto Sema::Callee::visit(Visitor&& v) const {
    if (auto p = dyn_cast<ProcDecl>()) return std::invoke(v, p);
    if (auto p = dyn_cast<ProcTemplateDecl>()) return std::invoke(v, p);
    if (auto s = dyn_cast<RecordType>()) return std::invoke(v, s);
    return std::invoke(v, cast<ProcType>());
}

template <typename Visitor>
auto Sema::Callee::visit_type(Visitor&& v) const {
    if (auto p = dyn_cast<ProcDecl>()) return std::invoke(v, p->proc_type());
    if (auto p = dyn_cast<ProcTemplateDecl>()) return std::invoke(v, p);
    if (auto s = dyn_cast<RecordType>()) return std::invoke(v, s);
    return std::invoke(v, cast<ProcType>());
}

bool Sema::Callee::argument_count_matches_parameters(u32 num_args) {
    auto required = num_non_variadic_params();
    if (num_args == required) return true;
    if (num_args < required) return false;
    return has_c_varargs() or is_variadic();
}

bool Sema::Callee::has_c_varargs() const {
    return visit_type(utils::Overloaded{
        [](ProcType* ty) { return ty->has_c_varargs(); },
        [](ProcTemplateDecl* decl) { return decl->pattern->type->attrs.c_varargs; },
        [](RecordType*) { return false; },
    });
}

auto Sema::Callee::index_of_named_param(String name) -> std::optional<u32> {
    return visit(utils::Overloaded{
        [](ProcType* ty) -> std::optional<u32> { return std::nullopt; },
        [&](ProcDecl* decl) { return decl->index_of_named_param(name); },
        [&](ProcTemplateDecl* decl) {
            return utils::index_of<u32>(decl->pattern->params(), name, &ParsedVarDecl::name);
        },
        [&](RecordType* ty) -> std::optional<u32> {
            return utils::index_of<u32>(ty->layout().fields(), name, &FieldDecl::name);
        },
    });
}

bool Sema::Callee::is_variadic() const {
    return visit_type(utils::Overloaded{
        [](ProcType* ty) { return ty->is_variadic(); },
        [](ProcTemplateDecl* decl) { return decl->has_variadic_param; },
        [](RecordType* ty) { return false; },
    });
}

auto Sema::Callee::decl() const -> Ptr<Decl> {
    return visit(utils::Overloaded{
        [](auto* decl) { return Ptr<Decl>(decl); },
        [](ProcType* ty) { return Ptr<Decl>(); },
        [](RecordType* ty) -> Ptr<Decl> {
            auto s = llvm::dyn_cast<StructType>(ty);
            return s ? s->decl() : nullptr;
        },
    });
}

auto Sema::Callee::name() const -> DeclNameLoc {
    if (auto d = decl().get_or_null()) return DeclNameLoc{d->name, d->location()};
    return {};
}

auto Sema::Callee::param_count() const -> u32 {
    return visit_type(utils::Overloaded{
        [](ProcType* ty) { return ty->param_count(); },
        [](ProcTemplateDecl* decl) { return u32(decl->pattern->params().size()); },
        [](RecordType* ty) { return u32(ty->layout().fields().size()); },
    });
}

auto Sema::Callee::param_loc(u32 index) const -> SLoc {
    return visit(utils::Overloaded{
        [](ProcType* ty) { return SLoc(); },
        [&](ProcDecl* decl) { return decl->params()[index]->location(); },
        [&](ProcTemplateDecl* decl) {
            return decl->pattern->type->param_types()[index].type->loc;
        },
        [&](RecordType* ty) { return ty->layout().fields()[index]->location(); },
    });
}

auto Sema::Callee::param_name(u32 index) const -> DeclName {
    return visit(utils::Overloaded{
        [](ProcType* ty) { return DeclName(); },
        [&](ProcDecl* decl) { return decl->params()[index]->name; },
        [&](ProcTemplateDecl* decl) { return decl->pattern->params()[index]->name; },
        [&](RecordType* ty) { return ty->layout().fields()[index]->name; },
    });
}

bool Sema::Candidate::has_valid_proc_type() const {
    return status.is<Viable, ParamInitFailed>();
}

auto Sema::Candidate::proc_type() const -> ProcType* {
    Assert(has_valid_proc_type(), "proc_type() cannot be used if template substitution failed");
    auto d = callee.dyn_cast<ProcDecl>();
    if (d) return d->proc_type();
    return subst.success()->type;
}

auto Sema::Candidate::type_for_diagnostic() const -> SmallUnrenderedString {
    auto d = callee.dyn_cast<ProcDecl>();
    if (d) return d->proc_type()->print();
    if (subst.success()) return subst.success()->type->print();
    return SmallUnrenderedString("(template)");
}

Sema::CandidateArgumentLists::CandidateArgumentLists(
    TupleExpr* args
) : unordered{args} {
    if (not args->is_named()) argument_lists.emplace_back(args->values());
}

/// Add a list; only valid if we have named arguments.
auto Sema::CandidateArgumentLists::add(List args) -> Id {
    Assert(unordered->is_named());
    argument_lists.push_back(std::move(args));
    return Id(argument_lists.size() - 1);
}

/// Get a list by id.
auto Sema::CandidateArgumentLists::operator[](Id id) const -> const List& {
    if (not unordered->is_named()) return argument_lists.front();
    return argument_lists[+id];
}

u32 Sema::ConversionSequence::badness() {
    u32 badness = 0;
    for (auto& conv : conversions) {
        switch (conv.kind) {
            using K = Conversion::Kind;

            // These don’t perform type conversion.
            case K::DefaultInit:
            case K::ExpandTuple:
            case K::LValueToRValue:
            case K::MaterialiseTemporary:
            case K::ReorderTuple:
            case K::SelectOverload:
            case K::StripParens:
                break;

            // These are actual type conversions.
            case K::ArrayDecay:
            case K::IntegralCast:
            case K::MaterialisePoison:
            case K::MutableToImmutable:
            case K::NilToOptional:
            case K::OptionalUnwrap:
            case K::OptionalWrap:
            case K::PointerDeref:
            case K::RangeCast:
            case K::SliceFromArray:
            case K::StrLitToCStr:
            case K::TupleToFirstElement:
                badness++;
                break;

            // These contain other conversions.
            case K::ArrayBroadcast:
                badness += conv.data.get<Conversion::ArrayBroadcastData>().seq->badness();
                break;

            case K::ArrayInit: {
                auto& data = conv.data.get<Conversion::ArrayInitData>();
                for (auto& seq : data.elem_convs) badness += seq.badness();
            } break;

            case K::RecordInit: {
                auto& data = conv.data.get<Conversion::RecordInitData>();
                for (auto& seq : data.field_convs) badness += seq.badness();
            } break;

            case K::SliceFromPtrAndSize: {
                auto& data = conv.data.get<Conversion::SliceFromPtrAndSizeData>();
                badness += data.ptr->badness();
                badness += data.size->badness();
            } break;
        }
    }
    return badness;
}

void Sema::NoteParameter(Decl* proc, u32 i) {
    SLoc loc;
    String name;

    if (auto d = dyn_cast<ProcDecl>(proc)) {
        if (d->is_imported()) return; // FIXME: Report this location somehow.

        // FIXME: Parameters are only created when we translate the body, but
        // we need their source locations earlier than that if we are diagnosing
        // a call to a procedure before translating its body. Either build the
        // parameters earlier or store the locations elsewhere; for now, we just
        // don’t emit this diagnostic if the parameters haven’t been created yet.
        if (i >= d->params().size()) return;

        // Get the parameter name and location.
        auto p = d->params()[i];
        loc = p->location();
        name = p->name.str();
    } else if (auto d = dyn_cast<ProcTemplateDecl>(proc)) {
        auto p = d->pattern->params()[i];
        loc = p->loc;
        name = p->name.str();
    } else {
        return;
    }

    if (name.empty()) Note(loc, "In argument to parameter declared here");
    else Note(loc, "In argument to parameter '{}'", name);
}

/// Resolve named arguments, i.e. figure out what positional parameters
/// they map to.
auto Sema::ResolveNamedArguments(
    TupleExpr* args,
    Callee callee
) -> std::expected<SmallVector<Expr*>, Candidate::Status> {
    // There is nothing to check if we have no parameters. This is called
    // after ArgumentCountMatchesParameters(), so we a parameter count
    // mismatch would have already been diagnosed.
    if (args->num_values() == 0) return SmallVector<Expr*>{};
    Assert(args->is_named(), "Should not have called this if there are no named args");
    const auto num_non_variadic_args = callee.num_non_variadic_params();

    // Add named and positional parameters.
    u32 next_index = 0;
    SmallVector<Expr*> ordered_params(args->num_values());
    for (auto [arg_index, arg] : enumerate(args->elems())) {
        u32 param_index;

        // If the argument has a name, use the parameter index of the
        // parameter with that name.
        auto [expr, name] = arg;
        if (name.name.valid()) {
            auto idx = callee.index_of_named_param(name.name.str());
            if (not idx) return std::unexpected(Candidate::NamedParamNotFound{
                name.name.str(),
                u32(arg_index),
            });

            param_index = *idx;

            // The variadic parameter cannot be named this way.
            if (param_index >= num_non_variadic_args) return std::unexpected(
                Candidate::NamedArgReferencesVariadicParam{u32(arg_index)}
            );
        }

        // Otherwise, use the next unused positional index. This loop should
        // never go out of bounds (if it does, we have more arguments than
        // parameters, which we should have diagnosed before ever getting here).
        else {
            while (ordered_params[next_index] != nullptr) next_index++;
            param_index = next_index++;
        }

        // If the slot for that parameter is already mapped, then we’re trying to
        // assign two parameters to the same slot.
        if (ordered_params[param_index] != nullptr) return std::unexpected(Candidate::ParamSpecifiedTwice{
            .param_index = param_index,
            .arg_index_1 = utils::index_of<u32>(args->values(), ordered_params[param_index]).value(),
            .arg_index_2 = u32(arg_index),
        });

        // Parameter has no argument yet; assign it.
        ordered_params[param_index] = expr;
    }

    // If we get here, we have processed (at least) as many arguments as there are
    // non-variadic parameters. Since we don’t allow binding two arguments to the
    // same parameter, nor binding an argument by name to the variadic parameter,
    // and we bail on any error, then by the pigeonhole principle, we should have
    // an argument bound to each parameter.
    DebugAssert(all_of(
        ArrayRef(ordered_params).take_front(num_non_variadic_args),
        [](Expr* p) { return p != nullptr; }
    ));

    return ordered_params;
}

void Sema::ConvertArgumentsForCall(
    ArrayRef<Expr*> args,
    Callee c,
    llvm::function_ref<void(u32 param_index, ArrayRef<Expr*> args, SLoc loc)> ConvertArg,
    SLoc call_loc
) {
    auto num_required_args = c.num_non_variadic_params();

    // Convert the argument to each non-variadic parameter.
    for (auto [i, a] : enumerate(args.take_front(num_required_args)))
        ConvertArg(u32(i), a, a->location());

    // Convert any remaining arguments to the variadic parameter.
    if (c.is_variadic()) ConvertArg(
        u32(c.param_count() - 1),
        args.drop_front(num_required_args),
        not args.empty() ? args.front()->location() : call_loc
    );
}

bool Sema::CheckIntents(ProcType* ty, MutableArrayRef<Expr*> args) {
    bool ok = true;
    for (auto [p, a] : zip(ty->params(), args)) {
        if (
            p.type->pass_by_reference(tu->target(), p.intent) and
            (not isa<CastExpr>(a) or cast<CastExpr>(a)->kind != CastExpr::MaterialisePoisonValue) and
            not a->is_mutable_lvalue()
        ) {
            // We want to make sure that 'inout' and 'out' parameters always use the original
            // object since writes to it must update that object; for other intents, we can
            // just materialise it.
            //
            // The exception is that we cannot move immutable lvalues.
            if (
                p.intent != Intent::Inout and
                p.intent != Intent::Out and
                (a->is_rvalue() or p.intent != Intent::Move)
            ) {
                a = MaterialiseTemporary(a);
                continue;
            }

            // If this is itself a parameter, issue a better error.
            ok = false;
            if (auto dre = dyn_cast_if_present<LocalRefExpr>(a); dre and isa<ParamDecl>(dre->decl)) {
                auto a_param = cast<ParamDecl>(dre->decl);
                if (p.intent == Intent::Move) {
                    Error(
                        a->location(),
                        "Cannot move %1({}%) parameter",
                        a_param->intent()
                    );
                } else {
                    Error(
                        a->location(),
                        "Cannot pass parameter of intent %1({}%) to a parameter with intent %1({}%)",
                        a_param->intent(),
                        p.intent
                    );
                }

                Note(a_param->location(), "Parameter declared here");
            }

            // Better error for moving an immovable value.
            else if (a->is_immutable_lvalue()) {
                if (p.intent == Intent::Move) Error(a->location(), "Cannot move immutable value");
                else Error(a->location(), "Cannot pass immutable value to an %1({}%) parameter.", p.intent);
            }

            // Generic error; this could probably be improved a bit.
            else {
                Error(a->location(), "Cannot bind this expression to an %1({}%) parameter.", p.intent);
                Remark("Try storing this in a variable first.");
            }
        }
    }
    return ok;
}

bool Sema::CheckOverloadedOperator(ProcDecl* d, bool builtin_operator) {
    static constexpr usz AnyNumber = ~0zu;
    auto t = d->name.operator_name();

    // Check the arity.
    auto [min, max] = [&] -> std::pair<usz, usz> {
        switch (t) {
            default: Unreachable("Invalid overloaded operator '{}'", t);
            case Tk::LBrack:
            case Tk::LParen:
                return {AnyNumber, AnyNumber};

            case Tk::As:
            case Tk::AsBang:
            case Tk::Caret:
            case Tk::MinusMinus:
            case Tk::Not:
            case Tk::PlusPlus:
            case Tk::Tilde:
                return {1, 1};

            case Tk::Ampersand:
            case Tk::Minus:
            case Tk::Plus:
                return {1, 2};

            case Tk::And:
            case Tk::ColonPercent:
            case Tk::ColonSlash:
            case Tk::DotDot:
            case Tk::DotDotEq:
            case Tk::DotDotLess:
            case Tk::EqEq:
            case Tk::In:
            case Tk::MinusEq:
            case Tk::MinusTilde:
            case Tk::MinusTildeEq:
            case Tk::Neq:
            case Tk::Or:
            case Tk::Percent:
            case Tk::PercentEq:
            case Tk::PlusEq:
            case Tk::PlusTilde:
            case Tk::PlusTildeEq:
            case Tk::SGe:
            case Tk::SGt:
            case Tk::SLe:
            case Tk::SLt:
            case Tk::ShiftLeft:
            case Tk::ShiftLeftEq:
            case Tk::ShiftLeftLogical:
            case Tk::ShiftLeftLogicalEq:
            case Tk::ShiftRight:
            case Tk::ShiftRightEq:
            case Tk::ShiftRightLogical:
            case Tk::ShiftRightLogicalEq:
            case Tk::Slash:
            case Tk::SlashEq:
            case Tk::Star:
            case Tk::StarEq:
            case Tk::StarStar:
            case Tk::StarStarEq:
            case Tk::StarTilde:
            case Tk::StarTildeEq:
            case Tk::UGe:
            case Tk::UGt:
            case Tk::ULe:
            case Tk::ULt:
            case Tk::VBar:
            case Tk::Xor:
                return {2, 2};
        }
    }();

    if (min == max and min != AnyNumber and d->param_count() != min) return Error(
        d->location(),
        "Operator '{}' requires exactly {} parameter{}",
        t,
        min,
        min == 1 ? "" : "s"
    );

    if (min != AnyNumber and d->param_count() < min) return Error(
        d->location(),
        "Operator '{}' requires at least {} parameter{}",
        t,
        min,
        min == 1 ? "" : "s"
    );

    if (max != AnyNumber) {
        if (d->param_count() > max) return Error(
            d->location(),
            "Operator '{}' takes at most {} parameter{}",
            t,
            max,
            max == 1 ? "" : "s"
        );

        if (d->proc_type()->has_c_varargs())
            return Error(d->location(), "Operator '{}' cannot use C varargs", t);
        if (d->proc_type()->is_variadic())
            return Error(d->location(), "Operator '{}' cannot be variadic", t);
    }


    // Disallow overriding builtin operators or defining overloads that take
    // only builtin types.
    if (
        not builtin_operator and
        not IsUserDefinedOverloadedOperator(t, d->param_types_no_intent() | rgs::to<SmallVector<Type, 10>>())
    ) {
        Error(d->location(), "At least one argument of overloaded operator must be a struct type");
        Remark("Current arguments: {}", utils::join(d->param_types_no_intent()));
        return false;
    }

    // The return type of 'as' and 'as!' must not be deduced.
    if ((t == Tk::As or t == Tk::AsBang) and d->return_type() == Type::DeducedTy)
        return Error(d->location(), "Return type of operator '{}' cannot be deduced", t);

    return true;
}

bool Sema::IsUserDefinedOverloadedOperator(Tk tk, ArrayRef<Type> argument_types) {
    if (tk == Tk::Assign or tk == Tk::Swap) return false;
    auto CanOverload = [](Type t) { return isa<StructType>(t); };
    return any_of(argument_types, CanOverload);
}

auto Sema::PerformOverloadResolution(
    OverloadSetExpr* overload_set,
    TupleExpr* call_args,
    bool is_associated_call,
    SLoc call_loc
) -> std::pair<ProcDecl*, SmallVector<Expr*>> {
    // Since this may also entail template substitution etc. we always
    // treat calls as requiring overload resolution even if there is
    // only a single ‘overload’.
    //
    // FIXME: This doesn’t work for indirect calls.
    //
    // Source does have a restricted form of SFINAE: deduction failure
    // and deduction failure only is not an error. Random errors during
    // deduction are hard errors.
    SmallVector<Candidate, 4> candidates;

    // Are we resolving a call to a builtin operator?
    auto types = llvm::to_vector(call_args->values() | vws::transform([](Expr* e) { return e->type; }));
    bool resolving_builtin_operator = overload_set->name().is_operator_name() and
                                      not IsUserDefinedOverloadedOperator(overload_set->name().operator_name(), types);

    // Unnamed parameters; these are precomputed here to avoid storing them
    // separately for every overload. Only populated if we have no named arguments.
    CandidateArgumentLists arg_lists{call_args};

    // Add a candidate to the overload set.
    auto AddCandidate = [&](Decl* proc) -> bool {
        if (not proc->valid())
            return false;

        // If we’re resolving a builtin operator, only consider templates that
        // have been annotated with the appropriate attribute (and vice versa).
        //
        // Do this here since we should never fail to resolve a builtin operator;
        // and conversely, they should never be used for user types, so there is
        // little point in always including these as non-viable candidates.
        auto templ = dyn_cast<ProcTemplateDecl>(proc);
        bool is_builtin_operator_template = templ and templ->is_builtin_operator_template();
        if (is_builtin_operator_template != resolving_builtin_operator)
            return true;

        // Add the candidate.
        auto& c = candidates.emplace_back(proc);

        // Argument count mismatch is not allowed, unless the
        // function is variadic. For variadic templates, we allow
        // the variadic parameter to be empty.
        if (not c.callee.argument_count_matches_parameters(u32(call_args->num_values()))) {
            c.status = Candidate::ArgumentCountMismatch{};
            return true;
        }

        // Check named parameters.
        if (call_args->is_named()) {
            auto resolved = ResolveNamedArguments(call_args, c.callee);
            if (not resolved) {
                c.status = std::move(resolved.error());
                return true;
            }

            c.status.get<Candidate::Viable>().arg_list = arg_lists.add(std::move(*resolved));
        }

        // Candidate is a regular procedure.
        if (not templ) return true;

        // Candidate is a template.
        SmallVector<TypeLoc, 6> types;
        for (const auto& arg : arg_lists[c.arg_list()]) types.emplace_back(arg->type, arg->location());
        c.subst = SubstituteTemplate(templ, types);

        // If there was a hard error, abort overload resolution entirely.
        if (c.subst.data.is<SubstitutionResult::Error>()) return false;

        // Otherwise, we can still try and continue with overload resolution.
        if (not c.subst.success()) c.status = Candidate::DeductionError{};
        return true;
    };

    // Collect all candidates.
    if (auto proc = dyn_cast<ProcRefExpr>(overload_set)) {
        if (not AddCandidate(proc->decl)) return {};
    } else {
        auto os = cast<OverloadSetExpr>(overload_set);
        if (not rgs::all_of(os->overloads(), AddCandidate)) return {};
    }

    // Check if a single candidate is viable. Returns false if there
    // is a fatal error that prevents overload resolution entirely.
    auto CheckCandidate = [&](Candidate& c) -> bool {
        auto ty = c.proc_type();
        auto params = ty->params();

        // Convert an argument to the corresponding parameter type.
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, SLoc loc) {
            // Candidate may have become invalid in the meantime.
            auto st = c.status.get_if<Candidate::Viable>();
            if (not st) return false;

            // Check the next parameter.
            auto& p = params[param_index];
            auto seq_or_err = BuildConversionSequence(
                p.type,
                args,
                loc,
                p.type->pass_by_reference(tu->target(), p.intent),
                is_associated_call and param_index == 0
            );

            // If this failed, stop checking this candidate.
            if (not seq_or_err.has_value()) {
                c.status = Candidate::ParamInitFailed{std::move(seq_or_err.error()), param_index};
                return false;
            }

            // Otherwise, store the sequence for later and keep going.
            st->conversions.push_back(std::move(seq_or_err.value()));
            st->badness += st->conversions.back().badness();
            return true;
        };

        ConvertArgumentsForCall(arg_lists[c.arg_list()], c.callee, ConvertArg, call_loc);
        return true;
    };

    // Check if we have no candidates at all; this can happen if the only
    // candidates in the overload set were builtin operators.
    if (candidates.empty()) {
        Assert(
            overload_set->name().is_operator_name() and call_args->num_values() == 2,
            "Only a few binary operators are currently handled this way"
        );

        Error(
            call_loc,
            "Invalid operation: '%1({}%)' between '{}' and '{}'",
            Spelling(overload_set->name().operator_name()),
            call_args->nth(0).expr->type,
            call_args->nth(1).expr->type
        );
        return {};
    }

    // Check each candidate, computing viability etc.
    for (auto& c : candidates) {
        if (not c.viable()) continue;
        if (not CheckCandidate(c)) return {};
    }

    // Find the best viable unique overload, if there is one.
    Ptr<Candidate> best;
    bool ambiguous = false;
    auto viable = candidates | vws::filter(&Candidate::viable);
    for (auto& c : viable) {
        // First viable candidate.
        if (not best) best = &c;

        // We already have a candidate. If the badness of this
        // one is better, then it becomes the new best candidate.
        else if (c.badness() < best.get()->badness()) {
            best = &c;
            ambiguous = false;
        }

        // Otherwise, if its badness is the same, we have an
        // ambiguous candidate; else, ignore it entirely.
        else if (c.badness() == best.get()->badness())
            ambiguous = true;
    }

    // If overload resolution was ambiguous, then we don’t have
    // a best candidate.
    u32 badness = best.get_or_null() ? best.get()->badness() : 0;
    if (ambiguous) best = {};

    // We found a single best candidate!
    if (auto c = best.get_or_null()) {
        ProcDecl* final_callee;

        // Instantiate it now if it is a template.
        if (c->callee.is_template()) {
            auto inst = InstantiateTemplate(
                c->callee.cast<ProcTemplateDecl>(),
                *c->subst.data.get<TemplateSubstitution*>(),
                call_loc
            );

            if (not inst or not inst->is_valid) return {};
            final_callee = inst;
        } else {
            final_callee = c->callee.cast<ProcDecl>();
        }

        // Now is the time to apply the argument conversions.
        auto& args = arg_lists[c->arg_list()];
        SmallVector<Expr*> actual_args;
        actual_args.reserve(args.size());
        ArrayRef conversions(c->status.get<Candidate::Viable>().conversions);
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, SLoc loc) {
            actual_args.emplace_back(ApplyConversionSequence(
                args,
                conversions[param_index],
                loc
            ));
        };

        ConvertArgumentsForCall(args, final_callee, ConvertArg, call_loc);
        if (not CheckIntents(final_callee->proc_type(), actual_args)) return {};
        return {final_callee, std::move(actual_args)};
    }

    // Overload resolution failed. :(
    ReportOverloadResolutionFailure(candidates, call_args, call_loc, badness);
    return {};
}

void Sema::FormatTempSubstFailure(
    const SubstitutionResult& info,
    SmallString<256>& out,
    std::string_view indent
) {
    info.data.visit(utils::Overloaded{// clang-format off
        [](TemplateSubstitution*) { Unreachable("Invalid template even though substitution succeeded?"); },
        [](SubstitutionResult::Error) { Unreachable("Should have bailed out earlier on hard error"); },
        [&](SubstitutionResult::ConstraintNotSatisfied) {
            out += "'%1(where%)' clause evaluated to '%1(false%)'";
        },

        [&](SubstitutionResult::DeductionFailed f) {
            Format(
                out,
                "In param #{}: could not infer ${}",
                f.param_index + 1,
                f.param
            );
        },

        [&](const SubstitutionResult::DeductionAmbiguous& a) {
            Format(
                out,
                "Inference mismatch for template parameter %3(${}%):\n"
                "{}Argument #{}: Inferred as {}\n"
                "{}Argument #{}: Inferred as {}",
                a.param,
                indent,
                a.first + 1,
                a.first_type,
                indent,
                a.second + 1,
                a.second_type
            );
        }
    }); // clang-format on
}

void Sema::ReportSingleOverloadResolutionFailure(
    Callee callee,
    Candidate::Status status,
    std::optional<SubstitutionResult> subst,
    TupleExpr* args,
    SLoc call_loc
) { // clang-format off
    String entity_name = callee.is_record() ? "Type"_s : "Procedure"_s;
    String parameter_name = callee.is_record() ? "field"_s : "parameter"_s;
    String argument_name = callee.is_record() ? "initialiser"_s : "argument"_s;
    String callee_name = callee.name().name.str();
    auto entity_and_callee_name = entity_name + " '" + callee_name + "'";
    bool has_name = callee.name().name.valid();
    bool is_record = callee.is_record();
    bool should_note_callee = not is_record;
    status.visit(utils::Overloaded{
        [](const Candidate::Viable&) { Unreachable(); },
        [&](Candidate::ArgumentCountMismatch) {
            Error(
                call_loc,
                "{} {} {}{}, got {}",
                has_name ? entity_and_callee_name + " expects" : "Expected",
                callee.param_count(),
                argument_name,
                callee.param_count() == 1 ? "" : "s",
                args->num_values()
            );
        },

        [&](Candidate::DeductionError) {
            Assert(subst.has_value());
            SmallString<256> extra;
            FormatTempSubstFailure(*subst, extra, "  ");
            if (subst->data.is<SubstitutionResult::ConstraintNotSatisfied>()) {
                Error(
                    call_loc,
                    "Constraints{} not satisfied",
                    has_name ? " of '"_s + callee_name + "' " : ""
                );
                if (auto d = callee.dyn_cast<ProcTemplateDecl>()) {
                    Note(d->pattern->where.get()->loc, "{}", extra);
                    should_note_callee = false; // We already point to the where clause.
                }
            } else {
                Error(
                    call_loc,
                    "Template argument substitution failed{}",
                    has_name ? "in call to "_s + callee_name : ""
                );
                Remark("\r{}", extra);
            }
        },

        [&](Candidate::NamedParamNotFound p) {
            Error(
                args->nth(p.arg_index).name.loc,
                "{} has no {} named '{}'",
                has_name ? entity_and_callee_name : entity_name.sv(),
                parameter_name,
                p.param_name
            );
        },

        [&](Candidate::NamedArgReferencesVariadicParam p) {
            const auto& [name, loc] = args->nth(p.arg_index).name;
            Error(loc, "Named argument names the variadic parameter");
        },

        [&](Candidate::ParamInitFailed& p) {
            for (auto& d : p.diags) diags().report(std::move(d));
            if (auto d = callee.decl().get_or_null()) NoteParameter(d, p.param_index);
        },

        [&](Candidate::ParamSpecifiedTwice& p) {
            Error(
                args->nth(p.arg_index_2).expr->location(),
                "An {} was already {} for {} '{}'",
                argument_name,
                is_record ? "provided" : "passed",
                parameter_name,
                callee.param_name(p.param_index)
            );

            Note(
                args->nth(p.arg_index_1).expr->location(),
                "Previously {} here",
                is_record ? "initialised" : "passed"
            );
        },
    });

    if (not should_note_callee) return;
    if (auto [_, loc] = callee.name()) Note(loc, "Callee declared here");
} // clang-format on

void Sema::ReportOverloadResolutionFailure(
    MutableArrayRef<Candidate> candidates,
    TupleExpr* args,
    SLoc call_loc,
    u32 final_badness
) {
    // If there is only one overload, print the failure reason for
    // it and leave it at that.
    if (candidates.size() == 1) {
        auto& c = candidates.front();
        ReportSingleOverloadResolutionFailure(
            c.callee,
            std::move(c.status),
            std::move(c.subst),
            args,
            call_loc
        );
        return;
    }

    // Otherwise, we need to print all overloads, and why they failed.
    SmallString<256> message;
    message = "%b(Candidates:%)\n"sv;

    // Compute the width of the number field.
    u32 width = u32(std::to_string(candidates.size()).size());

    // First, print all overloads.
    for (auto [i, c] : enumerate(candidates)) {
        Format(message, "  %b({}.%) \v{}", i + 1, c.type_for_diagnostic());
        Format(message, "\f%b(at%) {}", c.callee.name().loc.format(ctx, true));
        message += "\n";
    }

    Format(message, "\n\r%b(Failure Reason:%)");

    // Collect ambiguous candidates.
    SmallVector<u32> ambiguous_indices;
    for (auto [i, c] : enumerate(candidates))
        if (auto v = c.status.get_if<Candidate::Viable>())
            if (v->badness == final_badness)
                ambiguous_indices.push_back(u32(i + 1));

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        Format(message, "\n  %b({:>{}}.%) ", i + 1, width);
        auto V = utils::Overloaded{
            // clang-format off
            [&] (const Candidate::Viable& v) {
                // Don’t print that a candidate conflicts with itself...
                auto ambiguous = ambiguous_indices;
                erase(ambiguous, u32(i + 1));

                // If the badness is equal to the final badness, then
                // this candidate was ambiguous. Otherwise, another
                // candidate was simply better.
                if (v.badness == final_badness) Format(message, "Ambiguous (with {})", utils::join(ambiguous, ", ", "#{}"));
                else Format(message, "Another candidate was a better match", v.badness);
            },

            [&](Candidate::ArgumentCountMismatch) {
                auto params = c.callee.param_count();
                Format(
                    message,
                    "Expected {} arg{}, got {}",
                    params,
                    params == 1 ? "" : "s",
                    args->num_values()
                );
            },

            [&](Candidate::NamedParamNotFound p) {
                Format(
                    message,
                    "Parameter with name '{}' not found",
                    args->nth(p.arg_index).name.name
                );
            },

            [&](Candidate::NamedArgReferencesVariadicParam p) {
                Format(
                    message,
                    "Named argument names the variadic parameter",
                    args->nth(p.arg_index).name.name
                );
            },

            [&](Candidate::ParamInitFailed& i) {
                Format(message, "In argument to parameter #{}:\n", i.param_index);
                message += utils::Indent(Diagnostic::Render(ctx, i.diags, diags().cols() - 5, false), 2);
            },

            [&](Candidate::ParamSpecifiedTwice& p) {
                Format(
                    message,
                    "Parameter #{} passed twice (via args #{} and #{})",
                    p.param_index + 1,
                    p.arg_index_1 + 1,
                    p.arg_index_2 + 1
                );
            },

            [&](Candidate::DeductionError) {
                message += c.subst.data.is<SubstitutionResult::ConstraintNotSatisfied>()
                    ? "Constraints not satisfied"sv
                    : "Template argument substitution failed"sv;
                FormatTempSubstFailure(c.subst, message, "        ");
            },
        }; // clang-format on
        c.status.visit(V);
    }

    // Remove a trailing newline because rendering nested diagnostics
    // sometimes adds one too many.
    if (message.back() == '\n') message.pop_back();
    ctx.diags().report(Diagnostic{
        Diagnostic::Level::Error,
        call_loc,
        std::format("Overload resolution failed in call to\f'%2({}%)'", candidates.front().callee.name().name),
        message.str().str(),
    });
}