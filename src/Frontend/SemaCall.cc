#include <srcc/Frontend/Sema.hh>

using namespace srcc;
using namespace srcc::sema;

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
            return utils::index_of<u32>(ty->scope()->fields(), name, &FieldDecl::name);
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
            auto s = srcc::dyn_cast<StructType>(ty);
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
        [](RecordType* ty) { return u32(ty->layout().field_offsets().size()); },
    });
}

auto Sema::Callee::param_loc(u32 index) const -> SLoc {
    return visit(utils::Overloaded{
        [](ProcType* ty) { return SLoc(); },
        [&](ProcDecl* decl) { return decl->params()[index]->location(); },
        [&](ProcTemplateDecl* decl) {
            return decl->pattern->type->param_types()[index].type->loc;
        },
        [&](RecordType* ty) { return ty->scope()->field(index)->location(); },
    });
}

auto Sema::Callee::param_name(u32 index) const -> DeclName {
    return visit(utils::Overloaded{
        [](ProcType* ty) { return DeclName(); },
        [&](ProcDecl* decl) { return decl->params()[index]->name; },
        [&](ProcTemplateDecl* decl) { return decl->pattern->params()[index]->name; },
        [&](RecordType* ty) { return ty->scope()->field(index)->name; },
    });
}

/// Overload Ranking Model
///
/// After checking viability, we compare argument conversions one by one. A
/// candidate is better if no argument conversion is worse and at least one is
/// better.
///
/// Primitive conversions, from best to worst:
///
///   Exact: identity, lvalue loads, temporary materialisation, parentheses,
///       tuple expansion/reordering, default initialisation, function overload
///       selection, and 'noreturn' conversion.
///
///   Qualification: mutable pointer/slice to immutable pointer/slice.
///
///   View: pointer dereference, optional unwrap, array pointer decay, array to
///       slice, string literal to C string, tuple reshaping/singleton unwrapping.
///
///   Value: integer/range conversion, optional wrapping, 'nil' to optional.
///
///   Initialiser: calling a struct's initialiser, array/record construction,
///       array broadcasting, slice from pointer+size.
///
///   Varargs: passing an argument through untyped C varargs.
///
/// Steps in the same group are indistinguishable; repeating them adds no penalty.
/// In particular, we don’t favour closer integer widths, fewer optional wrappers,
/// or fewer dereferences of the same rank. Initialiser calls are opaque: do not
/// favour a destination type because its selected initialiser happens to take
/// more convenient parameter types.
///
/// A sequence has the rank of its worst conversion, including conversions nested
/// in aggregate construction. For example, constructing either int[3] or i128[3]
/// from 1 has Initialiser rank, even though the latter also converts its element.
/// Those sequences compare equal rather than favouring one destination type.
///
/// Each candidate stores a vector of sequence ranks indexed by source argument.
/// Named arguments are bound to parameters before ranking, then their ranks are
/// stored at the original argument indices so call-site ordering is irrelevant.
/// No expression map or construction depth is needed.
///
/// Variadic array/slice packing is bookkeeping. A pack's worst element conversion
/// determines its sequence rank for all its arguments, which are compared against
/// the corresponding arguments of fixed-arity candidates. Passing an existing
/// array as a slice still requires a view conversion. C varargs arguments receive
/// the varargs rank despite having no declared conversion sequence.
///
/// Only when all conversions compare equal, apply these tie-breaks in order:
///
///   1. Prefer a signature that requires no type deduction ('$T' or 'var').
///      Compile-time parameters do not affect ranking.
///   2. Prefer a procedure with neither variadic parameters nor C varargs.
///
auto Sema::Candidate::compare(const Candidate& other) const -> std::partial_ordering {
    Assert(viable() and other.viable());

    // They are not equivalent; returnn the ordering.
    auto order = status.get<Viable>().ranking.compare(other.status.get<Viable>().ranking);
    if (order != OverloadRanking::Equivalent) return order;

    // Apply tie-breaking rules.
    auto RequiresTypeDeduction = [](const Candidate& candidate) {
        if (auto pattern = candidate.callee.dyn_cast<ProcTemplateDecl>())
            return pattern->requires_type_deduction;
        auto proc = candidate.callee.cast<ProcDecl>();
        return proc->is_instantiation() and proc->pattern()->requires_type_deduction;
    };

    // Type deduction is worse than no deduction.
    bool deduced = RequiresTypeDeduction(*this);
    bool other_deduced = RequiresTypeDeduction(other);
    if (deduced != other_deduced) return deduced
        ? OverloadRanking::Worse
        : OverloadRanking::Better;

    // Variadic functions are worse than non-variadic functions.
    bool variadic = callee.is_variadic() or callee.has_c_varargs();
    bool other_variadic = other.callee.is_variadic() or other.callee.has_c_varargs();
    if (variadic != other_variadic) return variadic
        ? OverloadRanking::Worse
        : OverloadRanking::Better;

    // They’re equal.
    return OverloadRanking::Equivalent;
}

bool Sema::Candidate::has_valid_proc_type() const {
    return status.is<Viable, ParamInitFailed>() and
           (not callee.is_template() or subst.success());
}

auto Sema::Candidate::proc_type() const -> ProcType* {
    Assert(has_valid_proc_type(), "proc_type() cannot be used if template substitution failed");
    auto d = callee.dyn_cast<ProcDecl>();
    if (d) return d->proc_type();
    return subst.success()->signature->runtime_type;
}

auto Sema::Candidate::source_param_index(u32 index) const -> u32 {
    if (callee.is_template()) return subst.success()->signature->source_index(index);
    return index;
}

auto Sema::Candidate::type_for_diagnostic() const -> SmallUnrenderedString {
    auto d = callee.dyn_cast<ProcDecl>();
    if (d) return d->proc_type()->print();
    if (subst.success()) return subst.success()->signature->print(callee.cast<ProcTemplateDecl>());
    return SmallUnrenderedString("(template)");
}

Sema::CandidateArgumentLists::CandidateArgumentLists(TupleExpr* args) : unordered{args} {
    if (args->is_named()) return;
    List list;
    list.values = llvm::to_vector(args->values());
    for (u32 index = 0; index < args->num_values(); ++index) list.source_indices.push_back(index);
    add(std::move(list));
}

auto Sema::CandidateArgumentLists::add(SmallVector<Expr*> args) -> Id {
    List list;
    list.values = std::move(args);
    for (auto value : list.values)
        list.source_indices.push_back(utils::index_of<u32>(unordered->values(), value).value());
    return add(std::move(list));
}

auto Sema::CandidateArgumentLists::add(List args) -> Id {
    argument_lists.push_back(std::move(args));
    return Id(argument_lists.size() - 1);
}

auto Sema::CandidateArgumentLists::operator[](Id id) const -> const List& {
    return argument_lists[+id];
}

void Sema::ConversionRanking::reset(u32 num_arguments) {
    ranks.assign(num_arguments, Rank::Exact);
}

void Sema::ConversionRanking::set(u32 argument, Rank rank) {
    ranks[argument] = rank;
}

auto Sema::ConversionRanking::compare(const ConversionRanking& other) const -> std::partial_ordering {
    Assert(ranks.size() == other.ranks.size());
    auto result = OverloadRanking::Equivalent;
    for (auto [lhs, rhs] : zip_equal(ranks, other.ranks)) {
        auto order = lhs <=> rhs;
        if (order == 0) continue;
        if (result != OverloadRanking::Equivalent and result != order)
            return OverloadRanking::Unordered;
        result = order;
    }
    return result;
}

auto Sema::ConversionSequence::rank(bool variadic) const -> ConversionRanking::Rank {
    using Rank = ConversionRanking::Rank;
    Rank worst = Rank::Exact;
    auto Merge = [&](Rank rank) { worst = std::max(worst, rank); };

    // Typed packs use an array/slice, and heterogeneous packs use a tuple.
    // Their construction is bookkeeping; only element conversions contribute.
    bool packing = variadic and rgs::any_of(conversions, [](const Conversion& conv) {
        return conv.kind == Conversion::Kind::ArrayInit or
               conv.kind == Conversion::Kind::RecordInit;
    });

    for (auto& conv : conversions) {
        switch (conv.kind) {
            using K = Conversion::Kind;
            case K::DefaultInit:
            case K::ExpandTuple:
            case K::LValueToRValue:
            case K::MaterialisePoison:
            case K::MaterialiseTemporary:
            case K::ReorderTuple:
            case K::SelectOverload:
            case K::StripParens:
                break;

            case K::MutableToImmutable:
                Merge(Rank::Qualification);
                break;

            case K::ArrayDecay:
            case K::OptionalUnwrap:
            case K::PointerDeref:
            case K::StrLitToCStr:
            case K::TupleNopCast:
            case K::TupleToFirstElement:
                Merge(Rank::View);
                break;

            case K::SliceFromArray:
                if (not packing) Merge(Rank::View);
                break;

            case K::IntegralCast:
            case K::NilToOptional:
            case K::OptionalWrap:
            case K::RangeCast:
                Merge(Rank::Value);
                break;

            // An initialiser is an opaque call; its argument conversions do not
            // make one destination type preferable to another.
            case K::InitialiserCall:
                if (variadic) {
                    // A heterogeneous pack's canonical tuple can have a generated
                    // initialiser. This is still packing, not an element conversion.
                    for (auto& seq : conv.data.get<Conversion::CallData>().conversions)
                        Merge(seq.rank());
                } else {
                    Merge(Rank::Initialiser);
                }
                break;

            case K::ArrayBroadcast:
                Merge(Rank::Initialiser);
                Merge(conv.data.get<Conversion::ArrayBroadcastData>().seq->rank());
                break;

            case K::ArrayInit:
                if (not packing) Merge(Rank::Initialiser);
                for (auto& seq : conv.data.get<Conversion::ArrayInitData>().elem_convs)
                    Merge(seq.rank());
                break;

            case K::RecordInit:
                if (not packing) Merge(Rank::Initialiser);
                for (auto& seq : conv.data.get<Conversion::RecordInitData>().field_convs)
                    Merge(seq.rank());
                break;

            case K::SliceFromPtrAndSize: {
                Merge(Rank::Initialiser);
                auto& data = conv.data.get<Conversion::SliceFromPtrAndSizeData>();
                Merge(data.ptr->rank());
                Merge(data.size->rank());
            } break;
        }
    }
    return worst;
}

void Sema::NoteParameter(Decl* proc, u32 i) {
    SLoc loc;
    String name;

    if (auto d = dyn_cast<ProcDecl>(proc)) {
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

    if (not loc.is_valid()) return;
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

bool Sema::CheckOverloadedOperator(
    ProcDecl* d,
    bool builtin_operator,
    Ptr<ProcTemplateSignature> signature
) {
    // Collect parameters, which might include compile-time ones.
    auto type = d->proc_type();
    auto count = signature ? signature->param_count() : type->param_count();
    SmallVector<Type> argument_types;
    for (u32 index = 0; index < count; ++index) {
        argument_types.push_back(
            signature
                ? signature->parameter(index).type
                : type->params()[index].type
        );
    }

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

    if (min == max and min != AnyNumber and count != min) return Error(
        d->location(),
        "Operator '{}' requires exactly {} parameter{}",
        t,
        min,
        min == 1 ? "" : "s"
    );

    if (min != AnyNumber and count < min) return Error(
        d->location(),
        "Operator '{}' requires at least {} parameter{}",
        t,
        min,
        min == 1 ? "" : "s"
    );

    if (max != AnyNumber) {
        if (count > max) return Error(
            d->location(),
            "Operator '{}' takes at most {} parameter{}",
            t,
            max,
            max == 1 ? "" : "s"
        );

        if (type->has_c_varargs())
            return Error(d->location(), "Operator '{}' cannot use C varargs", t);
        if (type->is_variadic())
            return Error(d->location(), "Operator '{}' cannot be variadic", t);
    }


    // Disallow overriding builtin operators or defining overloads that take
    // only builtin types.
    if (
        not builtin_operator and
        not IsUserDefinedOverloadedOperator(t, argument_types)
    ) {
        Error(d->location(), "At least one argument of overloaded operator must be a struct type");
        Remark("Current arguments: {}", utils::join(argument_types));
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
    SLoc call_loc,
    bool* hard_error
) -> std::optional<Conversion::CallData> {
    bool hard_error_local;
    if (hard_error == nullptr) hard_error = &hard_error_local;
    *hard_error = false;

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
        if (not proc->valid()) {
            *hard_error = true;
            return false;
        }

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

        // Initialise the conversion rank to Exact for each argument.
        auto& viable = c.status.get<Candidate::Viable>();
        viable.ranking.reset(call_args->num_values());

        // Candidate is a regular procedure.
        if (not templ) return true;

        // Handle a compile-time parameter.
        //
        // The rest of the overload resolution code below only deals with runtime
        // parameters; this includes computing the rank of an overload. Compile-time
        // parameters are ranked here ahead of time as part of substitution.
        auto& source_args = arg_lists[c.arg_list()];

        // Perform template substitution and prepare constant arguments.
        c.subst = RequestTemplateSubstitution(
            c,
            source_args,
            is_associated_call,
            *hard_error
        );

        // If there was a hard error, abort overload resolution entirely.
        if (*hard_error or c.subst.data.is<SubstitutionResult::Error>()) {
            *hard_error = true;
            return false;
        }

        // This candidate is not viable; mark it as invalid.
        if (auto failure = c.subst.data.get_if<SubstitutionResult::ArgumentNotConstant>()) {
            c.status = Candidate::ParamInitFailed{std::move(failure->diags), failure->param_index};
        } else if (not c.subst.success()) {
            c.status = Candidate::DeductionError{};
        }

        // This candidate is still viable; remove the compile-time parameters
        // from the signature if there are any.
        else if (
            auto constants = c.subst.success()->signature->compile_time_params();
            not constants.empty()
        ) {
            CandidateArgumentLists::List runtime_args;
            usz constant = 0;

            for (auto [index, arg] : enumerate(source_args.values)) {
                // This is a constant; skip it.
                if (constant < constants.size() and constants[constant].index == index) {
                    ++constant;
                    continue;
                }

                // This is a runtime parameter; compile-time parameters may be interspersed
                // arbitrarly with runtime ones, so we need to maintain a mapping between
                // the indices of the arguments as written in source and their corresponding
                // runtime parameters.
                runtime_args.values.push_back(arg);
                runtime_args.source_indices.push_back(source_args.source_indices[index]);
            }

            viable.arg_list = arg_lists.add(std::move(runtime_args));
        }

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
        auto& arguments = arg_lists[c.arg_list()];

        // Convert an argument to the corresponding parameter type.
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, SLoc loc) {
            // Candidate may have become invalid in the meantime.
            auto st = c.status.get_if<Candidate::Viable>();
            if (not st) return false;

            // Calling conventions and conversions require the parameter layout.
            auto& p = params[param_index];
            if (not CheckVariableType(p.type, loc)) {
                *hard_error = true;
                c.status = Candidate::ParamInitFailed{{}, c.source_param_index(param_index)};
                return false;
            }

            auto seq_or_err = BuildParameterConversionSequence(
                p,
                args,
                loc,
                is_associated_call and param_index == 0 and
                not arguments.source_indices.empty() and arguments.source_indices.front() == 0
            );

            // If this failed, stop checking this candidate.
            if (not seq_or_err.has_value()) {
                c.status = Candidate::ParamInitFailed{
                    std::move(seq_or_err.error()),
                    c.source_param_index(param_index),
                };
                return false;
            }

            // Otherwise, store the sequence for later and keep going.
            st->conversions.push_back(std::move(seq_or_err.value()));
            return true;
        };

        // Build the conversion sequence for each argument.
        ConvertArgumentsForCall(arguments.values, ty, ConvertArg, call_loc);

        // If this is no longer viable, stop.
        auto st = c.status.get_if<Candidate::Viable>();
        if (not st) return not *hard_error;

        // Compute the conversion rank for each parameter.
        SmallVector<ConversionRanking::Rank> parameter_ranks;
        for (auto [seq, param] : zip_equal(st->conversions, params))
            parameter_ranks.push_back(seq.rank(param.variadic));

        // Set ranks in *source order*, i.e. before named argument reordering;
        // this is because we care about how bad the conversion rank of any one
        // *argument* is, not whatever the rank for the nth parameter of a particular
        // candidate happens to be.
        //
        // This pass is for *runtime* arguments only; compile-time arguments have
        // already had their rank set during substitution.
        for (auto [i, source_index] : enumerate(arguments.source_indices)) {
            auto index = ty->is_variadic() ? std::min(i, params.size() - 1) : i;
            st->ranking.set(
                source_index,
                index < parameter_ranks.size()
                    ? parameter_ranks[index]
                    : ConversionRanking::Rank::Varargs
            );
        }

        return not *hard_error;
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

    // Do a linear scan to find a candidate that is not worse than any
    // other candidate.
    Ptr<Candidate> best;
    auto viable = candidates | vws::filter(&Candidate::viable);
    for (auto& c : viable)
        if (not best or c.compare(*best.get()) == OverloadRanking::Better)
            best = &c;

    // If this candidate is unordered or equivalent to any other candidate,
    // then we don’t actually have a best match.
    if (
        best and rgs::any_of(viable, [&](const Candidate& c) {
            return &c != best.get() and best->compare(c) != OverloadRanking::Better;
        })
    ) best = {};

    // We found a single best candidate!
    if (auto c = best.get_or_null()) {
        ProcDecl* final_callee;

        // Instantiate it now if it is a template.
        if (c->callee.is_template()) {
            auto inst = RequestTemplateInstantiation(
                c->callee.cast<ProcTemplateDecl>(),
                *c->subst.data.get<TemplateSubstitution*>(),
                call_loc
            );

            if (not inst or not inst->is_valid) {
                *hard_error = true;
                return {};
            }
            final_callee = inst.get();
        } else {
            final_callee = c->callee.cast<ProcDecl>();
        }

        Conversion::CallData call;
        call.proc = final_callee;
        call.args = arg_lists[c->arg_list()].values;
        auto& conversions = c->status.get<Candidate::Viable>().conversions;
        for (auto& conversion : conversions) call.conversions.push_back(std::move(conversion));
        return call;
    }

    ReportOverloadResolutionFailure(candidates, call_args, call_loc);
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
        [](const SubstitutionResult::ArgumentNotConstant&) { Unreachable("Reported as parameter initialisation failure"); },
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
    bool has_name = callee.name().name.valid();
    bool is_record = callee.is_record();
    bool should_note_callee = not is_record;
    auto entity_and_callee_name = std::format(
        "{} '%{}({}%)'",
        entity_name,
        is_record ? '6' : '2',
        callee_name
    );

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
            for (auto& d : p.diags) ReportDiag(std::move(d));
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
    SLoc call_loc
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

    // Compute what candidates, if any, were ambiguous.
    SmallVector<u32> ambiguous_indices;
    auto viable = candidates | vws::filter(&Candidate::viable);
    for (auto [i, c] : enumerate(candidates)) {
        if (
            c.viable() and rgs::none_of(viable, [&](const Candidate& other) {
                return other.compare(c) == OverloadRanking::Better;
            })
        ) ambiguous_indices.push_back(u32(i + 1));
    }

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        Format(message, "\n  %b({:>{}}.%) ", i + 1, width);
        auto V = utils::Overloaded{
            // clang-format off
            [&] (const Candidate::Viable&) {
                // Don’t print that a candidate conflicts with itself...
                auto ambiguous = ambiguous_indices;
                erase(ambiguous, u32(i + 1));

                if (rgs::contains(ambiguous_indices, u32(i + 1))) {
                    Format(message, "Ambiguous (with {})", utils::join(ambiguous, ", ", "#{}"));
                } else {
                    Format(message, "Another candidate was a better match");
                }
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
    ReportDiag(Diagnostic{
        Diagnostic::Level::Error,
        call_loc,
        std::format("Overload resolution failed in call to\f'%2({}%)'", candidates.front().callee.name().name),
        message.str().str(),
    });
}