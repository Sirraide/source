#include <srcc/Frontend/Sema.hh>

using namespace srcc;

// ============================================================================
//  Initialisation.
// ============================================================================
Sema::Conversion::~Conversion() = default;
void Sema::AddInitialiserToDecl(AnyVarDecl decl, Ptr<Expr> init) {
    // Deduce the type from the initialiser, if need be.
    if (decl.type() == Type::DeducedTy) {
        if (init) decl.type() = init.get()->type;
        else {
            Error(decl.decl()->location(), "Type inference requires an initialiser");
            decl.type() = Type::VoidTy;
            decl.set_invalid();
            return;
        }
    }

    // Now that the type has been deduced (if necessary), we can check
    // if we can even create a variable of this type.
    if (not CheckVariableType(decl.type(), decl.decl()->location())) {
        decl.set_invalid();
        return;
    }

    // Then, perform initialisation.
    init = BuildInitialiser(
        decl.type(),
        init ? init.get() : ArrayRef<Expr*>{},
        init ? init.get()->location() : decl.decl()->location()
    );

    // If this fails, make sure to mark the decl as invalid so that
    // e.g. an 'eval' doesn’t try to evaluate a decl with an invalid
    // initialiser.
    if (init.invalid()) decl.set_invalid();
    decl.set_init(init);
}

auto Sema::ApplySimpleConversion(Expr* e, const Conversion& conv, SLoc loc) -> Expr* {
    auto Cast = [&](CastExpr::CastKind kind, ValueCategory vc = Expr::RValue) {
        return new (*tu) CastExpr(
            conv.type(),
            kind,
            e,
            loc,
            true,
            vc
        );
    };

    switch (conv.kind) {
        using K = Conversion::Kind;
        default: Unreachable();

        case K::ArrayDecay: return Cast(CastExpr::Pointer);
        case K::IntegralCast: return Cast(CastExpr::Integral);
        case K::LValueToRValue: return LValueToRValue(e);
        case K::MaterialisePoison: return new (*tu) CastExpr(
            conv.type(),
            CastExpr::MaterialisePoisonValue,
            e,
            loc,
            true,
            conv.value_category()
        );

        case K::MaterialiseTemporary: return MaterialiseTemporary(e);
        case K::MutableToImmutable: return Cast(CastExpr::Nop, e->value_category);
        case K::NilToOptional: return Cast(CastExpr::NilToOptional);
        case K::OptionalWrap: return Cast(CastExpr::OptionalWrap);
        case K::OptionalUnwrap: return UnwrapOptional(e, loc);
        case K::PointerDeref: return BuildUnaryExpr(Tk::Caret, e, false, loc).get();
        case K::RangeCast: return Cast(CastExpr::Range);

        case K::SelectOverload: {
            auto proc = cast<OverloadSetExpr>(e)->overloads()[conv.data.get<u32>()];
            return CreateReference(proc, loc).get();
        }

        case K::SliceFromArray: return new (*tu) CastExpr(
            SliceType::Get(*tu, cast<ArrayType>(e->type)->elem(), false),
            CastExpr::SliceFromArray,
            MaterialiseTemporary(e),
            loc,
            true
        );

        case K::StripParens: {
            auto p = cast<ParenExpr>(e);
            return p->expr;
        }

        case K::StrLitToCStr: return BuildBuiltinMemberAccessExpr(
            BuiltinMemberAccessExpr::AccessKind::SliceData,
            e,
            loc
        ).get(); // Should never fail.

        case K::TupleToFirstElement: {
            if (auto te = dyn_cast<TupleExpr>(e)) return te->values().front();
            auto ty = cast<TupleType>(e->type);
            auto temp = MaterialiseTemporary(e);
            return new (*tu) MemberAccessExpr(temp, ty->layout().fields().front(), loc);
        }
    }
}

void Sema::ApplyConversion(SmallVectorImpl<Expr*>& exprs, const Conversion& conv, SLoc loc) {
    switch (conv.kind) {
        using K = Conversion::Kind;
        case K::ArrayBroadcast: {
            Assert(exprs.size() == 1);
            auto& data = conv.data.get<Conversion::ArrayBroadcastData>();
            exprs.front() = ApplyConversionSequence(exprs, *data.seq, loc);
            exprs.front() = new (*tu) ArrayBroadcastExpr(data.type, exprs.front(), loc);
            return;
        }

        case K::ArrayInit: {
            auto& data = conv.data.get<Conversion::ArrayInitData>();

            // Apply the conversions to the initialisers.
            for (auto [e, c] : zip(exprs, data.elem_convs)) e = ApplyConversionSequence(e, c, e->location());

            // And add the default conversion for the remaining elements if there is one.
            if (auto c = data.broadcast_initialiser()) exprs.push_back(ApplyConversionSequence({}, *c, loc));

            auto init = ArrayInitExpr::Create(*tu, data.ty, exprs, loc);
            exprs.clear();
            exprs.push_back(init);
            return;
        }

        case K::DefaultInit: {
            Assert(exprs.empty());
            exprs.push_back(new (*tu) DefaultInitExpr(conv.data.get<TypeAndValueCategory>().type(), loc));
            return;
        }

        case K::ExpandTuple: {
            Assert(exprs.size() == 1);
            auto e = cast<TupleExpr>(exprs.front());
            exprs.clear();
            append_range(exprs, e->values());
            return;
        }

        case K::ArrayDecay:
        case K::IntegralCast:
        case K::LValueToRValue:
        case K::MaterialisePoison:
        case K::MaterialiseTemporary:
        case K::MutableToImmutable:
        case K::NilToOptional:
        case K::OptionalUnwrap:
        case K::OptionalWrap:
        case K::PointerDeref:
        case K::RangeCast:
        case K::SelectOverload:
        case K::SliceFromArray:
        case K::StripParens:
        case K::StrLitToCStr:
        case K::TupleToFirstElement: {
            Assert(exprs.size() == 1);
            exprs.front() = ApplySimpleConversion(exprs.front(), conv, loc);
            return;
        }

        case K::RecordInit: {
            auto& data = conv.data.get<Conversion::RecordInitData>();
            for (auto [e, seq] : zip(exprs, data.field_convs)) e = ApplyConversionSequence(e, seq, loc);
            auto e = TupleExpr::Create(*tu, data.ty, exprs, {}, loc);
            exprs.clear();
            exprs.push_back(e);
            return;
        }

        case K::ReorderTuple: {
            Assert(exprs.size() == 1);
            auto e = cast<TupleExpr>(exprs.front());
            auto& indices = conv.data.get<SmallVector<u32>>();
            exprs.clear();
            for (auto i : indices) exprs.push_back(e->values()[i]);
            return;
        }

        case K::SliceFromPtrAndSize: {
            auto& data = conv.data.get<Conversion::SliceFromPtrAndSizeData>();
            exprs[0] = ApplyConversionSequence(exprs[0], *data.ptr, loc);
            exprs[1] = ApplyConversionSequence(exprs[1], *data.size, loc);
            auto e = new (*tu) SliceConstructExpr(data.slice, exprs[0], exprs[1], loc);
            exprs.clear();
            exprs.push_back(e);
            return;
        }
    }

    Unreachable("Invalid conversion");
}

auto Sema::ApplyConversionSequence(
    ArrayRef<Expr*> exprs,
    const ConversionSequence& seq,
    SLoc loc
) -> Expr* {
    SmallVector<Expr*, 4> exprs_copy{exprs};
    for (auto& c : seq.conversions) ApplyConversion(exprs_copy, c, loc);
    Assert(exprs_copy.size() == 1, "Conversion sequence should only produce one expression");
    return exprs_copy.front();
}

// There are three ways of initialising a struct type:
//
//   1. By calling a user-defined initialisation procedure;
//      overload resolution is performed against all declared
//      initialisers.
//
//   2. By iterating over every field, building an initialiser
//      from an empty argument list for it, and evaluating the
//      resulting rvalue into the field (this usually ends up
//      performing this step recursively).
//
//   3. By initialising each field directly from a preexisting
//      rvalue of the same type.
//
// Option 1 is called ‘call initialisation’.
// Option 2 is called ‘default/empty initialisation’.
// Option 3 is called ‘direct/literal initialisation’.
//
// A type is default-initialisable/constructible if option 2 is
// valid; this also means option 2 is always taken if the argument
// list is empty.
//
// To determine which option should be applied, first, check if
// there are any user-defined initialisers. If there are, always
// use option 1, unless the argument list is empty and the 'default'
// attribute was specified, in which case use option 2 (specifying
// both 'default' *and* defining an initialiser that takes an empty
// argument list is an error).
//
// If there are no initialisers, use option 2 if the argument list
// is empty, and option 3 otherwise.
auto Sema::BuildAggregateInitialiser(
    ConversionSequence& seq,
    RecordType* r,
    ArrayRef<Expr*> args,
    SLoc loc
) -> MaybeDiags {
    auto& rl = r->layout();

    // First case: option 1 or 2.
    if (auto s = dyn_cast<StructType>(r); s and not s->initialisers().empty())
        return CreateICE(loc, "TODO: Call struct initialiser");

    // Second case: option 3. Option 2 is handled before we get here,
    // i.e. at this point, we know we’re building a literal initialiser.
    Assert(not args.empty(), "Should have handled this in BuildConversionSequence()");
    Assert(rl.has_literal_init(), "Should have rejected before we ever get here");

    // Currently, we don't support initialising unions (apart from zero-initialisation).
    if (rl.bits().is_union)
        return CreateICE(loc, "TODO: Non-trivial union initialisation");

    // Handle initialisation of structs from a named argument list.
    SmallVector<Expr*> reordered_args;
    if (args.size() == 1) {
        auto tuple = dyn_cast<TupleExpr>(args.front());
        if (tuple and tuple->is_named()) {
            auto expected_args = ResolveNamedArguments(tuple, r);
            if (not expected_args) {
                // FIXME: Can we move this to the very top level of BuildConversionSequence()? That
                // way, we can just issue diagnostics directly everywhere in the conversion
                // sequence code, which should simplify it quite a bit.
                DiagnosticsTrap trap{*this};
                ReportSingleOverloadResolutionFailure(
                    r,
                    std::move(expected_args.error()),
                    std::nullopt,
                    tuple,
                    loc
                );
                return trap.get_trapped_diagnostics();
            }

            // Compute the reordering.
            auto indices = llvm::to_vector(vws::transform(*expected_args, [&](Expr* arg) {
                return utils::index_of<u32>(tuple->values(), arg).value();
            }));

            seq.add(Conversion::ReorderTuple(std::move(indices)));

            // Replace 'args'.
            reordered_args = std::move(*expected_args);
            args = reordered_args;
        }
    }

    // Recursively build an initialiser for each element that the user provided.
    std::vector<ConversionSequence> field_seqs;
    for (auto [field, arg] : zip(rl.fields(), args)) {
        auto seq = BuildConversionSequence(field->type, arg, arg->location());
        if (not seq.has_value()) {
            auto note = field->name.empty()
                ? CreateNote(field->location(), "In initialiser for field declared here")
                : CreateNote(field->location(), "In initialiser for field '%6({}%)'", field->name);
            seq.error().push_back(std::move(note));
            return std::move(seq.error());
        }
        field_seqs.push_back(std::move(seq.value()));
    }

    // For now, the number of arguments must match the number of fields in the struct.
    if (rl.fields().size() != args.size()) return CreateError(
        loc,
        "Cannot initialise '{}' from\f'%1(({})%)'",
        Type{r},
        utils::join(args | vws::transform(&Expr::type))
    );

    seq.add(Conversion::RecordInit(Conversion::RecordInitData{r, std::move(field_seqs)}));
    return {};
}

auto Sema::BuildArrayInitialiser(
    ConversionSequence& seq,
    ArrayType* a,
    ArrayRef<Expr*> args,
    SLoc loc
) -> MaybeDiags {
    // Error if there are more initialisers than array elements.
    if (args.size() > u64(a->dimension())) return CreateError(
        loc,
        "Too many elements in array initialiser for '{}'\f(elements: {})",
        Type(a),
        args.size()
    );

    // If there is exactly 1 element, fill the array with copies of it. Since
    // all array elements have the same type, it suffices to build the conversion
    // sequence for it once.
    if (args.size() == 1 && a->dimension() > 1) {
        auto conv = Try(BuildConversionSequence(a->elem(), args, loc));
        seq.add(Conversion::ArrayBroadcast({
            .type = a,
            .seq = std::make_unique<ConversionSequence>(std::move(conv)),
        }));
        return {};
    }

    // Otherwise, use any available arguments to initialise array elements.
    Conversion::ArrayInitData data{a};
    for (auto arg : args) {
        auto conv = Try(BuildConversionSequence(a->elem(), arg, arg->location()));
        data.elem_convs.push_back(std::move(conv));
    }

    // And build an extra conversion from no arguments if we have elements left.
    if (args.size() != u64(a->dimension())) {
        auto conv = Try(BuildConversionSequence(a->elem(), {}, loc));
        data.elem_convs.push_back(std::move(conv));
        data.has_broadcast_initialiser = true;
    }

    seq.add(Conversion::ArrayInit(std::move(data)));
    return {};
}

auto Sema::BuildSliceInitialiser(
    ConversionSequence& seq,
    SliceType* s,
    ArrayRef<Expr*> args,
    bool want_lvalue,
    SLoc loc
) -> MaybeDiags {
    if (args.empty()) {
        seq.add(Conversion::DefaultInit(s));
        return {};
    }

    // Arrays are implicitly convertible to slices.
    if (
        auto arr = dyn_cast<ArrayType>(args.front()->type);
        args.size() == 1 and arr and arr->elem() == s->elem()
    ) {
       seq.add(Conversion::SliceFromArray());
       return {};
    }

    // A mutable slice can be converted to an immutable slice.
    if (
        auto from = dyn_cast<SliceType>(args.front()->type);
        args.size() == 1 and from and from->elem() == s->elem() and not from->is_immutable()
    ) {
        Assert(s->is_immutable());
        if (not want_lvalue) seq.add(Conversion::LValueToRValue());
        seq.add(Conversion::MutableToImmutable(s));
        return {};
    }

    // If we have 2 arguments, and the first is a pointer and the second is
    // an integer, try initialisation from pointer + size.
    //
    // FIXME: This conversion should probably be 'unsafe' and a function (or hard cast!),
    // especially since it may sometimes be unclear whether we want to do this or construct
    // a slice with two elements...
    if (args.size() == 2 and isa<PtrType>(args[0]->type) and args[1]->type->is_integer()) {
        auto ptr_seq = BuildConversionSequence(s->data_ptr_type(), args[0], loc);
        auto size_seq = BuildConversionSequence(Type::IntTy, args[1], loc);
        if (ptr_seq and size_seq) {
            Conversion::SliceFromPtrAndSizeData data;
            data.slice = s;
            data.ptr = std::make_unique<ConversionSequence>(std::move(ptr_seq.value()));
            data.size = std::make_unique<ConversionSequence>(std::move(size_seq.value()));
            seq.add(Conversion::SliceFromPtrAndSize(std::move(data)));
            return {};
        }
    }

    // Build a temporary array and convert it to a slice.
    auto arr_ty = ArrayType::Get(*tu, s->elem(), i64(args.size()));
    Try(BuildArrayInitialiser(seq, arr_ty, args, loc));

    // And convert the array to a slice.
    seq.add(Conversion::SliceFromArray());
    return {};
}

auto Sema::BuildConversionSequence(
    Type var_type,
    ArrayRef<Expr*> args,
    SLoc init_loc,
    bool want_lvalue,
    bool allow_auto_deref
) -> ConversionSequenceOrDiags {
    ConversionSequence seq;

    // Note: 'RequireCompleteType()' may issue diagnostics, but this is a bit unavoidable;
    // generally, we should almost never encounter incomplete types that we fail to complete,
    // so this usually shouldn’t cause any problems even if we’re building conversion sequences
    // tentatively.
    if (not RequireCompleteType(var_type, init_loc)) return CreateError(
        init_loc,
        "Cannot create instance of incomplete type '{}'",
        var_type
    );

    // Simplify tuples and parenthesised expressions.
    //
    // Note that this only handles literal tuples, e.g. a function returning a tuple
    // won’t get unwrapped here, which is probably what we want.
    //
    // Do not simplify tuples with named elements.
    {
        auto single_arg = args.size() == 1 ? args.front() : nullptr;

        // Unwrap TupleExprs that are not structs (e.g. unwrap '(1, 2)', but leave
        // 'foo(1, 2)' as-is). This also means that '()' is converted to no arguments
        // at all, which is exactly what we want because it is supposed to be equivalent
        // to default initialisation in all contexts.
        if (
            auto t = dyn_cast_if_present<TupleExpr>(single_arg);
            t and not t->is_struct() and not t->is_named()
        ) {
            seq.add(Conversion::ExpandTuple());
            args = t->values();
        }

        // Parentheses are significant for array intialisers, e.g. if we have
        // an 'int[2][2]', then '(1, 2)' is different from '((1, 2))': the former
        // results in '[[1, 1], [2, 2]]', the latter in '[[1, 2], [0, 0]]'.
        //
        // Thus, we can’t just strip all parentheses around an expression; instead,
        // remove only a single level here.
        else if (auto p = dyn_cast_if_present<ParenExpr>(single_arg)) {
            seq.add(Conversion::StripParens());
            args = p->expr;
        }
    }

    // Handle a single argument of type 'noreturn'.
    //
    // As a special case, 'noreturn' can be converted to *any* type (and value
    // category). This is because 'noreturn' means we never actually reach the
    // point in the program where the value would be needed, so it’s fine to just
    // pretend that we have one.
    //
    // Materialise a poison value so we don’t crash in codegen; this is admittedly
    // a bit of a hack, but it avoids having to check if we have an insert point
    // everywhere in codegen.
    if (args.size() == 1 and args.front()->type == Type::NoReturnTy) {
        if (var_type != Type::NoReturnTy) seq.add(Conversion::Poison(var_type, Expr::RValue));
        return seq;
    }

    // If there are no arguments, this is default initialisation.
    if (args.empty()) {
        if (var_type->can_default_init()) {
            seq.add(Conversion::DefaultInit(var_type));
            return seq;
        }

        if (var_type->can_init_from_no_args()) {
            return CreateICE(
                init_loc,
                "TODO: non-default empty initialisation of '{}'",
                var_type
            );
        }

        return CreateError(init_loc, "Type '{}' requires a non-empty initialiser", var_type);
    }

    // There are only few (classes of) types that support initialisation
    // from more than one argument.
    if (args.size() > 1) {
        if (auto a = dyn_cast<ArrayType>(var_type)) {
            Try(BuildArrayInitialiser(seq, a, args, init_loc));
            return seq;
        }

        if (auto r = dyn_cast<RecordType>(var_type.ptr())) {
            Try(BuildAggregateInitialiser(seq, r, args, init_loc));
            return seq;
        }

        if (auto s = dyn_cast<SliceType>(var_type)) {
            Try(BuildSliceInitialiser(seq, s, args, want_lvalue, init_loc));
            return seq;
        }

        return CreateError(
            init_loc,
            "Cannot initialise '{}' from\f'%1(({})%)'",
            var_type,
            utils::join(args | vws::transform(&Expr::type))
        );
    }

    // If we get here, there is only a single argument.
    Assert(args.size() == 1);

    // If the types match, ensure this is an rvalue.
    auto ty = args.front()->type;
    if (ty == var_type) {
        if (args.front()->is_lvalue() and not want_lvalue) seq.add(Conversion::LValueToRValue());
        return seq;
    }

    // Perform auto-dereferencing.
    // FIXME: This does the wrong thing if we have e.g. 'int^^' and 'int??'.
    if (
        allow_auto_deref and
        ty->strip_pointers_and_optionals() == var_type->strip_pointers_and_optionals()
    ) {
        while (ty != var_type) {
            auto elem = cast<SingleElementTypeBase>(ty)->elem();
            if (isa<PtrType>(ty)) seq.add(Conversion::PointerDeref());
            else if (isa<OptionalType>(ty)) seq.add(Conversion::OptionalUnwrap(elem));
            else Unreachable("Should have been a pointer or optional");
            ty = elem;
        }

        if (not want_lvalue) seq.add(Conversion::LValueToRValue());
        return seq;
    }

    // Also allow unwrapping any number of nested 1-tuples here, but *only*
    // if the first non-1-tuple is an exact match (e.g. we only want to unwrap
    // '((1, 2))' if we’re initialising an '(int, int)', but not if the target
    // type is 'int[2]').
    //
    // This process is also applied to non-literal tuples, e.g. call to a function
    // returning '(int,)' can be assigned to an 'int'.
    auto a = args.front();
    if (isa<TupleType>(a->type)) {
        TentativeConversionContext tc{seq};
        Type unwrapped_type = a->type;
        for (;;) {
            // This is now an exact match; return it.
            if (unwrapped_type == var_type) {
                seq.add(Conversion::LValueToRValue());
                tc.commit();
                return seq;
            }

            // Unwrap 1-tuples.
            auto t = dyn_cast<TupleType>(unwrapped_type);
            if (not t or t->layout().fields().size() != 1) break;
            unwrapped_type = t->layout().fields().front()->type;
            seq.add(Conversion::TupleToFirstElement());
        }

        // We didn’t find an exact match; undo any conversions we may have added.
        tc.rollback();
    }

    // Allow unwrapping optionals here.
    if (auto opt = dyn_cast<OptionalType>(a->type); opt and opt->elem() == var_type) {
        seq.add(Conversion::OptionalUnwrap(var_type));
        if (not want_lvalue) seq.add(Conversion::LValueToRValue());
        return seq;
    }

    // At this point, we need to perform a type conversion.
    auto TypeMismatch = [&] {
        return CreateError(
            init_loc,
            "Cannot convert expression of type '{}' to '{}'",
            ty,
            var_type
        );
    };

    // A 'trivial' integer constant expression for the purposes
    // of this is a (possibly negated) integer literal.
    auto ConstantIntFits = [&](Expr* e, Type ty) {
        // If this is a (possibly parenthesised and negated) integer
        // that fits in the type of the lhs, convert it. If it doesn’t
        // fit, the type must be larger, so give up.
        Expr* lit = e;
        bool negated = false;
        for (;;) {
            auto u = dyn_cast<UnaryExpr>(lit);
            if (not u or u->op != Tk::Minus) break;
            lit = u->arg;
            negated = not negated;
        }

        // If we ultimately found a literal, evaluate the original expression.
        if (isa_and_present<IntLitExpr>(lit)) {
            auto val = Evaluate(e, false);
            return val and IntegerLiteralFitsInType(val->cast<APInt>(), ty, negated);
        }

        return false;
    };

    switch (var_type->canonical_kind()) {
#       define AST_TYPE(x)
#       define AST_TYPE_SUGAR(x) case TypeBase::Kind::x:
#       include "srcc/AST.inc"
            Unreachable("Canonical type should not be type sugar");

        case TypeBase::Kind::EnumType:
            return TypeMismatch();

        // Note: 'nil' does NOT convert to pointer types since pointers
        // are not nullable! That’s what optional pointers are for.
        case TypeBase::Kind::PtrType: {
            // Allow implicitly converting string literals to C string.
            if (isa<StrLitExpr>(a) and var_type == tu->StrLitTy->data_ptr_type()) {
                seq.add(Conversion::StrLitToCStr());
                return seq;
            }

            auto var_ptr = cast<PtrType>(var_type);
            auto var_pointee = var_ptr->elem();
            auto arg_ptr = dyn_cast<PtrType>(a->type);
            if (arg_ptr) {
                auto pointee = arg_ptr->elem();

                // Allow converting a mutable to an immutable pointer type.
                if (
                    pointee == var_pointee and
                    not arg_ptr->is_immutable()
                ) {
                    Assert(var_ptr->is_immutable());
                    if (not want_lvalue) seq.add(Conversion::LValueToRValue());
                    seq.add(Conversion::MutableToImmutable(var_type));
                    return seq;
                }

                // Allow implicitly converting a pointer to a (multidimensional)
                // array to a pointer to the pointee type.
                //
                // TODO: Investigate what happens if the array has dimension 0.
                while (isa<ArrayType>(pointee)) {
                    pointee = cast<ArrayType>(pointee)->elem();
                    if (pointee == var_pointee) {
                        seq.add(Conversion::ArrayDecay(var_type));
                        return seq;
                    }
                }
            }

            return TypeMismatch();
        }

        case TypeBase::Kind::OpaqueType:
            Unreachable("Attempted to initialise opaque type");

        case TypeBase::Kind::OptionalType: {
            if (a->type == Type::NilTy) {
                seq.add(Conversion::NilToOptional(var_type));
                return seq;
            }

            auto nested_seq = BuildConversionSequence(cast<OptionalType>(var_type)->elem(), args, init_loc);
            if (not nested_seq.has_value()) return TypeMismatch();
            for (auto& c : nested_seq->conversions) seq.add(std::move(c));
            seq.add(Conversion::OptionalWrap(var_type));
            return seq;
        }

        case TypeBase::Kind::SliceType:
            Try(BuildSliceInitialiser(seq, cast<SliceType>(var_type), args, want_lvalue, init_loc));
            return seq;

        case TypeBase::Kind::ArrayType:
            Try(BuildArrayInitialiser(seq, cast<ArrayType>(var_type), args, init_loc));
            return seq;

        case TypeBase::Kind::StructType:
        case TypeBase::Kind::TupleType:
            Try(BuildAggregateInitialiser(seq, cast<RecordType>(var_type), args, init_loc));
            return seq;

        case TypeBase::Kind::ProcType:
            // Type is an overload set; attempt to convert it.
            //
            // This is *not* the same algorithm as overload resolution, because
            // the types must match exactly here, and we also need to check the
            // return type.
            if (ty == Type::UnresolvedOverloadSetTy) {
                auto p_proc_type = dyn_cast<ProcType>(var_type.ptr());
                if (not p_proc_type) return TypeMismatch();

                // Instantiate templates and simply match function types otherwise; we
                // don’t need to do anything fancier here.
                auto overloads = cast<OverloadSetExpr>(a)->overloads();

                // Check non-templates first to avoid storing template substitution
                // for all of them.
                for (auto [j, o] : enumerate(overloads)) {
                    if (isa<ProcTemplateDecl>(o)) continue;

                    // We have a match!
                    //
                    // The internal consistency of an overload set was already verified
                    // when the corresponding declarations were added to their scope, so
                    // if one of them matches, it is the only one that matches.
                    if (cast<ProcDecl>(o)->type == var_type) {
                        seq.add(Conversion::SelectOverload(u16(j)));
                        return seq;
                    }
                }

                // Otherwise, we need to try and instantiate templates in this overload set.
                for (auto o : overloads) {
                    if (not isa<ProcTemplateDecl>(o)) continue;
                    Todo("Instantiate template in nested overload set");
                }

                // None of the overloads matched.
                return CreateError(
                    init_loc,
                    "Overload set for '{}' does not contain a procedure of type '{}'",
                    overloads.front()->name,
                    var_type
                );
            }

            // Otherwise, the types don’t match.
            return TypeMismatch();

        case TypeBase::Kind::RangeType: {
            auto el = cast<RangeType>(var_type)->elem();

            // If the initialising range is a constant expression, try
            // to see if it fits.
            if (
                auto range = dyn_cast<BinaryExpr>(a);
                range and
                (range->op == Tk::DotDotEq or range->op == Tk::DotDotLess) and
                ConstantIntFits(range->lhs, el) and ConstantIntFits(range->rhs, el)
            ) {
                seq.add(Conversion::RangeCast(var_type));
                return seq;
            }

            // Ranges that are not constant expressions require an explicit cast.
            return TypeMismatch();
        }

        // For integers, we can use the common type rule.
        case TypeBase::Kind::IntType: {
            if (ConstantIntFits(a, var_type)) {
                seq.add(Conversion::IntegralCast(var_type));
                return seq;
            }

            // Otherwise, if both are sized integer types, and the initialiser
            // is smaller, we can convert it as well.
            auto ivar = cast<IntType>(var_type);
            auto iinit = dyn_cast<IntType>(a->type);
            if (not iinit or iinit->bit_width() > ivar->bit_width()) return TypeMismatch();
            seq.add(Conversion::LValueToRValue());
            seq.add(Conversion::IntegralCast(var_type));
            return seq;
        }

        // For builtin types, it depends.
        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(var_type)->builtin_kind()) {
                case BuiltinKind::CallArgList:
                case BuiltinKind::Deduced:
                    return CreateError(init_loc, "Type deduction is not allowed here");

                // The only type that can initialise these is the exact
                // same type, so complain (integer literals are not of
                // type 'int' iff the literal doesn’t fit in an 'int',
                // so don’t even bother trying to convert it).
                case BuiltinKind::Void:
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Tree:
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::Nil:
                    return TypeMismatch();
            }

            Unreachable();
        }
    }

    Unreachable();
}

auto Sema::BuildInitialiser(
    Type var_type,
    ArrayRef<Expr*> args,
    SLoc loc,
    bool want_lvalue,
    bool allow_auto_deref
) -> Ptr<Expr> {
    auto seq_or_err = BuildConversionSequence(var_type, args, loc, want_lvalue, allow_auto_deref);

    // The conversion succeeded.
    if (seq_or_err.has_value())
        return ApplyConversionSequence(args, seq_or_err.value(), loc);

    // There was an error.
    for (auto& d : seq_or_err.error()) diags().report(std::move(d));
    return nullptr;
}

auto Sema::TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr> {
    auto seq_or_err = BuildConversionSequence(var_type, {arg}, arg->location());
    if (not seq_or_err.has_value()) return nullptr;
    return ApplyConversionSequence({arg}, seq_or_err.value(), arg->location());
}
