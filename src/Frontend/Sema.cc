#include <srcc/AST/AST.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <base/StringUtils.hh>

#include <ranges>

using namespace srcc;

#define TRY(expression) ({              \
    auto _res = expression;             \
    if (_res.invalid()) return nullptr; \
    _res.get();                         \
})

struct Err {
    bool value;
    Err(bool v) : value{v} {}
    Err(std::nullptr_t) : Err(false) {}
    explicit operator bool() const { return value; }
};

// ============================================================================
//  Helpers
// ============================================================================
void Sema::AddDeclToScope(Scope* scope, Decl* d) {
    // Do not add anonymous decls to the scope.
    if (d->name.empty()) return;

    // And make sure to check for duplicates. Duplicate declarations
    // are usually allowed, but we forbid redeclaring e.g. (template)
    // parameters.
    auto& ds = scope->decls_by_name[d->name];
    if (not ds.empty() and isa<FieldDecl, ParamDecl, TemplateTypeParamDecl>(d)) {
        Error(d->location(), "Redeclaration of '{}'", d->name);
        Note(ds.front()->location(), "Previous declaration was here");
    } else {
        ds.push_back(d);
    }
}

bool Sema::CheckFieldType(Type type, Location loc) {
    if (not CheckVariableType(type, loc)) return false;

    // TODO: Allow this and instead actually perform recursive translation and
    // cycle checking.
    if (not IsCompleteType(type)) return Error(
        loc,
        "Cannot declare field of incomplete type '{}'",
        type
    );

    return true;
}

bool Sema::CheckVariableType(Type ty, Location loc) {
    // Any places that want to do type deduction need to take
    // care of it *before* this is called.
    if (not ty) return false;
    if (ty == Type::DeducedTy) return Error(loc, "Type deduction is not allowed here");
    if (ty == Type::NoReturnTy) return Error(loc, "'{}' is not allowed here", Type::NoReturnTy);
    if (ty == Type::UnresolvedOverloadSetTy) return Error(loc, "Unresolved overload set in parameter declaration");
    return true;
}

auto Sema::ComputeCommonTypeAndValueCategory(MutableArrayRef<Expr*> exprs) -> TypeAndValueCategory {
    Assert(not exprs.empty());
    auto t = exprs.front()->type;
    auto vc = exprs.front()->value_category;
    for (auto e : exprs.drop_front()) {
        // If either type is 'noreturn', the common type is the type of the other
        // branch (unless both are noreturn, in which case the type is just 'noreturn').
        if (e->type == Type::NoReturnTy) continue;
        if (t == Type::NoReturnTy) {
            t = e->type;
            vc = e->value_category;
            continue;
        }

        // If both are lvalues of the same type, the result is an lvalue
        // of that type.
        if (
            t == e->type and
            vc == Expr::LValue and
            e->value_category == Expr::LValue
        ) continue;

        // Finally, if there is a common type, the result is an rvalue of
        // that type. If there isn’t, then we don’t convert lvalues to rvalues
        // because neither side will actually be used..
        // TODO: Actually implement calculating the common type; for now
        //       we just use 'void' if the types don’t match.
        if (t != e->type) return {Type::VoidTy, Expr::RValue};
        vc = Expr::RValue;
    }

    // Perform lvalue-to-rvalue conversion if the final result is an rvalue.
    if (vc == Expr::RValue) {
        for (auto& e : exprs) {
            if (e->type == Type::NoReturnTy) continue;
            e = LValueToRValue(e);
        }
    }

    return {t, vc};
}

auto Sema::CreateReference(Decl* d, Location loc) -> Ptr<Expr> {
    if (not d->valid()) return nullptr;
    switch (d->kind()) {
        default: return ICE(d->location(), "Cannot build a reference to this declaration yet");
        case Stmt::Kind::ProcDecl: return new (*tu) ProcRefExpr(cast<ProcDecl>(d), loc);
        case Stmt::Kind::ProcTemplateDecl: return OverloadSetExpr::Create(*tu, d, loc);

        case Stmt::Kind::TypeDecl:
        case Stmt::Kind::TemplateTypeParamDecl:
            return new (*tu) TypeExpr(cast<TypeDecl>(d)->type, loc);

        case Stmt::Kind::LocalDecl:
        case Stmt::Kind::ParamDecl: {
            auto local = cast<LocalDecl>(d);

            // Check if this variable is declared in a parent procedure and captured it
            // if so; do *not* capture zero-sized variables however since they’ll be deleted
            // entirely anyway.
            if (
                curr_proc().proc != local->parent and
                local->type->memory_size(*tu) != Size()
            ) {
                local->captured = true;
                local->parent->introduces_captures = true;

                // Find the active procedure scope that corresponds to the local’s parent.
                auto st = vws::reverse(proc_stack);
                auto parent_scope = rgs::find_if(st, [&](ProcScopeInfo* i) { return i->proc == local->parent; });
                Assert(parent_scope != st.end());

                // Walk the procedure stack between it and the current procedure.
                //
                // Captures need to be passed down transitively, i.e. if procedure 'a' contains
                // 'b' which contains 'c', and 'c' captures a variable declared in 'a', then
                // 'b' needs to capture it as well even if it doesn’t use it so it can pass it
                // down to 'c'.
                //
                // Because we always propagate captures up whenever we first encounter them, if
                // a local has already been marked for capture in a scope, this loop will have
                // also marked it for capture in any parent scopes, so we can stop if we encounter
                // this situation.
                for (auto p = curr_proc().proc; p != (*parent_scope)->proc; p = p->parent.get_or_null()) {
                    if (p->has_captures) break;
                    p->has_captures = true;
                }
            }

            return new (*tu) LocalRefExpr(
                cast<LocalDecl>(d),
                local->category,
                loc
            );
        }
    }
}

void Sema::DeclareLocal(LocalDecl* d) {
    Assert(d->parent == curr_proc().proc, "Must EnterProcedure before adding a local variable");
    curr_proc().locals.push_back(d);
    AddDeclToScope(curr_scope(), d);
}

void Sema::DiagnoseZeroSizedTypeInNativeProc(Type ty, Location use, bool is_return) {
    // Delay this check if this is an incomplete struct type.
    if (auto s = dyn_cast<StructType>(ty); s and not s->is_complete()) {
        incomplete_structs_in_native_proc_type.emplace_back(s, use, is_return);
        return;
    }

    Error(
        use,
        "{} {}'{}' {} a '%1(native%)' procedure is not supported",
        is_return ? "Returning"sv : "Passing"sv,
        ty != Type::VoidTy ? "zero-sized type "sv : ""sv,
        ty,
        is_return ? "from"sv : "to"sv
    );
}

auto Sema::Evaluate(Stmt* s, Location loc) -> Ptr<Expr> {
    auto value = tu->vm.eval(s);
    if (not value.has_value()) return nullptr;
    return MakeConstExpr(s, std::move(*value), loc);
}

auto Sema::GetScopeFromDecl(Decl* d) -> Ptr<Scope> {
    switch (d->kind()) {
        default: return {};
        case Stmt::Kind::BlockExpr: return cast<BlockExpr>(d)->scope;
        case Stmt::Kind::ProcDecl: return cast<ProcDecl>(d)->scope;
    }
}

bool Sema::IntegerLiteralFitsInType(const APInt& i, Type ty, bool negated) {
    Assert(ty->is_integer(), "Not an integer: '{}'", ty);
    auto bits = ty->bit_width(*tu);
    if (negated) return Size::Bits(i.getSignificantBits()) <= bits;
    else return Size::Bits(i.getActiveBits()) <= bits;
}

bool Sema::IsCompleteType(Type ty, bool null_type_is_complete) {
    if (auto s = dyn_cast_if_present<StructType>(ty.ptr()))
        return s->is_complete();

    if (not ty)
        return null_type_is_complete;

    return true;
}

bool Sema::IsBuiltinVarType(ParsedStmt* stmt) {
    auto b = dyn_cast<ParsedBuiltinType>(stmt);
    return b and b->ty == Type::DeducedTy;
}

bool Sema::IsZeroSizedOrIncomplete(Type ty) {
    Assert(ty, "Must check for null type before calling this");
    if (auto s = dyn_cast<StructType>(ty); s and not s->is_complete()) return true;
    return ty->memory_size(*tu) == Size();
}

auto Sema::LookUpQualifiedName(Scope* in_scope, ArrayRef<DeclName> names) -> LookupResult {
    Assert(names.size() > 1, "Should not be unqualified lookup");

    // The first segment is looked up using unqualified lookup, but don’t
    // complain immediately if we can’t find it because we also need to
    // check module names. The first segment is also allowed to be empty,
    // which means we’re looking up a name in the global scope.
    auto first = names.front();
    if (first.empty()) in_scope = global_scope();
    else {
        auto res = LookUpUnqualifiedName(in_scope, first, false);
        switch (res.result) {
            using enum LookupResult::Reason;
            case Success: {
                auto scope = GetScopeFromDecl(res.decls.front());
                if (scope.invalid()) return LookupResult::NonScopeInPath(first, res.decls.front());
                in_scope = scope.get();
            } break;

            // Can’t perform scope access on an ambiguous lookup.
            case Ambiguous:
                return res;

            // Failed imports can only happen with C++ declarations, which are handled
            // elsewhere; we should *never* fail to import something from one of our
            // own modules.
            case FailedToImport:
                Unreachable("Should never happen");

            // Unqualified lookup should never complain about this.
            case NonScopeInPath:
                Unreachable("Non-scope error in unqualified lookup?");

            // Search module names here.
            //
            // This way, we don’t have to try and represent modules in the AST,
            // and we also don’t have to deal with what happens if unqualified
            // lookup finds a module name, because a module name alone is useless
            // if it’s not on the lhs of `::`.
            case NotFound: {
                auto it = tu->logical_imports.find(first.str());
                if (it == tu->logical_imports.end() or not it->getValue()) return res;
                if (auto s = dyn_cast<ImportedSourceModuleDecl>(it->second)) {
                    in_scope = &s->exports;
                    break;
                }

                // We found an imported C++ header; do a C++ lookup.
                auto hdr = dyn_cast<ImportedClangModuleDecl>(it->second);
                return LookUpCXXName(hdr, names.drop_front());
            } break;
        }
    }

    // For all elements but the last, we have to look up scopes.
    for (auto name : names.drop_front().drop_back()) {
        // Perform lookup.
        auto it = in_scope->decls_by_name.find(name);
        if (it == in_scope->decls_by_name.end()) return LookupResult(name);

        // The declaration must not be ambiguous.
        Assert(not in_scope->decls_by_name.empty(), "Invalid scope entry");
        if (it->second.size() != 1) return LookupResult::Ambiguous(name, it->second);

        // The declaration must reference a scope.
        auto scope = GetScopeFromDecl(it->second.front());
        if (scope.invalid()) return LookupResult::NonScopeInPath(name, it->second.front());

        // Keep going down the path.
        in_scope = scope.get();
    }

    // Finally, look up the name in the last scope.
    return LookUpUnqualifiedName(in_scope, names.back(), true);
}

auto Sema::LookUpUnqualifiedName(Scope* in_scope, DeclName name, bool this_scope_only) -> LookupResult {
    if (name.empty()) return LookupResult(name);
    while (in_scope) {
        // Look up the name in the this scope.
        auto it = in_scope->decls_by_name.find(name);
        if (it == in_scope->decls_by_name.end()) {
            if (this_scope_only) break;
            in_scope = in_scope->parent();
            continue;
        }

        // Found something.
        Assert(not in_scope->decls_by_name.empty(), "Invalid scope entry");
        if (it->second.size() == 1) return LookupResult::Success(it->second.front());
        return LookupResult::Ambiguous(name, it->second);
    }

    // If we couldn’t find it, try to find it in the open modules.
    if (not tu->open_modules.empty()) {
        Assert(tu->open_modules.size() == 1, "TODO: Lookup involving multiple open modules");
        auto mod = tu->open_modules.front();
        if (auto s = dyn_cast<ImportedSourceModuleDecl>(mod))
            return LookUpUnqualifiedName(&s->exports, name, true);
        return LookUpCXXName(cast<ImportedClangModuleDecl>(mod), name);
    }

    return LookupResult::NotFound(name);
}

auto Sema::LookUpName(
    Scope* in_scope,
    ArrayRef<DeclName> names,
    Location loc,
    bool complain
) -> LookupResult {
    auto res = names.size() == 1
                 ? LookUpUnqualifiedName(in_scope, names[0], false)
                 : LookUpQualifiedName(in_scope, names);
    if (not res.successful() and complain) ReportLookupFailure(res, loc);
    return res;
}

auto Sema::LValueToRValue(Expr* expr) -> Expr* {
    if (expr->is_rvalue()) return expr;
    return new (*tu) CastExpr(
        expr->type,
        CastExpr::LValueToRValue,
        expr,
        expr->location(),
        true,
        Expr::RValue
    );
}

bool Sema::MakeCondition(Expr*& e, StringRef op) {
    if (auto ass = dyn_cast<BinaryExpr>(e); ass and ass->op == Tk::Assign)
        Warn(e->location(), "Assignment in condition. Did you mean to write '=='?");

    if (not MakeRValue(Type::BoolTy, e, "Condition", op))
        return false;

    return true;
}

auto Sema::MakeConstExpr(
    Stmt* evaluated_stmt,
    eval::RValue val,
    Location loc
) -> Expr* {
    if (isa_and_present<BoolLitExpr, StrLitExpr, IntLitExpr, TypeExpr>(evaluated_stmt))
        return cast<Expr>(evaluated_stmt);
    return new (*tu) ConstExpr(*tu, std::move(val), loc, evaluated_stmt);
}

auto Sema::MakeLocal(
    Type ty,
    ValueCategory vc,
    String name,
    Location loc
) -> LocalDecl* {
    auto local = new (*tu) LocalDecl(
        ty,
        vc,
        name,
        curr_proc().proc,
        loc
    );

    DeclareLocal(local);
    return local;
}

template <typename Callback>
bool Sema::MakeRValue(Type ty, Expr*& e, Callback EmitDiag) {
    auto init = TryBuildInitialiser(ty, e);
    if (init.invalid()) {
        EmitDiag();
        return false;
    }

    // Make sure it’s an srvalue.
    e = LValueToRValue(init.get());
    return true;
}

bool Sema::MakeRValue(Type ty, Expr*& e, StringRef elem_name, StringRef op) {
    return MakeRValue(ty, e, [&] {
        Error(
             e->location(),
             "{} of '%1({}%)' must be of type\f'{}', but was '{}'",
             elem_name,
             op,
             ty,
             e->type
         );
    });
}

auto Sema::MaterialiseTemporary(Expr* expr) -> Expr* {
    if (expr->is_lvalue()) return expr;
    return new (*tu) MaterialiseTemporaryExpr(expr, expr->location());
}

auto Sema::MaterialiseVariable(Expr* expr) -> Expr* {
    if (isa<LocalRefExpr>(expr)) return expr;
    auto init = BuildInitialiser(expr->type, expr, expr->location()).get(); // Should never fail.
    auto local = MakeLocal(
        expr->type,
        Expr::LValue,
        "",
        expr->location()
    );

    local->set_init(init);
    return CreateReference(local, expr->location()).get();
}

void Sema::ReportLookupFailure(const LookupResult& result, Location loc) {
    switch (result.result) {
        using enum LookupResult::Reason;
        case Success: Unreachable("Diagnosing a successful lookup?");
        case FailedToImport: break; // Already diagnosed.
        case NotFound: Error(loc, "Unknown symbol '{}'", result.name); break;
        case Ambiguous: {
            Error(loc, "Ambiguous symbol '{}'", result.name);
            for (auto d : result.decls) Note(d->location(), "Candidate here");
        } break;
        case NonScopeInPath: {
            Error(loc, "Invalid left-hand side for '::'");
            if (not result.decls.empty()) Note(
                result.decls.front()->location(),
                "'{}' does not contain a scope",
                result.name
            );
        } break;
    }
}

// ============================================================================
//  Initialisation.
// ============================================================================
Sema::Conversion::~Conversion() = default;
auto Sema::ApplySimpleConversion(Expr* e, const Conversion& conv, Location loc) -> Expr* {
    switch (conv.kind) {
        using K = Conversion::Kind;
        default: Unreachable();
        case K::IntegralCast: return new (*tu) CastExpr(
            conv.type(),
            CastExpr::Integral,
            e,
            loc,
            true
        );

        case K::LValueToRValue:
            return LValueToRValue(e);

        case K::MaterialisePoison: return new (*tu) CastExpr(
            conv.type(),
            CastExpr::MaterialisePoisonValue,
            e,
            loc,
            true,
            conv.value_category()
        );

        case K::MaterialiseTemporary:
            return MaterialiseTemporary(e);

        case K::RangeCast: return new (*tu) CastExpr(
            conv.type(),
            CastExpr::Range,
            e,
            loc,
            true
        );

        case K::SelectOverload: {
            auto proc = cast<OverloadSetExpr>(e)->overloads()[conv.data.get<u32>()];
            return CreateReference(proc, loc).get();
        }

        case K::SliceFromArray: return new (*tu) CastExpr(
            SliceType::Get(*tu, cast<ArrayType>(e->type)->elem()),
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

void Sema::ApplyConversion(SmallVectorImpl<Expr*>& exprs, const Conversion& conv, Location loc) {
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

        case K::IntegralCast:
        case K::LValueToRValue:
        case K::MaterialisePoison:
        case K::MaterialiseTemporary:
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
            auto e = TupleExpr::Create(*tu, data.ty, exprs, loc);
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
    Location loc
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
    Location loc
) -> MaybeDiags {
    auto& rl = r->layout();

    // First case: option 1 or 2.
    if (auto s = dyn_cast<StructType>(r); s and not s->initialisers().empty())
        return CreateICE(loc, "TODO: Call struct initialiser");

    // Second case: option 3. Option 2 is handled before we get here,
    // i.e. at this point, we know we’re building a literal initialiser.
    Assert(not args.empty(), "Should have called BuildDefaultInitialiser() instead");
    Assert(rl.has_literal_init(), "Should have rejected before we ever get here");

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
    Location loc
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
    SliceType* a,
    ArrayRef<Expr*> args,
    Location loc
) -> MaybeDiags {
    if (args.empty()) {
        seq.add(Conversion::DefaultInit(a));
        return {};
    }

    // Build a temporary array and convert it to a slice.
    auto arr_ty = ArrayType::Get(*tu, a->elem(), i64(args.size()));
    Try(BuildArrayInitialiser(seq, arr_ty, args, loc));

    // And convert the array to a slice.
    seq.add(Conversion::SliceFromArray());
    return {};
}

auto Sema::BuildConversionSequence(
    Type var_type,
    ArrayRef<Expr*> args,
    Location init_loc,
    bool want_lvalue
) -> ConversionSequenceOrDiags {
    ConversionSequence seq;

    // The type we’re initialising must be complete.
    if (not IsCompleteType(var_type)) return CreateError(
        init_loc,
        "Cannot create instance of incomplete type '{}'",
        var_type
    );

    // Simplify tuples and parenthesised expressions.
    //
    // Note that this only handles literal tuples, e.g. a function returning a tuple
    // won’t get unwrapped here, which is probably what we want.
    {
        auto single_arg = args.size() == 1 ? args.front() : nullptr;

        // Unwrap TupleExprs that are not structs (e.g. unwrap '(1, 2)', but leave
        // 'foo(1, 2)' as-is). This also means that '()' is converted to no arguments
        // at all, which is exactly what we want because it is supposed to be equivalent
        // to default initialisation in all contexts.
        if (
            auto t = dyn_cast_if_present<TupleExpr>(single_arg);
            t and not t->is_struct()
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
            Try(BuildSliceInitialiser(seq, s, args, init_loc));
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
            auto val = tu->vm.eval(e, false);
            return val and IntegerLiteralFitsInType(val->cast<APInt>(), ty, negated);
        }

        return false;
    };

    switch (var_type->kind()) {
        case TypeBase::Kind::PtrType:
            // Allow implicitly converting string literals to C string.
            if (isa<StrLitExpr>(a) and var_type == tu->I8PtrTy) {
                seq.add(Conversion::StrLitToCStr());
                return seq;
            }

            return TypeMismatch();

        case TypeBase::Kind::SliceType:
            Try(BuildSliceInitialiser(seq, cast<SliceType>(var_type), args, init_loc));
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
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
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
    Location loc,
    bool want_lvalue
) -> Ptr<Expr> {
    auto seq_or_err = BuildConversionSequence(var_type, args, loc, want_lvalue);

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

// ============================================================================
//  Templates.
// ============================================================================
auto Sema::DeduceType(ParsedStmt* parsed_type, Type input_type) -> Type {
    if (isa<ParsedTemplateType>(parsed_type))
        return input_type;

    // TODO: Support more complicated deduction.
    return Type();
}

auto Sema::InstantiateTemplate(
    ProcTemplateDecl* pattern,
    TemplateSubstitution& info,
    Location inst_loc
) -> ProcDecl* {
    if (info.instantiation) return info.instantiation;

    // Translate the declaration proper.
    info.instantiation = BuildProcDeclInitial(
        info.scope,
        info.type,
        pattern->name,
        pattern->location(),
        pattern->pattern->type->attrs,
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
    auto it = parsed_template_deduction_infos.find(proc_template->pattern);
    if (it != parsed_template_deduction_infos.end()) {
        for (const auto& [name, indices] : it->second) {
            for (auto i : indices) {
                auto parsed = param_types[i];

                // There might be no corresponding argument if we didn’t
                // pass any variadic arguments to this. In that case, use
                // what ever we deduced earlier, or 'void' if this is the
                // only place where this parameter is deduced.
                Type ty;
                if (input_types.size() == i) {
                    Assert(proc_template->has_variadic_param);
                    deduced.try_emplace(name, Deduced{u32(i), {Type::VoidTy, parsed.type->loc}});
                    continue;
                }

                // Otherwise, deduce the parameter from its corresponding argument(s).
                ty = DeduceType(parsed.type, input_types[i].ty);
                if (not ty) {
                    return SubstitutionResult::DeductionFailed{
                        tu->save(name),
                        i
                    };
                }

                if (parsed.variadic) {
                    Assert(proc_template->has_variadic_param);

                    // We already deduced 'ty' from the first variadic argument, so
                    // start at the one after it, if there is one, and make sure we
                    // get the same type for each argument.
                    for (u32 j = u32(param_types.size()), e = u32(input_types.size()); j < e; j++) {
                        Type next = DeduceType(parsed.type, input_types[j].ty);
                            if (not next) {
                            return SubstitutionResult::DeductionFailed{
                                tu->save(name),
                                j
                            };
                        }

                        if (next != ty) return SubstitutionResult::DeductionAmbiguous{
                            name,
                            u32(i),
                            u32(j),
                            ty,
                            next,
                        };
                    }
                }


                // If the type has not been deduced yet, remember it.
                auto [it, inserted] = deduced.try_emplace(name, Deduced{u32(i), {ty, parsed.type->loc}});
                if (inserted) continue;

                // Otherwise, check that the deduction result is the same.
                if (it->second.second.ty != ty) {
                    return SubstitutionResult::DeductionAmbiguous{
                        name,
                        it->second.first,
                        u32(i),
                        it->second.second.ty,
                        ty,
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
            auto ty = BuildTupleType(input_types.drop_front(i));
            if (not ty) return SubstitutionResult::Error(); // TODO: Maybe this should not be a hard error.
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
    EnterScope scope{*this, ScopeKind::Procedure};
    for (auto [name, d] : deduced) AddDeclToScope(
        scope.get(),
        new (*tu) TemplateTypeParamDecl(name, d.second)
    );

    // Now that that is done, we can convert the type properly.
    auto ty = TranslateProcType(
        proc_template->pattern->type,
        deduced_var_parameters
    );

    // Mark that we’re done substituting.
    for (auto d : scope.get()->decls())
        cast<TemplateTypeParamDecl>(d)->in_substitution = false;

    // Store the type for later if substitution succeeded.
    if (not ty) return {};
    info = new (*tu) TemplateSubstitution(
        id.Intern(tu->allocator()),
        cast<ProcType>(ty),
        scope.get()
    );

    template_substitutions[proc_template].InsertNode(info);
    return info;
}

// ============================================================================
//  Overloading.
// ============================================================================
bool Sema::Candidate::has_valid_proc_type() const {
    return status.is<Viable, ParamInitFailed>();
}

bool Sema::Candidate::has_c_varargs() const {
    if (auto proc = dyn_cast<ProcDecl>(decl)) return proc->proc_type()->has_c_varargs();
    return cast<ProcTemplateDecl>(decl)->pattern->type->attrs.c_varargs;
}

bool Sema::Candidate::is_variadic() const {
    if (auto t = dyn_cast<ProcTemplateDecl>(decl)) return t->has_variadic_param;
    return cast<ProcDecl>(decl)->proc_type()->is_variadic();
}

auto Sema::Candidate::non_variadic_params() const -> u32 {
    return u32(param_count() - is_variadic());
}

auto Sema::Candidate::param_count() const -> usz {
    if (auto proc = dyn_cast<ProcDecl>(decl)) return proc->param_count();
    return cast<ProcTemplateDecl>(decl)->pattern->params().size();
}

auto Sema::Candidate::param_loc(usz index) const -> Location {
    if (auto proc = dyn_cast<ProcDecl>(decl)) return proc->params()[index]->location();
    return cast<ProcTemplateDecl>(decl)->pattern->type->param_types()[index].type->loc;
}

auto Sema::Candidate::proc_type() const -> ProcType* {
    Assert(has_valid_proc_type(), "proc_type() cannot be used if template substitution failed");
    auto d = dyn_cast<ProcDecl>(decl);
    if (d) return d->proc_type();
    return subst.success()->type;
}

auto Sema::Candidate::type_for_diagnostic() const -> SmallUnrenderedString {
    auto d = dyn_cast<ProcDecl>(decl);
    if (d) return d->proc_type()->print();
    if (subst.success()) return subst.success()->type->print();
    return SmallUnrenderedString("(template)");
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
            case K::SelectOverload:
            case K::StripParens:
                break;

            // These are actual type conversions.
            case K::IntegralCast:
            case K::MaterialisePoison:
            case K::RangeCast:
            case K::TupleToFirstElement:
            case K::SliceFromArray:
            case K::StrLitToCStr:
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
        }
    }
    return badness;
}

static void NoteParameter(Sema& S, Decl* proc, u32 i) {
    Location loc;
    String name;

    if (auto d = dyn_cast<ProcDecl>(proc)) {
        if (d->is_imported()) return; // FIXME: Report this location somehow.
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

    if (name.empty()) S.Note(loc, "In argument to parameter declared here");
    else S.Note(loc, "In argument to parameter '{}'", name);
}

/// Check if the number of call arguments ‘matches’ the number of
/// parameters—this does not mean that they’re equal!
template <typename Callee>
static bool ArgumentCountMatchesParameters(usz num_args, Callee* callee) {
    auto required_param_count = callee->param_count() - usz(callee->is_variadic());
    if (num_args == required_param_count) return true;
    if (num_args < required_param_count) return false;
    return callee->has_c_varargs() or callee->is_variadic();
}

/// Convert arguments to parameter types.
template <typename Callee, typename Callback>
static void ConvertArgumentsForCall(
    ArrayRef<Expr*> args,
    Callee* c,
    Callback ConvertArg,
    Location call_loc
) {
    // Convert the argument to each non-variadic parameter.
    for (auto [i, a] : enumerate(args.take_front(c->non_variadic_params())))
        ConvertArg(u32(i), a, a->location());

    // Convert any remaining arguments to the variadic parameter.
    if (c->is_variadic()) ConvertArg(
        u32(c->param_count() - 1),
        args.drop_front(c->non_variadic_params()),
        not args.empty() ? args.front()->location() : call_loc
    );
}

bool Sema::CheckIntents(ProcType* ty, ArrayRef<Expr*> args) {
    bool ok = true;
    for (auto [p, a] : zip(ty->params(), args)) {
        if (
            (p.intent == Intent::Inout or p.intent == Intent::Out) and
            (not isa<CastExpr>(a) or cast<CastExpr>(a)->kind != CastExpr::MaterialisePoisonValue) and
            not a->is_lvalue()
        ) {
            ok = false;
            if (auto dre = dyn_cast_if_present<LocalRefExpr>(a); dre and isa<ParamDecl>(dre->decl)) {
                // If this is itself a parameter, issue a better error.
                Error(
                    a->location(),
                    "Cannot pass parameter of intent %1({}%) to a parameter with intent %1({}%)",
                    cast<ParamDecl>(dre->decl)->intent(),
                    p.intent
                );
                Note(dre->decl->location(), "Parameter declared here");
            } else {
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

    if (min == max and min != AnyNumber and d->param_count() != min)
        return Error(d->location(), "Operator '{}' requires exactly {} parameters", t, min);
    if (min != AnyNumber and d->param_count() < min)
        return Error(d->location(), "Operator '{}' requires at least {} parameters", t, min);
    if (max != AnyNumber and d->param_count() > max)
        return Error(d->location(), "Operator '{}' takes at most {} parameters", t, max);

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

bool Sema::IsUserDefinedOverloadedOperator(Tk, ArrayRef<Type> argument_types) {
    auto CanOverload = [](Type t) { return isa<StructType>(t); };
    return any_of(argument_types, CanOverload);
}

auto Sema::PerformOverloadResolution(
    OverloadSetExpr* overload_set,
    ArrayRef<Expr*> args,
    Location call_loc
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
    auto types = vws::transform(args, &Expr::type) | rgs::to<SmallVector<Type, 10>>();
    bool resolving_builtin_operator = overload_set->name().is_operator_name() and
                                      not IsUserDefinedOverloadedOperator(overload_set->name().operator_name(), types);

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
        if (not ArgumentCountMatchesParameters(args.size(), &c)) {
            c.status = Candidate::ArgumentCountMismatch{};
            return true;
        }

        // Candidate is a regular procedure.
        if (not templ) return true;

        // Candidate is a template.
        SmallVector<TypeLoc, 6> types;
        for (auto arg : args) types.emplace_back(arg->type, arg->location());
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
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, Location loc) {
            // Candidate may have become invalid in the meantime.
            auto st = c.status.get_if<Candidate::Viable>();
            if (not st) return false;

            // Check the next parameter.
            auto& p = params[param_index];
            auto seq_or_err = BuildConversionSequence(
                p.type,
                args,
                loc,
                p.intent == Intent::Out or p.intent == Intent::Inout
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

        ConvertArgumentsForCall(args, &c, ConvertArg, call_loc);
        return true;
    };

    // Check if we have no candidates at all; this can happen if the only
    // candidates in the overload set were builtin operators.
    if (candidates.empty()) {
        Assert(
            overload_set->name().is_operator_name() and args.size() == 2,
            "Only a few binary operators are currently handled this way"
        );

        Error(
            call_loc,
            "Invalid operation: '%1({}%)' between '{}' and '{}'",
            Spelling(overload_set->name().operator_name()),
            args[0]->type,
            args[1]->type
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
        if (c->is_template()) {
            auto inst = InstantiateTemplate(
                cast<ProcTemplateDecl>(c->decl),
                *c->subst.data.get<TemplateSubstitution*>(),
                call_loc
            );

            if (not inst or not inst->is_valid) return {};
            final_callee = inst;
        } else {
            final_callee = cast<ProcDecl>(c->decl);
        }

        // Now is the time to apply the argument conversions.
        SmallVector<Expr*> actual_args;
        actual_args.reserve(args.size());
        ArrayRef conversions(c->status.get<Candidate::Viable>().conversions);
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, Location loc) {
            actual_args.emplace_back(ApplyConversionSequence(
                args,
                conversions[param_index],
                loc
            ));
        };

        ConvertArgumentsForCall(args, c, ConvertArg, call_loc);
        if (not CheckIntents(final_callee->proc_type(), actual_args)) return {};
        return {final_callee, std::move(actual_args)};
    }

    // Overload resolution failed. :(
    ReportOverloadResolutionFailure(candidates, args, call_loc, badness);
    return {};
}

void Sema::ReportOverloadResolutionFailure(
    MutableArrayRef<Candidate> candidates,
    ArrayRef<Expr*> call_args,
    Location call_loc,
    u32 final_badness
) {
    auto FormatTempSubstFailure = [&](const SubstitutionResult& info, std::string& out, std::string_view indent) {
        info.data.visit(utils::Overloaded{// clang-format off
            [](TemplateSubstitution*) { Unreachable("Invalid template even though substitution succeeded?"); },
            [](SubstitutionResult::Error) { Unreachable("Should have bailed out earlier on hard error"); },
            [&](SubstitutionResult::DeductionFailed f) {
                out += std::format(
                    "In param #{}: could not infer ${}",
                    f.param_index + 1,
                    f.param
                );
            },

            [&](const SubstitutionResult::DeductionAmbiguous& a) {
                out += std::format(
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
    };

    // If there is only one overload, print the failure reason for
    // it and leave it at that.
    if (candidates.size() == 1) {
        auto& c = candidates.front();
        auto V = utils::Overloaded{
            // clang-format off
            [](const Candidate::Viable&) { Unreachable(); },
            [&](Candidate::ArgumentCountMismatch) {
                Error(
                    call_loc,
                    "Procedure '%2({}%)' expects {} argument{}, got {}",
                    c.decl->name,
                    c.param_count(),
                    c.param_count() == 1 ? "" : "s",
                    call_args.size()
                );
                Note(c.decl->location(), "Declared here");
            },

            [&](Candidate::DeductionError) {
                std::string extra;
                FormatTempSubstFailure(c.subst, extra, "  ");
                Error(call_loc, "Template argument substitution failed");
                Remark("\r{}", extra);
                Note(c.decl->location(), "Declared here");
            },

            [&](Candidate::ParamInitFailed& p) {
                for (auto& d : p.diags) diags().report(std::move(d));
                NoteParameter(*this, c.decl, p.param_index);
            },
        }; // clang-format on
        c.status.visit(V);
        return;
    }

    // Otherwise, we need to print all overloads, and why they failed.
    std::string message = std::format("%b(Candidates:%)\n");

    // Compute the width of the number field.
    u32 width = u32(std::to_string(candidates.size()).size());

    // First, print all overloads.
    for (auto [i, c] : enumerate(candidates)) {
        // Print the type.
        message += std::format("  %b({}.%) \v{}", i + 1, c.type_for_diagnostic());

        // And include the location if it is valid.
        auto loc = c.decl->location();
        auto lc = loc.seek_line_column(ctx);
        if (lc) {
            message += std::format(
                "\f%b(at%) {}:{}:{}",
                ctx.file_name(loc.file_id),
                lc->line,
                lc->col
            );
        }

        message += "\n";
    }

    message += std::format("\n\r%b(Failure Reason:%)");

    // Collect ambiguous candidates.
    SmallVector<u32> ambiguous_indices;
    for (auto [i, c] : enumerate(candidates))
        if (auto v = c.status.get_if<Candidate::Viable>())
            if (v->badness == final_badness)
                ambiguous_indices.push_back(u32(i + 1));

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        message += std::format("\n  %b({:>{}}.%) ", i + 1, width);
        auto V = utils::Overloaded{
            // clang-format off
            [&] (const Candidate::Viable& v) {
                // Don’t print that a candidate conflicts with itself...
                auto ambiguous = ambiguous_indices;
                erase(ambiguous, u32(i + 1));

                // If the badness is equal to the final badness, then
                // this candidate was ambiguous. Otherwise, another
                // candidate was simply better.
                message += v.badness == final_badness
                    ? std::format("Ambiguous (with {})", utils::join(ambiguous, ", ", "#{}"))
                    : std::format("Another candidate was a better match", v.badness);
            },

            [&](Candidate::ArgumentCountMismatch) {
                auto params = c.param_count();
                message += std::format(
                    "Expected {} arg{}, got {}",
                    params,
                    params == 1 ? "" : "s",
                    call_args.size()
                );
            },

            [&](Candidate::ParamInitFailed& i) {
                message += std::format("In argument to parameter #{}:\n", i.param_index);
                message += utils::Indent(Diagnostic::Render(ctx, i.diags, diags().cols() - 5, false), 2);
            },

            [&](Candidate::DeductionError) {
                message += "Template argument substitution failed";
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
        std::format("Overload resolution failed in call to\f'%2({}%)'", candidates.front().decl->name),
        std::move(message),
    });
}

// ============================================================================
//  Pattern matching.
// ============================================================================
auto Sema::BoolMatchContext::add_constant_pattern(
    const eval::RValue& pattern,
    Location loc
) -> AddResult {
    if (pattern.type() != Type::BoolTy) return InvalidType();
    bool v = pattern.cast<APInt>().getBoolValue();
    auto& matched = v ? true_loc : false_loc;
    auto& other = v ? false_loc : true_loc;
    if (matched.is_valid()) return Subsumed(matched);
    matched = loc;
    return other.is_valid() ? Exhaustive() : Ok();
}

auto Sema::BoolMatchContext::build_comparison(
    Expr* control_expr,
    Expr* pattern_expr
) -> Ptr<Expr> {
    return S.BuildBinaryExpr(
        Tk::EqEq,
        control_expr,
        pattern_expr,
        pattern_expr->location()
    );
}

auto Sema::BoolMatchContext::preprocess(Expr* pattern) -> Ptr<Expr> {
    return pattern;
}

void Sema::BoolMatchContext::note_missing(Location match_loc) {
    if (true_loc.is_valid()) {
        S.Note(match_loc, "Possible value '%1(false%)' is not handled");
    } else if (false_loc.is_valid()) {
        S.Note(match_loc ,"Possible value '%1(true%)' is not handled");
    } else {
        S.Note(match_loc ,"Possible values '%1(true%)' and '%1(false%)' are not handled");
    }
}

Sema::IntMatchContext::IntMatchContext(Sema& s, Type ty) : MatchContext{s}, ty{ty} {
    auto int_width = ty->bit_width(*S.tu);
    min = APInt::getSignedMinValue(unsigned(int_width.bits()));
    max = APInt::getSignedMaxValue(unsigned(int_width.bits()));
}

auto Sema::IntMatchContext::add_constant_pattern(
    const eval::RValue& pattern,
    Location loc
) -> AddResult {
    if (pattern.type() == ty) {
        auto& i = pattern.cast<APInt>();
        return add_range(Range(i, i, loc));
    }

    if (auto r = dyn_cast<RangeType>(pattern.type()); r and r->elem() == ty) {
        const auto& [start, end] = pattern.cast<eval::Range>();
        return add_range(Range(start, end - 1 /* End is exclusive */, loc));
    }

    return InvalidType();
}

auto Sema::IntMatchContext::add_range(Range r) -> AddResult {
    if (ranges.empty()) {
        ranges.push_back(std::move(r));
    } else {
        // If we get here, there is at least one other range; find the
        // smallest range that either overlaps 'r' or is entirely before
        // 'r'.
        auto it = rgs::lower_bound(ranges, r, [](const Range& a, const Range& b) {
            return a.end.slt(b.start);
        });

        // If we found an element, then that means there is a range that
        // is not entirely before 'r'; if it subsumes 'r', drop 'r' entirely,
        // otherwise, merge it into r.
        if (it != ranges.end() and it->overlaps(r)) {
            if (it->subsumes(r)) return Subsumed(it->locations);
            it->merge(r);
        }

        // Either we didn’t find anything, or the range we found is entirely
        // after 'r'; insert 'r' before it (in the former case, this just
        // appends 'r').
        else {
            it = ranges.insert(it, std::move(r));
        }

        // We just modified a range; check if we need to merge it with the
        // range before or after it.
        for (;;) {
            if (it != ranges.end() - 1 and it->adjacent(*(it + 1))) {
                it->merge(*(it + 1));
                ranges.erase(it + 1);
            } else if (it != ranges.begin() and it->adjacent(*(it - 1))) {
                (it -1)->merge(*it);
                ranges.erase(std::exchange(it, it - 1));
            } else {
                break;
            }
        }
    }

    if (
        ranges.size() == 1 and
        ranges.front().start == min and
        ranges.front().end == max
    ) return Exhaustive();
    return Ok();
}

auto Sema::IntMatchContext::build_comparison(
    Expr* control_expr,
    Expr* pattern_expr
) -> Ptr<Expr> {
    Tk op = isa<RangeType>(pattern_expr->type) ? Tk::In : Tk::EqEq;
    return S.BuildBinaryExpr(
        op,
        control_expr,
        pattern_expr,
        pattern_expr->location()
    );
}

auto Sema::IntMatchContext::preprocess(Expr* pattern) -> Ptr<Expr> {
    // Only preprocess integers and ranges of integers. We don’t want to
    // try and convert other types to integers here (that should be done
    // by the code that handles the '==' or 'in' operator).
    if (not isa<RangeType>(pattern->type) and not pattern->type->is_integer())
        return pattern;

    // If the expression is a range, convert it to a range rather than a
    // single integer.
    if (isa<RangeType>(pattern->type)) return S.TryBuildInitialiser(
        RangeType::Get(*S.tu, ty),
        pattern
    );

    return S.TryBuildInitialiser(ty, pattern);
}

void Sema::IntMatchContext::note_missing(Location loc) {
    std::string msg;
    auto Format = [&](const APInt& start, const APInt& end) {
        auto FormatVal = [&](const APInt& i) {
            if (i == min) return std::format("{}%1(.%)%5(min%)", ty);
            if (i == max) return std::format("{}%1(.%)%5(max%)", ty);
            return std::format("%5({}%)", i);
        };

        if (start == end) {
            msg += std::format("\n    {},", FormatVal(start));
        } else {
            msg += std::format(
                "\n    {}%1(..=%){},",
                FormatVal(start),
                FormatVal(end)
            );
        }
    };

#if 0
    // For debugging the merge algorithm.
    for (auto& r : ranges) {
        msg += std::format("\n    {}, {}", r.start, r.end);
    }

    S.Note(loc, "DEBUG. VALUE RANGES ARE\n%r({}%)", msg);
    return;
#endif

    // It’s easiest to handle this case separately.
    auto sz = ranges.size();
    if (sz == 0) {
        Format(min, max);
    } else {
        // Compute the first unsatisfied value.
        //
        // This is normally '<type>.min', unless the first range starts
        // with that value. In that case, take that range’s 'end' value
        // plus 1. Note that this cannot overflow since in that case,
        // the span of that range would be the entire value domain, in
        // which case we should never get here in the first place since
        // the match would exhaustive.
        bool first_is_min = ranges.front().start == min;
        APInt first_unsatisfied = first_is_min ? ranges.front().end + 1 : min;

        // Go through all ranges (except the first if its start is the
        // minimum value) and collect all unsatisfied values between
        // them.
        APInt one{min.getBitWidth(), 1};
        for (auto& r : ArrayRef(ranges).drop_front(first_is_min ? 1 : 0)) {
            Format(first_unsatisfied, r.start.ssub_sat(one));
            first_unsatisfied = r.end.sadd_sat(one);
        }

        // Append any remaining values after the last range.
        //
        // Note that there is an edge case here: the last range may
        // end immediately before '<type>.max'.
        if (
            first_unsatisfied != max or
            (sz != 0 and ranges.back().end == max - 1)
        ) Format(first_unsatisfied, max);
    }


    if (msg.ends_with(",")) msg.pop_back();
    S.Note(loc, "Possible value ranges not handled:\n%r({}%)", msg);
}

void Sema::MarkUnreachableAfter(auto it, MutableArrayRef<MatchCase> cases) {
    Assert(it != cases.end());
    if (std::next(it) != cases.end()) {
        auto next = std::next(it);
        Warn(next->loc, "This and any following patterns will never be matched");
        Note(it->loc, "Because this pattern already makes the 'match' fully exhaustive");
        for (auto& u : MutableArrayRef<MatchCase>{next, cases.end()}) u.unreachable = true;
    }
}

// TODO:
//   - integer patterns
//   - wildcard pattern
//   - named wildcard pattern (`match x { var y: }`)

template <typename MContext>
bool Sema::CheckMatchExhaustive(
    MContext& mc,
    Location match_loc,
    Expr* control_expr,
    Type ty,
    MutableArrayRef<MatchCase> cases
) {
    bool ok = true;
    for (auto [i, c] : enumerate(cases)) {
        auto BuildComparison = [&] {
            auto cmp = mc.build_comparison(control_expr, c.cond.expr());
            if (cmp) c.cond = cmp.get();
        };

        auto WarnAlreadyExhaustive = [&] {
            MarkUnreachableAfter(cases.begin() + i, cases);
        };

        // A wildcard pattern means we don’t need to look at anything else.
        if (c.cond.is_wildcard()) {
            WarnAlreadyExhaustive();
            return true;
        }

        // For some types, we want to perform an initial type conversion (e.g.
        // we want to convert integer literals to 'i8' if we’re matching an 'i8'
        // *before* evaluating the expression).
        //
        // This is allowed to fail and should never issue any diagnostics.
        if (auto init = mc.preprocess(c.cond.expr()).get_or_null())
            c.cond.expr() = init;

        // We only really care about constants here.
        auto e = c.cond.expr();
        auto rv = tu->vm.eval(LValueToRValue(e), false);
        if (not rv.has_value()) {
            BuildComparison();
            continue;
        }

        // Add the constant.
        auto [res, locations] = mc.add_constant_pattern(rv.value(), e->location());
        c.cond = MakeConstExpr(e, std::move(rv.value()), e->location());
        switch (res) {
            using K = MatchContext::AddResult::Kind;
            case K::Ok: BuildComparison(); break;
            case K::Exhaustive: {
                WarnAlreadyExhaustive();
                BuildComparison();
                return true;
            }

            case K::InvalidType: {
                Error(e->location(), "Cannot match '{}' against '{}'", e->type, ty);
                ok = false;
                c.unreachable = true;
                break;
            }

            case K::Subsumed: {
                Warn(e->location(), "Pattern will never be matched");

                // FIXME: Stuff like this should *really* just be rendered in-line.
                if (locations.size() == 1) {
                    Note(locations.front(), "Because it is subsumed by this preceding pattern");
                } else {
                    for (auto loc : locations) {
                        Note(loc, "Because it is partially subsumed by this preceding pattern");
                    }
                }

                c.unreachable = true;
                break;
            }
        }
    }

    if (ok) {
        Error(match_loc, "'match' is not exhaustive");
        mc.note_missing(match_loc);
    }
    return false;
}

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildAssertExpr(
    Expr* cond,
    Ptr<Expr> msg,
    bool is_compile_time,
    Location loc
) -> Ptr<Expr> {
    if (not MakeCondition(cond, "assert")) return {};

    // Message must be a string literal.
    // TODO: Allow other string-like expressions.
    if (auto m = msg.get_or_null(); m and not isa<StrLitExpr>(m)) return Error(
        m->location(),
        "Assertion message must be a string literal",
        m->type
    );

    auto a = new (*tu) AssertExpr(cond, std::move(msg), false, loc);
    if (not is_compile_time) return a;
    return Evaluate(a, loc);
}

auto Sema::BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, Location loc) -> BlockExpr* {
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
    Location loc
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

    auto BuildArithmeticOrComparisonOperator = [&](bool comparison) -> Ptr<BinaryExpr> {
        if (not CheckIntegral() or not ConvertToCommonType()) return nullptr;
        return Build(comparison ? Type::BoolTy : lhs->type);
    };

    auto BuildCall = [&](DeclName fun) -> Ptr<Expr> {
        auto ref = BuildDeclRefExpr(fun, loc);
        if (not ref) return nullptr;
        return BuildCallExpr(ref.get(), {lhs, rhs}, loc);
    };

    if (IsUserDefinedOverloadedOperator(op, {lhs->type, rhs->type}))
        return BuildCall(DeclName(op));

    switch (op) {
        default: Unreachable("Invalid binary operator: '{}'", op);

        // Array or slice subscript.
        case Tk::LBrack: {
            if (auto ty = dyn_cast<TypeExpr>(lhs)) {
                auto arr = BuildArrayType({ty->value, ty->location()}, rhs);
                if (not arr) return {};
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
            if (isa<TupleType, ArrayType>(lhs->type)) lhs = MaterialiseTemporary(lhs);
            else lhs = LValueToRValue(lhs);

            // For tuples, the integer must be a compile-time constant, and
            // the result of a subscript operation is a member access.
            if (auto ty = dyn_cast<TupleType>(lhs->type)) {
                auto res = tu->vm.eval(rhs);
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
            return Build(cast<SingleElementTypeBase>(lhs->type)->elem(), LValue);
        }

        case Tk::As: {
            auto ty_expr = dyn_cast<TypeExpr>(rhs);
            if (not ty_expr) return Error(rhs->location(), "Expected type");
            auto ty = ty_expr->value;

            // This is a no-op if the types are the same.
            if (ty == lhs->type) return lhs;

            // Casting between integer types is always allowed.
            if (lhs->type->is_integer() and ty->is_integer()) return new (*tu) CastExpr(
                ty,
                CastExpr::Integral,
                LValueToRValue(lhs),
                loc
            );

            // Casting to void does nothing.
            if (ty == Type::VoidTy) return new (*tu) CastExpr(
                ty,
                CastExpr::ExplicitDiscard,
                lhs,
                loc
            );

            // For everything else, just try to build an initialiser.
            return BuildInitialiser(ty, lhs, loc);
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
                auto max = tu->vm.eval(rhs, false);
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
            if (not ConvertToCommonType()) return nullptr;

            // For slices, call an overloaded operator.
            if (isa<SliceType>(lhs->type)) {
                auto call = BuildCall(DeclName(Tk::EqEq));
                return op == Tk::Neq ? BuildUnaryExpr(Tk::Not, call.get(), false, loc) : call;
            }

            return Build(Type::BoolTy);
        }

        // Logical operator.
        case Tk::And:
        case Tk::Or:
        case Tk::Xor: {
            if (not MakeRValue(Type::BoolTy, lhs, "Left operand", Spelling(op))) return {};
            if (not MakeRValue(Type::BoolTy, rhs, "Right operand", Spelling(op))) return {};
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

            // Prohibit assignment to 'in' parameters.
            if (auto ref = dyn_cast<LocalRefExpr>(lhs)) {
                if (
                    auto param = dyn_cast<ParamDecl>(ref->decl);
                    param and param->intent() == Intent::In
                ) return Error(lhs->location(), "Cannot assign to '%1(in%)' parameter");
            }

            // LHS must be an lvalue.
            if (lhs->value_category != LValue)
                return Error(lhs->location(), "Invalid target for assignment");

            // Regular assignment.
            if (op == Tk::Assign) {
                if (isa<RecordType, ArrayType>(lhs->type)) return ICE(rhs->location(), "TODO: struct/array assignment");
                if (not MakeRValue(lhs->type, rhs, DiagnoseRHS)) return nullptr;
                return Build(lhs->type, LValue);
            }

            // Compound assignment.
            if (not CheckIntegral()) return nullptr;
            if (not MakeRValue(lhs->type, rhs, DiagnoseRHS)) return nullptr;
            if (not rhs) return nullptr;
            if (op != Tk::StarStarEq) return Build(lhs->type, LValue);

            // '**=' requires a separate function since it needs to return the lhs.
            if (lhs->type->bit_width(*tu) < Size::Bits(2)) return ErrorExpI1();
            return CastExpr::Dereference(*tu, TRY(BuildCall(DeclName(Tk::StarStarEq))));
        }
    }
}

auto Sema::BuildBuiltinCallExpr(
    BuiltinCallExpr::Builtin builtin,
    ArrayRef<Expr*> args,
    Location call_loc
) -> Ptr<BuiltinCallExpr> {
    auto ForbidArgs = [&](StringRef builtin_name) {
        if (args.size() == 0) return true;
        Error(call_loc, "{} takes no arguments", builtin_name);
        return false;
    };

    switch (builtin) {
        case BuiltinCallExpr::Builtin::Unreachable: {
            if (not ForbidArgs("__srcc_unreachable")) return nullptr;
            return BuiltinCallExpr::Create(*tu, builtin, Type::NoReturnTy, {}, call_loc);
        }
    }

    Unreachable("Invalid builtin type: {}", +builtin);
}

auto Sema::BuildBuiltinMemberAccessExpr(
    BuiltinMemberAccessExpr::AccessKind ak,
    Expr* operand,
    Location loc
) -> Ptr<BuiltinMemberAccessExpr> {
    auto type = [&] -> Type {
        switch (ak) {
            using AK = BuiltinMemberAccessExpr::AccessKind;
            case AK::TypeAlign:
            case AK::TypeArraySize:
            case AK::TypeBits:
            case AK::TypeBytes:
                return Type::IntTy;

            case AK::SliceSize:
                return Type::IntTy;

            case AK::TypeName:
                return tu->StrLitTy;

            case AK::RangeStart:
            case AK::RangeEnd:
                return cast<RangeType>(operand->type)->elem();

            case AK::TypeMaxVal:
            case AK::TypeMinVal:
                return cast<TypeExpr>(operand)->value;

            case AK::SliceData:
                return PtrType::Get(*tu, cast<SliceType>(operand->type)->elem());
        }
        Unreachable();
    }();

    return new (*tu) BuiltinMemberAccessExpr{
        type,
        Expr::RValue,
        operand,
        ak,
        loc
    };
}

auto Sema::BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr> {
    // If this is an overload set, perform overload resolution.
    Expr* resolved_callee = nullptr;
    SmallVector<Expr*> converted_args;
    if (auto os = dyn_cast<OverloadSetExpr>(callee_expr)) {
        ProcDecl* d{};
        std::tie(d, converted_args) = PerformOverloadResolution(os, args, loc);
        if (not d) return nullptr;
        resolved_callee = CreateReference(d, loc).get();
    }

    // If the ‘callee’ is a type, then this is an initialiser call.
    else if (isa<TypeExpr>(callee_expr)) {
        auto type = tu->vm.eval(callee_expr);
        if (not type) return ICE(
            callee_expr->location(),
            "Failed to evaluate expression designating a type"
        );

        return BuildInitialiser(
            type.value().cast<Type>(),
            args,
            loc
        );
    }

    // If the type of this is a procedure, then we can skip overload
    // resolution. While this also means we have some code duplication,
    // we don’t have to build conversion sequences here and can instead
    // apply them immediately, and overload resolution also wouldn’t
    // work for indirect calls since for those we don’t have a reference
    // to the procedure declaration.
    else if (auto ty = dyn_cast<ProcType>(callee_expr->type)) {
        resolved_callee = LValueToRValue(callee_expr);

        // Check arg count.
        if (not ArgumentCountMatchesParameters(args.size(), ty)) {
            auto decl = dyn_cast<ProcRefExpr>(callee_expr);
            Error(
                loc,
                "Procedure{} expects {} argument{}, got {}",
                decl and not decl->decl->name.empty() ? std::format(" '{}'", decl->decl->name) : "",
                ty->params().size(),
                ty->params().size() == 1 ? "" : "s",
                args.size()
            );

            if (decl) Note(decl->decl->location(), "Declared here");
            return nullptr;
        }

        bool ok = true;
        auto ConvertArg = [&](u32 param_index, ArrayRef<Expr*> args, Location loc) {
            auto p = ty->params()[param_index];
            auto arg = BuildInitialiser(
                p.type,
                args,
                loc,
                p.intent == Intent::Out or p.intent == Intent::Inout
            );

            // Point to the procedure if this is a direct call.
            if (not arg) {
                if (isa<ProcRefExpr>(callee_expr)) {
                    auto proc = cast<ProcRefExpr>(callee_expr);
                    NoteParameter(*this, proc->decl, u32(param_index));
                }

                ok = false;
                return;
            }

            converted_args.push_back(arg.get());
        };

        ConvertArgumentsForCall(args, ty, ConvertArg, loc);

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

    // And check variadic arguments.
    auto ty = cast<ProcType>(resolved_callee->type);
    if (ty->has_c_varargs()) {
        for (auto a : args.drop_front(ty->params().size())) {
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
        Expr::RValue,
        resolved_callee,
        converted_args,
        loc
    );
}

auto Sema::BuildDeclRefExpr(ArrayRef<DeclName> names, Location loc) -> Ptr<Expr> {
    auto res = LookUpName(curr_scope(), names, loc, false);
    if (res.successful()) return CreateReference(res.decls.front(), loc);

    // Overload sets are ok here.
    // TODO: Validate overload set; i.e. that there are no two functions that
    // differ only in return type, or not at all. Also: don’t allow overloading
    // on intent (for now).
    if (
        res.result == LookupResult::Reason::Ambiguous and
        isa<ProcDecl, ProcTemplateDecl>(res.decls.front())
    ) return OverloadSetExpr::Create(*tu, res.decls, loc);

    ReportLookupFailure(res, loc);
    return {};
}

auto Sema::BuildEvalExpr(Stmt* arg, Location loc) -> Ptr<Expr> {
    // An eval expression returns an rvalue.
    if (auto e = dyn_cast<Expr>(arg)) {
        auto init = BuildInitialiser(arg->type_or_void(), e, loc);
        if (not init) return nullptr;
        arg = init.get();
    }

    return Evaluate(arg, loc);
}

auto Sema::BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, Location loc) -> Ptr<IfExpr> {
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
    Location loc
) -> Ptr<Expr> {
    Expr* control = control_expr.get_or_null();
    bool exhaustive = false;
    if (control) {
        control = MaterialiseVariable(control);
        auto ty = control->type;
        if (ty->is_integer()) {
            IntMatchContext mc{*this, ty};
            exhaustive = CheckMatchExhaustive(mc, loc, control, ty, cases);
        } else if (ty == Type::BoolTy) {
            BoolMatchContext mc{*this};
            exhaustive = CheckMatchExhaustive(mc, loc, control, ty, cases);
        } else {
            return Error(
                control->location(),
                "Matching a value of type '{}' is not supported",
                control->type
            );
        }
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
                c.body->value_category_or_rvalue() == Expr::LValue
            );
        })) {
            tvc = {ty, Expr::LValue};
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

auto Sema::BuildParamDecl(
    ProcScopeInfo& proc,
    const ParamTypeData* param,
    u32 index,
    bool with_param,
    String name,
    Location loc
) -> ParamDecl* {
    auto decl = new (*tu) ParamDecl(param, Expr::LValue, name, proc.proc, index, with_param, loc);
    if (not param->type) decl->set_invalid();
    DeclareLocal(decl);
    return decl;
}

auto Sema::BuildProcDeclInitial(
    Scope* proc_scope,
    ProcType* ty,
    DeclName name,
    Location loc,
    ParsedProcAttrs attrs,
    ProcTemplateDecl* pattern
) -> ProcDecl* {
    // Get the parent procedure, which determines whether this is a nested
    // procedure; top-level procedures are *not* considered nested inside
    // the initialiser procedure (since the latter is just an implementation
    // detail).
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
    else parent = curr_proc().proc;
    if (parent == tu->initialiser_proc) parent = {};
    auto proc = ProcDecl::Create(
        *tu,
        nullptr,
        ty,
        name,
        attrs.extern_ ? Linkage::Imported : Linkage::Internal,
        attrs.nomangle or attrs.native ? Mangling::None : Mangling::Source,
        parent,
        loc
    );

    // Remember what template we were instantiated from.
    proc->instantiated_from = pattern;

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

auto Sema::BuildReturnExpr(Ptr<Expr> value, Location loc, bool implicit) -> ReturnExpr* {
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
    if (auto val = value.get_or_null())
        value = BuildInitialiser(proc->return_type(), {val}, val->location());

    return new (*tu) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildStaticIfExpr(
    Expr* cond,
    ParsedStmt* then,
    Ptr<ParsedStmt> else_,
    Location loc
) -> Ptr<Stmt> {
    // Otherwise, check this now.
    if (not MakeCondition(cond, "#if")) return {};
    auto val = tu->vm.eval(cond);
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

auto Sema::BuildTypeExpr(Type ty, Location loc) -> TypeExpr* {
    return new (*tu) TypeExpr(ty, loc);
}

auto Sema::BuildUnaryExpr(Tk op, Expr* operand, bool postfix, Location loc) -> Ptr<Expr> {
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

    // Handle overloaded operators.
    if (IsUserDefinedOverloadedOperator(op, operand->type)) {
        auto ref = BuildDeclRefExpr(DeclName(op), loc);
        if (not ref) return nullptr;
        return BuildCallExpr(ref.get(), operand, loc);
    }

    // Handle prefix operators.
    switch (op) {
        default: Unreachable("Invalid unary operator: {}", op);

        // Lvalue -> Pointer
        case Tk::Ampersand: {
            // FIXME: This message needs improving; we shouldn’t expect users
            // to know what an lvalue is (or why something isn’t an lvalue in
            // the case of e.g. if/match).
            if (not operand->is_lvalue()) return Error(loc, "Cannot take address of non-lvalue");
            return Build(PtrType::Get(*tu, operand->type), Expr::RValue);
        }

        // Pointer -> Lvalue.
        case Tk::Caret: {
            auto ptr = dyn_cast<PtrType>(operand->type);
            if (not ptr) return Error(
                loc,
                "Cannot dereference value of non-pointer type '{}'",
                operand->type
            );

            operand = LValueToRValue(operand);
            return Build(ptr->elem(), Expr::LValue);
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
            if (operand->value_category != Expr::LValue) return Error(
                operand->location(),
                "Invalid operand for '{}'",
                Spelling(op)
            );

            return BuildIntOp(Expr::LValue);
        }
    }
}

auto Sema::BuildWhileStmt(Expr* cond, Stmt* body, Location loc) -> Ptr<WhileStmt> {
    if (not MakeCondition(cond, "while")) return {};
    return new (*tu) WhileStmt(cond, body, loc);
}

// ============================================================================
//  Translation Driver
// ============================================================================
Sema::EnterProcedure::EnterProcedure(Sema& S, ProcDecl* proc)
    : info{S, proc} {
    Assert(proc->scope, "Entering procedure without scope?");
    S.proc_stack.emplace_back(&info);
}

Sema::EnterScope::EnterScope(Sema& S, ScopeKind kind, bool should_enter) : S{S} {
    if (not should_enter) return;
    Assert(not S.scope_stack.empty(), "Should not be used for the global scope");
    scope = S.tu->create_scope(S.curr_scope(), kind);
    S.scope_stack.push_back(scope);
}

Sema::EnterScope::EnterScope(Sema& S, bool should_enter)
    : EnterScope(S, ScopeKind::Block, should_enter) {}

Sema::EnterScope::EnterScope(Sema& S, Scope* scope) : S{S}, scope{scope} {
    if (not scope) return;

    // Allow entering the global scope multiple times; this is a bit
    // of a hack admittedly...
    Assert(
        S.scope_stack.empty() or S.curr_scope() != scope,
        "Entering the same scope twice in a row; this is probably a bug"
    );

    S.scope_stack.push_back(scope);
}

Sema::Sema(Context& ctx) : ctx(ctx) {}
Sema::~Sema() = default;

auto Sema::Translate(
    const LangOpts& opts,
    ParsedModule::Ptr preamble,
    SmallVector<ParsedModule::Ptr> modules,
    ArrayRef<std::string> module_search_paths,
    ArrayRef<std::string> clang_include_paths,
    bool load_runtime
) -> TranslationUnit::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};

    // Create the TU.
    S.tu = TranslationUnit::Create(
        first->context(),
        opts,
        first->name,
        first->is_module
    );

    // Take ownership of the modules.
    bool have_preamble = preamble != nullptr;
    if (have_preamble) S.parsed_modules.push_back(std::move(preamble));
    for (auto& m : modules) S.parsed_modules.push_back(std::move(m));
    S.search_paths = module_search_paths;
    S.clang_include_paths = clang_include_paths;

    // Translate it.
    S.Translate(have_preamble, load_runtime);
    return std::move(S.tu);
}

void Sema::Translate(bool have_preamble, bool load_runtime) {
    // Take ownership of any resources of the parsed modules.
    for (auto& p : parsed_modules) {
        tu->add_allocator(std::move(p->string_alloc));
        tu->add_integer_storage(std::move(p->integers));
        for (auto& [decl, info] : p->template_deduction_infos)
            parsed_template_deduction_infos[decl] = std::move(info);
    }

    // Set up scope stacks.
    tu->initialiser_proc->scope = global_scope();
    EnterProcedure _{*this, tu->initialiser_proc};

    // Initialise FFI types.
    auto DeclareBuiltinType = [&](String name, Type type) {
        auto decl = new (*tu) TypeDecl(type, name, Location());
        AddDeclToScope(global_scope(), decl);
    };

    DeclareBuiltinType("__srcc_ffi_char", tu->FFICharTy);
    DeclareBuiltinType("__srcc_ffi_wchar", tu->FFIWCharTy);
    DeclareBuiltinType("__srcc_ffi_short", tu->FFIShortTy);
    DeclareBuiltinType("__srcc_ffi_int", tu->FFIIntTy);
    DeclareBuiltinType("__srcc_ffi_long", tu->FFILongTy);
    DeclareBuiltinType("__srcc_ffi_longlong", tu->FFILongLongTy);

    // Translate the preamble first since the runtime and other modules rely
    // on it always being available.
    auto modules = ArrayRef(parsed_modules).drop_front(have_preamble ? 1 : 0);
    SmallVector<Stmt*> top_level_stmts;
    if (have_preamble) TranslateStmts(top_level_stmts, parsed_modules.front()->top_level);

    // Load the runtime.
    if (load_runtime) LoadModule(
        constants::RuntimeModuleName,
        constants::RuntimeModuleName,
        modules.front()->program_or_module_loc,
        false,
        false
    );

    // And process other imports.
    for (auto& m : modules) {
        for (auto& i : m->imports) {
            LoadModule(
                i.import_name,
                i.linkage_names,
                i.loc,
                i.is_open_import,
                i.is_header_import
            );
        }
    }

    // Bail if we couldn’t load a module.
    if (ctx.diags().has_error()) return;

    // Collect all statements and translate them.
    for (auto& p : modules) TranslateStmts(top_level_stmts, p->top_level);
    tu->file_scope_block = BlockExpr::Create(*tu, global_scope(), top_level_stmts, Location{});

    // File scope block should never be dependent.
    tu->initialiser_proc->finalise(
        BuildProcBody(tu->initialiser_proc, tu->file_scope_block),
        curr_proc().locals
    );

    // Perform any checks that require translation to be complete.
    for (auto [s, loc, is_return] : incomplete_structs_in_native_proc_type) {
        Assert(s->is_complete());
        if (s->layout().size() == Size()) DiagnoseZeroSizedTypeInNativeProc(s, loc, is_return);
    }

    // Sanity check.
    Assert(proc_stack.size() == 1);
    Assert(scope_stack.size() == 1);
}

void Sema::TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed, Type desired_type) {
    // Translate object declarations first since they may be out of order.
    //
    // Note that only the declaration part of definitions is translated here, e.g.
    // for a ProcDecl, we only translate the procedure type, not the body; the latter
    // is handled when we actually get to it later on.
    //
    // This translation only applies to *some* decls. It is allowed to do nothing,
    // but if it does fail, then we can’t process the rest of this scope.
    llvm::MapVector<ParsedStmt*, Decl*> translated_decls;
    bool ok = true;
    for (auto p : parsed) {
        if (auto d = dyn_cast<ParsedDecl>(p)) {
            auto proc = TranslateDeclInitial(d);

            // There was no initial translation here.
            if (not proc.has_value()) continue;

            // Initial translation encountered an error.
            if (proc->invalid()) ok = false;

            // Otherwise, store the translated decl for later.
            else translated_decls[d] = proc->get();
        }
    }

    // Stop if there was a problem.
    if (not ok) return;

    // Having collected out-of-order symbols, now translate all statements for real.
    for (auto p : parsed) {
        // Decls need the initial data that we passed to them earlier.
        if (auto d = dyn_cast<ParsedDecl>(p)) {
            auto decl = TranslateEntireDecl(translated_decls[p], d);
            if (decl.present()) stmts.push_back(decl.get());
            continue;
        }

        auto stmt = TranslateStmt(p, p == parsed.back() ? desired_type : Type());
        if (stmt.present()) stmts.push_back(stmt.get());
    }
}

// ============================================================================
//  Translation of Individual Statements
// ============================================================================
auto Sema::TranslateAssertExpr(ParsedAssertExpr* parsed, Type) -> Ptr<Stmt> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    Ptr<Expr> msg;
    if (auto m = parsed->message.get_or_null()) msg = TRY(TranslateExpr(m));
    return BuildAssertExpr(cond, msg, parsed->is_compile_time, parsed->loc);
}

auto Sema::TranslateBinaryExpr(ParsedBinaryExpr* expr, Type desired_type) -> Ptr<Stmt> {
    auto lhs = TRY(TranslateExpr(expr->lhs, desired_type));
    auto rhs = TRY(TranslateExpr(expr->rhs, desired_type));
    return BuildBinaryExpr(expr->op, lhs, rhs, expr->loc);
}

auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    EnterScope _{*this, parsed->should_push_scope};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, parsed->stmts(), desired_type);
    return BuildBlockExpr(curr_scope(), stmts, parsed->loc);
}

auto Sema::TranslateBoolLitExpr(ParsedBoolLitExpr* parsed, Type) -> Ptr<Stmt> {
    return new (*tu) BoolLitExpr(parsed->value, parsed->loc);
}

auto Sema::TranslateCallExpr(ParsedCallExpr* parsed, Type) -> Ptr<Stmt> {
    // Translate arguments.
    SmallVector<Expr*> args;
    bool errored = false;
    for (auto a : parsed->args()) {
        auto expr = TranslateExpr(a);
        if (expr.invalid()) errored = true;
        else args.push_back(expr.get());
    }

    // Stop if there was an error.
    if (errored) return nullptr;

    // Callee may be a builtin.
    if (auto dre = dyn_cast<ParsedDeclRefExpr>(parsed->callee); dre && dre->names().size() == 1) {
        using B = BuiltinCallExpr::Builtin;
        auto bk = llvm::StringSwitch<std::optional<B>>(dre->names().front().str())
            .Case("__srcc_unreachable", B::Unreachable)
            .Default(std::nullopt);

        // We have a builtin!
        if (bk.has_value()) return BuildBuiltinCallExpr(*bk, args, parsed->loc);
    }

    // Translate callee.
    auto callee = TRY(TranslateExpr(parsed->callee));
    return BuildCallExpr(callee, args, parsed->loc);
}

/// Translate a parsed name to a reference to the declaration it references.
auto Sema::TranslateDeclRefExpr(ParsedDeclRefExpr* parsed, Type) -> Ptr<Stmt> {
    return BuildDeclRefExpr(parsed->names(), parsed->loc);
}

/// Perform initial processing of a decl so it can be used by the rest
/// of the code. This only handles order-independent decls.
auto Sema::TranslateDeclInitial(ParsedDecl* d) -> std::optional<Ptr<Decl>> {
    // Unwrap exports.
    if (auto exp = dyn_cast<ParsedExportDecl>(d)) {
        auto decl = TranslateDeclInitial(exp->decl);
        if (decl and decl->present()) {
            AddDeclToScope(&tu->exports, decl->get());

            // If this declaration has linkage, adjust it now.
            if (auto object = dyn_cast<ObjectDecl>(decl->get()))
                object->linkage //
                    = object->linkage == Linkage::Imported
                        ? Linkage::Reexported
                        : Linkage::Exported;
        }
        return decl;
    }

    // Build procedure type now so we can forward-reference it.
    if (auto proc = dyn_cast<ParsedProcDecl>(d)) return TranslateProcDeclInitial(proc);
    if (auto s = dyn_cast<ParsedStructDecl>(d)) return TranslateStructDeclInitial(s);
    return std::nullopt;
}

auto Sema::TranslateDeferStmt(ParsedDeferStmt* stmt, Type) -> Ptr<Stmt> {
    auto body = TRY(TranslateStmt(stmt->body));
    return new (*tu) DeferStmt(body, stmt->loc);
}

/// Translate the body of a declaration.
auto Sema::TranslateEntireDecl(Decl* d, ParsedDecl* parsed) -> Ptr<Decl> {
    // Unwrap exports; pass along the actual parsed decl for this.
    if (auto exp = dyn_cast<ParsedExportDecl>(parsed))
        return TranslateEntireDecl(d, exp->decl);

    // Ignore this if there was a problem w/ the procedure type. Also,
    // if this is a template, there is nothing more to be done here.
    if (auto proc = dyn_cast<ParsedProcDecl>(parsed)) {
        if (not d) return nullptr;
        if (isa<ProcTemplateDecl>(d)) return d;
        return TranslateProc(cast<ProcDecl>(d), proc->body, proc->params());
    }

    // Complete struct declarations.
    if (auto s = dyn_cast<ParsedStructDecl>(parsed)) {
        if (not d) return nullptr;
        return TranslateStruct(cast<TypeDecl>(d), s);
    }

    // No special handling for anything else.
    auto res = TranslateStmt(parsed);
    if (res.invalid()) return nullptr;
    return cast<Decl>(res.get());
}

auto Sema::TranslateExportDecl(ParsedExportDecl*, Type) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

auto Sema::TranslateEmptyStmt(ParsedEmptyStmt* parsed, Type) -> Ptr<Stmt> {
    return new (*tu) EmptyStmt(parsed->loc);
}

/// Like TranslateStmt(), but checks that the argument is an expression.
auto Sema::TranslateExpr(ParsedStmt* parsed, Type desired_type) -> Ptr<Expr> {
    auto stmt = TranslateStmt(parsed, desired_type);
    if (stmt.invalid()) return nullptr;
    if (not isa<Expr>(stmt.get())) return Error(parsed->loc, "Expected expression");
    return cast<Expr>(stmt.get());
}

auto Sema::TranslateEvalExpr(ParsedEvalExpr* parsed, Type) -> Ptr<Stmt> {
    EnterScope _{*this};
    auto arg = TRY(TranslateStmt(parsed->expr));
    return BuildEvalExpr(arg, parsed->loc);
}

auto Sema::TranslateFieldDecl(ParsedFieldDecl*, Type) -> Decl* {
    Unreachable("Handled as part of StructDecl translation");
}

auto Sema::TranslateForStmt(ParsedForStmt* parsed, Type) -> Ptr<Stmt> {
    EnterScope _{*this};

    // The number of variables must be less than or equal to the number of ranges.
    if (parsed->vars().size() > parsed->ranges().size()) return Error(
        parsed->loc,
        "'%1(for%)' loop declares more variables than it has ranges ({} vs {})",
        parsed->vars().size(),
        parsed->ranges().size()
    );

    // The types of the variables depend on the ranges, so translate them first.
    SmallVector<Expr*> ranges;
    for (auto r : parsed->ranges())
        if (auto e = TranslateExpr(r).get_or_null())
            ranges.push_back(e);

    // Give up if something went wrong.
    if (ranges.size() != parsed->ranges().size()) return {};

    // Make sure the ranges are something we can actually iterate over.
    for (auto& r : ranges) {
        if (isa<RangeType, SliceType>(r->type)) {
            r = LValueToRValue(r);
        } else if (isa<ArrayType>(r->type)) {
            r = MaterialiseTemporary(r);
        } else {
            return Error(
                r->location(),
                "Invalid type '{}' for range of '%1(for%)' loop",
                r->type
            );
        }
    }

    // Declare the enumerator variable if there is one.
    Ptr<LocalDecl> enum_var;
    if (parsed->has_enumerator()) {
        enum_var = MakeLocal(
            Type::IntTy,
            Expr::RValue,
            parsed->enum_name,
            parsed->enum_loc
        );
    }

    // Create the loop variables; they have no initialisers since they’re really
    // just values bound to the elements of the ranges.
    SmallVector<LocalDecl*> vars;
    for (auto [v, r] : zip(parsed->vars(), ranges)) {
        auto MakeVar = [&](Type ty, ValueCategory cat) {
            auto var = MakeLocal(
                ty,
                cat,
                v.first,
                v.second
            );

            vars.push_back(var);
        };

        r->type->visit(utils::Overloaded{
            [&](auto*) { Unreachable(); },
            [&](ArrayType* ty) { MakeVar(ty->elem(), Expr::LValue); },
            [&](SliceType* ty) { MakeVar(ty->elem(), Expr::LValue); },
            [&](RangeType* ty) { MakeVar(ty->elem(), Expr::RValue); },
        });
    }

    // Now that we have the variables, translate the loop body.
    auto body = TRY(TranslateStmt(parsed->body));
    return ForStmt::Create(*tu, enum_var, vars, ranges, body, parsed->loc);
}

auto Sema::TranslateIfExpr(ParsedIfExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    EnterScope _{*this, not parsed->is_static};
    auto cond = TRY(TranslateExpr(parsed->cond));
    if (parsed->is_static) return BuildStaticIfExpr(cond, parsed->then, parsed->else_, parsed->loc);
    auto then = TRY(TranslateStmt(parsed->then, desired_type));
    Ptr else_ = parsed->else_ ? TRY(TranslateStmt(parsed->else_.get(), desired_type)) : nullptr;
    return BuildIfExpr(cond, then, else_, parsed->loc);
}

auto Sema::TranslateIntLitExpr(ParsedIntLitExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    // Determine the type of this.
    //
    // If we have a desired type, and the value fits in that type,
    // then the type of the literal is that type. Otherwise, if the
    // value fits in an 'int', its type is 'int'. If not, the type
    // is the smallest power of two large enough to store the value.
    Type ty;
    auto val = parsed->storage.value();
    if (desired_type and desired_type->is_integer() and IntegerLiteralFitsInType(val, desired_type, false)) {
        ty = desired_type;
    } else if (IntegerLiteralFitsInType(val, Type::IntTy, false)) {
        ty = Type::IntTy;
    } else if (auto bits = Size::Bits(llvm::PowerOf2Ceil(val.getActiveBits())); bits <= IntType::MaxBits) {
        ty = IntType::Get(*tu, bits);
    } else {
        // Print and colour the type names manually here since we can’t
        // even create a type this large properly...
        Error(parsed->loc, "Sorry, we can’t compile a number that big :(");
        Note(
            parsed->loc,
            "The maximum supported integer type is {}, "
            "which is smaller than an %6(i{:i}%), which would "
            "be required to store a value of {}",
            IntType::Get(*tu, IntType::MaxBits),
            bits,
            parsed->storage.str(false) // Parsed literals are unsigned.
        );
        return nullptr;
    }

    auto desired_bits = ty->bit_width(*tu);
    auto stored_bits = Size::Bits(val.getBitWidth());
    auto storage = desired_bits == stored_bits
        ? parsed->storage
        : tu->store_int(val.zextOrTrunc(unsigned(desired_bits.bits())));

    return new (*tu) IntLitExpr(
        ty,
        storage,
        parsed->loc
    );
}

auto Sema::TranslateLoopExpr(ParsedLoopExpr* parsed, Type) -> Ptr<Stmt> {
    EnterScope _{*this};
    Ptr<Stmt> body;
    if (auto b = parsed->body.get_or_null()) body = TRY(TranslateStmt(b));
    return new (*tu) LoopExpr(body, parsed->loc);
}

auto Sema::TranslateMatchExpr(ParsedMatchExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    EnterScope _{*this};
    SmallVector<MatchCase> cases;
    bool ok = true;

    // Give up here if translating the controlling expression fails since
    // the semantics of a 'match' w/ and w/o a controlling expression are
    // completely different.
    Ptr<Expr> control_expr = parsed->control_expr()
        ? TRY(TranslateExpr(parsed->control_expr().get()))
        : nullptr;

    // Translate the type if there is one.
    auto ty = parsed->declared_type()
        ? TranslateType(parsed->declared_type().get(), Type::VoidTy)
        : Type::DeducedTy;

    for (auto c : parsed->cases()) {
        auto body = TranslateStmt(c.body, desired_type);
        if (not body) ok = false;

        // The wildcard pattern isn’t an expression and requires special handling.
        if (
            auto dre = dyn_cast<ParsedDeclRefExpr>(c.cond);
            dre and
            dre->names().size() == 1 and
            dre->names().front().str() == "_"
        ) {
            if (body) cases.emplace_back(MatchCase::Pattern::Wildcard(), body.get(), dre->loc);
            continue;
        }

        auto cond = TranslateExpr(c.cond);
        if (cond and body) {
            cases.emplace_back(
                cond.get(),
                body.get(),
                c.cond->loc
            );
        }
    }

    if (not ok) return {};
    return BuildMatchExpr(control_expr, ty, cases, parsed->loc);
}

auto Sema::TranslateMemberExpr(ParsedMemberExpr* parsed, Type) -> Ptr<Stmt> {
    auto base = TRY(TranslateExpr(parsed->base));

    // Struct member access.
    if (auto s = dyn_cast<StructType>(base->type.ptr())) {
        if (not s->is_complete()) return Error(
            parsed->loc,
            "Member access on incomplete type '{}'",
            base->type
        );

        base = MaterialiseTemporary(base);
        auto field = LookUpUnqualifiedName(s->scope(), parsed->member, true);
        switch (field.result) {
            using enum LookupResult::Reason;
            case Success: break;
            case Ambiguous:
            case FailedToImport:
            case NonScopeInPath:
                Unreachable();

            case NotFound: return Error(
                parsed->loc,
                "Struct '{}' has no member named '{}'",
                base->type,
                parsed->member
            );
        }

        return new (*tu) MemberAccessExpr(
            base,
            cast<FieldDecl>(field.decls.front()),
            parsed->loc
        );
    }

    // Member access on builtin types.
    using AK = BuiltinMemberAccessExpr::AccessKind;
    static constexpr auto AlreadyDiagnosed = AK(255);
    auto kind = [&] -> Opt<AK> {
        using Switch = llvm::StringSwitch<Opt<AK>>;
        if (auto te = dyn_cast<TypeExpr>(base)) {
            auto is_int = te->value->is_integer();
            return Switch(parsed->member)
                .Case("align", AK::TypeAlign)
                .Case("arrsize", AK::TypeArraySize)
                .Case("bits", AK::TypeBits)
                .Case("bytes", AK::TypeBytes)
                .Case("name", AK::TypeName)
                .Case("size", AK::TypeBytes)
                .Case("min", is_int ? Opt<AK>(AK::TypeMinVal) : std::nullopt)
                .Case("max", is_int ? Opt<AK>(AK::TypeMaxVal) : std::nullopt)
                .Default(std::nullopt);
        }

        if (isa<SliceType>(base->type)) return Switch(parsed->member)
            .Case("data", AK::SliceData)
            .Case("size", AK::SliceSize)
            .Default(std::nullopt);

        if (isa<RangeType>(base->type)) return Switch(parsed->member)
            .Case("start", AK::RangeStart)
            .Case("end", AK::RangeEnd)
            .Default(std::nullopt);

        Error(parsed->loc, "Cannot perform member access on type '{}'", base->type);
        return AlreadyDiagnosed;
    }();

    if (kind == AlreadyDiagnosed) return {};
    if (kind == std::nullopt) return Error(
        parsed->loc,
        "'{}' has no member named '{}'",
        base->type,
        parsed->member
    );

    return BuildBuiltinMemberAccessExpr(kind.value(), base, parsed->loc);
}

auto Sema::TranslateParenExpr(ParsedParenExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    return new (*tu) ParenExpr(TRY(TranslateExpr(parsed->inner, desired_type)), parsed->loc);
}

auto Sema::TranslateTupleExpr(ParsedTupleExpr* parsed, Type) -> Ptr<Stmt> {
    SmallVector<Expr*> exprs;
    SmallVector<Type> types;
    bool ok = true;
    for (auto pe : parsed->exprs()) {
        if (auto e = TranslateExpr(pe).get_or_null()) {
            exprs.push_back(e);
            types.push_back(e->type);
            if (not CheckFieldType(e->type, pe->loc)) ok = false;
        } else {
            ok = false;
        }
    }

    if (not ok) return {};
    auto tt = TupleType::Get(*tu, types);
    return TupleExpr::Create(*tu, tt, exprs, parsed->loc);
}

auto Sema::TranslateVarDecl(ParsedVarDecl* parsed, Type) -> Decl* {
    if (parsed->is_static) Todo();
    auto decl = MakeLocal(
        TranslateType(parsed->type),
        Expr::LValue,
        parsed->name.str(),
        parsed->loc
    );

    // Don’t even bother with the initialiser if the type is ill-formed.
    if (not decl->type) {
        decl->type = Type::VoidTy;
        return decl->set_invalid();
    }

    // Translate the initialiser.
    Ptr<Expr> init;
    bool init_valid = true;
    if (auto val = parsed->init.get_or_null()) {
        init = TranslateExpr(val, decl->type != Type::DeducedTy ? decl->type : Type());
        init_valid = init.present();

        // If the initialiser is invalid, we can get bogus errors
        // if the variable type is deduced, so give up in that case.
        if (not init and decl->type == Type::DeducedTy)
            return decl->set_invalid();
    }

    // Deduce the type from the initialiser, if need be.
    if (decl->type == Type::DeducedTy) {
        if (init) decl->type = init.get()->type;
        else {
            Error(decl->location(), "Type inference requires an initialiser");
            decl->type = Type::VoidTy;
            return decl->set_invalid();
        }
    }

    // Now that the type has been deduced (if necessary), we can check
    // if we can even create a variable of this type.
    if (not CheckVariableType(decl->type, decl->location()))
        return decl->set_invalid();

    // Then, perform initialisation.
    //
    // If this fails, the initialiser is simply discarded; we can
    // still continue analysing this though as most of sema doesn’t
    // care about variable initialisers.
    //
    // Skip this if there was an error in the initialiser.
    if (init_valid) decl->set_init(BuildInitialiser(
        decl->type,
        init ? init.get() : ArrayRef<Expr*>{},
        init ? init.get()->location() : decl->location()
    ));
    return decl;
}

auto Sema::TranslateProc(
    ProcDecl* decl,
    Ptr<ParsedStmt> body,
    ArrayRef<ParsedVarDecl*> decls
) -> ProcDecl* {
    if (not body) return decl;

    // Translate the body.
    EnterProcedure _{*this, decl};
    auto res = TranslateProcBody(decl, body.get(), decls);

    // If there was an error, mark the procedure as errored.
    if (res.invalid()) decl->set_invalid();
    else decl->finalise(res, curr_proc().locals);
    return decl;
}

auto Sema::TranslateProcBody(
    ProcDecl* decl,
    ParsedStmt* parsed_body,
    ArrayRef<ParsedVarDecl*> decls
) -> Ptr<Stmt> {
    Assert(parsed_body);

    // Translate parameters.
    auto ty = decl->proc_type();
    for (auto [i, pair] : enumerate(zip(ty->params(), decls))) {
        auto [param_info, parsed_decl] = pair;
        BuildParamDecl(
            curr_proc(),
            &param_info,
            u32(i),
            false,
            parsed_decl->name.str(),
            parsed_decl->loc
        );
    }

    // Translate body.
    auto ret = decl->return_type();
    auto body = TranslateExpr(parsed_body, ret != Type::DeducedTy ? ret : Type());
    if (body.invalid()) {
        // If we’re attempting to deduce the return type of this procedure,
        // but the body contains an error, just set it to void.
        if (ret == Type::DeducedTy) {
            decl->type = ProcType::AdjustRet(*tu, decl->proc_type(), Type::VoidTy);
            decl->set_invalid();
        }
        return nullptr;
    }

    return BuildProcBody(decl, body.get());
}

auto Sema::TranslateProcDecl(ParsedProcDecl*, Type) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

/// Perform initial type checking on a procedure, enough to enable calls
/// to it to be translated, but without touching its body, if there is one.
auto Sema::TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<Decl> {
    // Diagnose invalid combinations of attributes.
    auto attrs = parsed->type->attrs;
    if (attrs.native and attrs.nomangle) Error(
        parsed->loc,
        "'%1(native%)' procedures should not be declared '%1(nomangle%)'"
    );

    // Variadic templates cannot have C varargs.
    bool has_variadic_param = parsed->type->has_variadic_param();
    if (has_variadic_param and attrs.c_varargs) {
        attrs.c_varargs = false;
        Error(parsed->loc, "Variadic function cannot be '%1(varargs%)'");
    }

    // Check if this is a template.
    auto IsTemplate = [&] {
        auto it = parsed_template_deduction_infos.find(parsed);
        if (it != parsed_template_deduction_infos.end()) return true;

        // A function with a 'var' parameter is also a template.
        return rgs::any_of(parsed->type->param_types(), [&](auto& p) {
            return IsBuiltinVarType(p.type);
        });
    };

    // If this is a template, we can’t do much right now.
    Decl* decl{};
    if (IsTemplate()) {
        decl = ProcTemplateDecl::Create(
            *tu,
            parsed,
            curr_proc().proc,
            has_variadic_param
        );
    }

    // Otherwise, convert its signature now.
    else {
        // TODO: Check for redeclaration here. Codegen will crash horribly if
        // there are two procedures w/ the same name.
        EnterScope scope{*this, ScopeKind::Procedure};
        auto type = TranslateProcType(parsed->type);
        auto ty = cast_if_present<ProcType>(type);
        if (not ty) ty = ProcType::Get(*tu, Type::VoidTy);
        decl = BuildProcDeclInitial(scope.get(), ty, parsed->name, parsed->loc, parsed->type->attrs);
        if (not type) decl->set_invalid();
    }

    // Currently, only the last parameter is allowed to be variadic.
    //
    // NOTE: If we ever change that (for instance, once we support
    // named parameters), then a number of places around overload
    // resolution and building calls need to be updated to handle
    // this (since they all assume that, if there is a variadic
    // parameter, it is the last parameter).
    //
    // In that case, we would probably want to handle positional and
    // named arguments separately.
    if (has_variadic_param) {
        auto params_except_last = parsed->type->param_types().drop_back();
        auto it = rgs::find_if(params_except_last, &ParsedParameter::variadic);
        if (it != params_except_last.end()) {
            decl->set_invalid();
            Error(
                parsed->params()[usz(it - params_except_last.begin())]->loc,
                "Only the last parameter can be variadic"
            );
        }
    }

    // Variadic parameters may not be 'inout' or 'out' since passing
    // things by reference gets complicated if they’re somehow also
    // supposed to be inside a slice or tuple.
    for (auto p : parsed->type->param_types()) {
        if (p.variadic and (p.intent == Intent::Inout or p.intent == Intent::Out)) {
            Error(p.type->loc, "Variadic parameter cannot have intent '%1({}%)'", p.intent);
            decl->set_invalid();
        }
    }

    AddDeclToScope(curr_scope(), decl);
    return decl;
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedStmt* parsed, Type desired_type) -> Ptr<Stmt> {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
#define PARSE_TREE_LEAF_TYPE(node)             \
    case K::node: {                            \
        auto ty = TranslateType(parsed);       \
        if (not ty) return {};                 \
        return BuildTypeExpr(ty, parsed->loc); \
    }
#define PARSE_TREE_LEAF_NODE(node) \
    case K::node: return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed), desired_type);
#include "srcc/ParseTree.inc"
    }

    Unreachable("Invalid parsed statement kind: {}", +parsed->kind());
}

auto Sema::TranslateStructDecl(ParsedStructDecl*, Type) -> Decl* {
    Unreachable("Should not be translated normally");
}

auto Sema::TranslateStruct(TypeDecl* decl, ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto s = cast<StructType>(decl->type);
    Assert(not s->is_complete(), "Type is already complete?");

    // Translate the fields and build the layout.
    EnterScope _{*this, s->scope()};
    RecordLayout::Builder lb{*tu};
    for (auto f : parsed->fields()) {
        auto ty = TranslateType(f->type);
        if (not CheckFieldType(ty, f->loc)) {
            // If the field’s type is invalid, we can’t query any of its
            // properties, so just insert a dummy field and continue.
            lb.add_field(Type::VoidTy, f->name.str(), f->loc)->set_invalid();
        } else {
            AddDeclToScope(s->scope(), lb.add_field(ty, f->name.str(), f->loc));
        }
    }

    s->finalise(lb.build());
    return decl;
}

auto Sema::TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto sc = tu->create_scope<StructScope>(curr_scope());
    auto ty = StructType::Create(
        *tu,
        sc,
        parsed->name.str(),
        parsed->loc
    );

    AddDeclToScope(curr_scope(), ty->decl());
    return ty->decl();
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed, Type) -> Ptr<Stmt> {
    return StrLitExpr::Create(*tu, parsed->value, parsed->loc);
}

/// Translate a return expression.
auto Sema::TranslateReturnExpr(ParsedReturnExpr* parsed, Type) -> Ptr<Stmt> {
    Ptr<Expr> ret_val;
    if (parsed->value.present()) {
        ret_val = TranslateExpr(
            parsed->value.get(),
            curr_proc().proc->return_type()
        );
    }
    return BuildReturnExpr(ret_val.get_or_null(), parsed->loc, false);
}

auto Sema::TranslateUnaryExpr(ParsedUnaryExpr* parsed, Type desired_type) -> Ptr<Stmt> {
    auto arg = TRY(TranslateExpr(parsed->arg, desired_type));
    return BuildUnaryExpr(parsed->op, arg, parsed->postfix, parsed->loc);
}

auto Sema::TranslateWhileStmt(ParsedWhileStmt* parsed, Type) -> Ptr<Stmt> {
    EnterScope _{*this};
    auto cond = TRY(TranslateExpr(parsed->cond));
    auto body = TRY(TranslateStmt(parsed->body));
    return BuildWhileStmt(cond, body, parsed->loc);
}

// ============================================================================
//  Translation of Types
// ============================================================================
auto Sema::BuildArrayType(TypeLoc base, Expr* size_expr) -> Type {
    auto size = tu->vm.eval(size_expr);
    if (not size) return Type();
    auto integer = size->dyn_cast<APInt>();

    // Check that the size is a 64-bit integer.
    if (not integer) return Error(
        size_expr->location(),
        "Array size must be an integer, but was '{}'",
        size->type()
    );

    if (not integer->isSingleWord()) return Error(
        size_expr->location(),
        "Array size must fit into a signed 64-bit integer"
    );

    auto v = integer->getSExtValue();
    return BuildArrayType(base, v, {base.loc, size_expr->location()});
}

auto Sema::BuildArrayType(TypeLoc base, i64 size, Location loc) -> Type {
    if (not CheckVariableType(base.ty, base.loc)) return Type();
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
    Location decl_loc
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

auto Sema::BuildSliceType(Type base, Location loc) -> Type {
    if (not CheckVariableType(base, loc)) return Type();
    return SliceType::Get(*tu, base);
}

auto Sema::BuildTupleType(ArrayRef<TypeLoc> types) -> Type {
    bool ok = true;
    for (auto [ty, loc] : types)
        if (not CheckFieldType(ty, loc))
            ok = false;

    if (not ok) return Type();
    return TupleType::Get(*tu, llvm::to_vector(vws::transform(types, &TypeLoc::ty)));
}

auto Sema::TranslateArrayType(ParsedBinaryExpr* parsed) -> Type {
    Assert(parsed->op == Tk::LBrack);
    auto elem = TranslateType(parsed->lhs);
    if (not elem) return Type();
    auto size = TRY(TranslateExpr(parsed->rhs));
    return BuildArrayType({elem, parsed->loc}, size);
}

auto Sema::TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type {
    return parsed->ty;
}

auto Sema::TranslateIntType(ParsedIntType* parsed) -> Type {
    if (parsed->bit_width > IntType::MaxBits) {
        Error(parsed->loc, "The maximum integer type is %6(i{:i}%)", IntType::MaxBits);
        return IntType::Get(*tu, IntType::MaxBits);
    }
    return IntType::Get(*tu, parsed->bit_width);
}

auto Sema::TranslateNamedType(ParsedDeclRefExpr* parsed) -> Type {
    auto res = LookUpName(curr_scope(), parsed->names(), parsed->loc);
    if (not res) return Type();

    // Template type.
    if (auto ttd = dyn_cast<TemplateTypeParamDecl>(res.decls.front()))
        return ttd->arg_type();

    // Type decl (struct or type alias).
    if (auto s = dyn_cast<TypeDecl>(res.decls.front()))
        return s->type;

    Error(parsed->loc, "'{}' does not name a type", utils::join(parsed->names(), "::"));
    Note(res.decls.front()->location(), "Declared here");
    return Type();
}

auto Sema::TranslateProcType(ParsedProcType* parsed, ArrayRef<Type> deduced_var_parameters) -> Type {
    // Sanity check.
    //
    // We use u32s for indices here and there, so ensure that this is small
    // enough. For now, only allow up to 65535 parameters because that’s
    // more than anyone should need.
    if (parsed->param_types().size() > std::numeric_limits<u16>::max()) {
        Error(
            parsed->loc,
            "Sorry, that’s too many parameters (max is {})",
            std::numeric_limits<u16>::max()
        );
        return Type();
    }

    SmallVector<ParamTypeData, 10> params;
    bool ok = true;
    u32 var_params = 0;
    for (auto a : parsed->param_types()) {
        auto ty = TranslateType(a.type);

        // If this parameter’s type is 'var', then substitute whatever we
        // deduced for it.
        bool is_var_param = ty == Type::DeducedTy;
        if (is_var_param) {
            Assert(var_params < deduced_var_parameters.size());
            ty = deduced_var_parameters[var_params++];
        }

        // Check this here, but don’t do anything about it; this is only
        // harmful if we emit LLVM IR for it, and we won’t be getting there
        // anyway because of this error.
        //
        // see DeferredNativeProcArgOrReturn.
        if (ty and parsed->attrs.native and IsZeroSizedOrIncomplete(ty))
            DiagnoseZeroSizedTypeInNativeProc(ty, a.type->loc, false);

        // If this is a variadic parameter, convert it to a slice.
        //
        // Do *not* do this if this is a 'var...' parameter since we pass
        // those as a tuple; this will have already been handled during
        // substitution.
        if (a.variadic and not is_var_param) ty = BuildSliceType(ty, a.type->loc);
        if (not CheckVariableType(ty, a.type->loc)) ok = false;
        params.emplace_back(a.intent, ty, a.variadic);
    }

    auto ret = TranslateType(parsed->ret_type);
    if (not ret) ret = Type::VoidTy;
    else if (
        parsed->attrs.native and
        ret != Type::VoidTy and
        ret != Type::NoReturnTy and
        IsZeroSizedOrIncomplete(ret)
    ) DiagnoseZeroSizedTypeInNativeProc(ret, parsed->ret_type->loc, true);

    if (not ok) return Type();
    return ProcType::Get(
        *tu,
        ret,
        params,
        parsed->attrs.native ? CallingConvention::Native : CallingConvention::Source,
        parsed->attrs.c_varargs
    );
}

auto Sema::TranslatePtrType(ParsedPtrType* stmt) -> Type {
    auto ty = TranslateType(stmt->elem);
    if (not CheckVariableType(ty, stmt->loc)) return Type();
    return PtrType::Get(*tu, ty);
}

auto Sema::TranslateRangeType(ParsedRangeType* parsed) -> Type {
    auto ty = TranslateType(parsed->elem);
    if (not ty) return Type();

    // Only ranges of integers are supported.
    if (not ty->is_integer()) {
        Error(
            parsed->loc,
            "Range element type must be an integer, but was '%1({}%)'",
            ty
        );
        return Type();
    }

    return RangeType::Get(*tu, ty);
}

auto Sema::TranslateSliceType(ParsedSliceType* parsed) -> Type {
    auto ty = TranslateType(parsed->elem);
    return BuildSliceType(ty, parsed->loc);
}

auto Sema::TranslateTemplateType(ParsedTemplateType* parsed) -> Type {
    auto res = LookUpUnqualifiedName(curr_scope(), parsed->name, true);
    if (not res) Error(parsed->loc, "Deduced template type cannot occur here");
    auto ty = cast<TemplateTypeParamDecl>(res.decls.front());
    if (not ty->in_substitution) Error(parsed->loc, "Deduced template type cannot occur here");
    return ty->arg_type();
}

auto Sema::TranslateType(ParsedStmt* parsed, Type fallback) -> Type {
    Type t;
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
        case K::BuiltinType: t = TranslateBuiltinType(cast<ParsedBuiltinType>(parsed)); break;
        case K::DeclRefExpr: t = TranslateNamedType(cast<ParsedDeclRefExpr>(parsed)); break;
        case K::IntType: t = TranslateIntType(cast<ParsedIntType>(parsed)); break;
        case K::ProcType: t = TranslateProcType(cast<ParsedProcType>(parsed)); break;
        case K::PtrType: t = TranslatePtrType(cast<ParsedPtrType>(parsed)); break;
        case K::RangeType: t = TranslateRangeType(cast<ParsedRangeType>(parsed)); break;
        case K::SliceType: t = TranslateSliceType(cast<ParsedSliceType>(parsed)); break;
        case K::TemplateType: t = TranslateTemplateType(cast<ParsedTemplateType>(parsed)); break;
        case K::ParenExpr: t = TranslateType(cast<ParsedParenExpr>(parsed)->inner); break;

        // Array types are parsed as subscript expressions.
        case K::BinaryExpr: {
            auto b = cast<ParsedBinaryExpr>(parsed);
            if (b->op != Tk::LBrack) goto default_;
            t = TranslateArrayType(b);
        } break;

        // Tuples can be treated as types.
        case K::TupleExpr: {
            SmallVector<TypeLoc> types;
            auto t = cast<ParsedTupleExpr>(parsed);
            bool ok = true;
            for (auto e : t->exprs()) {
                if (auto ty = TranslateType(e)) types.emplace_back(ty, e->loc);
                else ok = false;
            }

            if (ok) return BuildTupleType(types);
        } break;

        default:
        default_:
            Error(parsed->loc, "Expected type");
            break;
    }

    if (not t) t = fallback;
    return t;
}
