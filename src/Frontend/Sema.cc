#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Alignment.h>

#include <print>
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

auto Sema::ApplyConversion(Expr* e, Conversion conv) -> Expr* {
    switch (conv.kind) {
        using K = Conversion::Kind;
        case K::LValueToSRValue: return LValueToSRValue(e);
        case K::IntegralCast: return new (*M) CastExpr(conv.ty, CastExpr::Integral, e, e->location(), true);
        case K::SelectOverload: {
            auto proc = cast<OverloadSetExpr>(e)->overloads()[conv.index];
            return CreateReference(proc, e->location()).get();
        }
    }
    Unreachable("Invalid conversion");
}

auto Sema::ApplyConversionSequence(Expr* e, ConversionSequence& seq) -> Expr* {
    for (auto& c : seq) e = ApplyConversion(e, c);
    return e;
}

Type Sema::AdjustVariableType(Type ty, Location loc) {
    // Any places that want to do type deduction need to take
    // care of it *before* this is called.
    if (ty == Type::DeducedTy) {
        Error(loc, "Type deduction is not allowed here");
        return Type();
    }

    if (ty == Type::NoReturnTy or ty == Type::TypeTy) {
        Error(loc, "Cannot declare a variable of type '{}'", ty);
        return Type();
    }

    if (ty == Type::UnresolvedOverloadSetTy) {
        Error(loc, "Unresolved overload set in parameter declaration");
        return Type();
    }

    if (auto s = dyn_cast_if_present<StructType>(ty.ptr()); s and not s->is_complete()) {
        Error(loc, "Declaring a variable of type '{}' before it is complete", ty);
        Note(s->decl()->location(), "'{}' declared here", ty);
        return Type();
    }

    return ty;
}

auto Sema::CreateReference(Decl* d, Location loc) -> Ptr<Expr> {
    if (not d->valid()) return nullptr;
    switch (d->kind()) {
        default: return ICE(d->location(), "Cannot build a reference to this declaration yet");
        case Stmt::Kind::ProcDecl: return new (*M) ProcRefExpr(cast<ProcDecl>(d), loc);
        case Stmt::Kind::ProcTemplateDecl: return OverloadSetExpr::Create(*M, d, loc);
        case Stmt::Kind::TypeDecl: return new (*M) TypeExpr(cast<TypeDecl>(d)->type, loc);
        case Stmt::Kind::LocalDecl:
        case Stmt::Kind::ParamDecl:
            return new (*M) LocalRefExpr(cast<LocalDecl>(d), loc);
    }
}

void Sema::DeclareLocal(LocalDecl* d) {
    Assert(d->parent == curr_proc().proc, "Must EnterProcedure before adding a local variable");
    curr_proc().locals.push_back(d);
    AddDeclToScope(curr_scope(), d);
}

auto Sema::GetScopeFromDecl(Decl* d) -> Ptr<Scope> {
    switch (d->kind()) {
        default: return {};
        case Stmt::Kind::BlockExpr: return cast<BlockExpr>(d)->scope;
        case Stmt::Kind::ProcDecl: return cast<ProcDecl>(d)->scope;
    }
}

bool Sema::IntegerFitsInType(const APInt& i, Type ty) {
    if (not ty->is_integer()) return false;
    auto to_bits //
        = ty == Type::IntTy
            ? Type::IntTy->size(*M)
            : cast<IntType>(ty)->bit_width();
    return Size::Bits(i.getSignificantBits()) <= to_bits;
}

auto Sema::LookUpQualifiedName(Scope* in_scope, ArrayRef<String> names) -> LookupResult {
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
                auto it = M->imports.find(first);
                if (it == M->imports.end()) return res;
                if (isa<TranslationUnit*>(it->second.ptr())) Todo();

                // We found an imported C++ header; do a C++ lookup.
                auto hdr = dyn_cast<clang::ASTUnit*>(it->second.ptr());
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

auto Sema::LookUpUnqualifiedName(Scope* in_scope, String name, bool this_scope_only) -> LookupResult {
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

    return LookupResult::NotFound(name);
}

auto Sema::LookUpName(
    Scope* in_scope,
    ArrayRef<String> names,
    Location loc,
    bool complain
) -> LookupResult {
    auto res = names.size() == 1
                 ? LookUpUnqualifiedName(in_scope, names[0], false)
                 : LookUpQualifiedName(in_scope, names);
    if (not res.successful() and complain) ReportLookupFailure(res, loc);
    return res;
}

auto Sema::LValueToSRValue(Expr* expr) -> Expr* {
    if (expr->value_category == Expr::SRValue) return expr;
    Assert(expr->value_category == Expr::LValue);
    return new (*M) CastExpr(expr->type, CastExpr::LValueToSRValue, expr, expr->location(), true);
}

bool Sema::MakeSRValue(Type ty, Expr*& e, StringRef elem_name, StringRef op) {
    auto init = TryBuildInitialiser(ty, e);
    if (init.invalid()) {
        Error(
            e->location(),
            "{} of '%1({}%)' must be of type\f'{}', but was '{}'",
            elem_name,
            op,
            ty,
            e->type
        );
        return false;
    }

    // Make sure it’s an srvalue.
    e = LValueToSRValue(init.get());
    return true;
}

auto Sema::MaterialiseTemporary(Expr* expr) -> Expr* {
    if (expr->lvalue()) return expr;
    Todo();
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
//  Modules.
// ============================================================================
auto ModuleLoader::LoadModuleFromArchive(
    StringRef name,
    Location import_loc
) -> Opt<ImportHandle> {
    if (module_search_paths.empty()) {
        ICE(import_loc, "No module search path");
        return std::nullopt;
    }

    // Append extension.
    std::string filename{name.str()};
    if (not name.ends_with(".mod")) filename += ".mod";

    // Try to find the module in the search path.
    fs::Path path;
    for (auto& base : module_search_paths) {
        auto combined = fs::Path{base} / filename;
        if (fs::File::Exists(combined)) {
            path = std::move(combined);
            break;
        }
    }

    // Couldn’t find it :(.
    if (path.empty()) {
        Error(import_loc, "Could not find module '{}'", name);
        Remark("Search paths:\n  {}", utils::join(module_search_paths, "\n  "));
        return std::nullopt;
    }

    auto tu = TranslationUnit::Deserialise(ctx, name, path.string(), import_loc);
    if (not tu) return std::nullopt;
    return ImportHandle(std::move(tu.value()));
}

auto ModuleLoader::load(
    String logical_name,
    String linkage_name,
    Location import_loc,
    bool is_cxx_header
) -> Opt<ImportHandle> {
    if (auto it = modules.find(linkage_name); it != modules.end())
        return Opt<ImportHandle>{it->second.copy(logical_name, import_loc)};

    auto h //
        = is_cxx_header
            ? ImportCXXHeader(linkage_name, import_loc)
            : LoadModuleFromArchive(linkage_name, import_loc);

    if (not h) return std::nullopt;
    auto [it, _] = modules.try_emplace(linkage_name, std::move(h.value()));
    return it->second.copy(logical_name, import_loc);
}

// ============================================================================
//  Initialisation.
// ============================================================================
// Initialisation context that applies conversions and diagnoses
// failures to do so immediately.
class Sema::ImmediateInitContext {
    Sema& S;
    Type target_type;
    Location loc;

public:
    Expr* res;

    ImmediateInitContext(Sema& S, Expr* e, Type target_type, Location loc = {})
        : S{S},
          target_type{target_type},
          loc{loc == Location() ? e->location() : loc},
          res{e} {}

    void apply(Conversion c) { res = S.ApplyConversion(res, c); }

    auto location() const -> Location { return loc; }

    bool report_lvalue_intent_mismatch(Intent intent) {
        // If this is itself a parameter, issue a better error.
        if (auto dre = dyn_cast<LocalRefExpr>(res->strip_parens()); dre and isa<ParamDecl>(dre->decl)) {
            S.Error(
                res->location(),
                "Cannot pass parameter of intent %1({}%) to a parameter with intent %1({}%)",
                cast<ParamDecl>(dre->decl)->intent(),
                intent
            );
            S.Note(dre->decl->location(), "Parameter declared here");
        } else {
            S.Error(res->location(), "Cannot bind this expression to an %1({}%) parameter.", intent);
        }
        S.Remark("Try storing this in a variable first.");
        return false;
    }

    bool report_nested_resolution_failure() {
        Todo("Report this");
    }

    bool report_type_mismatch() {
        S.Error(
            res->location(),
            "Cannot convert expression of type '{}' to '{}'",
            res->type,
            target_type
        );
        return false;
    }

    bool report_same_type_lvalue_required(Intent intent) {
        S.Error(
            res->location(),
            "Cannot pass type {} to %1({}%) parameter of type {}",
            res->type,
            intent,
            target_type
        );
        return false;
    }

    auto result() -> Expr* { return res; }
};

// Initialisation context that converts conversion failures into
// making an overload not viable.
class Sema::OverloadInitContext {
    [[maybe_unused]] Sema& S;
    Candidate& c;
    u32 param_index;

public:
    OverloadInitContext(Sema& S, Candidate& c, u32 param_index)
        : S{S},
          c{c},
          param_index{param_index} {}

    void apply(Conversion conv) {
        auto& s = c.status.get<Candidate::Viable>();
        s.conversions.back().push_back(conv);

        // Don’t forget to increment the badness so we can actually rank them.
        switch (conv.kind) {
            using K = Conversion::Kind;

            // The simplest thing we can do. This is only here so we can
            // allow overloading on lvalue vs rvalue.
            case K::LValueToSRValue: s.badness++; break;

            // Other simple conversions have a badness score of 2.
            case K::IntegralCast: s.badness += 2; break;

            // This is essentially a no-op because we 'have' to select one
            // of them anyway, as we can't just pass an overload set around.
            case K::SelectOverload: break;
        }
    }

    bool report_lvalue_intent_mismatch(Intent) {
        c.status = Candidate::LValueIntentMismatch{u32(param_index)};
        return true; // Not a fatal error.
    }

    bool report_nested_resolution_failure() {
        c.status = Candidate::NestedResolutionFailure{u32(param_index)};
        return true; // Not a fatal error.
    }

    bool report_same_type_lvalue_required(Intent) {
        c.status = Candidate::SameTypeLValueRequired{u32(param_index)};
        return true; // Not a fatal error.
    }

    bool report_type_mismatch() {
        c.status = Candidate::TypeMismatch{u32(param_index)};
        return true; // Not a fatal error.
    }
};

// Init context that doesn’t report a diagnostic when initialisation fails.
class Sema::TentativeInitContext : public ImmediateInitContext {
public:
    TentativeInitContext(Sema& S, Expr* e, Type target_type)
        : ImmediateInitContext{S, e, target_type, e->location()} {}

    bool report_lvalue_intent_mismatch(Intent) { return false; }
    bool report_nested_resolution_failure() { return false; }
    bool report_type_mismatch() { return false; }
    bool report_same_type_lvalue_required(Intent) { return false; }
};

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
    StructType* s,
    ArrayRef<Expr*> args,
    Location loc
) -> Ptr<Expr> {
    // First case: option 1 or 2.
    if (not s->initialisers().empty()) {
        return ICE(loc, "TODO: Call struct initialiser");
    }

    // Second case: option 3. Option 2 is handled before we get here,
    // i.e. at this point, we know we’re building a literal initialiser.
    Assert(not args.empty(), "Should have called BuildDefaultInitialiser() instead");
    Assert(s->has_literal_init(), "Should have rejected before we ever get here");

    // For this initialiser, the number of arguments must match the number
    // of fields in the struct.
    if (s->fields().size() != args.size()) {
        Error(
            loc,
            "Struct '{}' has {} field{}, but got {} argument{}",
            Type{s},
            s->fields().size(),
            s->fields().size() == 1 ? "" : "s",
            args.size(),
            args.size() == 1 ? "" : "s"
        );

        Remark(
            "\vIf you want to be able to initialise the struct using fewer "
            "arguments, either define an initialisation procedure or provide "
            "default values for the remaining fields."
        );

        return {};
    }

    // Recursively build an initialiser for each element.
    SmallVector<Expr*> inits;
    for (auto [field, arg] : zip(s->fields(), args)) {
        auto init = BuildInitialiser(field->type, arg, arg->location());
        if (init) inits.push_back(init.get());
        else {
            Note(field->location(), "In initialiser for field '{}'", field->name);
            return {};
        }
    }

    return StructInitExpr::Create(*M, s, inits, loc);
}

template <typename InitContext>
bool Sema::BuildInitialiser(
    InitContext& init,
    Type var_type,
    Expr* a,
    Intent intent,
    CallingConvention cc,
    bool in_call
) {
    Assert(var_type, "Null type in initialisation?");
    Assert(a, "Initialiser must not be null");

    // If the intent resolves to pass by reference, then we
    // need to bind to it; the type must match exactly for
    // that.
    if (in_call and var_type->pass_by_lvalue(cc, intent)) {
        if (a->type != var_type) return init.report_same_type_lvalue_required(intent);
        if (not a->lvalue()) return init.report_lvalue_intent_mismatch(intent);
        return true;
    }

    // Type matches exactly.
    if (a->type == var_type) {
        // We’re passing by value. For srvalue types, convert lvalues
        // to srvalues here.
        if (a->type->rvalue_category() == Expr::SRValue) {
            if (a->lvalue()) init.apply(Conversion::LValueToSRValue());
            return true;
        }

        // Otherwise, we expect an mrvalue here.
        if (a->value_category == Expr::MRValue) return true;
        ICE(
            a->location(),
            "TODO: {} a struct by copy",
            in_call ? "Passing" : "Assigning"
        );
        return false;
    }

    // We need to perform conversion. What we do here depends on the type.
    switch (var_type->kind()) {
        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::SliceType:
        case TypeBase::Kind::ReferenceType:
            return init.report_type_mismatch();

        case TypeBase::Kind::ProcType:
            // Type is an overload set; attempt to convert it.
            //
            // This is *not* the same algorithm as overload resolution, because
            // the types must match exactly here, and we also need to check the
            // return type.
            if (a->type == Type::UnresolvedOverloadSetTy) {
                auto p_proc_type = dyn_cast<ProcType>(var_type.ptr());
                if (not p_proc_type) return init.report_type_mismatch();

                // Instantiate templates and simply match function types otherwise; we
                // don’t need to do anything fancier here.
                auto overloads = cast<OverloadSetExpr>(a->strip_parens())->overloads();

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
                        init.apply(Conversion::SelectOverload(u16(j)));
                        return true;
                    }
                }

                // Otherwise, we need to try and instantiate templates in this overload set.
                for (auto o : overloads) {
                    if (not isa<ProcTemplateDecl>(o)) continue;
                    Todo("Instantiate template in nested overload set");
                }

                // None of the overloads matched.
                return init.report_nested_resolution_failure();
            }

            // Otherwise, the types don’t match.
            return init.report_type_mismatch();

        // For integers, we can use the common type rule.
        case TypeBase::Kind::IntType: {
            // If this is a (possibly parenthesised and negated) integer
            // that fits in the type of the lhs, convert it. If it doesn’t
            // fit, the type must be larger, so give up.
            Expr* lit = a;
            for (;;) {
                lit = lit->strip_parens();
                auto u = dyn_cast<UnaryExpr>(lit);
                if (not u or u->op != Tk::Minus) break;
                lit = u->arg;
            }

            // If we ultimately found a literal, evaluate the original expression.
            if (isa_and_present<IntLitExpr>(lit)) {
                auto val = M->vm.eval(a, false);
                if (val and IntegerFitsInType(val->cast<APInt>(), var_type)) {
                    // Integer literals are srvalues so no need fo l2r conv here.
                    init.apply(Conversion::IntegralCast(var_type));
                    return true;
                }
            }

            // Otherwise, if both are sized integer types, and the initialiser
            // is smaller, we can convert it as well.
            auto ivar = cast<IntType>(var_type);
            auto iinit = dyn_cast<IntType>(a->type);
            if (not iinit or iinit->bit_width() > ivar->bit_width()) return init.report_type_mismatch();
            init.apply(Conversion::LValueToSRValue());
            init.apply(Conversion::IntegralCast(var_type));
            return true;
        }

        // For builtin types, it depends.
        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(var_type)->builtin_kind()) {
                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::Deduced:
                case BuiltinKind::NoReturn:
                    Unreachable("A variable of this type should not exist: {}", var_type);

                // The only type that can initialise these is the exact
                // same type, so complain (integer literals are not of
                // type 'int' iff the literal doesn’t fit in an 'int',
                // so don’t even bother trying to convert it).
                case BuiltinKind::Void:
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::Type:
                    return init.report_type_mismatch();
            }

            Unreachable();
        }

        case TypeBase::Kind::StructType: {
            // FIXME: HACK.
            //
            // TODO: Do we want to support implicit conversions to struct types
            // in calls and tentative conversion? Note that if the type of the
            // argument *is* a struct type, this is already supported by the
            // check for type identity above.
            if constexpr (utils::is<InitContext, ImmediateInitContext>) {
                auto result = BuildAggregateInitialiser(
                    cast<StructType>(var_type.ptr()),
                    a,
                    init.location()
                );

                if (not result) return false;
                init.res = result.get();
                return true;
            } else {
                return init.report_type_mismatch();
            }
        }
    }

    Unreachable();
}

auto Sema::BuildInitialiser(
    Type var_type,
    Expr* arg,
    Intent intent,
    CallingConvention cc,
    Location loc
) -> Ptr<Expr> {
    // Passing in 'arg' and 'var_type' twice here is unavoidable because
    // InitContexts generally don't store either; it’s just this one that
    // does...
    ImmediateInitContext init{*this, arg, var_type, loc};
    if (not BuildInitialiser(init, var_type, arg, intent, cc, true)) return nullptr;
    return init.result();
}

auto Sema::BuildInitialiser(Type var_type, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr> {
    var_type = AdjustVariableType(var_type, loc);
    if (not var_type) return nullptr;

    // Easy case: no arguments.
    if (args.empty()) {
        if (var_type->can_default_init()) return new (*M) DefaultInitExpr(var_type, loc);
        if (var_type->can_init_from_no_args()) return ICE(
            loc,
            "TODO: non-default empty initialisation of '{}'",
            var_type
        );

        return Error(loc, "Type '{}' has requires a non-empty initialiser", var_type);
    }

    // If there is exactly one argument, delegate to the rest of the
    // initialisation machinery.
    if (args.size() == 1) {
        ImmediateInitContext init{*this, args.front(), var_type, loc};
        if (not BuildInitialiser(init, var_type, args.front(), {}, {}, false)) return {};
        return init.result();
    }

    // There are only few (classes of) types that support initialisation
    // from more than one argument. We only support immediate initialisation
    // of these for now.
    if (isa<ArrayType>(var_type)) return ICE(loc, "TODO: Array initialiser");
    if (auto s = dyn_cast<StructType>(var_type.ptr())) return BuildAggregateInitialiser(s, args, loc);
    return Error(
        loc,
        "Cannot create a value of type '{}' from more than one argument",
        var_type
    );
}

auto Sema::TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr> {
    TentativeInitContext init{*this, arg, var_type};
    if (not BuildInitialiser(init, var_type, arg)) return nullptr;
    return init.result();
}

// ============================================================================
//  Templates.
// ============================================================================
auto Sema::DeduceType(
    ParsedStmt* parsed_type,
    u32 parsed_type_index,
    ArrayRef<TypeLoc> input_types
) -> Type {
    if (isa<ParsedTemplateType>(parsed_type))
        return input_types[parsed_type_index].ty;

    // TODO: Support more complicated deduction.
    return Type();
}

auto Sema::InstantiateTemplate(SubstitutionInfo& info, Location inst_loc) -> ProcDecl* {
    auto s = info.success();
    Assert(s, "Instantiating failed substitution?");
    if (s->instantiation) return s->instantiation;

    // Translate the declaration proper.
    s->instantiation = BuildProcDeclInitial(
        s->scope,
        s->type,
        info.pattern->name,
        info.pattern->location(),
        info.pattern->pattern->type->attrs
    );

    // Remember what pattern we were instantiated from.
    s->instantiation->instantiated_from = info.pattern;
    M->template_instantiations[info.pattern].push_back(s->instantiation);

    // Translate the body and record the instantiation.
    return TranslateProc(
        s->instantiation,
        info.pattern->pattern->body,
        info.pattern->pattern->params()
    );
}

auto Sema::SubstituteTemplate(
    ProcTemplateDecl* proc_template,
    ArrayRef<TypeLoc> input_types
) -> SubstitutionInfo& {
    auto params = proc_template->pattern->type->param_types();
    Assert(input_types.size() >= params.size(), "Not enough arguments");
    input_types = input_types.take_front(params.size());

    // Check if this has already been substituted.
    auto& substs = template_substitutions[proc_template];
    auto inst = find_if(substs, [&](auto& info) {
        return equal(info->input_types, input_types, [](Type a, TypeLoc b) {
            return a == b.ty;
        });
    });

    // We’ve substituted this template with these inputs before.
    if (inst != substs.end()) return *inst->get();

    // Otherwise, perform template deduction now.
    using Deduced = std::pair<u32, TypeLoc>;
    HashMap<String, Deduced> deduced;
    auto& info = *substs.emplace_back(
        std::make_unique<SubstitutionInfo>(
            proc_template,
            input_types | vws::transform(&TypeLoc::ty) | rgs::to<SmallVector<Type>>()
        )
    );

    // First, handle all deduction sites.
    auto& template_info = parsed_template_deduction_infos.at(proc_template->pattern);
    for (const auto& [name, indices] : template_info) {
        for (auto i : indices) {
            auto parsed = proc_template->pattern->type->param_types()[i];
            Type ty = DeduceType(parsed.type, i, input_types);
            if (not ty) {
                info.data = SubstitutionInfo::DeductionFailed{
                    M->save(name),
                    i
                };

                return info;
            }

            // If the type has not been deduced yet, remember it.
            auto [it, inserted] = deduced.try_emplace(name, Deduced{u32(i), {ty, parsed.type->loc}});
            if (inserted) continue;

            // Otherwise, check that the deduction result is the same.
            if (it->second.second.ty != ty) {
                info.data = SubstitutionInfo::DeductionAmbiguous{
                    name,
                    it->second.first,
                    u32(i),
                    it->second.second.ty,
                    ty,
                };

                return info;
            }
        }
    }

    // Create a scope for the procedure and save the template arguments there.
    EnterScope scope{*this, ScopeKind::Procedure};
    for (auto [name, d] : deduced)
        AddDeclToScope(scope.get(), new (*M) TemplateTypeParamDecl(name, d.second));

    // Now that that is done, we can convert the type properly.
    auto ty = TranslateType(proc_template->pattern->type);

    // Mark that we’re done substituting.
    for (auto d : scope.get()->decls())
        cast<TemplateTypeParamDecl>(d)->in_substitution = false;

    // Store the type for later if substitution succeeded.
    if (not ty) return info;
    info.data = SubstitutionInfo::Success{cast<ProcType>(ty), scope.get()};
    return info;
}

// ============================================================================
//  Overloading.
// ============================================================================
void Sema::ReportOverloadResolutionFailure(
    ArrayRef<Candidate> candidates,
    ArrayRef<Expr*> call_args,
    Location call_loc,
    u32 final_badness
) {
    /*auto FormatTempSubstFailure = [&](const SubstitutionInfo& ti, std::string& out, std::string_view indent) {
        ti.res.data.visit(utils::Overloaded{// clang-format off
            [](const TempSubstRes::Success&) { Unreachable("Invalid template even though substitution succeeded?"); },
            [](TempSubstRes::Error) { Unreachable("Should have bailed out earlier on hard error"); },
            [&](TempSubstRes::DeductionFailed f) {
                out += std::format(
                    "In param #{}: cannot deduce ${} in {} from {}",
                    f.param_index + 1,
                    f.ttd->name,
                    ti.pattern->param_types()[f.param_index].type,
                    call_args[f.param_index]->type
                );
            },

            [&](const TempSubstRes::DeductionAmbiguous& a) {
                out += std::format(
                    "Template deduction mismatch for parameter %3(${}%):\n"
                    "{}#{}: Deduced as {}\n"
                    "{}#{}: Deduced as {}",
                    a.ttd->name,
                    indent,
                    a.first,
                    a.first_type,
                    indent,
                    a.second,
                    a.second_type
                );
            }
        }); // clang-format on
    };*/

    /*
    // If there is only one overload, print the failure reason for
    // it and leave it at that.
    if (candidates.size() == 1) {
        auto c = candidates.front();
        auto ty = c.type();
        auto Ctx = [&](usz idx) {
            return ImmediateInitContext{
                *this,
                call_args[idx],
                ty->params()[idx].type,
            };
        };

        auto V = utils::Overloaded{// clang-format off
            [](const Candidate::Viable&) { Unreachable(); },
            [&](Candidate::ArgumentCountMismatch) {
                Error(
                    call_loc,
                    "Procedure '%2({}%)' expects {} argument{}, got {}",
                    c.proc->name,
                    ty->params().size(),
                    ty->params().size() == 1 ? "" : "s",
                    call_args.size()
                );
                Note(c.proc->location(), "Declared here");
            },

            [&](Candidate::InvalidTemplate) {
                Todo();
                /*std::string extra;
                FormatTempSubstFailure(GetTemplateSubstitutionInfo(c.proc), extra, "  ");
                Error(call_loc, "Template argument substitution failed");
                Remark("\r{}", extra);
                Note(c.location(), "Declared here");#1#
            },

            [&](Candidate::LValueIntentMismatch m) {
                Ctx(m.mismatch_index).report_lvalue_intent_mismatch(ty->params()[m.mismatch_index].intent);
            },

            [&](Candidate::NestedResolutionFailure) {
                Todo("Report this");
            },

            [&](Candidate::SameTypeLValueRequired m) {
                Ctx(m.mismatch_index).report_same_type_lvalue_required(ty->params()[m.mismatch_index].intent);
            },

            [&](Candidate::TypeMismatch m) {
                Error(
                    call_args[m.mismatch_index]->location(),
                    "Argument of type '{}' does not match expected type '{}'",
                    call_args[m.mismatch_index]->type,
                    ty->params()[m.mismatch_index].type
                );
                Note(c.param_loc(m.mismatch_index), "Parameter declared here");
            },

            [&](Candidate::UndeducedReturnType) {
                Error(call_loc, "Cannot call procedure before its return type has been deduced");
                Note(c.proc->location(), "Declared here");
                Remark("\rTry specifying the return type explicitly: '%1(->%) <type>'");
            }
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
        message += std::format(
            "  %b({}.%) \v{}",
            i + 1,
            c.proc->type
        );

        // And include the location if it is valid.
        auto loc = c.proc->location();
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

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        message += std::format("\n  %b({:>{}}.%) ", i + 1, width);
        auto V = utils::Overloaded{// clang-format off
            [&] (const Candidate::Viable& v) {
                // If the badness is equal to the final badness,
                // then this candidate was ambiguous. Otherwise,
                // another candidate was simply better.
                message += v.badness == final_badness
                    ? std::format("Ambiguous (matches as well as another candidate; score: {})"sv, v.badness)
                    : std::format("Not selected (another candidate matches better; score: {})"sv, v.badness);
            },

            [&](Candidate::ArgumentCountMismatch) {
                auto params = c.proc->proc_type()->params();
                message += std::format(
                    "Expected {} arg{}, got {}",
                    params.size(),
                    params.size() == 1 ? "" : "s",
                    call_args.size()
                );
            },

            [&](Candidate::InvalidTemplate) {
                Todo();
                /*message += "Template argument substitution failed";
                FormatTempSubstFailure(c.proc.get<Candidate::TemplateInfo>(), message, "        ");#1#
            },

            [&](Candidate::LValueIntentMismatch m) {
                auto& p = c.proc->proc_type()->params()[m.mismatch_index];
                message += std::format(
                    "Arg #{}: {} {} requires an lvalue of the same type",
                    m.mismatch_index + 1,
                    p.intent,
                    p.type
                );
            },

            [&](const Candidate::NestedResolutionFailure& n) {
                Todo("Report this");
            },

            [&](Candidate::TypeMismatch t) {
                message += std::format(
                    "Arg #{} should be '{}' but was '{}'",
                    t.mismatch_index + 1,
                    c.proc->proc_type()->params()[t.mismatch_index].type,
                    call_args[t.mismatch_index]->type
                );
            },

            [&](Candidate::SameTypeLValueRequired m) {
                auto& p = c.proc->proc_type()->params()[m.mismatch_index];
                message += std::format(
                    "Arg #{}: {} {} requires an lvalue of the same type",
                    m.mismatch_index + 1,
                    p.intent,
                    p.type
                );
            },

            [&](Candidate::UndeducedReturnType) {
                message += "Return type has not been deduced yet";
            }
        }; // clang-format on
        c.status.visit(V);
    }*/

    Error(call_loc, "Overload resolution failed");

    /*ctx.diags().report(Diagnostic{
        Diagnostic::Level::Error,
        call_loc,
        std::format("Overload resolution failed in call to\f'%2({}%)'", candidates.front().proc->name),
        std::move(message),
    });*/
}

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildAssertExpr(Expr* cond, Ptr<Expr> msg, Location loc) -> Ptr<AssertExpr> {
    if (not MakeSRValue(Type::BoolTy, cond, "Condition", "assert"))
        return {};

    // Message must be a string literal.
    // TODO: Allow other string-like expressions.
    if (auto m = msg.get_or_null(); m and not isa<StrLitExpr>(m)) return Error(
        m->location(),
        "Assertion message must be a string literal",
        m->type
    );

    return new (*M) AssertExpr(cond, std::move(msg), false, loc);
}

auto Sema::BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, Location loc) -> BlockExpr* {
    return BlockExpr::Create(
        *M,
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
) -> Ptr<BinaryExpr> {
    using enum ValueCategory;
    auto Build = [&](Type ty, ValueCategory cat = SRValue) {
        return new (*M) BinaryExpr(ty, cat, op, lhs, rhs, loc);
    };

    auto ConvertToCommonType = [&] {
        // Find the common type of the two. We need the same logic
        // during initialisation (and it actually turns out to be
        // easier to write it that way), so reuse it here.
        if (auto lhs_conv = TryBuildInitialiser(rhs->type, lhs).get_or_null()) {
            lhs = lhs_conv;
        } else if (auto rhs_conv = TryBuildInitialiser(lhs->type, rhs).get_or_null()) {
            rhs = rhs_conv;
        } else {
            Error(
                loc,
                "Invalid operation: %1({}%) between {} and {}",
                Spelling(op),
                lhs->type,
                rhs->type
            );
            return false;
        }

        // Now they’re the same type, so ensure both are srvalues.
        lhs = LValueToSRValue(lhs);
        rhs = LValueToSRValue(rhs);
        return true;
    };

    auto BuildArithmeticOrComparisonOperator = [&](bool comparison) -> Ptr<BinaryExpr> {
        auto Check = [&](std::string_view which, Expr* e) {
            if (e->type->is_integer()) return true;
            Error(e->location(), "{} of %1({}%) must be an integer", which, Spelling(op));
            return false;
        };

        // Both operands must be integers.
        if (not Check("Left operand", lhs) or not Check("Right operand", rhs)) return nullptr;
        if (not ConvertToCommonType()) return nullptr;
        return Build(comparison ? Type::BoolTy : lhs->type);
    };

    // Otherwise, each builtin operator needs custom handling.
    switch (op) {
        default: Unreachable("Invalid binary operator: {}", op);

        // Array or slice subscript.
        case Tk::LBrack: {
            if (not isa<SliceType, ArrayType>(lhs->type)) return Error(
                lhs->location(),
                "Cannot subscript non-array, non-slice type '{}'",
                lhs->type
            );

            if (not MakeSRValue(Type::IntTy, rhs, "Index", "[]")) return {};

            // Arrays need to be in memory before we can do anything
            // with them; slices are srvalues and should be loaded
            // whole.
            if (isa<ArrayType>(lhs->type)) lhs = MaterialiseTemporary(lhs);
            else lhs = LValueToSRValue(lhs);

            // A subscripting operation yields an lvalue.
            return Build(cast<SingleElementTypeBase>(lhs->type)->elem(), LValue);
        }

        // TODO: Allow for slices and arrays.
        case Tk::In: return ICE(loc, "Operator 'in' not yet implemented");

        // Arithmetic operation.
        case Tk::StarStar:
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
            return Build(Type::BoolTy);
        }

        // Logical operator.
        case Tk::And:
        case Tk::Or:
        case Tk::Xor: {
            if (not MakeSRValue(Type::BoolTy, lhs, "Left operand", Spelling(op))) return {};
            if (not MakeSRValue(Type::BoolTy, rhs, "Right operand", Spelling(op))) return {};
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
            // LHS must be an lvalue.
            if (lhs->value_category != LValue) return Error(
                lhs->location(),
                "Invalid target for assignment"
            );

            // Compound assignment.
            if (op != Tk::Assign) {
                // The right-hand side has to be convertible to the left, hand
                if (lhs->type != rhs->type) return Error(
                    loc,
                    "Cannot assign '{}' to '{}'",
                    rhs->type,
                    lhs->type
                );

                // The RHS an RValue.
                if (rhs->type->rvalue_category() == MRValue) return ICE(
                    rhs->location(),
                    "Sorry, assignment to a variable of type '{}' is not yet supported",
                    rhs->type
                );

                rhs = LValueToSRValue(rhs);

                // For arithmetic operations, both sides must be ints.
                if (lhs->type != Type::IntTy) Error(
                    lhs->location(),
                    "Compound assignment operator '{}' is not supported for type '{}'",
                    Spelling(op),
                    lhs->type
                );

                // The LHS is returned as an lvalue.
                return Build(lhs->type, LValue);
            }

            // Delegate to initialisation.
            if (auto init = BuildInitialiser(lhs->type, rhs, rhs->location()).get_or_null()) rhs = init;
            else return nullptr;
            return Build(lhs->type, LValue);
        }
    }
}

auto Sema::BuildBuiltinCallExpr(
    BuiltinCallExpr::Builtin builtin,
    ArrayRef<Expr*> args,
    Location call_loc
) -> Ptr<BuiltinCallExpr> {
    switch (builtin) {
        // __srcc_print takes a sequence of arguments and prints them all;
        // the arguments must be strings or integers.
        case BuiltinCallExpr::Builtin::Print: {
            SmallVector<Expr*> actual_args{args};
            if (args.empty()) return Error(call_loc, "__srcc_print takes at least one argument");
            for (auto& arg : actual_args) {
                if (
                    arg->type != M->StrLitTy and
                    arg->type != Type::IntTy and
                    arg->type != Type::BoolTy
                ) return Error( //
                    arg->location(),
                    "__srcc_print only accepts i8[] and integers, but got {}",
                    arg->type
                );
                arg = LValueToSRValue(arg);
            }
            return BuiltinCallExpr::Create(*M, builtin, Type::VoidTy, actual_args, call_loc);
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
            case AK::SliceData: return ReferenceType::Get(*M, cast<SliceType>(operand->type)->elem());
            case AK::SliceSize: return Type::IntTy;
            case AK::TypeAlign: return Type::IntTy;
            case AK::TypeArraySize: return Type::IntTy;
            case AK::TypeBits: return Type::IntTy;
            case AK::TypeBytes: return Type::IntTy;
            case AK::TypeName: return M->StrLitTy;
            case AK::TypeMaxVal: return cast<TypeExpr>(operand)->value;
            case AK::TypeMinVal: return cast<TypeExpr>(operand)->value;
        }
        Unreachable();
    }();

    return new (*M) BuiltinMemberAccessExpr{
        type,
        Expr::SRValue,
        operand,
        ak,
        loc
    };
}

auto Sema::BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr> {
    // Check variadic arguments.
    auto CheckVarargs = [&](SmallVectorImpl<Expr*>& actual_args, ArrayRef<Expr*> varargs) {
        bool ok = true;
        for (auto a : varargs) {
            // Codegen is not set up to handle variadic arguments that are larger than a word,
            // so reject these here. If you need one of those, then seriously, wtf are you doing.
            if (
                a->type == Type::IntTy or
                a->type == Type::BoolTy or
                a->type == Type::NoReturnTy or
                (isa<IntType>(a->type) and cast<IntType>(a->type)->bit_width() <= Size::Bits(64)) or
                isa<ReferenceType>(a->type)
            ) {
                actual_args.push_back(LValueToSRValue(a));
            } else {
                ok = false;
                Error(
                    a->location(),
                    "Passing a value of type '{}' as a varargs argument is not supported",
                    a->type
                );
            }
        }
        return ok;
    };

    // If this is not a procedure reference or overload set, then we don’t
    // need to perform overload resolution nor template instantiation, so
    // just typecheck the arguments directly.
    auto callee_no_parens = callee_expr->strip_parens();
    if (not isa<OverloadSetExpr, ProcRefExpr>(callee_no_parens)) {
        auto ty = dyn_cast<ProcType>(callee_expr->type.ptr());

        // If the type is 'type', then this is actually an initialiser call.
        if (callee_expr->type == Type::TypeTy) {
            auto type = M->vm.eval(callee_expr);
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

        // If does not have procedure type, then we can’t call it.
        if (not ty) return Error(
            callee_expr->location(),
            "Expression of type '{}' is not callable",
            callee_expr->type
        );

        // Check arg count.
        if (ty->params().size() != args.size()) {
            return Error(
                loc,
                "Procedure expects {} argument{}, got {}",
                ty->params().size(),
                ty->params().size() == 1 ? "" : "s",
                args.size()
            );
        }

        // Check each parameter.
        SmallVector<Expr*> actual_args;
        actual_args.reserve(args.size());
        for (auto [p, a] : zip(ty->params(), args)) {
            auto arg = TRY(BuildInitialiser(p.type, a, p.intent, ty->cconv()));
            actual_args.push_back(arg);
        }

        // And check variadic arguments.
        if (not CheckVarargs(actual_args, args.drop_front(ty->params().size())))
            return nullptr;

        // And create the call.
        return CallExpr::Create(
            *M,
            ty->ret(),
            LValueToSRValue(callee_expr),
            actual_args,
            loc
        );
    }

    // Otherwise, perform overload resolution and instantiation.
    //
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

    // Add a candidate to the overload set.
    auto AddCandidate = [&](Decl* proc) -> bool {
        if (not proc->valid())
            return false;

        // Candidate is a regular procedure.
        auto p = dyn_cast<ProcTemplateDecl>(proc);
        if (not p) {
            candidates.emplace_back(cast<ProcDecl>(proc));
            return true;
        }

        // Candidate is a template.
        SmallVector<TypeLoc, 6> types;
        for (auto arg : args) types.emplace_back(arg->type, arg->location());
        auto& subst = SubstituteTemplate(p, types);
        if (subst.data.is<SubstitutionInfo::Error>()) return false;
        auto c = candidates.emplace_back(&subst);
        if (not subst.success()) c.status = Candidate::InvalidTemplate{};
        return true;
    };

    // Collect all candidates.
    if (auto proc = dyn_cast<ProcRefExpr>(callee_no_parens)) {
        if (not AddCandidate(proc->decl)) return {};
    } else {
        auto os = cast<OverloadSetExpr>(callee_no_parens);
        if (not rgs::all_of(os->overloads(), AddCandidate)) return {};
    }

    // Check if a single candidate is viable. Returns false if there
    // is a fatal error that prevents overload resolution entirely.
    auto CheckCandidate = [&](Candidate& c) -> bool {
        auto ty = c.type();
        auto params = ty->params();

        // If the candidate’s return type is deduced, we’re trying to
        // call it before it has been fully analysed. Disallow this.
        if (ty->ret() == Type::DeducedTy) {
            c.status = Candidate::UndeducedReturnType{};
            return true;
        }

        // Argument count mismatch is not allowed, unless the
        // function is variadic.
        //
        // TODO: Default arguments.
        if (args.size() != params.size()) {
            if (args.size() < params.size() or not ty->variadic()) {
                c.status = Candidate::ArgumentCountMismatch{};
                return true;
            }
        }

        // Check that we can initialise each parameter with its
        // corresponding argument. Variadic arguments are checked
        // later when the call is built.
        for (auto [i, a] : enumerate(args.take_front(params.size()))) {
            // Candidate may have become invalid in the meantime.
            auto st = c.status.get_if<Candidate::Viable>();
            if (not st) break;

            // Check the next parameter.
            auto& p = params[i];
            st->conversions.emplace_back();
            OverloadInitContext init{*this, c, u32(i)};
            if (not BuildInitialiser(init, p.type, a, p.intent, ty->cconv(), true)) return false;
        }

        // No fatal error.
        return true;
    };

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
        if (not best) {
            best = &c;
            continue;
        }

        // We already have a candidate. If the badness of this
        // one is better, then it becomes the new best candidate.
        if (c.badness() < best.get()->badness()) {
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
        if (auto* temp = dyn_cast<SubstitutionInfo*>(c->candidate)) {
            auto inst = InstantiateTemplate(*temp, loc);
            if (not inst or not inst->is_valid) return nullptr;
            final_callee = inst;
        } else {
            final_callee = cast<ProcDecl*>(c->candidate);
        }

        // Now is the time to apply the argument conversions.
        SmallVector<Expr*> actual_args;
        actual_args.reserve(args.size());
        for (auto [i, conv] : enumerate(c->status.get<Candidate::Viable>().conversions))
            actual_args.emplace_back(ApplyConversionSequence(args[i], conv));

        // And check variadic arguments.
        if (not CheckVarargs(actual_args, args.drop_front(final_callee->proc_type()->params().size())))
            return nullptr;

        // Finally, create the call.
        return CallExpr::Create(
            *M,
            final_callee->return_type(),
            CreateReference(final_callee, callee_expr->location()).get(),
            actual_args,
            loc
        );
    }

    // Overload resolution failed. :(
    ReportOverloadResolutionFailure(candidates, args, loc, badness);
    return nullptr;
}

auto Sema::BuildEvalExpr(Stmt* arg, Location loc) -> Ptr<Expr> {
    // Always create an EvalExpr to represent this in the AST.
    auto eval = new (*M) EvalExpr(arg, loc);

    // If the expression is not dependent, evaluate it now.
    auto value = M->vm.eval(arg);
    if (not value.has_value()) return eval;

    // And cache the value for later.
    return new (*M) ConstExpr(*M, std::move(*value), loc, eval);
}

auto Sema::BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, Location loc) -> Ptr<IfExpr> {
    auto Build = [&](Type ty, ValueCategory val) {
        return new (*M) IfExpr(
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
    if (cond->type == Type::NoReturnTy) return Build(Type::NoReturnTy, Expr::SRValue);

    // Condition must be a bool.
    if (not MakeSRValue(Type::BoolTy, cond, "Condition", "if"))
        return {};

    // If there is no else branch, or if either branch is not an expression,
    // the type of the 'if' is 'void'.
    if (
        not else_ or
        not isa<Expr>(then) or
        not isa<Expr>(else_.get())
    ) return Build(Type::VoidTy, Expr::SRValue);

    // Otherwise, if either branch is dependent, we can’t determine the type yet.
    auto t = cast<Expr>(then);
    auto e = cast<Expr>(else_.get());

    // Next, if either branch is 'noreturn', the type of the 'if' is the type
    // of the other branch (unless both are noreturn, in which case the type
    // is just 'noreturn').
    if (t->type == Type::NoReturnTy or e->type == Type::NoReturnTy) {
        bool both = t->type == Type::NoReturnTy and e->type == Type::NoReturnTy;
        return Build(
            both                          ? Type::NoReturnTy
            : t->type == Type::NoReturnTy ? e->type
                                          : t->type,
            both                          ? Expr::SRValue
            : t->type == Type::NoReturnTy ? e->value_category
                                          : t->value_category
        );
    }

    // If both are lvalues of the same type, the result is an lvalue
    // of that type.
    if (
        t->type == e->type and
        t->value_category == Expr::LValue and
        e->value_category == Expr::LValue
    ) return Build(t->type, Expr::LValue);

    // Finally, if there is a common type, the result is an rvalue of
    // that type.
    // TODO: Actually implement calculating the common type; for now
    //       we just use 'void' if the types don’t match.
    auto common_ty = t->type == e->type ? t->type : Type::VoidTy;

    // We don’t convert lvalues to rvalues in this case because neither
    // side will actually be used if there is no common type.
    // TODO: If there is no common type, both sides need to be marked
    //       as discarded.
    if (common_ty == Type::VoidTy) return Build(Type::VoidTy, Expr::SRValue);

    // Permitting MRValues here is non-trivial.
    if (common_ty->rvalue_category() == Expr::MRValue) return ICE(
        loc,
        "Sorry, we don’t support returning a value of type '{}' from an '%1(if%)' expression yet.",
        common_ty
    );

    // Make sure both sides are rvalues.
    t = LValueToSRValue(t);
    e = LValueToSRValue(e);
    return new (*M) IfExpr(common_ty, Expr::SRValue, cond, t, e, false, loc);
}

auto Sema::BuildParamDecl(
    ProcScopeInfo& proc,
    const ParamTypeData* param,
    u32 index,
    bool with_param,
    String name,
    Location loc
) -> ParamDecl* {
    auto decl = new (*M) ParamDecl(param, name, proc.proc, index, with_param, loc);
    if (not param->type) decl->set_invalid();
    DeclareLocal(decl);
    return decl;
}

auto Sema::BuildProcDeclInitial(
    Scope* proc_scope,
    ProcType* ty,
    String name,
    Location loc,
    ParsedProcAttrs attrs
) -> ProcDecl* {
    // Create the declaration. A top-level procedure is not considered
    // 'nested' inside the initialiser procedure, which means that this
    // is only local if the procedure stack contains at least 3 entries
    // (the initialiser, our parent, and us).
    auto proc = ProcDecl::Create(
        *M,
        ty,
        name,
        attrs.extern_ ? Linkage::Imported : Linkage::Internal,
        attrs.nomangle or attrs.native ? Mangling::None : Mangling::Source,
        proc_stack.size() >= 3 ? proc_stack.back()->proc : nullptr,
        loc
    );

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
        return BlockExpr::Create(*M, nullptr, {body, ret}, body->location());
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
        proc->type = ProcType::AdjustRet(*M, proc_type, deduced);
    }

    // Or complain if the type doesn’t match.
    else {
        Type ret = value.invalid() ? Type::VoidTy : value.get()->type;
        if (ret != proc->return_type()) Error(
            loc,
            "Return type '{}' does not match procedure return type '{}'",
            ret,
            proc->return_type()
        );
    }

    // Perform any necessary conversions.
    if (auto val = value.get_or_null()) {
        if (val->type == Type::VoidTy) {
            // Nop.
        } else if (val->type == Type::IntTy or isa<IntType>(val->type)) {
            value = LValueToSRValue(val);
        } else {
            ICE(loc, "Cannot compile this return type yet: {}", val->type);
        }
    }

    return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildStaticIfExpr(
    Expr* cond,
    ParsedStmt* then,
    Ptr<ParsedStmt> else_,
    Location loc
) -> Ptr<Stmt> {
    // Otherwise, check this now.
    if (not MakeSRValue(Type::BoolTy, cond, "Condition", "static if")) return {};
    auto val = M->vm.eval(cond);
    if (not val) {
        Error(loc, "Condition of 'static if' must be a constant expression");
        return {};
    }

    // If there is no else clause, and the condition is false, return
    // an empty statement.
    auto cond_val = val->cast<bool>();
    if (not cond_val and not else_) return new (*M) ConstExpr(*M, {}, loc, nullptr);

    // Otherwise, translate the appropriate branch now, and throw
    // away the other one.
    return TranslateStmt(cond_val ? then : else_.get());
}

auto Sema::BuildTypeExpr(Type ty, Location loc) -> TypeExpr* {
    return new (*M) TypeExpr(ty, loc);
}

auto Sema::BuildUnaryExpr(Tk op, Expr* operand, bool postfix, Location loc) -> Ptr<UnaryExpr> {
    auto Build = [&](Type ty, ValueCategory cat) {
        return new (*M) UnaryExpr(ty, cat, op, operand, postfix, loc);
    };

    if (postfix) return ICE(loc, "Postfix unary operators are not yet implemented");

    // Handle prefix operators.
    switch (op) {
        default: Unreachable("Invalid unary operator: {}", op);

        // Boolean negation.
        case Tk::Not: {
            if (not MakeSRValue(Type::BoolTy, operand, "Operand", "not")) return {};
            return Build(Type::BoolTy, Expr::SRValue);
        }

        // Arithmetic operators.
        case Tk::Minus:
        case Tk::Plus:
        case Tk::Tilde: {
            if (not MakeSRValue(Type::IntTy, operand, "Operand", Spelling(op))) return {};
            return Build(Type::IntTy, Expr::SRValue);
        }

        // Increment and decrement.
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            // Operand must be an lvalue.
            if (operand->value_category != Expr::LValue) return Error(
                operand->location(),
                "Invalid operand for '{}'",
                Spelling(op)
            );

            // Operand must be an integer.
            if (operand->type != Type::IntTy) return Error(
                operand->location(),
                "Operand of '{}' must be an integer",
                Spelling(op)
            );

            // Result is an lvalue.
            return Build(Type::IntTy, Expr::LValue);
        }
    }
}

auto Sema::BuildWhileStmt(Expr* cond, Stmt* body, Location loc) -> Ptr<WhileStmt> {
    if (not MakeSRValue(Type::BoolTy, cond, "Condition", "while")) return {};
    return new (*M) WhileStmt(cond, body, loc);
}

// ============================================================================
//  Translation Driver
// ============================================================================
auto Sema::Translate(
    const LangOpts& opts,
    ArrayRef<ParsedModule::Ptr> modules,
    StringMap<ImportHandle> imported_modules
) -> TranslationUnit::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};
    S.M = TranslationUnit::Create(first->context(), opts, first->name, first->is_module);
    S.parsed_modules = modules;
    S.M->imports = std::move(imported_modules);
    S.Translate();
    return std::move(S.M);
}

void Sema::Translate() {
    // Initialise sema.
    scope_stack.push_back(M->create_scope(nullptr));

    // Take ownership of any resources of the parsed modules.
    for (auto& p : parsed_modules) {
        M->add_allocator(std::move(p->string_alloc));
        M->add_integer_storage(std::move(p->integers));
        for (auto& [decl, info] : p->template_deduction_infos)
            parsed_template_deduction_infos[decl] = std::move(info);
    }

    // Collect all statements and translate them.
    M->initialiser_proc->scope = global_scope();
    EnterProcedure _{*this, M->initialiser_proc};
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, Location{});

    // File scope block should never be dependent.
    M->initialiser_proc->finalise(BuildProcBody(M->initialiser_proc, M->file_scope_block), curr_proc().locals);
}

void Sema::TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed) {
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

        auto stmt = TranslateStmt(p);
        if (stmt.present()) stmts.push_back(stmt.get());
    }
}

// ============================================================================
//  Translation of Individual Statements
// ============================================================================
auto Sema::TranslateAssertExpr(ParsedAssertExpr* parsed) -> Ptr<Stmt> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    Ptr<Expr> msg;
    if (auto m = parsed->message.get_or_null()) msg = TRY(TranslateExpr(m));
    return BuildAssertExpr(cond, msg, parsed->loc);
}

auto Sema::TranslateBinaryExpr(ParsedBinaryExpr* expr) -> Ptr<Stmt> {
    // Translate LHS and RHS.
    auto lhs = TRY(TranslateExpr(expr->lhs));
    auto rhs = TRY(TranslateExpr(expr->rhs));
    return BuildBinaryExpr(expr->op, lhs, rhs, expr->loc);
}

auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed) -> Ptr<Stmt> {
    EnterScope scope{*this};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, parsed->stmts());
    return BuildBlockExpr(scope.get(), stmts, parsed->loc);
}

auto Sema::TranslateBoolLitExpr(ParsedBoolLitExpr* parsed) -> Ptr<Stmt> {
    return new (*M) BoolLitExpr(parsed->value, parsed->loc);
}

auto Sema::TranslateCallExpr(ParsedCallExpr* parsed) -> Ptr<Stmt> {
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
        auto bk = llvm::StringSwitch<std::optional<B>>(dre->names().front())
                      .Case("__srcc_print", B::Print)
                      .Default(std::nullopt);

        // We have a builtin!
        if (bk.has_value()) return BuildBuiltinCallExpr(*bk, args, parsed->loc);
    }

    // Translate callee.
    auto callee = TRY(TranslateExpr(parsed->callee));
    return BuildCallExpr(callee, args, parsed->loc);
}

/// Translate a parsed name to a reference to the declaration it references.
auto Sema::TranslateDeclRefExpr(ParsedDeclRefExpr* parsed) -> Ptr<Stmt> {
    auto res = LookUpName(curr_scope(), parsed->names(), parsed->loc, false);
    if (res.successful()) return CreateReference(res.decls.front(), parsed->loc);

    // Overload sets are ok here.
    // TODO: Validate overload set; i.e. that there are no two functions that
    // differ only in return type, or not at all. Also: don’t allow overloading
    // on intent (for now).
    if (
        res.result == LookupResult::Reason::Ambiguous and
        isa<ProcDecl>(res.decls.front())
    ) return OverloadSetExpr::Create(*M, res.decls, parsed->loc);

    ReportLookupFailure(res, parsed->loc);
    return {};
}

/// Perform initial processing of a decl so it can be used by the rest
/// of the code. This only handles order-independent decls.
auto Sema::TranslateDeclInitial(ParsedDecl* d) -> std::optional<Ptr<Decl>> {
    // Unwrap exports.
    if (auto exp = dyn_cast<ParsedExportDecl>(d)) {
        auto decl = TranslateDeclInitial(exp->decl);
        if (decl and decl->present()) {
            AddDeclToScope(&M->exports, decl->get());

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

auto Sema::TranslateExportDecl(ParsedExportDecl*) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

/// Like TranslateStmt(), but checks that the argument is an expression.
auto Sema::TranslateExpr(ParsedStmt* parsed) -> Ptr<Expr> {
    auto stmt = TranslateStmt(parsed);
    if (stmt.invalid()) return nullptr;
    if (not isa<Expr>(stmt.get())) return Error(parsed->loc, "Expected expression");
    return cast<Expr>(stmt.get());
}

auto Sema::TranslateEvalExpr(ParsedEvalExpr* parsed) -> Ptr<Stmt> {
    auto arg = TRY(TranslateStmt(parsed->expr));
    return BuildEvalExpr(arg, parsed->loc);
}

auto Sema::TranslateFieldDecl(ParsedFieldDecl*) -> Decl* {
    Unreachable("Handled as part of StructDecl translation");
}

auto Sema::TranslateIfExpr(ParsedIfExpr* parsed) -> Ptr<Stmt> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    if (parsed->is_static) return BuildStaticIfExpr(cond, parsed->then, parsed->else_, parsed->loc);
    auto then = TRY(TranslateStmt(parsed->then));
    Ptr else_ = parsed->else_ ? TRY(TranslateStmt(parsed->else_.get())) : nullptr;
    return BuildIfExpr(cond, then, else_, parsed->loc);
}

auto Sema::TranslateIntLitExpr(ParsedIntLitExpr* parsed) -> Ptr<Stmt> {
    // If the value fits in an 'int', its type is 'int'.
    auto val = parsed->storage.value();
    auto small = val.tryZExtValue();
    if (small.has_value()) return new (*M) IntLitExpr(
        Type::IntTy,
        M->store_int(APInt(u32(Type::IntTy->size(*M).bits()), u64(*small), false)),
        parsed->loc
    );

    // Otherwise, the type is the smallest power of two large enough
    // to store the value.
    auto bits = Size::Bits(llvm::PowerOf2Ceil(val.getActiveBits()));

    // Too big.
    if (bits > IntType::MaxBits) {
        // Print and colour the type names manually here since we can’t
        // even create a type this large properly...
        Error(parsed->loc, "Sorry, we can’t compile a number that big :(");
        Note(
            parsed->loc,
            "The maximum supported integer type is {}, "
            "which is smaller than an %6(i{:i}%), which would "
            "be required to store a value of {}",
            IntType::Get(*M, IntType::MaxBits),
            bits,
            parsed->storage.str(false) // Parsed literals are unsigned.
        );
        return nullptr;
    }

    return new (*M) IntLitExpr(
        IntType::Get(*M, bits),
        parsed->storage,
        parsed->loc
    );
}

auto Sema::TranslateMemberExpr(ParsedMemberExpr* parsed) -> Ptr<Stmt> {
    auto base = TRY(TranslateExpr(parsed->base));

    // Struct member access.
    if (auto s = dyn_cast<StructType>(base->type.ptr())) {
        if (not s->is_complete()) return Error(
            parsed->loc,
            "Member access on incomplete type '{}'",
            base->type
        );

        if (not base->lvalue()) return ICE(parsed->loc, "TODO: Materialise temporary");
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

        return new (*M) MemberAccessExpr(
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

        Error(parsed->loc, "Cannot perform member access on type '{}'", base->type);
        return AlreadyDiagnosed;
    }();

    if (kind == AlreadyDiagnosed) return {};
    if (kind == std::nullopt) {
        Error(parsed->loc, "'{}' has no member named '{}'", base->type, parsed->member);
        return {};
    }

    return BuildBuiltinMemberAccessExpr(kind.value(), base, parsed->loc);
}

auto Sema::TranslateParenExpr(ParsedParenExpr* parsed) -> Ptr<Stmt> {
    return new (*M) ParenExpr(TRY(TranslateExpr(parsed->inner)), parsed->loc);
}

auto Sema::TranslateLocalDecl(ParsedLocalDecl* parsed) -> Decl* {
    auto decl = new (*M) LocalDecl(
        TranslateType(parsed->type),
        parsed->name,
        curr_proc().proc,
        parsed->loc
    );

    // Add the declaration to the current scope.
    DeclareLocal(decl);

    // Don’t even bother with the initialiser if the type is ill-formed.
    if (not decl->type)
        return decl->set_invalid();

    // Translate the initialiser.
    Ptr<Expr> init;
    if (auto val = parsed->init.get_or_null()) {
        init = TranslateExpr(val);

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
    decl->type = AdjustVariableType(decl->type, decl->location());
    if (not decl->type) return decl->set_invalid();

    // Then, perform initialisation.
    //
    // If this fails, the initialiser is simply discarded; we can
    // still continue analysing this though as most of sema doesn’t
    // care about variable initialisers.
    decl->set_init(BuildInitialiser(
        decl->type,
        init ? init.get() : ArrayRef<Expr*>{},
        init ? init.get()->location() : decl->location()
    ));
    return decl;
}

auto Sema::TranslateProc(
    ProcDecl* decl,
    Ptr<ParsedStmt> body,
    ArrayRef<ParsedLocalDecl*> decls
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
    ArrayRef<ParsedLocalDecl*> decls
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
            parsed_decl->name,
            parsed_decl->loc
        );
    }

    // Translate body.
    auto body = TranslateExpr(parsed_body);
    if (body.invalid()) {
        // If we’re attempting to deduce the return type of this procedure,
        // but the body contains an error, just set it to void.
        if (decl->return_type() == Type::DeducedTy) {
            decl->type = ProcType::AdjustRet(*M, decl->proc_type(), Type::VoidTy);
            decl->set_invalid();
        }
        return nullptr;
    }

    return BuildProcBody(decl, body.get());
}

auto Sema::TranslateProcDecl(ParsedProcDecl*) -> Decl* {
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

    // If this is a template, we can’t do much right now.
    auto it = parsed_template_deduction_infos.find(parsed);
    auto is_template = it != parsed_template_deduction_infos.end();
    if (is_template) {
        auto decl = ProcTemplateDecl::Create(*M, parsed, curr_proc().proc);
        AddDeclToScope(curr_scope(), decl);
        return decl;
    }

    // Convert the type.
    EnterScope scope{*this, ScopeKind::Procedure};
    auto type = TranslateProcType(parsed->type);
    auto ty = cast_if_present<ProcType>(type);
    if (not ty) ty = ProcType::Get(*M, Type::VoidTy);
    auto decl = BuildProcDeclInitial(scope.get(), ty, parsed->name, parsed->loc, parsed->type->attrs);
    AddDeclToScope(scope.get()->parent(), decl);
    return decl;
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedStmt* parsed) -> Ptr<Stmt> {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
#define PARSE_TREE_LEAF_TYPE(node)             \
    case K::node: {                            \
        auto ty = TranslateType(parsed);       \
        if (not ty) return {};                 \
        return BuildTypeExpr(ty, parsed->loc); \
    }
#define PARSE_TREE_LEAF_NODE(node) \
    case K::node: return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed));
#include "srcc/ParseTree.inc"
    }

    Unreachable("Invalid parsed statement kind: {}", +parsed->kind());
}

auto Sema::TranslateStructDecl(ParsedStructDecl*) -> Decl* {
    Unreachable("Should not be translated normally");
}

auto Sema::TranslateStruct(TypeDecl* decl, ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto s = cast<StructType>(decl->type);
    Assert(not s->is_complete(), "Type is already complete?");

    // Translate the fields. While we’re at it, also keep track
    // of the struct’s size, and alignment.
    // FIXME: Move most of this logic into some BuildStructType() function.
    EnterScope _{*this, s->scope()};
    Size size{};
    Align align{1};
    SmallVector<FieldDecl*> fields;
    StructType::Bits bits;
    for (auto f : parsed->fields()) {
        auto ty = AdjustVariableType(TranslateType(f->type), f->loc);

        // If the field’s type is invalid, we can’t query any of its
        // properties, so just insert a dummy field and continue.
        if (not ty) {
            fields.emplace_back(new (*M) FieldDecl(Type::VoidTy, size, f->name, f->loc))->set_invalid();
            continue;
        }

        // Otherwise, add the field and adjust our size and alignment.
        // TODO: Optimise layout if this isn’t meant for FFI.
        size = size.align(ty->align(*M));
        fields.push_back(new (*M) FieldDecl(ty, size, f->name, f->loc));
        size += ty->size(*M);
        align = std::max(align, ty->align(*M));
        AddDeclToScope(s->scope(), fields.back());
    }

    // TODO: Initialisers are declared out-of-line, but they should
    // have been picked up during initial translation when we find
    // all the procedures in the current scope. Add any that we found
    // here, and if we didn’t find any (or the 'default' attribute was
    // specified on the struct declaration), declare the default initialiser.

    // TODO: If we decide to allow this:
    //
    // struct S { ... }
    // proc f {
    //    init S(...) { ... }
    // }
    //
    // Then the initialiser should still respect normal lookup rules (i.e.
    // it should only be visible within 'f'). Perhaps we want to store
    // local member functions outside the struct itself and in the local
    // scope instead?
    if (s->initialisers().empty()) {
        // Compute whether we can define a default initialiser for this.
        bits.init_from_no_args = bits.default_initialiser = rgs::all_of(
            fields,
            [](FieldDecl* d) { return d->type->can_init_from_no_args(); }
        );

        // We always provide a literal initialiser in this case.
        bits.literal_initialiser = true;
    }

    // Finally, mark the struct as complete.
    s->finalise(fields, size, align, bits);
    return decl;
}

auto Sema::TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto sc = M->create_scope<StructScope>(curr_scope());
    auto ty = StructType::Create(
        *M,
        sc,
        parsed->name,
        u32(parsed->fields().size()),
        parsed->loc
    );

    AddDeclToScope(curr_scope(), ty->decl());
    return ty->decl();
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed) -> Ptr<Stmt> {
    return StrLitExpr::Create(*M, parsed->value, parsed->loc);
}

/// Translate a return expression.
auto Sema::TranslateReturnExpr(ParsedReturnExpr* parsed) -> Ptr<Stmt> {
    Ptr<Expr> ret_val;
    if (parsed->value.present()) ret_val = TranslateExpr(parsed->value.get());
    return BuildReturnExpr(ret_val.get_or_null(), parsed->loc, false);
}

auto Sema::TranslateUnaryExpr(ParsedUnaryExpr* parsed) -> Ptr<Stmt> {
    auto arg = TRY(TranslateExpr(parsed->arg));
    return BuildUnaryExpr(parsed->op, arg, parsed->postfix, parsed->loc);
}

auto Sema::TranslateWhileStmt(ParsedWhileStmt* parsed) -> Ptr<Stmt> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    auto body = TRY(TranslateStmt(parsed->body));
    return BuildWhileStmt(cond, body, parsed->loc);
}

// ============================================================================
//  Translation of Types
// ============================================================================
auto Sema::TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type {
    return parsed->ty;
}

auto Sema::TranslateIntType(ParsedIntType* parsed) -> Type {
    if (parsed->bit_width > IntType::MaxBits) {
        Error(parsed->loc, "The maximum integer type is %6(i{:i}%)", IntType::MaxBits);
        return IntType::Get(*M, IntType::MaxBits);
    }
    return IntType::Get(*M, parsed->bit_width);
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

auto Sema::TranslateProcType(ParsedProcType* parsed) -> Type {
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
    for (auto a : parsed->param_types()) {
        auto ty = AdjustVariableType(TranslateType(a.type, Type::VoidTy), a.type->loc);
        params.emplace_back(a.intent, ty);
    }

    return ProcType::Get(
        *M,
        TranslateType(parsed->ret_type, Type::VoidTy),
        params,
        parsed->attrs.native ? CallingConvention::Native : CallingConvention::Source
    );
}

auto Sema::TranslateSliceType(ParsedSliceType* parsed) -> Type {
    auto ty = AdjustVariableType(TranslateType(parsed->elem), parsed->loc);
    return SliceType::Get(*M, ty);
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
        case K::IntType: t = TranslateIntType(cast<ParsedIntType>(parsed)); break;
        case K::SliceType: t = TranslateSliceType(cast<ParsedSliceType>(parsed)); break;
        case K::TemplateType: t = TranslateTemplateType(cast<ParsedTemplateType>(parsed)); break;
        case K::DeclRefExpr: t = TranslateNamedType(cast<ParsedDeclRefExpr>(parsed)); break;
        case K::ProcType: t = TranslateProcType(cast<ParsedProcType>(parsed)); break;
        default: Error(parsed->loc, "Expected type"); break;
    }

    if (not t) t = fallback;
    return t;
}
