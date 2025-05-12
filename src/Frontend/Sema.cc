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

auto Sema::Evaluate(Stmt* s, Location loc) -> Ptr<Expr> {
    // Evaluate the expression.
    auto value = M->vm.eval(s);
    if (not value.has_value()) return nullptr;

    // And cache the value for later.
    return new (*M) ConstExpr(*M, std::move(*value), loc, s);
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

bool Sema::IsCompleteType(Type ty, bool null_type_is_complete) {
    if (auto s = dyn_cast_if_present<StructType>(ty.ptr()))
        return s->is_complete();

    if (not ty)
        return null_type_is_complete;

    return true;
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

bool Sema::MakeCondition(Expr*& e, StringRef op) {
    if (auto ass = dyn_cast<BinaryExpr>(e); ass and ass->op == Tk::Assign)
        Warn(e->location(), "Assignment in condition. Did you mean to write '=='?");

    if (not MakeSRValue(Type::BoolTy, e, "Condition", op))
        return false;

    return true;
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
auto Sema::ApplySimpleConversion(Expr* e, const Conversion& conv, Location loc) -> Expr* {
    switch (conv.kind) {
        using K = Conversion::Kind;
        default: Unreachable();
        case K::IntegralCast: return new (*M) CastExpr(
            conv.type(),
            CastExpr::Integral,
            e,
            loc,
            true
        );

        case K::LValueToSRValue: return LValueToSRValue(e);
        case K::MaterialisePoison: return new (*M) CastExpr(
            conv.type(),
            CastExpr::MaterialisePoisonValue,
            e,
            loc,
            true,
            conv.value_category()
        );

        case K::SelectOverload: {
            auto proc = cast<OverloadSetExpr>(e)->overloads()[conv.data.get<u32>()];
            return CreateReference(proc, loc).get();
        }
    }
}

void Sema::ApplyConversion(SmallVectorImpl<Expr*>& exprs, const Conversion& conv, Location loc) {
    switch (conv.kind) {
        using K = Conversion::Kind;
        case K::DefaultInit: {
            Assert(exprs.empty());
            exprs.push_back(new (*M) DefaultInitExpr(conv.data.get<TypeAndValueCategory>().type(), loc));
            return;
        }

        case K::IntegralCast:
        case K::LValueToSRValue:
        case K::MaterialisePoison:
        case K::SelectOverload: {
            Assert(exprs.size() == 1);
            exprs.front() = ApplySimpleConversion(exprs.front(), conv, loc);
            return;
        }

        case K::StructInit: {
            auto& data = conv.data.get<Conversion::StructInitData>();
            for (auto [e, seq] : zip(exprs, data.field_convs)) e = ApplyConversionSequence(e, seq, loc);
            auto e = StructInitExpr::Create(*M, data.ty, exprs, loc);
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
    StructType* s,
    ArrayRef<Expr*> args,
    Location loc
) -> ConversionSequenceOrDiags {
    // First case: option 1 or 2.
    if (not s->initialisers().empty())
        return CreateICE(loc, "TODO: Call struct initialiser");

    // Second case: option 3. Option 2 is handled before we get here,
    // i.e. at this point, we know we’re building a literal initialiser.
    Assert(not args.empty(), "Should have called BuildDefaultInitialiser() instead");
    Assert(s->has_literal_init(), "Should have rejected before we ever get here");

    // Recursively build an initialiser for each element that the user provided.
    std::vector<ConversionSequence> field_seqs;
    for (auto [field, arg] : zip(s->fields(), args)) {
        auto seq = BuildConversionSequence(field->type, arg, arg->location());
        if (not seq.result.has_value()) {
            auto note = field->name.empty()
                ? CreateNote(field->location(), "In initialiser for field declared here")
                : CreateNote(field->location(), "In initialiser for field '%6({}%)'", field->name);
            seq.result.error().push_back(std::move(note));
            return seq;
        }
        field_seqs.push_back(std::move(seq.result.value()));
    }

    // For now, the number of arguments must match the number of fields in the struct.
    if (s->fields().size() != args.size()) {
        auto err = CreateError(
            loc,
            "Struct '{}' has {} field{}, but got {} argument{}",
            Type{s},
            s->fields().size(),
            s->fields().size() == 1 ? "" : "s",
            args.size(),
            args.size() == 1 ? "" : "s"
        );

        err.extra =
            "\vIf you want to be able to initialise the struct using fewer "
            "arguments, either define an initialisation procedure or provide "
            "default values for the remaining fields.";

        return err;
    }

    ConversionSequence seq;
    seq.add(Conversion::StructInit(Conversion::StructInitData{s, std::move(field_seqs)}));
    return seq;
}

auto Sema::BuildConversionSequence(
    Type var_type,
    ArrayRef<Expr*> args,
    Location init_loc,
    Intent intent,
    CallingConvention cc,
    bool in_call
) -> ConversionSequenceOrDiags {
    Assert(var_type, "Null type in initialisation?");
    ConversionSequence seq;

    // As a special case, 'noreturn' can be converted to *any* type (and value
    // category). This is because 'noreturn' means we never actually reach the
    // point in the program where the value would be needed, so it’s fine to just
    // pretend that we have one.
    auto a = args.empty() ? nullptr : args.front();
    if (a and a->type == Type::NoReturnTy) {
        auto cat = var_type->pass_value_category(cc, intent);
        if (var_type != Type::NoReturnTy or a->value_category != cat)
            seq.add(Conversion::Poison(var_type, cat));
        return seq;
    }

    // The type we’re initialising must be complete.
    if (not IsCompleteType(var_type)) return CreateError(
        init_loc,
        "Cannot create instance of incomplete type '{}'",
        var_type
    );

    // If there are no arguments, this is default initialisation.
    if (args.empty()) {
        if (var_type->can_default_init()) {
            seq.add(Conversion::DefaultInit(var_type));
            return seq;
        }

        if (var_type->can_init_from_no_args()) return CreateICE(
            init_loc,
            "TODO: non-default empty initialisation of '{}'",
            var_type
        );

        return CreateError(init_loc, "Type '{}' requires a non-empty initialiser", var_type);
    }

    // There are only few (classes of) types that support initialisation
    // from more than one argument.
    if (args.size() > 1) {
        if (isa<ArrayType>(var_type)) return CreateICE(args.front()->location(), "TODO: Array initialiser");
        if (auto s = dyn_cast<StructType>(var_type.ptr())) return BuildAggregateInitialiser(s, args, init_loc);
        return CreateError(init_loc, "Cannot create a value of type '{}' from more than one argument", var_type);
    }

    // Ok, we have exactly one argument.
    auto TypeMismatch = [&] {
        return CreateError(
            init_loc,
            "Cannot convert expression of type '{}' to '{}'",
            a->type,
            var_type
        );
    };

    // If the intent resolves to pass by reference, then we
    // need to bind to it; the type must match exactly for
    // that.
    if (in_call and var_type->pass_by_lvalue(cc, intent)) {
        // If we’re passing by lvalue, the type must match exactly.
        if (a->type != var_type) return CreateError(
            init_loc,
            "Cannot pass type {} to %1({}%) parameter of type {}",
            a->type,
            intent,
            var_type
        );

        // If the argument is not an lvalue, try to explain what the issue is.
        if (not a->lvalue()) {
            ConversionSequenceOrDiags::Diags diags;

            // If this is itself a parameter, issue a better error.
            if (auto dre = dyn_cast<LocalRefExpr>(a); dre and isa<ParamDecl>(dre->decl)) {
                diags.push_back(CreateError(
                    init_loc,
                    "Cannot pass parameter of intent %1({}%) to a parameter with intent %1({}%)",
                    cast<ParamDecl>(dre->decl)->intent(),
                    intent
                ));
                diags.push_back(CreateNote(dre->decl->location(), "Parameter declared here"));
            } else {
                diags.push_back(CreateError(init_loc, "Cannot bind this expression to an %1({}%) parameter.", intent));
            }

            diags.back().extra = "Try storing this in a variable first.";
            return diags;
        }

        // Otherwise, we have an lvalue of the same type; there is nothing
        // to be done here.
        return seq;
    }

    // Type matches exactly.
    if (a->type == var_type) {
        // We’re passing by value. For srvalue types, convert lvalues
        // to srvalues here.
        if (a->type->rvalue_category() == Expr::SRValue) {
            if (a->lvalue()) seq.add(Conversion::LValueToSRValue());
            return seq;
        }

        // If we have an mrvalue, use it directly.
        if (a->value_category == Expr::MRValue) return seq;

        // Otherwise, we have an lvalue here that we need to move from; if
        // moving is the same as copying, just leave it as it.
        if (a->type->move_is_copy()) return seq;
        return CreateICE(a->location(), "TODO: Moving a value of type '{}'", var_type);
    }

    // We need to perform conversion. What we do here depends on the type.
    switch (var_type->kind()) {
        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::SliceType:
        case TypeBase::Kind::PtrType:
            return TypeMismatch();

        case TypeBase::Kind::StructType:
            return BuildAggregateInitialiser(cast<StructType>(var_type), args, init_loc);

        case TypeBase::Kind::ProcType:
            // Type is an overload set; attempt to convert it.
            //
            // This is *not* the same algorithm as overload resolution, because
            // the types must match exactly here, and we also need to check the
            // return type.
            if (a->type == Type::UnresolvedOverloadSetTy) {
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

        // For integers, we can use the common type rule.
        case TypeBase::Kind::IntType: {
            // If this is a (possibly parenthesised and negated) integer
            // that fits in the type of the lhs, convert it. If it doesn’t
            // fit, the type must be larger, so give up.
            Expr* lit = a;
            for (;;) {
                auto u = dyn_cast<UnaryExpr>(lit);
                if (not u or u->op != Tk::Minus) break;
                lit = u->arg;
            }

            // If we ultimately found a literal, evaluate the original expression.
            if (isa_and_present<IntLitExpr>(lit)) {
                auto val = M->vm.eval(a, false);
                if (val and IntegerFitsInType(val->cast<APInt>(), var_type)) {
                    // Integer literals are srvalues so no need fo l2r conv here.
                    seq.add(Conversion::IntegralCast(var_type));
                    return seq;
                }
            }

            // Otherwise, if both are sized integer types, and the initialiser
            // is smaller, we can convert it as well.
            auto ivar = cast<IntType>(var_type);
            auto iinit = dyn_cast<IntType>(a->type);
            if (not iinit or iinit->bit_width() > ivar->bit_width()) return TypeMismatch();
            seq.add(Conversion::LValueToSRValue());
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
    bool in_call,
    Intent intent,
    CallingConvention cc
) -> Ptr<Expr> {
    auto seq_or_err = BuildConversionSequence(var_type, args, loc, intent, cc, in_call);

    // The conversion succeeded.
    if (seq_or_err.result.has_value())
        return ApplyConversionSequence(args, seq_or_err.result.value(), loc);

    // There was an error.
    for (auto& d : seq_or_err.result.error()) diags().report(std::move(d));
    return nullptr;
}

auto Sema::TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr> {
    auto seq_or_err = BuildConversionSequence(var_type, {arg}, arg->location());
    if (not seq_or_err.result.has_value()) return nullptr;
    return ApplyConversionSequence({arg}, seq_or_err.result.value(), arg->location());
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
    Assert(input_types.size() == params.size(), "Template argument count mismatch");

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
bool Sema::Candidate::has_valid_proc_type() const {
    return status.is<Viable, ParamInitFailed>();
}

bool Sema::Candidate::is_variadic() const {
    if (auto proc = dyn_cast<ProcDecl>(decl)) return proc->proc_type()->variadic();
    return cast<ProcTemplateDecl>(decl)->pattern->type->attrs.variadic;
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
    return subst->success()->type;
}

auto Sema::Candidate::type_for_diagnostic() const -> SmallUnrenderedString {
    auto d = dyn_cast<ProcDecl>(decl);
    if (d) return d->proc_type()->print();
    if (subst and subst->success()) return subst->success()->type->print();
    return SmallUnrenderedString("(template)");
}

u32 Sema::ConversionSequence::badness() {
    u32 badness = 0;
    for (auto& conv : conversions) {
        switch (conv.kind) {
            using K = Conversion::Kind;

            // These don’t perform type conversion.
            case K::LValueToSRValue:
            case K::SelectOverload:
            case K::DefaultInit:
                break;

            // These are actual type conversions.
            case K::IntegralCast:
            case K::MaterialisePoison:
                badness++;
                break;

            // These contain other conversions.
            case K::StructInit: {
                auto& data = conv.data.get<Conversion::StructInitData>();
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
        auto p = d->params()[i];
        loc = p->location();
        name = p->name;
    } else if (auto d = dyn_cast<ProcTemplateDecl>(proc)) {
        auto p = d->pattern->params()[i];
        loc = p->loc;
        name = p->name;
    } else {
        return;
    }

    if (name.empty()) S.Note(loc, "In argument to parameter declared here");
    else S.Note(loc, "In argument to parameter '{}'", name);
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

    // Add a candidate to the overload set.
    auto AddCandidate = [&](Decl* proc) -> bool {
        if (not proc->valid())
            return false;

        // Add the candidate.
        auto& c = candidates.emplace_back(proc);

        // Argument count mismatch is not allowed, unless the
        // function is variadic.
        auto param_count = c.param_count();
        if (args.size() != param_count) {
            if (args.size() < param_count or not c.is_variadic()) {
                c.status = Candidate::ArgumentCountMismatch{};
                return true;
            }
        }

        // Candidate is a regular procedure.
        auto templ = dyn_cast<ProcTemplateDecl>(proc);
        if (not templ) return true;

        // Candidate is a template. Check that we have enough arguments
        // to perform
        SmallVector<TypeLoc, 6> types;
        for (auto arg : args | vws::take(param_count)) types.emplace_back(arg->type, arg->location());
        c.subst = &SubstituteTemplate(templ, types);

        // If there was a hard error, abort overload resolution entirely.
        if (c.subst->data.is<SubstitutionInfo::Error>()) return false;

        // Otherwise, ee can still try and continue with overload resolution.
        if (not c.subst->success()) c.status = Candidate::DeductionError{};
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

        // Check that we can initialise each parameter with its
        // corresponding argument. Variadic arguments are checked
        // later when the call is built.
        for (auto [i, a] : enumerate(args.take_front(params.size()))) {
            // Candidate may have become invalid in the meantime.
            auto st = c.status.get_if<Candidate::Viable>();
            if (not st) break;

            // Check the next parameter.
            auto& p = params[i];
            auto seq_or_err = BuildConversionSequence(
                p.type,
                {a},
                a->location(),
                p.intent,
                ty->cconv(),
                true
            );

            // If this failed, stop checking this candidate.
            if (not seq_or_err.result.has_value()) {
                c.status = Candidate::ParamInitFailed{std::move(seq_or_err.result.error()), u32(i)};
                break;
            }

            // Otherwise, store the sequence for later and keep going.
            st->conversions.push_back(std::move(seq_or_err.result.value()));
            st->badness += st->conversions.back().badness();
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
        if (c->subst) {
            auto inst = InstantiateTemplate(*c->subst, call_loc);
            if (not inst or not inst->is_valid) return {};
            final_callee = inst;
        } else {
            final_callee = cast<ProcDecl>(c->decl);
        }

        // Now is the time to apply the argument conversions.
        SmallVector<Expr*> actual_args;
        actual_args.reserve(args.size());
        for (auto [i, conv] : enumerate(c->status.get<Candidate::Viable>().conversions))
            actual_args.emplace_back(ApplyConversionSequence(args[i], conv, args[i]->location()));
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
    auto FormatTempSubstFailure = [&](const SubstitutionInfo& info, std::string& out, std::string_view indent) {
        info.data.visit(utils::Overloaded{// clang-format off
            [](const SubstitutionInfo::Success&) { Unreachable("Invalid template even though substitution succeeded?"); },
            [](SubstitutionInfo::Error) { Unreachable("Should have bailed out earlier on hard error"); },
            [&](SubstitutionInfo::DeductionFailed f) {
                out += std::format(
                    "In param #{}: could not infer ${}",
                    f.param_index + 1,
                    f.param
                );
            },

            [&](const SubstitutionInfo::DeductionAmbiguous& a) {
                out += std::format(
                    "Inference mismatch for template parameter %3(${}%):\n"
                    "{}#{}: Inferred as {}\n"
                    "{}#{}: Inferred as {}",
                    a.param,
                    indent,
                    a.first,
                    a.first_type,
                    indent,
                    a.second,
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
                Assert(c.subst, "DeductionError requires a SubstitutionInfo");
                std::string extra;
                FormatTempSubstFailure(*c.subst, extra, "  ");
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
                Assert(c.subst, "DeductionError requires a SubstitutionInfo");
                message += "Template argument substitution failed";
                FormatTempSubstFailure(*c.subst, message, "        ");
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

    auto a = new (*M) AssertExpr(cond, std::move(msg), false, loc);
    if (not is_compile_time) return a;
    return Evaluate(a, loc);
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
) -> Ptr<Expr> {
    using enum ValueCategory;
    auto Build = [&](Type ty, ValueCategory cat = SRValue) {
        return new (*M) BinaryExpr(ty, cat, op, lhs, rhs, loc);
    };

    auto CheckIntegral = [&] -> bool {
        // Either operand must be an integer.
        bool lhs_int = lhs->type->is_integer();
        bool rhs_int = rhs->type->is_integer();
        if (not lhs_int and not rhs_int) {
            Error(
                loc,
                "Unsupported %1({}%) of '{}' and '{}'",
                Spelling(op),
                lhs->type,
                rhs->type
            );
            return false;
        }
        return true;
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
        if (not CheckIntegral() or not ConvertToCommonType()) return nullptr;
        return Build(comparison ? Type::BoolTy : lhs->type);
    };

    auto BuildExpCall = [&](String exp_fun) -> Ptr<Expr> {
        auto ref = LookUpName(global_scope(), exp_fun, loc, true);
        if (not ref) return nullptr;
        return BuildCallExpr(CreateReference(ref.decls.front(), loc).get(), {lhs, rhs}, loc);
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

        // This is implemented as a function template.
        case Tk::StarStar: {
            if (not CheckIntegral() or not ConvertToCommonType()) return nullptr;
            return BuildExpCall("__srcc_exp_i");
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
            if (lhs->value_category != LValue) {
                // Issue a better diagnostic for 'in' parameters.
                if (auto ref = dyn_cast<LocalRefExpr>(lhs)) {
                    if (
                        auto param = dyn_cast<ParamDecl>(ref->decl);
                        param and param->intent() == Intent::In
                    ) return Error(lhs->location(), "Cannot assign to '%1(in%)' parameter");
                }

                return Error(
                    lhs->location(),
                    "Invalid target for assignment"
                );
            }

            // Regular assignment.
            if (op == Tk::Assign) {
                if (auto init = BuildInitialiser(lhs->type, rhs, rhs->location()).get_or_null()) rhs = init;
                else return nullptr;
                return Build(lhs->type, LValue);
            }

            // Compound assignment.
            if (not CheckIntegral()) return nullptr;
            rhs = BuildInitialiser(lhs->type, rhs, rhs->location()).get_or_null();
            if (not rhs) return nullptr;
            if (op != Tk::StarStarEq) return Build(lhs->type, LValue);

            // '**=' requires a separate function since it needs to return the lhs.
            return CastExpr::Dereference(*M, TRY(BuildExpCall("__srcc_exp_assign_i")));
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

        case BuiltinCallExpr::Builtin::Unreachable: {
            if (not ForbidArgs("__srcc_unreachable")) return nullptr;
            return BuiltinCallExpr::Create(*M, builtin, Type::NoReturnTy, {}, call_loc);
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
            case AK::SliceData: return PtrType::Get(*M, cast<SliceType>(operand->type)->elem());
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

    // If the type of this is a procedure, then we can skip overload
    // resolution. While this also means we have some code duplication,
    // we don’t have to build conversion sequences here and can instead
    // apply them immediately, and overload resolution also wouldn’t
    // work for indirect calls since for those we don’t have a reference
    // to the procedure declaration.
    else if (auto ty = dyn_cast<ProcType>(callee_expr->type)) {
        resolved_callee = LValueToSRValue(callee_expr);

        // Check arg count.
        auto params = ty->params().size();
        auto argn = args.size();
        if (ty->variadic() ? params > argn : params != argn) {
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

        // Convert each non-variadic parameter.
        converted_args.reserve(args.size());
        for (auto [i, p, a] : enumerate(ty->params(), args.take_front(ty->params().size()))) {
            auto arg = BuildInitialiser(p.type, a, a->location(), true, p.intent, ty->cconv());

            // Point to the procedure if this is a direct call.
            if (not arg and isa<ProcRefExpr>(callee_expr)) {
                auto proc = cast<ProcRefExpr>(callee_expr);
                NoteParameter(*this, proc->decl, u32(i));
                return nullptr;
            }

            converted_args.push_back(arg.get());
        }
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
            converted_args.push_back(LValueToSRValue(a));
        } else {
            Error(
                a->location(),
                "Passing a value of type '{}' as a varargs argument is not supported",
                a->type
            );
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
    return CallExpr::Create(
        *M,
        cast<ProcType>(resolved_callee->type)->ret(),
        resolved_callee,
        converted_args,
        loc
    );
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
    if (not MakeCondition(cond, "if")) return {};

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

    // If we get here and the type is an mrvalue type, then we either have two
    // mrvalues or an mrvalue and an lvalue; either way, codegen knows how to
    // emit an lvalue as an mrvalue, so for types that are trivially copyable,
    // we don’t need to do anything here.
    if (common_ty->rvalue_category() == Expr::MRValue) {
        if (common_ty->move_is_copy()) return Build(common_ty, Expr::MRValue);
        return ICE(loc, "TODO: Move a value of type '{}'", common_ty);
    }

    // The type is an srvalue type. Make sure both sides are srvalues.
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
        if (ret != proc->return_type()) {
            value = nullptr;
            Error(
                loc,
                "Return type '{}' does not match procedure return type '{}'",
                ret,
                proc->return_type()
            );
        }
    }

    // Perform any necessary conversions.
    //
    // If the type is zero-sized, there is no need to do anything since we’ll
    // drop it anyway.
    if (auto val = value.get_or_null(); val and val->type->size(*M) != Size())
        value = BuildInitialiser(proc->return_type(), {val}, loc);

    return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildStaticIfExpr(
    Expr* cond,
    ParsedStmt* then,
    Ptr<ParsedStmt> else_,
    Location loc
) -> Ptr<Stmt> {
    // Otherwise, check this now.
    if (not MakeCondition(cond, "#if")) return {};
    auto val = M->vm.eval(cond);
    if (not val) return {};

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

        // Lvalue -> Pointer
        case Tk::Ampersand: {
            if (not operand->lvalue()) return Error(loc, "Cannot take address of non-lvalue");
            return Build(PtrType::Get(*M, operand->type), Expr::SRValue);
        }

        // Pointer -> Lvalue.
        case Tk::Caret: {
            auto ptr = dyn_cast<PtrType>(operand->type);
            if (not ptr) return Error(
                loc,
                "Cannot dereference value of non-pointer type '{}'",
                operand->type
            );

            operand = LValueToSRValue(operand);
            return Build(ptr->elem(), Expr::LValue);
        }

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
    if (not MakeCondition(cond, "while")) return {};
    return new (*M) WhileStmt(cond, body, loc);
}

// ============================================================================
//  Translation Driver
// ============================================================================
Sema::EnterProcedure::EnterProcedure(Sema& S, ProcDecl* proc)
    : info{S, proc} {
    Assert(proc->scope, "Entering procedure without scope?");
    S.proc_stack.emplace_back(&info);
}

Sema::EnterScope::EnterScope(Sema& S, ScopeKind kind) : S{S} {
    Assert(not S.scope_stack.empty(), "Should not be used for the global scope");
    scope = S.M->create_scope(S.curr_scope(), kind);
    S.scope_stack.push_back(scope);
}

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

auto Sema::Translate(
    const LangOpts& opts,
    ParsedModule::Ptr preamble,
    SmallVector<ParsedModule::Ptr> modules,
    StringMap<ImportHandle> imported_modules
) -> TranslationUnit::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};

    // Create the TU.
    S.M = TranslationUnit::Create(
        first->context(),
        opts,
        first->name,
        first->is_module
    );

    // Take ownership of the modules.
    if (preamble) S.parsed_modules.push_back(std::move(preamble));
    for (auto& m : modules) S.parsed_modules.push_back(std::move(m));
    S.M->imports = std::move(imported_modules);

    // Translate it.
    S.Translate();
    return std::move(S.M);
}

void Sema::Translate() {
    // Take ownership of any resources of the parsed modules.
    for (auto& p : parsed_modules) {
        M->add_allocator(std::move(p->string_alloc));
        M->add_integer_storage(std::move(p->integers));
        for (auto& [decl, info] : p->template_deduction_infos)
            parsed_template_deduction_infos[decl] = std::move(info);
    }

    // Set up scope stacks.
    M->initialiser_proc->scope = M->create_scope(nullptr);
    EnterProcedure _{*this, M->initialiser_proc};

    // Collect all statements and translate them.
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, Location{});

    // File scope block should never be dependent.
    M->initialiser_proc->finalise(
        BuildProcBody(M->initialiser_proc, M->file_scope_block),
        curr_proc().locals
    );

    // Sanity check.
    Assert(proc_stack.size() == 1);
    Assert(scope_stack.size() == 1);
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
    return BuildAssertExpr(cond, msg, parsed->is_compile_time, parsed->loc);
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
    return TranslateExpr(parsed->inner);
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
    if (not type) decl->set_invalid();
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
        bool complete = IsCompleteType(ty);
        if (not ty or not complete) {
            // TODO: Allow this and instead actually perform recursive translation and
            // cycle checking.
            if (not complete) Error(
                f->loc,
                "Cannot declare field of incomplete type '{}'",
                ty
            );

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
    bool ok = true;
    for (auto a : parsed->param_types()) {
        auto ty = TranslateType(a.type);

        // Diagnose this here, but don’t do anything about it; this is only
        // harmful if we emit LLVM IR for it, and we won’t be getting there
        // anyway because of this error.
        if (ty and parsed->attrs.native) {
            if (ty == Type::VoidTy) Error(
                a.type->loc,
                "Passing '%1(void%)' to a '%1(native%)' procedure is not supported"
            );

            else if (ty->size(*M) == Size()) Error(
                a.type->loc,
                "Passing zero-sized type '%1({}%)' to a '%1(native%)' procedure is not supported",
                ty
            );
        }

        ty = AdjustVariableType(ty, a.type->loc);
        if (not ty) ok = false;
        params.emplace_back(a.intent, ty);
    }

    if (not ok) return Type();
    return ProcType::Get(
        *M,
        TranslateType(parsed->ret_type, Type::VoidTy),
        params,
        parsed->attrs.native ? CallingConvention::Native : CallingConvention::Source
    );
}

auto Sema::TranslatePtrType(ParsedPtrType* stmt) -> Type {
    auto ty = AdjustVariableType(TranslateType(stmt->elem), stmt->loc);
    if (not ty) return Type();
    return PtrType::Get(*M, ty);
}

auto Sema::TranslateSliceType(ParsedSliceType* parsed) -> Type {
    auto ty = AdjustVariableType(TranslateType(parsed->elem), parsed->loc);
    if (not ty) return Type();
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
        case K::DeclRefExpr: t = TranslateNamedType(cast<ParsedDeclRefExpr>(parsed)); break;
        case K::IntType: t = TranslateIntType(cast<ParsedIntType>(parsed)); break;
        case K::ProcType: t = TranslateProcType(cast<ParsedProcType>(parsed)); break;
        case K::PtrType: t = TranslatePtrType(cast<ParsedPtrType>(parsed)); break;
        case K::SliceType: t = TranslateSliceType(cast<ParsedSliceType>(parsed)); break;
        case K::TemplateType: t = TranslateTemplateType(cast<ParsedTemplateType>(parsed)); break;
        default: Error(parsed->loc, "Expected type"); break;
    }

    if (not t) t = fallback;
    return t;
}
