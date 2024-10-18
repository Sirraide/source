module;

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Alignment.h>
#include <print>
#include <ranges>
#include <srcc/ClangForward.hh>
#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.utils;
import srcc.ast;
import srcc.token;
import base.fs;
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
    auto& ds = scope->decls[d->name];
    if (not ds.empty() and isa<ParamDecl, TemplateTypeDecl>(d)) {
        Error(d->location(), "Redeclaration of parameter '{}'", d->name);
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

auto Sema::CheckVariableType(Type ty, Location loc) -> Type {
    // Any places that want to do type deduction need to take
    // care of it *before* this is called.
    if (ty == Types::DeducedTy) {
        Error(loc, "Type deduction is not allowed here");
        return Types::ErrorDependentTy;
    }

    if (ty == Types::NoReturnTy or ty == Types::TypeTy) {
        Error(loc, "Cannot declare a variable of type '{}'", ty);
        return Types::ErrorDependentTy;
    }

    if (ty == Types::UnresolvedOverloadSetTy) {
        Error(loc, "Unresolved overload set in parameter declaration");
        return Types::ErrorDependentTy;
    }

    if (auto s = dyn_cast<StructType>(ty.ptr()); s and not s->is_complete()) {
        Error(loc, "Declaring a variable of type '{}' before it is complete", ty);
        Note(s->decl()->location(), "'{}' declared here", ty);
        return Types::ErrorDependentTy;
    }

    return ty;
}

auto Sema::CreateReference(Decl* d, Location loc) -> Ptr<Expr> {
    switch (d->kind()) {
        default: return ICE(d->location(), "Cannot build a reference to this declaration yet");
        case Stmt::Kind::ProcDecl: return new (*M) ProcRefExpr(cast<ProcDecl>(d), loc);
        case Stmt::Kind::TypeDecl: return new (*M) TypeExpr(cast<TypeDecl>(d)->type, loc);
        case Stmt::Kind::LocalDecl:
        case Stmt::Kind::ParamDecl:
            return new (*M) LocalRefExpr(cast<LocalDecl>(d), loc);
    }
}

void Sema::DeclareLocal(LocalDecl* d) {
    Assert(d->parent == curr_proc().proc, "Must EnterProcedure before adding a local variable");
    curr_proc().locals.push_back(d);

    // If the current procedure is a template instantiation, do not
    // add this to the procedure scope again since it’s the same one.
    if (not isa<InstantiationScopeInfo>(curr_proc()))
        AddDeclToScope(curr_scope(), d);
}

auto Sema::GetScopeFromDecl(Decl* d) -> Ptr<Scope> {
    switch (d->kind()) {
        default: return {};
        case Stmt::Kind::BlockExpr: return cast<BlockExpr>(d)->scope;
        case Stmt::Kind::ProcDecl: return cast<ProcDecl>(d)->scope;
    }
}

bool Sema::IntegerLiteralFitsInType(IntLitExpr* i, Type ty) {
    if (not ty->is_integer()) return false;
    auto to_bits //
        = ty == Types::IntTy
            ? Types::IntTy->size(*M)
            : cast<IntType>(ty)->bit_width();

    return Size::Bits(i->storage.value().getActiveBits()) <= to_bits;
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
                if (it->second.is<TranslationUnit*>()) Todo();

                // We found an imported C++ header; do a C++ lookup.
                auto hdr = it->second.get<clang::ASTUnit*>();
                return LookUpCXXName(hdr, names.drop_front());
            } break;
        }
    }

    // For all elements but the last, we have to look up scopes.
    for (auto name : names.drop_front().drop_back()) {
        // Perform lookup.
        auto it = in_scope->decls.find(name);
        if (it == in_scope->decls.end()) return LookupResult(name);

        // The declaration must not be ambiguous.
        Assert(not in_scope->decls.empty(), "Invalid scope entry");
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
        auto it = in_scope->decls.find(name);
        if (it == in_scope->decls.end()) {
            if (this_scope_only) break;
            in_scope = in_scope->parent();
            continue;
        }

        // Found something.
        Assert(not in_scope->decls.empty(), "Invalid scope entry");
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
    auto init = TryPerformVariableInitialisation(ty, e);
    if (not init) {
        Error(
            e->location(),
            "{} of '%1({})' must be of type\f'{}', but was '{}'",
            elem_name,
            op,
            ty,
            e->type
        );
        return false;
    }

    // Make sure it’s an srvalue.
    e = LValueToSRValue(init);
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
    base::File::Path path;
    for (auto& base : module_search_paths) {
        auto combined = base::File::Path{base} / filename;
        if (base::File::Exists(combined)) {
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
    Expr* res;
    Type target_type;

public:
    ImmediateInitContext(Sema& S, Expr* e, Type target_type)
        : S{S},
          res{e},
          target_type{target_type} {}

    void apply(Conversion c) { res = S.ApplyConversion(res, c); }

    bool report_lvalue_intent_mismatch(Intent intent) {
        // If this is itself a parameter, issue a better error.
        if (auto dre = dyn_cast<LocalRefExpr>(res->strip_parens()); dre and isa<ParamDecl>(dre->decl)) {
            S.Error(
                res->location(),
                "Cannot pass parameter of intent %1({}) to a parameter with intent %1({})",
                cast<ParamDecl>(dre->decl)->intent(),
                intent
            );
            S.Note(dre->decl->location(), "Parameter declared here");
        } else {
            S.Error(res->location(), "Cannot bind this expression to an %1({}) parameter.", intent);
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
            "Cannot pass type {} to %1({}) parameter of type {}",
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
        : ImmediateInitContext{S, e, target_type} {}

    bool report_lvalue_intent_mismatch(Intent) { return false; }
    bool report_nested_resolution_failure() { return false; }
    bool report_type_mismatch() { return false; }
    bool report_same_type_lvalue_required(Intent) { return false; }
};

template <typename InitContext>
bool Sema::PerformVariableInitialisation(
    InitContext& init,
    Type var_type,
    Expr* a,
    Intent intent,
    CallingConvention cc,
    bool in_call
) {
    Assert(not var_type->dependent(), "Initialising dependent variable?");
    Assert(not a->dependent(), "Dependent initialiser?");

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
        // We’re passing by value. Currently, we can only handle srvalues here.
        if (a->type->value_category() != Expr::SRValue) {
            ICE(
                a->location(),
                "Sorry, we currently don’t support {} of type {}",
                in_call ? "by-value arguments" : "assigning to variables",
                a->type
            );
            return false;
        }

        // Convert lvalues to srvalues here.
        if (a->lvalue()) init.apply(Conversion::LValueToSRValue());
        return true;
    }

    // We need to perform conversion. What we do here depends on the type.
    switch (var_type->kind()) {
        case TypeBase::Kind::TemplateType:
            Unreachable("Attempting to initialise dependent type?");

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
            if (a->type == Types::UnresolvedOverloadSetTy) {
                auto p_proc_type = dyn_cast<ProcType>(var_type.ptr());
                if (not p_proc_type) return init.report_type_mismatch();

                // Instantiate templates and simply match function types otherwise; we
                // don’t need to do anything fancier here.
                auto overloads = cast<OverloadSetExpr>(a->strip_parens())->overloads();

                // Check non-templates first to avoid storing template substitution
                // for all of them.
                for (auto [j, o] : enumerate(overloads)) {
                    if (o->is_template()) continue;

                    // We have a match!
                    //
                    // The internal consistency of an overload set was already verified
                    // when the corresponding declarations were added to their scope, so
                    // if one of them matches, it is the only one that matches.
                    if (o->type == var_type) {
                        init.apply(Conversion::SelectOverload(u16(j)));
                        return true;
                    }
                }

                // Otherwise, we need to try and instantiate templates in this overload set.
                for (auto o : overloads) {
                    if (not o->is_template()) continue;
                    Todo("Instantiate template in nested overload set");
                }

                // None of the overloads matched.
                return init.report_nested_resolution_failure();
            }

            // Otherwise, the types don’t match.
            return init.report_type_mismatch();

        // For integers, we can use the common type rule.
        case TypeBase::Kind::IntType: {
            // If the rhs is an integer literal that fits in the type of
            // the lhs, convert it. If it doesn’t fit, the type must be
            // larger, so give up.
            if (auto lit = dyn_cast<IntLitExpr>(a->strip_parens())) {
                if (IntegerLiteralFitsInType(lit, var_type)) {
                    // Integer literals are srvalues so no need fo l2r conv here.
                    init.apply(Conversion::IntegralCast(var_type));
                    return true;
                }

                return init.report_type_mismatch();
            }

            // Otherwise, if both are sized integer types, and the initialiser
            // is smaller, we can convert it as well.
            auto ivar = cast<IntType>(var_type);
            auto iinit = dyn_cast<IntType>(a->type);
            if (not iinit or iinit.value()->bit_width() > ivar->bit_width()) return init.report_type_mismatch();
            init.apply(Conversion::LValueToSRValue());
            init.apply(Conversion::IntegralCast(var_type));
            return true;
        }

        // For builtin types, it depends.
        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(var_type)->builtin_kind()) {
                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
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
            Todo();
        }
    }

    Unreachable();
}

auto Sema::PerformVariableInitialisation(
    Type var_type,
    Expr* arg,
    Intent intent,
    CallingConvention cc,
    bool is_call
) -> Expr* {
    // Passing in 'arg' and 'var_type' twice here is unavoidable because
    // InitContexts generally don't store either; it’s just this one that
    // does...
    ImmediateInitContext init{*this, arg, var_type};
    if (not PerformVariableInitialisation(init, var_type, arg, intent, cc, is_call)) return nullptr;
    return init.result();
}

auto Sema::TryPerformVariableInitialisation(Type var_type, Expr* arg) -> Expr* {
    TentativeInitContext init{*this, arg, var_type};
    if (not PerformVariableInitialisation(init, var_type, arg)) return nullptr;
    return init.result();
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
    auto FormatTempSubstFailure = [&](const Candidate::TemplateInfo& ti, std::string& out, std::string_view indent) {
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
                    "Template deduction mismatch for parameter %3(${}):\n"
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
    };

    // If there is only one overload, print the failure reason for
    // it and leave it at that.
    if (candidates.size() == 1) {
        auto c = candidates.front();
        auto ty = c.type_for_diagnostic();
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
                    "Procedure '%2({})' expects {} argument{}, got {}",
                    c.name(),
                    ty->params().size(),
                    ty->params().size() == 1 ? "" : "s",
                    call_args.size()
                );
                Note(c.location(), "Declared here");
            },

            [&](Candidate::InvalidTemplate) {
                std::string extra;
                FormatTempSubstFailure(c.proc.get<Candidate::TemplateInfo>(), extra, "  ");
                Error(call_loc, "Template argument substitution failed");
                Remark("\r{}", extra);
                Note(c.location(), "Declared here");
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
                Note(c.location(), "Declared here");
                Remark("\rTry specifying the return type explicitly: '%1(->) <type>'");
            }
        }; // clang-format on
        c.status.visit(V);
        return;
    }

    // Otherwise, we need to print all overloads, and why they failed.
    std::string message = std::format("%b(Candidates:)\n");

    // Compute the width of the number field.
    u32 width = u32(std::to_string(candidates.size()).size());

    // First, print all overloads.
    for (auto [i, c] : enumerate(candidates)) {
        // Print the type.
        message += std::format(
            "  %b({}.) \v{}",
            i + 1,
            c.type_for_diagnostic()
        );

        // And include the location if it is valid.
        auto loc = c.location();
        auto lc = loc.seek_line_column(ctx);
        if (lc) {
            message += std::format(
                "\f%b(at) {}:{}:{}",
                ctx.file(loc.file_id)->name(),
                lc->line,
                lc->col
            );
        }

        message += "\n";
    }

    message += std::format("\n\r%b(Failure Reason:)");

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        message += std::format("\n  %b({:>{}}.) ", i + 1, width);
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
                auto params = c.type_for_diagnostic()->params();
                message += std::format(
                    "Expected {} arg{}, got {}",
                    params.size(),
                    params.size() == 1 ? "" : "s",
                    call_args.size()
                );
            },

            [&](Candidate::InvalidTemplate) {
                message += "Template argument substitution failed";
                FormatTempSubstFailure(c.proc.get<Candidate::TemplateInfo>(), message, "        ");
            },

            [&](Candidate::LValueIntentMismatch m) {
                auto& p = c.type_for_diagnostic()->params()[m.mismatch_index];
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
                    c.type_for_diagnostic()->params()[t.mismatch_index].type,
                    call_args[t.mismatch_index]->type
                );
            },

            [&](Candidate::SameTypeLValueRequired m) {
                auto& p = c.type_for_diagnostic()->params()[m.mismatch_index];
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
    }

    ctx.diags().report(Diagnostic{
        Diagnostic::Level::Error,
        call_loc,
        std::format("Overload resolution failed in call to\f'%2({})'", candidates.front().name()),
        std::move(message),
    });
}

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildAssertExpr(Expr* cond, Ptr<Expr> msg, Location loc) -> Ptr<AssertExpr> {
    // Condition must be a bool.
    if (not cond->dependent()) {
        if (not MakeSRValue(Types::BoolTy, cond, "Condition", "assert"))
            return {};
    }

    // Message must be a string literal.
    // TODO: Allow other string-like expressions.
    if (auto m = msg.get_or_null(); m and not m->dependent()) {
        if (not isa<StrLitExpr>(m)) return Error(
            m->location(),
            "Assertion message must be a string literal",
            m->type
        );
    }

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
        if (auto lhs_conv = TryPerformVariableInitialisation(rhs->type, lhs)) {
            lhs = lhs_conv;
        } else if (auto rhs_conv = TryPerformVariableInitialisation(lhs->type, rhs)) {
            rhs = rhs_conv;
        } else {
            Error(
                loc,
                "Invalid operation: %1({}) between {} and {}",
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
            Error(e->location(), "{} of %1({}) must be an integer", which, Spelling(op));
            return false;
        };

        // Both operands must be integers.
        if (not Check("Left operand", lhs) or not Check("Right operand", rhs)) return nullptr;
        if (not ConvertToCommonType()) return nullptr;
        return Build(comparison ? Types::BoolTy : lhs->type);
    };

    // If either operand is dependent, then we can’t do much.
    if (lhs->dependent() or rhs->dependent()) return Build(
        Types::DependentTy,
        DValue
    );

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

            if (not MakeSRValue(Types::IntTy, rhs, "Index", "[]")) return {};

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
            return Build(Types::BoolTy);
        }

        // Logical operator.
        case Tk::And:
        case Tk::Or:
        case Tk::Xor: {
            if (not MakeSRValue(Types::BoolTy, lhs, "Left operand", Spelling(op))) return {};
            if (not MakeSRValue(Types::BoolTy, rhs, "Right operand", Spelling(op))) return {};
            return Build(Types::BoolTy);
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
                if (rhs->type->value_category() == MRValue) return ICE(
                    rhs->location(),
                    "Sorry, assignment to a variable of type '{}' is not yet supported",
                    rhs->type
                );

                rhs = LValueToSRValue(rhs);

                // For arithmetic operations, both sides must be ints.
                if (lhs->type != Types::IntTy) Error(
                    lhs->location(),
                    "Compound assignment operator '{}' is not supported for type '{}'",
                    Spelling(op),
                    lhs->type
                );

                // The LHS is returned as an lvalue.
                return Build(lhs->type, LValue);
            }

            // Delegate to initialisation.
            rhs = PerformVariableInitialisation(lhs->type, rhs);
            if (not rhs) return nullptr;
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
                if (arg->dependent()) continue;
                if (
                    arg->type != M->StrLitTy and
                    arg->type != Types::IntTy and
                    arg->type != Types::BoolTy
                ) return Error( //
                    arg->location(),
                    "__srcc_print only accepts i8[] and integers, but got {}",
                    arg->type
                );
                arg = LValueToSRValue(arg);
            }
            return BuiltinCallExpr::Create(*M, builtin, Types::VoidTy, actual_args, call_loc);
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
        switch (ak)  {
            using AK = BuiltinMemberAccessExpr::AccessKind;
            case AK::SliceData: return ReferenceType::Get(*M, cast<SliceType>(operand->type)->elem());
            case AK::SliceSize: return Types::IntTy;
            case AK::TypeAlign: return Types::IntTy;
            case AK::TypeArraySize: return Types::IntTy;
            case AK::TypeBits: return Types::IntTy;
            case AK::TypeBytes: return Types::IntTy;
            case AK::TypeName: return M->StrLitTy;
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

auto Sema::BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, Location loc) -> Ptr<CallExpr> {
    // Build a call expression that we can resolve later, usually
    // because the callee is dependent.
    auto BuildDependentCallExpr = [&](Expr* callee) {
        return CallExpr::Create(
            *M,
            Types::DependentTy,
            callee,
            args,
            loc
        );
    };

    // Check variadic arguments.
    auto CheckVarargs = [&](SmallVectorImpl<Expr*>& actual_args, ArrayRef<Expr*> varargs) {
        bool ok = true;
        for (auto a : varargs) {
            if (
                a->type == Types::IntTy or
                a->type == Types::BoolTy or
                a->type == Types::NoReturnTy or
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

    // Calls with dependent arguments are checked when they’re instantiated.
    if (rgs::any_of(args, [](Expr* e) { return e->dependent(); }))
        return BuildDependentCallExpr(callee_expr);

    // If this is not a procedure reference or overload set, then we don’t
    // need to perform overload resolution nor template instantiation, so
    // just typecheck the arguments directly.
    auto callee_no_parens = callee_expr->strip_parens();
    if (not isa<OverloadSetExpr, ProcRefExpr>(callee_no_parens)) {
        auto ty = dyn_cast<ProcType>(callee_expr->type.ptr());

        // If this is not a literal procedure and still dependent, then we
        // can’t check this yet.
        if (callee_expr->dependent()) return BuildDependentCallExpr(callee_expr);

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
            auto arg = PerformVariableInitialisation(p.type, a, p.intent, ty->cconv(), true);
            if (not arg) return {};
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
    bool dependent = false;

    // Add a candidate to the overload set.
    auto AddCandidate = [&](ProcDecl* proc) -> bool {
        // Candidate is a template.
        if (not proc->is_template()) {
            candidates.emplace_back(proc);
            return true;
        }

        // Collect the types of all arguments (and their locations for
        // diagnostics) for substitution.
        SmallVector<TypeLoc, 6> types;
        for (auto arg : args) types.emplace_back(arg->type, arg->location());

        // Perform template substitution.
        auto res = SubstituteTemplate(proc, types, callee_expr->location());
        if (res.data.is<TempSubstRes::Error>()) return false;
        if (
            auto subst = res.data.get_if<TempSubstRes::Success>();
            subst and subst->type->dependent()
        ) dependent = true;
        candidates.emplace_back(proc, std::move(res));
        return true;
    };

    // Collect all candidates.
    if (auto proc = dyn_cast<ProcRefExpr>(callee_no_parens)) {
        if (not AddCandidate(proc->decl)) return {};
    } else {
        auto os = cast<OverloadSetExpr>(callee_no_parens);
        if (not rgs::all_of(os->overloads(), AddCandidate)) return {};
    }

    // If any of the candidates are still dependent, we’re either in
    // a nested template, or there was an error; either way, we can’t
    // instantiate anything right now, so defer overload resolution
    // until we can.
    if (dependent) return BuildDependentCallExpr(callee_expr);

    // Check if a single candidate is viable. Returns false if there
    // is a fatal error that prevents overload resolution entirely.
    auto CheckCandidate = [&](Candidate& c) -> bool {
        auto ty = c.type();
        auto params = ty->params();

        // If the candidate’s return type is deduced, we’re trying to
        // call it before it has been fully analysed. Disallow this.
        if (ty->ret() == Types::DeducedTy and not c.is_template()) {
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
            if (not PerformVariableInitialisation(init, p.type, a, p.intent, ty->cconv(), true)) return false;
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
        if (auto* temp = c->proc.get_if<Candidate::TemplateInfo>()) {
            auto& subst = temp->res.data.get<TempSubstRes::Success>();
            auto inst = InstantiateTemplate(
                temp->pattern,
                subst.type,
                subst.args,
                callee_expr->location()
            );

            // And call it.
            if (not inst) return nullptr;
            final_callee = inst.get();
        }

        // Otherwise, just grab the procedure.
        else { final_callee = c->proc.get<ProcDecl*>(); }

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
    if (arg->dependent()) return eval;

    // If the expression is not dependent, evaluate it now.
    auto value = eval::Evaluate(*M, arg);
    if (not value.has_value()) {
        eval->set_errored();
        return eval;
    }

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
    if (cond->type == Types::NoReturnTy) return Build(Types::NoReturnTy, Expr::SRValue);

    // Condition must be a bool.
    if (not cond->dependent() and not MakeSRValue(Types::BoolTy, cond, "Condition", "if"))
        return {};

    // If there is no else branch, or if either branch is not an expression,
    // the type of the 'if' is 'void'.
    if (
        not else_ or
        not isa<Expr>(then) or
        not isa<Expr>(else_.get())
    ) return Build(Types::VoidTy, Expr::SRValue);

    // Otherwise, if either branch is dependent, we can’t determine the type yet.
    auto t = cast<Expr>(then);
    auto e = cast<Expr>(else_.get());
    if (t->type_dependent() or e->type_dependent()) return Build(
        Types::DependentTy,
        Expr::DValue
    );

    // Next, if either branch is 'noreturn', the type of the 'if' is the type
    // of the other branch (unless both are noreturn, in which case the type
    // is just 'noreturn').
    if (t->type == Types::NoReturnTy or e->type == Types::NoReturnTy) {
        bool both = t->type == Types::NoReturnTy and e->type == Types::NoReturnTy;
        return Build(
            both                           ? Types::NoReturnTy
            : t->type == Types::NoReturnTy ? e->type
                                           : t->type,
            both                           ? Expr::SRValue
            : t->type == Types::NoReturnTy ? e->value_category
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
    auto common_ty = t->type == e->type ? t->type : Types::VoidTy;

    // We don’t convert lvalues to rvalues in this case because neither
    // side will actually be used if there is no common type.
    // TODO: If there is no common type, both sides need to be marked
    //       as discarded.
    if (common_ty == Types::VoidTy) return Build(Types::VoidTy, Expr::SRValue);

    // Permitting MRValues here is non-trivial.
    if (common_ty->value_category() == Expr::MRValue) return ICE(
        loc,
        "Sorry, we don’t support returning a value of type '{}' from an '%1(if)' expression yet.",
        common_ty
    );

    // Make sure both sides are rvalues.
    t = LValueToSRValue(t);
    e = LValueToSRValue(e);
    return new (*M) IfExpr(common_ty, Expr::SRValue, cond, t, e, false, loc);
}

auto Sema::BuildLocalDecl(
    ProcScopeInfo& proc,
    Type ty,
    String name,
    Ptr<Expr> init,
    Location loc
) -> LocalDecl* {
    // Deduce the type from the initialiser, if need be.
    if (ty == Types::DeducedTy) {
        if (init) ty = init.get()->type;
        else {
            Error(loc, "Type inference requires an initialiser");
            ty = Types::ErrorDependentTy;
        }
    }

    // Adjust after inference.
    ty = CheckVariableType(ty, loc);

    // Then, perform initialisation.
    //
    // If this fails, the initialiser is simply discarded; we can
    // still continue analysing this though as most of sema doesn’t
    // care about variable initialisers.
    if (auto i = init.get_or_null(); i and not i->dependent())
        if (auto init_expr = PerformVariableInitialisation(ty, i))
            init = init_expr;

    auto param = new (*M) LocalDecl(CheckVariableType(ty, loc), name, proc.proc, init, loc);
    DeclareLocal(param);
    return param;
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
    DeclareLocal(decl);
    return decl;
}

auto Sema::BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr> {
    // If the body is not a block, build an implicit return.
    if (not isa<BlockExpr>(body)) body = BuildReturnExpr(body, body->location(), true);

    // Make sure all paths return a value. First, if the body is
    // 'noreturn', then that means we never actually get here.
    auto body_ret = body->type;
    if (body_ret->dependent() or body_ret == Types::NoReturnTy) return body;

    // Next, a function marked as returning void requires no checking
    // and is allowed to not return at all; invalid return expressions
    // are checked when we first encounter them.
    //
    // We do, however, need to synthesise a return statement in that case.
    if (proc->return_type() == Types::VoidTy) {
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
    if (value.present() and value.get()->dependent())
        return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);

    // Perform return type deduction.
    auto proc = curr_proc().proc;
    if (proc->return_type() == Types::DeducedTy) {
        auto proc_type = proc->proc_type();
        Type deduced = Types::VoidTy;
        if (auto val = value.get_or_null()) deduced = val->type;
        proc->type = ProcType::AdjustRet(*M, proc_type, deduced);
    }

    // Or complain if the type doesn’t match.
    else if (not proc->return_type()->dependent()) {
        Type ret = value.invalid() ? Types::VoidTy : value.get()->type;
        if (ret != proc->return_type()) Error(
            loc,
            "Return type '{}' does not match procedure return type '{}'",
            ret,
            proc->return_type()
        );
    }

    // Stop here if the procedure type is still dependent.
    if (proc->return_type()->dependent())
        return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);

    // Perform any necessary conversions.
    if (auto val = value.get_or_null()) {
        if (val->type == Types::VoidTy) {
            // Nop.
        } else if (val->type == Types::IntTy or isa<IntType>(val->type)) {
            value = LValueToSRValue(val);
        } else {
            ICE(loc, "Cannot compile this return type yet: {}", val->type);
        }
    }

    return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildTypeExpr(Type ty, Location loc) -> TypeExpr* {
    return new (*M) TypeExpr(ty, loc);
}

auto Sema::BuildUnaryExpr(Tk op, Expr* operand, bool postfix, Location loc) -> Ptr<UnaryExpr> {
    auto Build = [&](Type ty, ValueCategory cat) {
        return new (*M) UnaryExpr(ty, cat, op, operand, postfix, loc);
    };

    if (operand->dependent()) return Build(Types::DependentTy, Expr::DValue);
    if (postfix) return ICE(loc, "Postfix unary operators are not yet implemented");

    // Handle prefix operators.
    switch (op) {
        default: Unreachable("Invalid unary operator: {}", op);

        // Boolean negation.
        case Tk::Not: {
            if (not MakeSRValue(Types::BoolTy, operand, "Operand", "not")) return {};
            return Build(Types::BoolTy, Expr::SRValue);
        }

        // Arithmetic operators.
        case Tk::Minus:
        case Tk::Plus:
        case Tk::Tilde: {
            if (not MakeSRValue(Types::IntTy, operand, "Operand", Spelling(op))) return {};
            return Build(Types::IntTy, Expr::SRValue);
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
            if (operand->type != Types::IntTy) return Error(
                operand->location(),
                "Operand of '{}' must be an integer",
                Spelling(op)
            );

            // Result is an lvalue.
            return Build(Types::IntTy, Expr::LValue);
        }
    }
}

auto Sema::BuildWhileStmt(Expr* cond, Stmt* body, Location loc) -> Ptr<WhileStmt> {
    if (
        not cond->dependent() and
        not MakeSRValue(Types::BoolTy, cond, "Condition", "while")
    ) return {};
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
    all_scopes.push_back(std::make_unique<Scope>(nullptr));
    scope_stack.push_back(all_scopes.back().get());

    // Take ownership of any resources of the parsed modules.
    for (auto& p : parsed_modules) {
        M->add_allocator(std::move(p->string_alloc));
        M->add_integer_storage(std::move(p->integers));
    }

    // Collect all statements and translate them.
    M->initialiser_proc->scope = global_scope();
    EnterProcedure _{*this, M->initialiser_proc};
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, Location{});

    // File scope block should never be dependent.
    M->file_scope_block->set_dependence(Dependence::None);
    M->initialiser_proc->finalise(BuildProcBody(M->initialiser_proc, M->file_scope_block), curr_proc().locals);
}

void Sema::TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed) {
    // Translate object declarations first since they may be out of order.
    //
    // Note that only the declaration part of definitions is translated here, e.g.
    // for a ProcDecl, we only translate, not the body; the latter is handled later
    // on.
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
auto Sema::TranslateAssertExpr(ParsedAssertExpr* parsed) -> Ptr<Expr> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    Ptr<Expr> msg;
    if (auto m = parsed->message.get_or_null()) msg = TRY(TranslateExpr(m));
    return BuildAssertExpr(cond, msg, parsed->loc);
}

auto Sema::TranslateBinaryExpr(ParsedBinaryExpr* expr) -> Ptr<Expr> {
    // Translate LHS and RHS.
    auto lhs = TRY(TranslateExpr(expr->lhs));
    auto rhs = TRY(TranslateExpr(expr->rhs));
    return BuildBinaryExpr(expr->op, lhs, rhs, expr->loc);
}

auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed) -> Ptr<Expr> {
    EnterScope scope{*this};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, parsed->stmts());
    return BuildBlockExpr(scope.get(), stmts, parsed->loc);
}

auto Sema::TranslateBoolLitExpr(ParsedBoolLitExpr* parsed) -> Ptr<Expr> {
    return new (*M) BoolLitExpr(parsed->value, parsed->loc);
}

auto Sema::TranslateCallExpr(ParsedCallExpr* parsed) -> Ptr<Expr> {
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
auto Sema::TranslateDeclRefExpr(ParsedDeclRefExpr* parsed) -> Ptr<Expr> {
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

    // Ignore this if there was a problem w/ the procedure type.
    if (auto proc = dyn_cast<ParsedProcDecl>(parsed)) {
        if (not d) return nullptr;
        return TranslateProc(cast<ProcDecl>(d), proc);
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

auto Sema::TranslateEvalExpr(ParsedEvalExpr* parsed) -> Ptr<Expr> {
    auto arg = TRY(TranslateStmt(parsed->expr));
    return BuildEvalExpr(arg, parsed->loc);
}

auto Sema::TranslateFieldDecl(ParsedFieldDecl*) -> Decl* {
    Unreachable("Handled as part of StructDecl translation");
}

auto Sema::TranslateIfExpr(ParsedIfExpr* parsed) -> Ptr<Expr> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    auto then = TRY(TranslateStmt(parsed->then));
    Ptr else_ = parsed->else_ ? TRY(TranslateStmt(parsed->else_.get())) : nullptr;
    return BuildIfExpr(cond, then, else_, parsed->loc);
}

auto Sema::TranslateIntLitExpr(ParsedIntLitExpr* parsed) -> Ptr<Expr> {
    // If the value fits in an 'int', its type is 'int'.
    auto val = parsed->storage.value();
    auto small = val.tryZExtValue();
    if (small.has_value()) return new (*M) IntLitExpr(
        Types::IntTy,
        M->store_int(APInt(u32(Types::IntTy->size(*M).bits()), u64(*small), true)),
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
            "which is smaller than an %6(i{}), which would "
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

auto Sema::TranslateMemberExpr(ParsedMemberExpr* parsed) -> Ptr<Expr> {
    auto base = TRY(TranslateExpr(parsed->base));
    using AK = BuiltinMemberAccessExpr::AccessKind;
    static constexpr auto AlreadyDiagnosed = AK(255);
    auto kind = [&] -> Opt<AK> {
        using Switch = llvm::StringSwitch<Opt<AK>>;
        if (isa<TypeExpr>(base)) return Switch(parsed->member)
            .Case("align", AK::TypeAlign)
            .Case("arrsize", AK::TypeArraySize)
            .Case("bits", AK::TypeBits)
            .Case("bytes", AK::TypeBytes)
            .Case("name", AK::TypeName)
            .Case("size", AK::TypeBytes)
            .Default(std::nullopt);

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

auto Sema::TranslateParenExpr(ParsedParenExpr* parsed) -> Ptr<Expr> {
    return new (*M) ParenExpr(TRY(TranslateExpr(parsed->inner)), parsed->loc);
}

auto Sema::TranslateLocalDecl(ParsedLocalDecl* parsed) -> Decl* {
    return BuildLocalDecl(
        curr_proc(),
        TranslateType(parsed->type),
        parsed->name,
        parsed->init ? TranslateExpr(parsed->init.get()) : nullptr,
        parsed->loc
    );
}

auto Sema::TranslateProc(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    // Translate the body if there is one.
    if (parsed->body) {
        EnterProcedure _{*this, decl};
        auto res = TranslateProcBody(decl, parsed);
        if (res.invalid()) decl->set_errored();
        else decl->finalise(res, curr_proc().locals);
    }

    return decl;
}

auto Sema::TranslateProcBody(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<Stmt> {
    Assert(parsed->body);

    // Translate parameters.
    auto ty = decl->proc_type();
    for (auto [i, pair] : enumerate(zip(ty->params(), parsed->params()))) {
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
    auto body = TranslateExpr(parsed->body.get());
    if (body.invalid()) {
        // If we’re attempting to deduce the return type of this procedure,
        // but the body contains an error, then set the return type to errored.
        if (decl->return_type() == Types::DeducedTy)
            decl->type = ProcType::AdjustRet(*M, decl->proc_type(), Types::ErrorDependentTy);
        return nullptr;
    }

    return BuildProcBody(decl, body.get());
}

auto Sema::TranslateProcDecl(ParsedProcDecl*) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

/// Perform initial type checking on a procedure, enough to enable calls
/// to it to be translated, but without touching its body, if there is one.
auto Sema::TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    EnterScope scope{*this, true};
    SmallVector<TemplateTypeDecl*> ttds;

    // Create the declaration. A top-level procedure is not considered
    // 'nested' inside the initialiser procedure, which means that this
    // is only local if the procedure stack contains at least 3 entries
    // (the initialiser, our parent, and us).
    auto attrs = parsed->type->attrs;
    auto type = TranslateProcType(parsed->type, &ttds);
    auto proc = ProcDecl::Create(
        *M,
        type,
        parsed->name,
        attrs.extern_ ? Linkage::Imported : Linkage::Internal,
        attrs.nomangle or attrs.native ? Mangling::None : Mangling::Source,
        proc_stack.size() >= 3 ? proc_stack.back()->proc : nullptr,
        parsed->loc,
        ttds
    );

    // Diagnose invalid combinations of attributes.
    if (attrs.native and attrs.nomangle) Error(
        parsed->loc,
        "'%1(native)' procedures should not be declared '%1(nomangle)'"
    );

    // Add the procedure to the module and the parent scope.
    proc_decl_map[parsed] = proc;
    proc->scope = scope.get();
    AddDeclToScope(scope.get()->parent(), proc);
    return proc;
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedStmt* parsed) -> Ptr<Stmt> {
    switch (parsed->kind()) { // clang-format off
        using K = ParsedStmt::Kind;
#       define PARSE_TREE_LEAF_TYPE(node) case K::node: return BuildTypeExpr(TranslateType(parsed), parsed->loc);
#       define PARSE_TREE_LEAF_NODE(node) case K::node: return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed));
#       include "srcc/ParseTree.inc"



    } // clang-format on

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
    Size size{};
    Align align{1};
    SmallVector<FieldDecl*> fields;
    for (auto f : parsed->fields()) {
        auto ty = CheckVariableType(TranslateType(f->type), f->loc);

        // If the field’s type is invalid, we can’t query any of its
        // properties, so just insert an error field and continue.
        if (ty == Types::ErrorDependentTy) {
            fields.push_back(new (*M) FieldDecl(ty, size, f->name, f->loc));
            continue;
        }

        // Otherwise, add the field and adjust our size and alignment.
        // TODO: Optimise layout if this isn’t meant for FFI.
        size.align(ty->align(*M));
        fields.push_back(new (*M) FieldDecl(ty, size, f->name, f->loc));
        size += ty->size(*M);
        align = std::max(align, ty->align(*M));
    }

    // Finally, mark the struct as complete.
    s->finalise(fields, size, align);
    return decl;
}

auto Sema::TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto ty = StructType::Create(
        *M,
        parsed->name,
        u32(parsed->fields().size()),
        parsed->loc
    );

    AddDeclToScope(curr_scope(), ty->decl());
    return ty->decl();
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed) -> Ptr<Expr> {
    return StrLitExpr::Create(*M, parsed->value, parsed->loc);
}

/// Translate a return expression.
auto Sema::TranslateReturnExpr(ParsedReturnExpr* parsed) -> Ptr<Expr> {
    Ptr<Expr> ret_val;
    if (parsed->value.present()) ret_val = TranslateExpr(parsed->value.get());
    return BuildReturnExpr(ret_val.get_or_null(), parsed->loc, false);
}

auto Sema::TranslateUnaryExpr(ParsedUnaryExpr* parsed) -> Ptr<Expr> {
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
        Error(parsed->loc, "The maximum integer type is %6(i{})", IntType::MaxBits);
        return IntType::Get(*M, IntType::MaxBits);
    }
    return IntType::Get(*M, parsed->bit_width);
}

auto Sema::TranslateNamedType(ParsedDeclRefExpr* parsed) -> Type {
    auto res = LookUpName(curr_scope(), parsed->names(), parsed->loc);
    if (not res) return Types::ErrorDependentTy;

    // Template type.
    if (auto ttd = dyn_cast<TemplateTypeDecl>(res.decls.front()))
        return TemplateType::Get(*M, ttd);

    // Type decl (struct or type alias).
    if (auto s = dyn_cast<TypeDecl>(res.decls.front()))
        return s->type;

    Error(parsed->loc, "'{}' does not name a type", utils::join(parsed->names(), "::"));
    Note(res.decls.front()->location(), "Declared here");
    return Types::ErrorDependentTy;
}

auto Sema::TranslateProcType(
    ParsedProcType* parsed,
    SmallVectorImpl<TemplateTypeDecl*>* ttds
) -> ProcType* {
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
        return ProcType::GetInvalid(*M);
    }

    // We may have to handle template parameters here.
    //
    // In Source, template parameters are declared using a template type
    // token, e.g. '$type', in the parameter list of a function declaration;
    // these declarations serve a twofold purpose:
    //
    //    1. To introduce a new template parameter.
    //    2. To signify where that parameter should be deduced.
    //
    // The first is self-explanatory. The second has to do with the fact that
    // we sometimes want to enforce that certain arguments have the exact same
    // type, and sometimes, we just want the parameters to have the same type,
    // i.e. we want to permit implicit conversions at the call site.
    //
    // To accomplish this, a template type is deduced from a parameter, iff that
    // parameter’s type is a template type that uses the '$' sigil. Consider:
    //
    //   proc foo (T a, $T b, $T c) { ... }
    //
    // In this procedure, all of 'a', 'b', and 'c' will have the same type, that
    // being 'T'. However, what type 'T' actually is will be deduced from the
    // arguments passed in for 'b' and 'c' only. Furthermore, template parameters
    // are order-independent, since we sometimes want to be able to deduce a
    // parameter at an occurrence other than its first. Thus, consider the call:
    //
    //   foo(1 as i16, 2 as i32, 3 as i32)
    //
    // This succeeds because the type of 'T' is deduced to be 'i32' for both 'b'
    // and 'c', and the 'i16' passed for 'a' is converted to 'i32' accordingly;
    // next, consider:
    //
    //   foo(1 as i16, 2 as i16, 3 as i32)
    //
    // This call fails because we can’t deduce a type for 'T' that satisfies both
    // 'b' and 'c'.
    if (ttds) {
        struct TemplateDecl {
            String name;
            Location loc;
            SmallVector<u32, 1> deduced_indices;
        };

        // Template decl by template parameter name.
        StringMap<TemplateDecl> template_param_decls{};

        // First, do a prescan to collect template type defs.
        for (auto [i, p] : enumerate(parsed->param_types())) {
            if (auto ptt = dyn_cast<ParsedTemplateType>(p.type)) {
                auto& td = template_param_decls[ptt->name];
                if (td.deduced_indices.empty()) {
                    td.loc = ptt->loc;
                    td.name = ptt->name;
                }
                td.deduced_indices.push_back(u32(i));
            }
        }

        // Then build all the template type decls.
        for (const auto& entry : template_param_decls) {
            auto& td = entry.getValue();
            Assert(not td.deduced_indices.empty(), "Undeduced parameter?");
            auto ttd = ttds->emplace_back(TemplateTypeDecl::Create(*M, td.name, td.deduced_indices, td.loc));
            AddDeclToScope(curr_scope(), ttd);
        }
    }

    // Then, compute the actual parameter types.
    SmallVector<ParamTypeData, 10> params;
    for (auto a : parsed->param_types()) {
        // Template types encountered here introduce a template parameter
        // instead of referencing one, so we process them manually as the
        // usual translation machinery isn’t equipped to handle template
        // definitions.
        //
        // At this point, the only thing in the scope here should be the
        // template parameters, so lookup should never find anything else.
        if (auto ptt = dyn_cast<ParsedTemplateType>(a.type); ptt and ttds) {
            auto res = LookUpUnqualifiedName(curr_scope(), ptt->name, true);
            Assert(res.successful(), "Template parameter should have been declared earlier");
            Assert(res.decls.size() == 1 and isa<TemplateTypeDecl>(res.decls.front()));
            params.emplace_back(a.intent, TemplateType::Get(*M, cast<TemplateTypeDecl>(res.decls.front())));
        }

        // Anything else is parsed as a regular type.
        //
        // If this is a template parameter that occurs in a context where
        // it is not allowed (e.g. in a function type that is not part of
        // a procedure definition), type translation will handle that case
        // and return an error.
        else {
            auto ty = TranslateType(a.type);
            if (ty == Types::DeducedTy) {
                Error(a.type->loc, "'{}' is not a valid type for a procedure argument", Types::DeducedTy);
                ty = Types::ErrorDependentTy;
            }
            params.emplace_back(a.intent, ty);
        }
    }

    return ProcType::Get(
        *M,
        TranslateType(parsed->ret_type),
        params,
        parsed->attrs.native ? CallingConvention::Native : CallingConvention::Source
    );
}

auto Sema::TranslateSliceType(ParsedSliceType* parsed) -> Type {
    auto ty = CheckVariableType(TranslateType(parsed->elem), parsed->loc);
    return SliceType::Get(*M, ty);
}

auto Sema::TranslateTemplateType(ParsedTemplateType* parsed) -> Type {
    Error(parsed->loc, "A template type declaration is only allowed in the parameter list of a procedure");
    return Types::ErrorDependentTy;
}

auto Sema::TranslateType(ParsedStmt* parsed) -> Type {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
        case K::BuiltinType: return TranslateBuiltinType(cast<ParsedBuiltinType>(parsed));
        case K::IntType: return TranslateIntType(cast<ParsedIntType>(parsed));
        case K::SliceType: return TranslateSliceType(cast<ParsedSliceType>(parsed));
        case K::TemplateType: return TranslateTemplateType(cast<ParsedTemplateType>(parsed));
        case K::DeclRefExpr: return TranslateNamedType(cast<ParsedDeclRefExpr>(parsed));
        case K::ProcType: return TranslateProcType(cast<ParsedProcType>(parsed));
        default:
            Error(parsed->loc, "Expected type");
            return Types::ErrorDependentTy;
    }
}
