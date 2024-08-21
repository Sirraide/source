module;

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <print>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.utils;
import srcc.ast;
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
    // are usually allowed, but we forbid redeclaring e.g. parameters.
    auto& ds = scope->decls[d->name];
    if (not ds.empty() and isa<ParamDecl>(d)) {
        Error(d->location(), "Redeclaration of parameter '{}'", d->name);
        Note(ds.front()->location(), "Previous declaration was here");
    } else {
        ds.push_back(d);
    }
}

auto Sema::AdjustVariableType(Type ty, Location loc) -> Type {
    // 'noreturn' and 'type' are not a valid type for a variable.
    if (ty == Types::NoReturnTy or ty == Types::TypeTy) {
        Error(loc, "Cannot declare a variable of type '{}'", ty.print(ctx.use_colours()));
        return Types::ErrorDependentTy;
    }

    if (ty == Types::UnresolvedOverloadSetTy) {
        Error(loc, "Unresolved overload set in parameter declaration");
        return Types::ErrorDependentTy;
    }

    return ty;
}

auto Sema::ApplyConversionSequence(Expr* e, ConversionSequence& seq) -> Expr* {
    for (auto& c : seq) {
        switch (c.kind) {
            case Conversion::Kind::LValueToSRValue:
                e = LValueToSRValue(e);
                continue;
            case Conversion::Kind::SelectOverload:
                e = CreateReference(cast<OverloadSetExpr>(e)->overloads()[c.index], e->location()).get();
                continue;
        }
        Unreachable("Invalid conversion");
    }
    return e;
}

auto Sema::CreateReference(Decl* d, Location loc) -> Ptr<Expr> {
    switch (d->kind()) {
        default: return ICE(d->location(), "Cannot build a reference to this declaration");
        case Stmt::Kind::ProcDecl: return new (*M) ProcRefExpr(cast<ProcDecl>(d), loc);
        case Stmt::Kind::LocalDecl:
        case Stmt::Kind::ParamDecl:
            return new (*M) LocalRefExpr(cast<LocalDecl>(d), loc);
    }
}

void Sema::DeclareLocal(LocalDecl* d) {
    Assert(d->parent == curr_proc().proc, "Must EnterProcedure befor adding a local variable");
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
                auto it = imported_modules.find(first);
                if (it == imported_modules.end()) return res;
                in_scope = &it->second->exports;
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

    return LookupResult(name);
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

void Sema::ReportLookupFailure(const LookupResult& result, Location loc) {
    switch (result.result) {
        using enum LookupResult::Reason;
        case Success: Unreachable("Diagnosing a successful lookup?");
        case NotFound: Error(loc, "Unknown symbol '{}'", result.name); break;
        case Ambiguous: {
            Error(loc, "Ambiguous symbol '{}'", result.name);
            for (auto d : result.decls) Note(d->location(), "Candidate here");
        } break;
        case NonScopeInPath: {
            Error(loc, "Invalid left-hand side for '::'");
            Note(result.decls.front()->location(), "'{}' does not contain a scope", result.name);
        } break;
    }
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
    using enum utils::Colour;
    utils::Colours C{ctx.use_colours()};
    std::string message = std::format("{}Candidates:\n", C(Bold));

    // Compute the width of the number field.
    u32 width = u32(std::to_string(candidates.size()).size());

    // First, print all overloads.
    u32 term_width = ctx.diags().cols();
    for (auto [i, c] : enumerate(candidates)) {
        message += C(Bold);

        // Check if the location is valid.
        auto loc = c.location();
        auto lc = loc.seek_line_column(ctx);
        if (lc) {
            // We have a location. Compute the width of everything so we can print
            // it in a single line if it fits. We need to print the type twice since
            // the ANSI escape codes would throw everything off.
            auto type_width = c.type_for_diagnostic()->print(false).size();
            auto start = std::format(
                "  {:>{}}. ",
                i + 1,
                width
            );

            // Technically, we don’t have to do this calculation if the terminal width
            // is zero, but this is really not a place where we have to worry about
            // performance...
            auto total =
                type_width +
                start.size() +
                4 + // ' at '
                2 + // ':' twice
                ctx.file(loc.file_id)->name().size() +
                std::to_string(lc->line).size() +
                std::to_string(lc->col).size();

            // It fits!
            if (term_width == 0 or total < term_width) {
                message += std::format(
                    "{}{}{} {}at {}{}:{}:{}\n",
                    start,
                    C(Reset),
                    c.type_for_diagnostic()->print(C.use_colours),
                    C(Bold),
                    C(Reset),
                    ctx.file(loc.file_id)->name(),
                    lc->line,
                    lc->col
                );

                continue;
            }
        }

        // It doesn’t, or we have no location. Print the type first.
        if (i != 0) message += "\n";
        message += std::format(
            "  {:>{}}. {}{}\n",
            i + 1,
            width,
            C(Reset),
            c.type_for_diagnostic()->print(C.use_colours)
        );

        // And the location on the next line if there is one.
        if (lc) {
            message += std::format(
                "  {}{:>{}}  at {}{}:{}:{}\n",
                C(Bold),
                "",
                width,
                C(Reset),
                ctx.file(loc.file_id)->name(),
                lc->line,
                lc->col
            );
        }
    }

    message += std::format("\n{}Failure Reason:", C(Bold));

    // For each overload, print why there was an issue.
    for (auto [i, c] : enumerate(candidates)) {
        message += std::format("\n  {}{:>{}}. {}", C(Bold), i + 1, width, C(Reset));
        auto V = utils::Overloaded{// clang-format off
            [&] (const Candidate::Viable& v) {
                // If the badness is equal to the final badness,
                // then this candidate was ambiguous. Otherwise,
                // another candidate was simply better.
                message += v.badness == final_badness
                    ? "Matches as well as another candidate"sv
                    : "Another candidate was better"sv;
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
            [&](Candidate::TypeMismatch t) {
                message += std::format(
                    "Arg #{} should be '{}' but was '{}'",
                    t.mismatch_index + 1,
                    c.type_for_diagnostic()->params()[t.mismatch_index].print(C.use_colours),
                    call_args[t.mismatch_index]->type.print(C.use_colours)
                );
            },
            [&](Candidate::InvalidTemplate) {
                message += "Template argument deduction failed";
                const auto& ti = c.proc.get<Candidate::TemplateInfo>();
                auto TV = utils::Overloaded {
                    [](const TempSubstRes::Success&) { Unreachable("Invalid template even though deduction succeeded?"); },
                    [](TempSubstRes::Error) { Unreachable("Should have bailed out earlier on hard error"); },
                    [&](TempSubstRes::DeductionFailed f) {
                        message += std::format(
                            "In param #{}: cannot deduce ${} in {} from {}",
                            f.param_index + 1,
                            f.ttd->name,
                            ti.pattern->params()[f.param_index]->type.print(C.use_colours),
                            call_args[f.param_index]->type.print(C.use_colours)
                        );
                    },

                    [&](const TempSubstRes::DeductionAmbiguous& a) {
                        message += std::format(
                            "Template deduction mismatch for parameter {}${}{}:\n"
                            "        #{}: Deduced as {}\n"
                            "        #{}: Deduced as {}",
                            C(Yellow),
                            a.ttd->name,
                            C(Reset),
                            a.first,
                            a.first_type->print(C.use_colours),
                            a.second,
                            a.second_type->print(C.use_colours)
                        );
                    }
                };
                ti.res.data.visit(TV);
            },

            [&](const Candidate::NestedResolutionFailure& n) {
                Todo("Report this");
            }
        }; // clang-format on
        c.status.visit(V);
    }

    ctx.diags().report(Diagnostic{
        Diagnostic::Level::Error,
        call_loc,
        std::format("No matching overload for call to '{}{}{}'", C(Green), candidates.front().name(), C(Reset)),
        std::move(message),
    });
}

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, Location loc) -> BlockExpr* {
    return BlockExpr::Create(
        *M,
        scope,
        stmts,
        loc
    );
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
                if (not isa<StrLitExpr>(arg) and arg->type != Types::IntTy) {
                    return Error(
                        arg->location(),
                        "__srcc_print only accepts string literals and integers, but got {}",
                        arg->type.print(ctx.use_colours())
                    );
                }

                if (arg->type == Types::IntTy) arg = LValueToSRValue(arg);
            }
            return BuiltinCallExpr::Create(*M, builtin, Types::VoidTy, actual_args, call_loc);
        }
    }

    Unreachable("Invalid builtin type: {}", +builtin);
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

    // Calls with dependent arguments are checked when they’re instantiated.
    if (rgs::any_of(args, [](Expr* e) { return e->dependent(); }))
        return BuildDependentCallExpr(callee_expr);

    // Check that we can call this at all.
    auto callee_no_parens = callee_expr->strip_parens();
    if (not isa<OverloadSetExpr, ProcRefExpr>(callee_no_parens)) {
        if (callee_expr->dependent()) return BuildDependentCallExpr(callee_expr);
        return Error(callee_expr->location(), "Cannot call non-procedure");
    }

    // Perform overloading.
    //
    // Since this may also entail template substitution etc. we always
    // treat calls as requiring overload resolution even if there is
    // only a single ‘overload’.
    //
    // Source does have a restricted form of SFINAE: deduction failure
    // and deduction failure only is not an error. Random errors during
    // deduction are hard errors.
    // All candidates, whether viable or not.
    SmallVector<Candidate, 4> candidates;
    bool dependent = false;

    // Add a candidate to the overload set.
    auto AddCandidate = [&](ProcDecl* proc) -> bool {
        if (not proc->is_template()) {
            candidates.emplace_back(proc);
            return true;
        }

        // Collect the types of all arguments (and their locations for
        // diagnostics) for substitution.
        SmallVector<TypeLoc, 6> types;
        for (auto arg : args) types.emplace_back(arg->type, arg->location());

        // Perform template substitution.
        auto res = SubstituteTemplate(proc, types);
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

        // Argument count mismatch is not allowed.
        //
        // TODO: Default arguments and C-style variadic functions.
        if (args.size() != params.size()) {
            c.status = Candidate::ArgumentCountMismatch{};
            return true;
        }

        // Check that we can initialise each parameter with its
        // corresponding argument.
        for (auto [i, a] : enumerate(args)) {
            auto p = params[i];
            auto& conv = c.status.get<Candidate::Viable>().conversions.emplace_back();

            // Type matches exactly.
            if (a->type == p) {
                // We currently can only handle srvalues here.
                if (a->type->value_category() != ValueCategory::SRValue) {
                    ICE(
                        a->location(),
                        "Sorry, we currently don’t support arguments of type {}",
                        a->type.print(ctx.use_colours())
                    );
                    return false;
                }

                // Convert lvalues to srvalues here.
                if (a->lvalue()) conv.push_back(Conversion::LValueToSRValue());
            }

            // Type is an overload set; attempt to convert it.
            //
            // This is *not* the same algorithm as overload resolution, because
            // the types must match exactly here, and we also need to check the
            // return type.
            else if (a->type == Types::UnresolvedOverloadSetTy) {
                auto p_proc_type = dyn_cast<ProcType>(p.ptr());
                if (not p_proc_type) {
                    c.status = Candidate::TypeMismatch{u32(i)};
                    return true;
                }

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
                    if (o->type == p) {
                        conv.push_back(Conversion::SelectOverload(u16(j)));
                        goto next_param;
                    }
                }

                // Otherwise, we need to try and instantiate templates in this overload set.
                for (auto o : overloads) {
                    if (not o->is_template()) continue;
                    Todo("Instantiate template in nested overload set");
                }

                // None of the overloads matched.
                c.status = Candidate::NestedResolutionFailure{u32(i)};
                return true;
            }

            // Otherwise, the types don’t match.
            else {
                c.status = Candidate::TypeMismatch{u32(i)};
                return true;
            }
        next_param:
        }

        // All parameters match.
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
                subst.args
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

        // Finally, create the call.
        return CallExpr::Create(
            *M,
            final_callee->return_type(),
            CreateReference(final_callee, callee_expr->location()).get(),
            actual_args,
            loc
        );
    }

    // Overload resolution failed :(.
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
    ty = AdjustVariableType(ty, loc);

    // Then, perform initialisation.
    if (auto i = init.get_or_null()) {
        switch (ty->value_category()) {
            case ValueCategory::MRValue: Todo("Initialise MRValue");
            case ValueCategory::LValue: Todo("Initialise LValue");

            // Easy case: this behaves like an integer.
            case ValueCategory::SRValue:
                if (not i->dependent()) {
                    if (i->type != ty) {
                        Error(
                            i->location(),
                            "Initialiser of type '{}' does not match variable type '{}'",
                            i->type.print(ctx.use_colours()),
                            ty.print(ctx.use_colours())
                        );
                        init = nullptr;
                    } else {
                        init = LValueToSRValue(i);
                    }
                }
                break;

            // Dependent. We’ll come back to this later.
            case ValueCategory::DValue:
                break;
        }
    }

    auto param = new (*M) LocalDecl(AdjustVariableType(ty, loc), name, proc.proc, init, loc);
    DeclareLocal(param);
    return param;
}

auto Sema::BuildParamDecl(
    ProcScopeInfo& proc,
    Type ty,
    String name,
    Location loc
) -> ParamDecl* {
    auto param = new (*M) ParamDecl(AdjustVariableType(ty, loc), name, proc.proc, loc);
    DeclareLocal(param);
    return param;
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
        proc->type = ProcType::Get(
            *M,
            deduced,
            proc_type->params(),
            proc_type->cconv(),
            proc_type->variadic()
        );
    }

    // Or complain if the type doesn’t match.
    else if (not proc->return_type()->dependent()) {
        Type ret = value.invalid() ? Types::VoidTy : value.get()->type;
        if (ret != proc->return_type()) Error(
            loc,
            "Return type '{}' does not match procedure return type '{}'",
            ret.print(ctx.use_colours()),
            proc->return_type().print(ctx.use_colours())
        );
    }

    // Stop here if the procedure type is still dependent.
    if (proc->return_type()->dependent())
        return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);

    // Perform any necessary conversions.
    if (auto val = value.get_or_null()) {
        if (val->type == Types::VoidTy) {
            // Nop.
        } else if (val->type == Types::IntTy) {
            value = LValueToSRValue(val);
        } else {
            ICE(loc, "Cannot compile this return type yet: {}", val->type.print(ctx.use_colours()));
        }
    }

    return new (*M) ReturnExpr(value.get_or_null(), loc, implicit);
}

auto Sema::BuildTypeExpr(Type ty, Location loc) -> TypeExpr* {
    return new (*M) TypeExpr(ty, loc);
}

// ============================================================================
//  Translation Driver
// ============================================================================
auto Sema::Translate(ArrayRef<ParsedModule::Ptr> modules) -> TranslationUnit::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};
    S.M = TranslationUnit::Create(first->context(), first->name, first->is_module);
    S.parsed_modules = modules;
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

    // Resolve imports.
    for (auto& p : parsed_modules)
        for (auto& i : p->imports)
            M->imports[i.linkage_name] = {nullptr, i.loc, i.import_name};

    // FIXME: C++ headers should all be imported at the same time; it really
    // doesn’t make sense to import them separately...
    bool errored = false;
    for (auto& i : M->imports) {
        auto res = ImportCXXHeader(i.second.import_location, M->save(i.first()));
        if (not res) {
            errored = true;
            continue;
        }

        i.second.imported_module = std::move(res);
        imported_modules[i.second.import_name] = i.second.imported_module.get();
    }

    // Don’t attempt anything else if there was a problem.
    if (errored) return;

    // Collect all statements and translate them.
    EnterProcedure _{*this, M->initialiser_proc};
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, Location{});

    // File scope block should never be dependent.
    M->file_scope_block->set_dependence(Dependence::None);
    M->initialiser_proc->finalise(BuildProcBody(M->initialiser_proc, M->file_scope_block), {});
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
auto Sema::TranslateBinaryExpr(ParsedBinaryExpr*) -> Ptr<Expr> {
    Todo();
}

auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed) -> Ptr<BlockExpr> {
    EnterScope scope{*this};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, parsed->stmts());
    return BuildBlockExpr(scope.get(), stmts, parsed->loc);
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
    if (auto proc = dyn_cast<ParsedProcDecl>(d)) return TranslateProcDeclInitial(proc);
    return std::nullopt;
}

/// Translate the body of a declaration.
auto Sema::TranslateEntireDecl(Decl* d, ParsedDecl* parsed) -> Ptr<Decl> {
    // Ignore this if there was a problem w/ the procedure type.
    if (auto proc = dyn_cast<ParsedProcDecl>(parsed)) {
        if (not d) return nullptr;
        return TranslateProc(cast<ProcDecl>(d), proc);
    }

    // No special handling for anything else.
    auto res = TranslateStmt(parsed);
    if (res.invalid()) return nullptr;
    return cast<Decl>(res.get());
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
        using enum utils::Colour;
        utils::Colours C{ctx.use_colours()};
        Error(parsed->loc, "Sorry, we can’t compile a number that big :(");
        Note(
            parsed->loc,
            "The maximum supported integer type is {}, "
            "which is smaller than an {}i{}{}, which would "
            "be required to store a value of {}",
            IntType::Get(*M, IntType::MaxBits)->print(C.use_colours),
            C(Cyan),
            bits,
            C(Reset),
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
    if (isa<SliceType>(base->type)) {
        if (parsed->member == "data") return SliceDataExpr::Create(*M, base, parsed->loc);
        return Error(parsed->loc, "Slice has no member named '{}'", parsed->member);
    }

    return Error(parsed->loc, "Attempt to access member of type {}", base->type.print(true));
}

auto Sema::TranslateParenExpr(ParsedParenExpr*) -> Ptr<Expr> {
    Todo();
}

auto Sema::TranslateLocalDecl(ParsedLocalDecl* parsed) -> LocalDecl* {
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
    EnterScope scope{*this, decl->scope};
    Assert(parsed->body);

    // Translate parameters.
    auto ty = decl->proc_type();
    for (auto [i, pair] : vws::zip(ty->params(), parsed->params()) | vws::enumerate) {
        auto [param_ty, parsed_decl] = pair;
        BuildParamDecl(curr_proc(), param_ty, parsed_decl->name, parsed_decl->loc);
    }

    // Translate body.
    auto body = TranslateExpr(parsed->body);
    if (body.invalid()) return nullptr;
    return BuildProcBody(decl, body.get());
}

/// This is only called if we’re asked to translate a procedure by the expression
/// parser; actual procedure translation is handled elsewhere; only return a reference
/// here.
auto Sema::TranslateProcDecl(ParsedProcDecl*) -> Ptr<Expr> {
    Unreachable("Translating declaration as expression?");
    /*auto it = proc_decl_map.find(parsed);

    // This can happen if the procedure errored somehow.
    if (it == proc_decl_map.end()) return nullptr;

    // The procedure has already been created.
    return CreateReference(it->second, parsed->loc);*/
}

/// Perform initial type checking on a procedure, enough to enable calls
/// to it to be translated, but without touching its body, if there is one.
auto Sema::TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    EnterScope scope{*this, true};
    SmallVector<TemplateTypeDecl*> ttds;
    auto type = TranslateProcType(parsed->type, &ttds);
    auto proc = ProcDecl::Create(
        *M,
        type,
        parsed->name,
        Linkage::Internal,
        Mangling::Source,
        proc_stack.empty() ? nullptr : proc_stack.back()->proc,
        parsed->loc,
        ttds
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
#       define PARSE_TREE_LEAF_NODE(node) \
            case K::node: return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed));
#       include "srcc/ParseTree.inc"













    } // clang-format on

    Unreachable("Invalid parsed statement kind: {}", +parsed->kind());
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed) -> Ptr<StrLitExpr> {
    return StrLitExpr::Create(*M, parsed->value, parsed->loc);
}

/// Translate a return expression.
auto Sema::TranslateReturnExpr(ParsedReturnExpr* parsed) -> Ptr<Expr> {
    Ptr<Expr> ret_val;
    if (parsed->value.present()) ret_val = TranslateExpr(parsed->value.get());
    return BuildReturnExpr(ret_val.get_or_null(), parsed->loc, false);
}

auto Sema::TranslateUnaryExpr(ParsedUnaryExpr* parsed) -> Ptr<Expr> {
    Todo();
}

// ============================================================================
//  Translation of Types
// ============================================================================
auto Sema::TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type {
    return parsed->ty;
}

auto Sema::TranslateNamedType(ParsedDeclRefExpr* parsed) -> Type {
    auto res = LookUpName(curr_scope(), parsed->names(), parsed->loc);
    if (not res) return Types::ErrorDependentTy;

    // Currently, the only type declaration we can find this way is a template type.
    if (auto ttd = dyn_cast<TemplateTypeDecl>(res.decls.front()))
        return TemplateType::Get(*M, ttd);

    Error(parsed->loc, "'{}' does not name a type", utils::join(parsed->names(), "::"));
    Note(res.decls.front()->location(), "Declared here");
    return Types::ErrorDependentTy;
}

auto Sema::TranslateProcType(
    ParsedProcType* parsed,
    SmallVectorImpl<TemplateTypeDecl*>* ttds
) -> Type {
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
        return Types::ErrorDependentTy;
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

        // Can’t really do a DenseMap of strings too well, so we just use a vector.
        SmallVector<TemplateDecl, 4> template_param_decls{parsed->param_types().size()};

        // First, do a prescan to collect template type defs.
        for (auto [i, a] : enumerate(parsed->param_types())) {
            if (auto ptt = dyn_cast<ParsedTemplateType>(a)) {
                auto& [name, loc, deduced_indices] = template_param_decls[i];
                name = ptt->name;
                if (deduced_indices.empty()) loc = ptt->loc;
                deduced_indices.push_back(u32(i));
            }
        }

        // Then build all the template type decls.
        for (auto& [name, loc, deduced_indices] : template_param_decls) {
            if (deduced_indices.empty()) continue;
            auto ttd = ttds->emplace_back(TemplateTypeDecl::Create(*M, name, deduced_indices, loc));
            AddDeclToScope(curr_scope(), ttd);
        }
    }

    // Then, compute the actual parameter types.
    SmallVector<Type, 10> params;
    for (auto a : parsed->param_types()) {
        // Template types encountered here introduce a template parameter
        // instead of referencing one, so we process them manually as the
        // usual translation machinery isn’t equipped to handle template
        // definitions.
        //
        // At this point, the only thing in the scope here should be the
        // template parameters, so lookup should never find anything else.
        if (auto ptt = dyn_cast<ParsedTemplateType>(a); ptt and ttds) {
            auto res = LookUpUnqualifiedName(curr_scope(), ptt->name, true);
            Assert(res.successful(), "Template parameter should have been declared earlier");
            Assert(res.decls.size() == 1 and isa<TemplateTypeDecl>(res.decls.front()));
            params.push_back(TemplateType::Get(*M, cast<TemplateTypeDecl>(res.decls.front())));
        }

        // Anything else is parsed as a regular type.
        //
        // If this is a template parameter that occurs in a context where
        // it is not allowed (e.g. in a function type that is not part of
        // a procedure definition), type translation will handle that case
        // and return an error.
        else {
            auto ty = TranslateType(a);
            if (ty == Types::DeducedTy) {
                Error(a->loc, "'{}' is not a valid type for a procedure argument", Types::DeducedTy->print(ctx.use_colours()));
                ty = Types::ErrorDependentTy;
            }
            params.push_back(ty);
        }
    }

    return ProcType::Get(*M, TranslateType(parsed->ret_type), params);
}

auto Sema::TranslateTemplateType(ParsedTemplateType* parsed) -> Type {
    Error(parsed->loc, "A template type declaration is only allowed in the parameter list of a procedure");
    return Types::ErrorDependentTy;
}

auto Sema::TranslateType(ParsedStmt* parsed) -> Type {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
        case K::BuiltinType: return TranslateBuiltinType(cast<ParsedBuiltinType>(parsed));
        case K::TemplateType: return TranslateTemplateType(cast<ParsedTemplateType>(parsed));
        case K::DeclRefExpr: return TranslateNamedType(cast<ParsedDeclRefExpr>(parsed));
        case K::ProcType: return TranslateProcType(cast<ParsedProcType>(parsed));
        default:
            Error(parsed->loc, "Expected type");
            return Types::ErrorDependentTy;
    }
}
