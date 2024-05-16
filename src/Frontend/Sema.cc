module;

#include <llvm/ADT/MapVector.h>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.utils;
using namespace srcc;

#define Try(expression) ({              \
    auto _res = expression;             \
    if (_res.invalid()) return nullptr; \
    _res.get();                         \
})

// ============================================================================
//  Helpers
// ============================================================================
auto Sema::CreateReference(Decl* d, Location loc) -> Expr* {
    switch (d->kind()) {
        default: Diag::ICE(ctx, d->location(), "Cannot build a reference to this declaration");
        case Stmt::Kind::ProcDecl: return new (*M) ProcRefExpr(cast<ProcDecl>(d), loc);
    }
}

auto Sema::GetScopeFromDecl(Decl* d) -> Ptr<Scope> {
    switch (d->kind()) {
        default: return nullptr;
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
            case Success: in_scope = GetScopeFromDecl(res.decls.front()).get(); break;

            // These are a hard error here.
            case Ambiguous:
            case NonScopeInPath:
                return res;

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

void Sema::ReportLookupFailure(const LookupResult& result, Location loc){
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
            Note(result.decls.front()->location(), "'{}' does not contain to a scope", result.name);
        } break;
    }
}

// ============================================================================
//  Building nodes.
// ============================================================================
auto Sema::BuildCallExpr(Expr* callee, ArrayRef<Expr*> args) -> Ptr<CallExpr> {
    if (callee->dependent() or rgs::any_of(args, [](Expr* e) { return e->dependent(); })) {
        return CallExpr::Create(
            *M,
            M->DependentTy,
            callee,
            args,
            callee->location()
        );
    }

    // Check that we can call this.
    // TODO: overload resolution.
    auto callee_proc = dyn_cast<ProcRefExpr>(callee);
    if (not isa<ProcRefExpr>(callee_proc))
        return Error(callee->location(), "Attempt to call non-procedure");

    auto params = callee_proc->decl->proc_type()->params();
    if (args.size() != params.size()) {
        return Error(
            callee->location(),
            "Procedure '{}' expects {} arguments, but {} were provided",
            callee_proc->decl->name,
            params.size(),
            args.size()
        );
    }

    for (auto [a, p] : vws::zip(args, params)) {
        if (a->type != p) {
            return Error(
                a->location(),
                "Argument of type '{}' does not match expected type '{}'",
                a->type.print(ctx.use_colours()),
                p.print(ctx.use_colours())
            );
        }
    }

    return CallExpr::Create(
        *M,
        callee_proc->return_type(),
        callee,
        args,
        callee_proc->location()
    );
}

auto Sema::BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr> {
    // Make sure all paths return a value.
    //
    // A function marked as returning void requires no checking and is allowed
    // to not return at all; invalid return expressions are checked when we first
    // encounter them.
    //
    // Accordingly, a function that actually never returns is also always fine,
    // since 'noreturn' is convertible to any type.
    auto ret = proc->return_type();
    if (ret != M->VoidTy and body->type != M->NoReturnTy and body->type != M->DependentTy) {
        if (ret == M->NoReturnTy) Error(
            proc->location(),
            "Procedure '{}' returns despite being marked as 'noreturn'",
            proc->name
        );

        else Error(
            proc->location(),
            "Procedure '{}' does not return a value on all paths",
            proc->name
        );

        // Procedure can’t be evaluated anyway if its body is invalid.
        return nullptr;
    }

    return body;
}

// ============================================================================
//  Translation Driver
// ============================================================================
auto Sema::Translate(ArrayRef<ParsedModule::Ptr> modules) -> Module::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};
    S.M = Module::Create(first->context(), first->name, first->is_module);
    S.parsed_modules = modules;
    S.Translate();
    return std::move(S.M);
}

void Sema::Translate() {
    // Initialise sema.
    all_scopes.push_back(std::make_unique<Scope>(nullptr));
    scope_stack.push_back(all_scopes.back().get());

    // Resolve imports.
    for (auto& p : parsed_modules)
        for (auto& i : p->imports)
            M->imports[i.linkage_name] = {nullptr, i.loc, i.import_name};

    // FIXME: C++ headers should all be imported at the same time; it really
    // doesn’t make sense to import them separately...
    for (auto& i : M->imports) {
        auto res = ImportCXXHeader(M->save(i.first()));
        if (not res) continue;
        i.second.imported_module = std::move(*res);
        imported_modules[i.second.import_name] = i.second.imported_module.get();
    }

    // Don’t attempt anything else if there was a problem.
    if (has_error) return;

    // Collect all statements and translate them.
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) has_error = TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, BlockExpr::NoExprIndex, Location{});
    M->initialiser_proc->body = M->file_scope_block;
}

auto Sema::TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedExpr*> parsed) -> bool {
    // Translate object declarations first since they may be out of order.
    //
    // Note that only the declaration part of definitions is translated here, e.g.
    // for a ProcDecl, we only translate, not the body; the latter is handled later
    // on.
    //
    // This translation only applies to *some* decls. It is allowed to do nothing,
    // but if it does fail, then we can’t process the rest of this scope.
    llvm::MapVector<ParsedExpr*, Decl*> translated_decls;
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
    if (not ok) return false;

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
        else ok = false;
    }

    return ok;
}

// ============================================================================
//  Translation of Individual Statements
// ============================================================================
auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed) -> Ptr<BlockExpr> {
    ScopeRAII scope{*this};
    SmallVector<Stmt*> stmts;
    if (not TranslateStmts(stmts, parsed->stmts())) return nullptr;

    // Determine the expression that is returned from this block, if
    // any. Ignore declarations. If the last non-declaration statement
    // is an expression, we return it; if not, we return nothing.
    u32 return_index = BlockExpr::NoExprIndex;
    for (auto [i, stmt] : stmts | vws::reverse | vws::enumerate) {
        if (isa<Decl>(stmt)) continue;
        if (isa<Expr>(stmt)) return_index = u32(i);
        break;
    }

    return BlockExpr::Create(
        *M,
        scope.get(),
        stmts,
        return_index,
        parsed->loc
    );
}

auto Sema::TranslateCallExpr(ParsedCallExpr* parsed) -> Ptr<CallExpr> {
    // Translate arguments.
    SmallVector<Expr*> args;
    bool errored = false;
    for (auto a : parsed->args()) {
        auto expr = TranslateExpr(a);
        if (expr.invalid()) errored = true;
        else args.push_back(expr.get());
    }

    // Translate callee.
    auto callee = Try(TranslateExpr(parsed->callee));
    if (errored) return nullptr;
    return BuildCallExpr(callee, args);
}

/// Translate a parsed name to a reference to the declaration it references.
auto Sema::TranslateDeclRefExpr(ParsedDeclRefExpr* parsed) -> Ptr<Expr> {
    auto res = LookUpName(curr_scope(), parsed->names(), parsed->loc);
    if (not res.successful()) return nullptr;
    return CreateReference(res.decls.front(), parsed->loc);
}

/// Perform initial processing of a decl so it can be used by the rest
/// of the code. This only handles order-independent decls.
auto Sema::TranslateDeclInitial(ParsedDecl* d) -> std::optional<Ptr<Decl>> {
    if (auto proc = dyn_cast<ParsedProcDecl>(d)) return TranslateProcType(proc);
    return nullptr;
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
auto Sema::TranslateExpr(ParsedExpr* parsed) -> Ptr<Expr> {
    auto stmt = TranslateStmt(parsed);
    if (stmt.invalid()) return nullptr;
    if (not isa<Expr>(stmt.get())) return Error(parsed->loc, "Expected expression");
    return cast<Expr>(stmt.get());
}

auto Sema::TranslateMemberExpr(ParsedMemberExpr* parsed)-> Ptr<Expr> {
    auto base = Try(TranslateExpr(parsed->base));
    if (isa<SliceType>(base->type)) {
        if (parsed->member == "data") return SliceDataExpr::Create(*M, base, parsed->loc);
        return Error(parsed->loc, "Slice has no member named '{}'", parsed->member);
    }

    return Error(parsed->loc, "Attempt to access member of type {}", base->type.print(true));
}

auto Sema::TranslateProc(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    // Translate the body if there is one.
    if (parsed->body) {
        auto res = TranslateProcBody(decl, parsed);
        if (res.invalid()) decl->set_errored();
        else decl->body = res.get();
    }

    return decl;
}

auto Sema::TranslateProcBody(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<Stmt> {
    ScopeRAII scope{*this, true};
    Assert(parsed->body);

    // TODO: Create local variables for parameters.
    decl->scope = scope.get();

    // Translate body.
    auto body = TranslateExpr(parsed->body);
    if (body.invalid()) return nullptr;
    return BuildProcBody(decl, body.get());
}

/// This is only called if we’re asked to translate a procedure by the expression
/// parser; actual procedure translation is handled elsewhere; only return a reference
/// here.
auto Sema::TranslateProcDecl(ParsedProcDecl* parsed) -> Ptr<Expr> {
    auto it = proc_decl_map.find(parsed);

    // This can happen if the procedure errored somehow.
    if (it == proc_decl_map.end()) return nullptr;

    // The procedure has already been created.
    return CreateReference(it->second, parsed->loc);
}

/// Perform initial type checking on a procedure, enough to enable calls
/// to it to be translated, but without touching its body, if there is one.
auto Sema::TranslateProcType(ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    // We don’t actually have parameters or return types atm...
    auto type = ProcType::Get(*M, M->VoidTy);
    auto proc = new (*M) ProcDecl(
        type,
        parsed->name,
        Linkage::Internal,
        Mangling::None,
        curr_proc(),
        nullptr,
        parsed->loc
    );

    // Add the procedure to the module and the current scope.
    proc_decl_map[parsed] = proc;
    if (not parsed->name.empty()) curr_scope()->add(proc);
    M->procs.push_back(proc);
    return proc;
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedExpr* parsed) -> Ptr<Stmt> {
    switch (parsed->kind()) {
        using K = ParsedExpr::Kind;
#define PARSE_TREE_LEAF_NODE(node) \
    case K::node: return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed));
#include "srcc/ParseTree.inc"
    }

    Unreachable("Invalid parsed statement kind: {}", +parsed->kind());
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed) -> Ptr<StrLitExpr> {
    return StrLitExpr::Create(*M, parsed->value, parsed->loc);
}
