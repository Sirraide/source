module;

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
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
        Error(
            d->location(),
            "Redeclaration of parameter '{}'",
            d->name
        );
    } else {
        ds.push_back(d);
    }
}

auto Sema::AdjustVariableType(Type ty) -> Type {
    // 'noreturn' is not a valid type for a variable.
    if (ty == Types::NoReturnTy) return Types::ErrorDependentTy;
    return ty;
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

            // This is a hard error here.
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

void Sema::LValueToSRValue(Expr*& expr) {
    expr = new (*M) CastExpr(expr->type, CastExpr::LValueToSRValue, expr, expr->location(), true);
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
//  Building nodes.
// ============================================================================
auto Sema::BuildBuiltinCallExpr(
    BuiltinCallExpr::Builtin builtin,
    ArrayRef<Expr*> args,
    Location call_loc
) -> Ptr<BuiltinCallExpr> {
    switch (builtin) {
        // __builtin_print takes a sequence of arguments and formats them.
        // FIXME: Actually implement that; it only prints one argument for now.
        case BuiltinCallExpr::Builtin::Print: {
            if (args.empty()) return Error(call_loc, "__builtin_print takes at least one argument");
            return BuiltinCallExpr::Create(*M, builtin, Types::VoidTy, args, call_loc);
        }
    }

    Unreachable("Invalid builtin type: {}", +builtin);
}

auto Sema::BuildCallExpr(Expr* callee, ArrayRef<Expr*> args) -> Ptr<CallExpr> {
    if (callee->dependent() or rgs::any_of(args, [](Expr* e) { return e->dependent(); })) {
        return CallExpr::Create(
            *M,
            Types::DependentTy,
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
            "Procedure '{}' expects {} argument{}, got {}",
            callee_proc->decl->name,
            params.size(),
            params.size() == 1 ? "" : "s",
            args.size()
        );
    }

    SmallVector<Expr*> actual_args{args};
    for (auto [a, p] : vws::zip(actual_args, params)) {
        if (a->type != p) {
            return Error(
                a->location(),
                "Argument of type '{}' does not match expected type '{}'",
                a->type.print(ctx.use_colours()),
                p.print(ctx.use_colours())
            );
        }

        if (a->type != Types::IntTy) return ICE(
            a->location(),
            "Only integer arguments are supported for now"
        );

        Assert(a->value_category == Expr::LValue or a->value_category == Expr::SRValue);
        if (a->value_category == Expr::LValue) LValueToSRValue(a);
    }

    return CallExpr::Create(
        *M,
        callee_proc->return_type(),
        callee,
        actual_args,
        callee_proc->location()
    );
}

auto Sema::BuildEvalExpr(Stmt* arg, Location loc) -> Ptr<Expr> {
    // Always create an EvalExpr to represent this in the AST.
    auto eval = new (*M) EvalExpr(arg, loc);
    if (arg->dependent()) return eval;

    // If the expression is not dependent, evaluate it now.
    auto value = srcc::eval::Evaluate(*M, arg);
    if (not value.has_value()) {
        eval->set_errored();
        return eval;
    }

    // And cache the value for later.
    return new (*M) ConstExpr(*M, std::move(*value), loc, eval);
}

auto Sema::BuildParamDecl(ProcScopeInfo& proc, Type ty, String name, Location loc) -> ParamDecl* {
    auto param = new (*M) ParamDecl(AdjustVariableType(ty), name, proc.proc, loc);
    DeclareLocal(param);
    return param;
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
    if (ret != Types::VoidTy and body->type != Types::NoReturnTy and body->type != Types::DependentTy) {
        if (ret == Types::NoReturnTy) Error(
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
    SmallVector<Stmt*> top_level_stmts;
    for (auto& p : parsed_modules) TranslateStmts(top_level_stmts, p->top_level);
    M->file_scope_block = BlockExpr::Create(*M, global_scope(), top_level_stmts, BlockExpr::NoExprIndex, Location{});
    M->initialiser_proc->body = M->file_scope_block;
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
auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed) -> Ptr<BlockExpr> {
    ScopeRAII scope{*this};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, parsed->stmts());

    // Determine the expression that is returned from this block, if
    // any. Ignore declarations. If the last non-declaration statement
    // is an expression, we return it; if not, we return nothing.
    u32 return_index = BlockExpr::NoExprIndex;
    for (auto [i, stmt] : llvm::enumerate(stmts | vws::reverse)) {
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
                      .Case("__builtin_print", B::Print)
                      .Default(std::nullopt);

        // We have a builtin!
        if (bk.has_value()) return BuildBuiltinCallExpr(*bk, args, parsed->loc);
    }

    // Translate callee.
    auto callee = TRY(TranslateExpr(parsed->callee));
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
    if (auto proc = dyn_cast<ParsedProcDecl>(d)) return TranslateProcDeclInitial(proc);
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
    auto small = val.trySExtValue();
    if (small.has_value()) return new (*M) IntLitExpr(Types::IntTy, parsed->storage, parsed->loc);

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

auto Sema::TranslateParamDecl(
    ParsedParamDecl* parsed,
    Type ty
) -> ParamDecl* {
    return BuildParamDecl(curr_proc(), ty, parsed->name, parsed->loc);
}

auto Sema::TranslateProc(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    // Translate the body if there is one.
    if (parsed->body) {
        EnterProcedure _{*this, decl};
        auto res = TranslateProcBody(decl, parsed);
        if (res.invalid()) decl->set_errored();
        else {
            decl->body = res.get();
            decl->finalise(curr_proc().locals);
        }
    }

    return decl;
}

auto Sema::TranslateProcBody(ProcDecl* decl, ParsedProcDecl* parsed) -> Ptr<Stmt> {
    ScopeRAII scope{*this, true};
    Assert(parsed->body);
    decl->scope = scope.get();

    // Translate parameters.
    auto ty = decl->proc_type();
    for (auto [i, pair] : vws::zip(ty->params(), parsed->params()) | vws::enumerate) {
        auto [param_ty, parsed_decl] = pair;
        TranslateParamDecl(parsed_decl, param_ty);
    }

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
auto Sema::TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<ProcDecl> {
    auto type = TranslateProcType(parsed->type);
    auto proc = ProcDecl::Create(
        *M,
        type,
        parsed->name,
        Linkage::Internal,
        Mangling::None,
        proc_stack.empty() ? nullptr : proc_stack.back().proc,
        nullptr,
        parsed->loc
    );

    // Add the procedure to the module and the current scope.
    proc_decl_map[parsed] = proc;
    AddDeclToScope(curr_scope(), proc);
    M->procs.push_back(proc);
    return proc;
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedStmt* parsed) -> Ptr<Stmt> {
    switch (parsed->kind()) { // clang-format off
        using K = ParsedStmt::Kind;
#       define PARSE_TREE_LEAF_TYPE(node) case K::node: Todo("Translate type to expr");
#       define NODE_PARAM_DECL(node) case K::node: Unreachable("Param decls are emitted elsewhere");
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

// ============================================================================
//  Translation of Types
// ============================================================================
auto Sema::TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type {
    switch (parsed->builtin_kind) {
        using K = ParsedBuiltinType::Kind;
        case K::Int: return Types::IntTy;
    }

    Unreachable("Invalid builtin type");
}

auto Sema::TranslateProcType(ParsedProcType* parsed) -> Type {
    SmallVector<Type, 10> params;
    for (auto a : parsed->param_types()) params.push_back(TranslateType(a));
    return ProcType::Get(*M, Types::VoidTy, params);
}

auto Sema::TranslateType(ParsedType* parsed) -> Type { // clang-format off
    switch (parsed->kind()) {
        // Dispatch to individual type translators and reject everything that isn’t a type.
        using K = ParsedStmt::Kind;
#       define PARSE_TREE_LEAF_NODE(node) case K::node: break;
#       define PARSE_TREE_LEAF_TYPE(node) case K::node: \
            return SRCC_CAT(Translate, node)(cast<SRCC_CAT(Parsed, node)>(parsed));
#       include "srcc/ParseTree.inc"



    }

    Unreachable("Not a valid type kind: {}", +parsed->kind());
} // clang-format on
