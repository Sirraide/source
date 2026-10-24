#include <srcc/AST/AST.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/ASTImporterSharedState.h>

#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/MemoryBuffer.h>

#include <base/Assert.hh>
#include <base/StringUtils.hh>
#include <base/Utils.hh>

#include <ranges>

using namespace srcc;
using namespace srcc::sema;

#define TRY(...) ({                      \
    auto _res = (__VA_ARGS__);           \
    if (not _res) return utils::Falsy(); \
    *_res;                               \
})

// ============================================================================
//  RAII Types
// ============================================================================
Sema::DiagnosticsTrap::DiagnosticsTrap(Sema& S)
    : S{S}, prev_engine(&S.diags()),
      first_new_diag_index(u32(S.trapping_engine->diags.size())) {
    S.context().set_diags(S.trapping_engine);
}

Sema::DiagnosticsTrap::~DiagnosticsTrap() {
    Assert(
        S.trapping_engine->diags.size() == first_new_diag_index,
        "Forgot to use trapped diagnostics?"
    );

    S.context().set_diags(prev_engine);
}

auto Sema::DiagnosticsTrap::get_trapped_diagnostics() -> DiagsVector {
    auto& diags = S.trapping_engine->diags;
    DebugAssert(first_new_diag_index <= u32(diags.size()));

    // We didn’t trap anything.
    if (u32(diags.size()) == first_new_diag_index) return {};

    // Get the diagnostics that were trapped since the last time the trapping
    // engine was installed; note that is is possible to have multiple traps
    // on the stack recursively, so we can't just take all the diagnostics we
    // trapped unless the engine was empty before.
    if (first_new_diag_index == 0) return std::exchange(diags, DiagsVector());

    // Extract diagnostics starting at the index.
    DiagsVector new_vector{
        std::make_move_iterator(diags.begin() + first_new_diag_index),
        std::make_move_iterator(diags.end()),
    };

    diags.erase(diags.begin() + first_new_diag_index, diags.end());
    return new_vector;
}

Sema::EnterProcedure::EnterProcedure(Sema& S, ProcDecl* proc)
    : S{S}, scope{S, not S.scope_stack.empty() and S.curr_scope() == proc->scope ? nullptr : proc->scope} {
    Assert(proc->scope, "Entering procedure without scope?");
    auto& info = S.request_data.procedure_contexts.emplace_back(
        std::make_unique<ProcScopeInfo>(
            proc,
            S.proc_stack.empty() ? nullptr : S.proc_stack.back()
        )
    );

    append_range(info->locals, proc->params());
    S.proc_stack.push_back(info.get());
}

Sema::EnterProcedure::~EnterProcedure() {
    S.proc_stack.pop_back();
}

Sema::EnterLoop::EnterLoop(Sema& S) : S{S} {
    ++S.curr_proc().loop_depth;
    save_current_loop_has_break = std::exchange(S.curr_proc().current_loop_has_break, false);
}

Sema::EnterLoop::~EnterLoop() {
    --S.curr_proc().loop_depth;
    S.curr_proc().current_loop_has_break = save_current_loop_has_break;
}

auto Sema::EnterLoop::token() -> LoopToken {
    return S.curr_proc().loop_depth;
}

Sema::EnterScope::EnterScope(Sema& S, bool should_enter)
    : EnterScope(S, should_enter ? S.tu->create_scope<BlockScope>(S.curr_scope()) : nullptr) {}

Sema::EnterScope::EnterScope(Sema& S, Tag<ProcScope>, Opt<Type> associated_type)
    : EnterScope(S, S.tu->create_scope<ProcScope>(S.curr_scope(), associated_type)) {}

Sema::EnterScope::EnterScope(Sema& S, Tag<RecordScope>)
    : EnterScope(S, S.tu->create_scope<RecordScope>(S.curr_scope())) {}

Sema::EnterScope::EnterScope(Sema& S, Scope* scope) : S{S}, scope{scope} {
    if (not scope) return;

    Assert(
        S.scope_stack.empty() or S.curr_scope() != scope,
        "Entering the same scope twice in a row; this is probably a bug"
    );

    S.scope_stack.push_back(scope);
}

Sema::EnterScope::~EnterScope() {
    if (not scope) return;
    S.scope_stack.pop_back();
}

void Sema::ActivateWithParameters(ArrayRef<ParamSpec> specs) {
    for (auto [param, spec] : zip_equal(curr_proc().proc->params(), specs)) {
        if (spec.with_loc.is_valid() or spec.is_this) AddEntryToWithStack(
            curr_scope(),
            Save(RequestReference(param, param->location()).get()),
            spec.with_loc,
            spec.is_this
        );
    }
}

void Sema::AddDeclToScope(Scope* scope, Decl* d) {
    // Do not add anonymous decls to the scope.
    if (d->name.empty()) return;

    // And make sure to check for duplicates. Duplicate declarations
    // are usually allowed, but we forbid redeclaring e.g. (template)
    // parameters.
    auto ds = scope->decls_with_name(d->name);
    if (not ds.empty()) {
        if (isa<FieldDecl, ParamDecl, TemplateTypeParamDecl>(d)) {
            Error(d->location(), "Redeclaration of '{}'", d->name);
            NotePrevDecl(ds.front());
            return;
        }

        if (auto new_proc = dyn_cast<ProcDecl>(d); new_proc and new_proc->valid()) {
            // The existing decl is also a procedure; validate that the new one isn’t a
            // redeclaration of any existing procedure.
            for (auto old : ds) {
                // Skip invalid declarations here.
                if (!old->valid()) continue;

                // As well as templates.
                if (isa<ProcTemplateDecl>(old)) continue;

                // A procedure conflicts with a declaration of something else; this is a bit
                // of an issue since it would mess w/ overload resolution if we suddenly found
                // other nonsense when looking for one.
                if (not isa<ProcDecl>(old)) {
                    d->set_invalid();
                    Error(d->location(), "Declaration of procedure '%2({}%)' shadows a different entity", d->name);
                    NotePrevDecl(old);
                    return;
                }

                // Check if this overload is well-formed.
                auto old_proc = cast<ProcDecl>(old);
                auto CheckOverload = [&] -> bool {
                    // If the parameter count doesn’t match, this is fine.
                    auto old_ty = old_proc->proc_type();
                    auto new_ty = new_proc->proc_type();
                    auto num_old_params = old_ty->non_variadic_params();
                    auto num_new_params = new_ty->non_variadic_params();
                    if (num_old_params != num_new_params) return true;

                    // If a parameter type doesn’t match, this is fine.
                    auto old_params = old_ty->params().take_front(num_old_params);
                    auto new_params = new_ty->params().take_front(num_new_params);
                    for (auto [o, n] : zip_equal(old_params, new_params))
                        if (o.type != n.type)
                            return true;

                    // They do; check if we’re overloading on intent.
                    for (auto [i, pair] : enumerate(zip_equal(old_params, new_params))) {
                        auto [o, n] = pair;
                        if (o.intent != n.intent) return Error(
                            d->location(),
                            "Cannot overload procedure '%2({}%)' on parameter intent",
                            d->name
                        );
                    }

                    // A varargs procedure cannot overload a variadic procedure, and vice-versa.
                    if (
                        (old_ty->has_c_varargs() or old_ty->is_variadic()) and
                        (new_ty->has_c_varargs() or new_ty->is_variadic()) and
                        old_ty->has_c_varargs() != new_ty->has_c_varargs()
                    ) return Error(
                        d->location(),
                        "{} procedure '%2({}%)' redefined as {}",
                        old_ty->has_c_varargs() ? "'%1(varargs%)'" : "Variadic",
                        d->name,
                        new_ty->has_c_varargs() ? "'%1(varargs%)'" : "variadic"
                    );

                    // If both procedures are variadic, the variadic parameter can
                    // have a different type. This is valid; it’s up to overload
                    // resolution to figure out what to do w/ this.
                    if (
                        old_ty->is_variadic() and
                        old_ty->params().back().type != new_ty->params().back().type
                    ) return true;

                    // Procedures may not differ in return type alone, unless this is 'as'.
                    if (old_ty->ret() != new_ty->ret()) {
                        if (d->name == Tk::As) return true;
                        return Error(
                            d->location(),
                            "Cannot overload procedure '%2({}%)' on return type",
                            d->name
                        );
                    }

                    // The procedure are the exact same.
                    Assert(Type(old_ty) == new_ty);
                    return Error(d->location(), "Procedure '%2({}%)' redefined", d->name);
                };

                if (not CheckOverload()) {
                    new_proc->set_invalid();
                    NotePrevDecl(old);
                    return;
                }
            }
        }

        if (isa<WithFieldRefDecl, WithBuiltinFieldRefDecl>(d)) {
            Warn(d->location(), "Implicit field decl '{}' shadows existing declaration", d->name);
            NotePrevDecl(ds.front());
        }
    }

    scope->add(d);
}

void Sema::AddEntryToWithStack(Scope* scope, SaveExpr* object, SLoc with, bool is_this) {
    Assert(object);

    // The type must be complete.
    Type ty = object->type->strip_pointers_and_optionals();
    if (not RequestCompleteType(ty, with)) return;

    // Declare the object’s members in the current scope, if there are any.
    if (auto s = dyn_cast<RecordType>(ty)) {
        Assert(object->is_lvalue());
        for (auto f : s->scope()->fields()) {
            auto d = new (*tu) WithFieldRefDecl{object, f, with};
            AddDeclToScope(scope, d);
        }
    }

    // Also do the same for types with builtin members.
    else if (auto members = BuiltinMemberAccessExpr::GetAllBuiltinMembersOf(ty); not members.empty()) {
        for (const auto& [_, member] : members) {
            auto d = new (*tu) WithBuiltinFieldRefDecl(object, member, with);
            AddDeclToScope(scope, d);
        }
    }

    // Warn on non-struct types with no members if we’re not in a template. Also
    // don't warn if this is 'this' because the with entry for it is added implicitly.
    else if (not curr_proc().proc->is_instantiation() and not is_this) {
        Warn(with, "'%1(with%)' has no effect as type '{}' has no members", ty);
    }
}

auto Sema::BuildImplicitProcedure(
    ProcType* ty,
    DeclName name,
    ArrayRef<ParamSpec> params,
    Linkage linkage,
    Mangling mangling,
    SLoc loc,
    std::move_only_function<Ptr<Expr>(ProcDecl*)> BuildBody,
    bool defer_body
) -> ProcDecl* {
    auto scope = tu->create_scope<ProcScope>(curr_scope(), std::nullopt);
    auto proc = ProcDecl::Create(
        *tu,
        nullptr,
        ty,
        name,
        linkage,
        mangling,
        curr_proc().proc,
        InheritedProcedureProperties(),
        loc
    );

    // Set the parent, else captures will break horribly.
    proc->parent = curr_proc().proc;
    proc->scope = scope;

    // Build parameters.
    auto context = GetRequestContext();
    BuildParamDecls(proc, params);
    EnterProcedure procedure{*this, proc};
    ActivateWithParameters(params);

    // Defer building the body if requested, else build it now.
    proc->will_have_body = true;
    request_data.procedure_sources[proc] = {std::move(BuildBody), context};
    if (defer_body) request_data.pending_procedure_bodies[context.procedure->proc].push_back(proc);
    else RequestProcedureBody(proc, loc);
    return proc;
};

bool Sema::CheckFieldType(Type type, SLoc loc) {
    if (not CheckVariableType(type, loc)) return false;
    return true;
}

bool Sema::CheckVariableType(Type ty, SLoc loc) {
    // Any places that want to do type deduction need to take
    // care of it *before* this is called.
    if (not ProhibitDeducedTypes(ty, loc)) return false;
    return RequestCompleteType(ty, loc);
}

bool Sema::ProhibitDeducedTypes(Type ty, SLoc loc) {
    if (ty == Type::DeducedTy) return Error(loc, "Type deduction is not allowed here");
    if (ty == Type::NoReturnTy) return Error(loc, "'{}' is not allowed here", Type::NoReturnTy);
    if (ty == Type::UnresolvedOverloadSetTy) return Error(loc, "Unresolved overload set is not allowed here");
    return true;
}

auto Sema::ComputeCommonTypeAndValueCategory(MutableArrayRef<Expr*> exprs) -> TypeAndValueCategory {
    Assert(not exprs.empty());
    auto t = exprs.front()->type;
    auto vc = exprs.front()->value_category;
    for (auto [i, e] : enumerate(exprs.drop_front())) {
        auto MergeVC = [&] {
            if (vc == Expr::RValue or e->value_category == Expr::RValue) {
                vc = Expr::RValue;
            } else if (vc == Expr::ILValue or e->value_category == Expr::ILValue) {
                vc = Expr::ILValue;
            } else {
                vc = Expr::MLValue;
            }
        };

        // If either type is 'noreturn', the common type is the type of the other
        // branch (unless both are noreturn, in which case the type is just 'noreturn').
        if (e->type == Type::NoReturnTy) continue;
        if (t == Type::NoReturnTy) {
            t = e->type;
            vc = e->value_category;
            continue;
        }

        // If both values have the same type, the common type is that type. The value
        // category needs to be merged as well.
        if (t == e->type) {
            MergeVC();
            continue;
        }


        // If either type is an optional, and the other is that optional’s
        // element type, unwrap the optional. The result is an lvalue of the
        // element type if both are lvalues, and an rvalue of the element type
        // otherwise.
        //
        // If 'vc' is LValue, we want to set it to LValue, which is a no-op;
        // the same applies if 'vc' is RValue, so just don’t do anything here.
        //
        // First case: the new element is an optional; unwrap it.
        if (auto o = dyn_cast<OptionalType>(e->type); o and o->elem() == t) {
            e = UnwrapOptional(e, e->location());
            continue;
        }

        // Second case: all elements before this one were optionals; unwrap them.
        if (auto o = dyn_cast<OptionalType>(t); o and o->elem() == e->type) {
            t = e->type;

            // If the new element is an rvalue, force all other elements to be
            // rvalues as well.
            if (e->is_rvalue()) vc = Expr::RValue;

            // Offset the index by '1' since we dropped the first element above.
            for (auto& expr: exprs.take_front(i + 1)) expr = UnwrapOptional(expr, expr->location());
            continue;
        }

        // If both types are pointers with the same element type that differ only
        // in mutability, the result is an immutable pointer type with merged value
        // category.
        if (
            auto t_ptr = dyn_cast<PtrType>(t), e_ptr = dyn_cast<PtrType>(e->type);
            t_ptr and e_ptr and t_ptr->elem() == e_ptr->elem()
        ) {
            Assert(t_ptr->is_immutable() != e_ptr->is_immutable());
            t = t_ptr->is_immutable() ? t_ptr : e_ptr;
            MergeVC();
            continue;
        }

        // Similarly for slices.
        if (
            auto t_slice = dyn_cast<SliceType>(t), e_slice = dyn_cast<SliceType>(e->type);
            t_slice and e_slice and t_slice->elem() == e_slice->elem()
        ) {
            Assert(t_slice->is_immutable() != e_slice->is_immutable());
            t = t_slice->is_immutable() ? t_slice : e_slice;
            MergeVC();
            continue;
        }

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

void Sema::DefineSpecialProcedures(
    RecordType* r,
    RecordLayout::Props props,
    Ptr<ParsedStmt> parsed_deleter,
    Ptr<ParsedStmt> parsed_copy,
    SLoc loc,
    bool generate_initialisers
) {
    Assert(r->is_canonical());
    if (generate_initialisers) {
        auto Register = [&](ProcDecl* proc) {
            proc->props.associated_type = r;
            proc->props.mangling_number = next_proc_mangling_number++;
            if (auto structure = dyn_cast<StructType>(r)) proc->parent = structure->parent;
            else proc->parent = nullptr; // Tuples have no lexical parent.
            AddDeclToScope(r->scope(), proc);
        };

        // Build the body of the default initialiser.
        auto BuildDefault = [this, r, loc](ProcDecl*) -> Ptr<Expr> {
            SmallVector<Expr*> fields;
            for (auto field : r->scope()->fields()) {
                auto init = BuildInitialiser(field->type, {}, loc);
                if (not init) return {};
                fields.push_back(init.get());
            }
            return TupleExpr::Create(*tu, r, fields, {}, loc);
        };

        // Build the body of the aggregate initialiser.
        auto BuildAggregate = [this, r, loc](ProcDecl* proc) -> Ptr<Expr> {
            SmallVector<Expr*> fields;
            for (auto param : proc->params()) {
                auto ref = RequestReference(param, loc);
                if (not ref) return {};
                auto init = BuildInitialiser(param->type, ref.get(), loc);
                if (not init) return {};
                fields.push_back(init.get());
            }
            return TupleExpr::Create(*tu, r, fields, {}, loc);
        };

        // Build the default initialiser.
        Register(BuildImplicitProcedure(
            ProcType::Get(*tu, {r, Expr::RValue}),
            Tk::Init,
            {},
            Linkage::Exported,
            Mangling::Source,
            loc,
            std::move(BuildDefault),
            /*defer_body=*/true
        ));

        // Build the aggregate initialiser.
        SmallVector<ParamTypeData> params;
        SmallVector<ParamSpec> specs;
        for (auto field : r->scope()->fields()) {
            params.emplace_back(Intent::Move, field->type);
            specs.push_back({.name = {field->name, field->location()}});
        }

        Register(BuildImplicitProcedure(
            ProcType::Get(*tu, {r, Expr::RValue}, params),
            Tk::Init,
            specs,
            Linkage::Exported,
            Mangling::Source,
            loc,
            std::move(BuildAggregate),
            /*defer_body=*/true
        ));
    }

    // If we don’t need a deleter/copy, we’re done.
    if (
        parsed_deleter.invalid() and
        parsed_copy.invalid() and
        not props.must_define_delete and
        not props.must_define_copy
    ) return;

    // Build the body of 'delete'.
    auto delete_loc = parsed_deleter ? parsed_deleter.get()->loc : loc;
    auto BuildDeleteBody = [=, this](ProcDecl* proc) -> Ptr<Expr> {
        EnterScope scope{*this};
        SmallVector<Stmt*> stmts;

        // Insert the user-defined deleter if there is one.
        auto this_ptr = RequestReference(curr_proc().locals[0], delete_loc).get();
        if (auto body = parsed_deleter.get_or_null()) {
            AddEntryToWithStack(curr_scope(), Save(this_ptr), delete_loc, true);
            auto user_del = TranslateStmt(body);
            if (user_del) {
                stmts.push_back(user_del.get());
                if (user_del.get()->type_or_void() == Type::NoReturnTy) return nullptr;
            }
        }

        // Call the deleter of each field in reverse order.
        for (auto f : reverse(llvm::to_vector(r->scope()->fields()))) {
            if (not f->type->requires_deletion()) continue;
            auto ref = BuildMemberAccessExpr(this_ptr, f, delete_loc);
            if (not ref) continue;
            auto del = MaybeBuildDeleteExpr(ref.get(), true, delete_loc);
            if (del) stmts.push_back(del.get());
        }

        return BuildBlockExpr(scope.get(), stmts, delete_loc);
    };

    // Build the body of 'copy'.
    auto copy_loc = parsed_copy ? parsed_copy.get()->loc : loc;
    auto BuildCopyBody = [this, r, parsed_copy](ProcDecl* proc) -> Ptr<Expr> {
        // If the user provided a body, just use it.
        if (auto body = parsed_copy.get()) return TranslateExpr(body, r);

        // If not, build the equivalent of 'copy = (*this);'
        Todo("Spread this");
    };

    // FIXME: Once structs have linkage, reuse the struct's linkage.
    Linkage linkage = Linkage::Exported;
    if (props.must_define_delete or parsed_deleter.present()) {
        auto proc = BuildImplicitProcedure(
            ProcType::Get(*tu, {Type::VoidTy, Expr::RValue}, {{Intent::Inout, r}}),
            Tk::Delete,
            {ParamSpec{.name = {String("this"), delete_loc}, .is_this = true}},
            linkage,
            Mangling::Source,
            delete_loc,
            BuildDeleteBody,
            /*defer_body=*/true
        );

        r->set_delete(proc);
    }

    if (props.must_define_copy or parsed_copy.present()) {
        auto proc = BuildImplicitProcedure(
            ProcType::Get(*tu, {r, Expr::RValue}, {{Intent::In, r}}),
            Tk::Copy,
            {ParamSpec{.name = {String("this"), copy_loc}, .is_this = true}},
            linkage,
            Mangling::Source,
            copy_loc,
            BuildCopyBody,
            /*defer_body=*/true
        );

        r->set_copy(proc);
    }
}

void Sema::DeclareLocal(LocalDecl* d) {
    Assert(d->parent == curr_proc().proc, "Must EnterProcedure before adding a local variable");
    curr_proc().locals.push_back(d);
    AddDeclToScope(curr_scope(), d);
}

void Sema::DiagnoseZeroSizedTypeInNativeProc(Type ty, SLoc use, bool is_return) {
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

auto Sema::GetParamSpecs(ArrayRef<ParsedVarDecl*> params) -> SmallVector<ParamSpec> {
    return llvm::to_vector(params | vws::transform([](auto param) {
        return ParamSpec{
            .name = {param->name, param->loc},
            .with_loc = param->with_loc,
            .immutable = isa<ParsedValueType>(param->type),
            .is_this = param->is_this_param,
        };
    }));
}

auto Sema::GetScopeFromDecl(Decl* d) -> Ptr<Scope> {
    return d->visit(utils::Overloaded{
        [](auto*) -> Scope* { return nullptr; },
        [](ProcDecl* p) { return p->scope; },
        [](TypeDecl* td) {
            auto e = dyn_cast<EnumType>(td->type);
            return e ? e->scope() : nullptr;
        },
    });
}

bool Sema::InjectTree(Expr* injected, Opt<Type> desired_type, InjectionContext context) {
    if (not MakeRValue(Type::TreeTy, injected, [&]{
        Error(
            injected->location(),
            "Cannot inject value of type '{}'",
            injected->type
        );
    })) return {};

    // Evaluate the injection.
    auto res = RequestEvaluation(injected);
    if (not res) return {};
    auto top_level_tree = res->cast<TreeValue*>();

    // Collect the tokens and parse them.
    TokenStream tokens{tu->allocator()};
    auto CollectTokens = [&](this auto& self, TreeValue* tree) -> void {
        u32 unquote_idx = 0;
        for (const auto& t : tree->pattern()->quoted->tokens()) {
            if (t.is(Tk::Unquote)) self(tree->unquotes()[unquote_idx++]);
            else tokens.push(t);
        }
    };

    CollectTokens(top_level_tree);

    // Make sure to terminate the token stream as the parser gets very
    // angry if the token stream doesn’t end with an end-of-file token.
    tokens.finish(top_level_tree->pattern()->location());
    auto fragment = AddParsedModule(Parser::ParseFragment(ctx, tokens));
    if (not fragment) return false;

    // Keep diagnostic token locations, but compare visibility at the injection
    // site when a reference crosses the boundary of this expansion.
    auto parent = request_data.expansion;
    auto& expansion = request_data.expansions.emplace_back(
        parent,
        injected->location(),
        parent ? parent->depth + 1 : 1
    );

    tempset request_data.expansion = &expansion;

    // Translate the parse tree.
    SmallVector<Stmt*> stmts;
    if (not TranslateStmts(stmts, fragment->top_level, desired_type)) return false;
    if (auto multiple = dyn_cast<SmallVectorImpl<Stmt*>*>(context)) {
        multiple->append(stmts);
        return true;
    }

    auto single = cast<Stmt**>(context);
    if (stmts.size() != 1) return Error(
        top_level_tree->pattern()->location(),
        "An '#inject' in this context must result in exactly 1 statement; got {}",
        stmts.size()
    );

    *single = stmts.front();
    return true;
}

bool Sema::IntegerLiteralFitsInType(const APInt& i, Type ty, bool negated) {
    Assert(ty->is_integer(), "Not an integer: '{}'", ty);
    auto bits = ty->bit_width(*tu);
    if (negated) return Size::Bits(i.getSignificantBits()) <= bits;
    else return Size::Bits(i.getActiveBits()) <= bits;
}

void Sema::ReportDiag(Diagnostic&& d) {
    if (d.is_error()) curr_proc().proc->contains_error = true;
    diags().report_diag(std::move(d));
}

bool Sema::IsBuiltinVarType(ParsedStmt* stmt) {
    auto b = dyn_cast<ParsedBuiltinType>(stmt);
    return b and b->ty == Type::DeducedTy;
}

bool Sema::IsZeroSizedOrIncomplete(Type ty) {
    if (auto s = dyn_cast<StructType>(ty); s and not s->is_complete()) return true;
    return ty->memory_size(*tu) == Size();
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
    SLoc loc
) -> Expr* {
    if (isa_and_present<BoolLitExpr, StrLitExpr, IntLitExpr, NilExpr, TypeExpr>(evaluated_stmt))
        return cast<Expr>(evaluated_stmt);
    return new (*tu) ConstExpr(*tu, std::move(val), loc, evaluated_stmt);
}

auto Sema::MakeLocal(
    Type ty,
    ValueCategory vc,
    String name,
    SLoc loc
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

bool Sema::MakeRValue(Type ty, Expr*& e, llvm::function_ref<void()> EmitDiag) {
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

void Sema::NotePrevDecl(Decl *decl) {
    using Ret = std::pair<StringRef, int>;
    auto [what, colour] = decl->visit(utils::Overloaded{
        [&](FieldDecl*) -> Ret { return {"field", 5}; },
        [&](ModuleDecl*) -> Ret { return {"module", 4}; },
        [&](ProcDecl*) -> Ret { return {"procedure", 2}; },
        [&](ProcTemplateDecl*) -> Ret { return {"procedure", 2}; },
        [&](TypeDecl*) -> Ret { return {"type", 5}; },
        [&](GlobalDecl*) -> Ret { return {"global variable", 8}; },
        [&](CXXMacroExpansionDecl*) -> Ret { return {"C++ macro", 2}; },
        [&](EnumeratorDecl*) -> Ret { return {"enumerator", 4}; },
        [&](LocalDecl*) -> Ret { return {"local variable", 8}; },
        [&](ParamDecl*) -> Ret { return {"parameter", 4}; },
        [&](WithFieldRefDecl*) -> Ret { return {"'%1(with%)' binding", 3}; },
        [&](WithBuiltinFieldRefDecl*) -> Ret { return {"'%1(with%)' binding", 3}; },
    });

    Note(
        decl->location(),
        "Conflicting declaration of {} '%{}({}%)' is here",
        what,
        colour,
        decl->name
    );
}

void Sema::ReportLookupFailure(LookupResult&& res) {
    switch (res.result) {
        using enum LookupResult::Reason;
        case Success: Unreachable("Diagnosing a successful lookup?");
        case FailedToImport: Error(res.name.loc, "Could not import symbol '{}' from Clang", res.name); break;
        case NotFound: Error(res.name.loc, "Unknown symbol '{}'", res.name); break;
        case Ambiguous: {
            Error(res.name.loc, "Ambiguous symbol '{}'", res.name);
            for (auto d : res.decls) Note(d->location(), "Candidate here");
        } break;
        case NonScopeInPath: {
            Error(res.name.loc, "Invalid left-hand side for '::'");
            if (not res.decls.empty()) Note(
                res.decls.front()->location(),
                "'{}' does not contain a scope",
                res.name
            );
        } break;
    }

    if (res.note) {
        ReportDiag(std::move(*res.note));
        res.note.reset();
    }
}

bool Sema::RequiresManglingNumber(const ParsedProcAttrs& attrs) {
    // Obviously don’t allocate a mangling number if the thing isn’t even
    // mangled to begin with.
    if (attrs.nomangle) return false;

    // Do not allocate a mangling number to builtin operators as there should
    // never be any conflicting declarations of those since they’re under our
    // control, and this avoids invalidating the IR of every CG test after updating
    // the preamble.
    if (attrs.builtin_operator) return false;

    // There is also an attribute to disable the mangling number.
    return not attrs.no_mangling_number;
}

auto Sema::Save(Expr* e) -> SaveExpr* {
    // If we have a SaveExpr already, just reuse it.
    auto se = dyn_cast<SaveExpr>(e);
    if (se) return se;

    // A memory rvalue isn’t really a value we can refer to, so
    // materialise it first.
    if (e->is_rvalue() and e->type->eval_mode() == EvalMode::Memory)
        e = MaterialiseTemporary(e);

    return new (*tu) SaveExpr(e, e->location());
}

auto Sema::UnwrapOptional(Expr* expr, SLoc loc) -> Expr* {
    return new (*tu) CastExpr(
        cast<OptionalType>(expr->type)->elem(),
        CastExpr::OptionalUnwrap,
        MaterialiseTemporary(expr),
        loc,
        true,
        Expr::MLValue
    );
}

auto Sema::UnwrapPointersAndOptionals(Expr* base, Opt<Type> stop_at) -> Ptr<Expr> {
    while (base->type != stop_at) {
        if (isa<PtrType>(base->type)) {
            base = TRY(BuildUnaryExpr(Tk::Caret, base, false, base->location()));
        } else if (isa<OptionalType>(base->type)) {
            base = UnwrapOptional(base, base->location());
        } else {
            break;
        }
    }

    return base;
}

// ============================================================================
//  Pattern matching.
// ============================================================================
auto Sema::BoolMatchContext::add_constant_pattern(
    const eval::RValue& pattern,
    SLoc loc
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

void Sema::BoolMatchContext::note_missing(SLoc match_loc) {
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
    SLoc loc
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

void Sema::IntMatchContext::note_missing(SLoc loc) {
    SmallString<128> msg;
    auto Fmt = [&](const APInt& start, const APInt& end) {
        auto FormatVal = [&](const APInt& i) {
            if (i == min) return Format("{}%1(.%)%5(min%)", ty);
            if (i == max) return Format("{}%1(.%)%5(max%)", ty);
            return Format("%5({}%)", i);
        };

        if (start == end) {
            Format(msg, "\n    {},", FormatVal(start));
        } else {
            Format(
                msg,
                "\n    {}%1(..=%){},",
                FormatVal(start),
                FormatVal(end)
            );
        }
    };

#if 0
    // For debugging the merge algorithm.
    for (auto& r : ranges) {
        Format(msg, "\n    {}, {}", r.start, r.end);
    }

    S.Note(loc, "DEBUG. VALUE RANGES ARE\n%r({}%)", msg);
    return;
#endif

    // It’s easiest to handle this case separately.
    auto sz = ranges.size();
    if (sz == 0) {
        Fmt(min, max);
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
            Fmt(first_unsatisfied, r.start.ssub_sat(one));
            first_unsatisfied = r.end.sadd_sat(one);
        }

        // Append any remaining values after the last range.
        //
        // Note that there is an edge case here: the last range may
        // end immediately before '<type>.max'.
        if (
            first_unsatisfied != max or
            (sz != 0 and ranges.back().end == max - 1)
        ) Fmt(first_unsatisfied, max);
    }


    if (msg.ends_with(",")) msg.pop_back();
    S.Note(loc, "Possible value ranges not handled:\n%r({}%)", msg);
}

Sema::OptionalMatchContext::OptionalMatchContext(
    Sema& s,
    OptionalType* optional,
    std::unique_ptr<MatchContext> inner
) : MatchContext{s}, optional{optional}, inner{std::move(inner)} {}

auto Sema::OptionalMatchContext::add_constant_pattern(
    const eval::RValue& pattern,
    SLoc loc
) -> AddResult {
    if (pattern.type() == Type::NilTy) {
        if (nil_loc.is_valid()) return Subsumed(nil_loc);
        nil_loc = loc;
        return Exhaustive(inner_exhaustive);
    }

    auto res = inner->add_constant_pattern(pattern, loc);
    if (res.kind != AddResult::Kind::Exhaustive) return res;
    inner_exhaustive = true;
    return Exhaustive(nil_loc.is_valid());
}

auto Sema::OptionalMatchContext::build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr> {
    if (pattern_expr->type == Type::NilTy) return new (*S.tu) OptionalNilTestExpr(
        control_expr,
        pattern_expr,
        true,
        pattern_expr->location()
    );

    control_expr = S.UnwrapOptional(control_expr, pattern_expr->location());
    return inner->build_comparison(control_expr, pattern_expr);
}

void Sema::OptionalMatchContext::note_missing(SLoc match_loc) {
    if (not nil_loc.is_valid()) S.Note(match_loc, "'nil' value not handled");
    if (not inner_exhaustive) inner->note_missing(match_loc);
}

auto Sema::OptionalMatchContext::preprocess(Expr* pattern) -> Ptr<Expr> {
    return inner->preprocess(pattern);
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
//   - named wildcard pattern (`match x { var y: }`)
bool Sema::CheckMatchExhaustive(
    SLoc loc,
    Expr* control,
    Type type,
    MutableArrayRef<MatchCase> cases
) {
    // FIXME: Try to avoid heap-allocating these if possible.
    // TODO: Implementing a caching mechanism should help.
    auto AllocateMatchContext = [&] (this auto& self, Type ty) -> std::unique_ptr<MatchContext> {
        if (ty->is_integer()) {
            return std::make_unique<IntMatchContext>(*this, ty);
        } else if (ty == Type::BoolTy) {
            return std::make_unique<BoolMatchContext>(*this);
        } else if (auto o = dyn_cast<OptionalType>(ty)) {
            auto inner = self(o->elem());
            if (not inner) return nullptr;
            return std::make_unique<OptionalMatchContext>(*this, o, std::move(inner));
        } else {
            return Error(
                control->location(),
                "Matching a value of type '{}' is not supported",
                control->type
            );
        }
    };

    auto mc = AllocateMatchContext(type);
    if (not mc) return false;
    return CheckMatchExhaustiveImpl(*mc, loc, control, type, cases);
}

bool Sema::CheckMatchExhaustiveImpl(
    MatchContext& mc,
    SLoc match_loc,
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
        auto rv = RequestEvaluation(LValueToRValue(e), false);
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
