module;

#include <base/Assert.hh>
#include <base/Macros.hh>
#include <llvm/ADT/STLExtras.h>
#include <ranges>

module srcc.frontend.sema;
import srcc.utils;
using namespace srcc;

#undef Try
#define Try(X)                ({ auto _x = (X); if (not _x) return nullptr; _x.get(); })
#define TryInstantiateExpr(X) Try(InstantiateExpr(X))
#define TryInstantiateStmt(X) Try(InstantiateStmt(X))

class srcc::TemplateInstantiator {
    friend Sema;

    Sema& S;
    Sema::TemplateArguments template_arguments;

    struct InstantiationScopeInfo : Sema::ProcScopeInfo {
        ProcDecl* pattern;
        DenseMap<LocalDecl*, LocalDecl*> local_instantiations;

        InstantiationScopeInfo(
            ProcDecl* pattern,
            ProcDecl* instantiation
        ) : ProcScopeInfo{instantiation}, pattern{pattern} {}

        static bool classof(const ProcScopeInfo* info) {
            return info->proc->instantiated_from != nullptr;
        }
    };

    using EnterInstantiation = Sema::EnterProcedure<InstantiationScopeInfo>;

    explicit TemplateInstantiator(
        Sema& S,
        Sema::TemplateArguments args
    ) : S{S}, template_arguments{std::move(args)} {}

    // Steal the template arguments from the instantiator.
    auto take_args() -> Sema::TemplateArguments&& { return std::move(template_arguments); }

    /// Traverse scopes of procedures that we’re instantiating.
    auto InstantiationStack() {
        return S.proc_stack | vws::filter(InstantiationScopeInfo::classof) | vws::transform([](auto x) { return static_cast<InstantiationScopeInfo*>(x); });
    }

    auto InstantiateProcedure(ProcDecl* proc, ProcType* substituted_type) -> Ptr<ProcDecl>;

    // Type Instantiation
    auto InstantiateTemplateTypeDecl(TemplateTypeDecl* d) -> Decl*;
    auto InstantiateLocalDecl(LocalDecl* d) -> LocalDecl*;
    auto InstantiateParamDecl(ParamDecl* d) -> ParamDecl*;
    auto InstantiateProcDecl(ProcDecl* d) -> ProcDecl*;

    // Stmt instantiation.
    auto InstantiateStmt(Stmt* stmt) -> Ptr<Stmt>;
    auto InstantiateBlockExpr(BlockExpr* e) -> Ptr<Expr>;
    auto InstantiateBuiltinCallExpr(BuiltinCallExpr* e) -> Ptr<Expr>;
    auto InstantiateCallExpr(CallExpr* e) -> Ptr<Expr>;
    auto InstantiateCastExpr(CastExpr* e) -> Ptr<Expr>;
    auto InstantiateConstExpr(ConstExpr* e) -> Ptr<Expr>;
    auto InstantiateEvalExpr(EvalExpr* e) -> Ptr<Expr>;
    auto InstantiateExpr(Expr* e) -> Ptr<Expr>;
    auto InstantiateIntLitExpr(IntLitExpr* e) -> Ptr<Expr>;
    auto InstantiateLocalRefExpr(LocalRefExpr* e) -> Ptr<Expr>;
    auto InstantiateProcRefExpr(ProcRefExpr* e) -> Ptr<Expr>;
    auto InstantiateReturnExpr(ReturnExpr* e) -> Ptr<Expr>;
    auto InstantiateSliceDataExpr(SliceDataExpr* e) -> Ptr<Expr>;
    auto InstantiateStrLitExpr(StrLitExpr* e) -> Ptr<Expr>;

    // Type Instantiation
    auto InstantiateType(Type ty) -> Type;
    auto InstantiateArrayType(ArrayType*) -> Type { Todo(); }
    auto InstantiateBuiltinType(BuiltinType* ty) -> Type;
    auto InstantiateIntType(IntType*) -> Type { Unreachable("Never dependent"); }
    auto InstantiateProcType(ProcType* ty) -> Type;
    auto InstantiateReferenceType(ReferenceType*) -> Type { Todo(); }
    auto InstantiateSliceType(SliceType*) -> Type { Todo(); }
    auto InstantiateTemplateType(TemplateType* ty) -> Type;
};

// ============================================================================
//  Deduction
// ============================================================================
auto Sema::DeduceType(TemplateTypeDecl* decl, Type param, Type arg) -> Type {
    // TODO: More complicated deduction, e.g. $T[4].
    if (isa<TemplateType>(param)) return arg;
    Error(
        decl->location(),
        "Cannot deduce ${} in {} from {}",
        decl->name,
        param->print(ctx.use_colours()),
        arg->print(ctx.use_colours())
    );
    return Types::ErrorDependentTy;
}

// ============================================================================
//  Instantiation
// ============================================================================
auto Sema::InstantiateTemplate(
    ProcDecl* proc,
    ProcType* substituted_type,
    TemplateArguments& args
) -> Ptr<ProcDecl> {
    // Reuse the instantiation if we’ve already done it.
    auto& cache = M->template_instantiations[proc];
    if (auto it = cache.find(substituted_type); it != cache.end())
        return it->second;

    // Dew it.
    TemplateInstantiator I{*this, args};
    auto instantiated = I.InstantiateProcedure(proc, substituted_type);

    // Record the result, irrespective of whether we succeeded or not.
    M->template_instantiations[proc][substituted_type] = instantiated.get_or_null();
    return instantiated;
}

auto TemplateInstantiator::InstantiateProcedure(
    ProcDecl* proc,
    ProcType* substituted_type
) -> Ptr<ProcDecl> {
    Assert(proc->body(), "Instantiating procedure w/o body?");
    auto inst = ProcDecl::Create(
        *S.M,
        substituted_type,
        proc->name,
        Linkage::Merge,
        proc->mangling,
        proc->parent,
        proc->location(),
        {}
    );

    // Remember that we’re an instantiation.
    inst->instantiated_from = proc;

    // Instantiate the parameters.
    EnterInstantiation _{S, proc, inst};
    for (auto param : proc->params()) InstantiateParamDecl(param);
    auto body = InstantiateStmt(proc->body().get());
    inst->finalise(body, S.curr_proc().locals);
    return inst;
}

// ============================================================================
//  Decl Instantiation
// ============================================================================
auto TemplateInstantiator::InstantiateLocalDecl(LocalDecl* d) -> LocalDecl* {
    Todo("Instantiate local decl");
}

auto TemplateInstantiator::InstantiateParamDecl(ParamDecl* d) -> ParamDecl* {
    auto ty = InstantiateType(d->type);
    auto param = S.BuildParamDecl(S.curr_proc(), ty, d->name, d->location());
    cast<InstantiationScopeInfo>(S.curr_proc()).local_instantiations[d] = param;
    return param;
}

auto TemplateInstantiator::InstantiateProcDecl(ProcDecl*) -> ProcDecl* {
    Todo("Instantiate nested proc decl");
}

auto TemplateInstantiator::InstantiateTemplateTypeDecl(TemplateTypeDecl*) -> Decl* {
    Todo("Instantiate nested template type decl");
}

// ============================================================================
//  Stmt Instantiation
// ============================================================================
auto TemplateInstantiator::InstantiateStmt(Stmt* stmt) -> Ptr<Stmt> {
    if (not stmt->dependent()) return stmt;

    switch (stmt->kind()) {
#define AST_STMT_LEAF(node) \
    case Stmt::Kind::node: return Instantiate##node(cast<node>(stmt));
#include "srcc/AST.inc"
    }

    Unreachable("Invalid stmt kind");
}

auto TemplateInstantiator::InstantiateBlockExpr(BlockExpr* e) -> Ptr<Expr> {
    SmallVector<Stmt*> stmts;
    for (auto stmt : e->stmts()) stmts.push_back(TryInstantiateStmt(stmt));
    return S.BuildBlockExpr(e->scope, stmts, e->location());
}

auto TemplateInstantiator::InstantiateBuiltinCallExpr(BuiltinCallExpr* e) -> Ptr<Expr> {
    // Some builtins are instantiated as though they were
    // a regular call expression.
    auto InstantiateAsCall = [&] -> Ptr<Expr> {
        SmallVector<Expr*> args;
        for (auto arg : e->args()) args.push_back(TryInstantiateExpr(arg));
        return S.BuildBuiltinCallExpr(e->builtin, args, e->location());
    };

    switch (e->builtin) {
        using B = BuiltinCallExpr::Builtin;
        case B::Print:
            return InstantiateAsCall();
    }

    Unreachable("Invalid builtin");
}

auto TemplateInstantiator::InstantiateCallExpr(CallExpr* e) -> Ptr<Expr> {
    SmallVector<Expr*> args;
    auto callee = TryInstantiateExpr(e->callee);
    for (auto arg : e->args()) args.push_back(TryInstantiateExpr(arg));
    return S.BuildCallExpr(callee, args, e->location());
}

auto TemplateInstantiator::InstantiateCastExpr(CastExpr* e) -> Ptr<Expr> {
    [[maybe_unused]] auto arg = TryInstantiateExpr(e->arg);
    switch (e->kind) {
        case CastExpr::LValueToSRValue: Unreachable("Never dependent");
    }
    Unreachable("Invalid cast");
}

auto TemplateInstantiator::InstantiateConstExpr(ConstExpr* e) -> Ptr<Expr> {
    // This is never dependent and only created after evaluation; the
    // eval expression possibly contained within is always marked as
    // dependent because it isn’t supposed to be used anymore after
    // evaluation, so never instantiate the argument here.
    Assert(not e->dependent(), "Dependent ConstExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateEvalExpr(EvalExpr* e) -> Ptr<Expr> {
    return S.BuildEvalExpr(TryInstantiateStmt(e->stmt), e->location());
}

auto TemplateInstantiator::InstantiateExpr(Expr* e) -> Ptr<Expr> {
    auto expr = TryInstantiateStmt(e);
    Assert(isa<Expr>(expr), "Expression instantiated to statement?");
    return cast<Expr>(expr);
}

auto TemplateInstantiator::InstantiateIntLitExpr(IntLitExpr* e) -> Ptr<Expr> {
    Assert(not e->dependent(), "Dependent IntLitExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateLocalRefExpr(LocalRefExpr* e) -> Ptr<Expr> {
    for (auto inst : InstantiationStack() | vws::reverse)
        if (auto d = inst->local_instantiations.find(e->decl); d != inst->local_instantiations.end())
            return new (*S.M) LocalRefExpr{d->second, e->location()};
    Unreachable("Local not instantiated?");
}

auto TemplateInstantiator::InstantiateSliceDataExpr(SliceDataExpr* e) -> Ptr<Expr> {
    return SliceDataExpr::Create(*S.M, TryInstantiateExpr(e->slice), e->location());
}

auto TemplateInstantiator::InstantiateProcRefExpr(ProcRefExpr* e) -> Ptr<Expr> {
    Assert(not e->decl->is_template(), "TODO: Instantiate reference to template");
    return e;
}

auto TemplateInstantiator::InstantiateStrLitExpr(StrLitExpr* e) -> Ptr<Expr> {
    Assert(not e->dependent(), "Dependent StrLitExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateReturnExpr(ReturnExpr* e) -> Ptr<Expr> {
    Assert(e->value.present(), "Dependent return expression w/o argument?");
    return S.BuildReturnExpr(TryInstantiateExpr(e->value.get()), e->location(), e->implicit);
}

// ============================================================================
//  Type Instantiation
// ============================================================================
auto TemplateInstantiator::InstantiateType(Type ty) -> Type {
    if (not ty->dependent()) return ty;

    switch (ty->kind()) {
#define AST_TYPE_LEAF(node) \
    case TypeBase::Kind::node: return Instantiate##node(cast<node>(ty).ptr());
#include "srcc/AST.inc"
    }

    Unreachable("Invalid type kind");
}

auto TemplateInstantiator::InstantiateBuiltinType(BuiltinType* ty) -> Type {
    // This can happen if someone tries to instantiate e.g. DependentTy;
    // unfortunately for them, there is nothing to be done here.
    return ty;
}

auto TemplateInstantiator::InstantiateProcType(ProcType* ty) -> Type {
    SmallVector<Type, 4> params;
    auto ret = InstantiateType(ty->ret());
    for (auto param : ty->params()) params.push_back(InstantiateType(param));
    return ProcType::Get(*S.M, ret, params, ty->cconv(), ty->variadic());
}

auto TemplateInstantiator::InstantiateTemplateType(TemplateType* ty) -> Type {
    auto it = template_arguments.find(ty->template_decl());

    // Can’t instantiate this yet.
    //
    // TODO: If we’re instantiating a template that contains another
    // template, we need to rebuild any template types that refer to
    // the template parameters of the nested template to refer to the
    // template parameters of the copy of the nested template created
    // for the instantiation of the outer template.
    if (it == template_arguments.end()) return ty;

    // Resolve it!
    return it->second;
}

// ============================================================================
//  Substitution
// ============================================================================
auto Sema::SubstituteTemplate(
    ProcDecl* proc_template,
    ArrayRef<TypeLoc> input_types
) -> std::pair<Type, TemplateArguments> {
    Assert(proc_template->is_template(), "Instantiating non-template?");
    auto proc_type = proc_template->proc_type();
    TemplateArguments args;

    // First, perform template deduction against any parameters that need it.
    for (auto param : proc_template->template_params()) {
        // Iterate over all arguments for which this parameter needs to be deduced.
        auto idxs = param->deduced_indices();
        Assert(not idxs.empty());

        // Also keep track of the index of the index which we’re processing so we
        // can use it to get the previous index for diagnostics.
        for (auto [index_of_index, idx] : enumerate(idxs)) {
            auto& input_ty = input_types[idx];
            auto deduced = DeduceType(param, proc_type->params()[idx], input_ty.ty);

            // There was an error deducing this; throw it away.
            if (deduced->dependence() & Dependence::Error)
                continue;

            // If the value was already set, but the two don’t match, we have to
            // report the mismatch.
            if (auto it = args.find(param); it != args.end() and it->second != deduced) {
                Assert(index_of_index != 0, "Mismatch on first argument?");
                using enum utils::Colour;
                utils::Colours C{ctx.use_colours()};
                Error(
                    input_ty.loc,
                    "Template deduction mismatch for parameter {}${}{}:\n"
                    "    Argument #{}: Deduced as {}\n"
                    "    Argument #{}: Deduced as {}",
                    C(Yellow),
                    param->name,
                    C(Reset),
                    idxs[index_of_index - 1], // Mismatch was w/ previous index.
                    it->second->print(C.use_colours),
                    idx,
                    deduced.print(C.use_colours)
                );

                // Trying to deduce from other args is only going
                // to result in more errors.
                break;
            }

            // Otherwise, just remember the value and keep going.
            args[param] = deduced.ptr();
        }

        // We couldn’t deduce this. Give up.
        if (not args.contains(param)) return {ProcType::GetInvalid(*M), TemplateArguments{}};
    }

    TemplateInstantiator I{*this, std::move(args)};
    auto ty = I.InstantiateType(proc_template->type);
    return {ty, I.take_args()};
}
