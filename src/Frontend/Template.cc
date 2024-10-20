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
    Sema::TemplateArguments& template_arguments;
    Location inst_loc;

    using EnterInstantiation = Sema::EnterProcedure<Sema::InstantiationScopeInfo>;

    explicit TemplateInstantiator(
        Sema& S,
        Sema::TemplateArguments& args,
        Location inst_loc
    ) : S{S}, template_arguments{args}, inst_loc{inst_loc} {}

    auto InstantiateProcedure(ProcDecl* proc, ProcType* substituted_type) -> Ptr<ProcDecl>;

#define AST_DECL_LEAF(Class) [[nodiscard]] auto Instantiate##Class(Class* n)->Class*;
#define AST_STMT_LEAF(Class) [[nodiscard]] auto Instantiate##Class(Class* n)->Ptr<Stmt>;
#define AST_TYPE_LEAF(Class) [[nodiscard]] auto Instantiate##Class(Class* n)->Type;
#include "srcc/AST.inc"

    auto InstantiateStmt(Stmt* stmt) -> Ptr<Stmt>;
    auto InstantiateExpr(Expr* e) -> Ptr<Expr>;
    auto InstantiateType(Type ty) -> Type;
};

// ============================================================================
//  Deduction
// ============================================================================
auto Sema::DeduceType(TemplateTypeDecl* decl, Type param, Type arg) -> Opt<Type> {
    // TODO: More complicated deduction, e.g. $T[4].
    if (isa<TemplateType>(param)) return arg;
    return Type{Types::ErrorDependentTy};
}

// ============================================================================
//  Instantiation
// ============================================================================
auto Sema::InstantiateTemplate(
    ProcDecl* proc,
    ProcType* substituted_type,
    TemplateArguments& args,
    Location inst_loc
) -> Ptr<ProcDecl> {
    // Reuse the instantiation if we’ve already done it.
    auto& cache = M->template_instantiations[proc];
    if (auto it = cache.find(substituted_type); it != cache.end())
        return it->second;

    // Dew it.
    TemplateInstantiator I{*this, args, inst_loc};
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
    inst->scope = proc->scope;

    // Instantiate params.
    EnterInstantiation _{S, proc, inst, inst_loc};
    auto& info = cast<Sema::InstantiationScopeInfo>(S.curr_proc());
    for (auto [i, param] : enumerate(proc->params())) {
        auto new_decl = new (*S.M) ParamDecl{
            &substituted_type->params()[i],
            param->name,
            inst,
            param->index(),
            param->is_with_param(),
            param->location(),
        };

        S.DeclareLocal(new_decl);
        info.instantiated_decls[param] = new_decl;
    }

    // Instantiate procedure body.
    auto body = InstantiateStmt(proc->body().get());
    inst->finalise(body, S.curr_proc().locals);
    return inst;
}

// ============================================================================
//  Decl Instantiation
// ============================================================================
auto TemplateInstantiator::InstantiateFieldDecl(FieldDecl* n) -> FieldDecl* {
    Todo();
}

auto TemplateInstantiator::InstantiateLocalDecl(LocalDecl* d) -> LocalDecl* {
    Todo("Instantiate local decl");
}

auto TemplateInstantiator::InstantiateParamDecl(ParamDecl*) -> ParamDecl* {
    Unreachable("Params are not supposed to be instantiated this way");
}

auto TemplateInstantiator::InstantiateProcDecl(ProcDecl*) -> ProcDecl* {
    Todo("Instantiate nested proc decl");
}

auto TemplateInstantiator::InstantiateTemplateTypeDecl(TemplateTypeDecl*) -> TemplateTypeDecl* {
    Todo("Instantiate nested template type decl");
}

auto TemplateInstantiator::InstantiateTypeDecl(TypeDecl* n) -> TypeDecl* {
    Todo();
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

auto TemplateInstantiator::InstantiateAssertExpr(AssertExpr* n) -> Ptr<Stmt> {
    auto cond = TryInstantiateExpr(n->cond);
    auto msg = n->message ? TryInstantiateExpr(n->message.get()) : nullptr;
    return S.BuildAssertExpr(cond, msg, n->location());
}

auto TemplateInstantiator::InstantiateBinaryExpr(BinaryExpr* n) -> Ptr<Stmt> {
    auto lhs = TryInstantiateExpr(n->lhs);
    auto rhs = TryInstantiateExpr(n->rhs);
    return S.BuildBinaryExpr(n->op, lhs, rhs, n->location());
}

auto TemplateInstantiator::InstantiateBlockExpr(BlockExpr* e) -> Ptr<Stmt> {
    SmallVector<Stmt*> stmts;
    for (auto stmt : e->stmts()) stmts.push_back(TryInstantiateStmt(stmt));
    return S.BuildBlockExpr(e->scope, stmts, e->location());
}

auto TemplateInstantiator::InstantiateBoolLitExpr(BoolLitExpr*) -> Ptr<Stmt> {
    Unreachable("Never dependent");
}

auto TemplateInstantiator::InstantiateBuiltinCallExpr(BuiltinCallExpr* e) -> Ptr<Stmt> {
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

auto TemplateInstantiator::InstantiateBuiltinMemberAccessExpr(
    BuiltinMemberAccessExpr* n
) -> Ptr<Stmt> {
    auto expr = TryInstantiateExpr(n->operand);
    return S.BuildBuiltinMemberAccessExpr(n->access_kind, expr, n->location());
}

auto TemplateInstantiator::InstantiateCallExpr(CallExpr* e) -> Ptr<Stmt> {
    SmallVector<Expr*> args;
    auto callee = TryInstantiateExpr(e->callee);
    for (auto arg : e->args()) args.push_back(TryInstantiateExpr(arg));
    return S.BuildCallExpr(callee, args, e->location());
}

auto TemplateInstantiator::InstantiateCastExpr(CastExpr* e) -> Ptr<Stmt> {
    [[maybe_unused]] auto arg = TryInstantiateExpr(e->arg);
    switch (e->kind) {
        case CastExpr::LValueToSRValue:
        case CastExpr::Integral:
            Unreachable("Never dependent");
    }
    Unreachable("Invalid cast");
}

auto TemplateInstantiator::InstantiateConstExpr(ConstExpr* e) -> Ptr<Stmt> {
    // This is never dependent and only created after evaluation; the
    // eval expression possibly contained within is always marked as
    // dependent because it isn’t supposed to be used anymore after
    // evaluation, so never instantiate the argument here.
    Assert(not e->dependent(), "Dependent ConstExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateDefaultInitExpr(DefaultInitExpr*) -> Ptr<Stmt> {
    Unreachable("Never dependent");
}

auto TemplateInstantiator::InstantiateEvalExpr(EvalExpr* e) -> Ptr<Stmt> {
    return S.BuildEvalExpr(TryInstantiateStmt(e->stmt), e->location());
}

auto TemplateInstantiator::InstantiateExpr(Expr* e) -> Ptr<Expr> {
    auto expr = TryInstantiateStmt(e);
    Assert(isa<Expr>(expr), "Expression instantiated to statement?");
    return cast<Expr>(expr);
}

auto TemplateInstantiator::InstantiateIfExpr(IfExpr* n) -> Ptr<Stmt> {
    auto cond = TryInstantiateExpr(n->cond);
    auto then_ = TryInstantiateStmt(n->then);
    auto else_ = n->else_ ? TryInstantiateStmt(n->else_.get()) : nullptr;
    return S.BuildIfExpr(cond, then_, else_, n->location());
}

auto TemplateInstantiator::InstantiateIntLitExpr(IntLitExpr* e) -> Ptr<Stmt> {
    Assert(not e->dependent(), "Dependent IntLitExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateLocalRefExpr(LocalRefExpr* e) -> Ptr<Stmt> {
    for (auto inst : S.InstantiationStack() | vws::reverse)
        if (auto d = inst->instantiated_decls.find(e->decl); d != inst->instantiated_decls.end())
            return new (*S.M) LocalRefExpr{cast<LocalDecl>(d->second), e->location()};
    Unreachable("Local not instantiated?");
}

auto TemplateInstantiator::InstantiateOverloadSetExpr(OverloadSetExpr* n) -> Ptr<Stmt> {
    Todo();
}

auto TemplateInstantiator::InstantiateParenExpr(ParenExpr* n) -> Ptr<Stmt> {
    auto inner = TryInstantiateExpr(n->expr);
    return new (*S.M) ParenExpr(inner, n->location());
}

auto TemplateInstantiator::InstantiateProcRefExpr(ProcRefExpr* e) -> Ptr<Stmt> {
    Assert(not e->decl->is_template(), "TODO: Instantiate reference to template");
    return e;
}

auto TemplateInstantiator::InstantiateStaticIfExpr(StaticIfExpr* n) -> Ptr<Stmt> {
    auto cond = TryInstantiateExpr(n->cond);
    return S.BuildStaticIfExpr(
        cond,
        static_cast<ParsedStmt*>(n->then),
        static_cast<ParsedStmt*>(n->else_.get_or_null()),
        n->location()
    );
}

auto TemplateInstantiator::InstantiateStrLitExpr(StrLitExpr* e) -> Ptr<Stmt> {
    Assert(not e->dependent(), "Dependent StrLitExpr?");
    return e;
}

auto TemplateInstantiator::InstantiateTypeExpr(TypeExpr* e) -> Ptr<Stmt> {
    return new (*S.M) TypeExpr(InstantiateType(e->type), e->location());
}

auto TemplateInstantiator::InstantiateReturnExpr(ReturnExpr* e) -> Ptr<Stmt> {
    Assert(e->value.present(), "Dependent return expression w/o argument?");
    return S.BuildReturnExpr(TryInstantiateExpr(e->value.get()), e->location(), e->implicit);
}

auto TemplateInstantiator::InstantiateUnaryExpr(UnaryExpr* n) -> Ptr<Stmt> {
    Todo();
}

auto TemplateInstantiator::InstantiateWhileStmt(WhileStmt* n) -> Ptr<Stmt> {
    Todo();
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

auto TemplateInstantiator::InstantiateArrayType(ArrayType* ty) -> Type {
    return ArrayType::Get(*S.M, InstantiateType(ty->elem()), ty->dimension());
}

auto TemplateInstantiator::InstantiateBuiltinType(BuiltinType* ty) -> Type {
    // This can happen if someone tries to instantiate e.g. DependentTy;
    // unfortunately for them, there is nothing to be done here.
    return ty;
}

auto TemplateInstantiator::InstantiateIntType(IntType*) -> Type {
    Unreachable("Never dependent");
}

auto TemplateInstantiator::InstantiateMemberAccessExpr(MemberAccessExpr* n) -> Ptr<Stmt> {
    Todo();
}

auto TemplateInstantiator::InstantiateProcType(ProcType* ty) -> Type {
    SmallVector<ParamTypeData, 4> params;
    auto ret = InstantiateType(ty->ret());
    for (auto param : ty->params()) params.emplace_back(param.intent, InstantiateType(param.type));
    return ProcType::Get(*S.M, ret, params, ty->cconv(), ty->variadic());
}

auto TemplateInstantiator::InstantiateReferenceType(ReferenceType* ty) -> Type {
    return ReferenceType::Get(*S.M, InstantiateType(ty->elem()));
}

auto TemplateInstantiator::InstantiateSliceType(SliceType* ty) -> Type {
    return SliceType::Get(*S.M, InstantiateType(ty->elem()));
}

auto TemplateInstantiator::InstantiateStructType(StructType* n) -> Type {
    Todo();
}

auto TemplateInstantiator::InstantiateStructInitExpr(StructInitExpr*) -> Ptr<Stmt> {
    Unreachable("Never dependent");
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
    ArrayRef<TypeLoc> input_types,
    Location inst_loc
) -> TempSubstRes {
    Assert(proc_template->is_template(), "Instantiating non-template?");
    TemplateArguments args;
    auto proc_type = proc_template->proc_type();

    // First, perform template deduction against any parameters that need it.
    for (auto param : proc_template->template_params()) {
        // Iterate over all arguments for which this parameter needs to be deduced.
        auto idxs = param->deduced_indices();
        Assert(not idxs.empty());

        // Also keep track of the index of the index which we’re processing so we
        // can use it to get the previous index for diagnostics.
        for (auto [index_of_index, idx] : enumerate(idxs)) {
            auto& input_ty = input_types[idx];
            auto deduced_opt = DeduceType(param, proc_type->params()[idx].type, input_ty.ty);

            // There was an error.
            if (not deduced_opt) return {};
            if (deduced_opt.value() == Types::ErrorDependentTy) return TempSubstRes::DeductionFailed{
                param,
                idx,
            };

            // If the value was already set, but the two don’t match, we have to
            // report the mismatch.
            auto deduced = deduced_opt.value();
            if (auto it = args.find(param); it != args.end() and it->second != deduced) {
                Assert(index_of_index != 0, "Mismatch on first argument?");
                return TempSubstRes::DeductionAmbiguous{
                    .ttd = param,
                    .first = idxs[index_of_index - 1], // Mismatch was w/ previous index.
                    .second = idx,
                    .first_type = args[param],
                    .second_type = deduced.ptr(),
                };
            }

            // Otherwise, just remember the value and keep going.
            args[param] = deduced.ptr();
        }
    }

    TemplateInstantiator I{*this, args, inst_loc};
    auto substituted_type = cast<ProcType>(I.InstantiateType(proc_template->type)).ptr();
    return TempSubstRes::Success{substituted_type, std::move(args)};
}
