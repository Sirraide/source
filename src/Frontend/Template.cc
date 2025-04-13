#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Sema.hh>

#include <llvm/ADT/STLExtras.h>

#include <base/Assert.hh>
#include <base/Macros.hh>

#include <ranges>

/*
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
    return std::nullopt;
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
            if (not deduced_opt) return TempSubstRes::DeductionFailed{
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
*/
