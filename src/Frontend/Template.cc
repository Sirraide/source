module;

#include <base/Assert.hh>
#include <llvm/ADT/STLExtras.h>
#include <ranges>

module srcc.frontend.sema;
import srcc.utils;
using namespace srcc;

using TemplateArguments = DenseMap<TemplateTypeDecl*, TypeBase*>;

class srcc::TemplateInstantiator {
    friend Sema;

    Sema& S;
    TemplateArguments template_arguments;

    explicit TemplateInstantiator(
        Sema& S,
        TemplateArguments args
    ) : S(S), template_arguments(args) {}

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
//  Deduce
// ============================================================================
auto Sema::DeduceType(TemplateTypeDecl* ty, Type arg) -> Type {
    Todo();
}


// ============================================================================
//  Instantiation
// ============================================================================
auto Sema::InstantiateTemplate(ProcDecl*, ProcType*) -> Ptr<ProcDecl> {
    Todo("Instantiate template");
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
auto Sema::SubstituteTemplate(ProcDecl* proc_template, ArrayRef<TypeLoc> input_types) -> Type {
    Assert(proc_template->is_template(), "Instantiating non-template?");
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
            auto deduced = DeduceType(param, input_ty.ty);

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
        if (not args.contains(param)) return ProcType::GetInvalid(*M);
    }

    TemplateInstantiator I{*this, args};
    return I.InstantiateType(proc_template->type);
}
