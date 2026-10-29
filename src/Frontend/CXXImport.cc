#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/ExternalASTSource.h>
#include <clang/AST/DeclarationName.h>
#include <clang/AST/RecordLayout.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <srcc/AST/CXX.hh>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/MultiplexConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Parse/Parser.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Driver/CreateInvocationFromArgs.h>

#include <base/Assert.hh>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/VirtualFileSystem.h>

using namespace srcc;
using clang::QualType;

#define TRY(expr) ({                                                       \
    auto _v = expr;                                                        \
    if (not _v.has_value()) return std::unexpected(std::move(_v.error())); \
    *_v;                                                                   \
})

template <typename Value>
struct Res : std::expected<Value, std::unique_ptr<Diagnostic>> {
    using Base = std::expected<Value, std::unique_ptr<Diagnostic>>;
    using Base::Base;

    Res() = delete("Return a diagnostic instead");
    Res(std::nullptr_t) = delete("Return a diagnostic instead");
    Res(std::nullopt_t) = delete("Return a diagnostic instead");
};

class srcc::CXXImporter {
    Sema& S;
    ImportedClangModuleDecl* clang_module;

    static constexpr String ConstructorWrapperAnnotation = "cxx_constructor_wrapper";

public:
    explicit CXXImporter(Sema& S, ImportedClangModuleDecl* clang_module) : S(S), clang_module(clang_module) {}
    auto AST() -> clang::ASTContext& { return S.tu->cxx_compilation()->getASTContext(); }

    auto ImportDecl(clang::Decl* d) -> Res<Decl*>;
    auto ImportName(clang::TagDecl* td) -> StringRef;
    auto ImportReturnType(SLoc sloc, QualType ty) -> Res<TypeAndValueCategory>;
    auto ImportType(clang::SourceLocation sloc, QualType ty) -> Res<Type>;
    auto ImportType(SLoc sloc, QualType ty) -> Res<Type>;
    auto ImportSourceLocation(clang::SourceLocation sloc) -> SLoc;
    auto ImportValue(SLoc loc, const clang::APValue& val, Type ty) -> Res<eval::RValue>;

private:
    auto BuildCtorOrDtorBody(
        StructType* type,
        clang::CXXRecordDecl* record,
        Ptr<clang::CXXConstructorDecl> constructor,
        ProcDecl* proc
    ) -> Ptr<Expr>;

    static auto FormatAnnotation(String s, StringRef args = {}) -> SmallString<64>;
    auto GetAnnotation(clang::Decl* d, String s) -> Ptr<clang::AnnotateAttr>;
    auto ImportDeclImpl(clang::Decl* d) -> Res<Decl*>;
    auto ImportFunctionImpl(clang::FunctionDecl* d) -> Res<ProcDecl*>;
    void ImportCtorsAndDtor(StructType* type, clang::CXXRecordDecl* record);
    auto ImportRecordImpl(clang::RecordDecl* rd) -> Res<TypeDecl*>;

public:
    // Convert 'QualType' and other clang types that can't be formatted w/o an ASTContext
    // to a string so it can be used in a diagnostic.
    // FIXME: This should really be a member function.
    template <typename T>
    static auto PreprocessDiagArg(CXXImporter& i, T val) {
        using NonRef = std::remove_cvref_t<T>;
        if constexpr (utils::is<NonRef, QualType, clang::CanQualType>) {
            SmallString<32> str;
            llvm::raw_svector_ostream os{str};
            val.print(os, clang::PrintingPolicy(i.S.tu->cxx_compilation()->getLangOpts()));
            return str;
        } else if constexpr (utils::is<NonRef, clang::Type*, const clang::Type*>) {
            return PreprocessDiagArg(i, QualType(val, 0));
        } else if constexpr (utils::is<NonRef, clang::APValue>) {
            SmallString<128> str;
            llvm::raw_svector_ostream os{str};
            val.dump(os, i.S.tu->cxx_compilation()->getASTContext());
            return str;
        } else if constexpr (std::convertible_to<NonRef, clang::Decl*>) {
            if (auto n = dyn_cast<clang::NamedDecl>(val)) {
                return std::format("{}", n->getQualifiedNameAsString());
            } else {
                return std::string(static_cast<const clang::Decl*>(val)->getDeclKindName());
            }
        } else {
            return std::move(val);
        }
    }

    template <typename Entity>
    auto GetEntityLoc(Entity e) -> SLoc {
        using Pointee = std::remove_pointer_t<Entity>;
        if constexpr (std::derived_from<Pointee, clang::Decl>) {
            return ImportSourceLocation(e->getLocation());
        } else if constexpr (std::derived_from<Pointee, clang::Stmt>) {
            return ImportSourceLocation(e->getBeginLoc());
        } else if constexpr (std::is_same_v<Entity, clang::SourceLocation>) {
            return ImportSourceLocation(e);
        } else if constexpr (std::is_same_v<Entity, SLoc>) {
            return e;
        } else {
            static_assert(false, "Unknown entity");
        }
    }

    template <typename Entity, typename... Args>
    auto MakeErr(
        Entity e,
        std::format_string<decltype(PreprocessDiagArg(std::declval<CXXImporter&>(), std::declval<Args>()))...> fmt,
        Args&&... args
    ) {
        auto diag = S.CreateNote(GetEntityLoc(e), fmt, PreprocessDiagArg(*this, std::forward<Args>(args))...);
        return std::unexpected(std::make_unique<Diagnostic>(diag));
    }

    template <typename Entity, typename... Args>
    auto NYI(
        Entity e,
        std::format_string<decltype(PreprocessDiagArg(std::declval<CXXImporter&>(), std::declval<Args>()))...> fmt,
        Args&&... args
    ){
        auto msg = Format(fmt, PreprocessDiagArg(*this, std::forward<Args>(args))...);
        if (S.tu->lang_opts().wcxx_import) Warn(GetEntityLoc(e), "Importing {} is not yet supported", msg);
        return MakeErr(e, "Importing {} is not yet supported", msg);
    }

    template <typename... Args>
    void Warn(
        SLoc loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        if (not S.tu->lang_opts().wcxx_import) return;
        S.ReportDiag(DiagsProducer::CreateWarning(loc, fmt, std::forward<Args>(args)...));
    }

    auto AlreadyDiagnosed() {
        return std::unexpected(std::unique_ptr<Diagnostic>());
    }
};

// The 'canonical decl' of a decl can change, because of course it can,
// so this function instead tries to provide a *stable* 'canonical' decl.
static auto GetStableCanonicalDecl(clang::Decl* d) -> clang::Decl* {
    return llvm::TypeSwitch<clang::Decl*, clang::Decl*>(d)
        .Case<clang::RecordDecl>([](auto* rd) { return rd->getDefinition() ?: rd->getFirstDecl(); })
        .Case<clang::FunctionDecl>([](auto* fd) { return fd->getDefinition() ?: fd->getFirstDecl(); })
        .Case<clang::EnumConstantDecl>([](auto* fd) { return fd->getFirstDecl(); })
        .Default([&](auto* d){ return d->getCanonicalDecl(); });
}

static auto MangleCXXDecl(clang::MangleContext& context, clang::GlobalDecl decl) -> SmallString<128> {
    SmallString<128> name;
    llvm::raw_svector_ostream stream(name);
    context.mangleName(decl, stream);
    return name;
}

/// Build the body of a constructor or destructor. If 'constructor' is
/// null, we’re building the destructor.
auto srcc::CXXImporter::BuildCtorOrDtorBody(
    StructType* type,
    clang::CXXRecordDecl* record,
    Ptr<clang::CXXConstructorDecl> constructor,
    ProcDecl* proc
) -> Ptr<Expr> {
    auto loc = proc->location();
    auto& compilation = *S.tu->cxx_compilation().get();

    // Get the mangled name of the ctor/dtor.
    auto global = constructor
        ? clang::GlobalDecl(constructor.get(), clang::Ctor_Complete)
        : clang::GlobalDecl(record->getDestructor(), clang::Dtor_Complete);
    auto name = Format("__srcc_cxx_glue_{}", MangleCXXDecl(*compilation.mangler(), global));

    // Get the class name.
    auto policy = clang::PrintingPolicy(compilation.getLangOpts());
    policy.FullyQualifiedName = true;
    auto class_name = QualType(AST().getCanonicalTagType(record)).getAsString(policy);

    // Create a string that contains all the parameter declarations.
    std::string params = "void* __destination";
    std::string arguments;
    if (constructor) {
        for (auto [index, param] : enumerate(constructor->parameters())) {
            auto qt = param->getType();
            auto arg_name = Format("__arg{}", index);
            auto glue_type = qt->isReferenceType() ? AST().getPointerType(qt->getPointeeType()) : qt;
            std::string spelling;
            llvm::raw_string_ostream os(spelling);
            glue_type.print(os, policy, arg_name);
            params += ", " + spelling;
            if (index) arguments += ", ";
            arguments += qt->isReferenceType()
                ? Format("static_cast<{}>(*{})", qt.getAsString(policy), arg_name)
                : Format("static_cast<{}>({})", AST().getRValueReferenceType(qt).getAsString(policy), arg_name);
        }
    }

    // FIXME: Catch exceptions here and return an error once we have proper
    // result types.
    SmallString<256> body;
    if (constructor) body = Format("::new (__destination) _Type({});", arguments).str().str();
    else body = "static_cast<_Type*>(__destination)->_Type::~_Type();";
    auto code = Format(
        "#include <new>\n"
        "#include <type_traits>\n"
        "[[_Clang::__always_inline__]] {4}\n"
        "inline void {0}({1}) noexcept __asm__(\"{0}\");\n"
        "inline void {0}({1}) noexcept {{ using _Type = {2}; {3} }}\n",
        name,
        params,
        class_name,
        body,
        constructor ? FormatAnnotation(
            ConstructorWrapperAnnotation,
            Format("std::type_identity<{}>{{}}", class_name)
        ).str() : ""
    );

    // Parse the glue function.
    if (not compilation.parse(code)) return {};
    auto lookup = AST().getTranslationUnitDecl()->lookup(
        clang::DeclarationName(compilation.getPreprocessor().getIdentifierInfo(name))
    );

    auto helper_decl = cast<clang::FunctionDecl>(lookup.front());

    // If this is a constructor, ensure C++ overload resolution picked the right one;
    // if this ever fails, we need to fix our logic above.
    if (constructor) {
        auto definition = helper_decl->getDefinition();
        auto block = cast<clang::CompoundStmt>(definition->getBody());
        auto placement = cast<clang::CXXNewExpr>(block->body_back());
        if (auto construct = placement->getConstructExpr()) {
            auto selected = construct->getConstructor();
            if (selected->getCanonicalDecl() != constructor->getCanonicalDecl())
                return S.ICE(loc, "C++ overload resolution selected the wrong constructor for '{}'", type);
        }
    }

    // Mark it as used and emit it.
    helper_decl->setIsUsed();
    compilation.codegen().HandleTopLevelDecl(clang::DeclGroupRef(helper_decl));

    // Import it.
    auto imported = ImportDecl(helper_decl);
    if (not imported) {
        S.ReportDiag(std::move(*imported.error()));
        return {};
    }

    // Now, we need to call the glue function.
    auto helper = S.RequestReference(*imported, loc);
    if (not helper) return {};
    SmallVector<Expr*> args;

    // Forward all the parameters if this is a constructor.
    if (constructor) {
        for (auto param : proc->params()) {
            auto ref = S.RequestReference(param, loc);
            if (not ref) return {};
            args.push_back(ref.get());
        }
    }

    // If it is a destructor, pass our first argument along as a pointer.
    else {
        auto ref = S.RequestReference(proc->params().front(), loc);
        if (not ref) return {};
        auto pointer = S.BuildUnaryExpr(Tk::Ampersand, ref.get(), false, loc);
        if (not pointer) return {};
        args.push_back(new (*S.tu) CastExpr(
            PtrType::Get(*S.tu, Type::VoidTy, false),
            CastExpr::Pointer,
            pointer.get(),
            loc,
            true
        ));
    }

    // Call the glue function.
    auto tuple = TupleExpr::Create(*S.tu, Type::CallArgListTy, args, {}, loc);
    return S.BuildCallExpr(helper.get(), tuple, loc, false);
}

auto srcc::CXXImporter::FormatAnnotation(String s, StringRef args) -> SmallString<64> {
    SmallString<64> annot;
    Format(annot, "[[_Clang::__annotate__(\"__srcc__\", \"{}\"", s);
    if (not args.empty()) Format(annot, ", {}", args);
    annot += ")]]";
    return annot;
}

auto srcc::CXXImporter::GetAnnotation(clang::Decl* d, String annotation) -> Ptr<clang::AnnotateAttr> {
    for (auto attr : d->specific_attrs<clang::AnnotateAttr>()) {
        if (attr->getAnnotation() != "__srcc__" or attr->args_size() == 0) continue;
        auto kind = dyn_cast<clang::StringLiteral>((*attr->args_begin())->IgnoreParenImpCasts());
        if (kind and kind->getString() == annotation) return attr;
    }
    return {};
}

auto srcc::CXXImporter::ImportDecl(clang::Decl* d) -> Res<Decl*> {
    d = GetStableCanonicalDecl(d);

    // If we have attempted to find this before, do not do so again.
    if (auto it = S.imported_decls.find(d); it != S.imported_decls.end()){
        if (auto d = it->second.get_or_null()) return d;
        return AlreadyDiagnosed();
    }

    // Otherwise, try to create it now.
    //
    // Don’t hold on to a reference to 'S.imported_decls[D]' as importing a decl
    // might import more decls, which might invalidate it.
    S.imported_decls[d] = nullptr; // Mark that we’ve already tried doing this.
    auto imported = TRY(ImportDeclImpl(d));
    S.imported_decls[d] = imported;
    return imported;
}

auto srcc::CXXImporter::ImportDeclImpl(clang::Decl* d) -> Res<Decl*> {
    DebugAssert(
        S.imported_decls.at(d).invalid() and d == GetStableCanonicalDecl(d),
        "ImportDeclImpl() should only be called by ImportDecl()"
    );

    // Ignore invalid ones.
    Assert(d);
    if (d->isInvalidDecl()) return MakeErr(d, "declaration of '{}' is invalid", d);
    switch (d->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::Enum: {
            auto e = cast<clang::EnumDecl>(d);

            for (auto enumerator : e->enumerators())
                S.imported_decls[GetStableCanonicalDecl(enumerator)] = nullptr;

            auto underlying = TRY(ImportType(e->getLocation(), e->getIntegerType()));
            auto loc = ImportSourceLocation(e->getLocation());
            auto name = ImportName(e);
            auto scope = S.tu->create_scope<BlockScope>(S.global_scope());
            auto enum_ty = new (*S.tu) EnumType(*S.tu, scope, S.tu->save(name), underlying, loc);
            for (auto [index, enumerator] : enumerate(e->enumerators())) {
                auto decl = new (*S.tu) EnumeratorDecl(
                    enum_ty,
                    u32(index),
                    S.tu->save(enumerator->getName()),
                    ImportSourceLocation(enumerator->getLocation())
                );

                decl->value = S.tu->store_int(enumerator->getValue());
                S.AddDeclToScope(scope, decl);
                S.imported_decls[GetStableCanonicalDecl(enumerator)] = decl;
            }

            enum_ty->finalise();
            return enum_ty->decl();
        } break;

        case K::EnumConstant: {
            TRY(ImportDecl(cast<clang::EnumConstantDecl>(d)->getType()->getAsEnumDecl()));
            if (auto decl = S.imported_decls[d].get_or_null()) return decl;
            return AlreadyDiagnosed();
        }

        case K::Function:
            return ImportFunctionImpl(cast<clang::FunctionDecl>(d));

        case K::Namespace:
            // TODO
            break;

        case K::Record:
        case K::CXXRecord:
            return ImportRecordImpl(cast<clang::RecordDecl>(d));

        // There are 3 patterns that we want to handle here.
        //
        //  1. This is a typedef of some tag type whose name is the same as
        //     that of the typedef (e.g. 'typedef struct foo { ... } foo').
        //     Ignore the typedef and just import the underlying tag.
        //
        //  2. As 1, but the tag is anonymous (e.g. 'typedef struct { ... } foo');
        //     Import the tag and set its name to that of the typedef.
        //
        //  3. If the typedef is 'size_t' (and it's _actually_ size_t and not
        //     just called that), map it to 'int'.
        //
        //  4. None of the above; import the underlying type and create a type alias.
        case K::Typedef:
        case K::TypeAlias: {
            auto td = cast<clang::TypedefNameDecl>(d);

            // Case 2.
            //
            // Using the typedef name as the tag name is handled when we
            // import the tag, so no need to deal w/ that here.
            if (auto anon = td->getAnonDeclWithTypedefName(true))
                return TRY(ImportDecl(anon));

            // Case 1.
            //
            // Use 'dyn_cast' rather than 'getAs' because we don't want to
            // discard type sugar here.
            auto underlying = td->getUnderlyingType();
            if (
                auto tt = dyn_cast<clang::TagType>(underlying);
                tt and tt->getDecl()->getDeclName() == td->getDeclName()
            ) return ImportDecl(tt->getDecl());

            // Case 3.
            auto loc = ImportSourceLocation(td->getLocation());
            if (
                td->getName() == "size_t" and
                underlying.getCanonicalType() == AST().getSizeType().getCanonicalType()
            ) return AliasType::Create(*S.tu, Type::IntTy, String("size_t"), loc)->decl();

            // Case 4.
            //
            // Type aliases may be recursive (in that they may refer to a
            // struct that contains the a type alias); create the alias first
            // and fill in the type later.
            auto alias = AliasType::Create(*S.tu, Type::VoidTy, S.tu->save(td->getName()), loc);
            S.imported_decls[td] = alias->decl();
            alias->set_aliased_type(TRY(ImportType(loc, underlying)));
            return alias->decl();
        }

        case K::Var: {
            auto var = cast<clang::VarDecl>(d);
            if (var->isLocalVarDecl() or var->getKind() != clang::Decl::Kind::Var)
                return MakeErr(var, "Cannot import this variable");

            if (var->getFormalLinkage() != clang::Linkage::External)
                return MakeErr(var, "Cannot import variable with non-external linkage");

            if (var->getTSCSpec() != clang::TSCS_unspecified)
                return MakeErr(var, "Cannot import thread-local variable");

            auto name = S.tu->save(var->getName());
            auto loc = ImportSourceLocation(var->getLocation());
            auto ty = TRY(ImportType(loc, var->getType()));
            return new (*S.tu) GlobalDecl(
                S.tu.get(),
                clang_module,
                ty,
                var->getType().isConstQualified(),
                name,
                Linkage::Imported,
                Mangling::CXX,
                loc
            );
        }
    }

    return NYI(d, "declaration of kind {}", d->getDeclKindName());
}

auto srcc::CXXImporter::ImportFunctionImpl(clang::FunctionDecl* d) -> Res<ProcDecl*> {
    DebugAssert(
        S.imported_decls.at(d).invalid() and d == GetStableCanonicalDecl(d),
        "Call ImportDecl() instead of ImportFunctionImpl()"
    );

    if (isa<clang::CXXMethodDecl>(d)) return NYI(d, "member functions");

    // If the return type hasn’t been deduced yet, we can’t import it.
    if (d->getReturnType()->getAs<clang::AutoType>())
        return NYI(d, "function with deduced return type");

    // Don’t import immediate or inline functions for now.
    // TODO: Constant-evaluation of C++ functions.
    if (d->isImmediateFunction())
        return NYI(d, "'%1(consteval%)' function");

    // Don’t import functions with internal linkage, or anything
    // attached to a module.
    if (d->getLinkageInternal() != clang::Linkage::External)
        return NYI(d, "'%1(static%)' function");
    if (d->getOwningModule())
        return NYI(d, "function that is part of a C++20 module");

    // Do not import language builtins.
    if (auto builtin = d->getBuiltinID()) {
        // Note: Clang treats C standard library functions (e.g. 'puts') as
        // builtins as well, but those count as ‘library builtins’.
        if (not AST().BuiltinInfo.isPredefinedLibFunction(builtin))
            return NYI(d, "builtin function");
    }


    // Check if this is a constructor wrapper; these procedures act like
    // rvalues and have an extra leading 'void*' parameter that is not
    // reflected in the signature.
    ProcType* ty;
    auto constructor_wrapper = GetAnnotation(d, ConstructorWrapperAnnotation);
    if (constructor_wrapper) {
        // FIXME: Pass a reflection of the constructed type once Clang supports
        // reflecting arbitrary types. Until then, we use type_identity as a hack.
        QualType class_ty;
        auto identity = constructor_wrapper->args_begin()[1];
        Assert(S.tu->cxx_compilation()->getSema().isStdTypeIdentity(identity->getType(), &class_ty));
        auto result = TRY(ImportType(d->getLocation(), class_ty));

        // Build the procedure type.
        SmallVector<ParamTypeData> params;
        for (auto param : d->parameters().drop_front()) params.emplace_back(
            Intent::ByValue,
            TRY(ImportType(param->getLocation(), param->getType()))
        );

        ty = ProcType::Get(
            *S.tu,
            {result, Expr::RValue},
            params,
            CallingConvention::CXXConstructorWrapper
        );
    } else {
        ty = cast<ProcType>(TRY(ImportType(d->getLocation(), QualType(d->getFunctionType(), 0))));
    }

    // Reject immovable types for native value returns; constructor wrappers
    // instead initialise the result directly in its final storage.
    if (not constructor_wrapper and not ty->returns_lvalue() and not ty->ret()->can_move())
        return NYI(d, "returning an immovable class by value");

    for (auto param : ty->params())
        if (not param.type->can_move())
            return NYI(d, "passing an immovable class by value");

    // Create the procedure.
    auto proc = ProcDecl::Create(
        *S.tu,
        clang_module,
        ty,
        S.tu->save(d->getNameAsString()),
        Linkage::Imported,
        d->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        InheritedProcedureProperties{.is_cxx_inline_function = d->isInlineSpecified() and d->hasBody()},
        ImportSourceLocation(d->getNameInfo().getBeginLoc())
    );

    // We may need to mangle this, so remember where it came from.
    proc->set_clang_decl(d);

    // In C++, parameter names may differ across declarations, so only specify
    // a parameter name if it is the same across all declarations (in which that
    // parameter has a name).
    SmallVector<Sema::ParamSpec> param_specs;
    for (auto i : llvm::seq(constructor_wrapper ? 1u : 0u, d->getNumParams())) {
        std::optional<clang::DeclarationName> name;
        clang::SourceLocation name_loc;
        for (auto redecl : d->redecls()) {
            auto clang_param = redecl->getParamDecl(u32(i));
            auto n = clang_param->getDeclName();

            // This parameter declaration has no name.
            if (n.isEmpty()) continue;

            // We don't have a name yet, use this name.
            if (not name.has_value()) {
                name = n;
                name_loc = clang_param->getLocation();
            }

            // We already have a name; if it is different from the current one,
            // clear it and give up.
            else if (name.value() != n) {
                name = std::nullopt;
                break;
            }
        }

        // If we have a name, add it to the type.
        DeclNameLoc param_name{
            String(),
            ImportSourceLocation(d->getParamDecl(u32(i))->getLocation())
        };

        if (name.has_value()) {
            param_name = {
                S.tu->save(name.value().getAsIdentifierInfo()->getName()),
                ImportSourceLocation(name_loc),
            };
        }

        // We don't ever to anything to the body of a C++ function, so we don't
        // care about 'with' or 'this' here or whether the parameters are immutable,
        // so just preserve the name.
        param_specs.emplace_back(param_name);
    }

    // Create param decls.
    auto scope = S.tu->create_scope<ProcScope>(nullptr, std::nullopt);
    proc->scope = scope;
    S.BuildParamDecls(proc, param_specs);
    return proc;
}

void srcc::CXXImporter::ImportCtorsAndDtor(StructType* type, clang::CXXRecordDecl* record) {
    auto& sema = S.tu->cxx_compilation()->getSema();
    sema.ForceDeclarationOfImplicitMembers(record);

    // If we don’t have a destructor, give up.
    auto destructor = record->getDestructor();
    if (
        not destructor or
        destructor->isDeleted() or
        destructor->getAccess() != clang::AS_public
    ) return;

    // Return a lambda that creates a constructor or destructor body.
    auto MakeBody = [&](Ptr<clang::CXXConstructorDecl> constructor) {
        // The importer may have gone out of scope by the time the body is
        // requested, so don’t capture by reference here.
        return [=, sema = &S, module = clang_module](ProcDecl* proc) -> Ptr<Expr> {
            CXXImporter importer{*sema, module};
            return importer.BuildCtorOrDtorBody(type, record, constructor, proc);
        };
    };

    // Build the deleter.
    auto loc = type->decl()->location();
    auto deleter = S.BuildImplicitProcedure(
        ProcType::Get(*S.tu, {Type::VoidTy, Expr::RValue}, {{Intent::Inout, type}}),
        Tk::Delete,
        {Sema::ParamSpec::This(loc)},
        Linkage::Merge,
        Mangling::Source,
        loc,
        MakeBody({}),
        true
    );

    deleter->parent = type->parent;
    deleter->props.associated_type = type;
    type->set_delete(deleter);

    // Build an initialiser for each constructor we can handle.
    bool has_empty = false;
    if (not record->isAbstract()) {
        outer: for (auto constructor : record->ctors()) {
            if (
                constructor->isDeleted() or
                constructor->getAccess() != clang::AS_public or
                constructor->isCopyOrMoveConstructor() or
                constructor->getDescribedFunctionTemplate() or
                constructor->isVariadic() or
                constructor->isInheritingConstructor() or
                constructor->isImmediateFunction()
            ) continue;

            // Build ctor params.
            SmallVector<ParamTypeData> params;
            SmallVector<Sema::ParamSpec> specs;
            for (auto param : constructor->parameters()) {
                auto imported = ImportType(param->getLocation(), param->getType());
                if (not imported or not (*imported)->can_move()) continue outer;
                params.emplace_back(Intent::Move, *imported);
                specs.push_back({
                    .name = {
                        S.tu->save(param->getName()),
                        ImportSourceLocation(param->getLocation())
                    },
                });
            }

            // Declare the procedure.
            auto proc = S.BuildImplicitProcedure(
                ProcType::Get(*S.tu, {type, Expr::RValue}, params),
                Tk::Init,
                specs,
                Linkage::Merge,
                Mangling::Source,
                loc,
                MakeBody(constructor),
                true
            );

            proc->parent = type->parent;
            proc->props.associated_type = type;
            S.AddDeclToScope(type->scope(), proc);
            has_empty |= params.empty();
        }
    }

    type->update_initialisation(*S.tu, not type->initialisers().empty(), has_empty);
}

auto srcc::CXXImporter::ImportName(clang::TagDecl* td) -> StringRef {
    if (auto tdef = td->getTypedefNameForAnonDecl()) return tdef->getName();
    if (td->getDeclName().isIdentifier()) return td->getName();
    return "";
}

auto srcc::CXXImporter::ImportRecordImpl(clang::RecordDecl* rd) -> Res<TypeDecl*> {
    DebugAssert(
        S.imported_decls.at(rd).invalid() and rd == GetStableCanonicalDecl(rd),
        "Call ImportDecl() instead of ImportRecordImpl()"
    );

    // Incomplete types are treated as opaque.
    if (not rd->isCompleteDefinition()) return OpaqueType::Create(
        *S.tu,
        S.tu->save(ImportName(rd)),
        ImportSourceLocation(rd->getLocation())
    )->decl();

    auto cxx = dyn_cast<clang::CXXRecordDecl>(rd);
    bool is_cxx_class = cxx and not cxx->isCLike();
    if (is_cxx_class and rd->getDeclName().isEmpty()) return NYI(rd, "anonymous class type");

    // Declare the type now because it may be recursive.
    auto scope = S.tu->create_scope<RecordScope>(S.global_scope());
    auto name = S.tu->save(ImportName(rd));
    auto type = StructType::Create(
        *S.tu,
        scope,
        Linkage::Imported,
        name,
        ImportSourceLocation(rd->getLocation())
    );
    type->imported_from_module = clang_module;
    S.imported_decls[rd] = type->decl();

    // Import the fields.
    auto& layout = AST().getASTRecordLayout(rd);
    SmallVector<FieldDecl*> fields;
    SmallVector<Size> offsets;
    for (auto [i, f] : enumerate(rd->fields())) {
        // Ignore non-public members and bitfields.
        if (is_cxx_class and (f->getAccess() != clang::AS_public or f->isBitField())) continue;

        // We currently don’t support unaligned fields and in-class initialisers.
        //
        // FIXME: I think overaligned fields should 'just work' since we're getting
        // field offsets etc. from Clang.
        if (not is_cxx_class and f->getMaxAlignment() != 0) return NYI(f, "unaligned or overaligned field");
        if (not is_cxx_class and f->hasInClassInitializer()) return NYI(f, "field with in-class initialiser");

        // Import the field type.
        auto field_type = ImportType(f->getLocation(), f->getType());
        if (not field_type) std::unexpected(std::move(field_type.error()));

        // Declare the field.
        auto decl = new (*S.tu) FieldDecl(
            type,
            *field_type,
            S.tu->save(f->getName()),
            utils::safe_cast<u32>(fields.size()),
            ImportSourceLocation(f->getLocation())
        );

        fields.push_back(decl);
        offsets.push_back(Size::Bits(layout.getFieldOffset(u32(i))));
        S.AddDeclToScope(scope, decl);
    }

    // Determine if this contains a pointer.
    bool contains_pointer = any_of(fields, [&](auto *fd) {
        return fd->type->is_or_contains_pointer();
    });

    // Build the layout.
    bool can_persist_as_bytes =
        not contains_pointer and
        not rd->isUnion() and
        AST().getPaddingIntervals(AST().getCanonicalTagType(rd)).empty() and
        not any_of(fields, [&](auto* field) { return not field->type->can_persist_as_bytes(*S.tu); });

    auto bits = RecordLayout::Bits::CLikeType(
        contains_pointer,
        rd->isUnion(),
        can_persist_as_bytes
    );

    if (is_cxx_class) {
        bits.can_copy = false;
        bits.can_move = false;
        bits.can_persist_as_bytes = false;
        bits.contains_pointer = true;
        bits.init_from_no_args = false;
        bits.literal_initialiser = false;
        bits.zero_init = false;
    }

    auto rl = RecordLayout::Get(
        *S.tu,
        offsets,
        Size::Bytes(layout.getSize().getQuantity()),
        Size::Bytes(layout.getSize().getQuantity()),
        Align(layout.getAlignment().getQuantity()),
        bits
    );

    type->finalise(rl);
    if (is_cxx_class) ImportCtorsAndDtor(type, cxx);
    return type->decl();
}

auto srcc::CXXImporter::ImportReturnType(
    SLoc loc,
    QualType ty
) -> Res<TypeAndValueCategory> {
    auto vc = Expr::RValue;

    if (ty->isReferenceType()) {
        ty = ty->getPointeeType();
        vc = Expr::LValue(ty.isConstQualified());
    }

    return TypeAndValueCategory{TRY(ImportType(loc, ty)), vc};
}

auto srcc::CXXImporter::ImportType(clang::SourceLocation sloc, QualType ty) -> Res<Type> {
    return ImportType(ImportSourceLocation(sloc), ty);
}

auto srcc::CXXImporter::ImportType(SLoc sloc, QualType ty) -> Res<Type> {
    switch (ty->getTypeClass()) {
        using K = clang::Type::TypeClass;
        default: break;
        case K::Builtin: {
            switch (cast<clang::BuiltinType>(ty)->getKind()) {
                using K = clang::BuiltinType::Kind;
                default: break;
                case K::Void: return Type::VoidTy;
                case K::Bool: return Type::BoolTy;

                case K::SChar:
                case K::UChar:
                case K::Char_S:
                case K::Char_U:
                    return S.tu->FFICharTy;

                case K::WChar_S:
                case K::WChar_U:
                    return S.tu->FFIWCharTy;

                case K::Short:
                case K::UShort:
                    return S.tu->FFIShortTy;

                case K::Int:
                case K::UInt:
                    return S.tu->FFIIntTy;

                case K::Long:
                case K::ULong:
                    return S.tu->FFILongTy;

                case K::LongLong:
                case K::ULongLong:
                    return S.tu->FFILongLongTy;

                case K::Int128:
                case K::UInt128:
                    return S.tu->I128Ty;
            }
        } break;

        case K::LValueReference:
        case K::RValueReference: {
            auto elem = TRY(ImportType(sloc, ty->getPointeeType()));
            return PtrType::Get(*S.tu, elem, ty->getPointeeType().isConstQualified());
        }

        // C++ pointers are nullable, so wrap them in an optional.
        case K::Pointer: {
            auto elem = TRY(ImportType(sloc, ty->getPointeeType()));
            auto ptr = PtrType::Get(*S.tu, elem, ty->getPointeeType().isConstQualified());
            return OptionalType::Get(*S.tu, ptr);
        }

        case K::BitInt: {
            auto b = cast<clang::BitIntType>(ty);
            return IntType::Get(*S.tu, Size::Bits(b->getNumBits()));
        }

        case K::ConstantArray: {
            auto c = cast<clang::ConstantArrayType>(ty);
            auto elem = TRY(ImportType(sloc, c->getElementType()));
            return ArrayType::Get(*S.tu, elem, i64(c->getSize().getZExtValue()));
        }

        case K::Enum:
            return cast<TypeDecl>(TRY(ImportDecl(ty->getAsEnumDecl())))->type;

        case K::FunctionProto: {
            auto fpt = cast<clang::FunctionProtoType>(ty);
            if (fpt->getCallConv() != clang::CallingConv::CC_C) {
                return NYI(
                    sloc,
                    "function with '{}' calling convention",
                    clang::FunctionType::getNameForCallConv(fpt->getCallConv())
                );
            }

            auto ret = fpt->getExtInfo().getNoReturn()
                ? TypeAndValueCategory{Type::NoReturnTy, Expr::RValue}
                : TRY(ImportReturnType(sloc, fpt->getReturnType()));

            SmallVector<ParamTypeData, 6> params;
            for (auto param : fpt->param_types()) {
                auto param_ty = TRY(ImportType(sloc, param));
                params.emplace_back(Intent::ByValue, param_ty);
            }

            return ProcType::Get(
                *S.tu,
                ret,
                params,
                CallingConvention::Native,
                fpt->isVariadic()
            );
        }

        case K::Record: {
            auto rd = ty->castAsRecordDecl();
            return cast<TypeDecl>(TRY(ImportDecl(rd)))->type;
        }

        case K::Typedef: {
            auto td = cast<clang::TypedefType>(ty);
            return cast<TypeDecl>(TRY(ImportDecl(td->getDecl())))->type;
        }

        case K::Using: {
            auto td = cast<clang::UsingType>(ty);
            return cast<TypeDecl>(TRY(ImportDecl(td->getDecl()->getTargetDecl())))->type;
        }
    }

    if (ty != ty.getCanonicalType())
        return ImportType(sloc, ty.getCanonicalType());

    return NYI(sloc, "type '{}'", ty);
}

auto srcc::CXXImporter::ImportSourceLocation(clang::SourceLocation sloc) -> SLoc {
    if (not sloc.isValid()) return {};
    auto& sm = AST().getSourceManager();
    auto name = sm.getFilename(sloc);
    auto f = S.ctx.try_get_file(name.str());
    if (not f.has_value()) {
        Warn(SLoc(), "Could not import file for source location: {}", name);
        return {};
    }
    return SLoc(f.value()->data() + sm.getFileOffset(sloc));
}

auto srcc::CXXImporter::ImportValue(
    SLoc loc,
    const clang::APValue& val,
    Type ty
) -> Res<eval::RValue> {
    if (not ty->can_move()) return NYI(loc, "constant value of immovable type '{}'", ty);
    switch (val.getKind()) {
        case clang::APValue::Int: {
            Assert(ty->is_integer_or_bool());
            return eval::RValue(val.getInt(), ty);
        }

        case clang::APValue::Struct: {
            // Type must be a struct type.
            auto struct_ty = cast<RecordType>(ty);

            // Field count must match what we expect.
            u32 num_fields = val.getStructNumFields();
            Assert(struct_ty->layout().field_offsets().size() == num_fields);

            // Import each field.
            SmallVector<eval::RValue*> fields;
            for (auto [i, f] : vws::enumerate(struct_ty->scope()->fields())) {
                auto field = TRY(ImportValue(
                    loc,
                    val.getStructField(u32(i)),
                    f->type
                ));

                fields.push_back(S.tu->save(field));
            }

            return eval::RValue(
                eval::Record(ArrayRef(fields).copy(S.tu->allocator())),
                ty
            );
        }

        default: break;
    }

    return MakeErr(
        loc,
        "Unsupported {} APValue: {}",
        enchantum::to_string(val.getKind()),
        val
    );
}

CXXCompilation::~CXXCompilation() = default;

bool CXXCompilation::finish() {
    if (finished) return not getDiagnostics().hasErrorOccurred();
    getSema().ActOnEndOfTranslationUnit();
    getASTConsumer().HandleTranslationUnit(getASTContext());
    finished = true;
    return not getDiagnostics().hasErrorOccurred();
}

bool CXXCompilation::initialise(llvm::LLVMContext& context) {
    struct Action : clang::SyntaxOnlyAction {
        llvm::LLVMContext& context;
        clang::CodeGenerator*& generator;

        Action(llvm::LLVMContext& context, clang::CodeGenerator*& generator)
            : context{context}, generator{generator} {}

        auto CreateASTConsumer(
            clang::CompilerInstance& instance,
            llvm::StringRef name
        ) -> std::unique_ptr<clang::ASTConsumer> override {
            auto codegen = clang::CreateLLVMCodeGen(instance, name, context);
            generator = codegen.get();

            // Consumer that filters out things we don’t want to emit.
            struct Consumer : clang::MultiplexConsumer {
                explicit Consumer(std::unique_ptr<clang::ASTConsumer> codegen)
                    : MultiplexConsumer(std::move(codegen)) {}

                bool HandleTopLevelDecl(clang::DeclGroupRef group) override {
                    for (auto decl : group) {
                        // Ignore expanded macro declarations. They exist only so we can
                        // attempt to constant-fold their initialisers.
                        if (auto var = llvm::dyn_cast<clang::VarDecl>(decl);
                            var and var->getName().starts_with("__srcc_expanded_macro_"))
                            continue;

                        if (not MultiplexConsumer::HandleTopLevelDecl(clang::DeclGroupRef(decl)))
                            return false;
                    }
                    return true;
                }
            };

            return std::make_unique<Consumer>(std::move(codegen));
        }
    };

    Action action{context, code_generator};

    // Set up the parser.
    if (not createTarget()) return false;

    // Add a dummy main file.
    auto& input = getFrontendOpts().Inputs.front();
    input = clang::FrontendInputFile(
        llvm::MemoryBufferRef(";", "__srcc.imports.cc"),
        input.getKind()
    );

    // Set up the parser.
    if (not action.BeginSourceFile(*this, input)) return false;
    createSema(action.getTranslationUnitKind(), nullptr);
    auto& pp = getPreprocessor();
    pp.enableIncrementalProcessing();
    parser = std::make_unique<clang::Parser>(pp, getSema(), false);
    pp.EnterMainSourceFile();
    parser->Initialize();

    // Start the TU.
    if (auto external = getASTContext().getExternalSource())
        external->StartTranslationUnit(&getASTConsumer());
    getSema().ActOnStartOfTranslationUnit();
    mangle_context.reset(getASTContext().createMangleContext());

    // Parse until EOF.
    return ParseDeclarations();
}

bool CXXCompilation::parse(llvm::StringRef code) {
    Assert(not finished);
    if (getDiagnostics().hasUncompilableErrorOccurred()) return false;

    // Throw the code into a buffer and create a file for it.
    auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
        code,
        std::format("__srcc.glue.{}.cc", buffers++)
    );

    auto id = getSourceManager().createFileID(std::move(buffer));

    // Enter the source file.
    if (getPreprocessor().EnterSourceFile(id, nullptr, {})) return false;

    // Yeet the EOF token left over from the previous parsing run.
    Assert(parser->getCurToken().is(clang::tok::eof));
    parser->ConsumeAnyToken();

    // Parse until EOF.
    if (not ParseDeclarations()) return false;

    // Mark that we’re done with a part of the TU.
    getSema().ActOnEndOfTranslationUnitFragment(clang::TUFragmentKind::Normal);
    return not getDiagnostics().hasErrorOccurred();
}

bool CXXCompilation::ParseDeclarations() {
    auto state = clang::Sema::ModuleImportState::NotACXX20Module;

    // Parse until EOF, but leave EOF unparsed. Passing it to the parser
    // would run ActOnEndOfTranslationUnit() and prematurely finish this TU.
    while (parser->getCurToken().isNot(clang::tok::eof)) {
        clang::Parser::DeclGroupPtrTy declarations;
        parser->ParseTopLevelDecl(declarations, state);
        if (
            declarations and
            not getASTConsumer().HandleTopLevelDecl(declarations.get())
        ) return false;
    }

    return not getDiagnostics().hasErrorOccurred();
}

bool Sema::ImportCXXHeaders(ArrayRef<const ParsedModule::Import*> imports) {
    if (imports.empty()) return true;
    llvm::TimeTraceScope _{"[Clang] C++ Parsing"};

    SmallString<128> code;
    for (auto import : imports) {
        auto lc = import->loc.seek_line_column(context());
        auto file = import->loc.file_name(context());
        for (auto header : import->linkage_names) {
            Assert(header.starts_with("<") and header.ends_with(">"));
            if (lc and file) Format(code, "#line {} \"{}\"\n", lc->line, *file);
            Format(code, "#include \"{}\"\n", header.drop().drop_back());
        }
    }

    std::vector<std::string> args{
        SOURCE_CLANG_EXE,

        // We’ll handle codegen ourselves.
        "-fsyntax-only",

        // Use stdin as a dummy input file; we’ll supply a proper one later.
        "-x",
        "c++",
        "-",

        // Pass through whatever target we’re compiling for.
        "-target",
        tu->target().triple().getTriple(),

        // Use the latest standard.
        "-std=c++2c",

        // Diagnostic options.
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",

        // Forward our use_colours setting.
        std::format("-fdiagnostics-color={}", tu->context().use_colours ? "always" : "never"),

        // Make sure we always emit declarations if requested, even 'inline' ones.
        "-femit-all-decls",
    };

    // Add opt level.
    int clang_opt_level = 0;
    if (ctx.opt_level == 4) {
        args.push_back("-march=native");
        clang_opt_level = 3;
    } else {
        clang_opt_level = ctx.opt_level;
    }

    args.push_back(std::format("-O{}", clang_opt_level));

    // Add include paths.
    for (const auto& p : clang_include_paths) {
        args.push_back("-I");
        args.push_back(p);
    }

    // Add user-specified options.
    append_range(args, clang_options);

    // Our persistent parser stops before EOF, where Clang normally installs
    // its late-template parser callback (in ParseTopLevelDecl()). Keep delayed
    // parsing disabled for now.
    args.push_back("-fno-delayed-template-parsing");

    // Parse Clang command-line arguments.
    SmallVector<const char*> argv;
    for (const auto& arg : args) argv.push_back(arg.c_str());
    auto invocation = clang::createInvocation(argv);
    if (not invocation) return Error(imports.front()->loc, "Header import failed");

    // Dump header search paths if requested.
    if (tu->lang_opts().dump_include_paths) {
        std::println(stderr, "Clang header search paths:");
        for (const auto& entry : invocation->getHeaderSearchOpts().UserEntries)
            std::println(stderr, "  - {}", entry.Path);
    }

    // Initialise the compiler instance.
    auto compilation = std::make_unique<CXXCompilation>(std::move(invocation));
    compilation->setVirtualFileSystem(llvm::vfs::getRealFileSystem());
    compilation->createFileManager();
    compilation->createDiagnostics();
    compilation->createSourceManager();
    if (
        not compilation->initialise(tu->llvm_context) or
        not compilation->parse(code)
    ) return Error(imports.front()->loc, "Header import failed");
    tu->set_cxx_compilation(std::move(compilation));

    // Create a module declaration for each 'import' clause.
    for (auto import : imports) {
        auto decl = ImportedClangModuleDecl::Create(
            *tu,
            import->import_name,
            import->linkage_names,
            import->loc
        );

        if (import->is_open_import) {
            tu->open_modules.push_back(decl);
        } else {
            auto& previous = tu->logical_imports[import->import_name];
            if (previous) {
                Error(import->loc, "Cannot redefine header import '{}'", import->import_name);
                Note(previous->location(), "Previous definition was here");
                continue;
            }
            previous = decl;
        }
    }

    return not ctx.diags().has_error();
}

namespace {
enum class Kind : u8 {
    Nothing = 0,
    Type = 1,
    Function = 2,
    Enumerator = 4,
    Var = 8,
};

// FIXME: This should be a macro in base.
enum class NegatedKind : u8 {};
constexpr NegatedKind operator~(Kind a) { return NegatedKind(~+a); }
constexpr Kind operator&(Kind a, NegatedKind b) { return Kind(+a & +b); }
constexpr Kind& operator&=(Kind& a, NegatedKind b) {
    a = a & b;
    return a;
}

constexpr bool operator&(Kind a, Kind b) { return (+a & +b) == +b; }
constexpr Kind operator|(Kind a, Kind b) { return Kind(+a | +b); }
constexpr  Kind& operator|=(Kind& a, Kind b) {
    a = a | b;
    return a;
}
}

auto Sema::LookUpCXXNameImpl(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclNameLoc> names,
    LookupHint hint
) -> LookupResult {
    auto ast = tu->cxx_compilation();
    auto& clang_sema = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = clang_sema.getPreprocessor();

    // Look up all scopes in the path.
    clang::DeclContext* ctx = ast_ctx.getTranslationUnitDecl();
    for (auto n : names.drop_back()) {
        auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(n.name.str())));
        if (not res.isSingleResult()) return LookupResult::NonScopeInPath(n);
        auto new_ctx = dyn_cast<clang::DeclContext>(res.front());
        if (not new_ctx) return LookupResult::NonScopeInPath(n);
        ctx = new_ctx;
    }

    // Look up the last segment in the scope.
    // TODO: Support operators.
    auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(names.back().name.str())));

    // Filters.
    #define FILTER(F, ...)
    static const auto EnumeratorFilter = [](clang::Decl* d) { return isa<clang::EnumConstantDecl>(d); };
    static const auto FunctionFilter = [](clang::Decl* d) { return isa<clang::FunctionDecl>(d); };
    static const auto VarFilter = [](clang::Decl* d) { return isa<clang::VarDecl>(d); };
    static const auto TypeFilter = [](clang::Decl* d) {
        return isa<clang::TypedefNameDecl, clang::EnumDecl, clang::RecordDecl>(d);
    };

    // Figure out what we found.
    auto found = Kind::Nothing;
    CXXImporter i{*this, clang_module};
    for (auto d : res) {
        if (d->isInvalidDecl()) continue;
        if (VarFilter(d)) found |= Kind::Var;
        else if (FunctionFilter(d)) found |= Kind::Function;
        else if (TypeFilter(d)) found |= Kind::Type;
        else if (EnumeratorFilter(d)) found |= Kind::Enumerator;
        else {
            auto diag = CreateNote(
                i.ImportSourceLocation(d->getLocation()),
                "Importing a '{}Decl' is not supported (yet)",
                d->getDeclKindName()
            );

            return LookupResult::FailedToImport(
                names.back(),
                std::make_unique<Diagnostic>(std::move(diag))
            );
        }
    }

    // And import the declarations we care about.
    SmallVector<Decl*> converted;
    auto ImportFiltered = [&](auto filter) {
        for (auto d : res) {
            if (d->isInvalidDecl() or not std::invoke(filter, d)) continue;
            auto decl = i.ImportDecl(d);
            if (not decl.has_value()) return LookupResult::FailedToImport(names.back(), std::move(decl.error()));
            if (not is_contained(converted, *decl)) converted.push_back(*decl);
        }

        if (converted.empty()) return LookupResult::NotFound(names.back());
        if (converted.size() == 1) return LookupResult::Success(converted.front());
        return LookupResult::Ambiguous(names.back(), converted);
    };

    // If we want a type, and we found a type, keep only the types.
    if (hint == LookupHint::Type and found & Kind::Type)
        found = Kind::Type;

    // If we found multiple things, remove any types.
    if (hint != LookupHint::Type and found != Kind::Type)
        found &= ~Kind::Type;

    // If we still have multiple things left, then that's an error.
    if (std::popcount(+found) > 1) {
        auto diag = CreateNote(names.back().loc, "There are multiple incompatible declarations");
        return LookupResult::FailedToImport(
            names.back(),
            std::make_unique<Diagnostic>(std::move(diag))
        );
    }

    switch (found) {
        case Kind::Nothing: return LookupResult::NotFound(names.back());
        case Kind::Function: return ImportFiltered(FunctionFilter);
        case Kind::Enumerator: return ImportFiltered(EnumeratorFilter);
        case Kind::Type: return ImportFiltered(TypeFilter);
        case Kind::Var: return ImportFiltered(VarFilter);
    }

    Unreachable();
}

/// Import an object-like macro: resolve a single identifier directly, otherwise
/// parse a constexpr wrapper and an unevaluated-at-runtime probe variable into
/// the shared Clang TU. Constant results become ConstExprs; other results call
/// the wrapper at runtime. Cache both successes and failures by MacroInfo.
auto Sema::LookUpCXXMacro(
    ImportedClangModuleDecl* clang_module,
    clang::MacroInfo* mi,
    DeclNameLoc macro_name,
    LookupHint hint
) -> LookupResult {
    // Import the source location of the macro.
    CXXImporter importer{*this, clang_module};
    SLoc macro_loc = importer.ImportSourceLocation(mi->getDefinitionLoc());

    // Return an error.
    auto Err = [&]<typename ...Args>(std::format_string<Args...> fmt, Args&& ...args) {
        return LookupResult::FailedToImport(
            macro_name,
            std::make_unique<Diagnostic>(CreateNote(macro_loc, fmt, std::forward<Args>(args)...))
        );
    };

    // Refuse to import ‘builtin’ macros (i.e. __LINE__, __COUNTER__, and friends)
    // as well as function-like macros.
    if (mi->isBuiltinMacro())
        return Err("Cannot import builtin macro '{}'", macro_name);
    if (mi->isFunctionLike())
        return Err("Cannot import function-like macro '{}'", macro_name);

    // If we have attempted to find this before, do not do so again.
    if (auto it = imported_macros.find(mi); it != imported_macros.end()) {
        if (it->second.present()) return LookupResult::Success(it->second.get());
        return LookupResult::FailedToImport(macro_name);
    }

    // Cache that we attempted to import this.
    imported_macros[mi] = nullptr;

    // Enter the macro definition as a new file.
    auto& compilation = *tu->cxx_compilation().get();
    auto& ast = compilation.getASTContext();
    auto& pp = compilation.getPreprocessor();
    auto& sm = compilation.getSourceManager();
    auto clang_sloc = sm.getLocForStartOfFile(sm.getMainFileID());
    auto fid = sm.createFileID(
        llvm::MemoryBuffer::getMemBufferCopy(
            macro_name.name.str(),
            "<macro expansion>"
        )
    );

    if (pp.EnterSourceFile(fid, nullptr, clang_sloc))
        return Err("Failed to enter source file");

    // Lex the start of the macro.
    clang::Token first, token;
    pp.Lex(first);
    if (first.is(clang::tok::eof)) return Err("Macro has empty expansion");

    // Throw out all other tokens in the macro so the parser doesn’t
    // try to parse them later on
    bool single_identifier = first.is(clang::tok::identifier);
    bool unknown = first.is(clang::tok::unknown);
    for (;;) {
        pp.Lex(token);
        if (token.is(clang::tok::eof)) break;
        single_identifier = false;
        unknown |= token.is(clang::tok::unknown);
    }

    if (unknown) return Err("Unknown token in macro expansion");

    // This is a single identifier, just perform name lookup for it.
    if (single_identifier) return LookUpCXXNameImpl(
        clang_module,
        DeclNameLoc{tu->save(first.getIdentifierInfo()->getName()), macro_name.loc},
        hint
    );

    // Wrap the macro in a function that returns it and also create a variable initialised
    // with a call to it; the latter is used to try and constant-evaluate the function.
    //
    // Clang gets mad if we combine 'extern "C"' and 'decltype(auto)', so use '__asm__'
    // instead to set the function name; this however requires separating the definition and
    // declaration for some reason...
    //
    // The 'always_inline' is required, else inlining just doesn't happen at all (possibly
    // because the function has 'linkonce_odr' linkage).
    auto name_base = Format("__srcc_expanded_macro_{}", generated_cxx_macro_decls++);
    auto proc_name = Format("{}_init", name_base);
    auto var_name = Format("{}_var", name_base);
    auto code = Format(
        "#line 1 \"<macro-expansion>\"\n"
        "[[_Clang::__always_inline__]] constexpr decltype(auto) {0}() __asm__(\"{0}\");\n"
        "constexpr decltype(auto) {0}() {{ return {1}; }}\n"
        "decltype(auto) {2} = {0}();\n",
        proc_name,
        macro_name,
        var_name
    );

    if (not compilation.parse(code))
        // No diagnostic here; Clang already told the user what was wrong.
        // TODO: Introduce a custom Clang diags consumer (or at least set a
        // custom ostream).
        return LookupResult::FailedToImport(macro_name);

    // Retrieve a declaration.
    auto GetDecl = [&](StringRef name) -> clang::Decl* {
        clang::DeclarationName ident{pp.getIdentifierInfo(name)};
        auto res = ast.getTranslationUnitDecl()->lookup(ident);
        Assert(res.isSingleResult() and not res.front()->isInvalidDecl());
        return res.front();
    };

    // Retrieve the variable.
    auto var = cast<clang::VarDecl>(GetDecl(var_name));

    // Fold the macro expansion if possible.
    clang::Expr::EvalResult er;
    SmallVector<clang::PartialDiagnosticAt> diags;
    er.Diag = &diags;
    Expr* value = nullptr;

    // Folding succeeded.
    if (
        var->getInit()->EvaluateAsInitializer(
            ast,
            var,
            er,
            false
        ) and diags.empty()
    )  {
        // Import the type.
        auto ty = importer.ImportType(macro_loc, var->getType());
        if (not ty.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(ty.error())
        );

        // Import the value.
        auto v = importer.ImportValue(macro_loc, *var->getEvaluatedValue(), *ty);
        if (not v.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(v.error())
        );

        value = MakeConstExpr(nullptr, std::move(*v), macro_loc);
    }

    // Folding failed; emit the wrapper and create a 'declaration' that runs
    // the macro when it is referenced.
    else {
        // Retrieve the procedure.
        auto clang_proc = cast<clang::FunctionDecl>(GetDecl(proc_name));

        // Import its return type.
        auto clang_ret_ty = clang_proc->getType()->castAs<clang::FunctionProtoType>()->getReturnType();
        auto ret_ty = importer.ImportReturnType(macro_loc, clang_ret_ty);
        if (not ret_ty.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(ret_ty.error())
        );

        // Mark the procedure as used so it is actually emitted.
        clang_proc->setIsUsed();
        compilation.codegen().HandleTopLevelDecl(clang::DeclGroupRef(clang_proc));
        auto proc = ProcDecl::Create(
            *tu,
            clang_module,
            ProcType::Get(*tu, *ret_ty),
            tu->save(proc_name),
            Linkage::Imported,
            Mangling::None,
            nullptr,
            InheritedProcedureProperties{.always_inline = true},
            macro_loc
        );

        // Create a call to it. This should never fail.
        value = BuildCallExpr(
            RequestReference(proc, macro_loc).get(),
            {},
            macro_loc
        ).get();
    }

    auto md = new (*tu) CXXMacroExpansionDecl("", value, value->location());
    imported_macros[mi] = md;
    return LookupResult::Success(md);
}

auto Sema::LookUpCXXName(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclNameLoc> names,
    LookupHint hint
) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto DoNameLookup = [&] {
        return LookUpCXXNameImpl(clang_module, names, hint);
    };

    // If this is anything other than a single identifier, it can’t be a macro.
    if (names.size() != 1 or not names.front().name.is_str())
        return DoNameLookup();

    // Check if this is a macro.
    auto ast = tu->cxx_compilation();
    auto& pp = ast->getSema().getPreprocessor();
    auto id = pp.getIdentifierInfo(names.front().name.str());
    auto *mi = pp.getMacroInfo(id);
    if (not mi) return DoNameLookup();
    return LookUpCXXMacro(clang_module, mi, names.front(), hint);
}
