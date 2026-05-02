#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/ASTImporter.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/DeclarationName.h>
#include <clang/AST/RecordLayout.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/MacroArgs.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Parse/Parser.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Tooling/Tooling.h>

#include <base/Assert.hh>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>

using namespace srcc;

#define TRY(v) ({ auto _v = v; if (not _v) return utils::Falsy(); *_v; })

class Sema::Importer {
    Sema& S;
    ImportedClangModuleDecl* clang_module;

public:
    explicit Importer(Sema& S, ImportedClangModuleDecl* clang_module) : S(S), clang_module(clang_module) {}
    auto AST() -> clang::ASTContext& { return clang_module->clang_ast.getASTContext(); }

    auto FormatTypeFailure(DeclName name, clang::QualType ty) -> LookupResult;

    auto ImportDecl(clang::Decl* D) -> Ptr<Decl>;
    auto ImportDeclImpl(clang::Decl* D) -> Ptr<Decl>;
    auto ImportRecordImpl(clang::RecordDecl* RD) -> std::optional<Type>;
    auto ImportFunctionImpl(clang::FunctionDecl* D) -> Ptr<ProcDecl>;
    auto ImportName(clang::TagDecl* D) -> StringRef;
    auto ImportReturnType(clang::QualType T) -> std::optional<TypeAndValueCategory>;
    auto ImportType(const clang::Type* T) -> std::optional<Type>;
    auto ImportType(clang::QualType T) { return ImportType(T.getTypePtr()); }
    auto ImportSourceLocation(clang::SourceLocation sloc) -> SLoc;
    auto ImportValue(const clang::APValue& val, Type ty) -> std::optional<eval::RValue>;
};

auto Sema::Importer::FormatTypeFailure(DeclName name, clang::QualType ty) -> LookupResult {
    SmallString<128> str;
    llvm::raw_svector_ostream os{str};
    ty.print(os, clang::PrintingPolicy(clang_module->clang_ast.getLangOpts()));
    auto diag = CreateNote(SLoc(), "Unsupported Clang type: {}", str);
    return LookupResult::FailedToImport(
        name,
        std::make_unique<Diagnostic>(std::move(diag))
    );
}

auto Sema::Importer::ImportDecl(clang::Decl* D) -> Ptr<Decl> {
    D = D->getCanonicalDecl();

    // If we have attempted to find this before, do not do so again.
    if (auto it = S.imported_decls.find(D); it != S.imported_decls.end())
        return it->second;

    // Otherwise, try to create it now.
    //
    // Don’t hold on to a reference to 'S.imported_decls[D]' as importing a decl
    // might import more decls, which might invalidate it.
    S.imported_decls[D] = nullptr; // Mark that we’ve already tried doing this.
    auto imported = ImportDeclImpl(D);
    if (imported) {
        S.imported_decls[D] = imported;
        return imported;
    }

    if (auto n = dyn_cast<clang::NamedDecl>(D)) {
        S.Error(
            ImportSourceLocation(n->getLocation()),
            "Importing declaration of '{}' from C++ is not supported",
            n->getQualifiedNameAsString()
        );
    } else {
        S.Error(
            ImportSourceLocation(D->getBeginLoc()),
            "Importing this C++ declaration is not supported"
        );
    }

    return {};
}

auto Sema::Importer::ImportDeclImpl(clang::Decl* D) -> Ptr<Decl> {
    // Ignore invalid ones.
    if (D->isInvalidDecl()) return {};
    switch (D->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::Enum: {
            auto e = cast<clang::EnumDecl>(D);
            for (auto enumerator : e->enumerators())
                S.imported_decls[enumerator->getCanonicalDecl()] = nullptr;

            auto underlying = TRY(ImportType(e->getIntegerType()));
            auto loc = ImportSourceLocation(e->getLocation());
            auto name = ImportName(e);
            auto scope = S.tu->create_scope(S.global_scope());
            auto enum_ty = new (*S.tu) EnumType(*S.tu, scope, S.tu->save(name), underlying, loc);
            for (auto enumerator : e->enumerators()) {
                auto decl = new (*S.tu) EnumeratorDecl(
                    enum_ty,
                    S.tu->save(enumerator->getName()),
                    ImportSourceLocation(enumerator->getLocation())
                );

                decl->value = S.tu->store_int(enumerator->getValue());
                S.AddDeclToScope(scope, decl);
                S.imported_decls[enumerator->getCanonicalDecl()] = decl;
            }

            enum_ty->finalise();
            return enum_ty->decl();
        } break;

        case K::EnumConstant: {
            TRY(ImportDecl(cast<clang::EnumConstantDecl>(D)->getType()->getAsEnumDecl()));
            return S.imported_decls[D];
        }

        case K::Function:
            return ImportFunctionImpl(cast<clang::FunctionDecl>(D));

        case K::Namespace:
            // TODO
            break;

        case K::Record:
        case K::CXXRecord:
            return cast<StructType>(TRY(ImportRecordImpl(cast<clang::RecordDecl>(D))))->decl();

        case K::Typedef: {
            auto td = cast<clang::TypedefDecl>(D);
            auto clang_ty = AST().getTypedefType(
                clang::ElaboratedTypeKeyword::None,
                std::nullopt,
                td
            );

            if (clang_ty->isRecordType()) {
                auto ty = TRY(ImportType(clang_ty));
                return cast<StructType>(ty)->decl();
            }
        } break;

        case K::Using:
            // TODO
            break;

        case K::Var:
            // TODO
            break;
    }

    return nullptr;
}

auto Sema::Importer::ImportFunctionImpl(clang::FunctionDecl* D) -> Ptr<ProcDecl> {
    D = D->getDefinition() ?: D->getFirstDecl();
    if (isa<clang::CXXMethodDecl>(D)) return {};
    auto FPT = D->getType()->getAs<clang::FunctionProtoType>();
    Assert(FPT, "No prototype in C++?");

    // If the return type hasn’t been deduced yet, we can’t import it.
    if (D->getReturnType()->getAs<clang::AutoType>()) return {};

    // Don’t import immediate or inline functions for now.
    if (D->isImmediateFunction() or D->isInlineSpecified()) return {};

    // Don’t import functions with internal linkage, or anything
    // attached to a module.
    if (D->getLinkageInternal() != clang::Linkage::External) return {};
    if (D->getOwningModule()) return {};

    // Do not import language builtins.
    if (auto ID = D->getBuiltinID()) {
        // Note: Clang treats C standard library functions (e.g. 'puts') as
        // builtins as well, but those count as ‘library builtins’.
        if (not AST().BuiltinInfo.isPredefinedLibFunction(ID))
            return {};
    }

    // Import the type.
    auto T = ImportType(FPT);
    if (not T) return {};
    if (not isa<ProcType>(*T)) return {};

    // Create the procedure.
    auto PD = ProcDecl::Create(
        *S.tu,
        clang_module,
        cast<ProcType>(T.value().ptr()),
        S.tu->save(D->getNameAsString()),
        Linkage::Imported,
        D->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        InheritedProcedureProperties(),
        ImportSourceLocation(D->getNameInfo().getBeginLoc())
    );

    // Create param decls.
    SmallVector<LocalDecl*> Params;
    for (auto [I, P] : enumerate(D->parameters())) {
        Params.push_back(new (*S.tu) ParamDecl(
            &PD->param_types()[I],
            Expr::MLValue, // We pass by value so this is irrelevant.
            S.tu->save(P->getName()),
            PD,
            u32(I),
            false,
            ImportSourceLocation(P->getLocation())
        ));
    }

    PD->finalise(nullptr, Params);
    return PD;
}

auto Sema::Importer::ImportName(clang::TagDecl* td) -> StringRef {
    if (not td->hasNameForLinkage()) return "";
    if (td->getDeclName().isIdentifier()) return td->getName();
    if (auto tdef = td->getTypedefNameForAnonDecl()) return tdef->getName();
    return "";
}

auto Sema::Importer::ImportRecordImpl(clang::RecordDecl* RD) -> std::optional<Type> {
    Assert(RD);
    RD = cast<clang::RecordDecl>(RD->getDefinition() ?: RD->getFirstDecl());
    auto it = S.imported_records.find(RD);
    if (it != S.imported_records.end()) return it->second;

    // Create the cache entry now so we fail fast next time if we can’t import
    // this; don’t hold on to a reference to the cache entry here since we’re
    // about to import more types, which might invalidate it.
    S.imported_records[RD] = std::nullopt;

    // Skip unions and incomplete types.
    if (not RD or not RD->isCompleteDefinition() or RD->isUnion()) return std::nullopt;

    // Import the fields.
    auto& RL = AST().getASTRecordLayout(RD);
    SmallVector<FieldDecl*> Fields;
    for (auto [I, F] : enumerate(RD->fields())) {
        if (F->isBitField()) return std::nullopt;
        if (F->getMaxAlignment() != 0) return std::nullopt;
        if (F->hasInClassInitializer()) return std::nullopt;
        auto FTY = ImportType(F->getType());
        if (not FTY) return std::nullopt;
        Fields.push_back(new (*S.tu) FieldDecl(
            FTY.value(),
            Size::Bits(RL.getFieldOffset(unsigned(I))),
            S.tu->save(F->getName()),
            ImportSourceLocation(F->getLocation())
        ));
    }

    // Validate other properties of this type.
    if (auto CXX = dyn_cast<clang::CXXRecordDecl>(RD)) {
        if (not CXX->isCLike()) return std::nullopt;
    }

    // Determine the name of this type.
    StringRef Name = ImportName(RD);

    // Determine if this contains a pointer.
    bool contains_pointer = any_of(Fields, [&](auto *fd) {
        return fd->type->is_or_contains_pointer();
    });

    // Build the layout.
    auto rl = RecordLayout::Create(
        *S.tu,
        Fields,
        Size::Bytes(RL.getSize().getQuantity()),
        Size::Bytes(RL.getSize().getQuantity()),
        Align(RL.getAlignment().getQuantity()),
        RecordLayout::Bits::Trivial(contains_pointer)
    );

    auto Struct = S.BuildCompleteStructType(
        S.tu->save(Name),
        rl,
        ImportSourceLocation(RD->getLocation())
    );

    S.imported_records[RD] = Struct;
    return Struct;
}

auto Sema::Importer::ImportReturnType(clang::QualType T) -> std::optional<TypeAndValueCategory> {
    auto vc = Expr::RValue;
    if (T->isReferenceType()) {
        T = T.getNonReferenceType();
        vc = Expr::LValue(T.isConstQualified());
    }

    auto ty = TRY(ImportType(T));
    return TypeAndValueCategory{ty, vc};
}

auto Sema::Importer::ImportType(const clang::Type* T) -> std::optional<Type> {
    // FIXME: C++ pointers should be imported as nullable pointers once
    // we support optionals.

    // Handle known type sugar first.
    if (
        auto TD = T->getAs<clang::TypedefType>();
        TD and TD->getDecl()->getName() == "size_t" and
        T->getCanonicalTypeUnqualified() == AST().getSizeType()
    ) return Type::IntTy;

    // Only handle canonical types from here on.
    T = T->getCanonicalTypeUnqualified().getTypePtr();
    switch (T->getTypeClass()) {
        using K = clang::Type::TypeClass;
        default: return std::nullopt;
        case K::Builtin: {
            switch (cast<clang::BuiltinType>(T)->getKind()) {
                using K = clang::BuiltinType::Kind;
                default: return std::nullopt;
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
        }

        case K::LValueReference:
        case K::RValueReference: {
            auto Elem = ImportType(T->getPointeeType());
            if (not Elem) return std::nullopt;
            return PtrType::Get(*S.tu, *Elem, T->getPointeeType().isConstQualified());
        }

        // C++ pointers are nullable, so wrap them in an optional.
        case K::Pointer: {
            auto Elem = ImportType(T->getPointeeType());
            if (not Elem) return std::nullopt;
            auto Ptr = PtrType::Get(*S.tu, *Elem, T->getPointeeType().isConstQualified());
            return OptionalType::Get(*S.tu, Ptr);
        }

        case K::BitInt: {
            auto B = cast<clang::BitIntType>(T);
            return IntType::Get(*S.tu, Size::Bits(B->getNumBits()));
        }

        case K::ConstantArray: {
            auto C = cast<clang::ConstantArrayType>(T);
            auto Elem = ImportType(C->getElementType());
            if (not Elem) return std::nullopt;
            return ArrayType::Get(*S.tu, *Elem, i64(C->getSize().getZExtValue()));
        }

        case K::Enum:
            return cast<TypeDecl>(TRY(ImportDecl(T->getAsEnumDecl())))->type;

        case K::FunctionProto: {
            auto FPT = cast<clang::FunctionProtoType>(T);
            if (FPT->getCallConv() != clang::CallingConv::CC_C) return std::nullopt;

            auto Ret = FPT->getExtInfo().getNoReturn()
                ? TypeAndValueCategory{Type::NoReturnTy, Expr::RValue}
                : TRY(ImportReturnType(FPT->getReturnType()));

            SmallVector<ParamTypeData, 6> Params;
            for (auto P : FPT->param_types()) {
                auto Ty = TRY(ImportType(P));
                Params.emplace_back(Intent::Copy, Ty);
            }

            return ProcType::Get(
                *S.tu,
                Ret,
                Params,
                CallingConvention::Native,
                FPT->isVariadic()
            );
        }

        case K::Record: {
            auto RD = T->getAsRecordDecl();
            if (not RD) return std::nullopt;
            return cast<TypeDecl>(TRY(ImportDecl(RD)))->type;
        }
    }
}

auto Sema::Importer::ImportSourceLocation(clang::SourceLocation sloc) -> SLoc {
    if (not sloc.isValid()) return {};
    auto& sm = AST().getSourceManager();
    auto f = S.ctx.try_get_file(sm.getFilename(sloc).str());
    if (not f.has_value()) return {};
    return SLoc(f.value()->data() + sm.getFileOffset(sloc));
}

auto Sema::Importer::ImportValue(
    const clang::APValue& val,
    Type ty
) -> std::optional<eval::RValue> {
    switch (val.getKind()) {
        case clang::APValue::Int: {
            if (not ty->is_integer_or_bool()) return std::nullopt;
            return eval::RValue(val.getInt(), ty);
        }

        case clang::APValue::Struct: {
            // Type must be a struct type.
            auto struct_ty = dyn_cast<RecordType>(ty);
            if (not struct_ty) return std::nullopt;

            // Field count must match what we expect.
            u32 num_fields = val.getStructNumFields();
            Assert(struct_ty->layout().fields().size() == num_fields);

            // Import each field.
            SmallVector<eval::RValue*> fields;
            for (u32 i = 0; i < num_fields; i++) {
                auto field = TRY(ImportValue(
                    val.getStructField(i),
                    struct_ty->layout().fields()[i]->type
                ));

                fields.push_back(S.tu->save(field));
            }

            return eval::RValue(
                eval::Record(ArrayRef(fields).copy(S.tu->allocator())),
                ty
            );
        }

        default: return std::nullopt;
    }

    Unreachable();
}

auto Sema::ImportCXXDecl(ImportedClangModuleDecl* clang_module, CXXDecl* decl) -> Ptr<Decl> {
    Importer importer(*this, clang_module);
    auto d = importer.ImportDecl(decl);
    imported_decls[decl] = d;
    return d;
}

auto Sema::ParseCXX(
    StringRef code,
    std::optional<std::string> PCH
) -> std::unique_ptr<clang::ASTUnit> {
    // For PCHs, we need to remember this file and its contents.
    auto Name = std::format("__srcc.imports.{}.cc", cxx_import_file_counter++);
    PCHVFS->addFile(Name, 0, llvm::MemoryBuffer::getMemBufferCopy(code, Name));

    // The arguments need to include the tool name as well as the file name.
    std::vector<std::string> args{
        SOURCE_CLANG_EXE,
        "-x",
        "c++",
        Name,

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

    int clang_opt_level = 0;
    if (ctx.opt_level == 4) {
        args.push_back("-march=native");
        clang_opt_level = 3;
    } else {
        clang_opt_level = ctx.opt_level;
    }

    args.push_back(std::format("-O{}", clang_opt_level));

    if (PCH.has_value()) {
        args.push_back("-include-pch");
        args.push_back(std::move(PCH.value()));
    }

    for (const auto& p : clang_include_paths) {
        args.push_back("-I");
        args.push_back(p);
    }

    // We don't use clang::tooling::buildASTFromCodeWithArgs() because we handle
    // setting up the file system ourselves.
    struct Action : clang::tooling::ToolAction {
        std::unique_ptr<clang::ASTUnit> ast;
        bool runInvocation(
            std::shared_ptr<clang::CompilerInvocation> invocation,
            clang::FileManager* file_mgr,
            std::shared_ptr<clang::PCHContainerOperations> pch_ops,
            clang::DiagnosticConsumer* consumer
        ) override {
            Assert(not ast, "Expected at most one ASTUnit here!");
            auto clang_diags = clang::CompilerInstance::createDiagnostics(
                file_mgr->getVirtualFileSystem(),
                invocation->getDiagnosticOpts(),
                consumer,
                /*ShouldOwnClient=*/false
            );

            ast = clang::ASTUnit::LoadFromCompilerInvocation(
                std::move(invocation),
                std::move(pch_ops),
                nullptr,
                clang_diags,
                file_mgr,
                false,
                clang::CaptureDiagsKind::None
            );

            return ast != nullptr && not ast->getSema().hasUncompilableErrorOccurred();
        }
    };

    Action the_action;
    auto file_mgr = llvm::makeIntrusiveRefCnt<clang::FileManager>(clang::FileSystemOptions(), ImportVFS);
    clang::tooling::ToolInvocation invocation{
        args,
        &the_action,
        file_mgr.get(),
        std::make_shared<clang::PCHContainerOperations>()
    };

    if (not invocation.run()) return nullptr;
    Assert(the_action.ast);
    return std::move(the_action.ast);
}

auto Sema::ImportCXXHeaders(
    String logical_name,
    ArrayRef<String> header_names,
    SLoc import_loc
) -> Ptr<ImportedClangModuleDecl> {
    auto AST = ParseCXX(utils::join(header_names, "", "#include {}\n"));
    if (not AST) {
        Error(import_loc, "Header import failed");
        return {};
    }

    clang_ast_units.push_back(std::move(AST));
    return ImportedClangModuleDecl::Create(
        *tu,
        *clang_ast_units.back(),
        logical_name,
        header_names,
        import_loc
    );
}

namespace {
enum class Kind {
    Nothing,
    Unsupported,
    Function,
    Enumerator,
    Type,
};
}

auto Sema::LookUpCXXNameImpl(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclName> names,
    LookupHint hint
) -> LookupResult {
    auto ast = &clang_module->clang_ast;
    auto& clang_sema = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = clang_sema.getPreprocessor();

    // Look up all scopes in the path.
    clang::DeclContext* ctx = ast_ctx.getTranslationUnitDecl();
    for (auto n : names.drop_back()) {
        auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(n.str())));
        if (not res.isSingleResult()) return LookupResult::NonScopeInPath(n);
        auto new_ctx = dyn_cast<clang::DeclContext>(res.front());
        if (not new_ctx) return LookupResult::NonScopeInPath(n);
        ctx = new_ctx;
    }

    // Look up the last segment in the scope.
    // TODO: Support operators.
    auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(names.back().str())));

    // Figure out what we found.
    auto kind = Kind::Nothing;
    auto Merge = [&](Kind k) {
        using enum Kind;
        Assert(k != Nothing, "'Nothing' should only be used as an initial state");
        Assert(kind != Unsupported, "Should not process any more decls after reaching 'Unsupported'");
        Assert(k != Unsupported, "Should just early return instead of merging 'Unsupported'");
        if (k == kind) {} // Do nothing. They’re the same anyway.
        else if (kind == Nothing) kind = k;
        else if (kind == Type and hint == LookupHint::Type) {} // Prefer types if types were requested.
        else if (kind == Function or k == Function) kind = Function;
        else if (kind == Enumerator or k == Enumerator) kind = Enumerator;
        else {
            Assert(k == Type and kind == Type);
            kind = Type;
        }
    };

    for (auto d : res) {
        if (d->isInvalidDecl()) continue;
        if (isa<clang::FunctionDecl>(d)) Merge(Kind::Function);
        else if (isa<clang::TypedefDecl, clang::EnumDecl, clang::RecordDecl>(d)) Merge(Kind::Type);
        else if (isa<clang::EnumConstantDecl>(d)) Merge(Kind::Enumerator);
        else {
            kind = Kind::Unsupported;
            break;
        }
    }

    // And import the declarations we care about.
    SmallVector<Decl*> converted;
    auto ImportFiltered = [&](auto filter) {
        for (auto d : res) {
            if (d->isInvalidDecl() or not std::invoke(filter, d)) continue;
            auto decl = ImportCXXDecl(clang_module, d);
            if (not decl) return LookupResult::FailedToImport(names.back());
            if (not is_contained(converted, decl.get())) converted.push_back(decl.get());
        }

        if (converted.empty()) return LookupResult::NotFound(names.back());
        if (converted.size() == 1) return LookupResult::Success(converted.front());
        return LookupResult::Ambiguous(names.back(), converted);
    };

    switch (kind) {
        case Kind::Nothing:
            return LookupResult::NotFound(names.back());

        case Kind::Unsupported:
            return LookupResult::FailedToImport(names.back());

        case Kind::Function:
            return ImportFiltered([](auto* d){ return isa<clang::FunctionDecl>(d); });

        case Kind::Enumerator:
            return ImportFiltered([](auto* d){ return isa<clang::EnumConstantDecl>(d); });

        case Kind::Type:
            return ImportFiltered([](auto* d){
                return isa<
                    clang::TypedefDecl,
                    clang::EnumDecl,
                    clang::RecordDecl
                >(d);
            });
    }

    Unreachable();
}

auto Sema::LookUpCXXName(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclName> names,
    LookupHint hint
) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto DoNameLookup = [&] {
        return LookUpCXXNameImpl(clang_module, names, hint);
    };

    // If this is anything other than a single identifier, it can’t be a macro.
    if (names.size() != 1 or not names.front().is_str())
        return DoNameLookup();

    auto ast = &clang_module->clang_ast;
    auto& clang_sema = ast->getSema();
    auto& pp = clang_sema.getPreprocessor();
    auto id = pp.getIdentifierInfo(names.front().str());
    auto *mi = pp.getMacroInfo(id);
    if (not mi) return DoNameLookup();

    // Refuse to import ‘builtin’ macros (i.e. __LINE__, __COUNTER__, and friends)
    // as well as function-like macros.
    if (mi->isBuiltinMacro() or mi->isFunctionLike())
        return LookupResult::FailedToImport(names.back());

    // If we have attempted to find this before, do not do so again.
    if (auto it = imported_macros.find(mi); it != imported_macros.end()) {
        if (it->second.present()) return LookupResult::Success(it->second.get());
        return LookupResult::FailedToImport(names.back());
    }

    // Cache that we attempted to import this.
    imported_macros[mi] = nullptr;

    // Enter the macro definition as a new file.
    auto& sm = clang_sema.getSourceManager();
    auto clang_sloc = sm.getLocForStartOfFile(sm.getMainFileID());
    auto fid = sm.createFileID(
        llvm::MemoryBuffer::getMemBufferCopy(
            names.front().str(),
            "<macro expansion>"
        )
    );

    if (pp.EnterSourceFile(fid, nullptr, clang_sloc))
        return LookupResult::FailedToImport(names.back());

    // Ask the preprocessor to expand the macro.
    std::vector<clang::Token> toks;

    // Manual implementation of LexTokensUntilEOF that also keeps the
    // EOF token, because 'StringifyArgument' expects an EOF-terminated
    // token list, because of course it does.
    for (;;) {
        auto& tok = toks.emplace_back();
        pp.Lex(tok);
        if (tok.is(clang::tok::eof)) break;
        if (tok.is(clang::tok::unknown)) return LookupResult::FailedToImport(names.back());
    }

    // Macro expanded to nothing.
    if (toks.size() == 1) return LookupResult::FailedToImport(names.back());

    // If it is a single identifier, just look up that name.
    if (
        toks.size() == 2 and
        toks.front().is(clang::tok::identifier)
    ) return LookUpCXXNameImpl(
        clang_module,
        DeclName(tu->save(toks.front().getIdentifierInfo()->getName())),
        hint
    );

    // Otherwise, we need to parse this thing.
    //
    // First, emit the Clang module to a PCH if we haven't already done that.
    auto pch_name = std::format("__srcc_pch_{}", static_cast<void*>(clang_module));
    if (not PCHVFS->exists(pch_name)) {
        SmallString<0> pch;
        clang::CodeGenOptions default_opts{};
        llvm::raw_svector_ostream os{pch};
        if (clang_module->clang_ast.serialize(os))
            return LookupResult::FailedToImport(names.back());
        PCHVFS->addFile(pch_name, 0, llvm::MemoryBuffer::getMemBufferCopy(pch));
    }

    // Wrap the macro in a function that returns it and also create a variable initialised
    // with a call to it; the latter is used to try and constant-evaluate the function.
    //
    // Clang gets mad if we combine 'extern "C"' and 'decltype(auto)', so use '__asm__'
    // instead to set the function name; this however requires separating the definition and
    // declaration for some reason...
    //
    // The 'always_inline' is required, else inlining just doesn't happen at all (possibly
    // because the function has 'linkonce_odr' linkage).
    auto name_base = Format("__srcc_expanded_macro_{}", generated_cxx_macro_decls);
    auto proc_name = Format("{}_init", name_base);
    auto var_name = Format("{}_var", name_base);
    auto code = Format(
        "[[clang::always_inline]] constexpr decltype(auto) {0}() __asm__(\"{0}\");\n"
        "constexpr decltype(auto) {0}() {{ return {1}; }}\n"
        "decltype(auto) {2} = {0}();\n",
        proc_name,
        names.front().str(),
        var_name
    );

    auto macro_tu = ParseCXX(code, std::move(pch_name));
    if (not macro_tu or macro_tu->getSema().hasUncompilableErrorOccurred())
        return LookupResult::FailedToImport(names.back());

    // Get the procedure or variable decl. Take care to retrieve the II
    // in the preprocessor of the new TU.
    auto GetDecl = [&](StringRef name) -> Ptr<clang::Decl> {
        clang::DeclarationName ident{macro_tu->getPreprocessor().getIdentifierInfo(name)};
        auto res = macro_tu->getASTContext().getTranslationUnitDecl()->lookup(ident);
        if (not res.isSingleResult() or res.front()->isInvalidDecl())
            return nullptr;
        return res.front();
    };

    // Retrieve the variable.
    auto var = cast_if_present<clang::VarDecl>(GetDecl(var_name));
    if (not var) return LookupResult::FailedToImport(names.back());

    // Import the source location of the macro.
    SLoc macro_loc;
    {
        Importer i{*this, clang_module};
        macro_loc = i.ImportSourceLocation(mi->getDefinitionLoc());
    }

    // The importer requires a module, so fabricate one for this.
    auto fake_module = ImportedClangModuleDecl::Create(
        *tu,
        *macro_tu,
        "__srcc_macro_expansion__",
        {},
        SLoc()
    );

    Importer main_importer{*this, clang_module};
    Importer fake_importer{*this, fake_module};

    // Import a type from the Clang ASTUnit used for expansion to Source.
    auto ImportTypeFromExpansionAST = [&](clang::QualType clang_ty) -> std::expected<Type, LookupResult> {
        // Import the type of the value from the new ASTUnit to the main one;
        // this is to make sure that two references to the same type actually
        // resolve to the same Source type.
        clang::ASTImporter clang_importer{
            ast->getASTContext(),
            ast->getFileManager(),
            fake_module->clang_ast.getASTContext(),
            fake_module->clang_ast.getFileManager(),
            /*MinimalImport=*/false,
            clang_importer_state
        };

        auto imported_type = clang_importer.Import(clang_ty);
        if (auto err = imported_type.takeError()) {
            return std::unexpected(fake_importer.FormatTypeFailure(
                names.back(),
                clang_ty
            ));
        }

        // Then import the type from the main module to Source.
        Importer i{*this, clang_module};
        auto ty = i.ImportType(*imported_type);
        if (not ty) return std::unexpected(i.FormatTypeFailure(names.back(), *imported_type));
        return *ty;
    };

    // If we can’t evaluate this as a constant, instead emit the function
    // into an LLVM module and create a *local* value whose initialiser is
    // a call to it.
    clang::APValue init_val;
    SmallVector<clang::PartialDiagnosticAt> diags;
    Expr* value = nullptr;
    if (
        not var->getInit()->EvaluateAsInitializer(
            init_val,
            macro_tu->getASTContext(),
            var,
            diags,
            false
        ) or not diags.empty()
    )  {
        // Retrieve the procedure.
        auto clang_proc = cast_if_present<clang::FunctionDecl>(GetDecl(proc_name));
        if (not clang_proc) return LookupResult::FailedToImport(names.back());

        // Import its return type.
        auto clang_ret_ty = clang_proc->getType()->getAs<clang::FunctionProtoType>()->getReturnType();
        auto ret_ty = fake_importer.ImportReturnType(clang_ret_ty);
        if (not ret_ty) return LookupResult::FailedToImport(names.back());

        // Emit it.
        defer { generated_cxx_macro_decls++; };
        std::unique_ptr<clang::CodeGenerator> cg{clang::CreateLLVMCodeGen(
            macro_tu->getDiagnostics(),
            fake_module->name.str(),
            macro_tu->getVirtualFileSystemPtr(),
            macro_tu->getPreprocessor().getHeaderSearchInfo().getHeaderSearchOpts(),
            macro_tu->getPreprocessor().getPreprocessorOpts(),
            macro_tu->getCodeGenOpts(),
            tu->llvm_context
        )};

        // Make sure we actually emit this.
        clang_proc->setIsUsed();

        // I need to fix this API at some point, because the fact that you have to
        // *remember* to call Initialize() is... not great.
        cg->Initialize(macro_tu->getASTContext());
        cg->HandleTopLevelDecl(clang::DeclGroupRef(clang_proc));
        cg->HandleTranslationUnit(macro_tu->getASTContext());
        if (not cg->GetModule()) return LookupResult::FailedToImport(names.back());
        tu->link_llvm_modules.emplace_back(cg->ReleaseModule());

        // Declare the procedure on our end.
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
            CreateReference(proc, macro_loc).get(),
            {},
            macro_loc
        ).get();
    }

    // Ok, we managed to constant-evaluate the initialiser; convert
    // it to an RValue.
    else {
        auto ty = ImportTypeFromExpansionAST(var->getType());
        if (not ty) return std::move(ty.error());

        // Finally, import the value.
        auto v = fake_importer.ImportValue(*var->getEvaluatedValue(), *ty);
        if (not v.has_value()) {
            SmallString<128> str;
            llvm::raw_svector_ostream os{str};
            var->getEvaluatedValue()->printPretty(os, fake_module->clang_ast.getASTContext(), var->getType());
            auto diag = CreateNote(
                macro_loc,
                "Unsupported '{}' APValue: {}",
                enchantum::to_string(var->getEvaluatedValue()->getKind()),
                str
            );

            return LookupResult::FailedToImport(
                names.back(),
                std::make_unique<Diagnostic>(std::move(diag))
            );
        }

        value = MakeConstExpr(nullptr, std::move(*v), macro_loc);
    }

    // Do NOT wrap the value in a SaveExpr! We want it to be evaluated every
    // time the macro is referenced so that it actually behaves like a C++ macro.
    auto vd = new (*tu) ValueDecl("", value, value->location());
    imported_macros[mi] = vd;
    return LookupResult::Success(vd);
}
