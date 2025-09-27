#include <srcc/AST/Type.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/Decl.h>
#include <clang/AST/RecordLayout.h>
#include <clang/Basic/FileManager.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>

using namespace srcc;

class Sema::Importer {
    Sema& S;
    clang::ASTUnit& AST;

public:
    explicit Importer(Sema& S, clang::ASTUnit& AST) : S(S), AST(AST) {}
    auto ImportDecl(clang::Decl* D) -> Ptr<Decl>;
    auto ImportRecord(clang::RecordDecl* RD) -> std::optional<Type>;
    auto ImportFunction(clang::FunctionDecl* D) -> Ptr<ProcDecl>;
    auto ImportType(const clang::Type* T) -> std::optional<Type>;
    auto ImportType(clang::QualType T) { return ImportType(T.getTypePtr()); }
    auto ImportSourceLocation(clang::SourceLocation sloc) -> Location;
};

auto Sema::Importer::ImportDecl(clang::Decl* D) -> Ptr<Decl> {
    D = D->getCanonicalDecl();

    // If we have attempted to find this before, do not do so again.
    if (auto it = S.imported_decls.find(D); it != S.imported_decls.end())
        return it->second;

    // Otherwise, try to create it now. First, mark that we’ve already
    // tried doing this.
    S.imported_decls[D] = nullptr;

    // Ignore invalid ones.
    if (D->isInvalidDecl()) return {};
    switch (D->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::Enum:
            // TODO
            break;

        case K::Function: {
            auto f = ImportFunction(cast<clang::FunctionDecl>(D));
            if (not f) break;
            return f;
        }

        case K::Namespace:
            // TODO
            break;

        case K::Record:{
            auto ty = ImportRecord(cast<clang::RecordDecl>(D));
            if (ty) return cast<StructType>(ty.value())->decl();
        } break;

        case K::Typedef: {
            auto td = cast<clang::TypedefDecl>(D);
            auto clang_ty = AST.getASTContext().getTypedefType(
                clang::ElaboratedTypeKeyword::None,
                std::nullopt,
                td
            );

            if (clang_ty->isRecordType()) {
                auto ty = ImportType(clang_ty);
                if (ty) return cast<StructType>(ty.value())->decl();
            }
        } break;

        case K::Using:
            // TODO
            break;

        case K::Var:
            // TODO
            break;
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

auto Sema::Importer::ImportFunction(clang::FunctionDecl* D) -> Ptr<ProcDecl> {
    D = D->getDefinition() ?: D->getCanonicalDecl();
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
        if (not AST.getASTContext().BuiltinInfo.isPredefinedLibFunction(ID))
            return {};
    }

    // Import the type.
    auto T = ImportType(FPT);
    if (not T) return {};
    if (not isa<ProcType>(*T)) return {};

    // Create the procedure.
    auto PD = ProcDecl::Create(
        *S.M,
        cast<ProcType>(T.value().ptr()),
        S.M->save(D->getNameAsString()),
        Linkage::Imported,
        D->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        ImportSourceLocation(D->getNameInfo().getBeginLoc())
    );

    // Create param decls.
    SmallVector<LocalDecl*> Params;
    for (auto [I, P] : enumerate(D->parameters())) {
        Params.push_back(new (*S.M) ParamDecl(
            &PD->param_types()[I],
            Expr::LValue,
            S.M->save(P->getName()),
            PD,
            u32(I),
            false,
            ImportSourceLocation(P->getLocation())
        ));
    }

    PD->finalise(nullptr, Params);
    return PD;
}

auto Sema::Importer::ImportRecord(clang::RecordDecl* RD) -> std::optional<Type> {
    Assert(RD);
    auto it = S.imported_records.find(RD);
    if (it != S.imported_records.end()) return it->second;

    // Create the cache entry now so we fail fast next time if we can’t import
    // this; don’t hold on to a reference to the cache entry here since we’re
    // about to import more types, which might invalidate it.
    S.imported_records[RD] = std::nullopt;

    // Skip unions and incomplete types.
    if (not RD or not RD->isCompleteDefinition() or RD->isUnion()) return std::nullopt;

    // Import the fields.
    auto& RL = AST.getASTContext().getASTRecordLayout(RD);
    SmallVector<FieldDecl*> Fields;
    for (auto [I, F] : enumerate(RD->fields())) {
        if (F->isBitField()) return std::nullopt;
        if (F->getMaxAlignment() != 0) return std::nullopt;
        if (F->hasInClassInitializer()) return std::nullopt;
        auto FTY = ImportType(F->getType());
        if (not FTY) return std::nullopt;
        Fields.push_back(new (*S.M) FieldDecl(
            FTY.value(),
            Size::Bits(RL.getFieldOffset(unsigned(I))),
            S.M->save(F->getName()),
            ImportSourceLocation(F->getLocation())
        ));
    }

    // Validate other properties of this type.
    if (auto CXX = dyn_cast<clang::CXXRecordDecl>(RD)) {
        if (not CXX->isCLike()) return std::nullopt;
    }

    // Determine the name of this type.
    StringRef Name;
    if (RD->hasNameForLinkage()) {
        if (RD->getDeclName().isIdentifier()) {
            Name = RD->getName();
        } else if (auto TD = RD->getTypedefNameForAnonDecl()) {
            Name = TD->getName();
        }
    }

    // Build the scope for the declaration.
    auto Scope = S.M->create_scope<StructScope>(S.global_scope());
    for (auto F : Fields) S.AddDeclToScope(Scope, F);

    // Build the layout.
    auto rl = RecordLayout::Create(
        *S.M,
        Fields,
        Size::Bytes(RL.getSize().getQuantity()),
        Size::Bytes(RL.getSize().getQuantity()),
        Align(RL.getAlignment().getQuantity()),
        RecordLayout::Bits::Trivial()
    );

    // Build the type.
    auto Struct = StructType::Create(
        *S.M,
        Scope,
        S.M->save(Name),
        ImportSourceLocation(RD->getLocation()),
        rl
    );

    S.imported_records[RD] = Struct;
    return Struct;
}

auto Sema::Importer::ImportType(const clang::Type* T) -> std::optional<Type> {
    // Handle known type sugar first.
    if (
        auto TD = T->getAs<clang::TypedefType>();
        TD and TD->getDecl()->getName() == "size_t" and
        T->getCanonicalTypeUnqualified() == AST.getASTContext().getSizeType()
    ) return S.M->FFISizeTy;

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
                    return S.M->FFICharTy;

                case K::Short:
                case K::UShort:
                    return S.M->FFIShortTy;

                case K::Int:
                case K::UInt:
                    return S.M->FFIIntTy;

                case K::Long:
                case K::ULong:
                    return S.M->FFILongTy;

                case K::LongLong:
                case K::ULongLong:
                    return S.M->FFILongLongTy;
            }
        }

        case K::LValueReference:
        case K::RValueReference:
        case K::Pointer: {
            auto Elem = ImportType(T->getPointeeType());
            if (not Elem) return std::nullopt;
            return PtrType::Get(*S.M, *Elem);
        }

        case K::BitInt: {
            auto B = cast<clang::BitIntType>(T);
            return IntType::Get(*S.M, Size::Bits(B->getNumBits()));
        }

        case K::ConstantArray: {
            auto C = cast<clang::ConstantArrayType>(T);
            auto Elem = ImportType(C->getElementType());
            if (not Elem) return std::nullopt;
            return ArrayType::Get(*S.M, *Elem, i64(C->getSize().getZExtValue()));
        }

        case K::FunctionProto: {
            auto FPT = cast<clang::FunctionProtoType>(T);
            if (FPT->getCallConv() != clang::CallingConv::CC_C) return std::nullopt;

            auto Ret = FPT->getExtInfo().getNoReturn() ? Type::NoReturnTy : ImportType(FPT->getReturnType());
            if (not Ret) return std::nullopt;

            SmallVector<ParamTypeData, 6> Params;
            for (auto P : FPT->param_types()) {
                auto Ty = ImportType(P);
                if (not Ty) return std::nullopt;
                Params.emplace_back(Intent::Copy, *Ty);
            }

            return ProcType::Get(
                *S.M,
                *Ret,
                Params,
                CallingConvention::Native,
                FPT->isVariadic()
            );
        }

        case K::Record: {
            auto RD = T->getAsRecordDecl();
            if (not RD) return std::nullopt;
            return ImportRecord(RD);
        }
    }
}

auto Sema::Importer::ImportSourceLocation(clang::SourceLocation sloc) -> Location {
    Location loc;
    if (not sloc.isValid()) return {};
    auto& sm = AST.getASTContext().getSourceManager();
    auto& F = S.ctx.get_file(sm.getFilename(sloc).str());
    loc.pos = sm.getFileOffset(sloc);
    loc.len = u16(clang::Lexer::MeasureTokenLength(sloc, sm, AST.getLangOpts()));
    loc.file_id = u16(F.file_id());
    return loc;
}

auto Sema::ImportCXXDecl(clang::ASTUnit& ast, CXXDecl* decl) -> Ptr<Decl> {
    Importer importer(*this, ast);
    auto d = importer.ImportDecl(decl);
    imported_decls[decl] = d;
    return d;
}

auto Sema::ImportCXXHeaders(
    String logical_name,
    ArrayRef<String> header_names,
    Location import_loc
) -> Ptr<ImportedClangModuleDecl> {
    std::vector<std::string> args{
        "-x",
        "c++",
        "-Xclang",
        "-triple",
        "-Xclang",
        llvm::sys::getDefaultTargetTriple(),
        "-std=c++2c",
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",
        "-fcolor-diagnostics",
        "-fsyntax-only"
    };

    for (const auto& p : clang_include_paths) {
        args.push_back("-I");
        args.push_back(p);
    }

    auto AST = clang::tooling::buildASTFromCodeWithArgs(
        utils::join(header_names, "", "#include {}\n"),
        args,
        "__srcc.imports.cc",
        SOURCE_CLANG_EXE
    );

    if (not AST) {
        Error(import_loc, "Header import failed");
        return {};
    }

    clang_ast_units.push_back(std::move(AST));
    return ImportedClangModuleDecl::Create(
        *M,
        *clang_ast_units.back(),
        logical_name,
        header_names,
        import_loc
    );
}

auto Sema::LookUpCXXName(clang::ASTUnit* ast, ArrayRef<DeclName> names) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto& actions = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = actions.getPreprocessor();

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
    if (res.empty()) return LookupResult::NotFound(names.back());

    // We found exactly one name.
    if (res.isSingleResult()) {
        auto decl = ImportCXXDecl(*ast, res.front());
        if (not decl) return LookupResult::FailedToImport();
        return LookupResult::Success(decl.get());
    }

    // If we found function declarations, import them all.
    SmallVector<Decl*> converted;
    if (llvm::all_of(res, [](auto* d) { return isa<clang::FunctionDecl>(d); })) {
        for (auto d : res)
            if (auto decl = ImportCXXDecl(*ast, d))
                converted.push_back(decl.get());
    }

    // 'typedef struct X {} X' is a common pattern in C; in the Clang
    // AST, we end up with both a RecordDecl and TypedefDecl for this.
    else if (
        auto it = rgs::find_if(res, [](auto* d) { return isa<clang::TypedefDecl>(d); });
        it != res.end()
    ) {
        auto decl = ImportCXXDecl(*ast, *it);
        if (decl) converted.push_back(decl.get());
    }

    // We might end up with only one—or even none—if we couldn’t import
    // one of them.
    if (converted.empty()) return LookupResult::FailedToImport();
    if (converted.size() == 1) return LookupResult::Success(converted.front());
    return LookupResult::Ambiguous(names.back(), converted);
}
