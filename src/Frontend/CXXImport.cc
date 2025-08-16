#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/Basic/FileManager.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>

#include <print>

using namespace srcc;

class Sema::Importer {
    Sema& S;
    clang::ASTUnit& AST;

public:
    explicit Importer(Sema& S, clang::ASTUnit& AST) : S(S), AST(AST) {}
    auto ImportDecl(clang::Decl* D) -> Ptr<Decl>;
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
            S.imported_decls[D] = f;
            return f;
        }

        case K::Namespace:
            // TODO
            break;

        case K::Record:
            // TODO
            break;

        case K::Typedef:
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
                case K::Bool: return S.M->FFIBoolTy;

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
    return importer.ImportDecl(decl);
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

auto Sema::LookUpCXXName(clang::ASTUnit* ast, ArrayRef<String> names) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto& actions = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = actions.getPreprocessor();

    // Look up all scopes in the path.
    clang::DeclContext* ctx = ast_ctx.getTranslationUnitDecl();
    for (auto n : names.drop_back()) {
        auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(n)));
        if (not res.isSingleResult()) return LookupResult::NonScopeInPath(n);
        auto new_ctx = dyn_cast<clang::DeclContext>(res.front());
        if (not new_ctx) return LookupResult::NonScopeInPath(n);
        ctx = new_ctx;
    }

    // Look up the last segment in the scope.
    auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(names.back())));
    if (res.empty()) return LookupResult::NotFound(names.back());

    // We found exactly one name.
    if (res.isSingleResult()) {
        auto decl = ImportCXXDecl(*ast, res.front());
        if (not decl) return LookupResult::FailedToImport();
        return LookupResult::Success(decl.get());
    }

    // We found more than one; return them all if they’re all functions.
    SmallVector<Decl*> converted;
    if (not llvm::all_of(res, [](auto* d) { return isa<clang::FunctionDecl>(d); }))
        return LookupResult::NotFound(names.back());

    // They are. Import them all.
    for (auto d : res)
        if (auto decl = ImportCXXDecl(*ast, d))
            converted.push_back(decl.get());

    // We might end up with only one—or even none—if we couldn’t import
    // one of them.
    if (converted.empty()) return LookupResult::FailedToImport();
    if (converted.size() == 1) return LookupResult::Success(converted.front());
    return LookupResult::Ambiguous(names.back(), converted);
}
