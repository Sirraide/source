module;

#include <clang/Basic/FileManager.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Frontend/CompilerInstance.h>
#include <fmt/core.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>
#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.ast;
using namespace srcc;

class Sema::Importer {
    Sema& S;
    clang::ASTUnit& AST;
    Module::Ptr Mod;
    llvm::DenseSet<clang::Decl*> imported_decls;

public:
    explicit Importer(Sema& S, clang::ASTUnit& AST) : S(S), AST(AST) {}

    /// Imports a module.
    [[nodiscard]] auto Import(String name) -> Module::Ptr;
    void ImportDecl(clang::Decl* D);
    void ImportFunction(clang::FunctionDecl* D);
    auto ImportType(const clang::Type* T) -> std::optional<Type>;
    auto ImportType(clang::QualType T) { return ImportType(T.getTypePtr()); }
};

auto Sema::Importer::Import(String name) -> Module::Ptr {
    Mod = Module::Create(S.context(), name, true);
    auto* TU = AST.getASTContext().getTranslationUnitDecl();
    for (auto D : TU->decls()) ImportDecl(D);
    return std::move(Mod);
}

void Sema::Importer::ImportDecl(clang::Decl* D) {
    if (imported_decls.contains(D->getCanonicalDecl())) return;
    imported_decls.insert(D->getCanonicalDecl());
    if (D->isInvalidDecl()) return;
    switch (D->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::LinkageSpec:
            for (auto L = cast<clang::LinkageSpecDecl>(D); auto D : L->decls())
                ImportDecl(D);
            break;

        case K::Enum:
            // TODO
            break;

        case K::Function:
            ImportFunction(cast<clang::FunctionDecl>(D));
            break;

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
}

void Sema::Importer::ImportFunction(clang::FunctionDecl* D) {
    D = D->getDefinition() ?: D->getCanonicalDecl();
    if (isa<clang::CXXMethodDecl>(D)) return;
    auto FPT = D->getType()->getAs<clang::FunctionProtoType>();
    Assert(FPT, "No prototype in C++?");

    // If the return type hasn’t been deduced yet, we can’t import it.
    if (D->getReturnType()->getAs<clang::AutoType>()) return;

    // Don’t import immediate functions.
    if (D->isImmediateFunction()) return;

    // Do not import language builtins.
    //
    // Note: Clang treats C standard library functions (e.g. 'puts') as
    // builtins as well, but those count as ‘library builtins’.
    switch (clang::Builtin::ID(D->getBuiltinID())) {
        // Ignore everything we don’t know what to do with.
        default:
            return;

            // Import library builtins only.
#define BUILTIN(ID, ...)
#define LIBBUILTIN(ID, ...) case clang::Builtin::BI##ID:
#include "clang/Basic/Builtins.inc"

        case clang::Builtin::NotBuiltin:
            break;
    }

    // Don’t import constexpr or inline functions for now.
    if (D->isConstexpr() or D->isInlineSpecified()) return;

    // Don’t import functions with internal linkage.
    if (D->getLinkageInternal() != clang::Linkage::External) return;

    // Import the type.
    auto T = ImportType(FPT);
    if (not T) return;

    // Create the procedure.
    auto* Proc = new (*Mod) ProcDecl(
        *T,
        Mod->save(D->getNameAsString()),
        Linkage::Imported,
        D->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        nullptr,
        {}
    );

    Mod->procs.push_back(Proc);
    Mod->exports.add(Proc);
}

// The types in this function are currently created in the module we’re
// importing into because canonical types are compared by pointer; we’ll
// have to import them into the destiation module instead when we switch
// to processing and deduplicating imports before Sema.
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
                case K::Void: return S.M->VoidTy;
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
            if (not Elem) return nullptr;
            return ReferenceType::Get(*S.M, *Elem);
        }

        case K::BitInt: {
            auto B = cast<clang::BitIntType>(T);
            return IntType::Get(*S.M, i64(B->getNumBits()));
        }

        case K::ConstantArray: {
            auto C = cast<clang::ConstantArrayType>(T);
            auto Elem = ImportType(C->getElementType());
            if (not Elem) return nullptr;
            return ArrayType::Get(*S.M, *Elem, i64(C->getSize().getZExtValue()));
        }

        case K::FunctionProto: {
            auto FPT = cast<clang::FunctionProtoType>(T);
            if (FPT->getCallConv() != clang::CallingConv::CC_C) return nullptr;

            auto Ret = FPT->getExtInfo().getNoReturn() ? S.M->NoReturnTy : ImportType(FPT->getReturnType());
            if (not Ret) return nullptr;

            SmallVector<Type> Params;
            for (auto P : FPT->param_types()) {
                auto T = ImportType(P);
                if (not T) return nullptr;
                Params.push_back(*T);
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

auto Sema::ImportCXXHeader(String name) -> Result<Module::Ptr> {
    // TODO: Try using `clang::tooling::buildASTFromCodeWithArgs()`.
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay;
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> mem;

    // Create a filesystem where we can add the ‘header’.
    overlay = std::make_unique<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    mem = std::make_unique<llvm::vfs::InMemoryFileSystem>();
    overlay->pushOverlay(mem);

    // Initialise clang.
    clang::CompilerInstance clang;
    clang.createDiagnostics();
    clang.getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();
    if (not clang.createTarget()) Diag::ICE("Failed to create target for importing '{}'", name);
    clang.createSourceManager(*clang.createFileManager(overlay));
    clang.createPreprocessor(clang::TU_Prefix);
    clang.createASTContext();
    clang.getDiagnostics().setShowColors(ctx.use_colours());

    // Create the file we’re going to parse.
    auto code = fmt::format("#include <{}>\n", name);
    auto buffer = llvm::MemoryBuffer::getMemBuffer(code);
    mem->addFile("__srcc_imports.cc", /*mtime=*/0, std::move(buffer));

    // Construct Clang command line.
    //
    // TODO: This path may not exist if we ever ship srcc
    // on its own; we may have to ship Clang’s resource dir w/ it.
    std::array args{
        SOURCE_CLANG_EXE,
        "-x",
        "c++",
        "__srcc_imports.cc",
        "-std=c++2c",
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",
        "-fcolor-diagnostics",
        "-fsyntax-only"
    };

    // Parse it.
    clang::CreateInvocationOptions opts;
    opts.VFS = overlay;
    opts.Diags = &clang.getDiagnostics();
    auto CI = std::shared_ptr<clang::CompilerInvocation>{clang::createInvocation(args, opts).release()};
    auto AST = clang::ASTUnit::LoadFromCompilerInvocation(
        CI,
        std::make_shared<clang::PCHContainerOperations>(),
        clang.getDiagnosticsPtr(),
        &clang.getFileManager()
    );

    if (not AST) return Diag::Error("Failed to import C++ header '{}'", name);
    Importer I{*this, *AST};
    return I.Import(name);
}
