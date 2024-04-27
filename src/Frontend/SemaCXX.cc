module;

#include <clang/Basic/FileManager.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <fmt/core.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>
#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.ast;

using namespace srcc;
namespace srcc {
class Importer;
}

class srcc::Importer {
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
    auto ImportType(const clang::Type* Ty) -> Type*;
    auto ImportType(clang::QualType Ty) { return ImportType(Ty.getTypePtr()); }
};

auto Importer::Import(String name) -> Module::Ptr {
    Mod = Module::Create(S.Context(), name, true);
    auto* TU = AST.getASTContext().getTranslationUnitDecl();
    for (auto D : TU->decls()) ImportDecl(D);
    return std::move(Mod);
}

void Importer::ImportDecl(clang::Decl* D) {
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

void Importer::ImportFunction(clang::FunctionDecl* D) {
    D = D->getDefinition() ?: D->getCanonicalDecl();
    if (isa<clang::CXXMethodDecl>(D)) return;
    auto FPT = D->getType()->getAs<clang::FunctionProtoType>();
    Assert(FPT, "No prototype in C++?");

    std::string s = D->getNameAsString();

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
    auto* Ty = ImportType(FPT);
    if (not Ty) return;

    // Create the procedure.
    auto* Proc = new (*Mod) ProcDecl(
        Ty,
        Mod->save(D->getNameAsString()),
        Linkage::Imported,
        D->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        nullptr,
        {}
    );

    Mod->procs.push_back(Proc);
    Mod->exports[Proc->name].push_back(Proc);
}

auto Importer::ImportType(const clang::Type* Ty) -> Type* {
    // Handle known type sugar first.
    if (
        auto TD = Ty->getAs<clang::TypedefType>();
        TD and TD->getDecl()->getName() == "size_t" and
        Ty->getCanonicalTypeUnqualified() == AST.getASTContext().getSizeType()
    ) return Mod->FFISizeTy;

    // Only handle canonical types from here on.
    Ty = Ty->getCanonicalTypeUnqualified().getTypePtr();
    switch (Ty->getTypeClass()) {
        using K = clang::Type::TypeClass;
        default: return nullptr;
        case K::Builtin: {
            switch (cast<clang::BuiltinType>(Ty)->getKind()) {
                using K = clang::BuiltinType::Kind;
                default: return nullptr;
                case K::Void: return Mod->VoidTy;
                case K::Bool: return Mod->FFIBoolTy;

                case K::SChar:
                case K::UChar:
                case K::Char_S:
                case K::Char_U:
                    return Mod->FFICharTy;

                case K::Short:
                case K::UShort:
                    return Mod->FFIShortTy;

                case K::Int:
                case K::UInt:
                    return Mod->FFIIntTy;

                case K::Long:
                case K::ULong:
                    return Mod->FFILongTy;

                case K::LongLong:
                case K::ULongLong:
                    return Mod->FFILongLongTy;
            }
        }

        case K::LValueReference:
        case K::RValueReference:
        case K::Pointer: {
            auto Elem = ImportType(Ty->getPointeeType());
            if (not Elem) return nullptr;
            return ReferenceType::Get(*Mod, Elem);
        }

        case K::BitInt: {
            auto B = cast<clang::BitIntType>(Ty);
            return IntType::Get(*Mod, i64(B->getNumBits()));
        }

        case K::ConstantArray: {
            auto C = cast<clang::ConstantArrayType>(Ty);
            auto Elem = ImportType(C->getElementType());
            if (not Elem) return nullptr;
            return ArrayType::Get(*Mod, Elem, i64(C->getSize().getZExtValue()));
        }

        case K::FunctionProto: {
            auto FPT = cast<clang::FunctionProtoType>(Ty);
            if (FPT->getCallConv() != clang::CallingConv::CC_C) return nullptr;

            auto Ret = FPT->getExtInfo().getNoReturn() ? Mod->NoReturnTy : ImportType(FPT->getReturnType());
            if (not Ret) return nullptr;

            SmallVector<Type*> Params;
            for (auto P : FPT->param_types()) {
                auto* T = ImportType(P);
                if (not T) return nullptr;
                Params.push_back(T);
            }

            return ProcType::Get(
                *Mod,
                Ret,
                Params,
                CallingConvention::Native,
                FPT->isVariadic()
            );
        }
    }
}

auto Sema::ImportCXXHeader(String name) -> Result<Module::Ptr> {
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
