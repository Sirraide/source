#ifndef SRCC_AST_HH
#define SRCC_AST_HH

#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Core/Core.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <clang/Basic/TargetInfo.h>
#include <clang/Frontend/ASTUnit.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <base/DSA.hh>

namespace srcc {
class ImportHandle;
class TranslationUnit;
}

/// Ref-counted handle to an import.
class srcc::ImportHandle : public llvm::PointerUnion<TranslationUnit*, clang::ASTUnit*> {
    // So we don’t have to deal with implementing a refcount; we won’t
    // be moving these around much anyway.
    Variant<
        std::shared_ptr<TranslationUnit>,
        std::shared_ptr<clang::ASTUnit>>
        shared_handle;

    // Logical name of this import.
    String import_name;

    // Location of the import.
    Location import_location;

    ImportHandle(const ImportHandle&) = default;
    ImportHandle& operator=(const ImportHandle&) = default;

public:
    explicit ImportHandle(std::unique_ptr<TranslationUnit> h);
    explicit ImportHandle(std::unique_ptr<clang::ASTUnit> h);
    ImportHandle(ImportHandle&&) = default;
    ImportHandle& operator=(ImportHandle&&) = default;

    auto copy(String logical_name, Location loc) -> ImportHandle;
    auto logical_name() const -> String { return import_name; }
    auto location() const -> Location { return import_location; }
    auto ptr() -> PointerUnion<TranslationUnit*, clang::ASTUnit*> { return *this; }
};

/// Representation of a single program or module. NOT thread-safe.
class srcc::TranslationUnit {
    SRCC_IMMOVABLE(TranslationUnit);

public:
    struct Serialiser;
    struct Deserialiser;
    using Ptr = std::unique_ptr<TranslationUnit>;

    class Target {
        LIBBASE_IMMOVABLE(Target);

        friend TranslationUnit;
        std::unique_ptr<clang::CompilerInstance> ci;
        llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI;
        Target();
        ~Target();

    public:
        [[nodiscard]] auto closure_align() const -> Align { return ptr_align(); }
        [[nodiscard]] auto closure_size() const -> Size { return 2 * ptr_size(); }
        [[nodiscard]] auto int_align() const -> Align { return ptr_align(); }
        [[nodiscard]] auto int_size() const -> Size { return ptr_size(); }
        [[nodiscard]] auto int_align(const IntType* ty) const -> Align {
            return Align(TI->getBitIntAlign(u32(ty->bit_width().bits())) / 8);
        }

        [[nodiscard]] auto int_size(const IntType* ty) const -> Size {
            return Size::Bits(TI->getBitIntWidth(u32(ty->bit_width().bits())));
        }

        [[nodiscard]] auto ptr_align() const -> Align { return Align(TI->PointerAlign / 8); }
        [[nodiscard]] auto ptr_size() const -> Size { return Size::Bits(TI->PointerWidth); }
        [[nodiscard]] auto slice_align() const -> Align { return std::max(ptr_align(), int_align()); }
        [[nodiscard]] auto slice_size() const -> Size { return ptr_size().align(int_align()) + int_size(); }
    };

private:
    /// Context that owns this module.
    Context& ctx;

    /// Target information.
    Target tgt{};

    /// Language options for this module.
    LangOpts language_opts;

    /// Files that make up this module.
    SmallVector<const File*> files;

    /// Allocators that store strings used by this module.
    ///
    /// These result from the fact that we parse files in parallel; once we’re
    /// done w/ that, files that are part of a single module get merged into a
    /// module, which is then processed all at once. At that point, we don’t need
    /// separate allocators anymore, so we just use the first of these here, but
    /// we need to hold on to all of them as they store the strings that we use
    /// while compiling the module.
    SmallVector<std::unique_ptr<llvm::BumpPtrAllocator>> allocs;
    llvm::BumpPtrAllocator alloc;

    /// Same thing, but for integers.
    SmallVector<IntegerStorage> integers;

    /// Not unique because we’ll mostly just be creating unique symbols from now on.
    llvm::StringSaver saver{alloc};

    /// Used to store evaluated constant expressions.
    SmallVector<std::unique_ptr<eval::SRValue>> evaluated_constants;

    /// All scopes in this TU.
    std::vector<std::unique_ptr<Scope>> all_scopes;

    /// If this was imported, the path to the module file.
    String import_path;

    explicit TranslationUnit(Context& ctx, const LangOpts& opts, StringRef name, bool is_module);

public:
    /// Map from module names to imported modules.
    StringMap<ImportHandle> imports;

    /// The name of this program or module.
    String name;

    /// Whether this is a program or module.
    const bool is_module;

    /// Module initialiser.
    ProcDecl* initialiser_proc{};
    BlockExpr* file_scope_block{};

    /// All procedures in the module.
    SmallVector<ProcDecl*> procs;

    /// Declarations exported from this module.
    Scope exports{nullptr};

    /// Template instantiations by template.
    DenseMap<ProcTemplateDecl*, SmallVector<ProcDecl*>> template_instantiations;

    /// Compile-time virtual machine for constant evaluation.
    eval::VM vm{*this};

    /// LLVM context for this module.
    llvm::LLVMContext llvm_context;

    /// FFI Types.
    Type FFIBoolTy;
    Type FFICharTy;
    Type FFIShortTy;
    Type FFIIntTy;
    Type FFILongTy;
    Type FFILongLongTy;
    Type FFISizeTy;

    /// Convenience accessors because they’re used often.
    Type I8Ty;
    Type I16Ty;
    Type I32Ty;
    Type I64Ty;
    Type I128Ty;
    SliceType* StrLitTy;

    /// Type caches.
    FoldingSet<ArrayType> array_types;
    FoldingSet<IntType> int_types;
    FoldingSet<PtrType> ptr_types;
    FoldingSet<ProcType> proc_types;
    FoldingSet<SliceType> slice_types;

    /// Create a new module.
    static auto Create(Context& ctx, const LangOpts& opts, StringRef name, bool is_module) -> Ptr;
    static auto CreateEmpty(Context& ctx, const LangOpts& opts) -> Ptr;

    /// Deserialise a module.
    static auto Deserialise(Context& ctx, ArrayRef<char> data) -> Ptr;

    /// Deserialise a module from an archive.
    ///
    /// \return std::nullopt if extracting the module description
    /// from the file failed. Deserialisation errors are still
    /// fatal errors!
    static auto Deserialise(
        Context& ctx,
        StringRef module_name,
        StringRef archive_path,
        Location err_loc
    ) -> Opt<Ptr>;

    /// Allocate data.
    void* allocate(usz size, usz align) { return allocator().Allocate(size, align); }

    /// Allocate an object.
    template <typename T, typename... Args>
    auto AllocateAndConstruct(Args&&... args) -> T* {
        static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
        return new (allocate(sizeof(T), alignof(T))) T{std::forward<Args>(args)...};
    }

    /// Get the module’s allocator.
    [[nodiscard]] auto allocator() -> llvm::BumpPtrAllocator& { return alloc; }

    /// Get the owning context.
    [[nodiscard]] Context& context() const { return ctx; }

    /// Add an allocator to the module.
    void add_allocator(std::unique_ptr<llvm::BumpPtrAllocator> alloc) { allocs.push_back(std::move(alloc)); }

    /// Add a file to the module.
    void add_file(const File& file) { files.push_back(&file); }

    /// Add an integer storage unit.
    void add_integer_storage(IntegerStorage&& storage) {
        integers.push_back(std::move(storage));
    }

    /// Create a new scope.
    template <typename ScopeTy = Scope, typename... Args>
    auto create_scope(Args&&... args) -> ScopeTy* {
        all_scopes.emplace_back(new ScopeTy{std::forward<Args>(args)...});
        return static_cast<ScopeTy*>(all_scopes.back().get());
    }

    /// Dump the contents of the module.
    void dump() const;

    /// Whether this is an imported module.
    bool imported() const { return initialiser_proc == nullptr; }

    /// Get the language options for this module.
    [[nodiscard]] auto lang_opts() const -> const LangOpts& { return language_opts; }

    /// The path to the file that needs to be linked against when importing this module.
    auto link_path() const -> String { return import_path; }

    /// Save a string in the module.
    auto save(StringRef s) -> String { return String::Save(saver, s); }

    /// Save a constant in the module.
    auto save(eval::SRValue val) -> eval::SRValue*;

    /// Store an integer in the module.
    auto store_int(APInt value) -> StoredInteger;

    /// Serialise this module to a memory buffer
    void serialise(SmallVectorImpl<char>& buffer) const;

    /// Get the target info.
    auto target() const -> const Target& { return tgt; }
};

#endif // SRCC_AST_HH
