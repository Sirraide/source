#ifndef SRCC_AST_HH
#define SRCC_AST_HH

#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Core/Core.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <clang/Frontend/ASTUnit.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <base/DSA.hh>

namespace srcc {
class TranslationUnit;
class Target;
}

/// Representation of a single program or module. NOT thread-safe.
class srcc::TranslationUnit {
    SRCC_IMMOVABLE(TranslationUnit);

public:
    using Ptr = std::unique_ptr<TranslationUnit>;

private:
    /// Context that owns this module.
    Context& ctx;

    /// Clang compiler instance.
    std::unique_ptr<clang::CompilerInstance> ci;

    /// Target information.
    std::unique_ptr<Target> tgt{};

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
    SmallVector<std::unique_ptr<eval::RValue>> evaluated_constants;

    /// All scopes in this TU.
    std::vector<std::unique_ptr<Scope>> all_scopes;

    /// If this was imported, the path to the module file.
    String import_path;

    explicit TranslationUnit(Context& ctx, const LangOpts& opts, StringRef name, bool is_module);

public:
    /// Map from linkage names to imported source modules.
    StringMap<ImportedSourceModuleDecl*> linkage_imports;

    /// Map from logical names to imported modules.
    StringMap<ModuleDecl*> logical_imports;

    /// List of open modules.
    SmallVector<ModuleDecl*> open_modules;

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
    Type FFICharTy;
    Type FFIWCharTy;
    Type FFIShortTy;
    Type FFIIntTy;
    Type FFILongTy;
    Type FFILongLongTy;

    /// Convenience accessors because they’re used often.
    Type I8Ty;
    Type I16Ty;
    Type I32Ty;
    Type I64Ty;
    Type I128Ty;
    Type AbortInfoTy; // TODO: Get this from the runtime.
    Type I8PtrTy;
    SliceType* StrLitTy;
    TupleType* SliceEquivalentTupleTy;
    TupleType* ClosureEquivalentTupleTy;

    /// Type caches.
    FoldingSet<ArrayType> array_types;
    FoldingSet<IntType> int_types;
    FoldingSet<PtrType> ptr_types;
    FoldingSet<ProcType> proc_types;
    FoldingSet<RangeType> range_types;
    FoldingSet<SliceType> slice_types;
    FoldingSet<TupleType> tuple_types;

    /// The declaration of '__src_abort_info', if it exists.
    StructType* abort_info_type;

    ~TranslationUnit();

    /// Create a new module.
    static auto Create(Context& ctx, const LangOpts& opts, StringRef name, bool is_module) -> Ptr;

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

    /// Get the global scope.
    auto global_scope() const -> Scope* {
        Assert(not all_scopes.empty(), "Scopes not initialised");
        return all_scopes.front().get();
    }

    /// Get the language options for this module.
    [[nodiscard]] auto lang_opts() const -> const LangOpts& { return language_opts; }

    /// The path to the file that needs to be linked against when importing this module.
    auto link_path() const -> String { return import_path; }

    /// Save a string in the module.
    auto save(StringRef s) -> String { return String::Save(saver, s); }

    /// Save a constant in the module.
    auto save(eval::RValue val) -> eval::RValue*;

    /// Store an integer in the module.
    auto store_int(APInt value) -> StoredInteger;

    /// Serialise this module.
    auto serialise() -> SmallString<0>;

    /// Get the target info.
    auto target() const -> const Target& { return *tgt; }
};

#endif // SRCC_AST_HH
