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

#define ALLOCATE_IN_TU(TYPE)                                                  \
    void* operator new(usz) = SRCC_DELETED("Use `new (tu) { ... }` instead"); \
    void* operator new(usz size, TranslationUnit& tu) { return tu.allocate<TYPE>(); }

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
    /// These are stored here because the parse tree (as well as all of
    /// Sema) is deallocated before codegen proper, and string data needed
    /// during codegen is stored in these allocators.
    SmallVector<std::unique_ptr<llvm::BumpPtrAllocator>> allocs;
    llvm::BumpPtrAllocator alloc;

    /// Same thing, but for integers.
    SmallVector<IntegerStorage> integers;

    /// And for tokens.
    std::vector<std::unique_ptr<TokenStream>> quoted_tokens;

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

    /// All global variables in the module.
    SmallVector<GlobalDecl*> global_vars;

    /// Declarations exported from this module.
    Scope exports{nullptr};

    /// Template instantiations by template.
    DenseMap<ProcTemplateDecl*, SmallVector<ProcDecl*>> template_instantiations;

    /// Compile-time virtual machine for constant evaluation.
    eval::VM vm{*this};

    /// LLVM context for this module.
    llvm::LLVMContext llvm_context;

    /// LLVM modules to link into this TU.
    SmallVector<std::unique_ptr<llvm::Module>> link_llvm_modules;

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
    Type I8PtrTy;
    SliceType* StrLitTy;
    TupleType* SliceEquivalentTupleTy;
    TupleType* ClosureEquivalentTupleTy;
    TupleType* AbortInfoEquivalentTy;

    /// Type caches.
    FoldingSet<ArrayType> array_types;
    FoldingSet<IntType> int_types;
    FoldingSet<OptionalType> optional_types;
    FoldingSet<PtrType> ptr_types;
    FoldingSet<ProcType> proc_types;
    FoldingSet<RangeType> range_types;
    FoldingSet<SliceType> slice_types;
    FoldingSet<TupleType> tuple_types;

    /// Whether we’ve already emitted LLVM IR for this TU; we can do
    /// so only once due to the presence of additional LLVM modules that
    /// need to be merged, a process which consumes the modules.
    bool emitted_llvm = false;

    ~TranslationUnit();

    /// Create a new module.
    static auto Create(Context& ctx, const LangOpts& opts, StringRef name, bool is_module) -> Ptr;

    /// Allocate data.
    void* allocate(usz size, usz align) { return allocator().Allocate(size, align); }

    template <typename T>
    void* allocate() { return allocate(sizeof(T), alignof(T)); }

    /// Allocate an object.
    template <typename T, typename... Args>
    auto AllocateAndConstruct(Args&&... args) -> T* {
        static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
        return new (allocate<T>()) T{std::forward<Args>(args)...};
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

    /// Add tokens.
    void add_quoted_tokens(std::vector<std::unique_ptr<TokenStream>> tokens) {
        rgs::move(tokens, std::back_inserter(quoted_tokens));
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
    auto serialise() -> ByteBuffer;

    /// Get the target info.
    auto target() const -> const Target& { return *tgt; }
};

#endif // SRCC_AST_HH
