#ifndef SRCC_AST_EVAL_HH
#define SRCC_AST_EVAL_HH

#include <srcc/AST/Type.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/PointerIntPair.h>

#include <memory>
#include <optional>
#include <variant>

namespace srcc {
class Stmt;
class ProcDecl;
class StrLitExpr;
class TranslationUnit;
class Target;
class QuoteExpr;
} // namespace srcc

namespace mlir {
class Operation;
}

namespace srcc::eval {
class Eval;
class VM;
class VirtualMemoryMap;
class RValue;

/// Memory holding the result of constant evaluation.
///
/// This is non-owning. The actual data is allocated permanently
/// in the TranslationUnit allocator.
class MemoryValue {
    friend VM;

    void* storage;
    Size sz;

    MemoryValue(void* data, Size sz)
        : storage(std::move(data)), sz(sz) {}

public:
    /// Access the data pointer.
    [[nodiscard]] auto data() -> void* { return storage; }
    [[nodiscard]] auto data() const -> const void* { return storage; }

    /// Get the size of this allocation.
    [[nodiscard]] auto size() const -> Size { return sz; }
    [[nodiscard]] auto operator<=>(const MemoryValue& other) const = default;
};

/// Evaluated range.
struct Range {
    APInt start;
    APInt end;
};

/// Evaluated slice.
struct Slice {
    RValue* pointer;
    APInt size;
};

/// Evaluated '#quote' with all unquotes substituted.
class TreeValue final : llvm::TrailingObjects<TreeValue, TreeValue*> {
    friend VM;
    friend TrailingObjects;

    QuoteExpr* const q;

    TreeValue(QuoteExpr* pattern, ArrayRef<TreeValue*> unquotes);

public:
    [[nodiscard]] auto dump(const Context* ctx = nullptr) const -> SmallUnrenderedString;
    [[nodiscard]] auto pattern() const -> QuoteExpr* { return q; }
    [[nodiscard]] auto unquotes() const -> ArrayRef<TreeValue*>;
};

/// Tag used to indicate a pointer to stack memory in the evaluator; such
/// pointers cannot be emitted (as they are only valid within a single
/// evaluation) and any attempt to do so should error.
struct InvalidStackPointer {};

/// Evaluated rvalue.
class RValue {
    Variant<
        APInt,
        Range,
        Slice,
        Type,
        TreeValue*,
        MemoryValue,
        InvalidStackPointer,
        std::monostate
    > value;
    Type ty{Type::VoidTy};

public:
    RValue() = default;
    RValue(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    RValue(bool val) : value(APInt(1, val ? 1 : 0)), ty(Type::BoolTy) {}
    RValue(Type ty) : value(ty), ty(Type::TypeTy) {}
    RValue(TreeValue* tree) : value(tree), ty(Type::TreeTy) {}
    RValue(Range r, Type ty) : value(std::move(r)), ty(ty) {}
    RValue(Slice s, Type ty) : value(std::move(s)), ty(ty) {}
    RValue(MemoryValue val, Type ty) : value(val), ty(ty) {}
    RValue(InvalidStackPointer sp, Type ty) : value(sp), ty(ty) {}

    /// cast<>() the contained value.
    template <typename Ty>
    auto cast() -> Ty& { return std::get<Ty>(value); }

    template <typename Ty>
    auto cast() const -> const Ty& { return std::get<Ty>(value); }

    /// Dump this value to stderr.
    void dump() const;

    /// dyn_cast<>() the contained value.
    template <typename Ty>
    auto dyn_cast() const -> const Ty* {
        return std::holds_alternative<Ty>(value) ? &std::get<Ty>(value) : nullptr;
    }

    /// isa<>() on the contained value.
    template <typename Ty>
    auto isa() const -> bool { return std::holds_alternative<Ty>(value); }

    /// Print the value to a string.
    auto print(const Context* ctx = nullptr) const -> SmallUnrenderedString;

    /// Get the type of this value.
    [[nodiscard]] auto type() const -> Type { return ty; }

    /// Run a visitor over this value.
    template <typename Visitor>
    auto visit(Visitor&& visitor) const -> decltype(auto) {
        return std::visit(std::forward<Visitor>(visitor), value);
    }
};

/// Virtual machine used for constant evaluation; one of these is
/// created for every translation unit and reused across constant
/// evaluations.
class VM {
    LIBBASE_IMMOVABLE(VM);
    friend Eval;

    /// The tu that this vm belongs to.
    TranslationUnit& owner_tu;

    /// Symbol table for native procedures.
    DenseMap<mlir::Operation*, void*> native_symbols;

    /// Whether FFI calls are supported.
    bool supports_ffi_calls = false;

    /// Whether init() has been called.
    bool initialised = false;

    /// Whether we’re currently performing constant evaluation.
    bool evaluating = false;

    /// Virtual memory manager.
    std::unique_ptr<VirtualMemoryMap> memory;

public:
    explicit VM(TranslationUnit& owner_tu);
    ~VM();

    /// Allocate an mrvalue.
    [[nodiscard]] auto allocate_memory_value(Type ty) -> MemoryValue;

    /// Allocate a tree.
    [[nodiscard]] auto allocate_tree_value(QuoteExpr* quote, ArrayRef<TreeValue*> unquotes) -> TreeValue*;

    /// Attempt to evaluate a statement.
    ///
    /// Callers must check dependence before this is called; attempting
    /// to evaluate a dependent statement will assert.
    ///
    /// \param stmt Statement to evaluate.
    /// \param complain Whether to emit diagnostics if the statement cannot be evaluated.
    /// \param dump_ir Print the IR used for evaluation, if any.
    /// \return The value of the statement, if it can be evaluated.
    [[nodiscard]] auto eval(Stmt* stmt, bool complain = true, bool dump_ir = false) -> std::optional<RValue>;

    /// Initialise the VM.
    void init(const Target& tgt);

    /// Get the translation unit that owns this vm.
    [[nodiscard]] auto owner() -> TranslationUnit& { return owner_tu; }
};
} // namespace srcc::eval

namespace srcc {
using eval::TreeValue;
}

#endif // SRCC_AST_EVAL_HH
