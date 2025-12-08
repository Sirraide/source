#ifndef SRCC_AST_EVAL_HH
#define SRCC_AST_EVAL_HH

#include <srcc/AST/Type.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/PointerUnion.h>
#include <base/Assert.hh>
#include <base/Macros.hh>

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
///
/// This is only used for aggregate types (structs, arrays, tuples)
/// that do not contain pointers and whose memory representation can
/// just be emitted as an array of bytes (rather than requiring us to
/// emit references to other objects).
class RawByteBuffer {
    friend VM;

    void* storage;
    Size sz;
    bool is_str;

    RawByteBuffer(void* data, Size sz, bool is_str)
        : storage(std::move(data)), sz(sz), is_str(is_str) {}

public:
    /// Access the data pointer.
    [[nodiscard]] auto data() -> void* { return storage; }
    [[nodiscard]] auto data() const -> const void* { return storage; }

    /// Whether this is a string; this only affects how this is printed.
    [[nodiscard]] bool is_string() const { return is_str; }

    /// Get the size of this allocation.
    [[nodiscard]] auto size() const -> Size { return sz; }

    /// Get this as a string.
    [[nodiscard]] auto str() const -> String {
        // These buffers are allocated in the TU, so turning them into a string is fine.
        return String::CreateUnsafe(static_cast<char*>(storage), usz(sz.bits()));
    }

    /// Compare two buffers.
    [[nodiscard]] auto operator<=>(const RawByteBuffer& other) const = default;
};

/// Evaluated range.
struct Range {
    APInt start;
    APInt end;
};

/// A pointer that is the result of constant evaluation.
class EvaluatedPointer {
public:
    enum class Kind : u8 {
        /// The null pointer.
        Null,

        /// A pointer to stack memory in the evaluator; such pointers cannot be
        /// emitted (as they are only valid within a single evaluation) and any
        /// attempt to do so should error.
        InvalidStack,

        /// A pointer to a procedure declared in this TU or imported from a module.
        Procedure,

        /// We don’t know where this pointer came from or what it points to.
        Unknown,
    };

    using enum Kind;

private:
    llvm::PointerIntPair<ProcDecl*, 2> kind_and_pointer;
    EvaluatedPointer(ProcDecl* proc, Kind k) : kind_and_pointer(proc, +k) {}

public:
    /// Create a null pointer.
    EvaluatedPointer() : EvaluatedPointer(nullptr, Null) {}

    /// Create a procedure pointer.
    EvaluatedPointer(ProcDecl* proc) : EvaluatedPointer(proc, Procedure) {}

    /// Create an invalid stack pointer.
    [[nodiscard]] static auto GetInvalidStack() -> EvaluatedPointer {
        return {nullptr, InvalidStack};
    }

    /// Get an unknown pointer.
    [[nodiscard]] static auto GetUnknown() -> EvaluatedPointer {
        return {nullptr, Unknown};
    }

    /// Get the procedure pointer if this is one.
    [[nodiscard]] auto proc() const -> ProcDecl* {
        Assert(is_procedure());
        return kind_and_pointer.getPointer();
    }

    /// Check if this is an invalid pointer. Notably, the null pointer
    /// is *valid* since we can emit it (unlike e.g. an invalid stack
    /// pointer).
    [[nodiscard]] bool is_invalid() const {
        switch (kind()) {
            case InvalidStack:
            case Unknown:
                return true;

            case Null:
            case Procedure:
                return false;
        }

        Unreachable();
    }

    /// Check if this is the null pointer.
    [[nodiscard]] bool is_null() const { return kind() == Null; }

    /// Check if this is a procedure pointer.
    [[nodiscard]] bool is_procedure() const { return kind() == Procedure; }

    /// Get the pointer kind.
    [[nodiscard]] auto kind() const -> Kind { return Kind(kind_and_pointer.getInt()); }
};

/// Evaluated slice.
struct Slice {
    EvaluatedPointer pointer;
    APInt size;
};

/// Evaluated closure.
struct Closure {
    EvaluatedPointer proc;
    EvaluatedPointer env;
};

/// A struct/tuple value.
struct Record {
    ArrayRef<RValue*> fields;
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

/// Evaluated rvalue.
class RValue {
    Variant<
        APInt,
        Range,
        Slice,
        Closure,
        Type,
        TreeValue*,
        RawByteBuffer,
        Record,
        EvaluatedPointer,
        std::monostate
    > value;
    Type ty{Type::VoidTy};

public:
    explicit RValue() = default;
    explicit RValue(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    explicit RValue(std::same_as<bool> auto val) : value(APInt(1, val ? 1 : 0)), ty(Type::BoolTy) {}
    explicit RValue(Type ty) : value(ty), ty(Type::TypeTy) {}
    explicit RValue(TreeValue* tree) : value(tree), ty(Type::TreeTy) {}
    explicit RValue(Range r, Type ty) : value(std::move(r)), ty(ty) {}
    explicit RValue(Slice s, Type ty) : value(std::move(s)), ty(ty) {}
    explicit RValue(RawByteBuffer val, Type ty) : value(val), ty(ty) {}
    explicit RValue(EvaluatedPointer p, Type ty) : value(p), ty(ty) {}
    explicit RValue(Closure c, Type ty) : value(std::move(c)), ty(ty) {}
    explicit RValue(Record r, Type ty) : value(std::move(r)), ty(ty) {}

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
    [[nodiscard]] auto allocate_memory_value(Type ty) -> RawByteBuffer;

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
