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
class Sema;
class ProcDecl;
class StrLitExpr;
class TranslationUnit;
class Target;
class QuoteExpr;
} // namespace srcc

namespace mlir {
class Operation;
}

namespace mlir::LLVM {
class GlobalOp;
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
    friend Eval;

    void* storage;
    Size sz;
    bool is_str;

    RawByteBuffer(void* data, Size sz, bool is_str = false)
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

struct InvalidStackPointer{};
struct UnknownPointer{};

/// An object a pointer is based on.
using PointerBase = Variant<
    /// The null pointer.
    std::nullptr_t,

    /// A pointer to stack memory in the evaluator; such pointers cannot be
    /// emitted (as they are only valid within a single evaluation) and any
    /// attempt to do so should error.
    InvalidStackPointer,

    /// We don’t know where this pointer came from or what it points to.
    UnknownPointer,

    /// A pointer to a procedure declared in this TU or imported from a module.
    ProcDecl*,

    /// A pointer into a string.
    String
>;

/// A pointer that is the result of constant evaluation.
class EvaluatedPointer {
private:
    PointerBase base_object;
    Size offs;

public:
    /// Create a null pointer.
    EvaluatedPointer() : base_object(nullptr) {}

    /// Create a procedure pointer.
    EvaluatedPointer(ProcDecl* proc) : base_object(proc) {}

    /// Create a pointer to a global variable.
    EvaluatedPointer(String s, Size offset) : base_object(s), offs(offset) {}

    /// Create an invalid stack pointer.
    [[nodiscard]] static auto GetInvalidStack() -> EvaluatedPointer {
        EvaluatedPointer p;
        p.base_object = InvalidStackPointer{};
        return p;
    }

    /// Get an unknown pointer.
    [[nodiscard]] static auto GetUnknown() -> EvaluatedPointer {
        EvaluatedPointer p;
        p.base_object = UnknownPointer{};
        return p;
    }

    /// Get the object this pointer is based on.
    [[nodiscard]] auto base() const -> const PointerBase& { return base_object; }

    /// Check if this is an invalid pointer. Notably, the null pointer
    /// is *valid* since we can emit it (unlike e.g. an invalid stack
    /// pointer).
    [[nodiscard]] bool is_invalid() const {
        return base_object.is<InvalidStackPointer, UnknownPointer>();
    }

    /// Check if this is the null pointer.
    [[nodiscard]] bool is_null() const { return base_object.is<std::nullptr_t>(); }

    /// Check if this is a procedure pointer.
    [[nodiscard]] bool is_procedure() const { return base_object.is<ProcDecl*>(); }

    /// Get the offset added to this pointer; only valid for global constants.
    [[nodiscard]] auto offset() const -> Size { return offs; }
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

struct Nil {

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
        Nil,
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
    explicit RValue(Nil n) : value(n), ty(Type::NilTy) {}

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
    [[nodiscard]] auto allocate_memory_value(Size sz, Align a) -> RawByteBuffer;

    /// Allocate a tree.
    [[nodiscard]] auto allocate_tree_value(QuoteExpr* quote, ArrayRef<TreeValue*> unquotes) -> TreeValue*;

    /// Attempt to evaluate a statement.
    ///
    /// Callers must check dependence before this is called; attempting
    /// to evaluate a dependent statement will assert.
    ///
    /// \param sema Sema instance; may be null.
    /// \param stmt Statement to evaluate.
    /// \param complain Whether to emit diagnostics if the statement cannot be evaluated.
    /// \param dump_ir Print the IR used for evaluation, if any.
    /// \param allow_globals Allow evaluating global variables; 'true' when evaluating entire TUs.
    /// \return The value of the statement, if it can be evaluated.
    [[nodiscard]] auto eval(
        Sema* sema,
        Stmt* stmt,
        bool complain = true,
        bool dump_ir = false,
        bool allow_globals = false
    ) -> std::optional<RValue>;

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
