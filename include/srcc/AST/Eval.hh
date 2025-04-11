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
namespace cg::ir {
class InstValue;
class Proc;
}
class Stmt;
class ProcDecl;
class StrLitExpr;
class TranslationUnit;
} // namespace srcc

namespace srcc::eval {
class Eval;
class SRValue;
enum struct LifetimeState : u8;

class PointerValue {
    uptr data{};

public:
    PointerValue() = default;
    PointerValue(usz stack_offs) : data(stack_offs + 1) {}

    /// Create a stack pointer from a raw pointer.
    static auto FromRawValue(uptr raw) -> PointerValue {
        PointerValue sp{};
        sp.data = raw;
        return sp;
    }

    /// Get the encoded value.
    [[nodiscard]] auto encoded() const -> uptr { return data; }

    /// Check if this is the null pointer.
    [[nodiscard]] bool is_null() const { return data == 0; }

    /// Get the value provided it isn’t null.
    [[nodiscard]] auto non_null_value() const {
        Assert(not is_null());
        return data - 1;
    }
};
} // namespace srcc::eval

template <>
struct llvm::PointerLikeTypeTraits<srcc::eval::PointerValue> {
    // We can shift the value over to free up some bits.
    static constexpr int NumLowBitsAvailable = 4;

    static void* getAsVoidPointer(srcc::eval::PointerValue p) {
        return reinterpret_cast<void*>(p.encoded() << NumLowBitsAvailable);
    }

    static auto getFromVoidPointer(void* p) -> srcc::eval::PointerValue {
        return srcc::eval::PointerValue{reinterpret_cast<srcc::uptr>(p) >> NumLowBitsAvailable};
    }
};

namespace srcc::eval {
/// Pointer to compile-time data.
///
/// This can be a stack/heap pointer (i.e. a PointerValue), which
/// is an index into one of two big arrays of bytes, or an actual
/// pointer to compiler-internal data (e.g. a string literal).
///
/// We want to be able to pass numbers that behave like actual
/// pointers (e.g. you can cast them to integers, add 1, cast
/// them back, and then still dereference them); this means we
/// can’t have any metadata in the actual pointer, so we instead
/// store a list of valid memory ranges (for the heap and compiler
/// data) in the evaluator. The stack pointer is just one contiguous
/// array.
class Pointer {
    /// The int is only used to distinguish between stack and heap pointers.
    using DataTy = llvm::PointerIntPair<llvm::PointerUnion<PointerValue, void*>, 1>;
    DataTy data;

    Pointer(PointerValue val, bool stack_pointer) : data{val, stack_pointer} {}

public:
    Pointer() = default;
    Pointer(void* ptr) : data{ptr} {}

    /// Create a Pointer from its raw encoding.
    static auto FromRaw(uptr raw) -> Pointer {
        Pointer p{};
        p.data = DataTy::getFromOpaqueValue(reinterpret_cast<void*>(raw));
        return p;
    }

    /// Create a heap pointer.
    static auto Heap(PointerValue p) -> Pointer { return {p, false}; }

    /// Create a stack pointer.
    static auto Stack(PointerValue p) -> Pointer { return {p, true}; }

    /// Encode the pointer in a uintptr_t.
    [[nodiscard]] auto encode() const -> uptr {
        return reinterpret_cast<uptr>(data.getOpaqueValue());
    }

    /// Whether this points to compiler-internal data.
    [[nodiscard]] auto internal_ptr() const -> void* {
        Assert(not is_vm_ptr());
        return cast<void*>(data.getPointer());
    }

    /// Whether this is a heap pointer.
    [[nodiscard]] bool is_heap_ptr() const {
        return is_vm_ptr() and not is_stack_ptr();
    }

    /// Whether this is a stack pointer.
    [[nodiscard]] bool is_stack_ptr() const {
        return is_vm_ptr() and bool(data.getInt());
    }

    /// Whether this points to the VM stack or heap.
    [[nodiscard]] bool is_vm_ptr() const {
        return isa<PointerValue>(data.getPointer());
    }

    /// Get the VM pointer value.
    [[nodiscard]] auto vm_ptr() const -> PointerValue {
        Assert(is_vm_ptr());
        return cast<PointerValue>(data.getPointer());
    }

    /// Offset this pointer by an integer.
    [[nodiscard]] auto operator+(usz offs) const -> Pointer {
        Pointer p{*this};
        if (is_vm_ptr()) p.data.setPointer(PointerValue::FromRawValue(vm_ptr().encoded() + offs));
        else p.data.setPointer(static_cast<char*>(internal_ptr()) + offs);
        return p;
    }

    /// Check if two pointers are the same.
    [[nodiscard]] auto operator==(const Pointer& other) const -> bool {
        return data.getOpaqueValue() == other.data.getOpaqueValue();
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

public:
    explicit VM(TranslationUnit& owner_tu);
    ~VM();

    /// Attempt to evaluate a statement.
    ///
    /// Callers must check dependence before this is called; attempting
    /// to evaluate a dependent statement will assert.
    ///
    /// \param stmt Statement to evaluate.
    /// \param complain Whether to emit diagnostics if the statement cannot be evaluated.
    /// \return The value of the statement, if it can be evaluated.
    [[nodiscard]] auto eval(Stmt* stmt, bool complain = true) -> std::optional<SRValue>;

    /// Get the translation unit that owns this vm.
    [[nodiscard]] auto owner() -> TranslationUnit& { return owner_tu; }
};
} // namespace srcc::eval

/// A compile-time srvalue.
class srcc::eval::SRValue {
    Variant<cg::ir::Proc*, APInt, bool, Type, Pointer, std::monostate> value{std::monostate{}};
    Type ty{Types::VoidTy};

public:
    SRValue() = default;
    SRValue(cg::ir::Proc* proc);
    SRValue(bool b) : value{b}, ty{Types::BoolTy} {}
    SRValue(Type ty) : value{ty}, ty{Types::TypeTy} {}
    SRValue(Pointer p, Type ptr_ty) : value{p}, ty{ptr_ty} {}
    SRValue(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    SRValue(i64 val) : value{APInt{64, u64(val)}}, ty{Types::IntTy} {}

    /// Check if two values hold the same value.
    bool operator==(const SRValue& other) const;

    /// cast<>() the contained value.
    template <typename Ty>
    auto cast() -> Ty& { return std::get<Ty>(value); }

    template <typename Ty>
    auto cast() const -> const Ty& { return std::get<Ty>(value); }

    /// dyn_cast<>() the contained value.
    template <typename Ty>
    auto dyn_cast() const -> const Ty* {
        return std::holds_alternative<Ty>(value) ? &std::get<Ty>(value) : nullptr;
    }

    /// Print this value.
    void dump(bool use_colour = true) const;
    void dump_colour() const { dump(true); }

    /// Check if the value is empty. This is also used to represent '()'.
    auto empty() const -> bool { return std::holds_alternative<std::monostate>(value); }

    /// isa<>() on the contained value.
    template <typename Ty>
    auto isa() const -> bool { return std::holds_alternative<Ty>(value); }

    /// Print the value to a string.
    auto print() const -> SmallUnrenderedString;

    /// Get the type of the value.
    auto type() const -> Type { return ty; }

    /// Run a visitor over this value.
    template <typename Visitor>
    auto visit(Visitor&& visitor) const -> decltype(auto) {
        return std::visit(std::forward<Visitor>(visitor), value);
    }
};

template <>
struct std::formatter<srcc::eval::SRValue> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const srcc::eval::SRValue& val, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{val.print().str()}, ctx);
    }
};

#endif // SRCC_AST_EVAL_HH
