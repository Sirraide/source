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

/// Virtual pointer to compile-time data.
///
/// We want to be able to pass numbers that behave like actual
/// pointers (e.g. you can cast them to integers, add 1, cast
/// them back, and then still dereference them); this means we
/// can’t have any metadata in the actual pointer, so instead
/// these ‘pointers’ are really virtual memory addresses that
/// map to various internal buffers and data.
class Pointer {
public:
    enum struct AddressSpace : u8 {
        Stack,
        Heap,
        Host,
    };

private:
    /// Pointer value.
    uptr ptr : sizeof(void*) * 8 - 2 = 0;

    /// Pointer address space.
    uptr aspace : 2 = 0;

    /// The actual numeric value that is stored is 1 + the actual
    /// offset to make sure that the null pointer is never valid;
    /// this also means that we need to subtract 1 when converting
    /// to an integer or retrieving the actual pointer.
    Pointer(uptr p, AddressSpace k) : ptr(p + 1), aspace(u8(k)) {}

public:
    Pointer() = default;

    /// Create a Pointer from its raw encoding.
    static auto FromRaw(uptr raw) -> Pointer {
        return std::bit_cast<Pointer>(raw);
    }

    /// Create a heap pointer.
    static auto Heap(usz virtual_offs) -> Pointer {
        return {virtual_offs, AddressSpace::Heap};
    }

    /// Return a pointer to host memory.
    static auto Host(usz virtual_offs) -> Pointer {
        return Pointer{virtual_offs, AddressSpace::Host};
    }

    /// Create a stack pointer.
    static auto Stack(usz stack_offs) -> Pointer {
        return {stack_offs, AddressSpace::Stack};
    }

    /// Get the address space of this pointer.
    [[nodiscard]] auto address_space() const -> AddressSpace {
        return AddressSpace(aspace);
    }

    /// Encode the pointer in a uintptr_t.
    [[nodiscard]] auto encode() const -> uptr {
        return std::bit_cast<uptr>(*this);
    }

    /// Whether this is a heap pointer.
    [[nodiscard]] bool is_heap_ptr() const {
        return address_space() == AddressSpace::Heap;
    }

    /// Whether this points to the VM stack or heap.
    [[nodiscard]] bool is_host_ptr() const {
        return address_space() == AddressSpace::Host;
    }

    /// Whether this is the null pointer.
    [[nodiscard]] bool is_null_ptr() const {
        return ptr == 0;
    }

    /// Whether this is a stack pointer.
    [[nodiscard]] bool is_stack_ptr() const {
        return address_space() == AddressSpace::Stack;
    }

    /// Get the actual pointer value.
    [[nodiscard]] auto value() const -> usz {
        return ptr - 1;
    }

    /// Offset this pointer by an integer.
    [[nodiscard]] auto operator+(usz offs) const -> Pointer {
        Pointer p{*this};
        p.ptr += offs;
        return p;
    }

    /// Compute the difference between two pointers.
    [[nodiscard]] auto operator-(Pointer other) const -> usz {
        return ptr - other.ptr;
    }

    /// Compare two pointers in the same address space.
    [[nodiscard]] auto operator<=>(const Pointer& other) const {
        Assert(address_space() == other.address_space());
        return ptr <=> other.ptr;
    }

    /// Check if two pointers are the same.
    [[nodiscard]] auto operator==(const Pointer& other) const -> bool {
        return encode() == other.encode();
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
    DenseMap<cg::ir::Proc*, void*> native_symbols;

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
    explicit SRValue(cg::ir::Proc* proc);
    explicit SRValue(bool b) : value{b}, ty{Types::BoolTy} {}
    explicit SRValue(Type ty) : value{ty}, ty{Types::TypeTy} {}
    explicit SRValue(Pointer p, Type ptr_ty) : value{p}, ty{ptr_ty} {}
    explicit SRValue(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    explicit SRValue(i64 val) : value{APInt{64, u64(val)}}, ty{Types::IntTy} {}

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
