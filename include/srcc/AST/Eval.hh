#ifndef SRCC_AST_EVAL_HH
#define SRCC_AST_EVAL_HH

#include <srcc/AST/Type.hh>
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
}

namespace srcc::eval {
class LValue;
class Reference;
class Slice;
class Value;
class Memory;
enum struct LifetimeState : u8;

/*/// Virtual machine used for constant evaluation; one of these is
/// created for every translation unit and reused across constant
/// evaluations.
class VM {
    LIBBASE_IMMOVABLE(VM);

    /// Machine word.
    using word = uptr;

    /// Address space that stores either code or data.
    class AddressSpace {
        ByteBuffer mem;

    public:
        void contains(word ptr);
    };

    /// Start of executable memory.
    static constexpr word MemExecStart = 0x10000;

    /// Start of data memory.
    ///
    /// This is 1 << 30 on 32-bit systems and 1 << 46 on 64-bit systems.
    static constexpr word MemDataStart = 1 << std::min<word>(46, Size::Of<word>().bits() - 2);

public:
    /// Maximum number of registers.
    static constexpr usz MaxRegisters =  1 << 16;

private:
    /// The tu that this vm belongs to.
    TranslationUnit& owner_tu;

    /// Executable memory; this is where procedures and constant
    /// expressions are allocated.
    AddressSpace mem_exec;

    /// Data memory; this is for constant data.
    AddressSpace mem_data;

    /// VM registers.
    std::vector<word> registers;

public:
    explicit VM(TranslationUnit& owner_tu) : owner_tu{owner_tu}, registers(MaxRegisters) {}

    /// Attempt to evaluate a statement.
    ///
    /// Callers must check dependence before this is called; attempting
    /// to evaluate a dependent statement will assert.
    ///
    /// \param stmt Statement to evaluate.
    /// \param complain Whether to emit diagnostics if the statement cannot be evaluated.
    /// \return The value of the statement, if it can be evaluated.
    [[nodiscard]] auto eval(Stmt* stmt, bool complain = true) -> std::optional<Value>;

    /// Get the translation unit that owns this vm.
    [[nodiscard]] auto owner() -> TranslationUnit& { return owner_tu; }

private:
    /// Compile a statement and return a memory buffer containing the byte code.
    auto CompileSingleStmt(Stmt* stmt, bool complain) -> std::optional<ByteBuffer>;
};*/
} // namespace srcc::eval

namespace srcc::eval {
class EvaluationContext;
}

enum struct srcc::eval::LifetimeState : base::u8 {
    Uninitialised,
    Initialised,
};

class srcc::eval::LValue {
public:
    /// Base of the lvalue. This is either:
    ///     - a string literal;
    ///     - stack or heap memory;
    Variant<String, Memory*> base;

    /// The type of this lvalue.
    Type type;

    /// The location where this was created.
    Location loc;

    /// Whether the lvalue is modifiable.
    mutable bool modifiable{true};

    LValue(String base, Type type, Location loc, bool modifiable = true)
        : base{std::move(base)},
          type{type},
          loc{loc},
          modifiable{modifiable} {}

    LValue(Memory* base, Type type, Location loc, bool modifiable = true)
        : base{base},
          type{type},
          loc{loc},
          modifiable{modifiable} {}

    bool operator==(const LValue& other) const {
        return base == other.base;
    }

    /// Make this readonly; this cannot be undone.
    void make_readonly() const { modifiable = false; }

    /// Print this.
    void dump(bool use_colour = true) const;
    auto print() const -> SmallUnrenderedString;
};

/// Memory location.
class srcc::eval::Memory {
    SRCC_IMMOVABLE(Memory);
    friend EvaluationContext;
    friend auto LValue::print() const -> SmallUnrenderedString;

    Size sz;
    llvm::PointerIntPair<void*, 1, LifetimeState> data_and_state;

public:
    explicit Memory(Size size, void* data)
        : sz{size},
          data_and_state{data, LifetimeState::Uninitialised} {}

    /// Check whether this is (still) alive.
    [[nodiscard]] auto alive() const {
        return data_and_state.getInt() == LifetimeState::Initialised;
    }

    /// Check whether this is not alive anymore.
    [[nodiscard]] auto dead() const { return not alive(); }

    /// End the lifetime of this memory location.
    void destroy();

    /// Start the lifetime of this memory location.
    void init();

    /// Get the size of this memory.
    [[nodiscard]] auto size() const -> Size { return sz; }

    /// Zero out the memory.
    void zero();

private:
    [[nodiscard]] auto data() -> void* { return data_and_state.getPointer(); }
    [[nodiscard]] auto data() const -> const void* { return data_and_state.getPointer(); }
};

class srcc::eval::Reference {
public:
    LValue lvalue;
    APInt offset;

    bool operator==(const Reference&) const = default;
};

class srcc::eval::Slice {
public:
    Reference data;
    APInt size;

    Slice(Reference data, APInt size)
        : data(std::move(data)),
          size(std::move(size)) {}

    bool operator==(const Slice&) const = default;
};

/// A compile-time value.
class srcc::eval::Value {
    friend EvaluationContext;
    friend Slice;

    Variant<ProcDecl*, LValue, Slice, Reference, APInt, bool, Type, std::monostate> value{std::monostate{}};
    Type ty{Types::VoidTy};

public:
    Value() = default;
    Value(ProcDecl* proc);
    Value(bool b) : value{b}, ty{Types::BoolTy} {}
    Value(Type ty) : value{ty}, ty{Types::TypeTy} {}
    Value(Slice lvalue, Type ty);
    Value(LValue lvalue) : value(std::move(lvalue)), ty(lvalue.type) {}
    Value(Reference ref, Type ty) : value(std::move(ref)), ty(ty) {}
    Value(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    Value(i64 val) : value{APInt{64, u64(val)}}, ty{Types::IntTy} {}

    /// Check if two values hold the same value.
    bool operator==(const Value& other) const;

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

    /// Get the value category of this value.
    auto value_category() const -> ValueCategory;

    /// Run a visitor over this value.
    template <typename Visitor>
    auto visit(Visitor&& visitor) const -> decltype(auto) {
        return std::visit(std::forward<Visitor>(visitor), value);
    }
};

template <>
struct std::formatter<srcc::eval::Value> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const srcc::eval::Value& val, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{val.print().str()}, ctx);
    }
};

namespace srcc::eval {
[[deprecated("Remove this and implement a proper VM")]]
inline std::optional<Value> Evaluate(TranslationUnit&, Stmt*, bool complain = true) {
    if (complain) std::println(stderr, "TODO: Implement VM");
    return std::nullopt;
}}

#endif // SRCC_AST_EVAL_HH
