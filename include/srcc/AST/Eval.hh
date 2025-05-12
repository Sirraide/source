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
class VM;

/// Evaluated mrvalue.
///
/// This is non-owning. The actual data is allocated permanently
/// in the TranslationUnit allocator.
class MRValue {
    friend VM;

    void* storage;
    Size sz;

    MRValue(void* data, Size sz)
        : storage(std::move(data)), sz(sz) {}

public:
    [[nodiscard]] auto data() -> void* { return storage; }
    [[nodiscard]] auto size() const -> Size { return sz; }
    [[nodiscard]] auto operator<=>(const MRValue& other) const = default;
};

/// Evaluated range.
struct Range {
    APInt start;
    APInt end;

    Range(APInt start,  APInt end)
        : start(std::move(start)), end(std::move(end)) {}

    [[nodiscard]] bool operator==(const Range& other) const {
        // operator== asserts that the bit width are the same, so we
        // don’t use it directly here.
        auto Eq = [](const APInt& a, const APInt& b) {
            return a.getBitWidth() == b.getBitWidth() and a == b;
        };
        return Eq(start, other.start) and Eq(end, other.end);
    }
};

/// Evaluated rvalue.
class RValue {
    Variant<APInt, Range, bool, Type, MRValue, std::monostate> value;
    Type ty{Type::VoidTy};

public:
    RValue() = default;
    RValue(APInt val, Type ty) : value(std::move(val)), ty(ty) {}
    RValue(bool val) : value(val), ty(Type::BoolTy) {}
    RValue(Type ty) : value(ty), ty(Type::TypeTy) {}
    RValue(MRValue val, Type ty) : value(val), ty(ty) {}
    RValue(Range range,  Type ty) : value(range), ty(ty) {}

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

    /// isa<>() on the contained value.
    template <typename Ty>
    auto isa() const -> bool { return std::holds_alternative<Ty>(value); }

    /// Print the value to a string.
    auto print() const -> SmallUnrenderedString;

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
    DenseMap<cg::ir::Proc*, void*> native_symbols;

public:
    explicit VM(TranslationUnit& owner_tu);
    ~VM();

    /// Allocate an mrvalue.
    [[nodiscard]] auto allocate_mrvalue(Type ty) -> MRValue;

    /// Attempt to evaluate a statement.
    ///
    /// Callers must check dependence before this is called; attempting
    /// to evaluate a dependent statement will assert.
    ///
    /// \param stmt Statement to evaluate.
    /// \param complain Whether to emit diagnostics if the statement cannot be evaluated.
    /// \return The value of the statement, if it can be evaluated.
    [[nodiscard]] auto eval(Stmt* stmt, bool complain = true) -> std::optional<RValue>;

    /// Get the translation unit that owns this vm.
    [[nodiscard]] auto owner() -> TranslationUnit& { return owner_tu; }
};
} // namespace srcc::eval



#endif // SRCC_AST_EVAL_HH
