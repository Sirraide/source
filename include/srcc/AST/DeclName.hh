#ifndef SRCC_AST_DECL_NAME_HH
#define SRCC_AST_DECL_NAME_HH

#include <srcc/Core/Core.hh>
#include <srcc/Core/Token.hh>

namespace srcc {
/// Represents the name of a declaration; this can also be an operator
/// name. This behaves like a 'Variant<String, Tk>'.
class DeclName  {
    template <typename, typename>
    friend struct llvm::DenseMapInfo;

    usz is_string : 1 = true;
    usz opaque_value : sizeof(usz) * CHAR_BIT - 1 = 0;
    const char* ptr = nullptr;

public:
    DeclName() {}
    DeclName(String s) : opaque_value{s.size()}, ptr{s.data()} {}
    DeclName(Tk t) : is_string{false}, opaque_value{usz(t)} {}

    /// Check if this is the empty DeclName.
    [[nodiscard]] bool empty() const { return *this == DeclName(); }

    /// Check if this is an operator.
    [[nodiscard]] bool is_operator_name() const { return not is_string; }

    /// Check if this is a string.
    [[nodiscard]] bool is_str() const { return is_string; }

    /// Get this name as an operator.
    [[nodiscard]] auto operator_name() const -> Tk {
        Assert(is_operator_name());
        return Tk(opaque_value);
    }

    /// Get this as a string. It is valid to call this even if this is an operator name.
    [[nodiscard]] auto str() const -> String;

private:
    /// Equality comparison.
    friend bool operator==(DeclName, DeclName);
};


template <typename T>
using DeclNameMap = DenseMap<DeclName, T>;
}

template <>
struct llvm::DenseMapInfo<srcc::DeclName> {
    static constexpr srcc::DeclName getEmptyKey() {
        srcc::DeclName d;
        d.ptr = reinterpret_cast<const char*>(~0zu);
        return d;
    }

    static constexpr srcc::DeclName getTombstoneKey() {
        srcc::DeclName d;
        d.ptr = reinterpret_cast<const char*>(~1zu);
        return d;
    }

    static unsigned getHashValue(srcc::DeclName v) {
        using Tk = std::underlying_type_t<srcc::Tk>;
        if (v.is_operator_name()) return DenseMapInfo<Tk>::getHashValue(Tk(v.opaque_value));
        return DenseMapInfo<StringRef>::getHashValue(StringRef(v.ptr, v.opaque_value));
    }

    static bool isEqual(srcc::DeclName lhs, srcc::DeclName rhs) {
        return lhs == rhs;
    }
};

template <>
struct std::formatter<srcc::DeclName> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const srcc::DeclName& n, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(n.str().sv(), ctx);
    }
};

#endif // SRCC_AST_DECL_NAME_HH
