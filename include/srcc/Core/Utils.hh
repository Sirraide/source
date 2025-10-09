#ifndef SRCC_CORE_UTILS_HH
#define SRCC_CORE_UTILS_HH

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/TrailingObjects.h>

#include <base/Base.hh>
#include <base/Colours.hh>
#include <base/Serialisation.hh>
#include <base/Text.hh>
#include <base/Size.hh>

#include <filesystem>
#include <print>
#include <source_location>
#include <string>

static_assert(
    CHAR_BIT == 8,
    "Targets where a byte is not 8 bits are not and will not be supported"
);

namespace srcc {
using namespace base;

using llvm::cast;
using llvm::cast_if_present;
using llvm::dyn_cast;
using llvm::dyn_cast_if_present;
using llvm::isa;
using llvm::isa_and_present;

using llvm::APInt;
using llvm::ArrayRef;
using llvm::DenseMap;
using llvm::FoldingSet;
using llvm::FoldingSetNode;
using llvm::FoldingSetNodeID;
using llvm::MutableArrayRef;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringMap;
using llvm::StringRef;
using llvm::TrailingObjects;

template <typename... arguments>
void debug(std::format_string<arguments...> fmt, arguments&&... args) {
    std::print(stderr, fmt, std::forward<arguments>(args)...);
    std::print(stderr, "\n");
}

/// A wrapper around std::optional that is specialised for some types.
template <typename Ty>
class Opt : public std::optional<Ty> {
    using std::optional<Ty>::optional;

    // Disallow these because they’re unsafe.
    auto operator->() -> Ty* = delete;
    auto operator*() -> Ty& = delete;
};

/// Nullable pointer.
template <typename Ty>
class Ptr {
    Ty* value;

    template <typename T>
    friend class Ptr;

public:
    constexpr Ptr() : value{nullptr} {}
    constexpr Ptr(Ty* value) : value{value} {}
    constexpr Ptr(std::nullptr_t) : value{nullptr} {}

    template <std::derived_from<Ty> Derived>
    Ptr(Ptr<Derived> other) : value{other.value} {}

    template <typename Derived>
    auto cast() const -> Ptr<Derived> {
        if (not value) return nullptr;
        return cast<Derived>(value);
    }

    template <typename Derived>
    auto dyn_cast() const -> Ptr<Derived> {
        if (not value) return nullptr;
        return dyn_cast<Derived>(value);
    }

    auto get() const -> Ty* {
        Assert(value, "Value must be present!");
        return value;
    }

    auto get_or(Ty* def) const -> Ty* {
        return value ? value : def;
    }

    auto get_or_null() const -> Ty* { return value; }

    auto invalid() const -> bool { return not present(); }
    auto present() const -> bool { return value != nullptr; }
    explicit operator bool() const { return present(); }
};

template <typename T>
auto ref(SmallVectorImpl<T>& vec) -> MutableArrayRef<T> {
    return MutableArrayRef<T>(vec);
}

/// A string that is saved somewhere.
///
/// This is used for strings that are guaranteed to ‘live long
/// enough’ to be passed around without having to worry about who
/// owns them. This typically means they are stored in a module
/// or static storage.
///
/// NEVER return a String to outside a single driver invocation!
class String {
    StringRef val;

public:
    constexpr String() = default;

    /// Construct from a string literal.
    template <usz size>
    consteval String(const char (&arr)[size]) : val{arr} {}

    /// Construct from a string literal.
    consteval String(llvm::StringLiteral lit) : val{lit} {}

    /// Create a 'String' from a 'StringRef'.
    ///
    /// This is an unsafe operation! The caller must ensure that the
    /// underlying value lives as long as the string is going to be
    /// used and that it is null-terminated. This is intended to be
    /// used e.g. by the lexer; always prefer to obtain a 'String'
    /// by other means.
    [[nodiscard]] static constexpr auto CreateUnsafe(StringRef value) {
        String s;
        s.val = value;
        return s;
    }

    [[nodiscard]] static constexpr auto CreateUnsafe(const char* data, usz sz) {
        return CreateUnsafe(StringRef(data, sz));
    }

    /// Save it in a string saver; this is how you’re supposed to create these.
    [[nodiscard]] static auto Save(llvm::StringSaver& ss, StringRef s) {
        return CreateUnsafe(ss.save(s));
    }

    [[nodiscard]] static auto Save(llvm::UniqueStringSaver& ss, StringRef s) {
        return CreateUnsafe(ss.save(s));
    }

    /// Get an iterator to the beginning of the string.
    [[nodiscard]] constexpr auto begin() const { return val.begin(); }

    /// Get the data of the string.
    [[nodiscard]] constexpr auto data() const -> const char* { return val.data(); }

    /// Drop characters from the beginning of the string.
    [[nodiscard]] constexpr auto drop(usz n = 1) const -> String { return CreateUnsafe(val.drop_front(n)); }

    /// Drop characters from the end of the string.
    [[nodiscard]] constexpr auto drop_back(usz n = 1) const -> String { return CreateUnsafe(val.drop_back(n)); }

    /// Check if the string is empty.
    [[nodiscard]] constexpr auto empty() const -> bool { return val.empty(); }

    /// Get an iterator to the end of the string.
    [[nodiscard]] constexpr auto end() const { return val.end(); }

    /// Check if the string ends with a given suffix.
    [[nodiscard]] constexpr auto ends_with(StringRef suffix) const -> bool {
        return val.ends_with(suffix);
    }

    /// Get the size of the string.
    [[nodiscard]] constexpr auto size() const -> usz { return val.size(); }

    /// Check if the string starts with a given prefix.
    [[nodiscard]] constexpr auto starts_with(StringRef prefix) const -> bool {
        return val.starts_with(prefix);
    }

    [[nodiscard]] constexpr auto starts_with(char c) const -> bool {
        return val.starts_with(c);
    }

    /// Get the string value as a std::string_view.
    [[nodiscard]] constexpr auto sv() const -> std::string_view { return val; }

    /// Take the first n characters of the string.
    [[nodiscard]] constexpr auto take(usz n) const -> String { return CreateUnsafe(val.take_front(n)); }

    /// Get the string value.
    [[nodiscard]] constexpr auto value() const -> StringRef { return val; }

    /// Get the string value, including the null terminator.
    [[nodiscard]] constexpr auto value_with_null() const -> StringRef {
        return StringRef{val.data(), val.size() + 1};
    }

    /// Get a character at a given index.
    [[nodiscard]] constexpr auto operator[](usz idx) const -> char { return val[idx]; }

    /// Comparison operators.
    [[nodiscard]] friend auto operator==(String a, StringRef b) { return a.val == b; }
    [[nodiscard]] friend auto operator==(String a, String b) { return a.value() == b.value(); }
    [[nodiscard]] friend auto operator==(String a, const char* b) { return a.value() == b; }
    [[nodiscard]] friend auto operator<=>(String a, String b) { return a.sv() <=> b.sv(); }
    [[nodiscard]] friend auto operator<=>(String a, std::string_view b) { return a.val <=> b; }

    /// Get the string.
    [[nodiscard]] constexpr operator StringRef() const { return val; }
    [[nodiscard]] constexpr operator std::string_view() const { return val; }
    [[nodiscard]] constexpr operator str() const { return val; }
};

auto operator+=(std::string& s, String str) -> std::string&;

class StoredInteger;
class IntegerStorage {
    friend StoredInteger;
    SmallVector<std::unique_ptr<APInt>> saved;

public:
    IntegerStorage() = default;
    IntegerStorage(IntegerStorage&&) = default;
    IntegerStorage(const IntegerStorage&) = delete;
    IntegerStorage& operator=(IntegerStorage&&) = default;
    IntegerStorage& operator=(const IntegerStorage&) = delete;

    auto store_int(APInt integer) -> StoredInteger;
};

/// Class to store APInts in AST nodes.
class StoredInteger {
    friend IntegerStorage;

    // Stored inline if small, and a pointer to an APInt otherwise.
    uintptr_t data;
    u64 bits;

    StoredInteger() = default;

public:
    /// Get the value if it is small enough.
    auto inline_value() const -> std::optional<i64>;

    /// Check if the value is stored inline.
    bool is_inline() const { return bits <= 64; }

    /// Convert this to a string for printing.
    auto str(bool is_signed) const -> std::string;

    /// Get the value of this integer.
    auto value() const -> APInt;
};

// String that contains formatting characters that are yet to be rendered.
using SmallUnrenderedString = SmallString<128>;

// Strip colours from an unrendered string.
auto StripColours(const SmallUnrenderedString& s) -> std::string;
} // namespace srcc

// Rarely used helpers go here.
//
// We use the 'utils' namespace of 'base' for this to avoid creating
// ambiguity between the two.
namespace base::utils {
/// Type that is implicitly convertible to various ‘falsy’
/// values, such as 'nullptr' or 'false'.
struct Falsy {
    // Allow conversion to false.
    constexpr /* implicit */ operator bool() const { return false; }

    // Allow conversion to any nullable type.
    template <typename ty>
    requires std::convertible_to<std::nullptr_t, ty>
    constexpr /* implicit */ operator ty() const { return nullptr; }
};

/// Format string that also stores the source location of the caller.
template <typename... Args>
struct FStringWithSrcLocImpl {
    std::format_string<Args...> fmt;
    std::source_location sloc;

    consteval FStringWithSrcLocImpl(
        std::convertible_to<std::string_view> auto fmt,
        std::source_location sloc = std::source_location::current()
    ) : fmt(fmt), sloc(sloc) {}
};

/// Inhibit template argument deduction.
template <typename... Args>
using FStringWithSrcLoc = FStringWithSrcLocImpl<std::type_identity_t<Args>...>;

/// This parameter is moved-from because we’ll never use the
/// same error more than once anyway.
auto FormatError(llvm::Error& e) -> std::string;

/// Negate a predicate.
[[nodiscard]] auto Not(auto Predicate) {
    return [Predicate = std::move(Predicate)]<typename... Args>(Args&&... args) {
        return not std::invoke(Predicate, std::forward<Args>(args)...);
    };
}

/// Determine the width of a number when printed.
[[nodiscard]] auto NumberWidth(usz number, usz base = 10) -> usz;

/// Print (coloured) text.
template <typename... Args>
void Print(bool use_colours, std::format_string<Args...> fmt, Args&&... args) {
    auto s = std::format(fmt, std::forward<Args>(args)...);
    auto r = text::RenderColours(use_colours, s);
    std::print("{}", r);
}
} // namespace base::utils

template <>
struct std::hash<srcc::String> {
    auto operator()(srcc::String s) const noexcept -> size_t {
        return std::hash<std::string_view>{}(s.sv());
    }
};

template <>
struct std::formatter<llvm::Align> : formatter<srcc::usz> {
    template <typename FormatContext>
    auto format(llvm::Align align, FormatContext& ctx) const {
        return formatter<srcc::usz>::format(align.value(), ctx);
    }
};

template <>
struct std::formatter<llvm::APInt> : formatter<std::string> {
    template <typename FormatContext>
    auto format(const llvm::APInt& i, FormatContext& ctx) const {
        return formatter<std::string>::format(llvm::toString(i, 10, true), ctx);
    }
};

template <>
struct std::formatter<llvm::APSInt> : formatter<std::string> {
    template <typename FormatContext>
    auto format(const llvm::APSInt& i, FormatContext& ctx) const {
        return formatter<std::string>::format(llvm::toString(i, 10), ctx);
    }
};

template <>
struct std::formatter<srcc::String> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::String s, FormatContext& ctx) const {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

template <>
struct std::formatter<llvm::StringRef> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(llvm::StringRef s, FormatContext& ctx) const {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

template <srcc::usz n>
struct std::formatter<llvm::SmallString<n>> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(llvm::StringRef s, FormatContext& ctx) const {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

template <typename Ty>
struct std::formatter<llvm::ArrayRef<Ty>> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(llvm::ArrayRef<Ty> vals, FormatContext& ctx) const {
        std::string s{"["};
        for (auto& val : vals) s += std::format("{}, ", val);
        if (not vals.empty()) s.resize(s.size() - 2);
        s += "]";
        return formatter<std::string_view>::format(s, ctx);
    }
};

// TODO: Remove once this is part of C++26.
template <>
struct std::formatter<std::filesystem::path> : std::formatter<std::string> {
    template <typename FormatContext>
    auto format(const std::filesystem::path& path, FormatContext& ctx) const {
        return std::formatter<std::string>::format(path.string(), ctx);
    }
};

#endif // SRCC_CORE_UTILS_HH
