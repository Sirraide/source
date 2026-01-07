#ifndef SRCC_CORE_TOKEN_HH
#define SRCC_CORE_TOKEN_HH

#include <srcc/Core/Location.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <deque>
#include <utility>

namespace srcc {
enum struct Tk : u8;
struct Token;
class TokenStream;
auto StripAssignment(Tk t) -> Tk;
auto Spelling(Tk t) -> String;
}

/// A token type.
enum struct srcc::Tk : base::u8 {
#   define TOKEN(name, spelling) name,
#   include "srcc/Tokens.inc"
};

/// A token.
struct srcc::Token {
    /// The type of the token.
    Tk type = Tk::Invalid;

    /// Whether this token was produced by backslash-escaping.
    bool artificial = false;

    /// Text that this token references.
    String text{};

    /// Integer value of this token.
    APInt integer{};

    /// Source location of this token.
    SLoc location{};

    /// Create an EOF token.
    [[nodiscard]] static auto Eof() -> Token {
        Token t;
        t.type = Tk::Eof;
        return t;
    }

    /// Check if this is an end-of-file token.
    [[nodiscard]] auto eof() const -> bool { return type == Tk::Eof; }

    /// Check if this token has a certain type.
    [[nodiscard]] auto is(std::same_as<Tk> auto... t) const -> bool {
        return ((type == t) or ...);
    }

    /// Compare two tokens for equality. This only checks if their
    /// types and values are equal and ignores e.g. whether they are
    /// artificial
    [[nodiscard]] bool operator==(const Token& b);
};

/// This stores and allocates tokens.
class srcc::TokenStream {
public:
    using Buffer = std::deque<Token>;
    using Iterator = Buffer::const_iterator;
    using Range = rgs::subrange<Iterator>;

private:
    Buffer tokens;
    llvm::UniqueStringSaver saver;

public:
    /// Construct a token stream.
    explicit TokenStream(llvm::BumpPtrAllocator& alloc) : saver(alloc) {}

    /// Allocate a token.
    ///
    /// This returns a stable pointer that may be retained.
    auto allocate() -> Token* { return &tokens.emplace_back(); }

    /// Get the last token.
    [[nodiscard]] auto back() -> Token& { return tokens.back(); }
    [[nodiscard]] auto back() const -> const Token& { return tokens.back(); }

    /// Get an iterator to the beginning of the token stream.
    [[nodiscard]] auto begin() { return tokens.begin(); }
    [[nodiscard]] auto begin() const { return tokens.begin(); }

    /// Get an iterator to the end of the token stream.
    [[nodiscard]] auto end() { return tokens.end(); }
    [[nodiscard]] auto end() const { return tokens.end(); }

    /// Signal that we’re done lexing. This terminates the stream
    /// with an EOF token.
    void finish(SLoc loc) {
        if (tokens.empty() or not back().eof()) {
            auto& t = tokens.emplace_back();
            t.type = Tk::Eof;
            t.location = loc;
        }
    }

    /// Add a token to the stream.
    void push(Token t) { tokens.push_back(std::move(t)); }

    /// Remove the last token from the stream.
    void pop() { tokens.pop_back(); }

    /// Save a string in the stream.
    ///
    /// \param str The string to store.
    /// \return A stable reference to the stored string.
    auto save(StringRef str) -> String {
        return String::Save(saver, str);
    }

    /// Access a token by index.
    auto operator[](usz idx) -> Token& {
        Assert(idx < tokens.size(), "Token index out of bounds");
        return tokens[idx];
    }

    auto operator[](usz idx) const -> const Token& {
        Assert(idx < tokens.size(), "Token index out of bounds");
        return tokens[idx];
    }

    /// Get the number of tokens in the stream.
    [[nodiscard]] auto size() const -> usz { return tokens.size(); }
};

template <>
struct std::formatter<srcc::Tk> : formatter<srcc::String> {
    template <typename FormatContext>
    auto format(srcc::Tk tk, FormatContext& ctx) const {
        return formatter<srcc::String>::format(Spelling(tk), ctx);
    }
};

#endif // SRCC_CORE_TOKEN_HH
