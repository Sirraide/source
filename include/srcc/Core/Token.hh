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

enum struct srcc::Tk : base::u8 {
    Invalid,
    Eof,
    Identifier,
    TemplateType,
    CXXHeaderName,
    Integer,
    StringLiteral,
    Comment,

    /// Keywords.
    Alias,
    And,
    As,
    AsBang,
    Asm,
    Assert,
    Bool,
    Break,
    Continue,
    Defer,
    Delete,
    Do,
    Dynamic,
    Elif,
    Else,
    Enum,
    Eval,
    Export,
    F32,
    F64,
    Fallthrough,
    False,
    For,
    ForReverse,
    Goto,
    If,
    Import,
    In,
    Init,
    Int,
    IntegerType,
    Is,
    Loop,
    Match,
    Nil,
    NoReturn,
    Not,
    Or,
    Pragma,
    Proc,
    Quote,
    Range,
    Return,
    Static,
    Struct,
    Then,
    Tree,
    True,
    Try,
    Type,
    Typeof,
    Unreachable,
    Val,
    Var,
    Variant,
    Void,
    Where,
    While,
    With,
    Xor,

    /// Punctuation.
    Semicolon,
    Colon,
    ColonColon,
    Comma,
    Dollar,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Ellipsis,
    Dot,
    LArrow,
    RArrow,
    RDblArrow,
    Question,

    /// Operators.
    Plus,
    PlusTilde,
    Minus,
    MinusTilde,
    Star,
    StarTilde,
    Slash,
    ColonSlash,
    Percent,
    ColonPercent,
    Caret,
    Ampersand,
    VBar,
    Tilde,
    Hash,
    Bang,
    Assign,
    DotDot,
    DotDotLess,
    DotDotEq,
    MinusMinus,
    PlusPlus,
    StarStar,
    SLt,
    SLe,
    SGt,
    SGe,
    Swap,
    ULt,
    ULe,
    UGt,
    UGe,
    EqEq,
    Neq,
    PlusEq,
    PlusTildeEq,
    MinusEq,
    MinusTildeEq,
    StarEq,
    StarTildeEq,
    SlashEq,
    PercentEq,
    ShiftLeft,
    ShiftLeftLogical,
    ShiftRight,
    ShiftRightLogical,
    ShiftLeftEq,
    ShiftLeftLogicalEq,
    ShiftRightEq,
    ShiftRightLogicalEq,
    StarStarEq,
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
    std::deque<Token> tokens;
    llvm::UniqueStringSaver saver;

public:
    using iterator = decltype(tokens)::iterator;

    /// Construct a token stream.
    explicit TokenStream(llvm::BumpPtrAllocator& alloc) : saver(alloc) {}

    /// Allocate a token.
    ///
    /// This returns a stable pointer that may be retained.
    auto allocate() -> Token* { return &tokens.emplace_back(); }

    /// Get the last token.
    [[nodiscard]] auto back() -> Token& { return tokens.back(); }

    /// Get an iterator to the beginning of the token stream.
    [[nodiscard]] auto begin() { return tokens.begin(); }

    /// Get an iterator to the end of the token stream.
    [[nodiscard]] auto end() { return tokens.end(); }

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
