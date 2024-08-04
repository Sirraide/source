module;

#include <llvm/ADT/StringExtras.h>
#include <print>
#include <srcc/Macros.hh>

module srcc.frontend.parser;
import srcc.frontend.token;
import srcc.utils;
using namespace srcc;

// ===========================================================================
//  Lexer — Helpers and Data.
// ===========================================================================
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return llvm::isAlpha(c) or c == '_' or c == '$';
}

/// Check if a character is allowed in an identifier.
constexpr bool IsContinue(char c) {
    return IsStart(c) or llvm::isDigit(c) or c == '!';
}

constexpr bool IsBinary(char c) { return c == '0' or c == '1'; }
constexpr bool IsDecimal(char c) { return c >= '0' and c <= '9'; }
constexpr bool IsOctal(char c) { return c >= '0' and c <= '7'; }
constexpr bool IsHex(char c) { return llvm::isHexDigit(c); }

// All keywords.
const StringMap<Tk> keywords = {
    {"alias", Tk::Alias},
    {"and", Tk::And},
    {"as!", Tk::AsBang},
    {"as", Tk::As},
    {"asm", Tk::Asm},
    {"assert", Tk::Assert},
    {"bool", Tk::Bool},
    {"break", Tk::Break},
    {"continue", Tk::Continue},
    {"delete", Tk::Delete},
    {"defer", Tk::Defer},
    {"do", Tk::Do},
    {"dynamic", Tk::Dynamic},
    {"elif", Tk::Elif},
    {"else", Tk::Else},
    {"enum", Tk::Enum},
    {"eval", Tk::Eval},
    {"export", Tk::Export},
    {"f32", Tk::F32},
    {"f64", Tk::F64},
    {"fallthrough", Tk::Fallthrough},
    {"false", Tk::False},
    {"for", Tk::For},
    {"goto", Tk::Goto},
    {"if", Tk::If},
    {"import", Tk::Import},
    {"in", Tk::In},
    {"init", Tk::Init},
    {"int", Tk::Int},
    {"is", Tk::Is},
    {"land", Tk::Land},
    {"lor", Tk::Lor},
    {"match", Tk::Match},
    {"noreturn", Tk::NoReturn},
    {"not", Tk::Not},
    {"or", Tk::Or},
    {"pragma", Tk::Pragma},
    {"proc", Tk::Proc},
    {"return", Tk::Return},
    {"static", Tk::Static},
    {"struct", Tk::Struct},
    {"then", Tk::Then},
    {"true", Tk::True},
    {"try", Tk::Try},
    {"type", Tk::Type},
    {"typeof", Tk::Typeof},
    {"unreachable", Tk::Unreachable},
    {"val", Tk::Val},
    {"var", Tk::Var},
    {"variant", Tk::Variant},
    {"void", Tk::Void},
    {"while", Tk::While},
    {"with", Tk::With},
    {"xor", Tk::Xor},
    {"__srcc_ffi_char", Tk::CChar},
    {"__srcc_ffi_char16", Tk::CChar16T},
    {"__srcc_ffi_char32", Tk::CChar32T},
    {"__srcc_ffi_int", Tk::CInt},
    {"__srcc_ffi_long", Tk::CLong},
    {"__srcc_ffi_longdouble", Tk::CLongDouble},
    {"__srcc_ffi_longlong", Tk::CLongLong},
    {"__srcc_ffi_short", Tk::CShort},
    {"__srcc_ffi_size_t", Tk::CSizeT},
    {"__srcc_ffi_wchar", Tk::CWCharT},
};

// ========================================================================
//  Main lexer implementation.
// ========================================================================
struct Lexer {
    TokenStream& tokens;
    const File& f;
    Parser::CommentTokenCallback comment_token_handler;
    const char* curr;
    const char* const end;
    char lastc;
    bool raw_mode = false;

    Lexer(TokenStream& into, const File& f, Parser::CommentTokenCallback cb);

    auto tok() -> Token& { return tokens.back(); }

    template <typename... Args>
    bool Error(Location where, std::format_string<Args...> fmt, Args&&... args) {
        f.context().diags().diag(Diagnostic::Level::Error, where, fmt, std::forward<Args>(args)...);
        return false;
    }

    bool Eat(std::same_as<char> auto... cs) {
        // Remove the pointless bool() cast once Clang bug #101863 is fixed.
        if ((bool(lastc == cs) or ...)) {
            NextChar();
            return true;
        }

        return false;
    }

    auto CurrOffs() -> u32;
    auto CurrLoc() -> Location;
    void HandleCommentToken();
    void LexCXXHeaderName();
    void LexEscapedId();
    void LexIdentifier(char first_char);
    bool LexNumber(char first_char);
    void LexString(char delim);
    void Next();
    void NextChar();
    void NextImpl();
    void SkipWhitespace();
};

void Parser::ReadTokens(const File& file, CommentTokenCallback cb) {
    Lexer(stream, file, std::move(cb));
    tok = stream.begin();
}

Lexer::Lexer(TokenStream& into, const File& f, Parser::CommentTokenCallback cb)
    : tokens{into},
      f{f},
      comment_token_handler{std::move(cb)},
      curr{f.data()},
      end{curr + f.size()} {
    Assert(f.size() <= std::numeric_limits<u32>::max(), "We can’t handle files this big right now");
    NextChar();
    do Next();
    while (not tok().eof());
    if (not tokens.back().eof()) tokens.allocate()->type = Tk::Eof;
}

auto Lexer::CurrLoc() -> Location { return {CurrOffs(), 1, u16(f.file_id())}; }
auto Lexer::CurrOffs() -> u32 { return u32(curr - f.data()) - 1; }

void Lexer::Next() {
    tokens.allocate();
    NextImpl();
}

void Lexer::NextChar() {
    if (curr == end) {
        lastc = 0;
        return;
    }

    lastc = *curr++;

    // Collapse CR LF and LF CR to a single newline,
    // but keep CR CR and LF LF as two newlines.
    if (lastc == '\r' || lastc == '\n') {
        // Two newlines in a row.
        if (curr != end && (*curr == '\r' || *curr == '\n')) {
            bool same = lastc == *curr;
            lastc = '\n';

            // CR CR or LF LF.
            if (same) return;

            // CR LF or LF CR.
            curr++;
        }

        // Either CR or LF followed by something else.
        lastc = '\n';
    }
}

void Lexer::NextImpl() {
    tok().location.file_id = u16(f.file_id());

    // Tokens are not artificial by default.
    tok().artificial = false;

    // Skip whitespace.
    SkipWhitespace();

    // Keep returning EOF if we're at EOF.
    if (lastc == 0) {
        tok().type = Tk::Eof;

        // Fudge the location to be *something* valid.
        tok().location.pos = u32(f.size() - 1);
        tok().location.len = 1;
        return;
    }

    // Reset the token. We set the token type to 'invalid' here so that,
    // if we encounter an error, we can just issue a diagnostic and return
    // without setting the token type. The parser will then stop because
    // it encounters an invalid token.
    auto& ty = tok().type = Tk::Invalid;
    tok().artificial = false;
    tok().location.pos = CurrOffs();
    tok().location.len = 1;

    // Lex the token.
    //
    // Warning: Ternary abuse incoming.
    switch (auto c = lastc; NextChar(), c) {
        // Single-character tokens.
        case ';': ty = Tk::Semicolon; break;
        case ',': ty = Tk::Comma; break;
        case '?': ty = Tk::Question; break;
        case '(': ty = Tk::LParen; break;
        case ')': ty = Tk::RParen; break;
        case '[': ty = Tk::LBrack; break;
        case ']': ty = Tk::RBrack; break;
        case '{': ty = Tk::LBrace; break;
        case '}': ty = Tk::RBrace; break;
        case '^': ty = Tk::Caret; break;
        case '&': ty = Tk::Ampersand; break;
        case '|': ty = Tk::VBar; break;
        case '~': ty = Tk::Tilde; break;

        // Two-character tokens.
        case ':': ty = Eat(':') ? Tk::ColonColon : Tk::Colon; break;
        case '%': ty = Eat('=') ? Tk::PercentEq : Tk::Percent; break;
        case '!': ty = Eat('=') ? Tk::Neq : Tk::Bang; break;

        // Multi-character tokens.
        case '.':
            ty = not Eat('.') ? Tk::Dot
               : Eat('.')     ? Tk::Ellipsis
               : Eat('<')     ? Tk::DotDotLess
               : Eat('=')     ? Tk::DotDotEq
                              : Tk::DotDot;

            break;

        case '-':
            ty = Eat('>') ? Tk::RArrow
               : Eat('-') ? Tk::MinusMinus
               : Eat('=') ? Tk::MinusEq
                          : Tk::Minus;
            break;

        case '+':
            ty = Eat('+') ? Tk::PlusPlus
               : Eat('=') ? Tk::PlusEq
                          : ty = Tk::Plus;
            break;

        case '*':
            ty = Eat('=') ? Tk::StarEq
               : Eat('*') ? (Eat('=') ? Tk::StarStarEq : Tk::StarStar)
                          : Tk::Star;
            break;

        case '=':
            ty = Eat('=') ? Tk::EqEq
               : Eat('>') ? Tk::RDblArrow
                          : Tk::Assign;
            break;

        case '>':
            ty = Eat('=')     ? Tk::Ge
               : not Eat('>') ? Tk::Gt
               : Eat('>')     ? (Eat('=') ? Tk::ShiftRightLogicalEq : Tk::ShiftRightLogical)
               : Eat('=')     ? Tk::ShiftRightEq
                              : Tk::ShiftRight;
            break;

        // Complex tokens.
        case '/':
            if (Eat('/')) {
                HandleCommentToken();
                NextImpl();
                return;
            }

            ty = Eat('=') ? Tk::SlashEq : Tk::Slash;
            break;

        case '<':
            // Handle C++ header names.
            if (tokens.size() > 1 and tokens[tokens.size() - 2].type == Tk::Import) {
                LexCXXHeaderName();
                break;
            }

            ty = Eat('=') ? Tk::Le
               : Eat('<') ? (Eat('=') ? Tk::ShiftLeftEq : Tk::ShiftLeft)
               : Eat('-') ? Tk::LArrow
                          : Tk::Lt;
            break;

        case '\\':
            LexEscapedId();
            return;

        case '"':
        case '\'':
            LexString(c);
            break;

        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            // Skip the rest of the broken literal if this fails.
            if (not LexNumber(c))
                while (llvm::isAlnum(lastc))
                    NextChar();
            break;

        default:
            if (IsStart(c)) LexIdentifier(c);
            else {
                Error(CurrLoc() << 1, "Unexpected <U+{:X}> character in program", lastc);
                break;
            }
    }

    // Set the end of the token.
    tok().location.len = u16(u64(curr - f.data()) - tok().location.pos - 1);
    if (curr == end and not lastc) tok().location.len++;
}

void Lexer::HandleCommentToken() {
    const auto KeepSkipping = [&] { return lastc != '\n' && lastc != 0; };

    // If we were asked to lex comment tokens, do so and dispatch it. To
    // keep the parser from having to deal with these, they never enter
    // the token stream.
    if (comment_token_handler) {
        Token token{Tk::Comment};
        std::string text;
        while (KeepSkipping()) {
            text += lastc;
            NextChar();
        }

        token.text = tokens.save(text);
        token.location = tok().location;
        token.location.len = u16(text.size()) + 2;
        comment_token_handler(token);
        return;
    }

    // Otherwise, just skip past the comment.
    while (KeepSkipping()) NextChar();
}

void Lexer::SkipWhitespace() {
    while (std::isspace(lastc)) NextChar();
}

void Lexer::LexIdentifier(char first_char) {
    tok().type = Tk::Identifier;
    SmallString<32> text;
    text += first_char;
    while (IsContinue(lastc)) {
        text += lastc;
        NextChar();
    }
    tok().text = tokens.save(text);

    // Helper to parse keywords and integer types.
    const auto LexSpecialToken = [&] {
        if (auto k = keywords.find(tok().text); k != keywords.end()) {
            tok().type = k->second;

            // Handle "for~".
            if (tok().type == Tk::For and Eat('~')) tok().type = Tk::ForReverse;
        } else if (tok().text.starts_with("i")) {
            // Note: this returns true on error.
            if (not StringRef(tok().text).substr(1).getAsInteger(10, tok().integer))
                tok().type = Tk::IntegerType;
        }
    };

    // In raw mode, special processing is disabled. This is used for
    // parsing the argument and expansion lists of macros, as well as
    // for handling __id.
    if (raw_mode) return LexSpecialToken();

    // TODO: If we decide to support lexer macros, check for macro
    // definitions and expansions here.

    // Check for keywords and ints.
    LexSpecialToken();
}

bool Lexer::LexNumber(char first_char) {
    SmallString<64> buf;
    buf += first_char;
    char delim = lastc;

    // Helper function that actually parses a number.
    auto LexNumberImpl = [&](bool pred(char), unsigned base = 10) -> bool {
        auto DiagnoseInvalidLiteral = [&] {
            auto kind = [=] -> std::string_view {
                switch (base) {
                    case 2: return "binary";
                    case 8: return "octal";
                    case 10: return "decimal";
                    case 16: return "hexadecimal";
                    default: Unreachable("Invalid base: {}", base);
                }
            }();
            return Error(CurrLoc(), "Invalid digit '{}' in {} integer literal", lastc, kind);
        };

        // Need at least one digit.
        if (base != 10 and not pred(lastc)) {
            // If this is not even any sort of digit, then issue
            // a more helpful error; this is so we don’t complain
            // about e.g. ‘;’ not being a digit.
            if (not llvm::isAlnum(lastc))
                return Error(CurrLoc(), "Expected at least one digit after '{}'", delim);
            return DiagnoseInvalidLiteral();
        }

        // Parse the literal.
        while (pred(lastc)) {
            buf += lastc;
            NextChar();
        }

        // The next character must not be a start character.
        if (IsStart(lastc)) return DiagnoseInvalidLiteral();

        // We have a valid integer literal!
        tok().type = Tk::Integer;

        // Note: This returns true on error!
        tok().integer = APInt{};
        Assert(not buf.str().getAsInteger(base, tok().integer));
        return true;
    };

    // If the first character is not 0, then we have a decimal literal.
    if (first_char != '0') return LexNumberImpl(IsDecimal);
    if (Eat('x', 'X')) return LexNumberImpl(IsHex, 16);
    if (Eat('o', 'O')) return LexNumberImpl(IsOctal, 8);
    if (Eat('b', 'B')) return LexNumberImpl(IsBinary, 2);

    // Leading 0’s must be followed by one of the above.
    if (llvm::isDigit(lastc)) return Error(
        CurrLoc() << 1,
        "Leading zeros are not allowed in integers. Use 0o/0O for octal literals"
    );

    // Integer literal must be a literal 0.
    if (IsStart(lastc)) return Error(
        CurrLoc() <<= 1,
        "Invalid character in integer literal: '{}'",
        lastc
    );

    // If we get here, this must be a literal 0.
    tok().type = Tk::Integer;
    tok().integer = 0;
    return true;
}

void Lexer::LexString(char delim) {
    SmallString<32> text;

    // Lex the string. If it’s a raw string, we don’t need to
    // do any escaping.
    if (delim == '\'') {
        while (lastc != delim && lastc != 0) {
            text += lastc;
            NextChar();
        }
    }

    // Otherwise, we also need to replace escape sequences.
    else if (delim == '"') {
        while (lastc != delim && lastc != 0) {
            if (Eat('\\')) {
                switch (lastc) {
                    case 'a': text += '\a'; break;
                    case 'b': text += '\b'; break;
                    case 'e': text += '\033'; break;
                    case 'f': text += '\f'; break;
                    case 'n': text += '\n'; break;
                    case 'r': text += '\r'; break;
                    case 't': text += '\t'; break;
                    case 'v': text += '\v'; break;
                    case '\\': text += '\\'; break;
                    case '\'': text += '\''; break;
                    case '"': text += '"'; break;
                    case '0': text += '\0'; break;
                    default:
                        Error({tok().location, CurrLoc()}, "Invalid escape sequence");
                        return;
                }
            } else {
                text += lastc;
            }
            NextChar();
        }
    }

    // Other string delimiters are invalid.
    else {
        Error(CurrLoc() << 1, "Invalid delimiter: {}", delim);
        return;
    }

    // Make sure we actually have a delimiter.
    if (lastc != delim) {
        Error(CurrLoc() << 1, "Unterminated string literal");
        return;
    }

    tok().text = tokens.save(text);
    NextChar();

    // This is a valid string.
    tok().type = Tk::StringLiteral;
}

void Lexer::LexCXXHeaderName() {
    tempset raw_mode = true;
    tok().type = Tk::CXXHeaderName;

    SmallString<32> text;
    while (lastc != '>' and lastc != 0) {
        text.push_back(lastc);
        NextChar();
    }

    if (lastc == 0) {
        Error(tok().location, "Expected '>'");
        return;
    }

    tok().text = tokens.save(text);

    // Bring the lexer back into sync.
    NextChar();
}

void Lexer::LexEscapedId() {
    // Yeet backslash.
    tempset raw_mode = true;
    auto start = tok().location;
    NextImpl();

    // Mark this token as ‘artificial’. This is so we can e.g. nest
    // macro definitions using `\expands` and `\endmacro`.
    tok().artificial = true;

    // If the next token is anything other than "(", then it becomes the name.
    if (tok().type != Tk::LParen) {
        tok().type = Tk::Identifier;
        tok().text = tok().spelling();
        tok().location = {start, tok().location};
        return;
    }

    // If the token is "(", then everything up to the next ")" is the name.
    tok().type = Tk::Identifier;
    SmallString<32> text;
    while (lastc != ')' and lastc != 0) {
        text += lastc;
        NextChar();
    }

    // EOF.
    if (lastc == 0) {
        Error(start, "EOF reached while lexing \\(...");
        tok().type = Tk::Invalid;
        return;
    }

    // Skip the ")".
    tokens.save(text);
    tok().location = {start, tok().location};
    NextChar();
}
