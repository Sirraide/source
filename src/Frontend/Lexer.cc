module;

#include <cctype>
#include <srcc/Macros.hh>

module srcc.frontend.parser;
import srcc.frontend.token;
import srcc.utils;
using namespace srcc;

/// ===========================================================================
///  Lexer — Helpers and Data.
/// ===========================================================================
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) or c == '_' or c == '$';
}

/// Check if a character is allowed in an identifier.
constexpr bool IsContinue(char c) {
    return IsStart(c) or isdigit(static_cast<unsigned char>(c)) or c == '!';
}

constexpr bool IsBinary(char c) { return c == '0' or c == '1'; }
constexpr bool IsDecimal(char c) { return c >= '0' and c <= '9'; }
constexpr bool IsOctal(char c) { return c >= '0' and c <= '7'; }
constexpr bool IsHex(char c) { return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F'); }

/// All keywords.
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

/// ========================================================================
///  Main lexer implementation.
/// ========================================================================
struct Lexer {
    TokenStream& tokens;
    const File& f;
    const char* curr;
    const char* const end;
    char lastc;
    bool raw_mode = false;

    Lexer(TokenStream& into, const File& f);

    auto tok() -> Token& { return tokens.back(); }

    template <typename ...Args>
    void Error(Location where, fmt::format_string<Args...> fmt, Args&& ...args) {
        f.context().diags().diag(Diagnostic::Level::Error, where, fmt, std::forward<Args>(args)...);
    }

    auto CurrOffs() -> u32;
    auto CurrLoc() -> Location;
    void LexEscapedId();
    void LexIdentifier();
    void LexNumber();
    void LexString(char delim);
    void Next();
    void NextChar();
    void NextImpl();
    void SkipLine();
    void SkipWhitespace();
};

void Parser::ReadTokens(const File& file) {
    Lexer(stream, file);
    tok = stream.begin();
}

Lexer::Lexer(TokenStream& into, const File& f)
    : tokens{into},
      f{f},
      curr{f.data()},
      end{curr + f.size()} {
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

    /// Collapse CR LF and LF CR to a single newline,
    /// but keep CR CR and LF LF as two newlines.
    if (lastc == '\r' || lastc == '\n') {
        /// Two newlines in a row.
        if (curr != end && (*curr == '\r' || *curr == '\n')) {
            bool same = lastc == *curr;
            lastc = '\n';

            /// CR CR or LF LF.
            if (same) return;

            /// CR LF or LF CR.
            curr++;
        }

        /// Either CR or LF followed by something else.
        lastc = '\n';
    }
}

void Lexer::NextImpl() {
    tok().location.file_id = u16(f.file_id());

    /// Tokens are not artificial by default.
    tok().artificial = false;

    /// Skip whitespace.
    SkipWhitespace();

    /// Keep returning EOF if we're at EOF.
    if (lastc == 0) {
        tok().type = Tk::Eof;
        return;
    }

    /// Reset the token. We set the token type to 'invalid' here so that,
    /// if we encounter an error, we can just issue a diagnostic and return
    /// without setting the token type. The parser will then stop because
    /// it encounters an invalid token.
    tok().artificial = false;
    tok().type = Tk::Invalid;
    tok().location.pos = CurrOffs();

    /// Lex the token.
    switch (lastc) {
        case '\\':
            LexEscapedId();
            return;

        case ';':
            NextChar();
            tok().type = Tk::Semicolon;
            break;

        case ':':
            NextChar();
            if (lastc == ':') {
                NextChar();
                tok().type = Tk::ColonColon;
            } else {
                tok().type = Tk::Colon;
            }
            break;

        case ',':
            NextChar();
            tok().type = Tk::Comma;
            break;

        case '?':
            NextChar();
            tok().type = Tk::Question;
            break;

        case '(':
            NextChar();
            tok().type = Tk::LParen;
            break;

        case ')':
            NextChar();
            tok().type = Tk::RParen;
            break;

        case '[':
            NextChar();
            tok().type = Tk::LBrack;
            break;

        case ']':
            NextChar();
            tok().type = Tk::RBrack;
            break;

        case '{':
            NextChar();
            tok().type = Tk::LBrace;
            break;

        case '}':
            NextChar();
            tok().type = Tk::RBrace;
            break;

        case '.':
            NextChar();
            if (lastc == '.') {
                NextChar();
                if (lastc == '.') {
                    NextChar();
                    tok().type = Tk::Ellipsis;
                } else if (lastc == '<') {
                    NextChar();
                    tok().type = Tk::DotDotLess;
                } else if (lastc == '=') {
                    NextChar();
                    tok().type = Tk::DotDotEq;
                } else {
                    tok().type = Tk::DotDot;
                }
            } else {
                tok().type = Tk::Dot;
            }
            break;

        case '-':
            NextChar();
            if (lastc == '>') {
                NextChar();
                tok().type = Tk::RArrow;
            } else if (lastc == '-') {
                NextChar();
                tok().type = Tk::MinusMinus;
            } else if (lastc == '=') {
                NextChar();
                tok().type = Tk::MinusEq;
            } else {
                tok().type = Tk::Minus;
            }
            break;

        case '+':
            NextChar();
            if (lastc == '+') {
                NextChar();
                tok().type = Tk::PlusPlus;
            } else if (lastc == '=') {
                NextChar();
                tok().type = Tk::PlusEq;
            } else {
                tok().type = Tk::Plus;
            }
            break;

        case '*':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::StarEq;
            } else if (lastc == '*') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok().type = Tk::StarStarEq;
                } else {
                    tok().type = Tk::StarStar;
                }
            } else {
                tok().type = Tk::Star;
            }
            break;

        case '/':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::SlashEq;
            } else if (lastc == '/') {
                SkipLine();
                NextImpl();
                return;
            } else {
                tok().type = Tk::Slash;
            }
            break;

        case '%':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::PercentEq;
            } else {
                tok().type = Tk::Percent;
            }
            break;

        case '^':
            NextChar();
            tok().type = Tk::Caret;
            break;

        case '&':
            NextChar();
            tok().type = Tk::Ampersand;
            break;

        case '|':
            NextChar();
            tok().type = Tk::VBar;
            break;

        case '~':
            NextChar();
            tok().type = Tk::Tilde;
            break;

        case '!':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::Neq;
            } else {
                tok().type = Tk::Bang;
            }
            break;

        case '=':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::EqEq;
            } else if (lastc == '>') {
                NextChar();
                tok().type = Tk::RDblArrow;
            } else {
                tok().type = Tk::Assign;
            }
            break;

        case '<':
            NextChar();

            /// Handle C++ header names.
            if (tokens.size() > 1 and tokens[tokens.size() - 2].type == Tk::Import) {
                tempset raw_mode = true;
                tok().type = Tk::CXXHeaderName;

                SmallString<32> text;
                while (lastc != '>' and lastc != 0) {
                    text.push_back(lastc);
                    NextChar();
                }

                if (lastc == 0) {
                    Error(tok().location, "Expected '>'");
                    break;
                }

                tok().text = tokens.save(text);

                /// Bring the lexer back into sync.
                NextChar();
                break;
            }

            if (lastc == '=') {
                NextChar();
                tok().type = Tk::Le;
            } else if (lastc == '<') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok().type = Tk::ShiftLeftEq;
                } else {
                    tok().type = Tk::ShiftLeft;
                }
            } else if (lastc == '-') {
                NextChar();
                tok().type = Tk::LArrow;
            } else {
                tok().type = Tk::Lt;
            }
            break;

        case '>':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok().type = Tk::Ge;
            } else if (lastc == '>') {
                NextChar();
                if (lastc == '>') {
                    NextChar();
                    if (lastc == '=') {
                        NextChar();
                        tok().type = Tk::ShiftRightLogicalEq;
                    } else {
                        tok().type = Tk::ShiftRightLogical;
                    }
                } else if (lastc == '=') {
                    NextChar();
                    tok().type = Tk::ShiftRightEq;
                } else {
                    tok().type = Tk::ShiftRight;
                }
            } else {
                tok().type = Tk::Gt;
            }
            break;

        case '"':
        case '\'':
            LexString(lastc);
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
            LexNumber();
            break;

        default:
            if (IsStart(lastc)) LexIdentifier();
            else {
                Error(CurrLoc() << 1, "Unexpected <U+{:X}> character in program", lastc);
                break;
            }
    }

    /// Set the end of the token.
    tok().location.len = u16(u64(curr - f.data()) - tok().location.pos - 1);
    if (curr == end and not lastc) tok().location.len++;
}

void Lexer::SkipLine() {
    while (lastc != '\n' && lastc != 0) NextChar();
}

void Lexer::SkipWhitespace() {
    while (std::isspace(lastc)) NextChar();
}

void Lexer::LexIdentifier() {
    tok().type = Tk::Identifier;
    SmallString<32> text;
    do {
        text += lastc;
        NextChar();
    } while (IsContinue(lastc));
    tok().text = tokens.save(text);

    /// Helper to parse keywords and integer types.
    const auto LexSpecialToken = [&] {
        if (auto k = keywords.find(tok().text); k != keywords.end()) {
            tok().type = k->second;

            /// Handle "for~".
            if (tok().type == Tk::For and lastc == '~') {
                NextChar();
                tok().type = Tk::ForReverse;
            }
        } else if (tok().text.starts_with("i")) {
            /// Note: this returns true on error.
            if (not StringRef(tok().text).substr(1).getAsInteger(10, tok().integer))
                tok().type = Tk::IntegerType;
        }
    };

    /// In raw mode, special processing is disabled. This is used for
    /// parsing the argument and expansion lists of macros, as well as
    /// for handling __id.
    if (raw_mode) return LexSpecialToken();

    /// TODO: If we decide to support lexer macros, check for macro
    /// definitions and expansions here.

    /// Check for keywords and ints.
    LexSpecialToken();
}

void Lexer::LexNumber() {
    /// Helper function that actually parses a number.
    auto lex_number_impl = [this](bool pred(char), unsigned base) {
        /// Need at least one digit.
        if (not pred(lastc)) {
            Error(CurrLoc() << 1 <<= 1, "Invalid integer literal");
            return;
        }

        /// Parse the literal.
        SmallString<64> buf;
        while (pred(lastc)) {
            buf += lastc;
            NextChar();
        }

        /// The next character must not be a start character.
        if (IsStart(lastc)) {
            Error(Location{tok().location, CurrLoc()}, "Invalid character in integer literal: '{}'", lastc);
            return;
        }

        /// We have a valid integer literal!
        tok().type = Tk::Integer;

        /// Note: This returns true on error!
        tok().integer = APInt{};
        Assert(not buf.str().getAsInteger(base, tok().integer));
    };

    /// If the first character is a 0, then this might be a non-decimal constant.
    if (lastc == 0) {
        NextChar();

        /// Hexadecimal literal.
        if (lastc == 'x' or lastc == 'X') {
            NextChar();
            return lex_number_impl(IsHex, 16);
        }

        /// Octal literal.
        if (lastc == 'o' or lastc == 'O') {
            NextChar();
            return lex_number_impl(IsOctal, 8);
        }

        /// Binary literal.
        if (lastc == 'b' or lastc == 'B') {
            NextChar();
            return lex_number_impl(IsBinary, 2);
        }

        /// Multiple leading 0’s are not permitted.
        if (std::isdigit(lastc)) {
            Error(CurrLoc() << 1, "Leading 0 in integer literal. (Hint: Use 0o/0O for octal literals)");
            return;
        }

        /// Integer literal must be a literal 0.
        if (IsStart(lastc)) {
            Error(CurrLoc() <<= 1, "Invalid character in integer literal: '{}'", lastc);
            return;
        }

        /// Integer literal is 0.
        tok().type = Tk::Integer;
        tok().integer = 0;
        return;
    }

    /// If the first character is not 0, then we have a decimal literal.
    return lex_number_impl(IsDecimal, 10);
}

void Lexer::LexString(char delim) {
    /// Yeet the delimiter.
    SmallString<32> text;
    NextChar();

    /// Lex the string. If it’s a raw string, we don’t need to
    /// do any escaping.
    if (delim == '\'') {
        while (lastc != delim && lastc != 0) {
            text += lastc;
            NextChar();
        }
    }

    /// Otherwise, we also need to replace escape sequences.
    else if (delim == '"') {
        while (lastc != delim && lastc != 0) {
            if (lastc == '\\') {
                NextChar();
                switch (lastc) {
                    case 'a': text += '\a'; break;
                    case 'b': text += '\b'; break;
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

    /// Other string delimiters are invalid.
    else {
        Error(CurrLoc() << 1, "Invalid delimiter: {}", delim);
        return;
    }

    /// Make sure we actually have a delimiter.
    if (lastc != delim) {
        Error(CurrLoc() << 1, "Unterminated string literal");
        return;
    }

    tok().text = tokens.save(text);
    NextChar();

    /// This is a valid string.
    tok().type = Tk::StringLiteral;
}

void Lexer::LexEscapedId() {
    /// Yeet backslash.
    tempset raw_mode = true;
    auto start = tok().location;
    NextImpl();

    /// Mark this token as ‘artificial’. This is so we can e.g. nest
    /// macro definitions using `\expands` and `\endmacro`.
    tok().artificial = true;

    /// If the next token is anything other than "(", then it becomes the name.
    if (tok().type != Tk::LParen) {
        tok().type = Tk::Identifier;
        tok().text = tok().spelling();
        tok().location = {start, tok().location};
        return;
    }

    /// If the token is "(", then everything up to the next ")" is the name.
    tok().type = Tk::Identifier;
    SmallString<32> text;
    while (lastc != ')' and lastc != 0) {
        text += lastc;
        NextChar();
    }

    /// EOF.
    if (lastc == 0) {
        Error(start, "EOF reached while lexing \\(...");
        tok().type = Tk::Invalid;
        return;
    }

    /// Skip the ")".
    tokens.save(text);
    tok().location = {start, tok().location};
    NextChar();
}
