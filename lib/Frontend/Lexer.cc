#include <source/Frontend/Lexer.hh>

/// ===========================================================================
///  Lexer — Helpers and Data.
/// ===========================================================================
namespace src {
namespace {
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return std::isalpha(c) or c == '_' or c == '$';
}

/// Check if a character is allowed in an identifier.
constexpr bool IsContinue(char c) {
    return IsStart(c) or isdigit(c) or c == '!';
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
} // namespace
} // namespace src
/// ========================================================================
///  Main lexer implementation.
/// ========================================================================
src::Lexer::Lexer(Context* ctx, TokenStream& into, File& f)
    : ctx{ctx}, f{f}, tokens(into) {
    /// Init state.
    curr = f.data();
    end = curr + f.size();

    /// Define predefined macros.
    AllocateMacroDefinition(
        "__filename__",
        {},
        {Token::Make(Tk::StringLiteral, into.save(f.path().filename().string()))}
    );

    AllocateMacroDefinition(
        "__directory__",
        {},
        {Token::Make(Tk::StringLiteral, into.save(f.path().parent_path().string()))}
    );

    /// Init first token.
    NextChar();
    Next();
}

void src::Lexer::LexEntireFile(Context* ctx, TokenStream& into, File& f) {
    Lexer L{ctx, into, f};
    do L.Next();
    while (L.tok.type != Tk::Eof);
}

auto src::Lexer::CurrLoc() const -> Location { return {CurrOffs(), 1, u16(f.file_id())}; }
auto src::Lexer::CurrOffs() const -> u32 { return u32(curr - f.data()) - 1; }

void src::Lexer::Next() {
    tokens.allocate();
    NextImpl();
}

void src::Lexer::NextImpl() {
    tok.location.file_id = u16(f.file_id());

    /// Tokens are not artificial by default.
    tok.artificial = false;

    /// Pop empty macro expansions off the expansion stack.
    while (not macro_expansion_stack.empty())
        if (macro_expansion_stack.back().done())
            macro_expansion_stack.pop_back();

    /// Insert tokens from macro expansion.
    if (not macro_expansion_stack.empty()) {
        auto& expansion = macro_expansion_stack.back();
        *tokens.back() = ++expansion;

        /// If this token is another macro definition, handle it.
        if (tok.type == Tk::Identifier and tok.text == "macro") LexMacroDefinition();
        return;
    }

    /// Skip whitespace.
    SkipWhitespace();

    /// Keep returning EOF if we're at EOF.
    if (lastc == 0) {
        tok.type = Tk::Eof;
        return;
    }

    /// Reset the token. We set the token type to 'invalid' here so that,
    /// if we encounter an error, we can just issue a diagnostic and return
    /// without setting the token type. The parser will then stop because
    /// it encounters an invalid token.
    tok.artificial = false;
    tok.type = Tk::Invalid;
    tok.location.pos = CurrOffs();

    /// Lex the token.
    switch (lastc) {
        case '\\':
            LexEscapedId();
            return;

        case ';':
            NextChar();
            tok.type = Tk::Semicolon;
            break;

        case ':':
            NextChar();
            if (lastc == ':') {
                NextChar();
                tok.type = Tk::ColonColon;
            } else {
                tok.type = Tk::Colon;
            }
            break;

        case ',':
            NextChar();
            tok.type = Tk::Comma;
            break;

        case '?':
            NextChar();
            tok.type = Tk::Question;
            break;

        case '(':
            NextChar();
            tok.type = Tk::LParen;
            break;

        case ')':
            NextChar();
            tok.type = Tk::RParen;
            break;

        case '[':
            NextChar();
            tok.type = Tk::LBrack;
            break;

        case ']':
            NextChar();
            tok.type = Tk::RBrack;
            break;

        case '{':
            NextChar();
            tok.type = Tk::LBrace;
            break;

        case '}':
            NextChar();
            tok.type = Tk::RBrace;
            break;

        /// Syntax of this token is `#<identifier>` or `#<number>`.
        case '#': {
            auto l = CurrLoc();
            NextChar();
            NextImpl();

            /// Validate name.
            if (tok.type == Tk::Identifier or tok.type == Tk::Integer) {
                if (tok.type == Tk::Integer) {
                    SmallString<32> buf;
                    tok.integer.toStringUnsigned(buf, 10);
                    tok.text = tokens.save(buf);
                }

                tok.type = Tk::MacroParameter;
                tok.location = {l, tok.location};
            } else {
                Error(CurrLoc(), "Expected identifier or integer after '#'");
            }

            /// Check if this token is even allowed here.
            if (not in_macro_definition) Error(tok.location, "Unexpected macro parameter outside of macro definition");
        } break;

        case '.':
            NextChar();
            if (lastc == '.') {
                NextChar();
                if (lastc == '.') {
                    NextChar();
                    tok.type = Tk::Ellipsis;
                } else if (lastc == '<') {
                    NextChar();
                    tok.type = Tk::DotDotLess;
                } else if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::DotDotEq;
                } else {
                    tok.type = Tk::DotDot;
                }
            } else {
                tok.type = Tk::Dot;
            }
            break;

        case '-':
            NextChar();
            if (lastc == '>') {
                NextChar();
                tok.type = Tk::RArrow;
            } else if (lastc == '-') {
                NextChar();
                tok.type = Tk::MinusMinus;
            } else if (lastc == '=') {
                NextChar();
                tok.type = Tk::MinusEq;
            } else {
                tok.type = Tk::Minus;
            }
            break;

        case '+':
            NextChar();
            if (lastc == '+') {
                NextChar();
                tok.type = Tk::PlusPlus;
            } else if (lastc == '=') {
                NextChar();
                tok.type = Tk::PlusEq;
            } else {
                tok.type = Tk::Plus;
            }
            break;

        case '*':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::StarEq;
            } else if (lastc == '*') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::StarStarEq;
                } else {
                    tok.type = Tk::StarStar;
                }
            } else {
                tok.type = Tk::Star;
            }
            break;

        case '/':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::SlashEq;
            } else if (lastc == '/') {
                SkipLine();
                NextImpl();
                return;
            } else {
                tok.type = Tk::Slash;
            }
            break;

        case '%':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::PercentEq;
            } else {
                tok.type = Tk::Percent;
            }
            break;

        case '^':
            NextChar();
            tok.type = Tk::Caret;
            break;

        case '&':
            NextChar();
            tok.type = Tk::Ampersand;
            break;

        case '|':
            NextChar();
            tok.type = Tk::VBar;
            break;

        case '~':
            NextChar();
            tok.type = Tk::Tilde;
            break;

        case '!':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Neq;
            } else {
                tok.type = Tk::Bang;
            }
            break;

        case '=':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::EqEq;
            } else if (lastc == '>') {
                NextChar();
                tok.type = Tk::RDblArrow;
            } else {
                tok.type = Tk::Assign;
            }
            break;

        case '<':
            NextChar();

            /// Handle C++ header names.
            if (tokens.size() > 1 and tokens[tokens.size() - 2].type == Tk::Import) {
                tok.type = Tk::CXXHeaderName;
                raw_mode = true;
                SmallString<32> text;
                while (lastc != '>' and lastc != 0) {
                    text.push_back(lastc);
                    NextChar();
                }

                if (lastc == 0) {
                    Error(tok.location, "Expected '>'");
                    break;
                }

                /// Bring the lexer back into sync.
                tok.text = tokens.save(text);
                raw_mode = false;
                NextChar();
                break;
            }

            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Le;
            } else if (lastc == '<') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::ShiftLeftEq;
                } else {
                    tok.type = Tk::ShiftLeft;
                }
            } else if (lastc == '-') {
                NextChar();
                tok.type = Tk::LArrow;
            } else {
                tok.type = Tk::Lt;
            }
            break;

        case '>':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Ge;
            } else if (lastc == '>') {
                NextChar();
                if (lastc == '>') {
                    NextChar();
                    if (lastc == '=') {
                        NextChar();
                        tok.type = Tk::ShiftRightLogicalEq;
                    } else {
                        tok.type = Tk::ShiftRightLogical;
                    }
                } else if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::ShiftRightEq;
                } else {
                    tok.type = Tk::ShiftRight;
                }
            } else {
                tok.type = Tk::Gt;
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
    tok.location.len = u16(u64(curr - f.data()) - tok.location.pos - 1);
    if (curr == end and not lastc) tok.location.len++;
}

void src::Lexer::NextChar() {
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

void src::Lexer::SkipLine() {
    while (lastc != '\n' && lastc != 0) NextChar();
}

void src::Lexer::SkipWhitespace() {
    while (std::isspace(lastc)) NextChar();
}

void src::Lexer::LexIdentifier() {
    tok.type = Tk::Identifier;
    SmallString<32> text;
    do {
        text += lastc;
        NextChar();
    } while (IsContinue(lastc));
    tok.text = tokens.save(text);

    /// Helper to parse keywords and integer types.
    const auto LexSpecialToken = [&] {
        if (auto k = keywords.find(tok.text); k != keywords.end()) {
            tok.type = k->second;

            /// Handle "for~".
            if (tok.type == Tk::For and lastc == '~') {
                NextChar();
                tok.type = Tk::ForReverse;
            }
        } else if (tok.text.starts_with("i")) {
            /// Note: this returns true on error, for some reason.
            if (not StringRef(tok.text).substr(1).getAsInteger(10, tok.integer))
                tok.type = Tk::IntegerType;
        }
    };

    /// In raw mode, special processing is disabled. This is used for
    /// parsing the argument and expansion lists of macros, as well as
    /// for handling __id.
    if (raw_mode) return LexSpecialToken();

    /// Handle macro expansions.
    if (auto m = macro_definitions_by_name.find(tok.text); m != macro_definitions_by_name.end())
        return LexMacroExpansion(m->second);

    /// Handle macro definitions.
    if (tok.text == "macro") return LexMacroDefinition();

    /// Check for keywords and ints.
    LexSpecialToken();
}

void src::Lexer::LexNumber() {
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
            Error(Location{tok.location, CurrLoc()}, "Invalid character in integer literal: '{}'", lastc);
            return;
        }

        /// We have a valid integer literal!
        tok.type = Tk::Integer;

        /// Note: This returns true on error!
        tok.integer = APInt{};
        Assert(not buf.str().getAsInteger(base, tok.integer));
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
        tok.type = Tk::Integer;
        tok.integer = 0;
        return;
    }

    /// If the first character is not 0, then we have a decimal literal.
    return lex_number_impl(IsDecimal, 10);
}

void src::Lexer::LexString(char delim) {
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
                        Error({tok.location, CurrLoc()}, "Invalid escape sequence");
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

    tok.text = tokens.save(text);
    NextChar();

    /// This is a valid string.
    tok.type = Tk::StringLiteral;
}

void src::Lexer::LexEscapedId() {
    /// Yeet backslash.
    tempset raw_mode = true;
    auto start = tok.location;
    NextImpl();

    /// Mark this token as ‘artificial’. This is so we can e.g. nest
    /// macro definitions using `\expands` and `\endmacro`.
    tok.artificial = true;

    /// If the next token is anything other than "(", then it becomes the name.
    if (tok.type != Tk::LParen) {
        tok.type = Tk::Identifier;
        tok.text = Spelling(tok.type);
        tok.location = {start, tok.location};
        return;
    }

    /// If the token is "(", then everything up to the next ")" is the name.
    tok.type = Tk::Identifier;
    SmallString<32> text;
    while (lastc != ')' and lastc != 0) {
        text += lastc;
        NextChar();
    }

    /// EOF.
    if (lastc == 0) {
        Error(start, "EOF reached while lexing \\(...");
        tok.type = Tk::Invalid;
        return;
    }

    /// Skip the ")".
    tokens.save(text);
    tok.location = {start, tok.location};
    NextChar();
}

/// ========================================================================
///  Macros.
/// ========================================================================
auto src::Lexer::AllocateMacroDefinition(
    StringRef name,
    Location location,
    SmallVector<Token>&& expansion
) -> Macro& {
    auto& m = macro_definitions.emplace_back();
    macro_definitions_by_name[m.name = std::move(name)] = &m;
    m.location = location;
    m.expansion = std::move(expansion);
    return m;
}

/// <macro-definition> ::= MACRO <identifier> <tokens> EXPANDS <tokens> ENDMACRO
void src::Lexer::LexMacroDefinition() {
    tempset raw_mode = true;
    tempset in_macro_definition = true;
    NextImpl(); /// Yeet 'macro'.

    /// Make sure we can define this macro.
    if (tok.type != Tk::Identifier) {
        Error(tok.location, "Expected identifier");

        /// Synchronise on 'endmacro'.
        while (tok.type != Tk::Eof and (tok.type != Tk::Identifier or tok.text != "endmacro")) NextImpl();
        return;
    }

    if (macro_definitions_by_name.find(tok.text) == macro_definitions_by_name.end())
        Error(tok.location, "Macro '{}' is already defined. Use `undef` to undefine it first.", tok.text);

    /// Allocate a new macro.
    auto& m = AllocateMacroDefinition(tok.text, tok.location);
    NextImpl();

    /// Helper to issue an error and print the current macro definition.
    auto Err = [&]<typename... arguments>(fmt::format_string<arguments...> fmt, arguments&&... args) {
        Diag::Note(ctx, m.location, "In definition of macro '{}' here", m.name);
        Error(tok.location, fmt, std::forward<arguments>(args)...);
    };

    /// Collect macro parameters to make sure they exist and that there aren’t any duplicates.
    llvm::StringSet parameter_names;

    /// Read tokens until we encounter 'expands'.
    for (;;) {
        switch (tok.type) {
            case Tk::Eof: return Err("EOF reached while reading macro definition");

            /// Parameter must not exist.
            case Tk::MacroParameter:
                if (parameter_names.contains(tok.text)) return Err("Duplicate macro parameter '{}'", tok.text);
                parameter_names.insert(tok.text);
                goto param_list_default_case;

            /// Check for 'expands'.
            case Tk::Identifier:
                if (tok.text == "expands" and not tok.artificial) goto expands;
                [[fallthrough]];

            /// Add the token to the parameter list.
            default:
            param_list_default_case:
                m.parameter_list.push_back(tok);
                NextImpl();
        }
    }

    /// Yeet 'expands'.
expands:
    NextImpl();

    /// Read tokens until we encounter 'endmacro'.
    for (;;) {
        switch (tok.type) {
            case Tk::Eof: return Err("EOF reached while reading macro definition");

            /// Parameter must exist.
            case Tk::MacroParameter:
                if (not parameter_names.contains(tok.text)) return Err("Undefined macro parameter '{}'", tok.text);
                goto expansion_default_case;

            /// Check for 'endmacro'.
            case Tk::Identifier:
                if (tok.text == "endmacro" and not tok.artificial) goto endmacro;
                [[fallthrough]];

            /// Add the token to the expansion.
            default:
            expansion_default_case:
                m.expansion.push_back(tok);
                NextImpl();
        }
    }

    /// Yeet 'endmacro' and lex the next token.
endmacro:
    raw_mode = false;
    in_macro_definition = false;
    NextImpl();
}

void src::Lexer::LexMacroExpansion(Lexer::Macro* m) {
    StringMap<Token> args;
    auto loc = tok.location;

    /// Disable macro expansion etc.
    tempset raw_mode = true;

    /// Read tokens according to the definition of the macro parameter list
    /// and bind macro parameters to their values.
    for (auto& t : m->parameter_list) {
        /// Yeet the previous token. We do this here to make sure
        /// that we don’t discard past the last token of the param
        /// list, since the next token after that must come from the
        /// expansion itself.
        NextImpl();

        /// EOF is not a token.
        if (tok.type == Tk::Eof) {
            Error(
                tok.location,
                "EOF reached while reading parameters of macro '{}'",
                m->name
            );
            return;
        }

        /// Handle macro parameters.
        if (t.type == Tk::MacroParameter) {
            args[t.text] = tok;
            continue;
        }

        /// Make sure the token matches the expected token.
        if (tok != t) Error(
            tok.location,
            "Expected {} token in expansion of macro '{}', got {}",
            Spelling(t.type),
            m->name,
            Spelling(tok.type)
        );
    }

    /// Push the expansion onto the stack.
    macro_expansion_stack.emplace_back(*this, *m, std::move(args), loc);

    /// Reënable macro expansion and read the next token.
    raw_mode = false;
    NextImpl();
}

auto src::Lexer::MacroExpansion::operator++() -> Token {
    Token ret;
    Assert(not done());

    /// If the token is a macro arg, get the bound argument.
    if (it->type == Tk::MacroParameter) {
        auto arg = bound_parameters.find(it->text);
        Assert(arg != bound_parameters.end(), "Unbound macro argument '{}'", it->text);
        it++;
        ret = arg->second;
    }

    /// Otherwise, return a copy of the token.
    else { ret = *it++; }

    /// Mark the token as non-artificial, because, for example,
    /// if we are inserting "endmacro", then we want the inserted
    /// identifier to *not* be artificial so we actually end up
    /// closing a macro definition.
    ret.artificial = false;
    return ret;
}

bool src::operator==(const Token& a, const Token& b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case Tk::Identifier:
        case Tk::MacroParameter:
        case Tk::StringLiteral:
        case Tk::CXXHeaderName:
            return a.text == b.text;

        case Tk::Integer:
        case Tk::IntegerType:
            return a.integer == b.integer;

        /// All these are trivially equal.
        case Tk::Invalid:
        case Tk::Eof:

        case Tk::Alias:
        case Tk::And:
        case Tk::As:
        case Tk::AsBang:
        case Tk::Asm:
        case Tk::Assert:
        case Tk::Bool:
        case Tk::Break:
        case Tk::Continue:
        case Tk::Defer:
        case Tk::Delete:
        case Tk::Do:
        case Tk::Dynamic:
        case Tk::Elif:
        case Tk::Else:
        case Tk::Enum:
        case Tk::Export:
        case Tk::F32:
        case Tk::F64:
        case Tk::Fallthrough:
        case Tk::False:
        case Tk::For:
        case Tk::ForReverse:
        case Tk::Goto:
        case Tk::If:
        case Tk::In:
        case Tk::Init:
        case Tk::Int:
        case Tk::Import:
        case Tk::Is:
        case Tk::Land:
        case Tk::Lor:
        case Tk::Match:
        case Tk::NoReturn:
        case Tk::Not:
        case Tk::Or:
        case Tk::Pragma:
        case Tk::Proc:
        case Tk::Return:
        case Tk::Static:
        case Tk::Struct:
        case Tk::Then:
        case Tk::True:
        case Tk::Try:
        case Tk::Type:
        case Tk::Typeof:
        case Tk::Unreachable:
        case Tk::Val:
        case Tk::Var:
        case Tk::Variant:
        case Tk::Void:
        case Tk::While:
        case Tk::With:
        case Tk::Xor:

        case Tk::CChar:
        case Tk::CChar8T:
        case Tk::CChar16T:
        case Tk::CChar32T:
        case Tk::CWCharT:
        case Tk::CShort:
        case Tk::CInt:
        case Tk::CLong:
        case Tk::CLongLong:
        case Tk::CLongDouble:
        case Tk::CSizeT:

        case Tk::Semicolon:
        case Tk::Colon:
        case Tk::ColonColon:
        case Tk::Comma:
        case Tk::LParen:
        case Tk::RParen:
        case Tk::LBrack:
        case Tk::RBrack:
        case Tk::LBrace:
        case Tk::RBrace:
        case Tk::Ellipsis:
        case Tk::Dot:
        case Tk::LArrow:
        case Tk::RArrow:
        case Tk::RDblArrow:
        case Tk::Question:
        case Tk::Plus:
        case Tk::Minus:
        case Tk::Star:
        case Tk::Slash:
        case Tk::Percent:
        case Tk::Caret:
        case Tk::Ampersand:
        case Tk::VBar:
        case Tk::Tilde:
        case Tk::Bang:
        case Tk::Assign:
        case Tk::DotDot:
        case Tk::DotDotLess:
        case Tk::DotDotEq:
        case Tk::MinusMinus:
        case Tk::PlusPlus:
        case Tk::StarStar:
        case Tk::Lt:
        case Tk::Le:
        case Tk::Gt:
        case Tk::Ge:
        case Tk::EqEq:
        case Tk::Neq:
        case Tk::PlusEq:
        case Tk::MinusEq:
        case Tk::StarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeft:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
        case Tk::StarStarEq:
            return true;
    }

    Unreachable();
}
