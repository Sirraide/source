#include <llvm/ADT/StringExtras.h>
#include <print>
#include <srcc/Macros.hh>
#include <srcc/Frontend/Parser.hh>

using namespace srcc;

// ===========================================================================
//  Lexer — Helpers and Data.
// ===========================================================================
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return llvm::isAlpha(c) or c == '_';
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
const llvm::StringMap<Tk> keywords = {
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
    {"loop", Tk::Loop},
    {"match", Tk::Match},
    {"noreturn", Tk::NoReturn},
    {"not", Tk::Not},
    {"or", Tk::Or},
    {"pragma", Tk::Pragma},
    {"proc", Tk::Proc},
    {"range", Tk::Range},
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

bool Parser::IsKeyword(Tk t) {
    return keywords.contains(Spelling(t));
}

// ========================================================================
//  Main lexer implementation.
// ========================================================================
struct Lexer {
    TokenStream& tokens;
    const srcc::File& f;
    Parser::CommentTokenCallback comment_token_handler;
    const char* curr;
    const char* const end;
    bool in_pragma = false;

    Lexer(TokenStream& into, const srcc::File& f, Parser::CommentTokenCallback cb);

    auto tok() -> Token& { return tokens.back(); }

    char Curr() { return Done() ? 0 : *curr; }

    bool Done() { return curr == end; }

    template <typename... Args>
    bool Error(Location where, std::format_string<Args...> fmt, Args&&... args) {
        f.context().diags().diag(Diagnostic::Level::Error, where, fmt, std::forward<Args>(args)...);
        return false;
    }

    bool Eat(std::same_as<char> auto... cs) {
        if (curr != end and ((*curr == cs) or ...)) {
            curr++;
            return true;
        }

        return false;
    }

    void FinishText() {
        tok().location.len = u16(curr - f.data() - tok().location.pos);
        tok().text = tok().location.text(f.context());
    }

    auto CurrOffs() -> u32;
    auto CurrLoc() -> Location;
    void HandleCommentToken();
    void HandlePragma();
    void LexCXXHeaderName();
    void LexEscapedId();
    void LexIdentifierRest(bool dollar);
    bool LexNumber(bool zero);
    void LexString(char delim);
    void Next();
    void NextImpl();
    void SkipWhitespace();
};

auto Parser::ReadTokens(
    const File& file,
    CommentTokenCallback comment_callback
) -> LexedTokenStream {
    LexedTokenStream stream;
    ReadTokens(stream.tokens, file, std::move(comment_callback));
    return stream;
}

void Parser::ReadTokens(TokenStream& s, const File& file, CommentTokenCallback cb) {
    Lexer(s, file, std::move(cb));
}

Lexer::Lexer(TokenStream& into, const srcc::File& f, Parser::CommentTokenCallback cb)
    : tokens{into},
      f{f},
      comment_token_handler{std::move(cb)},
      curr{f.data()},
      end{curr + f.size()} {
    Assert(f.size() <= std::numeric_limits<u32>::max(), "We can’t handle files this big right now");
    do Next();
    while (not tok().eof());
    if (not tokens.back().eof()) tokens.allocate()->type = Tk::Eof;
}

auto Lexer::CurrLoc() -> Location { return {CurrOffs(), 1, u16(f.file_id())}; }
auto Lexer::CurrOffs() -> u32 { return u32(curr - f.data()); }

void Lexer::Next() {
    Assert(not in_pragma, "May not allocate tokens while handling pragma");
    tokens.allocate();
    NextImpl();
}

void Lexer::NextImpl() {
    tok().location.file_id = u16(f.file_id());

    // Tokens are not artificial by default.
    tok().artificial = false;

    // Skip whitespace.
    SkipWhitespace();

    // Keep returning EOF if we're at EOF.
    if (Done()) {
        tok().type = Tk::Eof;
        tok().location.pos = u32(f.size());
        tok().location.len = 1;
        return;
    }

    // Reset the token. We set the token type to 'invalid' here so that,
    // if we encounter an error, we can just issue a diagnostic and return
    // without setting the token type. The parser will then stop because
    // it encounters an invalid token.
    auto& ty = tok().type = Tk::Invalid;
    auto start = curr;
    tok().artificial = false;
    tok().location.pos = CurrOffs();
    tok().location.len = 1;

    // Lex the token.
    //
    // Warning: Ternary abuse incoming.
    switch (auto c = *curr++) {
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
        case '#': ty = Tk::Hash; break;

        // Two-character tokens.
        case '%': ty = Eat('=') ? Tk::PercentEq : Tk::Percent; break;
        case '!': ty = Eat('=') ? Tk::Neq : Tk::Bang; break;

        // Multi-character tokens.
        case ':':
            ty = Eat(':') ? Tk::ColonColon
               : Eat('/') ? Tk::ColonSlash
               : Eat('%') ? Tk::ColonPercent
               : Eat('>') ? (Eat('=') ? Tk::UGe : Tk::UGt)
                          : Tk::Colon;
            break;

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
               : Eat('~') ? (Eat('=') ? Tk::MinusTildeEq : Tk::MinusTilde)
               : Eat('=') ? Tk::MinusEq
                          : Tk::Minus;
            break;

        case '+':
            ty = Eat('+') ? Tk::PlusPlus
               : Eat('=') ? Tk::PlusEq
               : Eat('~') ? (Eat('=') ? Tk::PlusTildeEq : Tk::PlusTilde)
                          : Tk::Plus;
            break;

        case '*':
            ty = Eat('=') ? Tk::StarEq
               : Eat('~') ? (Eat('=') ? Tk::StarTildeEq : Tk::StarTilde)
               : Eat('*') ? (Eat('=') ? Tk::StarStarEq : Tk::StarStar)
                          : Tk::Star;
            break;

        case '=':
            ty = Eat('=') ? Tk::EqEq
               : Eat('>') ? Tk::RDblArrow
                          : Tk::Assign;
            break;

        case '>':
            ty = Eat('=')     ? Tk::SGe
               : not Eat('>') ? Tk::SGt
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

            ty = Eat('=')     ? (Eat(':') ? Tk::ULe : Tk::SLe)
               : Eat('-')     ? Tk::LArrow
               : Eat(':')     ? Tk::ULt
               : not Eat('<') ? Tk::SLt
               : Eat('<')     ? (Eat('=') ? Tk::ShiftLeftLogicalEq : Tk::ShiftLeftLogical)
               : Eat('=')     ? Tk::ShiftLeftEq
                              : Tk::ShiftLeft;

            break;

        case '\\':
            LexEscapedId();
            return;

        case '$':
            LexIdentifierRest(true);
            return;

        case '"':
        case '\'':
            LexString(c);
            return;

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
            if (LexNumber(c == '0')) return;
            while (llvm::isAlnum(Curr()))
                curr++;
            NextImpl();
            return;

        default:
            if (IsStart(c)) return LexIdentifierRest(false);
            Error(CurrLoc() << 1, "Unexpected <U+{:X}> character in program", c);
            break;
    }

    // Set the end of the token.
    tok().location.len = u16(curr - start);
}

void Lexer::HandleCommentToken() {
    const auto KeepSkipping = [&] { return not Done() and * curr != '\n'; };

    // If we were asked to lex comment tokens, do so and dispatch it. To
    // keep the parser from having to deal with these, they never enter
    // the token stream.
    if (comment_token_handler) {
        tok().type = Tk::Comment;
        while (KeepSkipping()) curr++;
        FinishText();
        comment_token_handler(tok());
        return;
    }

    // Otherwise, just skip past the comment.
    while (KeepSkipping()) curr++;
}

/// <pragma> ::= <pragma-include>
/// <pragma-include> ::= PRAGMA "include" STRING-LITERAL
void Lexer::HandlePragma() {
    tempset in_pragma = true;

    // Yeet 'pragma'.
    auto pragma_loc = tok().location;
    NextImpl();

    // Next token must be an identifier.
    if (tok().type != Tk::Identifier) {
        Error(pragma_loc, "Expected identifier after 'pragma'");
        return;
    }

    // Include pragma.
    if (tok().text == "include") {
        NextImpl();
        if (tok().type != Tk::StringLiteral) {
            Error(pragma_loc, "Expected string literal after 'pragma include'");
            return;
        }

        // Search for the file.
        auto file = f.path().parent_path() / tok().text.sv();
        auto res = f.context().try_get_file(file);
        if (not res) {
            Error(pragma_loc, "{}", res.error());
            return;
        }

        // Prevent a file from including itself.
        //
        // No, this doesn’t work if there is more than one level of recursion,
        // but this feature is only supposed to be used very sparingly anyways.
        auto& new_f = res.value();
        if (f == new_f) {
            Error(
                tok().location,
                "File '{}' may not include itself",
                f.context().file_name(f.file_id())
            );
            return;
        }

        // Drop the current token entirely.
        tokens.pop();

        // Lex the entire file.
        Lexer(tokens, new_f, comment_token_handler);

        // Drop the EOF token it produced.
        tokens.pop();
        return;
    }

    Error(
        pragma_loc,
        "Unknown pragma '{}'",
        tok().text
    );
}


void Lexer::SkipWhitespace() {
    while (llvm::isSpace(Curr())) curr++;
}

void Lexer::LexIdentifierRest(bool dollar) {
    tok().type = dollar ? Tk::TemplateType : Tk::Identifier;
    while (IsContinue(Curr())) curr++;
    FinishText();

    // Inside of pragmas, treat keywords and integer types as raw identifiers.
    if (dollar or in_pragma) return;

    // Keywords.
    if (auto k = keywords.find(tok().text); k != keywords.end()) {
        tok().type = k->second;

        // Handle pragmas.
        if (tok().type == Tk::Pragma) {
            HandlePragma();
            Next();
            return;
        }

        // Handle "for~".
        if (tok().type == Tk::For and Eat('~')) {
            tok().type = Tk::ForReverse;
            FinishText();
        }
    }

    // Integer types.
    else if (tok().text.starts_with("i")) {
        // Note: this returns true on error.
        if (not StringRef(tok().text).substr(1).getAsInteger(10, tok().integer))
            tok().type = Tk::IntegerType;
    }
}

bool Lexer::LexNumber(bool zero) {
    auto second = Curr();

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
            return Error(CurrLoc(), "Invalid digit '{}' in {} integer literal", Curr(), kind);
        };

        // Need at least one digit.
        if (base != 10 and not pred(Curr())) {
            // If this is not even any sort of digit, then issue
            // a more helpful error; this is so we don’t complain
            // about e.g. ‘;’ not being a digit.
            if (not llvm::isAlnum(Curr()))
                return Error(CurrLoc(), "Expected at least one digit after '{}'", second);
            return DiagnoseInvalidLiteral();
        }

        // Parse the literal.
        while (pred(Curr())) curr++;

        // The next character must not be a start character.
        if (IsStart(Curr())) return DiagnoseInvalidLiteral();

        // We have a valid integer literal!
        FinishText();
        tok().type = Tk::Integer;

        // Note: This returns true on error!
        tok().integer = APInt{};
        Assert(
            not tok().text.value().drop_front(base != 10 ? 2 : 0).getAsInteger(base, tok().integer),
            "Failed to lex base-{} integer '{}'",
            base,
            tok().text
        );

        return true;
    };

    // If the first character is not 0, then we have a decimal literal.
    if (not zero) return LexNumberImpl(IsDecimal);
    if (Eat('x', 'X')) return LexNumberImpl(IsHex, 16);
    if (Eat('o', 'O')) return LexNumberImpl(IsOctal, 8);
    if (Eat('b', 'B')) return LexNumberImpl(IsBinary, 2);

    // Leading 0’s must be followed by one of the above.
    if (llvm::isDigit(Curr())) return Error(
        CurrLoc() << 1,
        "Leading zeros are not allowed in integers. Use 0o/0O for octal literals"
    );

    // Integer literal must be a literal 0.
    if (IsStart(Curr())) return Error(
        CurrLoc() <<= 1,
        "Invalid character in integer literal: '{}'",
        Curr()
    );

    // If we get here, this must be a literal 0.
    FinishText();
    tok().type = Tk::Integer;
    tok().integer = 0;
    return true;
}

void Lexer::LexString(char delim) {
    // Anything other than ', " is invalid.
    if (delim != '"' and delim != '\'') {
        Error(CurrLoc() << 1, "Invalid delimiter: {}", delim);
        return;
    }

    // Lex the string. If it’s a raw string, we don’t need to
    // do any escaping.
    tok().type = Tk::StringLiteral;
    if (delim == '\'') {
        while (not Done() and *curr != delim) curr++;
        if (not Eat(delim)) Error(CurrLoc() << 1, "Unterminated string literal");
        FinishText();
        tok().text = tok().text.drop().drop_back(); // Drop the quotes.
        tok().type = Tk::StringLiteral;
        return;
    }

    // We need to perform escaping here, so we can’t get away with
    // not allocating a buffer here unfortunately.
    SmallString<32> text;
    while (not Done() and *curr != delim) {
        if (not Eat('\\')) {
            text += *curr++;
            continue;
        }

        // If, somehow, the file ends with an unterminated escape sequence,
        // don’t even bother reporting it since we’ll complain about the
        // missing delimiter anyway.
        if (Done()) break;

        // Handle escape sequences now (we could save the original source text
        // and do this later, but then we’d have to constantly keep track of
        // whether we’ve already done that etc. etc., so just get this out of
        // the way now.
        switch (*curr++) {
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
            default: Error({tok().location, CurrLoc()}, "Invalid escape sequence");
        }
    }

    // Done!
    if (not Eat(delim)) Error(CurrLoc() << 1, "Unterminated string literal");
    FinishText();
    tok().text = tokens.save(text);
}

void Lexer::LexCXXHeaderName() {
    tok().type = Tk::CXXHeaderName;
    while (not Done() and Curr() != '>') curr++;
    if (not Eat('>')) Error(tok().location, "Expected '>'");
    FinishText();
}

void Lexer::LexEscapedId() {
    auto backslash = tok().location;
    NextImpl();
    tok().artificial = true;

    // If the next token is anything other than "(", then it becomes the name.
    if (tok().type != Tk::LParen) {
        tok().type = Tk::Identifier;
        tok().text = tok().location.text(f.context());
        tok().location = {backslash, tok().location};
        return;
    }

    // If the token is "(", then everything up to the next ")" is the name.
    tok().type = Tk::Identifier;
    tok().location = CurrLoc();
    while (not Done() and *curr != ')') curr++;
    FinishText();
    tok().location = {backslash, CurrLoc()};
    if (not Eat(')')) Error(CurrLoc() << 1, "EOF reached while lexing \\(...");
}
