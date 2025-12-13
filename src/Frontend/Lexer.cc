#include <llvm/ADT/StringSwitch.h>
#include <base/Text.hh>
#include <base/TrieMap.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Token.hh>
#include <srcc/Macros.hh>
#include <srcc/Frontend/Parser.hh>

using namespace srcc;

// ===========================================================================
//  Lexer — Helpers and Data.
// ===========================================================================
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return text::IsAlpha(c) or c == '_';
}

/// Check if a character is allowed in an identifier.
constexpr bool IsContinue(char c) {
    return IsStart(c) or text::IsDigit(c) or c == '!';
}

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
    {"nil", Tk::Nil},
    {"noreturn", Tk::NoReturn},
    {"not", Tk::Not},
    {"or", Tk::Or},
    {"pragma", Tk::Pragma},
    {"proc", Tk::Proc},
    {"quote", Tk::Quote},
    {"range", Tk::Range},
    {"return", Tk::Return},
    {"static", Tk::Static},
    {"struct", Tk::Struct},
    {"then", Tk::Then},
    {"tree", Tk::Tree},
    {"true", Tk::True},
    {"try", Tk::Try},
    {"type", Tk::Type},
    {"typeof", Tk::Typeof},
    {"unreachable", Tk::Unreachable},
    {"val", Tk::Val},
    {"var", Tk::Var},
    {"variant", Tk::Variant},
    {"void", Tk::Void},
    {"where", Tk::Where},
    {"while", Tk::While},
    {"with", Tk::With},
    {"xor", Tk::Xor},
};

const base::TrieMap<Tk> punctuators = {
    {";", Tk::Semicolon},
    {",", Tk::Comma},
    {"?", Tk::Question},
    {"(", Tk::LParen},
    {")", Tk::RParen},
    {"[", Tk::LBrack},
    {"]", Tk::RBrack},
    {"{", Tk::LBrace},
    {"}", Tk::RBrace},
    {"^", Tk::Caret},
    {"&", Tk::Ampersand},
    {"|", Tk::VBar},
    {"~", Tk::Tilde},
    {"#", Tk::Hash},
    {"%", Tk::Percent},
    {"%=", Tk::PercentEq},
    {"!", Tk::Bang},
    {"!=", Tk::Neq},
    {":", Tk::Colon},
    {"::", Tk::ColonColon},
    {":%", Tk::ColonPercent},
    {":/", Tk::ColonSlash},
    {":>", Tk::UGt},
    {":>=", Tk::UGe},
    {".", Tk::Dot},
    {"..", Tk::DotDot},
    {"...", Tk::Ellipsis},
    {"..<", Tk::DotDotLess},
    {"..=", Tk::DotDotEq},
    {"-", Tk::Minus},
    {"->", Tk::RArrow},
    {"--", Tk::MinusMinus},
    {"-~", Tk::MinusTilde},
    {"-~=", Tk::MinusTildeEq},
    {"-=", Tk::MinusEq},
    {"+", Tk::Plus},
    {"++", Tk::PlusPlus},
    {"+=", Tk::PlusEq},
    {"+~", Tk::PlusTilde},
    {"+~=", Tk::PlusTildeEq},
    {"*", Tk::Star},
    {"*=", Tk::StarEq},
    {"*~", Tk::StarTilde},
    {"*~=", Tk::StarTildeEq},
    {"**", Tk::StarStar},
    {"**=", Tk::StarStarEq},
    {"=", Tk::Assign},
    {"==", Tk::EqEq},
    {"=>", Tk::RDblArrow},
    {">", Tk::SGt},
    {"><", Tk::Swap},
    {">=", Tk::SGe},
    {">>", Tk::ShiftRight},
    {">>=", Tk::ShiftRightEq},
    {">>>", Tk::ShiftRightLogical},
    {">>>=", Tk::ShiftRightLogicalEq},
    {"/", Tk::Slash},
    {"/=", Tk::SlashEq},
    {"<", Tk::SLt},
    {"<:", Tk::ULt},
    {"<=", Tk::SLe},
    {"<=:", Tk::ULe},
    {"<-", Tk::LArrow},
    {"<<", Tk::ShiftLeft},
    {"<<=", Tk::ShiftLeftEq},
    {"<<<", Tk::ShiftLeftLogical},
    {"<<<=", Tk::ShiftLeftLogicalEq},
};

bool Parser::IsKeyword(Tk t) {
    return keywords.contains(Spelling(t));
}

// ========================================================================
//  Main lexer implementation.
// ========================================================================
struct [[nodiscard]] Lexer : str, DiagsProducer {
    TokenStream& tokens;
    const srcc::File& f;
    Parser::CommentTokenCallback comment_token_handler;
    bool in_pragma = false;
    const Token invalid_token;
    const char* const input_end;

    Lexer(
        TokenStream& into,
        const srcc::File& f,
        Parser::CommentTokenCallback cb = nullptr
    );

    auto data_or_end() const -> const char* {
        if (empty()) return input_end;
        return data();
    }

    auto tok() -> Token& { return tokens.back(); }

    using DiagsProducer::Error;

    template <typename... Args>
    auto Error(std::format_string<Args...> fmt, Args&&... args) -> utils::Falsy {
        return Error(tok().location, fmt, LIBBASE_FWD(args)...);
    }

    char curr() { return front().value_or(0); }
    auto diags() const -> DiagnosticsEngine& { return f.context().diags(); }

    void FinishText() {
        auto ptr = tok().location.pointer();
        tok().text = String::CreateUnsafe(ptr, usz(data_or_end() - ptr));
    }

    auto Prev(usz i = 1) -> const Token& {
        i++; // Exclude the current token.
        if (tokens.size() >= i) return tokens[tokens.size() - i];
        return invalid_token;
    }

    auto CurrLoc() -> SLoc;
    void HandleCommentToken();
    void HandlePragma();
    void LexCXXHeaderName();
    void LexEntireFile();
    void LexEscapedId();
    void LexIdentifier(bool doller);
    bool LexNumber();
    void LexString();
    void Next();
    void NextImpl();
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
    Lexer l(s, file, std::move(cb));
    l.LexEntireFile();
}

Lexer::Lexer(TokenStream& into, const srcc::File& f, Parser::CommentTokenCallback cb)
    : str(f.contents().sv()),
      tokens{into},
      f{f},
      comment_token_handler{std::move(cb)},
      input_end{end()} {
    Assert(
        f.size() <= std::numeric_limits<u32>::max(),
        "We can’t handle files this big right now"
    );
}

auto Lexer::CurrLoc() -> SLoc { return SLoc(data_or_end()); }
void Lexer::LexEntireFile() {
    do Next();
    while (not tok().eof());
    if (not tokens.back().eof()) {
        tokens.allocate();
        tok().type = Tk::Eof;
        tok().location = SLoc(f.end() - 1);
    }
}

void Lexer::Next() {
    Assert(not in_pragma, "May not allocate tokens while handling pragma");
    tokens.allocate();
    NextImpl();
}

void Lexer::NextImpl() {
    tok().artificial = false;

    // Skip whitespace.
    trim_front();
    tok().location = CurrLoc();

    // Keep returning EOF if we're at EOF.
    if (empty()) {
        tok().type = Tk::Eof;
        tok().location = SLoc(f.end() - 1);
        return;
    }

    // Handle special tokens first.
    switch (*front()) {
        default: break;
        case '/':
            // Comment.
            if (consume("//")) {
                HandleCommentToken();
                NextImpl();
                return;
            }

            break;

        case '<':
            // C++ header name.
            if (
                Prev().is(Tk::Import) or
                (Prev().is(Tk::Comma) and Prev(2).is(Tk::CXXHeaderName))
            ) {
                LexCXXHeaderName();
                return;
            }

            break;

        case '\\':
            LexEscapedId();
            return;

        case '$':
            drop();
            if (not starts_with(IsContinue)) {
                tok().type = Tk::Dollar;
                return;
            }

            LexIdentifier(true);
            return;

        case '"':
        case '\'':
            LexString();
            return;

        case '0' ... '9':
            if (LexNumber()) return;

            // Skip the rest of the broken literal.
            drop_while(text::IsAlnum);
            NextImpl();
            return;
    }

    auto tk = match_prefix(punctuators);
    if (tk.has_value()) {
        tok().type = *tk;
        return;
    }

    if (IsStart(*front())) return LexIdentifier(false);
    Error("Unexpected <U+{:X}> character in program", *front());
}

void Lexer::HandleCommentToken() {
    drop_until('\n');

    // If we were asked to lex comment tokens, do so and dispatch it. To
    // keep the parser from having to deal with these, they never enter
    // the token stream.
    if (comment_token_handler) {
        tok().type = Tk::Comment;
        FinishText();
        comment_token_handler(tok());
    }
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
        const srcc::File& new_f = res.value();
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
        Lexer l(tokens, new_f, comment_token_handler);
        l.LexEntireFile();

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

void Lexer::LexIdentifier(bool dollar) {
    tok().type = dollar ? Tk::TemplateType : Tk::Identifier;
    drop_while(IsContinue);
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
        if (tok().type == Tk::For and consume('~'))
            tok().type = Tk::ForReverse;
    }

    // Integer types.
    else if (tok().text.starts_with("i")) {
        // Note: this returns true on error.
        if (not StringRef(tok().text).substr(1).getAsInteger(10, tok().integer))
            tok().type = Tk::IntegerType;
    }
}

bool Lexer::LexNumber() {
    // Helper function that actually parses a number.
    auto LexNumberImpl = [&](bool pred(char), unsigned base = 10, char second = 0) -> bool {
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
            return Error("Invalid digit '{}' in {} integer literal", curr(), kind);
        };

        // Need at least one digit.
        if (base != 10 and not starts_with(pred)) {
            // If this is not even any sort of digit, then issue
            // a more helpful error; this is so we don’t complain
            // about e.g. ‘;’ not being a digit.
            if (not starts_with(text::IsAlnum))
                return Error("Expected at least one digit after '{}'", second);
            return DiagnoseInvalidLiteral();
        }

        // Parse the literal.
        drop_while(pred);

        // The next character must not be a start character.
        if (starts_with(IsStart)) return DiagnoseInvalidLiteral();

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
    if (not consume('0')) return LexNumberImpl(text::IsDigit);

    // Otherwise, try lexing a hex/binary/octal literal.
    char second = curr();
    if (consume_any("xX")) return LexNumberImpl(text::IsXDigit, 16, second);
    if (consume_any("oO")) return LexNumberImpl(text::IsOctal, 8, second);
    if (consume_any("bB")) return LexNumberImpl(text::IsBinary, 2, second);

    // Leading 0’s must be followed by one of the above.
    if (starts_with(text::IsDigit)) return Error(
        "Leading zeros are not allowed in integers. Use 0o/0O for octal literals"
    );

    // Integer literal must be a literal 0.
    if (starts_with(IsStart)) return Error(
        "Invalid character in integer literal: '{}'",
        *front()
    );

    // If we get here, this must be a literal 0.
    tok().type = Tk::Integer;
    tok().integer = 0;
    return true;
}

void Lexer::LexString() {
    char delim = take()[0];
    Assert(delim == '"' or delim == '\'');

    // Lex the string. If it’s a raw string, we don’t need to
    // do any escaping.
    tok().type = Tk::StringLiteral;
    if (delim == '\'') {
        drop_until('\'');
        if (not consume(delim)) Error("Unterminated string literal");
        FinishText();
        tok().text = tok().text.drop().drop_back(); // Drop the quotes.
        tok().type = Tk::StringLiteral;
        return;
    }

    // We need to perform escaping here, so we can’t get away with
    // not allocating a buffer here, unfortunately.
    SmallString<32> text;
    for (;;) {
        text += take_until_any("\\\"").text();
        if (starts_with("\"") or drop().empty()) break;
        switch (take()[0]) {
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
            default: Error("Invalid escape sequence");
        }
    }

    // Done!
    if (not consume(delim)) Error("Unterminated string literal");
    tok().text = tokens.save(text);
}

void Lexer::LexCXXHeaderName() {
    tok().type = Tk::CXXHeaderName;
    drop().drop_until('>');
    if (not consume('>')) Error("Expected '>'");
    FinishText();
}

void Lexer::LexEscapedId() {
    auto backslash = CurrLoc();
    drop(); // Yeet '\'.
    NextImpl();
    tok().artificial = true;

    // If the next token is anything other than "(", then it becomes the name.
    if (tok().type != Tk::LParen) {
        tok().type = Tk::Identifier;
        FinishText();
        tok().location = backslash;
        return;
    }

    // If the token is "(", then everything up to the next ")" is the name.
    tok().type = Tk::Identifier;
    drop_until(')');
    if (not consume(')')) Error(backslash, "EOF reached while lexing \\(...");
    FinishText();

    // Drop the '()' from the name.
    tok().text = tok().text.drop().drop_back();
    tok().location = backslash;
}

auto SLoc::measure_token_length(const Context& ctx) const -> std::optional<u64> {
    auto f = file(ctx);
    if (not f) return std::nullopt;
    llvm::BumpPtrAllocator temp_alloc;
    TokenStream temp{temp_alloc};
    Lexer l{temp, *f};
    l.drop(usz(ptr - f->data()));
    l.Next();
    return l.tok().eof() ? 0 : u64(l.data_or_end() - ptr);
}
