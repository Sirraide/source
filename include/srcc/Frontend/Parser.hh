#ifndef SRCC_FRONTEND_PARSER_HH
#define SRCC_FRONTEND_PARSER_HH

#include <srcc/Frontend/ParseTree.hh>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/TrailingObjects.h>

namespace srcc {
struct LexedTokenStream {
    llvm::BumpPtrAllocator alloc;
    TokenStream tokens{alloc};
};
}

namespace srcc::parser {
enum class DREContext {
    ColonColon,
    AfterHashInMacroCall,
};

struct ScopeSpec {
    SLoc global_loc;
    Ptr<ParsedStmt> expr;
    SmallVector<DeclNameLoc> names;

    bool empty() const {
        return not global_loc.is_valid() and
               expr.invalid() and
               names.empty();
    }
};

struct Signature {
    SmallVector<ParsedParameter, 10> param_types;
    Ptr<ParsedStmt> ret; // Unset if no return type is parsed.
    Ptr<ParsedStmt> where;
    DeclNameLoc name;
    Ptr<ParsedStmt> associated_type;
    SLoc proc_loc;
    SLoc tok_after_proc;
    ParsedProcAttrs attrs;
};

struct TentativeParseScope;
struct State {
    TokenStream::Iterator tok;
    TentativeParseScope* tentative = nullptr;
    u16 num_parens{}, num_brackets{}, num_braces{};
    bool has_error : 1 = false;
    bool has_diag : 1 = false;
};

struct TentativeParseScope {
    LIBBASE_IMMOVABLE(TentativeParseScope);

    Parser& p;
    State saved_state;
    SmallVector<Diagnostic, 0> caught_diags;

    TentativeParseScope(Parser& p);
    ~TentativeParseScope();

    void commit();
    bool ok();
    auto parent() -> TentativeParseScope*;
};

enum class ParseExprFlags {
    None = 0,
    ExpectType = 1,
};

LIBBASE_DEFINE_FLAG_ENUM(ParseExprFlags);
}

class srcc::Parser : DiagsProducer, parser::State {
    SRCC_IMMOVABLE(Parser);

    friend parser::TentativeParseScope;

public:
    using CommentTokenCallback = VerifyDiagnosticsEngine::CommentTokenCallback;

private:
    friend DiagsProducer;

    class BracketTracker {
        LIBBASE_IMMOVABLE(BracketTracker);
        Parser& p;
        const bool diagnose;
        Tk open_bracket, close_bracket;
        bool consumed_close = false;

    public:
        SLoc left, right;

        BracketTracker(Parser& p, Tk open, bool diagnose = true);
        ~BracketTracker();
        bool close();
        [[nodiscard]] auto corresponding_closing_bracket() -> Tk { return close_bracket; }

    private:
        void decrement();
    };

    ParsedModule* mod;
    parser::Signature* current_signature = nullptr;
    const TokenStream& stream;
    Context& ctx;
    const bool parsing_internal_file;

public:
    /// Parse a file.
    static auto Parse(
        const File& file,
        CommentTokenCallback comment_callback = {},
        bool is_internal_file = false
    ) -> ParsedModule::Ptr;

    /// Parse a TU fragment.
    static auto ParseFragment(Context& ctx, const TokenStream& toks) -> ParsedModule::Ptr;

    /// Read all tokens in a file.
    static auto ReadTokens(
        const File& file,
        CommentTokenCallback comment_callback = {}
    ) -> LexedTokenStream;

    /// Allocate data.
    void* allocate(usz size, usz align) { return mod->allocator().Allocate(size, align); }

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& { return ctx.diags(); }

    /// Get the module.
    auto module() -> ParsedModule& { return *mod; }

private:
    explicit Parser(
        ParsedModule* mod,
        const TokenStream& tokens,
        bool internal
    );

    /// Each of these corresponds to a production in the grammar.
    auto ParseAssert(bool is_compile_time) -> Ptr<ParsedAssertExpr>;
    auto ParseBlock() -> Ptr<ParsedBlockExpr>;
    auto ParseDeclRefExpr(parser::DREContext ctx, Ptr<ParsedStmt> root_expr = nullptr) -> Ptr<ParsedStmt>;
    auto ParseExpr(int precedence = -1, parser::ParseExprFlags Flags = {}) -> Ptr<ParsedStmt>;
    auto ParseForStmt() -> Ptr<ParsedStmt>;
    void ParseFile();
    void ParseHeader();
    auto ParseIf(bool is_static) -> Ptr<ParsedIfExpr>;
    void ParseImport();
    auto ParseIntent() -> std::pair<SLoc, Intent>;
    auto ParseMatchExpr() -> Ptr<ParsedMatchExpr>;
    auto ParseOptionalScopeSpec(Ptr<ParsedStmt> expr = nullptr) -> parser::ScopeSpec;
    auto ParseOverloadableOperatorName() -> std::optional<DeclNameLoc>;
    bool ParseParameter(parser::Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls);
    auto ParseParenExpr() -> Ptr<ParsedStmt>;
    void ParsePreamble();
    auto ParseProcBody() -> Ptr<ParsedStmt>;
    auto ParseProcDecl() -> Ptr<ParsedProcDecl>;
    auto ParseQuotedTokenSeq(SLoc quote_loc, bool in_macro_call) -> Ptr<ParsedStmt>;
    bool ParseSignature(parser::Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls, bool allow_constraint);
    bool ParseSignatureImpl(SmallVectorImpl<ParsedVarDecl*>* decls, bool allow_constraint);
    void ParseStmts(SmallVectorImpl<ParsedStmt*>& into, Tk stop_at = Tk::Eof);
    auto ParseStmt() -> Ptr<ParsedStmt>;
    auto ParseStructDecl() -> Ptr<ParsedStructDecl>;
    auto ParseType(int precedence = -1) -> Ptr<ParsedStmt>;
    auto ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedVarDecl>;

    auto CreateDRE(parser::ScopeSpec&& ss, DeclNameLoc name) -> ParsedDeclRefExpr*;
    auto CreateType(parser::Signature& sig) -> ParsedProcType*;

    void AddDiagRemark(std::string&& s);
    void ReportDiag(Diagnostic&& d);

    using DiagsProducer::Error;
    template <typename... Args>
    auto Error(std::format_string<Args...> fmt, Args&&... args) -> std::nullptr_t {
        return Error(tok->location, fmt, std::forward<Args>(args)...);
    }

    /// Consume a token or issue an error.
    bool ConsumeOrError(Tk tk);

    /// Check if we’re at a token.
    bool At(std::same_as<Tk> auto... tks) {
        return tok->is(tks...);
    }

    bool Is(Tk t, std::same_as<Tk> auto... tks) {
        return ((t == tks) or ...);
    }

    /// Check if we’re at the start of an expression/statement.
    bool AtStartOfExpression();
    bool AtStartOfStatement();

    /// Consume a token if it is present.
    bool Consume(Tk tk);
    bool Consume(SLoc& into, Tk tk);

    /// Consume a contextual keyword.
    bool ConsumeContextual(StringRef keyword);
    bool ConsumeContextual(SLoc& into, StringRef keyword);

    /// Consume a token if we’re at it, and issue an error about it otherwise.
    template <typename... Args>
    bool ExpectAndConsume(Tk t, std::format_string<Args...> fmt, Args&&... args) {
        if (Consume(t)) return true;
        Error(fmt, std::forward<Args>(args)...);
        return false;
    }

    /// Consume a semicolon and issue an error on the previous line if it is missing.
    bool ExpectSemicolon();

    /// Check if a token is a keyword (implemented in the lexer).
    bool IsKeyword(Tk t);

    /// Get a lookahead token; returns EOF if looking past the end.
    auto LookAhead(usz n = 1) -> const Token&;

    /// Consume a token and return it (or its location). If the parser is at
    /// end of file, the token iterator is not advanced.
    auto Next() -> SLoc;

    /// Actually advance to the next token. This only exists to implement Next(),
    /// SkipTo(), and BracketTracker; do not call this from anywhere else!
    auto NextTokenImpl() -> SLoc;

    /// Skip up to a token.
    bool SkipTo(std::same_as<Tk> auto... tks);

    /// Skip up and past a token.
    bool SkipPast(std::same_as<Tk> auto... tks);

    /// Read all tokens from a file.
    static void ReadTokens(TokenStream& s, const File& file, CommentTokenCallback cb);
};

#endif // SRCC_FRONTEND_PARSER_HH
