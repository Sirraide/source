#ifndef SRCC_FRONTEND_PARSER_HH
#define SRCC_FRONTEND_PARSER_HH

#include <srcc/AST/DeclName.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Core.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Token.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/TrailingObjects.h>

namespace srcc {
class Parser;
class ParsedModule;
class ParsedStmt;
struct ParsedMatchCase;
struct LexedTokenStream;
struct ImportedModule;
struct ParsedParameter;

/// The list of parameter indices, for each template parameter,
/// in which that template parameter is deduced.
using TemplateParamDeductionInfo = HashMap<String, llvm::SmallDenseSet<u32>>;

#define PARSE_TREE_NODE(node) class SRCC_CAT(Parsed, node);
#include "srcc/ParseTree.inc"
} // namespace srcc

/// Parsed representation of a single file. NOT thread-safe.
class srcc::ParsedModule {
    SRCC_IMMOVABLE(ParsedModule);
    friend Parser;
    Context& ctx;

public:
    using Ptr = std::unique_ptr<ParsedModule>;

    /// Allocator used for allocating strings and the parse tree.
    std::unique_ptr<llvm::BumpPtrAllocator> alloc = std::make_unique<llvm::BumpPtrAllocator>();

    /// Allocator used for integer literals.
    IntegerStorage integers;

    /// Top-level statements.
    SmallVector<ParsedStmt*> top_level;

    /// Template deduction information for each template.
    DenseMap<ParsedProcDecl*, TemplateParamDeductionInfo> template_deduction_infos;

    /// Token streams that are referenced by '#quote's.
    std::vector<std::unique_ptr<TokenStream>> quoted_tokens;

    /// The name of this program or module.
    SLoc name_loc;
    String name;

    /// Whether this is a program or module.
    SLoc program_or_module_loc;
    bool is_module = false;

    /// Imported modules.
    struct Import {
        SmallVector<String> linkage_names; ///< The name of the modules on disk and for linking.
        String import_name;  ///< The name it is imported as.
        SLoc loc;        ///< The location of the import
        bool is_open_import; ///< Whether this uses the 'as *' syntax.
        bool is_header_import;
    };
    SmallVector<Import> imports;

    /// Create a new parse context.
    explicit ParsedModule(Context& ctx) : ctx(ctx) {}

    /// Get the module’s allocator.
    auto allocator() -> llvm::BumpPtrAllocator& { return *alloc; }

    /// Get the owning context.
    Context& context() const { return ctx; }

    /// Dump the contents of the module.
    void dump() const;

    /// Format this module.
    void format() const;
};

// ============================================================================
//  Statements
// ============================================================================
/// Root of the parse tree hierarchy.
class srcc::ParsedStmt {
    struct Printer;
    friend Printer;

public:
    enum struct Kind : u8 {
#define PARSE_TREE_LEAF_NODE(node)             node,
#define PARSE_TREE_INHERITANCE_MARKER(m, node) m = node,
#include "srcc/ParseTree.inc"

    };

    const Kind expr_kind;
    SLoc loc;

protected:
    ParsedStmt(Kind kind, SLoc loc) : expr_kind{kind}, loc{loc} {}

public:
    // Only allow allocating these in the parser.
    void* operator new(usz) = SRCC_DELETED("Use `new (parser) { ... }` instead");
    void* operator new(usz size, Parser& parser);

    auto kind() const -> Kind { return expr_kind; }

    void dump(const ParsedModule* owner, bool use_colour) const;
    void dump(bool use_colour = false) const;
    void dump_color() const { dump(true); }
    auto dump_as_type() -> SmallUnrenderedString;
    auto dump_as_value(const Context* ctx = nullptr) -> SmallUnrenderedString;

    /// Visit this node.
    template <typename Visitor>
    auto visit(Visitor&& v) -> decltype(auto);
};

// ============================================================================
//  Types
// ============================================================================
class srcc::ParsedType : public ParsedStmt {
protected:
    ParsedType(Kind k, SLoc loc): ParsedStmt(k, loc) {}

public:
    static bool classof(const ParsedStmt* e) {
        return e->kind() >= Kind::Type$Start and e->kind() <= Kind::Type$End;
    }
};

class srcc::ParsedBuiltinType final : public ParsedType {
public:
    Type ty;

    ParsedBuiltinType(Type ty, SLoc loc)
        : ParsedType{Kind::BuiltinType, loc},
          ty{ty} {}

    static bool classof(const ParsedStmt* e) {
        return e->kind() == Kind::BuiltinType;
    }
};

class srcc::ParsedIntType final : public ParsedType {
public:
    const Size bit_width;

    ParsedIntType(Size bitwidth, SLoc loc)
        : ParsedType{Kind::IntType, loc}, bit_width{bitwidth} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IntType; }
};

class srcc::ParsedOptionalType final : public ParsedType {
public:
    ParsedStmt* elem;

    ParsedOptionalType(ParsedStmt* elem, SLoc loc)
        : ParsedType{Kind::OptionalType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::OptionalType; }
};

/// This only holds type information. The parameter name is stored
/// in the corresponding declaration instead.
struct srcc::ParsedParameter {
    Intent intent;
    ParsedStmt* type;
    bool variadic;
    ParsedParameter(Intent intent, ParsedStmt* type, bool variadic)
        : intent{intent}, type{type}, variadic{variadic} {}
};

struct ParsedProcAttrs {
    bool extern_ = false;
    bool nomangle = false;
    bool native = false;
    bool c_varargs = false;
    bool builtin_operator = false;
    bool no_mangling_number = false;
    bool inline_ = false;
};

class srcc::ParsedProcType final : public ParsedType
    , llvm::TrailingObjects<ParsedProcType, ParsedParameter> {
    friend TrailingObjects;
    const u32 num_params;
    auto numTrailingObjects(OverloadToken<ParsedParameter>) -> usz { return num_params; }

    ParsedProcType(
        ParsedStmt* ret_type,
        ArrayRef<ParsedParameter> params,
        ParsedProcAttrs attrs,
        SLoc loc
    );

public:
    ParsedStmt* const ret_type;
    ParsedProcAttrs attrs;

    static auto Create(
        Parser& parser,
        ParsedStmt* ret_type,
        ArrayRef<ParsedParameter> params,
        ParsedProcAttrs attrs,
        SLoc loc
    ) -> ParsedProcType*;

    bool has_variadic_param() {
        return rgs::any_of(param_types(), &ParsedParameter::variadic);
    }

    auto param_types() -> ArrayRef<ParsedParameter> {
        return getTrailingObjects(num_params);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ProcType; }
};

class srcc::ParsedPtrType final : public ParsedType {
public:
    ParsedStmt* elem;

    ParsedPtrType(ParsedStmt* elem, SLoc loc)
        : ParsedType{Kind::PtrType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::PtrType; }
};

class srcc::ParsedRangeType final : public ParsedType {
public:
    ParsedStmt* elem;

    ParsedRangeType(ParsedStmt* elem, SLoc loc)
        : ParsedType{Kind::RangeType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::RangeType; }
};

class srcc::ParsedSliceType final : public ParsedType {
public:
    ParsedStmt* elem;

    ParsedSliceType(ParsedStmt* elem, SLoc loc)
        : ParsedType{Kind::SliceType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::SliceType; }
};

class srcc::ParsedTemplateType final : public ParsedType {
public:
    /// The parameter name, *without* the '$' sigil.
    String name;

    ParsedTemplateType(String name, SLoc loc)
        : ParsedType{Kind::TemplateType, loc},
          name{name} {}

    static bool classof(const ParsedStmt* e) {
        return e->kind() == Kind::TemplateType;
    }
};

class srcc::ParsedTypeofType final : public ParsedType {
public:
    ParsedStmt* arg;

    ParsedTypeofType(ParsedStmt* arg, SLoc loc)
        : ParsedType{Kind::TypeofType, loc}, arg{arg} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::TypeofType; }
};

// ============================================================================
//  Expressions
// ============================================================================
class srcc::ParsedAssertExpr final : public ParsedStmt {
public:
    ParsedStmt* cond;
    Ptr<ParsedStmt> message;
    bool is_compile_time;
    SRange cond_range;

    ParsedAssertExpr(
        ParsedStmt* cond,
        Ptr<ParsedStmt> message,
        bool is_compile_time,
        SLoc location,
        SRange cond_range
    ) : ParsedStmt{Kind::AssertExpr, location},
        cond{cond},
        message{std::move(message)},
        is_compile_time{is_compile_time},
        cond_range{cond_range} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::AssertExpr; }
};

class srcc::ParsedBinaryExpr final : public ParsedStmt {
public:
    Tk op;
    ParsedStmt* lhs;
    ParsedStmt* rhs;

    ParsedBinaryExpr(
        Tk op,
        ParsedStmt* lhs,
        ParsedStmt* rhs,
        SLoc location
    ) : ParsedStmt{Kind::BinaryExpr, location}, op{op}, lhs{lhs}, rhs{rhs} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::BinaryExpr; }
};

/// A syntactic block that contains other expressions.
class srcc::ParsedBlockExpr final : public ParsedStmt
    , TrailingObjects<ParsedBlockExpr, ParsedStmt*> {
    friend TrailingObjects;
    const u32 num_stmts;
    auto numTrailingObjects(OverloadToken<ParsedStmt*>) -> usz { return num_stmts; }
    ParsedBlockExpr(ArrayRef<ParsedStmt*> stmts, SLoc location);

public:
    /// Whether this block should create a new scope; this is almost
    /// always true, unless this is the child of e.g. a static '#if'.
    bool should_push_scope = true;

    /// Create a new block.
    static auto Create(
        Parser& parser,
        ArrayRef<ParsedStmt*> stmts,
        SLoc location
    ) -> ParsedBlockExpr*;

    /// Get the statements stored in this block.
    auto stmts() -> ArrayRef<ParsedStmt*> {
        return getTrailingObjects(num_stmts);
    }

    static bool classof(const ParsedStmt* e) {
        return e->kind() == Kind::BlockExpr;
    }
};

class srcc::ParsedBoolLitExpr final : public ParsedStmt {
public:
    bool value;

    ParsedBoolLitExpr(bool value, SLoc location)
        : ParsedStmt{Kind::BoolLitExpr, location}, value{value} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::BoolLitExpr; }
};

class srcc::ParsedBreakContinueExpr final : public ParsedStmt {
public:
    bool is_continue;

    ParsedBreakContinueExpr(bool is_continue, SLoc location)
        : ParsedStmt{Kind::BreakContinueExpr, location}, is_continue{is_continue} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::BreakContinueExpr; }
};

/// A call to a function, or anything that syntactically
/// resembles one.
class srcc::ParsedCallExpr final : public ParsedStmt
    , TrailingObjects<ParsedCallExpr, ParsedStmt*> {
    friend TrailingObjects;

public:
    /// The expression that is called.
    ParsedStmt* callee;

private:
    const u32 num_args;

    auto numTrailingObjects(OverloadToken<ParsedStmt*>) -> usz { return num_args; }

    ParsedCallExpr(
        ParsedStmt* callee,
        ArrayRef<ParsedStmt*> args,
        SLoc location
    );

public:
    static auto Create(
        Parser& parser,
        ParsedStmt* callee,
        ArrayRef<ParsedStmt*> args,
        SLoc location
    ) -> ParsedCallExpr*;

    /// Get the arguments to the call.
    auto args() -> ArrayRef<ParsedStmt*> {
        return getTrailingObjects(num_args);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::CallExpr; }
};

/// A reference to a declaration.
class srcc::ParsedDeclRefExpr final : public ParsedStmt
    , TrailingObjects<ParsedDeclRefExpr, DeclName> {
    friend TrailingObjects;

    u32 num_parts;
    auto numTrailingObjects(OverloadToken<DeclName>) -> usz { return num_parts; }

    ParsedDeclRefExpr(
        ArrayRef<DeclName> names,
        SLoc location
    );

public:
    static auto Create(
        Parser& parser,
        ArrayRef<DeclName> names,
        SLoc location
    ) -> ParsedDeclRefExpr*;

    /// Get the parts of the declaration reference.
    auto names() -> ArrayRef<DeclName> { return getTrailingObjects(num_parts); }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::DeclRefExpr; }
};

class srcc::ParsedDeferStmt final : public ParsedStmt {
public:
    ParsedStmt* body;

    ParsedDeferStmt(
        ParsedStmt* body,
        SLoc location
    ) : ParsedStmt{Kind::DeferStmt, location}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::DeferStmt; }
};

/// A single semicolon.
class srcc::ParsedEmptyStmt final : public ParsedStmt {
public:
    ParsedEmptyStmt(SLoc loc) : ParsedStmt{Kind::EmptyStmt, loc} {}
    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::EmptyStmt; }
};

/// A statement to evaluate.
class srcc::ParsedEvalExpr final : public ParsedStmt {
public:
    ParsedStmt* expr;

    ParsedEvalExpr(
        ParsedStmt* expr,
        SLoc location
    ) : ParsedStmt{Kind::EvalExpr, location}, expr{expr} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::EvalExpr; }
};

/// A for loop.
class srcc::ParsedForStmt final : public ParsedStmt,
    TrailingObjects<ParsedForStmt, std::pair<String, SLoc>, ParsedStmt*> {
    friend TrailingObjects;

    u32 num_idents;
    u32 num_ranges;

public:
    using LoopVar = std::pair<String, SLoc>;

    SLoc enum_loc;
    String enum_name;
    ParsedStmt* body;

private:
    ParsedForStmt(
        SLoc for_loc,
        SLoc enum_loc,
        String enum_name,
        ArrayRef<LoopVar> vars,
        ArrayRef<ParsedStmt*> ranges,
        ParsedStmt* body
    );

    usz numTrailingObjects(OverloadToken<LoopVar>) const { return num_idents; }
    usz numTrailingObjects(OverloadToken<ParsedStmt*>) const { return num_ranges; }

public:
    static auto Create(
        Parser& parser,
        SLoc for_loc,
        SLoc enum_loc,
        String enum_name,
        ArrayRef<LoopVar> vars,
        ArrayRef<ParsedStmt*> ranges,
        ParsedStmt* body
    ) -> ParsedForStmt*;

    bool has_enumerator() const { return not enum_name.empty(); }
    auto ranges() const -> ArrayRef<ParsedStmt*> { return getTrailingObjects<ParsedStmt*>(num_ranges); }
    auto vars() const -> ArrayRef<LoopVar> { return getTrailingObjects<LoopVar>(num_idents); }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ForStmt; }
};

/// A string literal.
class srcc::ParsedStrLitExpr final : public ParsedStmt {
public:
    String value;

    ParsedStrLitExpr(
        String value,
        SLoc location
    ) : ParsedStmt{Kind::StrLitExpr, location}, value{value} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::StrLitExpr; }
};

class srcc::ParsedIfExpr final : public ParsedStmt {
public:
    ParsedStmt* cond;
    ParsedStmt* then;
    Ptr<ParsedStmt> else_;
    bool is_static;

    ParsedIfExpr(
        ParsedStmt* cond,
        ParsedStmt* then,
        Ptr<ParsedStmt> else_,
        bool is_static,
        SLoc location
    ) : ParsedStmt{Kind::IfExpr, location},
        cond{cond},
        then{then},
        else_{else_},
        is_static{is_static} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IfExpr; }
};

class srcc::ParsedInjectExpr final : public ParsedStmt {
public:
    ParsedStmt* injected;

    ParsedInjectExpr(
        ParsedStmt* injected,
        SLoc location
    ) : ParsedStmt{Kind::InjectExpr, location}, injected{injected} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::InjectExpr; }
};

/// An integer literal.
class srcc::ParsedIntLitExpr final : public ParsedStmt {
public:
    StoredInteger storage;

    ParsedIntLitExpr(
        Parser& p,
        APInt value,
        SLoc location
    );

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IntLitExpr; }
};

/// This is used for both the expression and statement form.
class srcc::ParsedLoopExpr final : public ParsedStmt {
public:
    Ptr<ParsedStmt> body;

    ParsedLoopExpr(Ptr<ParsedStmt> body, SLoc location)
        : ParsedStmt{Kind::LoopExpr, location}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::LoopExpr; }
};

struct srcc::ParsedMatchCase {
    ParsedStmt* cond;
    ParsedStmt* body;
    ParsedMatchCase(ParsedStmt* cond, ParsedStmt* body): cond{cond}, body{body} {}
};

/// Pattern matching.
class srcc::ParsedMatchExpr final : public ParsedStmt,
    TrailingObjects<ParsedMatchExpr, ParsedStmt*, ParsedMatchCase> {
    friend TrailingObjects;
    const u32 num_cases : 30;
    const u32 has_control_expr : 1;
    const u32 has_type : 1;

    auto numTrailingObjects(OverloadToken<ParsedStmt*>) const -> usz { return has_control_expr + has_type; }
    auto numTrailingObjects(OverloadToken<ParsedMatchCase>) const -> usz { return num_cases; }

private:
    ParsedMatchExpr(
        Ptr<ParsedStmt> control_expr,
        Ptr<ParsedStmt> declared_type,
        ArrayRef<ParsedMatchCase> cases,
        SLoc loc
    );

public:
    static auto Create(
        Parser& p,
        Ptr<ParsedStmt> control_expr,
        Ptr<ParsedStmt> declared_type,
        ArrayRef<ParsedMatchCase> cases,
        SLoc loc
    ) -> ParsedMatchExpr*;

    [[nodiscard]] auto cases() const -> ArrayRef<ParsedMatchCase> {
        return getTrailingObjects<ParsedMatchCase>(num_cases);
    }

    [[nodiscard]] auto control_expr() const -> Ptr<ParsedStmt> {
        return has_control_expr ? *getTrailingObjects<ParsedStmt*>() : nullptr;
    }

    [[nodiscard]] auto declared_type() const -> Ptr<ParsedStmt> {
        return has_type ? *(getTrailingObjects<ParsedStmt*>() + has_control_expr) : nullptr;
    }

    static bool classof(const ParsedStmt* s) { return s->kind() == Kind::MatchExpr; }
};

/// A member access.
class srcc::ParsedMemberExpr final : public ParsedStmt {
public:
    ParsedStmt* base;
    String member;

    ParsedMemberExpr(
        ParsedStmt* base,
        String member,
        SLoc location
    ) : ParsedStmt{Kind::MemberExpr, location}, base{base}, member{member} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::MemberExpr; }
};

class srcc::ParsedNilExpr final : public ParsedStmt {
public:
    ParsedNilExpr(SLoc location) : ParsedStmt{Kind::NilExpr, location} {}
    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::NilExpr; }
};

class srcc::ParsedParenExpr final : public ParsedStmt {
public:
    ParsedStmt* inner;

    ParsedParenExpr(
        ParsedStmt* inner,
        SLoc location
    ) : ParsedStmt{Kind::ParenExpr, location}, inner{inner} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ParenExpr; }
};

class srcc::ParsedQuoteExpr final : public ParsedStmt
    , TrailingObjects<ParsedQuoteExpr, ParsedUnquoteExpr*> {
    friend TrailingObjects;
    TokenStream* token_stream = nullptr;

public:
    const u32 num_unquotes;
    bool brace_delimited;

private:
    ParsedQuoteExpr(
        TokenStream* tokens,
        ArrayRef<ParsedUnquoteExpr*> unquotes,
        bool brace_delimited,
        SLoc location
    );

public:
    static auto Create(
        Parser& p,
        TokenStream* tokens,
        ArrayRef<ParsedUnquoteExpr*> unquotes,
        bool brace_delimited,
        SLoc location
    ) -> ParsedQuoteExpr*;

    [[nodiscard]] auto tokens() const -> TokenStream::Range {
        if (token_stream == nullptr) return {};
        return TokenStream::Range{token_stream->begin(), token_stream->end()};
    }

    [[nodiscard]] auto unquotes() const -> ArrayRef<ParsedUnquoteExpr*> {
        return getTrailingObjects(num_unquotes);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::QuoteExpr; }
};

/// A return from a function.
class srcc::ParsedReturnExpr final : public ParsedStmt {
public:
    const Ptr<ParsedStmt> value;

    ParsedReturnExpr(
        Ptr<ParsedStmt> value,
        SLoc location
    ) : ParsedStmt{Kind::ReturnExpr, location}, value{std::move(value)} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ReturnExpr; }
};

class srcc::ParsedTupleExpr final : public ParsedStmt,
    TrailingObjects<ParsedTupleExpr, ParsedStmt*> {
    friend TrailingObjects;
    const u32 num_exprs;

    ParsedTupleExpr(ArrayRef<ParsedStmt*> exprs, SLoc loc);

public:
    static auto Create(
        Parser& p,
        ArrayRef<ParsedStmt*> exprs,
        SLoc loc
    ) -> ParsedTupleExpr*;

    auto exprs() -> ArrayRef<ParsedStmt*> { return getTrailingObjects(num_exprs); }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::TupleExpr; }
};

class srcc::ParsedUnaryExpr final : public ParsedStmt {
public:
    Tk op;
    ParsedStmt* arg;
    bool postfix;

    ParsedUnaryExpr(
        Tk op,
        ParsedStmt* arg,
        bool postfix,
        SLoc location
    ) : ParsedStmt{Kind::UnaryExpr, location}, op{op}, arg{arg}, postfix{postfix} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::UnaryExpr; }
};

class srcc::ParsedUnquoteExpr final : public ParsedStmt {
public:
    ParsedStmt* arg;

    ParsedUnquoteExpr(
        ParsedStmt* arg,
        SLoc location
    ) : ParsedStmt{Kind::UnquoteExpr, location}, arg{arg} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::UnquoteExpr; }
};

class srcc::ParsedWhileStmt final : public ParsedStmt {
public:
    ParsedStmt* cond;
    ParsedStmt* body;

    ParsedWhileStmt(
        ParsedStmt* cond,
        ParsedStmt* body,
        SLoc location
    ) : ParsedStmt{Kind::WhileStmt, location}, cond{cond}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::WhileStmt; }
};

class srcc::ParsedWithExpr final : public ParsedStmt {
public:
    ParsedStmt* expr;
    ParsedStmt* body;

    ParsedWithExpr(
        ParsedStmt* expr,
        ParsedStmt* body,
        SLoc location
    ) : ParsedStmt{Kind::WithExpr, location}, expr{expr}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::WithExpr; }
};

// ============================================================================
//  Declarations
// ============================================================================
/// Base class for declarations.
class srcc::ParsedDecl : public ParsedStmt {
public:
    /// The name of the declaration; may be empty if it is
    /// compiler-generated.
    DeclName name;

protected:
    ParsedDecl(
        Kind kind,
        DeclName name,
        SLoc location
    ) : ParsedStmt{kind, location}, name{name} {}

public:
    static bool classof(const ParsedStmt* e) {
        return e->kind() >= Kind::Decl$Start and e->kind() <= Kind::Decl$End;
    }
};

/// An exported declaration.
class srcc::ParsedExportDecl final : public ParsedDecl {
public:
    ParsedDecl* decl;

    ParsedExportDecl(
        ParsedDecl* decl,
        SLoc location
    ) : ParsedDecl{Kind::ExportDecl, String(), location}, decl{decl} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ExportDecl; }
};

class srcc::ParsedFieldDecl final : public ParsedDecl {
public:
    ParsedStmt* type;

    ParsedFieldDecl(
        String name,
        ParsedStmt* type,
        SLoc location
    ) : ParsedDecl{Kind::FieldDecl, name, location}, type{type} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::FieldDecl; }
};

class srcc::ParsedVarDecl final : public ParsedDecl {
public:
    ParsedStmt* type;
    Ptr<ParsedStmt> init;
    Intent intent; // Only used for parameters.
    bool is_static;
    SLoc with_loc; // Only used for parameters.

    ParsedVarDecl(
        String name,
        ParsedStmt* param_type,
        SLoc location,
        Intent intent = Intent::Move,
        bool is_static = false,
        SLoc with = {}
    ) : ParsedDecl{Kind::VarDecl, name, location}, type{param_type}, intent{intent}, is_static{is_static}, with_loc{with} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::VarDecl; }
};

/// A procedure declaration.
class srcc::ParsedProcDecl final : public ParsedDecl
    , llvm::TrailingObjects<ParsedProcDecl, ParsedVarDecl*> {
    friend TrailingObjects;

public:
    /// The body of the procedure.
    ///
    /// This is not present if the procedure is only declared,
    /// not defined.
    Ptr<ParsedStmt> body;

    /// The type of the procedure.
    ParsedProcType* type;

    /// The constraint clause, if any.
    Ptr<ParsedStmt> where;

private:
    auto numTrailingObjects(OverloadToken<ParsedVarDecl*>) -> usz { return type->param_types().size(); }
    ParsedProcDecl(
        DeclName name,
        ParsedProcType* type,
        ArrayRef<ParsedVarDecl*> param_decls,
        Ptr<ParsedStmt> body,
        Ptr<ParsedStmt> where,
        SLoc location
    );

public:
    static auto Create(
        Parser& parser,
        DeclName name,
        ParsedProcType* type,
        ArrayRef<ParsedVarDecl*> param_names,
        Ptr<ParsedStmt> body,
        Ptr<ParsedStmt> where,
        SLoc location
    ) -> ParsedProcDecl*;

    auto params() -> ArrayRef<ParsedVarDecl*> {
        return getTrailingObjects(type->param_types().size());
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ProcDecl; }
};

class srcc::ParsedStructDecl final : public ParsedDecl
    , TrailingObjects<ParsedStructDecl, ParsedFieldDecl*> {
    friend TrailingObjects;

    u32 num_fields;
    auto numTrailingObjects(OverloadToken<ParsedFieldDecl*>) -> usz { return num_fields; }
    ParsedStructDecl(String name, ArrayRef<ParsedFieldDecl*> fields, SLoc loc);

public:
    static auto Create(
        Parser& parser,
        String name,
        ArrayRef<ParsedFieldDecl*> fields,
        SLoc loc
    ) -> ParsedStructDecl*;

    auto fields() -> ArrayRef<ParsedFieldDecl*> {
        return getTrailingObjects(num_fields);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::StructDecl; }
};

template <typename Visitor>
auto srcc::ParsedStmt::visit(Visitor&& v) -> decltype(auto) {
    switch (kind()) {
#       define PARSE_TREE_LEAF_NODE(node) case Kind::node: return std::invoke(v, cast<Parsed##node>(this));
#       include "srcc/ParseTree.inc"
    }
    Unreachable();
}

// ============================================================================
//  Parser
// ============================================================================
struct srcc::LexedTokenStream {
    llvm::BumpPtrAllocator alloc;
    TokenStream tokens{alloc};
};

class srcc::Parser : DiagsProducer {
    SRCC_IMMOVABLE(Parser);

public:
    using CommentTokenCallback = VerifyDiagnosticsEngine::CommentTokenCallback;

private:
    friend DiagsProducer;

    struct Signature {
        SmallVector<ParsedParameter, 10> param_types;
        TemplateParamDeductionInfo deduction_info;
        Ptr<ParsedStmt> ret; // Unset if no return type is parsed.
        Ptr<ParsedStmt> where;
        DeclName name;
        SLoc proc_loc;
        SLoc tok_after_proc;
        ParsedProcAttrs attrs;

        // Mark that a template parameter is deduced in the parameter
        // that is currently being parsed.
        void add_deduced_template_param(String name) {
            deduction_info[name].insert(u32(param_types.size()));
        }
    };

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
    const TokenStream& stream;
    TokenStream::Iterator tok;
    Context& ctx;
    Signature* current_signature = nullptr;
    int num_parens{}, num_brackets{}, num_braces{};
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
    auto ParseDeclRefExpr() -> Ptr<ParsedDeclRefExpr>;
    auto ParseExpr(int precedence = -1, bool expect_type = false) -> Ptr<ParsedStmt>;
    auto ParseForStmt() -> Ptr<ParsedStmt>;
    void ParseFile();
    void ParseHeader();
    auto ParseIf(bool is_static) -> Ptr<ParsedIfExpr>;
    void ParseImport();
    auto ParseIntent() -> std::pair<SLoc, Intent>;
    auto ParseMatchExpr() -> Ptr<ParsedMatchExpr>;
    void ParseOverloadableOperatorName(Signature& sig);
    bool ParseParameter(Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls);
    void ParsePreamble();
    auto ParseProcBody() -> Ptr<ParsedStmt>;
    auto ParseProcDecl() -> Ptr<ParsedProcDecl>;
    auto ParseQuotedTokenSeq(SLoc quote_loc, bool in_macro_call) -> Ptr<ParsedStmt>;
    bool ParseSignature(Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls, bool allow_constraint);
    bool ParseSignatureImpl(SmallVectorImpl<ParsedVarDecl*>* decls, bool allow_constraint);
    void ParseStmts(SmallVectorImpl<ParsedStmt*>& into, Tk stop_at = Tk::Eof);
    auto ParseStmt() -> Ptr<ParsedStmt>;
    auto ParseStructDecl() -> Ptr<ParsedStructDecl>;
    auto ParseType(int precedence = -1) { return ParseExpr(precedence, true); }
    auto ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedVarDecl>;

    auto CreateType(Signature& sig) -> ParsedProcType*;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, SLoc where, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

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
