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
    const File& file;

    /// Allocator used for allocating the parse tree.
    ///
    /// While strings need to stick around for longer, this can be deleted
    /// once we’re done building the AST.
    llvm::BumpPtrAllocator alloc;

public:
    using Ptr = std::unique_ptr<ParsedModule>;

    /// Allocator used for allocating strings.
    std::unique_ptr<llvm::BumpPtrAllocator> string_alloc = std::make_unique<llvm::BumpPtrAllocator>();

    /// Allocator used for integer literals.
    IntegerStorage integers;

    /// Top-level statements.
    SmallVector<ParsedStmt*> top_level;

    /// Template deduction information for each template.
    DenseMap<ParsedProcDecl*, TemplateParamDeductionInfo> template_deduction_infos;

    /// The name of this program or module.
    Location name_loc;
    String name;

    /// Whether this is a program or module.
    Location program_or_module_loc;
    bool is_module = false;

    /// Imported modules.
    struct Import {
        SmallVector<String> linkage_names; ///< The name of the modules on disk and for linking.
        String import_name;  ///< The name it is imported as.
        Location loc;        ///< The location of the import
        bool is_open_import; ///< Whether this uses the 'as *' syntax.
        bool is_header_import;
    };
    SmallVector<Import> imports;

    /// Create a new parse context for a file.
    explicit ParsedModule(const File& file) : file(file) {}

    /// Get the module’s allocator.
    auto allocator() -> llvm::BumpPtrAllocator& { return alloc; }

    /// Get the owning context.
    Context& context() const { return file.context(); }

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
    Location loc;

protected:
    ParsedStmt(Kind kind, Location loc) : expr_kind{kind}, loc{loc} {}

public:
    // Only allow allocating these in the parser.
    void* operator new(usz) = SRCC_DELETED("Use `new (parser) { ... }` instead");
    void* operator new(usz size, Parser& parser);

    auto kind() const -> Kind { return expr_kind; }

    void dump(const ParsedModule* owner, bool use_colour) const;
    void dump(bool use_colour = false) const;
    void dump_color() const { dump(true); }
    auto dump_as_type() -> SmallUnrenderedString;
};

// ============================================================================
//  Types
// ============================================================================
class srcc::ParsedBuiltinType final : public ParsedStmt {
public:
    Type ty;

    ParsedBuiltinType(Type ty, Location loc)
        : ParsedStmt{Kind::BuiltinType, loc},
          ty{ty} {}

    static bool classof(const ParsedStmt* e) {
        return e->kind() == Kind::BuiltinType;
    }
};

class srcc::ParsedIntType final : public ParsedStmt {
public:
    const Size bit_width;

    ParsedIntType(Size bitwidth, Location loc)
        : ParsedStmt{Kind::IntType, loc}, bit_width{bitwidth} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IntType; }
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
};

class srcc::ParsedProcType final : public ParsedStmt
    , llvm::TrailingObjects<ParsedProcType, ParsedParameter> {
    friend TrailingObjects;
    const u32 num_params;
    auto numTrailingObjects(OverloadToken<ParsedParameter>) -> usz { return num_params; }

    ParsedProcType(
        ParsedStmt* ret_type,
        ArrayRef<ParsedParameter> params,
        ParsedProcAttrs attrs,
        Location loc
    );

public:
    ParsedStmt* const ret_type;
    ParsedProcAttrs attrs;

    static auto Create(
        Parser& parser,
        ParsedStmt* ret_type,
        ArrayRef<ParsedParameter> params,
        ParsedProcAttrs attrs,
        Location loc
    ) -> ParsedProcType*;

    bool has_variadic_param() {
        return rgs::any_of(param_types(), &ParsedParameter::variadic);
    }

    auto param_types() -> ArrayRef<ParsedParameter> {
        return getTrailingObjects(num_params);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ProcType; }
};

class srcc::ParsedPtrType final : public ParsedStmt {
public:
    ParsedStmt* elem;

    ParsedPtrType(ParsedStmt* elem, Location loc)
        : ParsedStmt{Kind::PtrType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::PtrType; }
};

class srcc::ParsedRangeType final : public ParsedStmt {
public:
    ParsedStmt* elem;

    ParsedRangeType(ParsedStmt* elem, Location loc)
        : ParsedStmt{Kind::RangeType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::RangeType; }
};

class srcc::ParsedSliceType final : public ParsedStmt {
public:
    ParsedStmt* elem;

    ParsedSliceType(ParsedStmt* elem, Location loc)
        : ParsedStmt{Kind::SliceType, loc}, elem{elem} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::SliceType; }
};

class srcc::ParsedTemplateType final : public ParsedStmt {
public:
    /// The parameter name, *without* the '$' sigil.
    String name;

    ParsedTemplateType(String name, Location loc)
        : ParsedStmt{Kind::TemplateType, loc},
          name{name} {}

    static bool classof(const ParsedStmt* e) {
        return e->kind() == Kind::TemplateType;
    }
};

// ============================================================================
//  Expressions
// ============================================================================
class srcc::ParsedAssertExpr final : public ParsedStmt {
public:
    ParsedStmt* cond;
    Ptr<ParsedStmt> message;
    bool is_compile_time;

    ParsedAssertExpr(
        ParsedStmt* cond,
        Ptr<ParsedStmt> message,
        bool is_compile_time,
        Location location
    ) : ParsedStmt{Kind::AssertExpr, location},
        cond{cond},
        message{std::move(message)},
        is_compile_time{is_compile_time} {}

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
        Location location
    ) : ParsedStmt{Kind::BinaryExpr, location}, op{op}, lhs{lhs}, rhs{rhs} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::BinaryExpr; }
};

/// A syntactic block that contains other expressions.
class srcc::ParsedBlockExpr final : public ParsedStmt
    , TrailingObjects<ParsedBlockExpr, ParsedStmt*> {
    friend TrailingObjects;
    const u32 num_stmts;
    auto numTrailingObjects(OverloadToken<ParsedStmt*>) -> usz { return num_stmts; }
    ParsedBlockExpr(ArrayRef<ParsedStmt*> stmts, Location location);

public:
    /// Whether this block should create a new scope; this is almost
    /// always true, unless this is the child of e.g. a static '#if'.
    bool should_push_scope = true;

    /// Create a new block.
    static auto Create(
        Parser& parser,
        ArrayRef<ParsedStmt*> stmts,
        Location location
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

    ParsedBoolLitExpr(bool value, Location location)
        : ParsedStmt{Kind::BoolLitExpr, location}, value{value} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::BoolLitExpr; }
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
        Location location
    );

public:
    static auto Create(
        Parser& parser,
        ParsedStmt* callee,
        ArrayRef<ParsedStmt*> args,
        Location location
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
        Location location
    );

public:
    static auto Create(
        Parser& parser,
        ArrayRef<DeclName> names,
        Location location
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
        Location location
    ) : ParsedStmt{Kind::DeferStmt, location}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::DeferStmt; }
};

/// A single semicolon.
class srcc::ParsedEmptyStmt final : public ParsedStmt {
public:
    ParsedEmptyStmt(Location loc) : ParsedStmt{Kind::EmptyStmt, loc} {}
    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::EmptyStmt; }
};

/// A statement to evaluate.
class srcc::ParsedEvalExpr final : public ParsedStmt {
public:
    ParsedStmt* expr;

    ParsedEvalExpr(
        ParsedStmt* expr,
        Location location
    ) : ParsedStmt{Kind::EvalExpr, location}, expr{expr} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::EvalExpr; }
};

/// A for loop.
class srcc::ParsedForStmt final : public ParsedStmt,
    TrailingObjects<ParsedForStmt, std::pair<String, Location>, ParsedStmt*> {
    friend TrailingObjects;

    u32 num_idents;
    u32 num_ranges;

public:
    using LoopVar = std::pair<String, Location>;

    Location enum_loc;
    String enum_name;
    ParsedStmt* body;

private:
    ParsedForStmt(
        Location for_loc,
        Location enum_loc,
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
        Location for_loc,
        Location enum_loc,
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
        Location location
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
        Location location
    ) : ParsedStmt{Kind::IfExpr, location},
        cond{cond},
        then{then},
        else_{else_},
        is_static{is_static} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IfExpr; }
};

/// An integer literal.
class srcc::ParsedIntLitExpr final : public ParsedStmt {
public:
    StoredInteger storage;

    ParsedIntLitExpr(
        Parser& p,
        APInt value,
        Location location
    );

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::IntLitExpr; }
};

/// This is used for both the expression and statement form.
class srcc::ParsedLoopExpr final : public ParsedStmt {
public:
    Ptr<ParsedStmt> body;

    ParsedLoopExpr(Ptr<ParsedStmt> body, Location location)
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
        Location loc
    );

public:
    static auto Create(
        Parser& p,
        Ptr<ParsedStmt> control_expr,
        Ptr<ParsedStmt> declared_type,
        ArrayRef<ParsedMatchCase> cases,
        Location loc
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
        Location location
    ) : ParsedStmt{Kind::MemberExpr, location}, base{base}, member{member} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::MemberExpr; }
};

class srcc::ParsedParenExpr final : public ParsedStmt {
public:
    ParsedStmt* inner;

    ParsedParenExpr(
        ParsedStmt* inner,
        Location location
    ) : ParsedStmt{Kind::ParenExpr, location}, inner{inner} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ParenExpr; }
};

/// A return from a function.
class srcc::ParsedReturnExpr final : public ParsedStmt {
public:
    const Ptr<ParsedStmt> value;

    ParsedReturnExpr(
        Ptr<ParsedStmt> value,
        Location location
    ) : ParsedStmt{Kind::ReturnExpr, location}, value{std::move(value)} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ReturnExpr; }
};

class srcc::ParsedTupleExpr final : public ParsedStmt,
    TrailingObjects<ParsedTupleExpr, ParsedStmt*> {
    friend TrailingObjects;
    const u32 num_exprs;

    ParsedTupleExpr(ArrayRef<ParsedStmt*> exprs, Location loc);

public:
    static auto Create(
        Parser& p,
        ArrayRef<ParsedStmt*> exprs,
        Location loc
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
        Location location
    ) : ParsedStmt{Kind::UnaryExpr, location}, op{op}, arg{arg}, postfix{postfix} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::UnaryExpr; }
};

class srcc::ParsedWhileStmt final : public ParsedStmt {
public:
    ParsedStmt* cond;
    ParsedStmt* body;

    ParsedWhileStmt(
        ParsedStmt* cond,
        ParsedStmt* body,
        Location location
    ) : ParsedStmt{Kind::WhileStmt, location}, cond{cond}, body{body} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::WhileStmt; }
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
        Location location
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
        Location location
    ) : ParsedDecl{Kind::ExportDecl, String(), location}, decl{decl} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::ExportDecl; }
};

class srcc::ParsedFieldDecl final : public ParsedDecl {
public:
    ParsedStmt* type;

    ParsedFieldDecl(
        String name,
        ParsedStmt* type,
        Location location
    ) : ParsedDecl{Kind::FieldDecl, name, location}, type{type} {}

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::FieldDecl; }
};

class srcc::ParsedVarDecl final : public ParsedDecl {
public:
    ParsedStmt* type;
    Ptr<ParsedStmt> init;
    Intent intent; // Only used for parameters.
    bool is_static;

    ParsedVarDecl(
        String name,
        ParsedStmt* param_type,
        Location location,
        Intent intent = Intent::Move,
        bool is_static = false
    ) : ParsedDecl{Kind::VarDecl, name, location}, type{param_type}, intent{intent}, is_static{is_static} {}

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

private:
    auto numTrailingObjects(OverloadToken<ParsedVarDecl*>) -> usz { return type->param_types().size(); }
    ParsedProcDecl(
        DeclName name,
        ParsedProcType* type,
        ArrayRef<ParsedVarDecl*> param_decls,
        Ptr<ParsedStmt> body,
        Location location
    );

public:
    static auto Create(
        Parser& parser,
        DeclName name,
        ParsedProcType* type,
        ArrayRef<ParsedVarDecl*> param_names,
        Ptr<ParsedStmt> body,
        Location location
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
    ParsedStructDecl(String name, ArrayRef<ParsedFieldDecl*> fields, Location loc);

public:
    static auto Create(
        Parser& parser,
        String name,
        ArrayRef<ParsedFieldDecl*> fields,
        Location loc
    ) -> ParsedStructDecl*;

    auto fields() -> ArrayRef<ParsedFieldDecl*> {
        return getTrailingObjects(num_fields);
    }

    static bool classof(const ParsedStmt* e) { return e->kind() == Kind::StructDecl; }
};

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
        DeclName name;
        Location proc_loc;
        Location tok_after_proc;
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
        Location left, right;

        BracketTracker(Parser& p, Tk open, bool diagnose = true);
        ~BracketTracker();
        bool close();
        [[nodiscard]] auto span() -> Location { return Location(left, right); }
        [[nodiscard]] auto corresponding_closing_bracket() -> Tk { return close_bracket; }

    private:
        void decrement();
    };

    ParsedModule::Ptr mod;
    TokenStream stream;
    TokenStream::iterator tok;
    Context& ctx;
    Signature* current_signature = nullptr;
    int num_parens{}, num_brackets{}, num_braces{};

public:
    /// Parse a file.
    static auto Parse(
        const File& file,
        CommentTokenCallback comment_callback = {}
    ) -> ParsedModule::Ptr;

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
    explicit Parser(const File& file)
        : mod{std::make_unique<ParsedModule>(file)},
          stream{*mod->string_alloc},
          ctx{file.context()} {}

    /// Each of these corresponds to a production in the grammar.
    auto ParseAssert(bool is_compile_time) -> Ptr<ParsedAssertExpr>;
    auto ParseBlock() -> Ptr<ParsedBlockExpr>;
    auto ParseDeclRefExpr() -> Ptr<ParsedDeclRefExpr>;
    auto ParseExpr(int precedence = -1, bool expect_type = false) -> Ptr<ParsedStmt>;
    auto ParseForStmt() -> Ptr<ParsedStmt>;
    void ParseFile();
    void ParseHeader();
    auto ParseIf(bool is_static, bool is_expr) -> Ptr<ParsedIfExpr>;
    void ParseImport();
    auto ParseIntent() -> std::pair<Location, Intent>;
    auto ParseMatchExpr() -> Ptr<ParsedMatchExpr>;
    void ParseOverloadableOperatorName(Signature& sig);
    bool ParseParameter(Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls);
    void ParsePreamble();
    auto ParseProcDecl() -> Ptr<ParsedProcDecl>;
    bool ParseSignature(Signature& sig, SmallVectorImpl<ParsedVarDecl*>* decls);
    bool ParseSignatureImpl(SmallVectorImpl<ParsedVarDecl*>* decls);
    auto ParseStmt() -> Ptr<ParsedStmt>;
    auto ParseStructDecl() -> Ptr<ParsedStructDecl>;
    auto ParseType(int precedence = -1) { return ParseExpr(precedence, true); }
    auto ParseVarDecl(ParsedStmt* type) -> Ptr<ParsedVarDecl>;

    auto CreateType(Signature& sig) -> ParsedProcType*;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    using DiagsProducer::Error;
    template <typename... Args>
    auto Error(std::format_string<Args...> fmt, Args&&... args) -> std::nullptr_t {
        return Error(tok->location, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto ErrorSync(Location loc, std::format_string<Args...> fmt, Args&&... args) -> std::nullptr_t {
        Error(loc, fmt, std::forward<Args>(args)...);
        SkipTo(Tk::Semicolon);
        return {};
    }

    template <typename... Args>
    auto ErrorSync(std::format_string<Args...> fmt, Args&&... args) -> std::nullptr_t {
        Error(tok->location, fmt, std::forward<Args>(args)...);
        SkipTo(Tk::Semicolon);
        return {};
    }

    /// Consume a token or issue an error.
    bool ConsumeOrError(Tk tk);

    /// Check if we’re at a token.
    bool At(std::same_as<Tk> auto... tks) {
        return tok->is(tks...);
    }

    /// Check if we’re at the start of an expression.
    bool AtStartOfExpression();

    /// Consume a token if it is present.
    bool Consume(Tk tk);
    bool Consume(Location& into, Tk tk);

    /// Consume a contextual keyword.
    bool ConsumeContextual(StringRef keyword);
    bool ConsumeContextual(Location& into, StringRef keyword);

    /// Consume a token if we’re at it, and issue an error about it otherwise.
    template <typename... Args>
    bool ExpectAndConsume(Tk t, std::format_string<Args...> fmt, Args&&... args) {
        if (Consume(t)) return true;
        Error(fmt, std::forward<Args>(args)...);
        return false;
    }

    /// Consume a semicolon and issue an error on the previous line if it is missing.
    bool ExpectSemicolon();

    /// Check if a token is one of (), [], {}.
    bool IsBracket(Tk t);

    /// Check if a token is a keyword (implemented in the lexer).
    bool IsKeyword(Tk t);

    /// Get a lookahead token; returns EOF if looking past the end.
    auto LookAhead(usz n = 1) -> Token&;

    /// Consume a token and return it (or its location). If the parser is at
    /// end of file, the token iterator is not advanced.
    auto Next() -> Location;

    /// Actually advance to the next token. This only exists to implement Next(),
    /// SkipTo(), and BracketTracker; do not call this from anywhere else!
    auto NextTokenImpl() -> Location;

    /// Skip up to a token.
    bool SkipTo(std::same_as<Tk> auto... tks);

    /// Skip up and past a token.
    bool SkipPast(std::same_as<Tk> auto... tks);

    /// Read all tokens from a file.
    static void ReadTokens(TokenStream& s, const File& file, CommentTokenCallback cb);
};

#endif // SRCC_FRONTEND_PARSER_HH
