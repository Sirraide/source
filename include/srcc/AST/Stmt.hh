#ifndef SRCC_AST_STMT_HH
#define SRCC_AST_STMT_HH

#include <srcc/AST/DeclName.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Core/Token.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/TrailingObjects.h>

#include <functional>
#include <ranges>

namespace clang {
class ASTUnit;
}

namespace srcc {
#define AST_STMT(node) class node;
#include "srcc/AST.inc"

#define SRCC_ALL_BUILTINS(F)   \
    F(Memcpy, "__srcc_memcpy") \
    F(Ptradd, "__srcc_ptradd") \
    F(Unreachable, "__srcc_unreachable")

struct MatchCase;
class ParsedStmt;
class ParsedProcDecl;

// Token to identify a loop.
enum class LoopToken : u32;
constexpr void operator--(LoopToken& t) {
    Assert(+t != 0);
    t = LoopToken(+t - 1);
}

constexpr void operator++(LoopToken& t) {
    t = LoopToken(+t + 1);
}
} // namespace srcc

// ============================================================================
//  Statements
// ============================================================================
/// Root of the AST inheritance hierarchy.
class srcc::Stmt {
    SRCC_IMMOVABLE(Stmt);
    struct Printer;
    friend Printer;

public:
    enum struct Kind : u8 {
#define AST_STMT_LEAF(node) node,
#include "srcc/AST.inc"

    };

private:
    /// The kind of this statement.
    const Kind stmt_kind;

    /// Source location of this statement.
    SLoc loc;

protected:
    explicit Stmt(Kind kind, SLoc loc) : stmt_kind{kind}, loc{loc} {}

public:
    // Only allow allocating these in the module.
    void* operator new(usz) = SRCC_DELETED("Use `new (tu) { ... }` instead");
    void* operator new(usz size, TranslationUnit& tu);

    /// Dump the statement.
    void dump(bool use_colour = false) const;
    void dump_color() const { dump(true); }

    /// Get the kind of this statement.
    Kind kind() const { return stmt_kind; }

    /// Get the source location of this statement.
    auto location() const -> SLoc { return loc; }

    /// Get the type of this if it is an expression and Void otherwise.
    auto type_or_void() const -> Type;

    /// Get the value category of this if it is an expression and RValue otherwise.
    auto value_category_or_rvalue() const -> ValueCategory;

    /// Visit this statement.
    template <typename Visitor>
    auto visit(Visitor&& v) -> decltype(auto);
};

class srcc::DeferStmt : public Stmt {
public:
    Stmt* body;
    SLoc loc;
    DeferStmt(Stmt* body, SLoc loc) : Stmt{Kind::DeferStmt, loc}, body{body} {}
    static bool classof(const Stmt* e) { return e->kind() == Kind::DeferStmt; }
};

class srcc::EmptyStmt : public Stmt {
public:
    SLoc loc;
    EmptyStmt(SLoc loc) : Stmt{Kind::EmptyStmt, loc} {}
    static bool classof(const Stmt* e) { return e->kind() == Kind::EmptyStmt; }
};

class srcc::ForStmt final : public Stmt
    , TrailingObjects<ForStmt, LocalDecl*, Expr*> {
    friend TrailingObjects;
    u32 num_vars, num_ranges;

public:
    LoopToken token;
    Ptr<LocalDecl> enum_var;
    Stmt* body;

private:
    ForStmt(
        LoopToken token,
        Ptr<LocalDecl> enum_var,
        ArrayRef<LocalDecl*> vars,
        ArrayRef<Expr*> ranges,
        Stmt* body,
        SLoc location
    );

    usz numTrailingObjects(OverloadToken<LocalDecl*>) const { return num_vars; }
    usz numTrailingObjects(OverloadToken<Expr*>) const { return num_ranges; }

public:
    static auto Create(
        TranslationUnit& tu,
        LoopToken token,
        Ptr<LocalDecl> enum_var,
        ArrayRef<LocalDecl*> vars,
        ArrayRef<Expr*> ranges,
        Stmt* body,
        SLoc location
    ) -> ForStmt*;

    auto ranges() const -> ArrayRef<Expr*> { return getTrailingObjects<Expr*>(num_ranges); }
    auto vars() const -> ArrayRef<LocalDecl*> { return getTrailingObjects<LocalDecl*>(num_vars); }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ForStmt; }
};

class srcc::WhileStmt : public Stmt {
public:
    LoopToken token;
    Expr* cond;
    Stmt* body;

    WhileStmt(
        LoopToken token,
        Expr* cond,
        Stmt* body,
        SLoc location
    ) : Stmt{Kind::WhileStmt, location}, token{token}, cond{cond}, body{body} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::WhileStmt; }
};

// ============================================================================
//  Expressions
// ============================================================================
class srcc::Expr : public Stmt {
public:
    /// The type of this expression.
    Type type;

    /// The value category of this expression.
    ValueCategory value_category;

protected:
    Expr(
        Kind kind,
        Type type,
        ValueCategory category,
        SLoc location
    ) : Stmt(kind, location), type{type}, value_category{category} {}

public:
    using enum ValueCategory;

    /// Look through any parentheses.
    [[nodiscard]] auto ignore_parens() -> Expr*;

    /// Check if this expression is an lvalue.
    [[nodiscard]] bool is_lvalue() const { return value_category == LValue; }

    /// Check if this expression is an rvalue.
    [[nodiscard]] bool is_rvalue() const { return value_category == RValue; }

    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::ArrayBroadcastExpr and e->kind() <= Kind::UnaryExpr;
    }
};

class srcc::ArrayBroadcastExpr final : public Expr {
public:
    Expr* element;

    ArrayBroadcastExpr(ArrayType* type, Expr* element, SLoc loc)
        : Expr{Kind::ArrayBroadcastExpr, type, RValue, loc}, element{element} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::ArrayBroadcastExpr; }
};

class srcc::ArrayInitExpr final : public Expr
    , TrailingObjects<ArrayInitExpr, Expr*> {
    friend TrailingObjects;
    u32 num_inits;

    ArrayInitExpr(ArrayType* type, ArrayRef<Expr*> element, SLoc loc);

public:
    static auto Create(
        TranslationUnit& tu,
        ArrayType* type,
        ArrayRef<Expr*> element,
        SLoc loc
    ) -> ArrayInitExpr*;

    [[nodiscard]] auto broadcast_init() const -> Expr* { return initialisers().back(); }
    [[nodiscard]] auto initialisers() const -> ArrayRef<Expr*> { return getTrailingObjects(num_inits); }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ArrayInitExpr; }
};

class srcc::AssertExpr final : public Expr {
public:
    Expr* cond;
    Ptr<Expr> message;
    bool is_static;
    SRange cond_range;
    Ptr<ProcDecl> stringifier;

    AssertExpr(
        Expr* cond,
        Ptr<Expr> message,
        bool is_static,
        SLoc location,
        SRange cond_range,
        Ptr<ProcDecl> stringifier
    ) : Expr{Kind::AssertExpr, Type::VoidTy, RValue, location},
        cond{cond},
        message{message},
        is_static{is_static},
        cond_range{cond_range},
        stringifier{stringifier} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::AssertExpr; }
};

class srcc::BinaryExpr final : public Expr {
public:
    Tk op;
    Expr* lhs;
    Expr* rhs;

    BinaryExpr(
        Type type,
        ValueCategory cat,
        Tk op,
        Expr* lhs,
        Expr* rhs,
        SLoc location
    ) : Expr{Kind::BinaryExpr, type, cat, location},
        op{op},
        lhs{lhs},
        rhs{rhs} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BinaryExpr; }
};

class srcc::BlockExpr final : public Expr
    , TrailingObjects<BlockExpr, Stmt*> {
    friend TrailingObjects;
    const u32 num_stmts;
    auto numTrailingObjects(OverloadToken<Stmt*>) -> usz { return num_stmts; }

public:
    /// Scope associated with this block.
    Scope* scope;

private:
    BlockExpr(
        Scope* parent_scope,
        Type type,
        ArrayRef<Stmt*> stmts,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& mod,
        Scope* parent_scope,
        ArrayRef<Stmt*> stmts,
        SLoc location
    ) -> BlockExpr*;

    /// Get the statements in this block.
    auto stmts() -> ArrayRef<Stmt*> { return getTrailingObjects(num_stmts); }

    /// Get the expression whose value is returned from this block, if any.
    auto return_expr() -> Expr*;

    static bool classof(const Stmt* e) { return e->kind() == Kind::BlockExpr; }
};

class srcc::BoolLitExpr final : public Expr {
public:
    const bool value;

    BoolLitExpr(
        bool value,
        SLoc location
    ) : Expr{Kind::BoolLitExpr, Type::BoolTy, RValue, location}, value{value} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BoolLitExpr; }
};

class srcc::BreakContinueExpr final : public Expr {
public:
    const bool is_continue;
    LoopToken target_loop;

    BreakContinueExpr(bool is_continue, LoopToken target_loop, SLoc location)
    : Expr{Kind::BreakContinueExpr, Type::NoReturnTy, RValue, location},
    is_continue{is_continue},
    target_loop{target_loop} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BreakContinueExpr; }
};

class srcc::BuiltinCallExpr final : public Expr
    , TrailingObjects<BuiltinCallExpr, Expr*> {
    friend TrailingObjects;

public:
    enum struct Builtin : u8 {
        #define F(enumerator, ...) enumerator,
        SRCC_ALL_BUILTINS(F)
        #undef F
    };

    const Builtin builtin;

private:
    const u32 num_args;
    auto numTrailingObjects(OverloadToken<Expr*>) -> usz { return num_args; }

    BuiltinCallExpr(
        Builtin kind,
        Type return_type,
        ValueCategory vc,
        ArrayRef<Expr*> args,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        Builtin kind,
        Type return_type,
        ValueCategory vc,
        ArrayRef<Expr*> args,
        SLoc location
    ) -> BuiltinCallExpr*;

    /// Get the arguments.
    auto args() -> ArrayRef<Expr*> { return getTrailingObjects(num_args); }

    [[nodiscard]] static auto ToString(Builtin b) -> String;
    [[nodiscard]] static auto Parse(StringRef s) -> Opt<Builtin>;

    static bool classof(const Stmt* e) { return e->kind() == Kind::BuiltinCallExpr; }
};

class srcc::BuiltinMemberAccessExpr final : public Expr {
public:
    enum struct AccessKind : u8 {
        SliceData,
        SliceSize,
        RangeStart,
        RangeEnd,
        TypeAlign,
        TypeArraySize,
        TypeBits,
        TypeBytes,
        TypeName,
        TypeMaxVal,
        TypeMinVal,
    };

    Expr* operand;
    const AccessKind access_kind;

    BuiltinMemberAccessExpr(
        Type type,
        ValueCategory cat,
        Expr* operand,
        AccessKind kind,
        SLoc location
    ) : Expr{Kind::BuiltinMemberAccessExpr, type, cat, location},
        operand{operand},
        access_kind{kind} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BuiltinMemberAccessExpr; }
};

class srcc::CallExpr final : public Expr
    , TrailingObjects<CallExpr, Expr*> {
    friend TrailingObjects;

public:
    Expr* callee;

private:
    const u32 num_args;

    auto numTrailingObjects(OverloadToken<Expr*>) -> usz { return num_args; }

    CallExpr(
        Type type,
        ValueCategory vc,
        Expr* callee,
        ArrayRef<Expr*> args,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& mod,
        Type type,
        ValueCategory vc,
        Expr* callee,
        ArrayRef<Expr*> args,
        SLoc location
    ) -> CallExpr*;

    [[nodiscard]] auto args() -> ArrayRef<Expr*> { return getTrailingObjects(num_args); }

    static bool classof(const Stmt* e) { return e->kind() == Kind::CallExpr; }
};

class srcc::CastExpr final : public Expr {
public:
    enum class CastKind : u8 {
        /// Convert an rvalue 'T^' to an lvalue 'T'.
        Deref,

        /// This is a cast to void.
        ExplicitDiscard,

        /// Cast an rvalue integer to an rvalue integer.
        Integral,

        /// Cast a pointer to a pointer.
        Pointer,

        /// Convert an lvalue to an rvalue.
        LValueToRValue,

        /// Materialise a poison value of the given type.
        MaterialisePoisonValue,

        /// Convert between range types.
        Range,

        /// Convert an array to a slice.
        SliceFromArray,
    };

    using enum CastKind;

    Expr* arg;
    CastKind kind;
    bool implicit;

    CastExpr(
        Type type,
        CastKind kind,
        Expr* expr,
        SLoc location,
        bool implicit = false,
        ValueCategory value_category = RValue
    ) : Expr{Kind::CastExpr, type, value_category, location},
        arg{expr},
        kind{kind},
        implicit{implicit} {}

    static auto Dereference(TranslationUnit& tu, Expr* expr) -> CastExpr* {
        return new (tu) CastExpr{
            cast<PtrType>(expr->type)->elem(),
            Deref,
            expr,
            expr->location(),
            true,
            LValue
        };
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::CastExpr; }
};

class srcc::ConstExpr final : public Expr {
public:
    /// Constant value.
    eval::RValue* value;

    /// Evaluated statement. May be unset if this was created artificially.
    Ptr<Stmt> stmt;

    ConstExpr(
        TranslationUnit& tu,
        eval::RValue value,
        SLoc location,
        Ptr<Stmt> stmt = {}
    );

    static bool classof(const Stmt* e) { return e->kind() == Kind::ConstExpr; }
};

/// Default-initialise a type.
class srcc::DefaultInitExpr final : public Expr {
public:
    DefaultInitExpr(
        Type type,
        SLoc location
    ) : Expr{Kind::DefaultInitExpr, type, RValue, location} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::DefaultInitExpr; }
};

/// This is wrapped with a 'ConstExpr' after evaluation, so the
/// type of this itself is irrelevant.
class srcc::EvalExpr final : public Expr {
public:
    Stmt* stmt;

    EvalExpr(
        Stmt* stmt,
        SLoc location
    ) : Expr{Kind::EvalExpr, stmt->type_or_void(), LValue /* dummy */, location}, stmt{stmt} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::EvalExpr; }
};

class srcc::IfExpr final : public Expr {
public:
    Expr* cond;
    Stmt* then;
    Ptr<Stmt> else_;
    bool is_static;

    IfExpr(
        Type type,
        ValueCategory val,
        Expr* cond,
        Stmt* then,
        Ptr<Stmt> else_,
        bool is_static,
        SLoc location
    ) : Expr{Kind::IfExpr, type, val, location},
        cond{cond},
        then{then},
        else_{else_},
        is_static{is_static} {}

    [[nodiscard]] bool has_yield() const {
        return type != Type::VoidTy and type != Type::NoReturnTy;
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::IfExpr; }
};

class srcc::IntLitExpr final : public Expr {
public:
    StoredInteger storage;

    IntLitExpr(
        Type ty,
        StoredInteger integer,
        SLoc location
    ) : Expr{Kind::IntLitExpr, ty, RValue, location}, storage{integer} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::IntLitExpr; }
};

class srcc::LocalRefExpr final : public Expr {
public:
    LocalDecl* decl;
    LocalRefExpr(LocalDecl* decl, ValueCategory vc, SLoc location);
    static bool classof(const Stmt* e) { return e->kind() == Kind::LocalRefExpr; }
};

class srcc::LoopExpr final : public Expr {
public:
    LoopToken token;
    Ptr<Stmt> body;

    LoopExpr(LoopToken token, Ptr<Stmt> body, Type ty, SLoc loc)
        : Expr{Kind::LoopExpr, ty, RValue, loc},
          token{token},
          body{body} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::LoopExpr; }
};

class srcc::MaterialiseTemporaryExpr final : public Expr {
public:
    Expr* temporary;

    MaterialiseTemporaryExpr(Expr* temporary, SLoc location)
        : Expr(Kind::MaterialiseTemporaryExpr, temporary->type, LValue, location),
          temporary{temporary} {}

    static bool classof(const Stmt* e) {
        return e->kind() == Kind::MaterialiseTemporaryExpr;
    }
};

struct srcc::MatchCase {
    class Pattern {
        Expr* data = nullptr;

        Pattern() = default;

    public:
        Pattern(Expr* e) : data{e} { Assert(e); }

        /// Get the wildcard pattern.
        static auto Wildcard() -> Pattern { return Pattern(); }

        /// Check if this is a wildcard pattern.
        [[nodiscard]] bool is_wildcard() const { return data == nullptr; }

        /// Get the expression that makes up this pattern.
        [[nodiscard]] auto expr() const -> Expr* {
            Assert(not is_wildcard());
            return data;
        }

        /// Get the expression that makes up this pattern.
        [[nodiscard]] auto expr() -> Expr*& {
            Assert(not is_wildcard());
            return data;
        }
    };

    Pattern cond;
    Stmt* body;
    SLoc loc;
    bool unreachable = false;
    MatchCase(Pattern cond, Stmt* body, SLoc loc) : cond{cond}, body{body}, loc{loc} {}
};

class srcc::MatchExpr final : public Expr,
    TrailingObjects<MatchExpr, Expr*, MatchCase> {
    friend TrailingObjects;
    const u32 num_cases : 31;
    const u32 has_control_expr : 1;

    auto numTrailingObjects(OverloadToken<Expr*>) const -> usz { return has_control_expr; }
    auto numTrailingObjects(OverloadToken<MatchCase>) const -> usz { return num_cases; }

private:
    MatchExpr(
        Ptr<Expr> control_expr,
        Type ty,
        ValueCategory vc,
        ArrayRef<MatchCase> cases,
        SLoc loc
    );

public:
    static auto Create(
        TranslationUnit& tu,
        Ptr<Expr> control_expr,
        Type ty,
        ValueCategory vc,
        ArrayRef<MatchCase> cases,
        SLoc loc
    ) -> MatchExpr*;

    [[nodiscard]] auto cases() -> ArrayRef<MatchCase> {
        return getTrailingObjects<MatchCase>(num_cases);
    }

    [[nodiscard]] auto control_expr() -> Ptr<Expr> {
        return has_control_expr ? *getTrailingObjects<Expr*>() : nullptr;
    }

    static auto classof(const Stmt* s) { return s->kind() == Kind::MatchExpr; }
};

class srcc::MemberAccessExpr final : public Expr {
public:
    Expr* base;
    FieldDecl* field;

    MemberAccessExpr(
        Expr* base,
        FieldDecl* field,
        SLoc location
    );

    static bool classof(const Stmt* e) { return e->kind() == Kind::MemberAccessExpr; }
};
class srcc::OverloadSetExpr final : public Expr
    , TrailingObjects<OverloadSetExpr, Decl*> {
    friend TrailingObjects;
    const u32 num_overloads;
    auto numTrailingObjects(OverloadToken<Decl*>) -> usz { return num_overloads; }

    OverloadSetExpr(ArrayRef<Decl*> overloads, SLoc location);

public:
    static auto Create(
        TranslationUnit& tu,
        ArrayRef<Decl*> overloads,
        SLoc location
    ) -> OverloadSetExpr*;

    auto name() -> DeclName;
    auto overloads() -> ArrayRef<Decl*> { return getTrailingObjects(num_overloads); }

    static bool classof(const Stmt* e) { return e->kind() == Kind::OverloadSetExpr; }
};

class srcc::ParenExpr final : public Expr {
public:
    Expr* expr;

    ParenExpr(Expr* expr, SLoc loc)
        : Expr{Kind::ParenExpr, expr->type, expr->value_category, loc},
          expr{expr} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::ParenExpr; }
};

class srcc::ProcRefExpr final : public Expr {
public:
    ProcDecl* decl;

    ProcRefExpr(ProcDecl* decl, SLoc location);

    auto return_type() const -> Type;

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcRefExpr; }
};

class srcc::SliceConstructExpr final : public Expr {
public:
    Expr* ptr;
    Expr* size;

    SliceConstructExpr(SliceType* slice, Expr* ptr, Expr* size, SLoc loc)
        : Expr{Kind::SliceConstructExpr, slice, RValue, loc}, ptr{ptr}, size{size} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::SliceConstructExpr; }
};

class srcc::StrLitExpr final : public Expr {
public:
    String value;

private:
    StrLitExpr(
        Type ty,
        String value,
        SLoc location
    ) : Expr{Kind::StrLitExpr, ty, RValue, location}, value{value} {}

public:
    static auto Create(TranslationUnit& mod, String value, SLoc location) -> StrLitExpr*;

    static bool classof(const Stmt* e) { return e->kind() == Kind::StrLitExpr; }
};

class srcc::TupleExpr final : public Expr
    , TrailingObjects<TupleExpr, Expr*> {
    friend TrailingObjects;
    TupleExpr(
        RecordType* ty,
        ArrayRef<Expr*> fields,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        RecordType* type,
        ArrayRef<Expr*> fields,
        SLoc location
    ) -> TupleExpr*;

    /// Whether this is the empty tuple, i.e. '()' or ‘nil’.
    bool is_nil() { return not is_struct() and num_values() == 0; }

    /// Get the number of expressions in this tuple.
    auto num_values() -> u32 { return u32(record_type()->layout().fields().size()); }

    /// Get the type of this expression.
    auto record_type() const -> RecordType* { return cast<RecordType>(type.ptr()); }

    /// Whether this tuple is actually a struct initialiser.
    bool is_struct() { return isa<StructType>(type); }

    /// Get the expressions that make up this tuple.
    auto values() -> ArrayRef<Expr*> {
        return getTrailingObjects(num_values());
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::TupleExpr; }
};

class srcc::TypeExpr final : public Expr {
public:
    Type value;
    TypeExpr(
        Type type,
        SLoc location
    ) : Expr{Kind::TypeExpr, Type::TypeTy, RValue, location}, value{type} {}
    static bool classof(const Stmt* e) { return e->kind() == Kind::TypeExpr; }
};

class srcc::ReturnExpr final : public Expr {
public:
    Ptr<Expr> value;
    bool implicit;

    ReturnExpr(
        Ptr<Expr> value,
        SLoc location,
        bool implicit = false
    ) : Expr{
            Kind::ReturnExpr,
            Type::NoReturnTy,
            RValue,
            location,
        },
        value{value}, implicit{implicit} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::ReturnExpr; }
};

class srcc::UnaryExpr final : public Expr {
public:
    Tk op;
    Expr* arg;
    bool postfix;

    UnaryExpr(
        Type type,
        ValueCategory val,
        Tk op,
        Expr* arg,
        bool postfix,
        SLoc location
    ) : Expr{Kind::UnaryExpr, type, val, location},
        op{op},
        arg{arg},
        postfix{postfix} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::UnaryExpr; }
};

// ============================================================================
//  Declarations
// ============================================================================
class srcc::Decl : public Stmt {
public:
    DeclName name;
    bool is_valid = true;

protected:
    Decl(
        Kind kind,
        DeclName name,
        SLoc location
    ) : Stmt{kind, location}, name{name} {}

public:
    /// Mark this declaration as invalid and return itself for convenience.
    ///
    /// We prefer this over simply discarding it from the AST since decls
    /// may be referenced in other places. If a decl is ‘invalid’, then its
    /// type should be considered nonsense.
    auto set_invalid() -> Decl* {
        is_valid = false;
        return this;
    }

    /// Check whether this declaration is valid.
    [[nodiscard]] bool valid() const { return is_valid; }

    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::FieldDecl;
    }
};

class srcc::FieldDecl final : public Decl {
public:
    Type type;
    Size offset;

    FieldDecl(
        Type type,
        Size offset,
        String name,
        SLoc location
    ) : Decl{Kind::FieldDecl, name, location}, type{type}, offset{offset} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::FieldDecl; }
};

// A struct or type alias decl.
class srcc::TypeDecl : public Decl {
    friend StructType;

public:
    Type type;

protected:
    TypeDecl(
        Kind k,
        Type type,
        String name,
        SLoc location
    ) : Decl{k, name, location}, type{type} {}

public:
    TypeDecl(
        Type type,
        String name,
        SLoc location
    ) : TypeDecl{Kind::TypeDecl, type, name, location} {}

public:
    static bool classof(const Stmt* e) {
        return e->kind() == Kind::TypeDecl || e->kind() == Kind::TemplateTypeParamDecl;
    }
};

class srcc::TemplateTypeParamDecl final : public TypeDecl {
public:
    // Whether we’re currently substituting the type of the declaration that this
    // belongs to. This is used to allow translation of '$T' only in the parameter
    // list of a template.
    //
    // Note that this is only true during the *substitution* phase.
    bool in_substitution = true;

    TemplateTypeParamDecl(
        String name,
        TypeLoc tl
    ) : TypeDecl{Kind::TemplateTypeParamDecl, tl.ty, name, tl.loc} {}

    /// Get the template argument type bound to this parameter.
    auto arg_type() -> Type { return type; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::TemplateTypeParamDecl; }
};

/// Base class for declarations that represent modules.
class srcc::ModuleDecl : public Decl {
protected:
    ModuleDecl(
        Kind k,
        String logical_name,
        SLoc loc
    ) : Decl{k, logical_name, loc} {}

public:
    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::ImportedClangModuleDecl and
               e->kind() <= Kind::ImportedSourceModuleDecl;
    }
};

/// Class representing an imported Clang module.
class srcc::ImportedClangModuleDecl final : public ModuleDecl
    , TrailingObjects<ImportedClangModuleDecl, String> {
    friend TrailingObjects;
    const u32 num_headers;

public:
    clang::ASTUnit& clang_ast;

private:
    ImportedClangModuleDecl(
        clang::ASTUnit& clang_ast,
        String logical_name,
        ArrayRef<String> header_names,
        SLoc loc
    );

public:
    static auto Create(
        TranslationUnit& tu,
        clang::ASTUnit& clang_ast,
        String logical_name,
        ArrayRef<String> header_names,
        SLoc loc
    ) -> ImportedClangModuleDecl*;

    /// Get the headers that make up this module.
    auto headers() const -> ArrayRef<String> {
        return getTrailingObjects(num_headers);
    }

    static bool classof(const Stmt* e) {
        return e->kind() == Kind::ImportedClangModuleDecl;
    }
};

/// Class representing an imported Source module.
class srcc::ImportedSourceModuleDecl : public ModuleDecl {
public:
    Scope& exports;

    /// The path that we need to link against.
    String mod_path;

    /// The ‘linkage name’ is the actual name of the imported module or the
    /// actual header name, as opposed to the logical name, which is the name
    /// given to it when imported. E.g. in
    ///
    ///     import foo as bar;
    ///
    /// ‘foo’ is the linkage name and ‘bar’ the logical name.
    String linkage_name;

    ImportedSourceModuleDecl(
        Scope& exports,
        String logical_name,
        String linkage_name,
        String mod_path,
        SLoc loc
    ) : ModuleDecl{Kind::ImportedSourceModuleDecl, logical_name, loc},
        exports{exports},
        mod_path{mod_path},
        linkage_name{linkage_name} {
        Assert(not linkage_name.empty());
    }

    static bool classof(const Stmt* e) {
        return e->kind() == Kind::ImportedSourceModuleDecl;
    }
};

class srcc::LocalDecl : public Decl {
public:
    /// The immediate parent whose stack frame this belongs to.
    ProcDecl* parent;

    /// The type of this decl.
    Type type;

    /// The value category of this decl.
    ///
    /// This is usually LValue, but may be SRValue for some in parameters
    /// and loop variables.
    ValueCategory category;

    /// Whether this declaration is captured by a nested procedure.
    bool captured = false;

    /// Initialiser, if any.
    ///
    /// For SRValues, this is a normal expression that is emitted
    /// and then stored into the memory location. For other values,
    /// this is an initialiser that is evaluated with the memory
    /// location as an argument.
    Ptr<Expr> init;

protected:
    LocalDecl(
        Kind k,
        Type type,
        ValueCategory category,
        String name,
        ProcDecl* parent,
        SLoc location
    ) : Decl{k, name, location},
        parent{parent},
        type{type},
        category{category} {}

public:
    LocalDecl(
        Type type,
        ValueCategory category,
        String name,
        ProcDecl* parent,
        SLoc location
    ) : LocalDecl{Kind::LocalDecl, type, category, name, parent, location} {}

    /// Set the initialiser of this declaration.
    void set_init(Ptr<Expr> expr) { init = expr; }

    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::LocalDecl and e->kind() <= Kind::ParamDecl;
    }
};

class srcc::ParamDecl : public LocalDecl {
    u32 idx;
    bool with;
public:
    ParamDecl(
        const ParamTypeData* param,
        ValueCategory vc,
        String name,
        ProcDecl* parent,
        u32 index,
        bool with_param,
        SLoc location
    ) : LocalDecl{Kind::ParamDecl, param->type, vc, name, parent, location},
        idx{index},
        with{with_param} {
    }

    /// Get the parameter’s index.
    [[nodiscard]] auto index() const -> u32 { return idx; }

    /// Get the parameter’s intent.
    [[nodiscard]] auto intent() const -> Intent;

    /// Whether this is a 'with' parameter.
    [[nodiscard]] bool is_with_param() const { return with; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ParamDecl; }
};
/// Declaration with linkage.
class srcc::ObjectDecl : public Decl {
public:
    TranslationUnit* owner;
    ModuleDecl* imported_from_module;
    Linkage linkage;
    Mangling mangling;
    Type type;

protected:
    ObjectDecl(
        Kind kind,
        TranslationUnit* owner,
        ModuleDecl* imported_from_module,
        Type type,
        DeclName name,
        Linkage linkage,
        Mangling mangling,
        SLoc location
    ) : Decl{kind, name, location},
        owner{owner},
        imported_from_module{imported_from_module},
        linkage{linkage},
        mangling{mangling},
        type{type} {}

public:
    [[nodiscard]] bool is_imported() const {
        return linkage == Linkage::Imported or linkage == Linkage::Reexported;
    }

    static bool classof(const Stmt* e) { return e->kind() >= Kind::LocalDecl; }
};

/// Procedure declaration.
class srcc::ProcDecl final : public ObjectDecl {
    /// Not set if this is e.g. external.
    Ptr<Stmt> body_stmt;

public:
    /// May be null if this is a top-level procedure.
    Ptr<ProcDecl> parent;

    /// Scope associated with this procedure, if any.
    Scope* scope = nullptr;

    /// Local variables in this procedure.
    ///
    /// The parameter declarations are stored at the start of this array.
    ArrayRef<LocalDecl*> locals;

    /// The template this was instantiated from.
    ProcTemplateDecl* instantiated_from = nullptr;

    /// Whether this procedure (or any of its nested procedures) contain
    /// variable accesses that refer to variables declared in a parent
    /// procedure.
    bool has_captures = false;

    /// Whether any variables declared this procedure specifically are
    /// captured by any nested procedures.
    bool introduces_captures = false;

private:
    ProcDecl(
        TranslationUnit* owner,
        ModuleDecl* imported_from_module,
        ProcType* type,
        DeclName name,
        Linkage linkage,
        Mangling mangling,
        Ptr<ProcDecl> parent,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        ModuleDecl* imported_from_module,
        ProcType* type,
        DeclName name,
        Linkage linkage,
        Mangling mangling,
        Ptr<ProcDecl> parent,
        SLoc location
    ) -> ProcDecl*;

    /// Get the procedure body.
    auto body() -> Ptr<Stmt> { return body_stmt; }

    /// Get all variables declared in this procedure that are captured
    /// by any nested procedures.
    auto captured_vars() {
        return vws::filter(locals, [](auto* decl) { return decl->captured; });
    }

    /// Get the procedure's calling convention.
    auto cconv() const { return proc_type()->cconv(); }

    /// Finalise analysing a procedure.
    ///
    /// \param body The body of the procedure.
    /// \param locals The declarations for all parameters and
    ///        local variables of this procedure.
    void finalise(Ptr<Stmt> body, ArrayRef<LocalDecl*> locals);

    /// Check if this declaration declares an overloaded operator.
    bool is_overloaded_operator() const { return name.is_operator_name(); }

    /// Get the parameter declarations.
    auto params() const -> ArrayRef<ParamDecl*> {
        Assert(not is_imported(), "Attempted to access parameter declarations of imported function");
        auto arr = locals.take_front(param_count());
        return {reinterpret_cast<ParamDecl* const*>(arr.data()), arr.size()};
    }

    /// Get the number of parameters that this procedure has.
    auto param_count() const -> usz { return param_types().size(); }

    /// Get the parameter types.
    auto param_types() const -> ArrayRef<ParamTypeData> {
        return proc_type()->params();
    }

    /// Get the param types without the intent.
    auto param_types_no_intent() const {
        return proc_type()->params() | vws::transform(&ParamTypeData::type);
    }

    /// Iterate over all parents of this procedure.
    auto parents() -> std::generator<ProcDecl*> {
        for (auto p = parent.get_or_null(); p; p = p->parent.get_or_null())
            co_yield p;
    }

    /// Iterate over all parents of this procedure, top-down.
    auto parents_top_down() {
        SmallVector<ProcDecl*> ps;
        for (auto p : parents()) ps.push_back(p);
        rgs::reverse(ps);
        return ps;
    }

    /// Get the procedure type.
    auto proc_type() const -> ProcType*;

    /// Get the procedure's return type.
    auto return_type() -> Type;

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcDecl; }
};

/// Procedure template declaration.
class srcc::ProcTemplateDecl final : public Decl {
public:
    TranslationUnit* owner;

    /// May be null if this is a top-level procedure.
    Ptr<ProcDecl> parent;

    /// Pattern to instantiate.
    ParsedProcDecl* pattern;

    /// Whether this template has any variadic arguments (*not* C
    /// varargs, for that, see the 'c_varargs' attribute).
    bool has_variadic_param;

private:
    ProcTemplateDecl(
        TranslationUnit& tu,
        ParsedProcDecl* pattern,
        Ptr<ProcDecl> parent,
        bool has_variadic_param,
        SLoc location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        ParsedProcDecl* pattern,
        Ptr<ProcDecl> parent,
        bool has_variadic_param
    ) -> ProcTemplateDecl*;

    /// Get all instantiations of this template.
    auto instantiations() -> ArrayRef<ProcDecl*>;

    /// Whether this template is used to implement a builtin operator.
    bool is_builtin_operator_template() const;

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcTemplateDecl; }
};

// This can only be defined here because it needs to know how big 'Decl' is.
inline auto srcc::Scope::decls() {
    return decls_by_name                                                                           //
         | vws::transform([](auto& entry) -> llvm::TinyPtrVector<Decl*>& { return entry.second; }) //
         | vws::join;
}

// This requires the definition of 'FieldDecl', so put it here.
// FIXME: Stop storing decls in the record layout and store only offsets instead.
inline auto srcc::RecordLayout::field_types() const {
    return fields() | vws::transform([](FieldDecl* fd) -> Type { return fd->type; });
}

/// Visit this statement.
template <typename Visitor>
auto srcc::Stmt::visit(Visitor&& v) -> decltype(auto) {
    switch (kind()) {
#define AST_STMT_LEAF(node) \
    case Kind::node: return std::invoke(std::forward<Visitor>(v), static_cast<node*>(this));
#include "srcc/AST.inc"
    }
    Unreachable();
}

template <>
struct std::formatter<srcc::BuiltinCallExpr::Builtin> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::BuiltinCallExpr::Builtin b, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(srcc::BuiltinCallExpr::ToString(b), ctx);
    }
};

#endif // SRCC_AST_STMT_HH
