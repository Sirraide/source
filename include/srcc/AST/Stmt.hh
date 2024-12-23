#ifndef SRCC_AST_STMT_HH
#define SRCC_AST_STMT_HH

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
#include <memory>
#include <ranges>

namespace srcc {
#define AST_STMT(node) class node;
#include "srcc/AST.inc"

class ParsedStmt;
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

    /// Whether this statement is dependent, and how.
    Dependence dep = Dependence::None;

    /// Source location of this statement.
    Location loc;

protected:
    explicit Stmt(Kind kind, Location loc) : stmt_kind{kind}, loc{loc} {}

public:
    // Only allow allocating these in the module.
    void* operator new(usz) = SRCC_DELETED("Use `new (tu) { ... }` instead");
    void* operator new(usz size, TranslationUnit& tu);

    /// Get whether this statement is dependent.
    bool dependent() const { return dependence() != Dependence::None; }

    /// Get the dependence of this statement.
    auto dependence() const -> Dependence { return dep; }

    /// Check if this statement contains an error.
    bool errored() const { return dependence() & Dependence::ErrorDependent; }

    /// Dump the statement.
    void dump(bool use_colour = false) const;
    void dump_color() const { dump(true); }

    /// Get the kind of this statement.
    Kind kind() const { return stmt_kind; }

    /// Get the source location of this statement.
    auto location() const -> Location { return loc; }

    /// Set the dependence of this statement.
    void set_dependence(Dependence d) { dep = d; }

    /// Mark this node as errored.
    void set_errored() { dep |= Dependence::Error; }

    /// Check if this is type-dependent.
    bool type_dependent() const { return dependence() & Dependence::TypeDependent; }

    /// Visit this statement.
    template <typename Visitor>
    auto visit(Visitor&& v) -> decltype(auto);

protected:
    /// Compute the dependence of this expression.
    void ComputeDependence();
};

class srcc::WhileStmt : public Stmt {
public:
    Expr* cond;
    Stmt* body;

    WhileStmt(
        Expr* cond,
        Stmt* body,
        Location location
    ) : Stmt{Kind::WhileStmt, location}, cond{cond}, body{body} {
        ComputeDependence();
    }

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
        Location location
    ) : Stmt(kind, location), type{type}, value_category{category} {}

public:
    using enum ValueCategory;

    /// Check if this expression is an lvalue.
    [[nodiscard]] bool lvalue() const { return value_category == LValue; }

    /// Check if this expression is an rvalue.
    [[nodiscard]] bool rvalue() const { return not lvalue(); }

    /// Look through parentheses.
    [[nodiscard]] auto strip_parens() -> Expr*;

    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::AssertExpr and e->kind() <= Kind::UnaryExpr;
    }
};

class srcc::AssertExpr final : public Expr {
public:
    Expr* cond;
    Ptr<Expr> message;
    bool is_static;

    AssertExpr(
        Expr* cond,
        Ptr<Expr> message,
        bool is_static,
        Location location
    ) : Expr{Kind::AssertExpr, Types::VoidTy, SRValue, location},
        cond{cond},
        message{message},
        is_static{is_static} {
        ComputeDependence();
    }

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
        Location location
    ) : Expr{Kind::BinaryExpr, type, cat, location},
        op{op},
        lhs{lhs},
        rhs{rhs} {
        ComputeDependence();
    }

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
        Location location
    );

public:
    static auto Create(
        TranslationUnit& mod,
        Scope* parent_scope,
        ArrayRef<Stmt*> stmts,
        Location location
    ) -> BlockExpr*;

    /// Get the statements in this block.
    auto stmts() -> ArrayRef<Stmt*> { return {getTrailingObjects<Stmt*>(), num_stmts}; }

    /// Get the expression whose value is returned from this block, if any.
    auto return_expr() -> Expr*;

    static bool classof(const Stmt* e) { return e->kind() == Kind::BlockExpr; }
};

class srcc::BoolLitExpr final : public Expr {
public:
    const bool value;

    BoolLitExpr(
        bool value,
        Location location
    ) : Expr{Kind::BoolLitExpr, Types::BoolTy, SRValue, location}, value{value} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BoolLitExpr; }
};

class srcc::BuiltinCallExpr final : public Expr
    , TrailingObjects<BuiltinCallExpr, Expr*> {
    friend TrailingObjects;

public:
    enum struct Builtin : u8 {
        Print // __builtin_print
    };

    const Builtin builtin;

private:
    const u32 num_args;
    auto numTrailingObjects(OverloadToken<Expr*>) -> usz { return num_args; }

    BuiltinCallExpr(
        Builtin kind,
        Type return_type,
        ArrayRef<Expr*> args,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        Builtin kind,
        Type return_type,
        ArrayRef<Expr*> args,
        Location location
    ) -> BuiltinCallExpr*;

    /// Get the arguments.
    auto args() -> ArrayRef<Expr*> { return {getTrailingObjects<Expr*>(), num_args}; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::BuiltinCallExpr; }
};

class srcc::BuiltinMemberAccessExpr final : public Expr {
public:
    enum struct AccessKind : u8 {
        SliceData,
        SliceSize,
        TypeAlign,
        TypeArraySize,
        TypeBits,
        TypeBytes,
        TypeName,
    };

    Expr* operand;
    const AccessKind access_kind;

    BuiltinMemberAccessExpr(
        Type type,
        ValueCategory cat,
        Expr* operand,
        AccessKind kind,
        Location location
    ) : Expr{Kind::BuiltinMemberAccessExpr, type, cat, location},
        operand{operand},
        access_kind{kind} {
        ComputeDependence();
    }

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
        Expr* callee,
        ArrayRef<Expr*> args,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& mod,
        Type type,
        Expr* callee,
        ArrayRef<Expr*> args,
        Location location
    ) -> CallExpr*;

    [[nodiscard]] auto args() -> ArrayRef<Expr*> { return {getTrailingObjects<Expr*>(), num_args}; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::CallExpr; }
};

class srcc::CastExpr final : public Expr {
public:
    enum class CastKind : u8 {
        /// Convert an lvalue to an srvalue.
        ///
        /// This is only valid for types that can be srvalues.
        LValueToSRValue,

        /// Cast an srvalue integer to an srvalue integer.
        Integral,
    };

    using enum CastKind;

    Expr* arg;
    CastKind kind;
    bool implicit;

    CastExpr(
        Type type,
        CastKind kind,
        Expr* expr,
        Location location,
        bool implicit = false
    ) : Expr{Kind::CastExpr, type, SRValue, location},
        arg{expr},
        kind{kind},
        implicit{implicit} {
        ComputeDependence();
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::CastExpr; }
};

class srcc::ConstExpr final : public Expr {
public:
    /// Constant value.
    eval::Value* value;

    /// Evaluated statement. May be unset if this was created artificially.
    Ptr<Stmt> stmt;

    ConstExpr(
        TranslationUnit& tu,
        eval::Value value,
        Location location,
        Ptr<Stmt> stmt = {}
    );

    static bool classof(const Stmt* e) { return e->kind() == Kind::ConstExpr; }
};

/// Default-initialise a type.
class srcc::DefaultInitExpr final : public Expr {
public:
    DefaultInitExpr(
        Type type,
        Location location
    ) : Expr{Kind::DefaultInitExpr, type, type->value_category(), location} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::DefaultInitExpr; }
};

/// This is wrapped with a 'ConstExpr' after evaluation, so the
/// type of this itself is irrelevant.
class srcc::EvalExpr final : public Expr {
public:
    Stmt* stmt;

    EvalExpr(
        Stmt* stmt,
        Location location
    ) : Expr{Kind::EvalExpr, Types::DependentTy, DValue, location}, stmt{stmt} {
        ComputeDependence();
    }

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
        Location location
    ) : Expr{Kind::IfExpr, type, val, location},
        cond{cond},
        then{then},
        else_{else_},
        is_static{is_static} {
        ComputeDependence();
    }

    [[nodiscard]] bool has_yield() const {
        return type != Types::VoidTy and type != Types::NoReturnTy;
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::IfExpr; }
};

class srcc::IntLitExpr final : public Expr {
public:
    StoredInteger storage;

    IntLitExpr(
        Type ty,
        StoredInteger integer,
        Location location
    ) : Expr{Kind::IntLitExpr, ty, SRValue, location}, storage{integer} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::IntLitExpr; }
};

class srcc::LocalRefExpr final : public Expr {
public:
    LocalDecl* decl;
    LocalRefExpr(LocalDecl* decl, Location location);
    static bool classof(const Stmt* e) { return e->kind() == Kind::LocalRefExpr; }
};

class srcc::MemberAccessExpr final : public Expr {
public:
    Expr* base;
    FieldDecl* field;

    MemberAccessExpr(
        Expr* base,
        FieldDecl* field,
        Location location
    );

    static bool classof(const Stmt* e) { return e->kind() == Kind::MemberAccessExpr; }
};
class srcc::OverloadSetExpr final : public Expr
    , TrailingObjects<OverloadSetExpr, ProcDecl*> {
    friend TrailingObjects;
    const u32 num_overloads;
    auto numTrailingObjects(OverloadToken<ProcDecl*>) -> usz { return num_overloads; }

    OverloadSetExpr(ArrayRef<Decl*> overloads, Location location);

public:
    static auto Create(
        TranslationUnit& tu,
        ArrayRef<Decl*> overloads,
        Location location
    ) -> OverloadSetExpr*;

    auto overloads() -> ArrayRef<ProcDecl*> { return {getTrailingObjects<ProcDecl*>(), num_overloads}; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::OverloadSetExpr; }
};

class srcc::ParenExpr final : public Expr {
public:
    Expr* expr;

    ParenExpr(
        Expr* expr,
        Location location
    ) : Expr{Kind::ParenExpr, expr->type, expr->value_category, location}, expr{expr} {
        ComputeDependence();
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ParenExpr; }
};

class srcc::ProcRefExpr final : public Expr {
public:
    ProcDecl* decl;

    ProcRefExpr(
        ProcDecl* decl,
        Location location
    );

    auto return_type() const -> Type;

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcRefExpr; }
};

class srcc::StaticIfExpr final : public Expr {
public:
    Expr* cond;

    // Making the body of a 'static if' ‘dependent’ or deferring instantiation
    // in some other way is *really* complicated. Instead, we take a page out
    // of D’s book and simply... don’t translate the body until we know which
    // branch we actually want.
    ParsedStmt* then;
    Ptr<ParsedStmt> else_;

    StaticIfExpr(
        Expr* cond,
        ParsedStmt* then,
        Ptr<ParsedStmt> else_,
        Location location
    ) : Expr{Kind::StaticIfExpr, Types::DependentTy, DValue, location},
        cond{cond},
        then{then},
        else_{else_} {
        Assert(cond->dependent(), "Non-dependent StaticIfExpr???");
        ComputeDependence();
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::StaticIfExpr; }
};

class srcc::StrLitExpr final : public Expr {
public:
    String value;

private:
    StrLitExpr(
        Type ty,
        String value,
        Location location
    ) : Expr{Kind::StrLitExpr, ty, SRValue, location}, value{value} {}

public:
    static auto Create(TranslationUnit& mod, String value, Location location) -> StrLitExpr*;

    static bool classof(const Stmt* e) { return e->kind() == Kind::StrLitExpr; }
};

/// A literal initialiser for a struct.
class srcc::StructInitExpr final : public Expr
    , TrailingObjects<StructInitExpr, Expr*> {
    friend TrailingObjects;
    auto numTrailingObjects(OverloadToken<Expr*>) -> usz {
        return struct_type()->fields().size();
    }

    StructInitExpr(
        StructType* ty,
        ArrayRef<Expr*> fields,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        StructType* type,
        ArrayRef<Expr*> fields,
        Location location
    ) -> StructInitExpr*;

    auto struct_type() const -> StructType* { return cast<StructType>(type.ptr()); }
    auto values() -> ArrayRef<Expr*> { return {getTrailingObjects<Expr*>(), struct_type()->fields().size()}; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::StructInitExpr; }
};

class srcc::TypeExpr final : public Expr {
public:
    Type value;
    TypeExpr(
        Type type,
        Location location
    ) : Expr{Kind::TypeExpr, Types::TypeTy, SRValue, location}, value{type} { ComputeDependence(); }
    static bool classof(const Stmt* e) { return e->kind() == Kind::TypeExpr; }
};

class srcc::ReturnExpr final : public Expr {
public:
    Ptr<Expr> value;
    bool implicit;

    ReturnExpr(
        Ptr<Expr> value,
        Location location,
        bool implicit = false
    ) : Expr{
            Kind::ReturnExpr,
            Types::NoReturnTy,
            SRValue,
            location,
        },
        value{value}, implicit{implicit} {
        ComputeDependence();
    }

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
        Location location
    ) : Expr{Kind::UnaryExpr, type, val, location},
        op{op},
        arg{arg},
        postfix{postfix} {
        ComputeDependence();
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::UnaryExpr; }
};

// ============================================================================
//  Declarations
// ============================================================================
class srcc::Decl : public Stmt {
public:
    String name;

protected:
    Decl(
        Kind kind,
        String name,
        Location location
    ) : Stmt{kind, location}, name{name} {}

public:
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
        Location location
    ) : Decl{Kind::FieldDecl, name, location}, type{type}, offset{offset} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::FieldDecl; }
};

class srcc::TemplateTypeDecl final : public Decl
    , llvm::TrailingObjects<TemplateTypeDecl, u32> {
    friend TrailingObjects;
    u32 num_deduced_indices;
    auto numTrailingObjects(OverloadToken<u32>) -> usz { return num_deduced_indices; }

    TemplateTypeDecl(
        String name,
        ArrayRef<u32> deduced_indices,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        String name,
        ArrayRef<u32> deduced_indices,
        Location location
    ) -> TemplateTypeDecl*;

    /// Get the indices of any parameters of the procedure that this
    /// declaration belongs to from which the type of this declaration
    /// shall be deduced.
    auto deduced_indices() -> ArrayRef<u32> {
        return {getTrailingObjects<u32>(), num_deduced_indices};
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::TemplateTypeDecl; }
};

// A struct or type alias decl.
class srcc::TypeDecl : public Decl {
    friend StructType;

public:
    Type type;

private:
    TypeDecl(
        Type type,
        String name,
        Location location
    ) : Decl{Kind::TypeDecl, name, location}, type{type} {}

public:
    static bool classof(const Stmt* e) { return e->kind() == Kind::TypeDecl; }
};

class srcc::LocalDecl : public Decl {
public:
    /// The immediate parent whose stack frame this belongs to.
    ProcDecl* parent;

    /// The type of this decl.
    Type type;

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
        String name,
        ProcDecl* parent,
        Ptr<Expr> init,
        Location location
    ) : Decl{k, name, location},
        parent{parent},
        type{type},
        init{init} {
        ComputeDependence();
    }

public:
    LocalDecl(
        Type type,
        String name,
        ProcDecl* parent,
        Ptr<Expr> init,
        Location location
    ) : LocalDecl{Kind::LocalDecl, type, name, parent, init, location} {}

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
        String name,
        ProcDecl* parent,
        u32 index,
        bool with_param,
        Location location
    ) : LocalDecl{Kind::ParamDecl, param->type, name, parent, nullptr, location},
        idx{index},
        with{with_param} {}

    /// Get the parameter’s index.
    [[nodiscard]] auto index() const -> u32 { return idx; }

    /// Get the parameter’s intent.
    [[nodiscard]] auto intent() const -> Intent;

    /// Whether this is an 'in' parameter that is passed by value
    /// under the hood.
    [[nodiscard]] bool is_rvalue_in_parameter() const;

    /// Whether this is a 'with' parameter.
    [[nodiscard]] bool is_with_param() const { return with; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ParamDecl; }
};
/// Declaration with linkage.
class srcc::ObjectDecl : public Decl {
public:
    TranslationUnit* owner;
    Linkage linkage;
    Mangling mangling;
    Type type;

protected:
    ObjectDecl(
        Kind kind,
        TranslationUnit* owner,
        Type type,
        String name,
        Linkage linkage,
        Mangling mangling,
        Location location
    ) : Decl{kind, name, location},
        owner{owner},
        linkage{linkage},
        mangling{mangling},
        type{type} {}

public:
    static bool classof(const Stmt* e) { return e->kind() >= Kind::ProcDecl; }
};

/// Procedure declaration.
class srcc::ProcDecl final : public ObjectDecl
    , llvm::TrailingObjects<ProcDecl, TemplateTypeDecl*> {
    friend TrailingObjects;
    const u32 num_template_params;
    auto numTrailingObjects(OverloadToken<TemplateTypeDecl*>) -> usz { return num_template_params; }

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
    ProcDecl* instantiated_from = nullptr;

private:
    ProcDecl(
        TranslationUnit* owner,
        ProcType* type,
        String name,
        Linkage linkage,
        Mangling mangling,
        Ptr<ProcDecl> parent,
        ArrayRef<TemplateTypeDecl*> template_params,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        ProcType* type,
        String name,
        Linkage linkage,
        Mangling mangling,
        Ptr<ProcDecl> parent,
        Location location,
        ArrayRef<TemplateTypeDecl*> template_params = {}
    ) -> ProcDecl*;

    /// Get the procedure body.
    auto body() -> Ptr<Stmt> { return body_stmt; }

    /// Get the procedure's calling convention.
    auto cconv() const { return proc_type()->cconv(); }

    /// Finalise analysing a procedure.
    ///
    /// \param body The body of the procedure.
    /// \param locals The declarations for all parameters and
    ///        local variables of this procedure.
    void finalise(Ptr<Stmt> body, ArrayRef<LocalDecl*> locals);

    /// Whether this is a template.
    bool is_template() const { return num_template_params > 0; }

    /// Get the parameter declarations.
    auto params() const -> ArrayRef<ParamDecl*> {
        auto arr = locals.take_front(param_count());
        return {reinterpret_cast<ParamDecl* const*>(arr.data()), arr.size()};
    }

    /// Get the number of parameters that this procedure has.
    auto param_count() const -> usz { return param_types().size(); }

    /// Get the parameter types.
    auto param_types() const -> ArrayRef<ParamTypeData> {
        return proc_type()->params();
    }

    /// Get the procedure type.
    auto proc_type() const -> ProcType*;

    /// Get the procedure's return type.
    auto return_type() -> Type;

    /// Get the template parameters.
    auto template_params() -> ArrayRef<TemplateTypeDecl*> {
        return {getTrailingObjects<TemplateTypeDecl*>(), num_template_params};
    }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcDecl; }
};

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

#endif // SRCC_AST_STMT_HH
