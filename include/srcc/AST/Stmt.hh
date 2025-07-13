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
class ParsedProcDecl;
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
    Location loc;

protected:
    explicit Stmt(Kind kind, Location loc) : stmt_kind{kind}, loc{loc} {}

public:
    // Only allow allocating these in the module.
    void* operator new(usz) = SRCC_DELETED("Use `new (tu) { ... }` instead");
    void* operator new(usz size, TranslationUnit& tu);

    /// Dump the statement.
    void dump(bool use_colour = false) const;
    void dump_color() const { dump(true); }

    /// Get the kind of this statement.
    Kind kind() const { return stmt_kind; }

    /// Get whether this is an mrvalue.
    bool is_mrvalue() const { return value_category_or_srvalue() == ValueCategory::MRValue; }

    /// Get the source location of this statement.
    auto location() const -> Location { return loc; }

    /// Get the type of this if it is an expression and Void otherwise.
    auto type_or_void() const -> Type;

    /// Get the value category if this is an expression and SRValue otherwise.
    auto value_category_or_srvalue() const -> ValueCategory;

    /// Visit this statement.
    template <typename Visitor>
    auto visit(Visitor&& v) -> decltype(auto);
};

class srcc::EmptyStmt : public Stmt {
public:
    Location loc;
    EmptyStmt(Location loc) : Stmt{Kind::EmptyStmt, loc} {}
    static bool classof(const Stmt* e) { return e->kind() == Kind::EmptyStmt; }
};

class srcc::ForStmt final : public Stmt
    , TrailingObjects<ForStmt, LocalDecl*, Expr*> {
    friend TrailingObjects;
    u32 num_vars, num_ranges;

public:
    Ptr<LocalDecl> enum_var;
    Stmt* body;

private:
    ForStmt(
        Ptr<LocalDecl> enum_var,
        ArrayRef<LocalDecl*> vars,
        ArrayRef<Expr*> ranges,
        Stmt* body,
        Location location
    );

    usz numTrailingObjects(OverloadToken<LocalDecl*>) const { return num_vars; }
    usz numTrailingObjects(OverloadToken<Expr*>) const { return num_ranges; }

public:
    static auto Create(
        TranslationUnit& tu,
        Ptr<LocalDecl> enum_var,
        ArrayRef<LocalDecl*> vars,
        ArrayRef<Expr*> ranges,
        Stmt* body,
        Location location
    ) -> ForStmt*;

    auto ranges() const -> ArrayRef<Expr*> { return getTrailingObjects<Expr*>(num_ranges); }
    auto vars() const -> ArrayRef<LocalDecl*> { return getTrailingObjects<LocalDecl*>(num_vars); }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ForStmt; }
};

class srcc::WhileStmt : public Stmt {
public:
    Expr* cond;
    Stmt* body;

    WhileStmt(
        Expr* cond,
        Stmt* body,
        Location location
    ) : Stmt{Kind::WhileStmt, location}, cond{cond}, body{body} {}

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

    static bool classof(const Stmt* e) {
        return e->kind() >= Kind::ArrayBroadcastExpr and e->kind() <= Kind::UnaryExpr;
    }
};

class srcc::ArrayBroadcastExpr final : public Expr {
public:
    Expr* element;

    ArrayBroadcastExpr(ArrayType* type, Expr* element, Location loc)
        : Expr{Kind::ArrayBroadcastExpr, type, MRValue, loc}, element{element} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::ArrayBroadcastExpr; }
};

class srcc::ArrayInitExpr final : public Expr
    , TrailingObjects<ArrayInitExpr, Expr*> {
    friend TrailingObjects;
    u32 num_inits;

    ArrayInitExpr(ArrayType* type, ArrayRef<Expr*> element, Location loc);

public:
    static auto Create(
        TranslationUnit& tu,
        ArrayType* type,
        ArrayRef<Expr*> element,
        Location loc
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

    AssertExpr(
        Expr* cond,
        Ptr<Expr> message,
        bool is_static,
        Location location
    ) : Expr{Kind::AssertExpr, Type::VoidTy, SRValue, location},
        cond{cond},
        message{message},
        is_static{is_static} {}

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
    ) : Expr{Kind::BoolLitExpr, Type::BoolTy, SRValue, location}, value{value} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::BoolLitExpr; }
};

class srcc::BuiltinCallExpr final : public Expr
    , TrailingObjects<BuiltinCallExpr, Expr*> {
    friend TrailingObjects;

public:
    enum struct Builtin : u8 {
        Print,       // __srcc_print
        Unreachable, // __srcc_unreachable
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
        Location location
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
        /// Convert an srvalue 'T^' to an lvalue 'T'.
        Deref,

        /// Cast an srvalue integer to an srvalue integer.
        Integral,

        /// Convert an lvalue to an srvalue.
        ///
        /// This is only valid for types that can be srvalues.
        LValueToSRValue,

        /// Materialise a poison value of the given type.
        MaterialisePoisonValue,
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
        bool implicit = false,
        ValueCategory value_category = SRValue
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
    ) : Expr{Kind::DefaultInitExpr, type, type->rvalue_category(), location} {}

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
        Location location
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
        Location location
    ) : Expr{Kind::IntLitExpr, ty, SRValue, location}, storage{integer} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::IntLitExpr; }
};

class srcc::LocalRefExpr final : public Expr {
public:
    LocalDecl* decl;
    LocalRefExpr(LocalDecl* decl, ValueCategory vc, Location location);
    static bool classof(const Stmt* e) { return e->kind() == Kind::LocalRefExpr; }
};

class srcc::LoopExpr final : public Expr {
public:
    Ptr<Stmt> body;

    LoopExpr(Ptr<Stmt> body, Location loc)
        : Expr{Kind::LoopExpr, Type::NoReturnTy, SRValue, loc},
          body{body} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::LoopExpr; }
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
    , TrailingObjects<OverloadSetExpr, Decl*> {
    friend TrailingObjects;
    const u32 num_overloads;
    auto numTrailingObjects(OverloadToken<Decl*>) -> usz { return num_overloads; }

    OverloadSetExpr(ArrayRef<Decl*> overloads, Location location);

public:
    static auto Create(
        TranslationUnit& tu,
        ArrayRef<Decl*> overloads,
        Location location
    ) -> OverloadSetExpr*;

    auto overloads() -> ArrayRef<Decl*> { return {getTrailingObjects<Decl*>(), num_overloads}; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::OverloadSetExpr; }
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
    ) : Expr{Kind::TypeExpr, Type::TypeTy, SRValue, location}, value{type} {}
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
            Type::NoReturnTy,
            SRValue,
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
        Location location
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
    String name;
    bool is_valid = true;

protected:
    Decl(
        Kind kind,
        String name,
        Location location
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
        Location location
    ) : Decl{Kind::FieldDecl, name, location}, type{type}, offset{offset} {}

    static bool classof(const Stmt* e) { return e->kind() == Kind::FieldDecl; }
};

class srcc::TemplateTypeParamDecl final : public Decl {
    Type ty;

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
    ) : Decl{Kind::TemplateTypeParamDecl, name, tl.loc}, ty{tl.ty} {}

    /// Get the template argument type bound to this parameter.
    auto arg_type() -> Type { return ty; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::TemplateTypeParamDecl; }
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

    /// The value category of this decl.
    ///
    /// This is usually LValue, but may be SRValue for some in parameters
    /// and loop variables.
    ValueCategory category;

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
        Location location
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
        Location location
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
        String name,
        ProcDecl* parent,
        u32 index,
        bool with_param,
        Location location
    ) : LocalDecl{Kind::ParamDecl, param->type, Expr::LValue, name, parent, location},
        idx{index},
        with{with_param} {
        if (is_rvalue_in_parameter()) category = Expr::SRValue;
    }

    /// Get the parameter’s index.
    [[nodiscard]] auto index() const -> u32 { return idx; }

    /// Get the parameter’s intent.
    [[nodiscard]] auto intent() const -> Intent;

    /// Whether this is a 'with' parameter.
    [[nodiscard]] bool is_with_param() const { return with; }

    static bool classof(const Stmt* e) { return e->kind() == Kind::ParamDecl; }

private:
    /// Whether this is an 'in' parameter that is passed by value
    /// under the hood.
    [[nodiscard]] bool is_rvalue_in_parameter() const;
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

private:
    ProcDecl(
        TranslationUnit* owner,
        ProcType* type,
        String name,
        Linkage linkage,
        Mangling mangling,
        Ptr<ProcDecl> parent,
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
        Location location
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

private:
    ProcTemplateDecl(
        TranslationUnit& tu,
        ParsedProcDecl* pattern,
        Ptr<ProcDecl> parent,
        Location location
    );

public:
    static auto Create(
        TranslationUnit& tu,
        ParsedProcDecl* pattern,
        Ptr<ProcDecl> parent
    ) -> ProcTemplateDecl*;

    /// Get all instantiations of this template.
    auto instantiations() -> ArrayRef<ProcDecl*>;

    static bool classof(const Stmt* e) { return e->kind() == Kind::ProcTemplateDecl; }
};

// This can only be defined here because it needs to know how big 'Decl' is.
inline auto srcc::Scope::decls() {
    return decls_by_name                                                                           //
         | vws::transform([](auto& entry) -> llvm::TinyPtrVector<Decl*>& { return entry.second; }) //
         | vws::join;
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

#endif // SRCC_AST_STMT_HH
