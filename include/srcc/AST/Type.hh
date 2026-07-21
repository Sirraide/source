#ifndef SRCC_AST_TYPE_HH
#define SRCC_AST_TYPE_HH

#include <srcc/AST/DeclName.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

#include <base/Macros.hh>
#include <base/Serialisation.hh>

namespace srcc {
class Target;
class Scope;
class BlockScope;
class ModuleScope;
class ProcScope;
class StructScope;
class TemplateTypeParamDecl;
class TranslationUnit;
struct ParamTypeData;
class Expr;
class Decl;
class FieldDecl;
class ProcTemplateDecl;
class TypeDecl;
class ProcDecl;
class EnumeratorDecl;
class RecordLayout;
struct BuiltinTypes;

#define AST_TYPE_LEAF_WITH_IDENTITY(node) class node;
#define AST_TYPE(node) class SRCC_DIAGNOSE_POINTER_COMPARISON node;
#include "srcc/AST.inc"

class Type;
class TypeAndValueCategory;
class TypeLoc;

namespace types::detail {
struct CanonicalTypeDenseMapInfo;
struct NonCanonicalTypeDenseMapInfo;
auto MakeCanonical(TranslationUnit& tu, RecordLayout* rl) -> RecordLayout*;
}

/// Casting.
template <typename To> auto cast(Type from) -> To*;
template <typename To> auto cast_if_present(Opt<Type> from) -> To*;
template <typename To> auto dyn_cast(Type from) -> To*;
template <typename... Ts> auto isa(Type from) -> bool;
template <typename... Ts> auto isa_and_nonnull(Opt<Type> from) -> bool;
} // namespace srcc

/// Base class for all types.
///
/// The 'alignas(8)' is required to ensure that we have some low bits
/// available in the pointer for other purposes.
///
/// \see Type.
class alignas(8) srcc::TypeBase {
    LIBBASE_IMMOVABLE(TypeBase);

    #define AST_TYPE(node) friend node;
    #include "srcc/AST.inc"

    friend Type;
    friend types::detail::CanonicalTypeDenseMapInfo;
    friend auto types::detail::MakeCanonical(TranslationUnit& tu, RecordLayout* rl) -> RecordLayout*;
    template <typename To> friend auto ::srcc::cast(Type from) -> To*;
    template <typename To> friend auto ::srcc::dyn_cast(Type from) -> To*;
    template <typename... Ts> friend auto ::srcc::isa(Type from) -> bool;

public:
    enum struct Kind : u8 {
#define AST_TYPE_LEAF(node) node,
#include "srcc/AST.inc"

    };

private:
    /// Canonical type.
    TypeBase* canonical;

    /// Kind of this type.
    const Kind type_kind;

protected:
    explicit constexpr TypeBase(Kind kind) : canonical{this}, type_kind{kind} {}

public:
    SRCC_ALLOCATE_IN_CONTEXT(TypeBase, TranslationUnit);

    /// Get the alignment of this type.
    [[nodiscard]] auto align(TranslationUnit& tu) const -> Align;
    [[nodiscard]] auto align(const Target& t) const -> Align;

    /// Get the size of this type when stored in an array.
    ///
    /// \see bit_width(), memory_size()
    [[nodiscard]] auto array_size(TranslationUnit& tu) const -> Size;
    [[nodiscard]] auto array_size(const Target& t) const -> Size;

    /// Get the exact bit width of this type.
    ///
    /// \see array_size(), memory_size()
    [[nodiscard]] auto bit_width(TranslationUnit& tu) const -> Size;

    /// Get whether this type can be initialised using an empty
    /// argument list. For struct types, this can entail calling
    /// an initialiser with no arguments.
    [[nodiscard]] bool can_init_from_no_args() const;

    /// Get whether this type has a default initialiser.
    [[nodiscard]] bool can_default_init() const;

    /// Check if default initialisation for this type is zero-initialisation.
    [[nodiscard]] bool can_zero_init() const;

    /// Print a type to stdout.
    void dump(bool use_colour = false) const;
    void dump_colour() const { dump(true); }

    /// Get the evaluation mode of this type; this determines whether values
    /// of this type always need to live in memory or not.
    [[nodiscard]] auto eval_mode() const -> EvalMode;

    /// Check if this is an array/struct/range/slice/closure.
    [[nodiscard]] bool is_aggregate() const;

    /// Check if this is a complete type.
    [[nodiscard]] bool is_complete() const;

    /// Check if this is 'int' or a sized integer.
    [[nodiscard]] bool is_integer() const;

    /// Check if this is 'int', 'bool', or a sized integer.
    [[nodiscard]] bool is_integer_or_bool() const;

    /// Check whether the memory representation for this type is or
    /// contains a pointer at any level (this is especially also true
    /// for e.g. slices and closures).
    [[nodiscard]] bool is_or_contains_pointer() const;

    /// Check if this type is the builtin 'void' type.
    [[nodiscard]] bool is_void() const;

    /// Get the in-memory size of this type, excluding tail padding.
    ///
    /// \see array_size(), bit_width()
    [[nodiscard]] auto memory_size(TranslationUnit& tu) const -> Size;
    [[nodiscard]] auto memory_size(const Target& t) const -> Size;

    /// Whether moving this type is the same as a copy.
    [[nodiscard]] bool move_is_copy() const;

    /// Determine whether this parameter type is passed by reference under
    /// the given intent.
    ///
    /// No calling convention is passed to this since parameters to native
    /// procedures should always have the 'copy' intent, which by definition
    /// always passes by value.
    [[nodiscard]] bool pass_by_reference(const Target& t, Intent i) const;

    /// Get a string representation of this type.
    [[nodiscard]] auto print() const -> SmallUnrenderedString;

    /// Stream the fields that make up this aggregate together with
    /// their offsets.
    ///
    /// Do NOT use this for large arrays!!!
    /*[[nodiscard]] auto stream_fields(TranslationUnit& tu) {
        return stream_fields_impl(tu, {});
    }*/

    /// Whether this type has a destructor.
    [[nodiscard]] bool requires_deletion() const;

    /// Strip array types from this type.
    [[nodiscard]] auto strip_arrays() const -> Type;

    /// Strip pointers and optional types.
    [[nodiscard]] auto strip_pointers_and_optionals() const -> Type;

    /// Whether this type is trivially copyable.
    [[nodiscard]] bool trivially_copyable() const {
        return not requires_deletion();
    }

    /// Visit this type.
    template <typename Visitor>
    auto visit(Visitor&& v) const -> decltype(auto);

    /// Get the canonical type kind of this type.
    [[nodiscard]] auto canonical_kind() const -> Kind { return canonical->type_kind; }

    /// Get the exact type kind of this type, including type sugar if this is a sugared
    /// type. You almost always want 'canonical_kind()' instead.
    [[nodiscard]] auto exact_kind() const -> Kind { return type_kind; }

private:
    [[nodiscard]] auto desugar_once() const -> Type;
    [[nodiscard]] bool is_canonical() const;
    [[nodiscard]] bool is_sugar() const;
    [[nodiscard]] auto stream_fields_impl(TranslationUnit& tu, Size offs) -> std::generator<std::pair<Type, Size>>;
    [[nodiscard]] auto size_impl(const Target& t) const -> Size;

    template <typename ...Types>
    [[nodiscard]] auto strip_qualifiers() const -> Type;

    /// As visit() but also visits non-canonical types.
    template <typename Visitor>
    auto visit_exact(Visitor&& v) const -> decltype(auto);
};

/// Class that holds a type.
///
/// The main purpose of this class is to allow writing equality comparisons
/// between types as 'a == b'. We can’t simply do that w/ two 'TypeBase*'s
/// since the builtin pointer comparison operator doesn't work properly as
/// soon as type sugar is involved.
///
/// 'CanonicalType' is a variant of this that always stores a canonical type
/// pointer; this avoids having to apply desugaring in cases where we only
/// care about canonical types.
///
/// Generally, use 'CanonicalType' whenever you explicitly don't want to
/// preserve type sugar, and 'Type' everywhere else. Note that equality
/// comparison and hashing *always* look through type sugar. In situations
/// where you *do* want to distinguish type sugar, use a 'TypeBase*' (e.g.
/// serialisation).
class srcc::Type {
    TypeBase* pointer;

public:
    Type() = delete("Use 'Opt<Type>' instead of trying to construct a null 'Type'");
    Type(std::nullptr_t) = delete("Use 'Opt<Type>' instead of trying to construct a null 'Type'");
    constexpr Type(TypeBase* t) : pointer{t} { DebugAssert(t); }
    constexpr Type(const TypeBase* t) : Type{const_cast<TypeBase*>(t)} {}

    /// Get the type pointer.
    [[nodiscard]] auto ptr() const -> TypeBase* { return pointer; }

    /// Access the type pointer.
    [[nodiscard]] auto operator->() const -> TypeBase* { return pointer; }

    /// Check if two types are equal.
    [[nodiscard]] auto operator==(Type ty) const -> bool {
        // static_cast to void is required because we disallow pointer
        // comparisons involving 'TypeBase*'.
        return static_cast<void*>(pointer->canonical) ==
               static_cast<void*>(ty.pointer->canonical);
    }

    [[nodiscard]] auto operator==(const TypeBase* ty) const -> bool {
        return *this == Type{ty};
    }

    static const Type VoidTy;
    static const Type NoReturnTy;
    static const Type BoolTy;
    static const Type IntTy;
    static const Type DeducedTy;
    static const Type TreeTy;
    static const Type TypeTy;
    static const Type UnresolvedOverloadSetTy;
    static const Type NilTy;
};

/// Specialisation of 'Opt' for 'Type'.
template <>
class srcc::Opt<srcc::Type> {
    llvm::PointerIntPair<TypeBase*, 1, bool> type_and_is_engaged;

public:
    Opt() : type_and_is_engaged(nullptr, false) {}
    Opt(Type ty) : type_and_is_engaged(ty.ptr(), true) {}
    Opt(std::derived_from<TypeBase> auto* ty) : type_and_is_engaged(ty, true) {}
    Opt(std::nullopt_t) : Opt() {}

    [[nodiscard]] bool has_value() const { return type_and_is_engaged.getInt(); }
    [[nodiscard]] explicit operator bool() const { return has_value(); }
    [[nodiscard]] auto value() const -> Type {
        Assert(has_value());
        return Type{type_and_is_engaged.getPointer()};
    }

    [[nodiscard]] auto value_or(Type ty) const -> Type {
        if (has_value()) return value();
        return ty;
    }

    [[nodiscard]] auto operator->() const -> TypeBase* { return value().ptr(); }
    [[nodiscard]] auto operator*() const -> Type { return value(); }
    [[nodiscard]] bool operator==(const Opt<Type>& other) const = default;
};

/// Query whether a type class is type sugar.
template <typename Ty>
consteval bool IsTypeSugarClass() {
#   define AST_TYPE_SUGAR(Class) if constexpr (std::same_as<Ty, Class>) return true;
#   include "srcc/AST.inc"
    return false;
}

/// Scope that stores declarations.
///
/// These need to be allocated separately because we usually need
/// to be able to look up declarations in one before we create the
/// node that contains it.
///
/// Note that these are only valid during sema and should not be
/// referenced after that.
class alignas(8) srcc::Scope {
    LIBBASE_IMMOVABLE(Scope);

    /// The parent scope and scope kind.
    llvm::PointerIntPair<Scope*, 3, ScopeKind> parent_scope_and_kind;

protected:
    explicit Scope(Scope* parent, ScopeKind k);

public:
    /// Declarations in this scope.
    DeclNameMap<llvm::TinyPtrVector<Decl*>> decls_by_name;

    /// Defaulted out-of-line.
    virtual ~Scope();

    /// Get the associated type of this scope.
    auto associated_type() -> Opt<Type>;

    /// Get a flat list of all declarations in this scope.
    auto decls();

    /// Get this scope’s enclosing procedure scope (which may be itself).
    auto enclosing_proc() -> ProcScope*;

    /// Get the kind of this scope.
    auto kind() const -> ScopeKind { return parent_scope_and_kind.getInt(); }

    /// Check if this is a specific kind of scope.
    bool is_block_scope() const { return kind() == ScopeKind::Block; }
    bool is_module_scope() const { return kind() == ScopeKind::Module; }
    bool is_proc_scope() const { return kind() == ScopeKind::Procedure; }
    bool is_struct_scope() const { return kind() == ScopeKind::Struct; }

    /// Get the parent scope.
    ///
    /// This returns null if this is the global scope.
    auto parent() -> Scope* { return parent_scope_and_kind.getPointer(); }

    /// Get a list of declarations, sorted in source order.
    [[nodiscard]] auto sorted_decls() -> SmallVector<Decl*>;
};

/// Scope of a block.
class srcc::BlockScope : public Scope {
    friend TranslationUnit;
    BlockScope(Scope* parent) : Scope{parent, ScopeKind::Block} {}
    static bool classof(const Scope* s) { return s->is_block_scope(); }
};

/// Scope used for the exports of a module; NOTE: the global scope is *not* a module scope!
class srcc::ModuleScope : public Scope {
    friend TranslationUnit;
    ModuleScope() : Scope{nullptr, ScopeKind::Module} {}
    static bool classof(const Scope* s) { return s->is_module_scope(); }
};

/// Scope of a procedure.
class srcc::ProcScope : public Scope {
    friend TranslationUnit;

    ProcScope(Scope* parent, Opt<Type> associated_type)
        : Scope{parent, ScopeKind::Procedure},
          associated_type{associated_type} {}

public:
    /// The associated type of the procedure. We also store this here because
    /// we need to know this before the actual declaration is created.
    Opt<Type> associated_type;

    /// Initialiser declarations.
    SmallVector<Decl*> inits;

    /// Check if 's' is a proc scope.
    static bool classof(const Scope* s) { return s->is_proc_scope(); }
};

/// Scope that stores struct members, initialisers, etc.
class srcc::StructScope : public Scope {
    friend TranslationUnit;

    StructScope(Scope* parent) : Scope{parent, ScopeKind::Struct} {}

public:
    /// Initialiser declarations.
    SmallVector<Decl*> inits;

    static bool classof(const Scope* s) { return s->is_struct_scope(); }
};

class srcc::BuiltinType final : public TypeBase {
    friend BuiltinTypes;

    const BuiltinKind b_kind;

    explicit constexpr BuiltinType(BuiltinKind kind)
        : TypeBase{Kind::BuiltinType}, b_kind{kind} {}

public:
    /// Get the kind of this builtin type.
    auto builtin_kind() const -> BuiltinKind { return b_kind; }

    static auto Get(TranslationUnit& mod, BuiltinKind kind) = delete("Use Types::VoidTy and friends instead");
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::BuiltinType; }
};

class srcc::IntType final : public TypeBase
    , public FoldingSetNode {
    friend Type;

public:
    static constexpr Size MaxBits = Size::Bits(u64(llvm::IntegerType::MAX_INT_BITS));

private:
    Size bits;

    explicit constexpr IntType(Size bit_width) : TypeBase{Kind::IntType}, bits{bit_width} {
        Assert(bits >= Size::Bits(1), "Cannot create integer type with bit width 0");
        Assert(bits <= MaxBits, "Bit width too large: {:i}", bits);
    }

public:
    /// Get the bit width of this integer type.
    auto bit_width() const -> Size { return bits; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, bits); }
    static auto Get(TranslationUnit& mod, Size size) -> IntType*;
    static void Profile(FoldingSetNodeID& ID, Size bit_width);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::IntType; }
};

namespace srcc {
inline auto hash_value(Type ty) = delete("Cannot hash 'Type'");

/// For libassert.
inline auto operator<<(std::ostream& os, Type ty) -> std::ostream& {
    return os << text::RenderColours(false, ty->print().str());
}
}

class srcc::TypeAndValueCategory {
    llvm::PointerIntPair<TypeBase*, 2, ValueCategory> data;

public:
    /// Create a new type and value category pair.
    TypeAndValueCategory(Type ty, ValueCategory val)
        : data{ty.ptr(), val} {}

    /// Create a void SRValue pair.
    TypeAndValueCategory() : TypeAndValueCategory(Type::VoidTy, ValueCategory::RValue) {}

    /// Get the type.
    [[nodiscard]] auto type() const -> Type { return data.getPointer(); }
    void set_type(Type ty) { data.setPointer(ty.ptr()); }

    /// Get the value category.
    [[nodiscard]] auto value_category() const -> ValueCategory {
        return data.getInt();
    }
    void set_value_category(ValueCategory vc) { data.setInt(vc); }
};

template <>
struct std::tuple_size<srcc::TypeAndValueCategory>
    : std::integral_constant<base::usz, 2> {};

template <>
struct std::tuple_element<0, srcc::TypeAndValueCategory>
    : std::type_identity<srcc::Type> {};

template <>
struct std::tuple_element<1, srcc::TypeAndValueCategory>
    : std::type_identity<srcc::ValueCategory> {};

namespace srcc {
template <std::size_t i>
constexpr auto get(TypeAndValueCategory tvc) {
    if constexpr (i == 0) return tvc.type();
    else if constexpr (i == 1) return tvc.value_category();
    else static_assert(false, "Invalid index");
}
}

template <>
struct llvm::simplify_type<srcc::TypeAndValueCategory> {
    using SimpleType = srcc::TypeBase*;
    static SimpleType getSimplifiedValue(srcc::TypeAndValueCategory v) { return v.type().ptr(); }
};

template <>
struct llvm::simplify_type<const srcc::TypeAndValueCategory> {
    using SimpleType = srcc::TypeBase*;
    static SimpleType getSimplifiedValue(srcc::TypeAndValueCategory v) { return v.type().ptr(); }
};

class srcc::TypeLoc {
public:
    Type ty;
    SLoc loc;
    TypeLoc(Type ty, SLoc loc) : ty{ty}, loc{loc} {}
    TypeLoc(Expr* e);
};

class srcc::SingleElementTypeBase : public TypeBase {
    Type element_type;

protected:
    SingleElementTypeBase(Kind kind, Type elem) : TypeBase{kind}, element_type{elem} {}

public:
    /// Get the element type of this type, e.g. `int` for `int[3]`.
    auto elem() const -> Type { return element_type; }

    static bool classof(const TypeBase* e) {
        return e->exact_kind() >= Kind::ArrayType and e->exact_kind() <= Kind::RangeType;
    }
};

class srcc::ArrayType final : public SingleElementTypeBase
    , public FoldingSetNode {
    i64 elems;

    ArrayType(
        Type elem,
        i64 size
    ) : SingleElementTypeBase{Kind::ArrayType, elem}, elems{size} {
        Assert(size >= 0, "Negative array size?");
    }

public:
    /// Get the number of elements in this array.
    auto dimension() const -> i64 { return elems; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem(), elems); }
    static auto Get(TranslationUnit& mod, Type elem, i64 size) -> ArrayType*;
    static void Profile(FoldingSetNodeID& ID, Type elem, i64 size);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::ArrayType; }
};

class srcc::EnumType final : public SingleElementTypeBase {
    TypeDecl* type_decl{};
    Scope* enum_scope{};
    bool complete = false;

public:
    EnumType(TranslationUnit& tu, Scope* scope, DeclName name, Type underlying_type, SLoc loc);

    /// Get the declaration of this enum.
    auto decl() const -> TypeDecl* { return type_decl; }

    /// Get all enumerators.
    auto enumerators() const;

    /// Mark an enum type as complete.
    void finalise() { complete = true; }

    /// Whether all enumerators have been assigned a value.
    bool is_complete() const { return complete; }

    /// Get the name of this type.
    auto name() const -> DeclName;

    /// Get the scope of this enum.
    auto scope() const -> Scope* { return enum_scope; }

    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::EnumType; }
};

class srcc::OpaqueType final : public TypeBase {
    TypeDecl* type_decl{};
    explicit OpaqueType() : TypeBase(Kind::OpaqueType) {}

public:
    /// Create a type and the corresponding declaration.
    static auto Create(TranslationUnit& tu, String name, SLoc decl_loc) -> OpaqueType*;

    /// Get the declaration of this type.
    auto decl() const -> TypeDecl* { return type_decl; }

    /// Get the name of the type.
    auto name() const -> String;

    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::OpaqueType; }
};

class srcc::OptionalType final : public SingleElementTypeBase
    , public FoldingSetNode {
    llvm::PointerIntPair<const RecordLayout*, 1> layout_and_field_index = {};
    explicit OptionalType(Type elem, const RecordLayout* rl = nullptr, u32 field_index = 0)
        : SingleElementTypeBase{Kind::OptionalType, elem},
          layout_and_field_index{rl, field_index} {}

public:
    /// Whether this optional type has the same memory representation as
    /// its underlying types (this is the case e.g. for optional pointers).
    [[nodiscard]] bool has_transparent_layout() const {
        return layout_and_field_index.getPointer() == nullptr;
    }

    /// Get the record layout for this optional type.
    ///
    /// Optional types whose memory representation is essentially that of a
    /// 'bool' + the actual type are treated like record types with two fields.
    [[nodiscard]] auto get_equivalent_record_layout() const -> const RecordLayout* {
        Assert(not has_transparent_layout());
        return layout_and_field_index.getPointer();
    }

    /// If this optional type has record layout, the index of the field that
    /// stores whether this is engaged.
    [[nodiscard]] auto get_engaged_field_index() const -> u32 {
        Assert(not has_transparent_layout());
        return layout_and_field_index.getInt();
    }

    /// If this optional type has record layout, the offset of the byte that
    /// stores whether this is engaged.
    [[nodiscard]] auto get_engaged_offset() const -> Size;

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem()); }
    static auto Get(TranslationUnit& mod, Type elem) -> OptionalType*;
    static void Profile(FoldingSetNodeID& ID, Type elem);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::OptionalType; }
};

class srcc::PtrType final : public SingleElementTypeBase
    , public FoldingSetNode {
    const bool immutable;
    explicit PtrType(Type elem, bool immutable)
        : SingleElementTypeBase{Kind::PtrType, elem},
          immutable{immutable} {}

public:
    /// Whether this is an immutable pointer type (that is, whether the *pointee*
    /// is immutable).
    [[nodiscard]] bool is_immutable() const { return immutable; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem(), immutable); }
    static auto Get(TranslationUnit& mod, Type elem, bool immutable) -> PtrType*;
    static void Profile(FoldingSetNodeID& ID, Type elem, bool immutable);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::PtrType; }
};

/// Parts of a parameter that are relevant for the procedure type.
struct srcc::ParamTypeData {
    Intent intent;
    Type type;

    // This is only relevant in Sema when performing overload resolution
    // and has no effect in CodeGen!
    bool variadic;

    ParamTypeData(Intent intent, Type type, bool variadic = false) :
        intent{intent}, type{type}, variadic{variadic} {}
};

class srcc::ProcType final : public TypeBase
    , public FoldingSetNode
    , TrailingObjects<ProcType, ParamTypeData> {
    friend TrailingObjects;

    CallingConvention cc;
    bool is_varargs;
    const u32 num_params;
    TypeAndValueCategory return_type;

    auto numTrailingObjects(OverloadToken<ParamTypeData>) -> usz { return num_params; }

    ProcType(
        CallingConvention cconv,
        bool variadic,
        TypeAndValueCategory return_type,
        ArrayRef<ParamTypeData> param_types
    );

public:
    /// Get the calling convention of this procedure type.
    auto cconv() const -> CallingConvention { return cc; }

    /// Get whether this procedure type has C varargs.
    auto has_c_varargs() const -> bool { return is_varargs; }

    /// Get whether this is a variadic procedure.
    auto is_variadic() const -> bool {
        return num_params != 0 and params()[num_params - 1].variadic;
    }

    /// Get all non-variadic parameters.
    auto non_variadic_params() const -> u32 {
        return param_count() - u32(is_variadic());
    }

    /// Get the number of parameters that this type has.
    auto param_count() const -> u32 { return num_params; }

    /// Get the parameter types of this procedure type.
    auto params() const -> ArrayRef<ParamTypeData> { return getTrailingObjects(num_params); }

    /// Get the parameter types of this procedure type.
    auto param_types() const {
        return params() | vws::transform(&ParamTypeData::type);
    }

    /// Print the proc type, optionally with a name.
    auto print(
        DeclName proc_name = {},
        ProcDecl* decl = nullptr,
        bool include_proc_keyword = true
    ) const -> SmallUnrenderedString;

    /// Get the return type of this procedure type.
    auto ret() const -> Type { return return_type.type(); }
    auto return_value_category() const -> ValueCategory {
        return return_type.value_category();
    }

    bool returns_lvalue() const {
        return return_value_category() != ValueCategory::RValue;
    }

    void Profile(FoldingSetNodeID& ID) const {
        Profile(ID, return_type, params(), cc, is_varargs);
    }

    /// Get a copy of this type, with the return type adjusted
    /// to \p new_ret.
    static auto AdjustRet(TranslationUnit& mod, ProcType* ty, Type new_ret) -> ProcType*;

    static auto Get(
        TranslationUnit& mod,
        TypeAndValueCategory return_type,
        ArrayRef<ParamTypeData> param_types = {},
        CallingConvention cconv = CallingConvention::Native,
        bool variadic = false
    ) -> ProcType*;

    static void Profile(
        FoldingSetNodeID& ID,
        TypeAndValueCategory return_type,
        ArrayRef<ParamTypeData> param_types,
        CallingConvention cconv,
        bool variadic
    );

    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::ProcType; }
};

class srcc::RangeType final : public SingleElementTypeBase
    , public FoldingSetNode {
    TupleType* equivalent_tuple;
    explicit RangeType(Type elem, TupleType* equivalent_tuple)
        : SingleElementTypeBase{Kind::RangeType, elem}, equivalent_tuple{equivalent_tuple} {}

public:
    /// Get the tuple type that is structurally equivalent to this range.
    [[nodiscard]] auto equivalent_tuple_type() const -> TupleType* {
        return equivalent_tuple;
    }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem()); }
    static auto Get(TranslationUnit& mod, Type elem) -> RangeType*;
    static void Profile(FoldingSetNodeID& ID, Type elem);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::RangeType; }
};

// TODO: We also need an 'immutable' flag for slices.
class srcc::SliceType final : public SingleElementTypeBase
    , public FoldingSetNode {
    Type data_ty;
    const bool immutable;
    explicit SliceType(Type elem, Type data_ty, bool immutable)
        : SingleElementTypeBase{Kind::SliceType, elem},
          data_ty{data_ty},
          immutable{immutable} {}

public:
    /// Get the type of the 'data' member of this slice.
    [[nodiscard]] auto data_ptr_type() const -> Type { return data_ty; }

    /// Get whether this is an immutable slice type.
    [[nodiscard]] bool is_immutable() const { return immutable; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem(), immutable); }
    static auto Get(TranslationUnit& mod, Type elem, bool immutable) -> SliceType*;
    static void Profile(FoldingSetNodeID& ID, Type elem, bool immutable);
    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::SliceType; }
};

class srcc::RecordLayout final : llvm::TrailingObjects<RecordLayout, FieldDecl*> {
    LIBBASE_IMMOVABLE(RecordLayout);
    friend TrailingObjects;
    friend auto MakeCanonical(TranslationUnit& tu, const RecordLayout* rl) -> RecordLayout*;

public:
    struct Bits {
        bool default_initialiser : 1 = false;
        bool init_from_no_args   : 1 = false;
        bool literal_initialiser : 1 = false;
        bool zero_init           : 1 = false;
        bool contains_pointer    : 1 = false;
        bool is_union            : 1 = false;
        u8 padding : 2{};

        static auto Trivial(bool contains_pointer) -> Bits {
            return {true, true, true, true, contains_pointer, false};
        }

        void serialise(ByteWriter& w) const { w << std::bit_cast<u8>(*this); }
        static auto deserialise(ByteReader& r) -> Result<Bits> {
            return std::bit_cast<Bits>(Try(r.read<u8>()));
        }
    };

private:
    const u32 num_fields;

public:
    const Align computed_alignment;
    const Bits computed_bits;
    const Size computed_size;
    const Size computed_array_size;

private:
    RecordLayout(
        ArrayRef<FieldDecl*> fields,
        Size sz,
        Size arr_sz,
        Align a,
        Bits bits
    );

public:
    /// Helper to build a record layout.
    class Builder {
        TranslationUnit& tu;
        Size sz;
        Align a;
        RecordLayout::Bits bits;
        SmallVector<FieldDecl*> decls;

    public:
        explicit Builder(TranslationUnit& tu, bool is_union = false): tu{tu} {
            bits.is_union = is_union;
        }

        /// Add a field with the specified type and name.
        ///
        /// This does not perform any checking as to whether the type is even
        /// valid for a field; this must be done before calling this.
        auto add_field(Type ty, String name = "", SLoc loc = {}) -> FieldDecl*;

        /// Build the layout.
        [[nodiscard]] auto build(ArrayRef<ProcDecl*> initialisers = {}) -> RecordLayout*;
    };

    static auto Create(
        TranslationUnit& tu,
        ArrayRef<FieldDecl*> fields,
        Size sz,
        Size arr_sz,
        Align a,
        Bits bits
    ) -> RecordLayout*;

    /// Get the computed alignment of this record.
    auto align() const -> Align {
        return computed_alignment;
    }

    /// Get the size that this record has when stored in an
    /// array (this includes tail padding, unlike 'size()').
    auto array_size() const -> Size {
        return computed_array_size;
    }

    /// Get the record bits.
    auto bits() const -> Bits { return computed_bits; }

    /// Get the record’s fields.
    auto fields() const -> ArrayRef<FieldDecl*> {
        return getTrailingObjects(num_fields);
    }

    /// Get the record’s field types.
    // Defined out of line in Stmt.hh.
    auto field_types() const;

    /// Whether this record has a default initialiser (i.e.
    /// an initialiser that takes no arguments and is *not*
    /// declared by the user).
    bool has_default_init() const { return bits().default_initialiser; }

    /// Whether this record can be initialised with an empty
    /// argument list; this need not entail using a default
    /// initialiser.
    bool has_init_from_no_args() const { return bits().init_from_no_args; }

    /// Whether this record has a literal initialiser (i.e.
    /// an initialiser that takes a list of rvalues and emits
    /// them directly into the record’s memory).
    bool has_literal_init() const { return bits().literal_initialiser; }

    /// Check if this type can be zero-initialised if default
    /// initialisation is requested.
    bool has_zero_init() const { return bits().zero_init; }

    /// Get the size of a single instance of this type; this does
    /// *not* include tail padding (use 'array_size()' instead for
    /// that).
    auto size() const -> Size { return computed_size; }
};

/// Base class for 'TupleType' and 'StructType'.
class srcc::RecordType : public TypeBase {
protected:
    Ptr<ProcDecl> deleter_proc;
    RecordLayout* record_layout = nullptr;
    RecordType(Kind k) : TypeBase{k} {}

public:
    /// Get the deleter of this struct, if any.
    auto deleter() const -> Ptr<ProcDecl> { return deleter_proc; }

    /// Get whether this type is complete, i.e. whether we can
    /// declare variables and create objects of this type.
    bool is_complete() const { return record_layout != nullptr; }

    /// Get the layout of this type.
    auto layout() const -> const RecordLayout& {
        Assert(record_layout);
        return *record_layout;
    }

    /// Set the deleter of this record.
    void set_deleter(ProcDecl* deleter) {
        Assert(not deleter_proc, "Deleter already set");
        Assert(deleter);
        deleter_proc = deleter;
    }

    static bool classof(const TypeBase* t) {
        return t->exact_kind() == Kind::StructType or t->exact_kind() == Kind::TupleType;
    }
};

class srcc::TupleType final : public RecordType, public FoldingSetNode {
    explicit TupleType(RecordLayout* layout) : RecordType{Kind::TupleType} {
        record_layout = layout;
    }

public:
    void Profile(FoldingSetNodeID& ID) const;
    static auto Get(TranslationUnit& tu, ArrayRef<Type> elem_types) -> TupleType*;
    static auto Get(TranslationUnit& tu, RecordLayout* layout) -> TupleType*;
    static void Profile(FoldingSetNodeID& tu, auto elem_types);
    static bool classof(const TypeBase* t) {  return t->exact_kind() == Kind::TupleType; }
};

class srcc::StructType final : public RecordType {
private:
    TranslationUnit& owning_tu; // FIXME: Move this into TypeDecl.
    TypeDecl* type_decl = nullptr;
    StructScope* struct_scope;

    StructType(TranslationUnit& owner, StructScope* struct_scope)
        : RecordType{Kind::StructType},
          owning_tu{owner},
          struct_scope{struct_scope} {}

public:
    /// Create a type and the corresponding declaration.
    static auto Create(
        TranslationUnit& owner,
        StructScope* scope,
        String name,
        SLoc decl_loc,
        RecordLayout* layout = nullptr
    ) -> StructType*;

    /// Get the type declaration for this struct.
    auto decl() const -> TypeDecl* { return type_decl; }

    /// Finalise the struct.
    void finalise(RecordLayout* layout);

    /// Get the user-declared initialisers for this struct.
    auto initialisers() const -> ArrayRef<Decl*> { return struct_scope->inits; }

    /// Get the name of this type.
    auto name() const -> String;

    /// Get the translation unit this is attached to.
    auto owner() const -> TranslationUnit& { return owning_tu; }

    /// Get the scope containing the fields, member functions, and
    /// initialisers of this struct.
    auto scope() const -> StructScope* { return struct_scope; }

    static bool classof(const TypeBase* e) { return e->exact_kind() == Kind::StructType; }
};

/// Visit this type.
template <typename Visitor>
auto srcc::TypeBase::visit(Visitor&& v) const -> decltype(auto) {
    // We const_cast here because types are never modified anyway,
    // so the 'const' is just superfluous and does nothing.
    switch (canonical_kind()) {
        // Canonical types.
#       define AST_TYPE_SUGAR(node)
#       define AST_TYPE_LEAF(node)                                \
           case Kind::node: return std::invoke(                   \
                std::forward<Visitor>(v),                         \
                const_cast<node*>(static_cast<const node*>(this)) \
            );
#       include "srcc/AST.inc"

        // Type sugar.
#       define AST_TYPE_SUGAR(node) case Kind::node:
#       include "srcc/AST.inc"
            Unreachable("Canonical type should never be type sugar");
    }
    Unreachable();
}

template <>
struct std::formatter<srcc::Type> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::Type t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t->print().str()}, ctx);
    }
};

template <std::derived_from<srcc::TypeBase> Ty>
struct std::formatter<Ty*> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(Ty* t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t ? t->print().str() : "(null)"}, ctx);
    }
};

struct srcc::types::detail::CanonicalTypeDenseMapInfo {
    static constexpr Type getEmptyKey() {
        return reinterpret_cast<srcc::TypeBase*>(~0zu);
    }

    static constexpr Type getTombstoneKey() {
        return reinterpret_cast<srcc::TypeBase*>(~1zu);
    }

    static unsigned getHashValue(srcc::Type v) {
        return llvm::DenseMapInfo<void*>::getHashValue(v.ptr()->canonical);
    }

    static bool isEqual(srcc::Type lhs, srcc::Type rhs) {
        return lhs == rhs;
    }
};

struct srcc::types::detail::NonCanonicalTypeDenseMapInfo {
    static constexpr Type getEmptyKey() {
        return reinterpret_cast<srcc::TypeBase*>(~0zu);
    }

    static constexpr Type getTombstoneKey() {
        return reinterpret_cast<srcc::TypeBase*>(~1zu);
    }

    static unsigned getHashValue(srcc::Type v) {
        return llvm::DenseMapInfo<void*>::getHashValue(v.ptr());
    }

    static bool isEqual(srcc::Type lhs, srcc::Type rhs) {
        // static_cast to void is required because we disallow pointer
        // comparisons involving 'TypeBase*'.
        return static_cast<void*>(lhs.ptr()) == static_cast<void*>(rhs.ptr());
    }
};

namespace srcc {
/// Map that maps 'Type's to 'Value's, ignoring type sugar.
template <typename Value>
using CanonicalTypeMap = llvm::DenseMap<
    Type,
    Value,
    types::detail::CanonicalTypeDenseMapInfo
>;

/// Map that maps 'Type's to 'Value's, preserving type sugar.
template <typename Value>
using NonCanonicalTypeMap = llvm::DenseMap<
    Type,
    Value,
    types::detail::NonCanonicalTypeDenseMapInfo
>;
}

template <>
struct libassert::stringifier<srcc::Type> {
    static auto stringify(srcc::Type ty) -> std::string {
        return base::text::RenderColours(false, ty->print().str());
    }
};

template <typename To>
[[nodiscard]] auto srcc::dyn_cast(Type from) -> To* {
    if constexpr (IsTypeSugarClass<To>()) return llvm::dyn_cast<To>(from.ptr());
    return llvm::dyn_cast<To>(from->canonical);
}

template <typename To>
[[nodiscard]] auto srcc::cast(Type from) -> To* {
    if constexpr (IsTypeSugarClass<To>()) return llvm::dyn_cast<To>(from.ptr());
    return llvm::cast<To>(from->canonical);
}

template <typename To>
[[nodiscard]] auto srcc::cast_if_present(Opt<Type> ty) -> To* {
    if (ty.has_value()) return srcc::cast<To>(*ty);
    return nullptr;
}

template <typename... Ts>
[[nodiscard]] auto srcc::isa(Type from) -> bool {
    if constexpr ((IsTypeSugarClass<Ts>() or ...))
        if (llvm::isa<Ts...>(from.ptr()))
            return true;

    return llvm::isa<Ts...>(from->canonical);
}

template <typename... Ts>
[[nodiscard]] auto srcc::isa_and_nonnull(Opt<Type> ty) -> bool {
    if (ty.has_value()) return srcc::isa<Ts...>(ty.value());
    return false;
}

#endif // SRCC_AST_TYPE_HH
