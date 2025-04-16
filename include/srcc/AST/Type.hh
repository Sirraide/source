#ifndef SRCC_AST_TYPE_HH
#define SRCC_AST_TYPE_HH

#include <srcc/AST/Enums.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

namespace srcc {
class Scope;
class StructScope;
class TemplateTypeParamDecl;
class TranslationUnit;
struct ParamTypeData;
class Expr;
class Decl;
class FieldDecl;
class ProcTemplateDecl;
class TypeDecl;
struct BuiltinTypes;

#define AST_TYPE(node) class node;
#include "srcc/AST.inc"

class Type;
class TypeLoc;

/// Casting.
} // namespace srcc

/// Scope that stores declarations.
///
/// These need to be allocated separately because we usually need
/// to be able to look up declarations in one before we create the
/// node that contains it.
///
/// Note that these are only valid during sema and should not be
/// referenced after that.
class srcc::Scope {
    SRCC_IMMOVABLE(Scope);

    friend TranslationUnit;

    /// The parent scope.
    Scope* parent_scope;

    /// Whether this is a procedure scope.
    const ScopeKind kind;

protected:
    explicit Scope(Scope* parent, ScopeKind k = ScopeKind::Block);

public:
    /// Declarations in this scope.
    StringMap<llvm::TinyPtrVector<Decl*>> decls_by_name;

    virtual ~Scope(); // Defaulted out-of-line.

    /// Get a flat list of all declarations in this scope.
    auto decls();

    /// Check if this is a specific kind of scope.
    bool is_block_scope() const { return kind == ScopeKind::Block; }
    bool is_proc_scope() const { return kind == ScopeKind::Procedure; }
    bool is_struct_scope() const { return kind == ScopeKind::Struct; }

    /// Get the parent scope.
    ///
    /// This returns null if this is the global scope.
    auto parent() -> Scope* { return parent_scope; }
};

/// Scope that stores struct members, initialisers, etc.
class srcc::StructScope : public Scope {
    friend TranslationUnit;

    StructScope(Scope* parent) : Scope{parent, ScopeKind::Struct} {}

public:
    /// Initialiser declarations.
    SmallVector<Decl*> inits;
};

/// Base class for all types.
class srcc::TypeBase {
    SRCC_IMMOVABLE(TypeBase);

    friend Type;

public:
    enum struct Kind : u8 {
#define AST_TYPE_LEAF(node) node,
#include "srcc/AST.inc"

    };

    const Kind type_kind;

protected:
    explicit constexpr TypeBase(Kind kind) : type_kind{kind} {}

public:
    // Only allow allocating these in the module.
    void* operator new(usz) = SRCC_DELETED("Use `new (mod) { ... }` instead");
    void* operator new(usz size, TranslationUnit& mod);

    /// Get the alignment of this type.
    [[nodiscard]] auto align(TranslationUnit& tu) const -> Align;

    /// Get the size of this type when stored in an array.
    [[nodiscard]] auto array_size(TranslationUnit& tu) const -> Size;

    /// Get whether this type can be initialised using an empty
    /// argument list. For struct types, this can entail calling
    /// an initialiser with no arguments.
    [[nodiscard]] bool can_init_from_no_args() const;

    /// Get whether this type has a default initialiser.
    [[nodiscard]] bool can_default_init() const;

    /// Print a type to stdout.
    void dump(bool use_colour = false) const;
    void dump_colour() const;

    /// Check if this is 'int' or a sized integer.
    [[nodiscard]] bool is_integer() const;

    /// Whether values of this type are mrvalues.
    [[nodiscard]] bool is_mrvalue() const { return not is_srvalue(); }

    /// Whether values of this type are srvalues.
    [[nodiscard]] bool is_srvalue() const;

    /// Check if this type is the builtin 'void' type.
    [[nodiscard]] bool is_void() const;

    /// Whether this type should be passed as an lvalue given a specific intent.
    [[nodiscard]] bool pass_by_lvalue(CallingConvention cc, Intent intent) const {
        return not pass_by_rvalue(cc, intent);
    }

    /// Whether this type should be passed as an rvalue given a specific intent.
    [[nodiscard]] bool pass_by_rvalue(CallingConvention cc, Intent intent) const;

    /// Get a string representation of this type.
    [[nodiscard]] auto print() const -> SmallUnrenderedString;

    /// Get what kind of rvalue this type produced.
    [[nodiscard]] auto rvalue_category() const -> ValueCategory {
        return is_srvalue() ? ValueCategory::SRValue : ValueCategory::MRValue;
    }

    /// Get the size of this type. This does NOT include tail padding!
    [[nodiscard]] auto size(TranslationUnit& tu) const -> Size;

    /// Visit this type.
    template <typename Visitor>
    auto visit(Visitor&& v) const -> decltype(auto);

    /// Get the type kind.
    [[nodiscard]] auto kind() const -> Kind { return type_kind; }
};

class srcc::BuiltinType final : public TypeBase {
    friend BuiltinTypes;

    const BuiltinKind b_kind;

    explicit constexpr BuiltinType(BuiltinKind kind)
        : TypeBase{Kind::BuiltinType}, b_kind{kind} {}

public:
    /// Get the kind of this builtin type.
    auto builtin_kind() const -> BuiltinKind { return b_kind; }

    static auto Get(TranslationUnit& mod, BuiltinKind kind) = SRCC_DELETED("Use Types::VoidTy and friends instead");
    static bool classof(const TypeBase* e) { return e->kind() == Kind::BuiltinType; }
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
    static bool classof(const TypeBase* e) { return e->kind() == Kind::IntType; }
};

/// A type together with qualifiers.
class srcc::Type {
    TypeBase* pointer = nullptr;

public:
    explicit constexpr Type() = default;
    constexpr Type(TypeBase* t) : pointer{t} {}

    /// Get the type pointer.
    [[nodiscard]] auto ptr() const -> TypeBase* { return pointer; }

    /// Access the type pointer.
    [[nodiscard]] auto operator->() const -> TypeBase* { return pointer; }

    /// Check if two types are equal.
    [[nodiscard]] auto operator==(Type ty) const -> bool { return pointer == ty.pointer; }
    [[nodiscard]] auto operator==(const TypeBase* ty) const -> bool { return pointer == ty; }

    /// Check whether this holds a valid type.
    [[nodiscard]] explicit operator bool() const { return pointer != nullptr; }

    static const Type VoidTy;
    static const Type NoReturnTy;
    static const Type BoolTy;
    static const Type IntTy;
    static const Type DeducedTy;
    static const Type TypeTy;
    static const Type UnresolvedOverloadSetTy;
};

template <>
struct llvm::PointerLikeTypeTraits<srcc::Type> {
    static constexpr int NumLowBitsAvailable = PointerLikeTypeTraits<srcc::TypeBase*>::NumLowBitsAvailable;
    static constexpr bool isPtrLike = true;
    static auto getAsVoidPointer(srcc::Type t) -> void* { return t.ptr(); }
    static auto getFromVoidPointer(void* p) -> srcc::Type { return srcc::Type{static_cast<srcc::TypeBase*>(p)}; }
};

template <>
struct llvm::simplify_type<srcc::Type> {
    using SimpleType = srcc::TypeBase*;
    static SimpleType getSimplifiedValue(srcc::Type v) { return v.ptr(); }
};

template <>
struct llvm::simplify_type<const srcc::Type> {
    using SimpleType = srcc::TypeBase*;
    static SimpleType getSimplifiedValue(srcc::Type v) { return v.ptr(); }
};

class srcc::TypeLoc {
public:
    Type ty;
    Location loc;
    TypeLoc(Type ty, Location loc) : ty{ty}, loc{loc} {}
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
        return e->kind() >= Kind::ArrayType and e->kind() <= Kind::PtrType;
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
    static bool classof(const TypeBase* e) { return e->kind() == Kind::ArrayType; }
};

class srcc::PtrType final : public SingleElementTypeBase
    , public FoldingSetNode {
    explicit PtrType(Type elem) : SingleElementTypeBase{Kind::PtrType, elem} {}

public:
    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem()); }
    static auto Get(TranslationUnit& mod, Type elem) -> PtrType*;
    static void Profile(FoldingSetNodeID& ID, Type elem);
    static bool classof(const TypeBase* e) { return e->kind() == Kind::PtrType; }
};

/// Parts of a parameter that are relevant for the procedure type.
struct srcc::ParamTypeData {
    Intent intent;
    Type type;
    ParamTypeData(Intent intent, Type type) : intent{intent}, type{type} {}
};

class srcc::ProcType final : public TypeBase
    , public FoldingSetNode
    , TrailingObjects<ProcType, ParamTypeData> {
    friend TrailingObjects;

    CallingConvention cc;
    bool is_variadic;
    const u32 num_params;
    Type return_type;

    auto numTrailingObjects(OverloadToken<ParamTypeData>) -> usz { return num_params; }

    ProcType(
        CallingConvention cconv,
        bool variadic,
        Type return_type,
        ArrayRef<ParamTypeData> param_types
    );

public:
    /// Get the calling convention of this procedure type.
    auto cconv() const -> CallingConvention { return cc; }

    /// Get the parameter types of this procedure type.
    auto params() const -> ArrayRef<ParamTypeData> { return {getTrailingObjects<ParamTypeData>(), num_params}; }

    /// Print the proc type, optionally with a name.
    auto print(StringRef proc_name = "", bool number_params = false) const -> SmallUnrenderedString;

    /// Get the return type of this procedure type.
    auto ret() const -> Type { return return_type; }

    /// Get whether this procedure type is variadic.
    auto variadic() const -> bool { return is_variadic; }

    void Profile(FoldingSetNodeID& ID) const {
        Profile(ID, return_type, params(), cc, is_variadic);
    }

    /// Get a copy of this type, with the return type adjusted
    /// to \p new_ret.
    static auto AdjustRet(TranslationUnit& mod, ProcType* ty, Type new_ret) -> ProcType*;

    static auto Get(
        TranslationUnit& mod,
        Type return_type,
        ArrayRef<ParamTypeData> param_types = {},
        CallingConvention cconv = CallingConvention::Source,
        bool variadic = false
    ) -> ProcType*;

    static void Profile(
        FoldingSetNodeID& ID,
        Type return_type,
        ArrayRef<ParamTypeData> param_types,
        CallingConvention cconv,
        bool variadic
    );

    static bool classof(const TypeBase* e) { return e->kind() == Kind::ProcType; }
};

class srcc::SliceType final : public SingleElementTypeBase
    , public FoldingSetNode {
    explicit SliceType(Type elem) : SingleElementTypeBase{Kind::SliceType, elem} {}

public:
    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem()); }
    static auto Get(TranslationUnit& mod, Type elem) -> SliceType*;
    static void Profile(FoldingSetNodeID& ID, Type elem);
    static bool classof(const TypeBase* e) { return e->kind() == Kind::SliceType; }
};

class srcc::StructType final : public TypeBase
    , TrailingObjects<StructType, FieldDecl*> {
    friend TrailingObjects;

    u32 num_fields;
    auto numTrailingObjects(OverloadToken<FieldDecl*>) -> usz { return num_fields; }

public:
    struct Bits {
        bool default_initialiser : 1 = false;
        bool init_from_no_args   : 1 = false;
        bool literal_initialiser : 1 = false;
    };

private:
    Size computed_size;
    Size computed_array_size;
    Align computed_alignment;
    bool finalised : 1 = false;
    Bits bits;

    TranslationUnit& owning_tu;
    TypeDecl* type_decl = nullptr;
    StructScope* struct_scope;

    StructType(TranslationUnit& owner, StructScope* struct_scope, u32 num_fields)
        : TypeBase{Kind::StructType},
          num_fields{num_fields},
          owning_tu{owner},
          struct_scope{struct_scope} {}

public:
    /// Create a type and the corresponding declaration.
    ///
    /// The fields will be filled in later, but note that the
    /// field count cannot be changed once the type has been
    /// created.
    static auto Create(
        TranslationUnit& owner,
        StructScope* scope,
        String name,
        u32 num_fields,
        Location decl_loc
    ) -> StructType*;

    /// Get the computed alignment of this struct.
    auto align() const -> Align {
        Assert(finalised);
        return computed_alignment;
    }

    /// Get the size that this struct has when stored in an
    /// array (this includes tail padding, unlike 'size()').
    auto array_size() const -> Size {
        Assert(finalised);
        return computed_array_size;
    }

    /// Get the type declaration for this struct.
    auto decl() const -> TypeDecl* { return type_decl; }

    /// Get the struct’s fields.
    auto fields() const -> ArrayRef<FieldDecl*> {
        Assert(finalised);
        return {getTrailingObjects<FieldDecl*>(), num_fields};
    }

    /// Initialise fields and other properties; this marks
    /// the struct as complete.
    void finalise(
        ArrayRef<FieldDecl*> fields,
        Size size,
        Align alignment,
        Bits bits
    );

    /// Whether this struct has a default initialiser (i.e.
    /// an initialiser that takes no arguments and is *not*
    /// declared by the user).
    bool has_default_init() const { return bits.default_initialiser; }

    /// Whether this struct can be initialised with an empty
    /// argument list; this need not entail using a default
    /// initialiser.
    bool has_init_from_no_args() const { return bits.init_from_no_args; }

    /// Whether this struct has a literal initialiser (i.e.
    /// an initialiser that takes a list of rvalues and emits
    /// them directly into the struct’s memory).
    bool has_literal_init() const { return bits.literal_initialiser; }

    /// Get the user-declared initialisers for this struct.
    auto initialisers() const -> ArrayRef<Decl*> { return struct_scope->inits; }

    /// Get whether this type is complete, i.e. whether we can
    /// declare variables and create objects of this type.
    bool is_complete() const { return finalised; }

    /// Get the name of this type.
    auto name() const -> String;

    /// Get the translation unit this is attached to.
    auto owner() const -> TranslationUnit& { return owning_tu; }

    /// Get the scope containing the fields, member functions, and
    /// initialisers of this struct.
    auto scope() const -> StructScope* { return struct_scope; }

    /// Get the size of a single instance of this type; this does
    /// *not* include tail padding (use 'array_size()' instead for
    /// that).
    auto size() const -> Size { return computed_size; }

    static bool classof(const TypeBase* e) { return e->kind() == Kind::StructType; }
};

/// Visit this type.
template <typename Visitor>
auto srcc::TypeBase::visit(Visitor&& v) const -> decltype(auto) {
    // We const_cast here because types are never modified anyway,
    // so the 'const' is just superfluous and does nothing.
    switch (kind()) {
#define AST_TYPE_LEAF(node)                               \
    case Kind::node: return std::invoke(                  \
        std::forward<Visitor>(v),                         \
        const_cast<node*>(static_cast<const node*>(this)) \
    );
#include "srcc/AST.inc"
    }
    Unreachable();
}

template <>
struct std::formatter<srcc::Type> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::Type t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t ? t->print().str() : "(null)"}, ctx);
    }
};

template <std::derived_from<srcc::TypeBase> Ty>
struct std::formatter<Ty*> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(Ty* t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t ? t->print().str() : "(null)"}, ctx);
    }
};

/*
template <typename To>
auto srcc::cast(srcc::Type from) -> To* {
    return llvm::cast<To>(from.ptr());
}

template <typename To>
auto srcc::dyn_cast(srcc::Type from) -> To* {
    return llvm::dyn_cast<To>(from.ptr());
}

template <typename... Ts>
auto srcc::isa(srcc::Type from) -> bool {
    return llvm::isa<Ts...>(from.ptr());
}*/

#endif // SRCC_AST_TYPE_HH
