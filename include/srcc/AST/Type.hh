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
class TemplateTypeDecl;
class TranslationUnit;
class Types;
struct ParamTypeData;
class Expr;
class Decl;
class FieldDecl;
class TypeDecl;

#define AST_TYPE(node) class node;
#include "srcc/AST.inc"

template <typename Wrapped>
class TypeWrapper;
using Type = TypeWrapper<TypeBase>;
class TypeLoc;

/// Casting.
template <typename To, typename From> auto cast(TypeWrapper<From> from) -> TypeWrapper<To>;
template <typename To, typename From> auto dyn_cast(TypeWrapper<From> from) -> std::optional<TypeWrapper<To>>;
template <typename To, typename From> auto isa(TypeWrapper<From> from) -> bool;
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
    StringMap<llvm::TinyPtrVector<Decl*>> decls;

    virtual ~Scope(); // Defaulted out-of-line.

    /// Check if this is a procedure scope.
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

/// Type of an expression or declaration.
///
/// Types are immutable, so it’s fine to pass them by non-`const`
/// reference or pointer in most cases.
class srcc::TypeBase {
    SRCC_IMMOVABLE(TypeBase);

    template <typename Wrapped>
    friend class TypeWrapper;

public:
    enum struct Kind : u8 {
#define AST_TYPE_LEAF(node) node,
#include "srcc/AST.inc"

    };

    const Kind type_kind;
    Dependence dep = Dependence::None;

protected:
    explicit constexpr TypeBase(Kind kind, Dependence dep = Dependence::None) : type_kind{kind}, dep{dep} {}

public:
    // Only allow allocating these in the module.
    void* operator new(usz) = SRCC_DELETED("Use `new (mod) { ... }` instead");
    void* operator new(usz size, TranslationUnit& mod);

    /// Get the alignment of this type.
    LLVM_READONLY auto align(TranslationUnit& tu) const -> Align;

    /// Get the size of this type when stored in an array.
    LLVM_READONLY auto array_size(TranslationUnit& tu) const -> Size;

    /// Get whether this type can be initialised using an empty
    /// argument list. For struct types, this can entail calling
    /// an initialiser with no arguments.
    LLVM_READONLY bool can_init_from_no_args() const;

    /// Get whether this type has a default initialiser.
    LLVM_READONLY bool can_default_init() const;

    /// Get the dependence of this type.
    auto dependence() const -> Dependence { return dep; }

    /// Check if this is dependent.
    bool dependent() const { return dep != Dependence::None; }

    /// Print a type to stdout.
    void dump(bool use_colour = false) const;
    void dump_colour() const;

    /// Check if this type contains an error.
    bool errored() const { return dep & Dependence::Error; }

    /// Get the kind of this type.
    auto kind() const -> Kind { return type_kind; }

    /// Check if this is 'int' or a sized integer.
    LLVM_READONLY bool is_integer() const;

    /// Check if this type is the builtin 'void' type.
    LLVM_READONLY bool is_void() const;

    /// Whether this type should be passed as an lvalue given a specific intent.
    LLVM_READONLY bool pass_by_lvalue(CallingConvention cc, Intent intent) const {
        return not pass_by_rvalue(cc, intent);
    }

    /// Whether this type should be passed as an rvalue given a specific intent.
    LLVM_READONLY bool pass_by_rvalue(CallingConvention cc, Intent intent) const;

    /// Get a string representation of this type.
    LLVM_READONLY auto print() const -> SmallUnrenderedString;

    /// Get the size of this type. This does NOT include tail padding!
    LLVM_READONLY auto size(TranslationUnit& tu) const -> Size;

    /// Get the value category that an expression with this type
    /// has by default. This will usually be some kind of rvalue.
    LLVM_READONLY auto value_category() const -> ValueCategory;

    /// Visit this type.
    template <typename Visitor>
    auto visit(Visitor&& v) const -> decltype(auto);

protected:
    void ComputeDependence();
};

class srcc::BuiltinType final : public TypeBase {
    friend Types;
    const BuiltinKind b_kind;

    explicit constexpr BuiltinType(BuiltinKind kind, Dependence dep = Dependence::None)
        : TypeBase{Kind::BuiltinType, dep}, b_kind{kind} {}

public:
    /// Get the kind of this builtin type.
    auto builtin_kind() const -> BuiltinKind { return b_kind; }

    static auto Get(TranslationUnit& mod, BuiltinKind kind) = SRCC_DELETED("Use Types::VoidTy and friends instead");
    static bool classof(const TypeBase* e) { return e->kind() == Kind::BuiltinType; }
};

class srcc::IntType final : public TypeBase
    , public FoldingSetNode {
    friend Types;

public:
    static constexpr Size MaxBits = Size::Bits(u64(llvm::IntegerType::MAX_INT_BITS));

private:
    Size bits;

    explicit constexpr IntType(Size bit_width) : TypeBase{Kind::IntType}, bits{bit_width} {
        Assert(bits >= Size::Bits(1), "Cannot create integer type with bit width 0");
        Assert(bits <= MaxBits, "Bit width too large: {}", bits);
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
template <typename Wrapped = srcc::TypeBase>
class srcc::TypeWrapper {
    template <typename> friend class TypeWrapper;

public:
    using WrappedType = Wrapped;

private:
    WrappedType* pointer;

    TypeWrapper() = default;

public:
    constexpr TypeWrapper(WrappedType* t) : pointer{t} { Check(); }

    template <std::derived_from<WrappedType> Derived>
    TypeWrapper(Derived* derived) : pointer{derived} { Check(); }

    template <std::derived_from<WrappedType> Derived>
    TypeWrapper(TypeWrapper<Derived> derived) : pointer{derived.ptr()} { Check(); }

    TypeWrapper(std::nullptr_t) = SRCC_DELETED("Use `TypeWrapper::UnsafeNull()` if you must");

    /// Get this as an opaque pointer.
    auto as_opaque_ptr() const -> void* { return pointer; }

    /// Get the type pointer.
    auto ptr() const -> WrappedType* { return pointer; }

    /// Access the type pointer.
    auto operator->() const -> WrappedType* { return pointer; }

    /// Check if two types are equal.
    template <typename T>
    auto operator==(TypeWrapper<T> other) const -> bool { return pointer == other.pointer; }
    auto operator==(const TypeBase* ty) const -> bool { return pointer == ty; }

    /// Construct a `TypeWrapper` from an opaque pointer.
    static auto FromOpaquePointer(void* ptr) -> TypeWrapper {
        return TypeWrapper{static_cast<WrappedType*>(ptr)};
    }

    /// Construct a null `TypeWrapper`.
    ///
    /// This should be used very sparingly; a type that is passed
    /// to a function or returned from one must never be null; this
    /// is only to allow late initialisation of fields.
    static constexpr auto UnsafeNull() -> TypeWrapper {
        TypeWrapper t;
        t.pointer = nullptr;
        return t;
    }

private:
    constexpr void Check() {
        Assert(pointer, "Null type pointer can only be constructed with `TypeWrapper::UnsafeNull()`");
    }
};

class srcc::TypeLoc {
public:
    Type ty;
    Location loc;
    TypeLoc(Type ty, Location loc) : ty{ty}, loc{loc} {}
    TypeLoc(Expr* e);
};

// Specialisation that uses the null state to represent a null type.
template <typename Ty>
class srcc::Opt<srcc::TypeWrapper<Ty>> {
    using Type = TypeWrapper<Ty>;
    Type val;

public:
    constexpr Opt() : val{Type::UnsafeNull()} {}
    constexpr Opt(std::nullptr_t) : Opt() {}
    constexpr Opt(std::nullopt_t) : Opt() {}
    constexpr Opt(Type ty) : val{ty} {}
    constexpr Opt(Ty* ty) : val{ty} {}

    constexpr bool has_value() const { return val != Type::UnsafeNull(); }
    constexpr auto value() const -> Type {
        Assert(has_value());
        return val;
    }

    constexpr auto value() -> Type& {
        Assert(has_value());
        return val;
    }

    constexpr explicit operator bool() const { return has_value(); }
};

// The const_cast here is fine because we never modify types anyway.
#define BUILTIN_TYPE(Name, ...)                                                              \
private:                                                                                     \
    static constexpr BuiltinType Name##TyImpl{BuiltinKind::Name __VA_OPT__(, ) __VA_ARGS__}; \
public:                                                                                      \
    static constexpr TypeWrapper<BuiltinType> Name##Ty = const_cast<BuiltinType*>(&Name##Ty##Impl)

// FIXME: Eliminate 'TypeWrapper' in favour of a single 'Type' class and move these there?
class srcc::Types {
    BUILTIN_TYPE(Void);
    BUILTIN_TYPE(Dependent, Dependence::Type);
    BUILTIN_TYPE(ErrorDependent, Dependence::Type | Dependence::Error);
    BUILTIN_TYPE(NoReturn);
    BUILTIN_TYPE(Bool);
    BUILTIN_TYPE(Int);
    BUILTIN_TYPE(Deduced);
    BUILTIN_TYPE(Type);
    BUILTIN_TYPE(UnresolvedOverloadSet);
};

class srcc::SingleElementTypeBase : public TypeBase {
    Type element_type;

protected:
    SingleElementTypeBase(Kind kind, Type elem) : TypeBase{kind}, element_type{elem} {}

public:
    /// Get the element type of this type, e.g. `int` for `int[3]`.
    auto elem() const -> Type { return element_type; }

    static bool classof(const TypeBase* e) {
        return e->kind() >= Kind::ArrayType and e->kind() <= Kind::ReferenceType;
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
        ComputeDependence();
    }

public:
    /// Get the number of elements in this array.
    auto dimension() const -> i64 { return elems; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem(), elems); }
    static auto Get(TranslationUnit& mod, Type elem, i64 size) -> ArrayType*;
    static void Profile(FoldingSetNodeID& ID, Type elem, i64 size);
    static bool classof(const TypeBase* e) { return e->kind() == Kind::ArrayType; }
};

class srcc::ReferenceType final : public SingleElementTypeBase
    , public FoldingSetNode {
    explicit ReferenceType(Type elem) : SingleElementTypeBase{Kind::ReferenceType, elem} {
        ComputeDependence();
    }

public:
    void Profile(FoldingSetNodeID& ID) const { Profile(ID, elem()); }
    static auto Get(TranslationUnit& mod, Type elem) -> ReferenceType*;
    static void Profile(FoldingSetNodeID& ID, Type elem);
    static bool classof(const TypeBase* e) { return e->kind() == Kind::ReferenceType; }
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

    static auto GetInvalid(TranslationUnit& tu) -> ProcType*;

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
    explicit SliceType(Type elem) : SingleElementTypeBase{Kind::SliceType, elem} {
        ComputeDependence();
    }

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
        bool init_from_no_args : 1 = false;
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

class srcc::TemplateType final : public TypeBase
    , public FoldingSetNode {
    TemplateTypeDecl* decl;
    explicit TemplateType(TemplateTypeDecl* decl)
        : TypeBase{Kind::TemplateType},
          decl{decl} {
        ComputeDependence();
    }

public:
    auto template_decl() const -> TemplateTypeDecl* { return decl; }

    void Profile(FoldingSetNodeID& ID) const { Profile(ID, decl); }
    static auto Get(TranslationUnit& mod, TemplateTypeDecl* decl) -> TemplateType*;
    static void Profile(FoldingSetNodeID& ID, TemplateTypeDecl* decl);
    static bool classof(const TypeBase* e) { return e->kind() == Kind::TemplateType; }
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

template <typename Ty>
struct std::formatter<srcc::TypeWrapper<Ty>> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::TypeWrapper<Ty> t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t->print().str()}, ctx);
    }
};

template <std::derived_from<srcc::TypeBase> Ty>
struct std::formatter<Ty*> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(Ty* t, FormatContext& ctx) const {
        return std::formatter<std::string_view>::format(std::string_view{t->print().str()}, ctx);
    }
};

template <typename To, typename From>
auto srcc::cast(srcc::TypeWrapper<From> from) -> srcc::TypeWrapper<To> {
    return srcc::TypeWrapper<To>{llvm::cast<To>(from.ptr())};
}

template <typename To, typename From>
auto srcc::dyn_cast(srcc::TypeWrapper<From> from) -> std::optional<srcc::TypeWrapper<To>> {
    if (auto* c = llvm::dyn_cast<To>(from.ptr())) return srcc::TypeWrapper<To>{c};
    return std::nullopt;
}

template <typename To, typename From>
auto srcc::isa(srcc::TypeWrapper<From> from) -> bool {
    return llvm::isa<To>(from.ptr());
}

#endif // SRCC_AST_TYPE_HH
