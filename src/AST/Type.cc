#include <srcc/AST/AST.hh>
#include <srcc/AST/Enums.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/Core/Utils.hh>

#include <clang/Basic/TargetInfo.h>

#include <llvm/ADT/FoldingSet.h>

#include <base/StringUtils.hh>
#include <base/Utils.hh>

#include <memory>
#include <print>

using namespace srcc;

struct srcc::BuiltinTypes {
    static constexpr BuiltinType VoidTyImpl{BuiltinKind::Void};
    static constexpr BuiltinType NoReturnTyImpl{BuiltinKind::NoReturn};
    static constexpr BuiltinType BoolTyImpl{BuiltinKind::Bool};
    static constexpr BuiltinType IntTyImpl{BuiltinKind::Int};
    static constexpr BuiltinType DeducedTyImpl{BuiltinKind::Deduced};
    static constexpr BuiltinType TreeTyImpl{BuiltinKind::Tree};
    static constexpr BuiltinType TypeTyImpl{BuiltinKind::Type};
    static constexpr BuiltinType UnresolvedOverloadSetTyImpl{BuiltinKind::UnresolvedOverloadSet};
    static constexpr BuiltinType CallArgListTyImpl{BuiltinKind::CallArgList};
    static constexpr BuiltinType NilTyImpl{BuiltinKind::Nil};
};

constexpr Type Type::VoidTy{const_cast<BuiltinType*>(&BuiltinTypes::VoidTyImpl)};
constexpr Type Type::NoReturnTy{const_cast<BuiltinType*>(&BuiltinTypes::NoReturnTyImpl)};
constexpr Type Type::BoolTy{const_cast<BuiltinType*>(&BuiltinTypes::BoolTyImpl)};
constexpr Type Type::IntTy{const_cast<BuiltinType*>(&BuiltinTypes::IntTyImpl)};
constexpr Type Type::DeducedTy{const_cast<BuiltinType*>(&BuiltinTypes::DeducedTyImpl)};
constexpr Type Type::TreeTy{const_cast<BuiltinType*>(&BuiltinTypes::TreeTyImpl)};
constexpr Type Type::TypeTy{const_cast<BuiltinType*>(&BuiltinTypes::TypeTyImpl)};
constexpr Type Type::UnresolvedOverloadSetTy{const_cast<BuiltinType*>(&BuiltinTypes::UnresolvedOverloadSetTyImpl)};
constexpr Type Type::CallArgListTy{const_cast<BuiltinType*>(&BuiltinTypes::CallArgListTyImpl)};
constexpr Type Type::NilTy{const_cast<BuiltinType*>(&BuiltinTypes::NilTyImpl)};

// ============================================================================
//  Helpers
// ============================================================================
template <typename T, typename... Args>
auto GetOrCreateType(FoldingSet<T>& Set, auto CreateNew, Args&&... args) -> T* {
    FoldingSetNodeID ID;
    T::Profile(ID, std::forward<Args>(args)...);

    llvm::FoldingSetInsertToken pos;
    auto* type = Set.lookup(ID, pos);
    if (not type) {
        type = CreateNew();
        Set.insert(type, pos);
    }

    return type;
}

TypeLoc::TypeLoc(Expr* e) : ty{e->type}, loc{e->location()} {}

// ============================================================================
//  Scope
// ============================================================================
Scope::~Scope() = default;
Scope::Scope(Scope* parent, ScopeKind k)
    : parent_scope_and_kind{parent, k} {}

// ============================================================================
//  Type
// ============================================================================
void* TypeBase::operator new(usz size, TranslationUnit& mod) {
    return mod.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

auto TypeBase::align(TranslationUnit& tu) const -> Align {
    return align(tu.target());
}

// TODO: Cache type size and alignment in the TU.
auto TypeBase::align(const Target& t) const -> Align { // clang-format off
    return visit(utils::Overloaded{
        [&](const ArrayType* ty) -> Align { return ty->elem()->align(t); },
        [&](const BuiltinType* ty) -> Align {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool: return Align{1};
                case BuiltinKind::Int: return t.int_align();
                case BuiltinKind::NoReturn: return Align{1};
                case BuiltinKind::Void: return Align{1};
                case BuiltinKind::Nil: return Align{1};
                case BuiltinKind::Tree: return Align::Of<TreeValue*>(); // This is a compile-time only type.
                case BuiltinKind::Type: return Align::Of<TypeBase*>(); // This is a compile-time only type.
                case BuiltinKind::UnresolvedOverloadSet: return t.closure_align();
                case BuiltinKind::CallArgList: Unreachable("Requested alignment of init list");
                case BuiltinKind::Deduced: Unreachable("Requested alignment of deduced type");
            }
            Unreachable();
        },
        [&](const EnumType* ty) { return ty->elem()->align(t); },
        [&](const IntType* ty) { return t.int_align(ty); },
        [&](const ProcType*) { return t.closure_align(); },
        [&](const PtrType*) { return t.ptr_align(); },
        [&](const RangeType* ty) { return ty->elem()->align(t); },
        [&](const SliceType*) { return t.slice_align(); },
        [&](const RecordType* ty) {
            Assert(ty->is_complete(), "Requested size of incomplete struct");
            return ty->layout().align();
        },
        [&](const OpaqueType* ty) -> Align { Unreachable("Querying property of opaque type"); },
        [&](const OptionalType* ty) {
            if (ty->has_transparent_layout()) return ty->elem()->align(t);
            return ty->get_equivalent_record_layout()->align();
        },
    });
} // clang-format on

bool TypeBase::pass_by_reference(const Target& t, Intent i) const {
    // Zero-sized types are passed by value.
    if (memory_size(t) == Size()) return false;

    // Large or non-trivially copyable 'in' parameters are references.
    if (i == Intent::In) {
        if (not trivially_copyable()) return true;
        return t.abi().pass_in_parameter_by_reference(this);
    }

    // 'inout' and 'out' parameters are always references.
    if (i == Intent::Inout or i == Intent::Out) return true;

    // Move parameters are references only if the type is not trivial;
    // that is, for trivially-copyable types, any modification of the
    // ‘moved’ value must not be reflected in the caller, so we *must*
    // pass by value rather than by reference.
    //
    // Specifically, moving for these types is *logically* a copy, that
    // is the ‘moved’ value is not actually considered ‘moved’, and the
    // caller may continue accessing it.
    if (i == Intent::Move) return not trivially_copyable();

    // Whether by-value parameters making a copy in the caller and passing a
    // pointer or whether they are passed in registers is up to the
    // target ABI and handled in a separate lowering pass.
    return false;
}

auto TypeBase::array_size(TranslationUnit& tu) const -> Size {
    return array_size(tu.target());
}

auto TypeBase::array_size(const Target& t) const -> Size {
    if (auto s = dyn_cast<RecordType>(canonical)) return s->layout().array_size();
    if (auto s = dyn_cast<RangeType>(canonical)) return s->elem()->array_size(t) * 2;
    return memory_size(t);
}

auto TypeBase::bit_width(TranslationUnit& tu) const -> Size {
    if (canonical == Type::BoolTy) return Size::Bits(1);
    if (auto i = dyn_cast<IntType>(canonical)) return i->bit_width();
    if (auto e = dyn_cast<EnumType>(canonical)) return e->elem()->bit_width(tu);
    return size_impl(tu.target());
}

template <bool (RecordLayout::*struct_predicate)() const>
bool InitCheckHelper(const TypeBase* type) { // clang-format off
    return type->visit(utils::Overloaded{
        [&](const ArrayType* ty) { return InitCheckHelper<struct_predicate>(ty->elem().ptr()); },
        [&](const BuiltinType* ty) {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::Void:
                case BuiltinKind::Nil:
                    return true;

                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Tree:
                case BuiltinKind::Type:
                    return false;

                case BuiltinKind::Deduced:
                case BuiltinKind::CallArgList:
                    Unreachable("Querying property of deduced type");
            }
            Unreachable();
        },
        [&](const EnumType*) { return true; },
        [&](const IntType*) { return true; },
        [&](const OpaqueType* ty) -> bool { Unreachable("Querying property of opaque type"); },
        [&](const OptionalType* ty) { return true; },
        [&](const ProcType*) { return false; },
        [&](const PtrType*) { return false; },
        [&](const RangeType*) { return true; },
        [&](const SliceType*) { return true; },
        [&](const RecordType* ty) { return (ty->layout().*struct_predicate)(); },
    });
} // clang-format on

bool TypeBase::can_copy() const {
    Type t = strip_arrays_and_optionals();
    if (t->trivially_copyable()) return true;
    return t->has_copy_proc();
}

bool TypeBase::can_default_init() const {
    return InitCheckHelper<&RecordLayout::has_default_init>(this);
}

bool TypeBase::can_init_from_no_args() const {
    return InitCheckHelper<&RecordLayout::has_init_from_no_args>(this);
}

bool TypeBase::can_zero_init() const {
    return InitCheckHelper<&RecordLayout::has_zero_init>(this);
}

auto TypeBase::desugar_once() const -> Type {
    return this;
}

void TypeBase::dump(bool use_colour) const {
    std::println("{}", text::RenderColours(use_colour, print().str()));
}

auto TypeBase::elem() const -> Type {
    return cast<SingleElementTypeBase>(this)->elem();
}

auto TypeBase::eval_mode() const -> EvalMode {
    switch (canonical_kind()) {
        using K = TypeBase::Kind;
        case K::BuiltinType:
        case K::EnumType:
        case K::IntType:
        case K::ProcType:
        case K::PtrType:
        case K::SliceType:
            return EvalMode::Scalar;

#       define AST_TYPE(x)
#       define AST_TYPE_SUGAR(x) case K::x:
#       include "srcc/AST.inc"
            Unreachable("Canonical type should not be type sugar");

        case K::OpaqueType:
            Unreachable("Querying property of opaque type");

        case K::OptionalType: {
            auto opt = cast<OptionalType>(this);
            if (opt->has_transparent_layout()) return opt->elem()->eval_mode();
            return EvalMode::Memory; // This is an aggregate.
        }

        // TODO: Ranges are weird in that both eval modes make sense: memory
        // for calls and scalar for casts and for how they’re created; maybe
        // this warrants a separate eval mode (like Clang’s complex eval mode)?
        case K::RangeType:
            return EvalMode::Scalar;

        case K::ArrayType:
        case K::StructType:
        case K::TupleType:
            return EvalMode::Memory;
    }

    Unreachable();
}

bool TypeBase::has_copy_proc() const {
    Type t = canonical->strip_arrays_and_optionals();
    auto r = dyn_cast<RecordType>(t);
    return r and r->get_copy().present();
}

bool TypeBase::is_aggregate() const {
    if (auto opt = dyn_cast<OptionalType>(canonical)) {
        if (not opt->has_transparent_layout()) return true;
        return opt->elem()->is_aggregate();
    }

    return isa<RecordType, ArrayType, SliceType, RangeType, ProcType>(canonical);
}

bool TypeBase::is_canonical() const {
    // static_cast to void is required because we disallow pointer
    // comparisons involving 'TypeBase*'.
    return static_cast<void*>(canonical) == static_cast<const void*>(this);
}

bool TypeBase::is_complete() const {
    if (isa<OpaqueType>(canonical)) return false;
    if (auto s = dyn_cast<RecordType>(canonical)) return s->is_complete();
    return true;
}

bool TypeBase::is_integer() const {
    return canonical == Type::IntTy or isa<IntType>(canonical);
}

bool TypeBase::is_integer_or_bool() const {
    return is_integer() or canonical == Type::BoolTy;
}

bool TypeBase::is_or_contains_pointer() const {
    return visit(utils::Overloaded{
        [&](const EnumType*) { return false; },
        [&](const IntType*) { return false; },
        [&](const ProcType*) { return true; },
        [&](const PtrType*) { return true; },
        [&](const RangeType*) { return false; },
        [&](const SliceType*) { return true; },
        [&](const ArrayType* ty) { return ty->elem()->is_or_contains_pointer(); },
        [&](const RecordType* ty) {
            Assert(ty->is_complete(), "Querying property of incomplete type");
            return ty->layout().bits().contains_pointer;
        },
        [&](const OpaqueType* ty) -> bool {
            Unreachable("Querying property of opaque type");
        },
        [&](const OptionalType* ty) {
            if (ty->has_transparent_layout()) return ty->elem()->is_or_contains_pointer();
            return ty->get_equivalent_record_layout()->bits().contains_pointer;
        },
        [&](const BuiltinType* ty) {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::Void:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Nil:
                return false;

                case BuiltinKind::Tree:
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
                return true;

                case BuiltinKind::Deduced:
                case BuiltinKind::CallArgList:
                    Unreachable("Querying property of deduced type");
            }
            Unreachable();
        },
    });
}

bool TypeBase::is_sugar() const {
    return isa<AliasType>(this);
}

bool TypeBase::is_void() const {
    return isa<BuiltinType>(canonical) and
           cast<BuiltinType>(canonical)->builtin_kind() == BuiltinKind::Void;
}

bool TypeBase::move_is_copy() const {
    // This will have to change once we have destructors.
    return not requires_deletion();
}

auto TypeBase::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out = print_impl();
    if (is_sugar()) {
        // FIXME: This ends up printing e.g. "foo (aka int)", which means
        // that if we put the type in single quotes, we get "'foo (aka int)'",
        // when really we want "'foo' (aka 'int')".
        //
        // This means the printing of 'aka' should be moved into the diagnostics
        // rendering, but that’s currently just plain format strings, so we need
        // to do something about that.
        out += " %0((aka%) ";
        out += canonical->print_impl();
        out += "%0()%)";
    }
    return out;
}

auto TypeBase::print_impl() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    auto MaybeParenthesise = [](Type ty) {
        if (not isa<ProcType>(ty)) return ty->print_impl();
        SmallUnrenderedString s;
        s += "%1((";
        s += ty->print_impl();
        s += ")%)";
        return s;
    };

    // Append the base type.
    visit_exact(utils::Overloaded{ // clang-format off
        [&](AliasType* alias) { Format(out, "%3({}%)", alias->decl()->name); },
        [&](ArrayType* arr) {
            Format(
                out,
                "{}%1([%5({}%)]%)",
                MaybeParenthesise(arr->elem()),
                arr->dimension()
            );
        },
        [&](BuiltinType* b) {
            switch (b->builtin_kind()) {
                case BuiltinKind::Bool: out += "%6(bool%)"; return;
                case BuiltinKind::Deduced: out += "%6(var%)"; return;
                case BuiltinKind::UnresolvedOverloadSet: out += "%6(<overload set>%)"; return;
                case BuiltinKind::Int: out += "%6(int%)"; return;
                case BuiltinKind::NoReturn: out += "%6(noreturn%)"; return;
                case BuiltinKind::Tree: out += "%6(tree%)"; return;
                case BuiltinKind::Type: out += "%6(type%)"; return;
                case BuiltinKind::Void: out += "%6(void%)"; return;
                case BuiltinKind::Nil: out += "%6(nil%)"; return;
                case BuiltinKind::CallArgList: out += "<call arg list>"; return;
            }
            Unreachable();
        },
        [&](EnumType* e) { Format(out, "%6({}%)", e->decl()->name); },
        [&](IntType* int_ty) { Format(out, "%6(i{:i}%)", int_ty->bit_width()); },
        [&](OpaqueType* ty) { Format(out, "%3({}%)", ty->name()); },
        [&](OptionalType* opt) { Format(out, "{}%1(?%)", MaybeParenthesise(opt->elem())); },
        [&](ProcType* proc) { out += proc->print(String()); },
        [&](PtrType* ptr) {
            Format(
                out,
                "{}%1({}^%)",
                MaybeParenthesise(ptr->elem()),
                ptr->is_immutable() ? " val" : ""
            );
        },
        [&](RangeType* range) { Format(out, "%6(range%)%1(<%){}%1(>%)", range->elem()->print_impl()); },
        [&](SliceType* slice) {
            Format(
                out,
                "{}%1({}[]%)",
                MaybeParenthesise(slice->elem()),
                slice->is_immutable() ? " val" : ""
            );
        },
        [&](StructType* s) { Format(out, "%6({}%)", s->name()); },
        [&](TupleType* s) {
            out += "%1((";
            bool first = true;
            for (auto decl : s->scope()->fields()) {
                if (first) first = false;
                else out += ", ";
                if (not decl->name.empty()) Format(out, "%5({}%): ", decl->name);
                out += decl->type->print_impl();
            }
            out += ")%)";
        },
    }); // clang-format on

    return out;
}

auto TypeBase::memory_size(TranslationUnit& tu) const -> Size {
    return memory_size(tu.target());
}

auto TypeBase::memory_size(const Target& t) const -> Size {
    return size_impl(t).as_bytes();
}

bool TypeBase::requires_deletion() const {
    Type t = canonical->strip_arrays_and_optionals();
    auto r = dyn_cast<RecordType>(t);
    return r and r->get_delete().present();
}

auto TypeBase::size_impl(const Target& t) const -> Size {
    return visit(utils::Overloaded{
        [&](BuiltinType* b) {
            switch (b->builtin_kind()) {
                case BuiltinKind::Bool: return Size::Bits(1);
                case BuiltinKind::Int: return t.int_size();
                case BuiltinKind::Tree: return Size::Of<TreeValue*>();
                case BuiltinKind::Type: return Size::Of<TypeBase*>();
                case BuiltinKind::UnresolvedOverloadSet: return t.closure_size();

                case BuiltinKind::NoReturn:
                case BuiltinKind::Void:
                case BuiltinKind::Nil:
                    return Size();

                case BuiltinKind::Deduced:
                case BuiltinKind::CallArgList:
                    Unreachable("Requested size of deduced type");
            }
        },
        [&](EnumType* e) { return e->elem()->size_impl(t); },
        [&](IntType* i) { return t.int_size(i); },
        [&](PtrType*) { return t.ptr_size(); },
        [&](ProcType*) { return t.closure_size(); },
        [&](SliceType*) { return t.slice_size(); },
        [&](RangeType* r) {
            auto elem = r->elem();
            return elem->array_size(t) + elem->memory_size(t);
        },

        [&](OpaqueType* ) -> Size { Unreachable("Querying size of opaque type"); },
        [&](OptionalType* opt) {
            if (opt->has_transparent_layout()) return opt->elem()->size_impl(t);
            return opt->get_equivalent_record_layout()->size();
        },

        [&](RecordType* s) { return s->layout().size(); },
        [&](ArrayType* arr) { return arr->elem()->array_size(t) * u64(arr->dimension()); },
    });
}

template <typename ...Types>
auto TypeBase::strip_qualifiers() const -> Type {
    static_assert((std::derived_from<Types, SingleElementTypeBase> and ...));
    for (Type ty = this;;) {
        // Strip qualifiers from the sugared type.
        while (isa<Types...>(ty)) ty = cast<SingleElementTypeBase>(ty)->elem();

        // If the canonical type no longer satisfies the predicate, stop.
        if (not isa<Types...>(ty->canonical)) return ty;

        // Otherwise, desugar until the predicate matches again.
        do ty = ty->desugar_once();
        while (not isa<Types...>(ty));
    }
}

auto TypeBase::strip_arrays() const -> Type {
    return strip_qualifiers<ArrayType>();
}

auto TypeBase::strip_arrays_and_optionals() const -> Type {
    return strip_qualifiers<ArrayType, OptionalType>();
}

auto TypeBase::strip_pointers_and_optionals() const -> Type {
    return strip_qualifiers<PtrType, OptionalType>();
}

// ============================================================================
//  Types
// ============================================================================
/// Create a new type decl that declares a type alias.
auto AliasType::Create(
    TranslationUnit& tu,
    Type type,
    DeclName name,
    SLoc decl_loc
) -> AliasType* {
    auto td = new (tu) TypeDecl(type, name, decl_loc);
    auto alias = new (tu) AliasType(td, type);
    td->type = alias;
    return alias;
}

void AliasType::set_aliased_type(Type ty) {
    element_type = ty;
    canonical = ty->canonical;
}

auto ArrayType::Get(TranslationUnit& mod, Type elem, i64 size) -> ArrayType* {
    auto CreateNew = [&] {
        auto arr = new (mod) ArrayType{elem, size};
        if (elem->is_canonical()) return arr;
        arr->canonical = new (mod) ArrayType{elem->canonical, size};
        return arr;
    };

    return GetOrCreateType(mod.array_types, CreateNew, elem, size);
}

void ArrayType::Profile(FoldingSetNodeID& ID, Type elem, i64 size) {
    ID.AddPointer(elem.ptr());
    ID.AddInteger(size);
}

EnumType::EnumType(
    TranslationUnit& tu,
    Scope* scope,
    DeclName name,
    Type underlying_type,
    SLoc loc
) : SingleElementTypeBase{Kind::EnumType, underlying_type}, enum_scope{scope} {
    type_decl = new (tu) TypeDecl(this, name, loc);
}

auto EnumType::name() const -> DeclName { return decl()->name; }

auto IntType::Get(TranslationUnit& mod, Size bits) -> IntType* {
    auto CreateNew = [&] { return new (mod) IntType{bits}; };
    return GetOrCreateType(mod.int_types, CreateNew, bits);
}

void IntType::Profile(FoldingSetNodeID& ID, Size bits) {
    ID.AddInteger(bits.bits());
}

auto OpaqueType::Create(TranslationUnit& tu, String name, SLoc decl_loc) -> OpaqueType* {
    auto type = new (tu) OpaqueType();
    type->type_decl = new (tu) TypeDecl{type, name, decl_loc};
    return type;
}

auto OpaqueType::name() const -> String {
    return type_decl->name.str();
}

auto OptionalType::Get(TranslationUnit& mod, Type elem) -> OptionalType* {
    auto CreateNew = [&] {
        auto Make = [&](Type elem) {
            if (isa<PtrType, ProcType>(elem)) return new (mod) OptionalType{elem};
            RecordLayout::Builder b{mod};
            b.add_field(elem);
            b.add_field(Type::BoolTy);

            // Ignore the properties of the record layout here; all code paths
            // that care about those know to look through optionals.
            auto [layout, _] = b.build();
            return new (mod) OptionalType{elem, layout};
        };

        OptionalType* ty = Make(elem);
        if (elem->is_canonical()) return ty;
        ty->canonical = Make(elem->canonical);
        return ty;
    };

    return GetOrCreateType(mod.optional_types, CreateNew, elem);
}

void OptionalType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.ptr());
}

auto OptionalType::get_engaged_offset() const -> Size {
    Assert(not has_transparent_layout());
    return layout->field_offsets()[1];
}

auto PtrType::Get(TranslationUnit& mod, Type elem, bool immutable) -> PtrType* {
    auto CreateNew = [&] {
        auto ty = new (mod) PtrType{elem, immutable};
        if (elem->is_canonical()) return ty;
        ty->canonical = new (mod) PtrType{elem->canonical, immutable};
        return ty;
    };

    return GetOrCreateType(mod.ptr_types, CreateNew, elem, immutable);
}

void PtrType::Profile(FoldingSetNodeID& ID, Type elem, bool immutable) {
    ID.AddPointer(elem.ptr());
    ID.AddBoolean(immutable);
}

auto ProcType::AdjustRet(TranslationUnit& mod, ProcType* ty, Type new_ret) -> ProcType* {
    if (ty->ret() == new_ret) return ty;
    return Get(
        mod,
        {new_ret, ty->return_value_category()},
        ty->params(),
        ty->cconv(),
        ty->has_c_varargs()
    );
}

auto ProcType::Get(
    TranslationUnit& mod,
    TypeAndValueCategory return_type,
    ArrayRef<ParamTypeData> param_types,
    CallingConvention cconv,
    bool c_varargs
) -> ProcType* {
    auto CreateNew = [&] {
        auto Make = [&](TypeAndValueCategory return_type, ArrayRef<ParamTypeData> param_types) {
            const auto size = totalSizeToAlloc<ParamTypeData>(param_types.size());
            auto mem = mod.allocate(size, alignof(ProcType));
            return ::new (mem) ProcType{
                cconv,
                c_varargs,
                return_type,
                param_types
            };
        };

        auto ty = Make(return_type, param_types);
        if (
            return_type.type()->is_canonical() and
            all_of(param_types, [](auto& p) { return p.type->is_canonical(); })
        ) return ty;

        // Build canonical type.
        TypeAndValueCategory canonical_ret{return_type.type()->canonical, return_type.value_category()};
        auto canonical_params = to_vector(vws::transform(
            param_types,
            [](auto& p) { return ParamTypeData{p.intent, p.type->canonical, p.variadic}; }
        ));

        ty->canonical = Make(canonical_ret, canonical_params);
        return ty;
    };

    return GetOrCreateType(
        mod.proc_types,
        CreateNew,
        return_type,
        param_types,
        cconv,
        c_varargs
    );
}

ProcType::ProcType(
    CallingConvention cconv,
    bool variadic,
    TypeAndValueCategory return_type,
    ArrayRef<ParamTypeData> param_types
) : TypeBase{Kind::ProcType},
    cc{cconv},
    is_varargs{variadic},
    num_params{utils::safe_cast<u32>(param_types.size())},
    return_type{return_type} {
    std::uninitialized_copy_n(
        param_types.begin(),
        param_types.size(),
        getTrailingObjects()
    );
}

void ProcType::Profile(
    FoldingSetNodeID& ID,
    TypeAndValueCategory return_type,
    ArrayRef<ParamTypeData> param_types,
    CallingConvention cc,
    bool is_varargs
) {
    ID.AddInteger(+cc);
    ID.AddBoolean(is_varargs);
    ID.AddPointer(return_type.type().ptr());
    ID.AddInteger(+return_type.value_category());
    ID.AddInteger(param_types.size());
    for (const auto& t : param_types) {
        ID.AddInteger(+t.intent);
        ID.AddPointer(t.type.ptr());
        ID.AddBoolean(t.variadic);
    }
}

auto ProcType::print(
    DeclName proc_name,
    ProcDecl* decl,
    bool include_proc_keyword
) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    out += "%1(";
    if (include_proc_keyword) out += "proc";

    // Add name.
    if (not proc_name.empty())
        Format(out, " %2({}%)", proc_name);

    // Add params.
    const auto& ps = params();
    if (not ps.empty()) {
        out += " (";
        bool first = true;
        for (const auto& [i, p] : enumerate(ps)) {
            if (first) first = false;
            else out += ", ";
            if (p.intent != Intent::Move) Format(out, "{} ", p.intent);
            out += p.type->print_impl();
        }
        out += ")";
    }

    // Add attributes.
    if (has_c_varargs()) out += " varargs";
    if (decl) {
        if (cconv() != CallingConvention::Native and decl->mangling == Mangling::None)
            out += " nomangle";
    }

    // Add return type and value category.
    if (not ret()->is_void()) Format(out, " -> {}", ret()->print_impl());
    if (return_type.value_category() == Expr::ILValue) out += " val ref";
    else if (return_type.value_category() == Expr::MLValue) out += " ref";

    out += "%)";
    return out;
}

auto RangeType::Get(TranslationUnit& mod, Type elem) -> RangeType* {
    auto CreateNew = [&] {
        auto tuple = TupleType::GetTrivial(mod, {elem, elem});
        auto ty = new (mod) RangeType{elem, tuple};
        if (elem->is_canonical()) return ty;
        ty->canonical = new (mod) RangeType{elem->canonical, cast<TupleType>(tuple->canonical)};
        return ty;
    };

    return GetOrCreateType(mod.range_types, CreateNew, elem);
}

void RangeType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.ptr());
}

auto SliceType::Get(TranslationUnit& tu, Type elem, bool immutable) -> SliceType* {
    auto CreateNew = [&] {
        auto ptr = PtrType::Get(tu, elem, immutable);
        auto ty = new (tu) SliceType{elem, ptr, immutable};
        if (elem->is_canonical()) return ty;
        ty->canonical = new (tu) SliceType{elem->canonical, ptr->canonical, immutable};
        return ty;
    };

    return GetOrCreateType(tu.slice_types, CreateNew, elem, immutable);
}

void SliceType::Profile(FoldingSetNodeID& ID, Type elem, bool immutable) {
    ID.AddPointer(elem.ptr());
    ID.AddBoolean(immutable);
}

// ============================================================================
//  Record Types
// ============================================================================
void RecordLayout::Builder::add_field(Type ty) {
    // The alignment of a struct is the alignment of its most-aligned field.
    auto fa = ty->align(tu);
    a = std::max(a, fa);

    // Compute the field offset; note that all union fields have offset 0.
    // TODO: Maybe optimise layout if this isn’t meant for FFI.
    if (not bits.is_union) {
        auto offset = sz.align(fa);
        sz = offset + ty->memory_size(tu);
        offsets.push_back(offset);
    } else {
        offsets.push_back(Size());
    }

    // Update bits.
    bits.default_initialiser = bits.init_from_no_args &= ty->can_init_from_no_args();
    bits.zero_init &= bits.default_initialiser and ty->can_zero_init();
    bits.contains_pointer |= ty->is_or_contains_pointer();
    bits.can_copy &= ty->can_copy();

    // Update props.
    props.must_define_delete |= ty->requires_deletion();
    props.must_define_copy = bits.can_copy and (props.must_define_copy or ty->has_copy_proc());
}

// TODO: Initialisers are declared out-of-line, but they should
// have been picked up during initial translation when we find
// all the procedures in the current scope. Add any that we found
// here, and if we didn’t find any (or the 'default' attribute was
// specified on the struct declaration), declare the default initialiser.

// TODO: If we decide to allow this:
//
// struct S { ... }
// proc f {
//    init S(...) { ... }
// }
//
// Then the initialiser should still respect normal lookup rules (i.e.
// it should only be visible within 'f'). Perhaps we want to store
// local member functions outside the struct itself and in the local
// scope instead?
auto RecordLayout::Builder::build() -> std::pair<RecordLayout*, RecordLayout::Props> {
    return {RecordLayout::Get(tu, offsets, sz, sz.align(a), a, bits), props};
}

RecordLayout::RecordLayout(
    ArrayRef<Size> offsets,
    Size sz,
    Size arr_sz,
    Align a,
    Bits bits
) : num_fields{utils::safe_cast<u32>(offsets.size())},
    computed_alignment{a},
    computed_bits{bits},
    computed_size{sz},
    computed_array_size{arr_sz} {
    std::uninitialized_copy_n(
        offsets.begin(),
        offsets.size(),
        getTrailingObjects()
    );
}

auto RecordLayout::Get(
    TranslationUnit &tu,
    ArrayRef<Size> offsets,
    Size sz,
    Size arr_sz,
    Align a,
    Bits bits
) -> RecordLayout* {
    // Check if this is an existing layout.
    FoldingSetNodeID id;
    llvm::FoldingSetInsertToken tok;
    Profile(id, offsets, sz, arr_sz, a, bits);
    if (auto rl = tu.record_layouts.lookup(id, tok)) return rl;

    // If not, create it.
    auto alloc_sz = totalSizeToAlloc<Size>(offsets.size());
    auto mem = tu.allocate(alloc_sz, alignof(RecordLayout));
    auto rl = ::new (mem) RecordLayout(offsets, sz, arr_sz, a, bits);
    tu.record_layouts.insert(rl, tok);
    return rl;
}

void RecordLayout::Profile(FoldingSetNodeID& id) const {
    Profile(id, field_offsets(), size(), array_size(), align(), bits());
}

void RecordLayout::Profile(
    FoldingSetNodeID& id,
    ArrayRef<Size> offsets,
    Size sz,
    Size arr_sz,
    Align a,
    Bits bits
) {
    for (auto o : offsets) id.AddInteger(o.bits());
    id.AddInteger(sz.bits());
    id.AddInteger(arr_sz.bits());
    id.AddInteger(a.log_repr());
    id.AddInteger(bits.encode());
}

auto RecordType::get_special_member(Tk tok) const -> Ptr<ProcDecl> {
    auto d = record_scope->decls_with_name(tok);
    if (d.empty()) return nullptr;
    Assert(utils::IsSingle(d));
    return cast<ProcDecl>(d.front());
}

void RecordType::set_special_member(Tk tok, ProcDecl* decl) {
    Assert(decl);
    Assert(record_scope->decls_with_name(tok).empty());
    record_scope->add(decl);
}

auto StructType::Create(
    TranslationUnit& owner,
    RecordScope* scope,
    String name,
    SLoc decl_loc,
    RecordLayout* layout
) -> StructType* {
    auto type = new (owner) StructType(owner, scope);
    type->type_decl = new (owner) TypeDecl{type, name, decl_loc};
    type->record_layout = layout;
    return type;
}

void StructType::finalise(RecordLayout* layout) {
    Assert(not is_complete(), "Struct already finalised");
    record_layout = layout;
}

auto StructType::name() const -> String {
    return type_decl->name.str();
}

TupleType::TupleType(
    RecordLayout* layout,
    RecordScope* scope
) : RecordType(Kind::TupleType, scope)  {
    record_layout = layout;
}

auto TupleType::CreateDeserialised(
    TranslationUnit& tu,
    RecordScope* scope,
    RecordLayout* layout
) -> TupleType* {
    // Fields will be filled in by the deserialiser.
    return new (tu) TupleType(layout, scope);
}

auto TupleType::GetTrivial(TranslationUnit& tu, ArrayRef<Type> elems) -> TupleType* {
    RecordLayout::Builder lb{tu};
    for (auto ty : elems) lb.add_field(ty);
    auto [layout, props] = lb.build();
    Assert(props.trivial(), "Non-trivial tuples must be build by Sema");
    return Get(
        tu,
        tu.global_scope(),
        layout,
        elems,
        llvm::to_vector(vws::repeat(String(), elems.size())),
        [](auto) { /* Nothing to do here. */ }
    );
}

auto TupleType::Get(
    TranslationUnit& tu,
    Scope* parent_scope,
    RecordLayout* layout,
    ArrayRef<Type> fields,
    ArrayRef<String> names,
    llvm::function_ref<void(TupleType* new_type)> DefineSpecialProcedures
) -> TupleType* {
    Assert(layout);
    auto GetCanonicalType = [](auto ty) { return Type(ty->canonical); };
    auto canonical_types = llvm::to_vector(vws::transform(fields, GetCanonicalType));

    // Profile the element types recursively.
    FoldingSetNodeID types_profile;
    for (auto t : canonical_types) {
        if (auto tuple = dyn_cast<TupleType>(t)) types_profile.AddNodeID(tuple->types_profile);
        else types_profile.AddPointer(t.ptr());
    }

    // Allocate the tuple.
    auto CreateTuple = [&](ArrayRef<Type> types) {
        auto s = tu.create_scope<RecordScope>(parent_scope);
        auto tuple = new (tu) TupleType{layout, s};
        u32 index = 0;
        for (auto [ty, name] : llvm::zip_equal(types, names))
            s->add(new (tu) FieldDecl(tuple, ty, name, index++, SLoc()));
        return tuple;
    };

    // Create the canonical tuple.
    //
    // This additionally defines the special functions. We only do this once
    // since we would otherwise generate the same code twice.
    auto CreateCanonical = [&] {
        auto tuple = CreateTuple(canonical_types);
        tuple->types_profile = types_profile.Intern(tu.allocator());
        if (DefineSpecialProcedures) std::invoke(DefineSpecialProcedures, tuple);
        return tuple;
    };

    auto canonical = GetOrCreateType(
        tu.tuple_types,
        CreateCanonical,
        types_profile,
        names
    );

    // Create the sugared tuple.
    //
    // Reuse the special members for the canonical tuple here.
    auto CreateSugared = [&] {
        auto tuple = CreateTuple(fields);
        tuple->canonical = canonical;
        tuple->types_profile = canonical->types_profile;
        for (auto t : {Tk::Copy, Tk::Delete}) {
            auto proc = canonical->get_special_member(t);
            if (proc.present()) tuple->set_special_member(t, proc.get());
        }
        return tuple;
    };

    // Create the non-canonical tuple.
    return GetOrCreateType(
        tu.tuple_types,
        CreateSugared,
        types_profile,
        names
    );
}

void TupleType::Profile(FoldingSetNodeID& id, const FoldingSetNodeID& types, auto names) {
    id.AddNodeID(types);
    for (auto n : names) id.AddString(n);
}

void TupleType::Profile(FoldingSetNodeID& id) const {
    auto Names = [](FieldDecl* f) { return f->name.str(); };
    Profile(id, types_profile, vws::transform(record_scope->fields(), Names));
}
