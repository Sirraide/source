#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>

#include <clang/Basic/TargetInfo.h>

#include <llvm/Support/MathExtras.h>

#include <memory>
#include <print>

using namespace srcc;

struct srcc::BuiltinTypes {
    static constexpr BuiltinType VoidTyImpl{BuiltinKind::Void};
    static constexpr BuiltinType NoReturnTyImpl{BuiltinKind::NoReturn};
    static constexpr BuiltinType BoolTyImpl{BuiltinKind::Bool};
    static constexpr BuiltinType IntTyImpl{BuiltinKind::Int};
    static constexpr BuiltinType DeducedTyImpl{BuiltinKind::Deduced};
    static constexpr BuiltinType TypeTyImpl{BuiltinKind::Type};
    static constexpr BuiltinType UnresolvedOverloadSetTyImpl{BuiltinKind::UnresolvedOverloadSet};
};

constexpr Type Type::VoidTy{const_cast<BuiltinType*>(&BuiltinTypes::VoidTyImpl)};
constexpr Type Type::NoReturnTy{const_cast<BuiltinType*>(&BuiltinTypes::NoReturnTyImpl)};
constexpr Type Type::BoolTy{const_cast<BuiltinType*>(&BuiltinTypes::BoolTyImpl)};
constexpr Type Type::IntTy{const_cast<BuiltinType*>(&BuiltinTypes::IntTyImpl)};
constexpr Type Type::DeducedTy{const_cast<BuiltinType*>(&BuiltinTypes::DeducedTyImpl)};
constexpr Type Type::TypeTy{const_cast<BuiltinType*>(&BuiltinTypes::TypeTyImpl)};
constexpr Type Type::UnresolvedOverloadSetTy{const_cast<BuiltinType*>(&BuiltinTypes::UnresolvedOverloadSetTyImpl)};

// ============================================================================
//  Helpers
// ============================================================================
template <typename T, typename... Args>
auto GetOrCreateType(FoldingSet<T>& Set, auto CreateNew, Args&&... args) -> T* {
    FoldingSetNodeID ID;
    T::Profile(ID, std::forward<Args>(args)...);

    void* pos = nullptr;
    auto* type = Set.FindNodeOrInsertPos(ID, pos);
    if (not type) {
        type = CreateNew();
        Set.InsertNode(type, pos);
    }

    return type;
}

TypeLoc::TypeLoc(Expr* e) : ty{e->type}, loc{e->location()} {}

// ============================================================================
//  Scope
// ============================================================================
Scope::~Scope() = default;
Scope::Scope(Scope* parent, ScopeKind k)
    : parent_scope{parent},
      kind{k} {}

// ============================================================================
//  Type
// ============================================================================
void* TypeBase::operator new(usz size, TranslationUnit& mod) {
    return mod.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

// TODO: Cache type size and alignment in the TU.
auto TypeBase::align(TranslationUnit& tu) const -> Align { // clang-format off
    return visit(utils::Overloaded{
        [&](const ArrayType* ty) -> Align { return ty->elem()->align(tu); },
        [&](const BuiltinType* ty) -> Align {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool: return Align{1};
                case BuiltinKind::Int: return tu.target().int_align();
                case BuiltinKind::NoReturn: return Align{1};
                case BuiltinKind::Void: return Align{1};
                case BuiltinKind::Type: return Align::Of<Type>(); // This is a compile-time only type.
                case BuiltinKind::UnresolvedOverloadSet: return tu.target().closure_align();;
                case BuiltinKind::Deduced: Unreachable("Requested alignment of deduced type");
            }
            Unreachable();
        },
        [&](const IntType* ty) { return tu.target().int_align(ty); },
        [&](const ProcType*) { return tu.target().closure_align(); },
        [&](const PtrType*) { return tu.target().ptr_align(); },
        [&](const SliceType*) { return tu.target().slice_align(); },
        [&](const StructType* ty) {
            Assert(ty->is_complete(), "Requested size of incomplete struct");
            return ty->align();
        },
    });
} // clang-format on

auto TypeBase::array_size(TranslationUnit& tu) const -> Size {
    if (auto s = dyn_cast<StructType>(this)) return s->array_size();
    return size(tu);
}

template <bool (StructType::*struct_predicate)() const>
bool InitCheckHelper(const TypeBase* type) { // clang-format off
    return type->visit(utils::Overloaded{
        [&](const ArrayType* ty) { return InitCheckHelper<struct_predicate>(ty->elem().ptr()); },
        [&](const BuiltinType* ty) {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::Void:
                    return true;

                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Type:
                    return false;

                case BuiltinKind::Deduced:
                    Unreachable("Querying property of deduced type");
            }
            Unreachable();
        },
        [&](const IntType*) { return true; },
        [&](const ProcType*) { return false; },
        [&](const PtrType*) { return false; },
        [&](const SliceType*) { return true; },
        [&](const StructType* ty) { return (ty->*struct_predicate)(); },
    });
} // clang-format on

bool TypeBase::can_default_init() const {
    return InitCheckHelper<&StructType::has_default_init>(this);
}

bool TypeBase::can_init_from_no_args() const {
    return InitCheckHelper<&StructType::has_init_from_no_args>(this);
}

void TypeBase::dump(bool use_colour) const {
    std::println("{}", text::RenderColours(use_colour, print().str()));
}

bool TypeBase::is_integer() const {
    return this == Type::IntTy or isa<IntType>(this);
}

bool TypeBase::is_srvalue() const {
    switch (type_kind) {
        case Kind::BuiltinType:
        case Kind::SliceType:
        case Kind::PtrType:
        case Kind::IntType:
        case Kind::ProcType:
            return true;

        case Kind::ArrayType:
        case Kind::StructType:
            return false;
    }

    Unreachable("Invalid type kind");
}

bool TypeBase::is_void() const {
    return kind() == Kind::BuiltinType and
           cast<BuiltinType>(this)->builtin_kind() == BuiltinKind::Void;
}

bool TypeBase::move_is_copy() const {
    // This will have to change once we have destructors.
    return true;
}

bool TypeBase::pass_by_rvalue(CallingConvention cc, Intent intent) const {
    // Always pass parameters to C or C++ functions by value.
    if (cc == CallingConvention::Native) return true;
    Assert(cc == CallingConvention::Source, "Unsupported calling convention");
    switch (intent) {
        // These allow modifying the original value, which means that
        // we always have to pass by reference here.
        case Intent::Inout:
        case Intent::Out:
            return false;

        // Always pass by value if we’re making a copy.
        case Intent::Copy:
            return true;

        // If we only want to inspect the value, pass by value if small,
        // and by reference otherwise.
        //
        // On the caller side, moving is treated the same as 'in', the
        // only difference is that the latter creates a variable in the
        // callee, whereas the former doesn’t.
        //
        // The intent behind the latter is that e.g. 'moving' a large
        // struct should not require a memcpy unless the callee actually
        // moves it somewhere else; otherwise, it doesn't matter where
        // it is stored, and we save a memcpy that way.
        case Intent::In:
        case Intent::Move:
            return visit(utils::Overloaded{
                // clang-format off
                [](ArrayType*) { return false; },                        // Arrays are usually big, so pass by reference.
                [](BuiltinType*) { return true; },                       // All builtin types are small.
                [](IntType* t) { return t->bit_width().bits() <= 128; }, // Only pass small ints by value.
                [](ProcType*) { return true; },                          // Closures are two pointers.
                [](PtrType*) { return true; },                           // Pointers are small.
                [](SliceType*) { return true; },                         // Slices are two pointers.
                [](StructType*) { return false; },                       // Pass structs by reference (TODO: small ones by value).
            }); // clang-format on
    }

    Unreachable();
}

auto TypeBase::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (kind()) {
        case Kind::ArrayType: {
            auto* arr = cast<ArrayType>(this);
            out += std::format("{}%1([%5({}%)]%)", arr->elem()->print(), arr->dimension());
        } break;

        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: out += "%6(bool%)"; break;
                case BuiltinKind::Deduced: out += "%6(var%)"; break;
                case BuiltinKind::UnresolvedOverloadSet: out += "%6(<overload set>%)"; break;
                case BuiltinKind::Int: out += "%6(int%)"; break;
                case BuiltinKind::NoReturn: out += "%6(noreturn%)"; break;
                case BuiltinKind::Type: out += "%6(type%)"; break;
                case BuiltinKind::Void: out += "%6(void%)"; break;
            }
        } break;

        case Kind::IntType: {
            auto* int_ty = cast<IntType>(this);
            out += std::format("%6(i{:i}%)", int_ty->bit_width());
        } break;

        case Kind::ProcType: {
            auto proc = cast<ProcType>(this);
            out += proc->print("");
        } break;

        case Kind::PtrType: {
            auto* ref = cast<PtrType>(this);
            out += std::format("{}%1(^%)", ref->elem()->print());
        } break;

        case Kind::SliceType: {
            auto* slice = cast<SliceType>(this);
            out += std::format("{}%1([]%)", slice->elem()->print());
        } break;

        case Kind::StructType: {
            auto* s = cast<StructType>(this);
            out += std::format("%6({}%)", s->name());
        } break;
    }

    return out;
}

auto TypeBase::size(TranslationUnit& tu) const -> Size {
    switch (type_kind) {
        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return Size::Bits(1);
                case BuiltinKind::Int: return tu.target().int_size();
                case BuiltinKind::Type: return Size::Of<Type>();
                case BuiltinKind::UnresolvedOverloadSet: return tu.target().closure_size();

                case BuiltinKind::NoReturn:
                case BuiltinKind::Void:
                    return Size();

                case BuiltinKind::Deduced:
                    Unreachable("Requested size of deduced type");
            }
        }

        case Kind::PtrType: return tu.target().ptr_size();
        case Kind::IntType: return cast<IntType>(this)->bit_width();
        case Kind::ProcType: return tu.target().closure_size();
        case Kind::SliceType: return tu.target().slice_size();
        case Kind::StructType: {
            auto s = cast<StructType>(this);
            return s->size();
        }

        case Kind::ArrayType: {
            auto arr = cast<ArrayType>(this);
            return arr->elem()->array_size(tu) * u64(arr->dimension());
        }
    }

    Unreachable("Invalid type kind");
}

// ============================================================================
//  Types
// ============================================================================
auto ArrayType::Get(TranslationUnit& mod, Type elem, i64 size) -> ArrayType* {
    auto CreateNew = [&] { return new (mod) ArrayType{elem, size}; };
    return GetOrCreateType(mod.array_types, CreateNew, elem, size);
}

void ArrayType::Profile(FoldingSetNodeID& ID, Type elem, i64 size) {
    ID.AddPointer(elem.ptr());
    ID.AddInteger(size);
}

auto IntType::Get(TranslationUnit& mod, Size bits) -> IntType* {
    auto CreateNew = [&] { return new (mod) IntType{bits}; };
    return GetOrCreateType(mod.int_types, CreateNew, bits);
}

void IntType::Profile(FoldingSetNodeID& ID, Size bits) {
    ID.AddInteger(bits.bits());
}

auto PtrType::Get(TranslationUnit& mod, Type elem) -> PtrType* {
    auto CreateNew = [&] { return new (mod) PtrType{elem}; };
    return GetOrCreateType(mod.ptr_types, CreateNew, elem);
}

void PtrType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.ptr());
}

auto ProcType::AdjustRet(TranslationUnit& mod, ProcType* ty, Type new_ret) -> ProcType* {
    return Get(
        mod,
        new_ret,
        ty->params(),
        ty->cconv(),
        ty->variadic()
    );
}

auto ProcType::Get(
    TranslationUnit& mod,
    Type return_type,
    ArrayRef<ParamTypeData> param_types,
    CallingConvention cconv,
    bool variadic
) -> ProcType* {
    auto CreateNew = [&] {
        const auto size = totalSizeToAlloc<ParamTypeData>(param_types.size());
        auto mem = mod.allocate(size, alignof(ProcType));
        return ::new (mem) ProcType{
            cconv,
            variadic,
            return_type,
            param_types
        };
    };

    return GetOrCreateType(
        mod.proc_types,
        CreateNew,
        return_type,
        param_types,
        cconv,
        variadic
    );
}

ProcType::ProcType(
    CallingConvention cconv,
    bool variadic,
    Type return_type,
    ArrayRef<ParamTypeData> param_types
) : TypeBase{Kind::ProcType},
    cc{cconv},
    is_variadic{variadic},
    num_params{u32(param_types.size())},
    return_type{return_type} {
    std::uninitialized_copy_n(
        param_types.begin(),
        param_types.size(),
        getTrailingObjects<ParamTypeData>()
    );
}

void ProcType::Profile(
    FoldingSetNodeID& ID,
    Type return_type,
    ArrayRef<ParamTypeData> param_types,
    CallingConvention cc,
    bool is_variadic
) {
    ID.AddInteger(+cc);
    ID.AddBoolean(is_variadic);
    ID.AddPointer(return_type.ptr());
    ID.AddInteger(param_types.size());
    for (const auto& t : param_types) {
        ID.AddInteger(+t.intent);
        ID.AddPointer(t.type.ptr());
    }
}

auto ProcType::print(StringRef proc_name, bool number_params) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    out += "%1(proc";

    // Add name.
    if (not proc_name.empty())
        out += std::format(" %2({}%)", proc_name);

    // Add params.
    const auto& ps = params();
    if (not ps.empty()) {
        out += " (";
        bool first = true;
        for (const auto& [i, p] : enumerate(ps)) {
            if (first) first = false;
            else out += ", ";
            if (p.intent != Intent::Move) out += std::format("{} ", p.intent);
            out += p.type->print();
            if (number_params) out += std::format(" %4(%%{}%)", i);
        }
        out += ")";
    }

    // Add attributes.
    if (cconv() == CallingConvention::Native) out += " native";
    if (variadic()) out += " variadic";

    // Add return type.
    if (not ret()->is_void())
        out += std::format(" -> {}", ret()->print());

    out += "%)";
    return out;
}

auto SliceType::Get(TranslationUnit& mod, Type elem) -> SliceType* {
    auto CreateNew = [&] { return new (mod) SliceType{elem}; };
    return GetOrCreateType(mod.slice_types, CreateNew, elem);
}

void SliceType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.ptr());
}

auto StructType::Create(
    TranslationUnit& owner,
    StructScope* scope,
    String name,
    u32 num_fields,
    Location decl_loc
) -> StructType* {
    auto type = ::new (owner.allocate(
        totalSizeToAlloc<FieldDecl*>(num_fields),
        alignof(StructType)
    )) StructType{owner, scope, num_fields};
    type->type_decl = new (owner) TypeDecl{type, name, decl_loc};
    return type;
}

auto StructType::name() const -> String {
    return type_decl->name;
}

void StructType::finalise(
    ArrayRef<FieldDecl*> fields,
    Size sz,
    Align align,
    Bits struct_bits
) {
    Assert(not finalised, "finalise() called twice?");
    finalised = true;
    bits = struct_bits;
    computed_size = sz;
    computed_alignment = align;
    computed_array_size = sz.align(align);
    std::uninitialized_copy_n(
        fields.begin(),
        fields.size(),
        getTrailingObjects<FieldDecl*>()
    );
}
