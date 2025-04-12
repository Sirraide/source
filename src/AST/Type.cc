#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>

#include <llvm/Support/MathExtras.h>

#include <memory>
#include <print>

using namespace srcc;

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

auto TypeBase::align(TranslationUnit& tu) const -> Align { // clang-format off
    return visit(utils::Overloaded{
        [&](const ArrayType* ty) -> Align { return ty->elem()->align(tu); },
        [&](const BuiltinType* ty) -> Align {
            switch (ty->builtin_kind()) {
                case BuiltinKind::Bool: return Align{1};
                case BuiltinKind::Int: return Align{8}; // FIXME: Get alignment from context.
                case BuiltinKind::NoReturn: return Align{1};
                case BuiltinKind::Void: return Align{1};
                case BuiltinKind::Type: return Align::Of<Type>();
                case BuiltinKind::UnresolvedOverloadSet: return Align{8};
                case BuiltinKind::ErrorDependent: return Align{1}; // Dummy value.
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                    Unreachable("Requested alignment of dependent type");
            }
            Unreachable();
        },
        [&](const IntType* ty) { return Align{std::min<u64>(64, llvm::PowerOf2Ceil(u64(ty->bit_width().bytes())))}; },
        [&](const ProcType*) { return Align{8}; }, // FIXME: Get alignment from context.
        [&](const ReferenceType*) { return Align{8}; }, // FIXME: Get alignment from context.
        [&](const SliceType*) { return Align{8}; }, // FIXME: Get alignment from context.
        [&](const StructType* ty) {
            Assert(ty->is_complete(), "Requested size of incomplete struct");
            return ty->align();
        },
        [&](const TemplateType*) -> Align { Unreachable("Requested size of dependent type"); },
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
                case BuiltinKind::ErrorDependent:
                    return true;

                case BuiltinKind::UnresolvedOverloadSet:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Type:
                    return false;

                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                    Unreachable("Querying property of dependent type");
            }
            Unreachable();
        },
        [&](const IntType*) { return true; },
        [&](const ProcType*) { return false; },
        [&](const ReferenceType*) { return false; },
        [&](const SliceType*) { return true; },
        [&](const StructType* ty) { return (ty->*struct_predicate)(); },
        [&](const TemplateType*) -> bool { Unreachable("Querying property of dependent type"); },
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
    return this == Types::IntTy or isa<IntType>(this);
}

bool TypeBase::is_void() const {
    return kind() == Kind::BuiltinType and
           cast<BuiltinType>(this)->builtin_kind() == BuiltinKind::Void;
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
                [](ReferenceType*) { return true; },                     // References are small.
                [](SliceType*) { return true; },                         // Slices are two pointers.
                [](StructType*) { return false; },                       // Pass structs by reference (TODO: small ones by value).
                [](TemplateType*) -> bool { Unreachable(); }             // Should never be called for these.
            }); // clang-format on
    }
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
                case BuiltinKind::Dependent: out += "%6(<dependent type>%)"; break;
                case BuiltinKind::ErrorDependent: out += "%6(<error>%)"; break;
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

        case Kind::ReferenceType: {
            auto* ref = cast<ReferenceType>(this);
            out += std::format("%1(ref%) {}", ref->elem()->print());
        } break;

        case Kind::SliceType: {
            auto* slice = cast<SliceType>(this);
            out += std::format("{}%1([]%)", slice->elem()->print());
        } break;

        case Kind::StructType: {
            auto* s = cast<StructType>(this);
            out += std::format("%6({}%)", s->name());
        } break;

        case Kind::TemplateType: {
            auto* tt = cast<TemplateType>(this);
            out += std::format("%3({}%)", tt->template_decl()->name);
        } break;
    }

    return out;
}

auto TypeBase::size(TranslationUnit& tu) const -> Size {
    switch (type_kind) {
        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return Size::Bits(1);
                case BuiltinKind::Int: return Size::Bytes(8); // FIXME: Get size from context.
                case BuiltinKind::Type: return Size::Of<Type>();
                case BuiltinKind::UnresolvedOverloadSet: return Size::Bytes(16); // FIXME: Get size from context.

                case BuiltinKind::ErrorDependent:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Void:
                    return Size();

                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                    Unreachable("Requested size of dependent type");
            }
        }

        case Kind::ReferenceType: return Size::Bytes(8); // FIXME: Get pointer size from context.
        case Kind::IntType: return cast<IntType>(this)->bit_width();
        case Kind::ProcType: return Size::Bytes(16);  // FIXME: Get closure size from context.
        case Kind::SliceType: return Size::Bytes(16); // FIXME: Get slice size from context.
        case Kind::StructType: {
            auto s = cast<StructType>(this);
            return s->size();
        }
        case Kind::ArrayType: {
            auto arr = cast<ArrayType>(this);
            return arr->elem()->array_size(tu) * u64(arr->dimension());
        }

        case Kind::TemplateType: Unreachable("Requested size of dependent type");
    }

    Unreachable("Invalid type kind");
}

auto TypeBase::value_category() const -> ValueCategory {
    switch (type_kind) {
        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                // 'void' is our unit type; 'noreturn' is never instantiated
                // by definition, so just making it an srvalue is fine.
                case BuiltinKind::Void:
                case BuiltinKind::NoReturn:
                case BuiltinKind::Bool:
                case BuiltinKind::Int:
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
                    return Expr::SRValue;

                // Don’t know yet.
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                    return Expr::DValue;
            }

            Unreachable("Invalid builtin kind");
        }

        // It’s not worth it to try and construct arrays in registers.
        case Kind::ArrayType: return Expr::MRValue;

        // Slices are a pointer+size, which our ABI just passes in registers.
        case Kind::SliceType: return Expr::SRValue;

        // Structs are always passed in memory.
        // TODO: Only if they’re big, have a non-trivial ctor,
        //       or are not trivially copyable/movable.
        case Kind::StructType: return Expr::MRValue;

        // Pointers are just scalars.
        case Kind::ReferenceType: return Expr::SRValue;

        // Integers may end up being rather large (e.g. i1024), but
        // that’s for the backend to deal with.
        case Kind::IntType: return Expr::SRValue;

        // Closures are srvalues.
        case Kind::ProcType: return Expr::SRValue;

        // Dependent.
        case Kind::TemplateType: return Expr::DValue;
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
    ID.AddPointer(elem.as_opaque_ptr());
    ID.AddInteger(size);
}

auto IntType::Get(TranslationUnit& mod, Size bits) -> IntType* {
    auto CreateNew = [&] { return new (mod) IntType{bits}; };
    return GetOrCreateType(mod.int_types, CreateNew, bits);
}

void IntType::Profile(FoldingSetNodeID& ID, Size bits) {
    ID.AddInteger(bits.bits());
}

auto ReferenceType::Get(TranslationUnit& mod, Type elem) -> ReferenceType* {
    auto CreateNew = [&] { return new (mod) ReferenceType{elem}; };
    return GetOrCreateType(mod.reference_types, CreateNew, elem);
}

void ReferenceType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.as_opaque_ptr());
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

auto ProcType::GetInvalid(TranslationUnit& tu) -> ProcType* {
    return Get(tu, Types::ErrorDependentTy, {});
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
    ComputeDependence();
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
    ID.AddPointer(return_type.as_opaque_ptr());
    ID.AddInteger(param_types.size());
    for (const auto& t : param_types) {
        ID.AddInteger(+t.intent);
        ID.AddPointer(t.type.as_opaque_ptr());
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


auto TemplateType::Get(TranslationUnit& tu, TemplateTypeDecl* decl) -> TemplateType* {
    auto CreateNew = [&] { return new (tu) TemplateType{decl}; };
    return GetOrCreateType(tu.template_types, CreateNew, decl);
}

void TemplateType::Profile(FoldingSetNodeID& ID, TemplateTypeDecl* decl) {
    ID.AddPointer(decl);
}

auto SliceType::Get(TranslationUnit& mod, Type elem) -> SliceType* {
    auto CreateNew = [&] { return new (mod) SliceType{elem}; };
    return GetOrCreateType(mod.slice_types, CreateNew, elem);
}

void SliceType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.as_opaque_ptr());
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
