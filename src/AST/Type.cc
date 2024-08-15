module;

#include <memory>
#include <print>
#include <srcc/Macros.hh>
#include <llvm/Support/MathExtras.h>

module srcc.ast;
import srcc;
import :type;
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

// ============================================================================
//  Type
// ============================================================================
void* TypeBase::operator new(usz size, TranslationUnit& mod) {
    return mod.allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

auto TypeBase::align(TranslationUnit& tu) const -> Align {
    switch (type_kind) {
        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return Align{1};
                case BuiltinKind::Int: return Align{8}; // FIXME: Get alignment from context.
                case BuiltinKind::NoReturn: return Align{1};
                case BuiltinKind::Void: return Align{1};
                case BuiltinKind::Type: return Align::Of<Type>();
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                    Unreachable("Requested alignment of dependent type");
            }
        }

        case Kind::ArrayType:
        case Kind::SliceType:
            return cast<SingleElementTypeBase>(this)->elem()->align(tu);

        case Kind::ReferenceType: return Align{8}; // FIXME: Get pointer alignment from context.
        case Kind::IntType: {
            auto bytes = cast<IntType>(this)->bit_width().bytes();
            return Align{llvm::PowerOf2Ceil(bytes)};
        }

        case Kind::ProcType: return Align{8}; // FIXME: Get closure alignment from context.

        case Kind::TemplateType: Unreachable("Requested size of dependent type");
    }

    Unreachable("Invalid type kind");
}

auto TypeBase::array_size(TranslationUnit& tu) const -> Size {
    // Currently identical for all types we support.
    return size(tu);
}

void TypeBase::dump(bool use_colour) const {
    std::print("{}", print(use_colour));
}

bool TypeBase::is_void() const {
    return kind() == Kind::BuiltinType and cast<BuiltinType>(this)->builtin_kind() == BuiltinKind::Void;
}

auto TypeBase::print(bool use_colour) const -> std::string {
    utils::Colours C{use_colour};
    std::string out = print_impl(C);
    out += C(utils::Colour::Reset);
    return out;
}

auto TypeBase::print_impl(utils::Colours C) const -> std::string {
    using enum utils::Colour;
    switch (kind()) {
        case Kind::ArrayType: {
            auto* arr = cast<ArrayType>(this);
            return std::format(
                "{}[{}{}{}]{}",
                arr->elem()->print_impl(C),
                C(Red),
                C(Magenta),
                arr->dimension(),
                C(Red)
            );
        }

        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return std::format("{}bool", C(Cyan));
                case BuiltinKind::Deduced: return std::format("{}var", C(Cyan));
                case BuiltinKind::Dependent: return std::format("{}<dependent type>", C(Cyan));
                case BuiltinKind::ErrorDependent: return std::format("{}<error>", C(Cyan));
                case BuiltinKind::Int: return std::format("{}int", C(Cyan));
                case BuiltinKind::NoReturn: return std::format("{}noreturn", C(Cyan));
                case BuiltinKind::Type: return std::format("{}type", C(Cyan));
                case BuiltinKind::Void: return std::format("{}void", C(Cyan));
            }
        }

        case Kind::IntType: {
            auto* int_ty = cast<IntType>(this);
            return std::format("{}i{}", C(Cyan), int_ty->bit_width());
        }

        case Kind::ProcType: {
            auto proc = cast<ProcType>(this);
            auto ret = std::format("{}proc", C(Red));

            // Add params.
            auto params = proc->params();
            if (not params.empty()) {
                ret += std::format("{} (", C(Red));
                bool first = true;
                for (auto p : params) {
                    if (first) first = false;
                    else ret += std::format("{}, ", C(Red));
                    ret += p->print_impl(C);
                }
                ret += std::format("{})", C(Red));
            }

            // Add attributes.
            if (proc->cconv() == CallingConvention::Native) ret += " native";
            if (proc->variadic()) ret += " variadic";

            // Add return type.
            if (not proc->ret()->is_void())
                ret += std::format(" {}-> {}", C(Red), proc->ret()->print_impl(C));

            return ret;
        }

        case Kind::ReferenceType: {
            auto* ref = cast<ReferenceType>(this);
            return std::format("{}ref {}", C(Red), ref->elem()->print_impl(C));
        }

        case Kind::SliceType: {
            auto* slice = cast<SliceType>(this);
            return std::format("{}{}[]", slice->elem()->print_impl(C), C(Red));
        }

        case Kind::TemplateType: {
            auto* tt = cast<TemplateType>(this);
            return std::format("{}{}", C(Yellow), tt->template_decl()->name);
        }
    }

    Unreachable("Invalid type kind");
}

auto TypeBase::size(TranslationUnit& tu) const -> Size {
    switch (type_kind) {
        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return Size::Bits(1);
                case BuiltinKind::Int: return Size::Bytes(8); // FIXME: Get size from context.
                case BuiltinKind::Type: return Size::Of<Type>();

                case BuiltinKind::NoReturn:
                case BuiltinKind::Void:
                    return Size();

                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                    Unreachable("Requested size of dependent type");
            }
        }

        case Kind::ReferenceType: return Size::Bytes(8); // FIXME: Get pointer size from context.
        case Kind::IntType: return cast<IntType>(this)->bit_width();
        case Kind::ProcType: return Size::Bytes(16); // FIXME: Get closure size from context.
        case Kind::SliceType: return Size::Bytes(16); // FIXME: Get slice size from context.
        case Kind::ArrayType: {
            auto arr = cast<ArrayType>(this);
            return arr->elem()->array_size(tu) * usz(arr->dimension());
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
        case Kind::ArrayType: return Expr::LValue;

        // Slices are a pointer+size, which our ABI just passes in registers.
        case Kind::SliceType: return Expr::SRValue;

        // Pointers are just scalars.
        case Kind::ReferenceType: return Expr::SRValue;

        // Integers may end up being rather large (e.g. i1024), but
        // that’s for the backend to deal with.
        case Kind::IntType: return Expr::SRValue;

        // Invalid, can’t be instantiated; return some random nonsense.
        case Kind::ProcType: return Expr::DValue;

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

auto ProcType::Get(
    TranslationUnit& mod,
    Type return_type,
    ArrayRef<Type> param_types,
    CallingConvention cconv,
    bool variadic
) -> ProcType* {
    auto CreateNew = [&] {
        const auto size = totalSizeToAlloc<Type>(param_types.size());
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
    ArrayRef<Type> param_types
) : TypeBase{Kind::ProcType},
    cc{cconv},
    is_variadic{variadic},
    num_param_types{u32(param_types.size())},
    return_type{return_type} {
    std::uninitialized_copy_n(
        param_types.begin(),
        param_types.size(),
        getTrailingObjects<Type>()
    );
    ComputeDependence();
}

void ProcType::Profile(FoldingSetNodeID& ID, Type return_type, ArrayRef<Type> param_types, CallingConvention cc, bool is_variadic) {
    ID.AddInteger(+cc);
    ID.AddBoolean(is_variadic);
    ID.AddPointer(return_type.as_opaque_ptr());
    ID.AddInteger(param_types.size());
    for (auto t : param_types) ID.AddPointer(t.as_opaque_ptr());
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
