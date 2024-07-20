module;

#include <memory>
#include <print>
#include <srcc/Macros.hh>

module srcc.ast;
import srcc;
import :type;
using namespace srcc;

// ============================================================================
//  Helpers
// ============================================================================
template <typename T, typename... Args>
auto FindOrCreateType(FoldingSet<T>& Set, auto CreateNew, Args&&... args) -> T* {
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
                case BuiltinKind::Void: return std::format("{}void", C(Cyan));
                case BuiltinKind::Dependent: return std::format("{}<dependent type>", C(Cyan));
                case BuiltinKind::Bool: return std::format("{}bool", C(Cyan));
                case BuiltinKind::NoReturn: return std::format("{}noreturn", C(Cyan));
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
    }

    Unreachable("Invalid type kind");
}

// ============================================================================
//  Types
// ============================================================================
auto ArrayType::Get(TranslationUnit& mod, Type elem, i64 size) -> ArrayType* {
    auto CreateNew = [&] { return new (mod) ArrayType{elem, size}; };
    return FindOrCreateType(mod.array_types, CreateNew, elem, size);
}

void ArrayType::Profile(FoldingSetNodeID& ID, Type elem, i64 size) {
    ID.AddPointer(elem.as_opaque_ptr());
    ID.AddInteger(size);
}

auto IntType::Get(TranslationUnit& mod, i64 bits) -> IntType* {
    auto CreateNew = [&] { return new (mod) IntType{bits}; };
    return FindOrCreateType(mod.int_types, CreateNew, bits);
}

void IntType::Profile(FoldingSetNodeID& ID, i64 bits) {
    ID.AddInteger(bits);
}

auto ReferenceType::Get(TranslationUnit& mod, Type elem) -> ReferenceType* {
    auto CreateNew = [&] { return new (mod) ReferenceType{elem}; };
    return FindOrCreateType(mod.reference_types, CreateNew, elem);
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

    return FindOrCreateType(
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
}

void ProcType::Profile(FoldingSetNodeID& ID, Type return_type, ArrayRef<Type> param_types, CallingConvention cc, bool is_variadic) {
    ID.AddInteger(+cc);
    ID.AddBoolean(is_variadic);
    ID.AddPointer(return_type.as_opaque_ptr());
    ID.AddInteger(param_types.size());
    for (auto t : param_types) ID.AddPointer(t.as_opaque_ptr());
}

auto SliceType::Get(TranslationUnit& mod, Type elem) -> SliceType* {
    auto CreateNew = [&] { return new (mod) SliceType{elem}; };
    return FindOrCreateType(mod.slice_types, CreateNew, elem);
}

void SliceType::Profile(FoldingSetNodeID& ID, Type elem) {
    ID.AddPointer(elem.as_opaque_ptr());
}
