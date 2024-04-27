module;

#include <fmt/format.h>
#include <memory>
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
void* Type::operator new(usz size, Module& mod) {
    return mod.Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

bool Type::is_void() const {
    return kind() == Kind::BuiltinType and cast<BuiltinType>(this)->builtin_kind() == BuiltinKind::Void;
}

auto Type::print(bool use_colour) const -> std::string {
    using enum utils::Colour;
    utils::Colours C{use_colour};
    switch (kind()) {
        case Kind::ArrayType: {
            auto* arr = cast<ArrayType>(this);
            return fmt::format(
                "{}[{}{}{}]",
                arr->elem()->print(use_colour),
                C(Red),
                C(Magenta),
                arr->dimension(),
                C(Red)
            );
        }

        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: return fmt::format("{}bool", C(Red));
                case BuiltinKind::Void: return fmt::format("{}void", C(Red));
                case BuiltinKind::NoReturn: return fmt::format("{}noreturn", C(Red));
            }
        }

        case Kind::IntType: {
            auto* int_ty = cast<IntType>(this);
            return fmt::format("{}i{}", C(Red), int_ty->bit_width());
        }

        case Kind::ProcType: {
            auto proc = cast<ProcType>(this);
            auto ret = fmt::format("{}proc", C(Red));

            // Add params.
            auto params = proc->params();
            if (not params.empty()) {
                ret += fmt::format("{}(", C(Red));
                for (auto p : params) {
                    ret += p->print(use_colour);
                    if (p != params.back()) ret += fmt::format("{}, ", C(Red));
                }
                ret += fmt::format("{})", C(Red));
            }

            // Add attributes.
            if (proc->cconv() == CallingConvention::Native) ret += " native";
            if (proc->variadic()) ret += " variadic";

            // Add return type.
            if (not proc->ret()->is_void())
                ret += fmt::format(" {}-> {}", C(Red), proc->ret()->print(use_colour));

            return ret;
        }

        case Kind::ReferenceType: {
            auto* ref = cast<ReferenceType>(this);
            return fmt::format("{}ref {}", C(Red), ref->elem()->print(use_colour));
        }

        case Kind::SliceType: {
            auto* slice = cast<SliceType>(this);
            return fmt::format("{}[]", C(Red), slice->elem()->print(use_colour));
        }
    }

    Unreachable("Invalid type kind");
}

// ============================================================================
//  Types
// ============================================================================
auto ArrayType::Get(Module& mod, Type* elem, i64 size) -> ArrayType* {
    auto CreateNew = [&] { return new (mod) ArrayType{elem, size}; };
    return FindOrCreateType(mod.array_types, CreateNew, elem, size);
}

void ArrayType::Profile(FoldingSetNodeID& ID, Type* elem, i64 size) {
    ID.AddPointer(elem);
    ID.AddInteger(size);
}

auto IntType::Get(Module& mod, i64 bits) -> IntType* {
    auto CreateNew = [&] { return new (mod) IntType{bits}; };
    return FindOrCreateType(mod.int_types, CreateNew, bits);
}

void IntType::Profile(FoldingSetNodeID& ID, i64 bits) {
    ID.AddInteger(bits);
}

auto ReferenceType::Get(Module& mod, Type* elem) -> ReferenceType* {
    auto CreateNew = [&] { return new (mod) ReferenceType{elem}; };
    return FindOrCreateType(mod.reference_types, CreateNew, elem);
}

void ReferenceType::Profile(FoldingSetNodeID& ID, Type* elem) {
    ID.AddPointer(elem);
}

auto ProcType::Get(
    Module& mod,
    Type* return_type,
    ArrayRef<Type*> param_types,
    CallingConvention cconv,
    bool variadic
) -> ProcType* {
    auto CreateNew = [&] {
        const auto size = totalSizeToAlloc<Type*>(param_types.size());
        auto mem = mod.Allocate(size, alignof(ProcType));
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
    Type* return_type,
    ArrayRef<Type*> param_types
) : Type{Kind::ProcType},
    cc{cconv},
    is_variadic{variadic},
    num_param_types{u32(param_types.size())},
    return_type{return_type} {
    std::uninitialized_copy_n(
        param_types.begin(),
        param_types.size(),
        getTrailingObjects<Type*>()
    );
}

void ProcType::Profile(FoldingSetNodeID& ID, Type* return_type, ArrayRef<Type*> param_types, CallingConvention cc, bool is_variadic) {
    ID.AddInteger(+cc);
    ID.AddBoolean(is_variadic);
    ID.AddPointer(return_type);
    ID.AddInteger(param_types.size());
    for (auto t : param_types) ID.AddPointer(t);
}

auto SliceType::Get(Module& mod, Type* elem) -> SliceType* {
    auto CreateNew = [&] { return new (mod) SliceType{elem}; };
    return FindOrCreateType(mod.slice_types, CreateNew, elem);
}

void SliceType::Profile(FoldingSetNodeID& ID, Type* elem) {
    ID.AddPointer(elem);
}
