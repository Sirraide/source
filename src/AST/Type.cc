module;

#include <llvm/Support/MathExtras.h>
#include <memory>
#include <print>
#include <srcc/Macros.hh>

module srcc.ast;
import srcc;
import base.colours;
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

TypeLoc::TypeLoc(Expr* e): ty{e->type}, loc{e->location()} {}

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
                case BuiltinKind::UnresolvedOverloadSet: return Align{8};
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
    std::print("{}", text::RenderColours(use_colour, print().str()));
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
            return visit(utils::Overloaded{ // clang-format off
                [](ArrayType*) { return false; },                        // Arrays are usually big, so pass by reference.
                [](BuiltinType*) { return true; },                       // All builtin types are small.
                [](IntType* t) { return t->bit_width().bits() <= 128; }, // Only pass small ints by value.
                [](ProcType*) { return true; },                          // Closures are two pointers.
                [](ReferenceType*) { return true; },                     // References are small.
                [](SliceType*) { return true; },                         // Slices are two pointers.
                [](TemplateType*) -> bool { Unreachable(); }             // Should never be called for these.
            }); // clang-format on
    }
}

auto TypeBase::print() const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (kind()) {
        case Kind::ArrayType: {
            auto* arr = cast<ArrayType>(this);
            out += std::format("{}%1([%5({})])", arr->elem()->print(), arr->dimension());
        } break;

        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind()) {
                case BuiltinKind::Bool: out += "%6(bool)"; break;
                case BuiltinKind::Deduced: out += "%6(var)"; break;
                case BuiltinKind::Dependent: out += "%6(<dependent type>)"; break;
                case BuiltinKind::ErrorDependent: out += "%6(<error>)"; break;
                case BuiltinKind::UnresolvedOverloadSet: out += "%6(<overload set>)"; break;
                case BuiltinKind::Int: out += "%6(int)"; break;
                case BuiltinKind::NoReturn: out += "%6(noreturn)"; break;
                case BuiltinKind::Type: out += "%6(type)"; break;
                case BuiltinKind::Void: out += "%6(void)"; break;
            }
        } break;

        case Kind::IntType: {
            auto* int_ty = cast<IntType>(this);
            out += std::format("%6(i{})", int_ty->bit_width());
        } break;

        case Kind::ProcType: {
            auto proc = cast<ProcType>(this);
            out += "%1(proc";

            // Add params.
            const auto& params = proc->params();
            if (not params.empty()) {
                out += " (";
                bool first = true;
                for (const auto& p : params) {
                    if (first) first = false;
                    else out += ", ";
                    if (p.intent != Intent::Move) out += std::format("{} ", p.intent);
                    out += p.type->print();
                }
                out += "\033)";
            }

            // Add attributes.
            if (proc->cconv() == CallingConvention::Native) out += " native";
            if (proc->variadic()) out += " variadic";

            // Add return type.
            if (not proc->ret()->is_void())
                out += std::format(" -> {}", proc->ret()->print());

            out += ")";
        } break;

        case Kind::ReferenceType: {
            auto* ref = cast<ReferenceType>(this);
            out += std::format("%1(ref) {}", ref->elem()->print());
        } break;

        case Kind::SliceType: {
            auto* slice = cast<SliceType>(this);
            out += std::format("{}%1([])", slice->elem()->print());
        } break;

        case Kind::TemplateType: {
            auto* tt = cast<TemplateType>(this);
            out += std::format("%3({})", tt->template_decl()->name);
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
        case Kind::ProcType: return Size::Bytes(16);  // FIXME: Get closure size from context.
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
    ArrayRef<Parameter> param_types,
    CallingConvention cconv,
    bool variadic
) -> ProcType* {
    auto CreateNew = [&] {
        const auto size = totalSizeToAlloc<Parameter>(param_types.size());
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
    ArrayRef<Parameter> param_types
) : TypeBase{Kind::ProcType},
    cc{cconv},
    is_variadic{variadic},
    num_params{u32(param_types.size())},
    return_type{return_type} {
    std::uninitialized_copy_n(
        param_types.begin(),
        param_types.size(),
        getTrailingObjects<Parameter>()
    );
    ComputeDependence();
}

void ProcType::Profile(
    FoldingSetNodeID& ID,
    Type return_type,
    ArrayRef<Parameter> param_types,
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
