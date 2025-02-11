#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>

#include <llvm/Transforms/Utils/ModuleUtils.h>

/*
using namespace srcc;
using namespace srcc::cg;

namespace Intrinsic = llvm::Intrinsic;

using llvm::BasicBlock;
using llvm::ConstantInt;
using llvm::IRBuilder;
using llvm::Value;

// ============================================================================
//  Helpers
// ============================================================================
namespace {
auto ConvertCC(CallingConvention cc) -> llvm::CallingConv::ID {
    switch (cc) {
        case CallingConvention::Source: return llvm::CallingConv::Fast;
        case CallingConvention::Native: return llvm::CallingConv::C;
    }

    Unreachable("Unknown calling convention");
}

auto ConvertLinkage(Linkage lnk) -> llvm::GlobalValue::LinkageTypes {
    switch (lnk) {
        using L = llvm::GlobalValue::LinkageTypes;
        case Linkage::Internal: return L::PrivateLinkage;
        case Linkage::Exported: return L::ExternalLinkage;
        case Linkage::Imported: return L::ExternalLinkage;
        case Linkage::Reexported: return L::ExternalLinkage;
        case Linkage::Merge: return L::LinkOnceODRLinkage;
    }

    Unreachable("Unknown linkage");
}

// ============================================================================
//  Type Conversion
// ============================================================================
auto CodeGen::ConvertTypeImpl(Type ty, bool array_elem) -> llvm::Type* {
    switch (ty->kind()) {
        case TypeBase::Kind::SliceType: return SliceTy;
        case TypeBase::Kind::ReferenceType: return PtrTy;
        case TypeBase::Kind::ProcType: return ConvertProcType(cast<ProcType>(ty).ptr());
        case TypeBase::Kind::IntType: return getIntNTy(u32(cast<IntType>(ty)->bit_width().bits()));

        case TypeBase::Kind::ArrayType: {
            // FIXME: This doesn’t handle structs correctly at the moment.
            auto arr = cast<ArrayType>(ty);
            auto elem = ConvertType(arr->elem(), true);
            return llvm::ArrayType::get(elem, u64(arr->dimension()));
        }

        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                    Unreachable("Dependent type in codegen?");

                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable("Unresolved overload set type in codegen?");

                case BuiltinKind::Type:
                    Unreachable("Cannot emit 'type' type");

                case BuiltinKind::Bool: return I1Ty;
                case BuiltinKind::Int: return IntTy;

                case BuiltinKind::Void:
                case BuiltinKind::NoReturn:
                    return VoidTy;
            }

            Unreachable("Unknown builtin type");
        }

        case TypeBase::Kind::TemplateType: Unreachable("TemplateType in codegen?");
        case TypeBase::Kind::StructType: {
            auto s = cast<StructType>(ty);
            auto sz = array_elem ? s->array_size() : s->size();
            return llvm::ArrayType::get(I8Ty, sz.bytes());
        }
    }

    Unreachable("Unknown type kind");
}
}

CGLLVM::CGLLVM(TranslationUnit& tu, llvm::TargetMachine& machine)
    : CodeGen{tu, Size::Bits(64) /* FIXME: Don’t hard-code this #1#}, machine{machine},
      llvm{std::make_unique<llvm::Module>(tu.name, tu.llvm_context)},
      builder{tu.llvm_context},
      IntTy{builder.getInt64Ty()},
      I1Ty{builder.getInt1Ty()},
      I8Ty{builder.getInt8Ty()},
      PtrTy{builder.getPtrTy()},
      FFIIntTy{llvm::Type::getIntNTy(tu.llvm_context, 32)}, // FIXME: Get size from target.
      SliceTy{llvm::StructType::get(PtrTy, IntTy)},
      ClosureTy{llvm::StructType::get(PtrTy, PtrTy)},
      VoidTy{builder.getVoidTy()} {
    llvm->setTargetTriple(machine.getTargetTriple().str());
    llvm->setDataLayout(machine.createDataLayout());
}

auto CGLLVM::GetStringPtr(StringRef s) -> llvm::Constant* {
    if (auto it = strings.find(s); it != strings.end()) return it->second;
    return strings[s] = builder.CreateGlobalString(s);
}

void CGLLVM::finalise() {
    if (tu.is_module) {
        SmallString<0> md;
        tu.serialise(md);
        auto mb = llvm::MemoryBuffer::getMemBuffer(md, "", false);
        llvm::embedBufferInModule(*llvm, mb->getMemBufferRef(), constants::ModuleDescriptionSectionName(tu.name));
    }
}

auto CGLLVM::CreateBool(bool value) -> SRValue {
    return box(value ? builder.getTrue() : builder.getFalse());
}

auto CGLLVM::CreateCallImpl(TypedSRValue callee, ArrayRef<SRValue> args) -> SRValue {
    llvm::FunctionCallee func{ConvertProcType(cast<ProcType>(callee.type).ptr()), unbox(callee.value)};
    return box(builder.CreateCall(func, unbox(args)));
}

auto CGLLVM::CreateEmptySlice() -> SRValue {
    return box(llvm::ConstantPointerNull::get(PtrTy), ConstantInt::get(IntTy, 0));
}

auto CGLLVM::CreateICmp(Tk op, SRValue a, SRValue b) -> SRValue {
    auto pred = [&] {
        switch (op) {
            case Tk::ULt: return llvm::CmpInst::Predicate::ICMP_ULT;
            case Tk::ULe: return llvm::CmpInst::Predicate::ICMP_ULE;
            case Tk::UGt: return llvm::CmpInst::Predicate::ICMP_UGT;
            case Tk::UGe: return llvm::CmpInst::Predicate::ICMP_UGE;
            case Tk::SLt: return llvm::CmpInst::Predicate::ICMP_SLT;
            case Tk::SLe: return llvm::CmpInst::Predicate::ICMP_SLE;
            case Tk::SGt: return llvm::CmpInst::Predicate::ICMP_SGT;
            case Tk::SGe: return llvm::CmpInst::Predicate::ICMP_SGE;
            case Tk::EqEq: return llvm::CmpInst::Predicate::ICMP_EQ;
            case Tk::Neq: return llvm::CmpInst::Predicate::ICMP_NE;
            default: Unreachable("Unknown comparison operator");
        }
    }();

    return box(builder.CreateICmp(pred, unbox(a), unbox(b)));
}

auto CGLLVM::CreateInt(const APInt& value) -> SRValue {
    return box(ConstantInt::get(tu.llvm_context, value));
}

auto CGLLVM::CreateInt(i64 integer, Type type) -> SRValue {
    return box(ConstantInt::get(ConvertType(type), u64(integer), true));
}

auto CGLLVM::CreateIMul(SRValue a, SRValue b) -> SRValue {
    return box(builder.CreateMul(unbox(a), unbox(b)));
}

auto CGLLVM::CreatePtrAdd(SRValue ptr, SRValue offs, bool inbounds) -> SRValue {
    auto flags = inbounds ? llvm::GEPNoWrapFlags::all() : llvm::GEPNoWrapFlags::none();
    return box(builder.CreatePtrAdd(unbox(ptr), unbox(offs), "", flags));
}

auto CGLLVM::CreateStringSlice(StringRef s) -> SRValue {
    auto ptr = GetStringPtr(s);
    auto size = ConstantInt::get(IntTy, s.size());
    return box(ptr, size);
}

void CGLLVM::CreateUnreachable() {
    builder.CreateUnreachable();
}
*/
