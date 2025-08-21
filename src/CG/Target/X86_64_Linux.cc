#include <srcc/AST/Stmt.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>

#include <llvm/ADT/BitVector.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace srcc;
using namespace srcc::cg;
using mlir::Value;

namespace {
enum class Class {
    INTEGER,
    SSE,
    SSEUP,
    X87,
    X87UP,
    COMPLEX_X87,
    NO_CLASS,
    MEMORY,
};


constexpr u32 MaxRegs = 6;
constexpr Size Eightbyte = Size::Bits(64);

using enum Class;
using Classification = SmallVector<Class, 8>;

struct Impl final : Target {
    Impl(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) : Target(std::move(TI)) {}
    bool is_returned_as_mrvalue(Type ty) const override;
    bool needs_indirect_return(Type ty) const override;
    auto get_call_builder(
        CodeGen& cg,
        ProcType* ty,
        Value indirect_ptr
    ) const -> std::unique_ptr<CallBuilder> override;
    bool pass_in_parameter_by_value(Type ty) const override;
};

struct Builder final : CallBuilder {
    SmallVector<Value> args;
    SmallVector<mlir::Attribute> args_attrs;
    ProcType* proc_ty;
    Classification ret_class;
    Value indirect_ptr;
    bool indirect = false;
    u32 num_regs = 0;

    Builder(CodeGen& cg, ProcType* ty, Value indirect_ptr);

    void add(Value v, mlir::DictionaryAttr a = {});
    void add_argument(Type param_ty, Value v, bool require_copy) override;
    void add_byval_arg(mlir::Type ty, Value ptr);
    void add_copy(Type ty, Value ptr, bool require_copy);
    void add_pointer_or_integer(Value v) override;
    void add_pointer(Value v) override;
    auto get_arg_attrs() -> mlir::ArrayAttr override;
    auto get_final_args() -> ArrayRef<Value> override;
    auto get_proc_type() -> mlir::FunctionType override;
    auto get_result_types() -> SmallVector<mlir::Type, 2> override;
    auto get_result_vals(ir::CallOp call) -> SRValue override;
};
}

static auto ClassifyEightbytes(const Target& t, Type ty) -> Classification {
    // If the size of an object is larger than eight eightbytes [...] it has class MEMORY'
    //
    // Additionally, the following rule applies aggregate that isn’t an XMM/YMM/ZMM register:
    // 'If the size of the aggregate exceeds two eightbytes [...] the whole argument is passed
    // in memory.'
    //
    // We don’t have floats at the moment, so short-circuit that here.
    auto sz = ty->size(t);
    if (sz > Eightbyte * 2) return {MEMORY};

    // Integer types and pointers are INTEGER. >64 bit integers are split into
    // 8-byte chunks, each of which are INTEGER.
    if (ty == Type::BoolTy or ty == Type::IntTy or isa<PtrType>(ty)) return {INTEGER};
    if (auto i = dyn_cast<IntType>(ty)) {
        if (i->bit_width() <= Eightbyte) return {INTEGER};
        if (i->bit_width() <= Eightbyte * 2) return {INTEGER, INTEGER};
        return {MEMORY};
    }

    // For arrays, since every element has the same type, we only need to classify them once.
    if (auto a = dyn_cast<ArrayType>(ty)) {
        // Shortcut if this is an array of memory elements.
        auto cls = ClassifyEightbytes(t, a->elem());
        if (cls.front() == MEMORY) return {MEMORY};

        // If this is a one-element array, it collapses to the containing class.
        if (a->dimension() == 1) return cls;

        // Because of how arrays work, there is no way that this isn’t passed in memory
        // if it’s larger than 128 bytes (the only objects larger than that that can be
        // passed in registers are YMM and ZMM registers; specifically the requirement
        // is that only the first eightbyte has class SSE and all other eightbytes class
        // SSEUP; since the nature of an array means that we will have multiple eightbytes
        // with class SSE, this rule does not apply here).
        //
        // Note: This check is technically redundant at the moment, but it won’t be once
        // we actually support e.g. __m256 because then the same check further up will have
        // to be removed.
        if (sz > Eightbyte * 2) return {MEMORY};

        // The only other option therefore is to split the array into word-sized chunks.
        if (sz > Eightbyte) return {INTEGER, INTEGER};
        return {INTEGER};
    }

    // For structs, we’re supposed to classify the fields recursively.
    //
    // However, this step, too, can be short-circuited because we don’t
    // support floats yet. (In conclusion, I’m *really* not looking forward
    // to adding support for floats—at least not YMM/XMM/ZMM registers...)
    if (sz > Eightbyte) return {INTEGER, INTEGER};
    return {INTEGER};
}

auto Impl::get_call_builder(
    CodeGen& cg,
    ProcType *ty,
    Value indirect_ptr
) const -> std::unique_ptr<CallBuilder> {
    return std::make_unique<Builder>(cg, ty, indirect_ptr);
}

bool Impl::pass_in_parameter_by_value(Type ty) const {
    return ty->size(*this) <= Eightbyte * 2;
}

Builder::Builder(CodeGen& cg, ProcType* ty, Value indirect_ptr)
    : CallBuilder{cg}, proc_ty{ty}, indirect_ptr{indirect_ptr} {
    auto ret = ty->ret();
    if (target().needs_indirect_return(ret)) {
        Assert(indirect_ptr);
        add_pointer(indirect_ptr);
        indirect = true;
    }

    ret_class = ClassifyEightbytes(target(), ret);
}

void Builder::add(Value v, mlir::DictionaryAttr a) {
    args.push_back(v);
    args_attrs.push_back(a);
}

void Builder::add_argument(Type param_ty, Value v, bool require_copy) {
    Assert(not cg.IsZeroSizedType(param_ty));
    auto conv = cg.C(param_ty).scalar();

    // 'bool' needs special handling because it has to be zero-extended
    // rather than sign-extended like 'i1'.
    if (param_ty == Type::BoolTy) {
        auto zext = cg.getNamedAttr(mlir::LLVM::LLVMDialect::getZExtAttrName(), mlir::UnitAttr::get(cg.mlir_context()));
        add(v, mlir::DictionaryAttr::get(cg.mlir_context(), {zext}));
        return;
    }

    if (isa<mlir::LLVM::LLVMPointerType, mlir::IntegerType>(conv)) {
        add_pointer_or_integer(v);
        return;
    }

    Assert(isa<mlir::LLVM::LLVMPointerType>(v.getType()), "Should have been an lvalue");
    auto cls = ClassifyEightbytes(target(), param_ty);
    if (cls.front() == MEMORY) {
        add_copy(param_ty, v, require_copy);
        return;
    }

    Assert(all_of(cls, [](Class c) { return c == INTEGER; }));
    if (cls.size() + num_regs > MaxRegs) {
        add_copy(param_ty, v, require_copy);
        return;
    }

    auto Load = [&](mlir::Type t) {
        return cg.create<ir::LoadOp>(v.getLoc(), t, v, Align(1));
    };

    num_regs += cls.size();
    Assert(cls.size() <= 2);
    Size remaining = param_ty->size(target());
    for (;;) {
        if (remaining <= Eightbyte) {
            add(Load(cg.IntTy(remaining)));
            break;
        }

        add(Load(cg.getI64Type()));
        remaining -= Eightbyte;
        v = cg.CreatePtrAdd(v.getLoc(), v, Eightbyte);
    }
}

void Builder::add_byval_arg(mlir::Type ty, Value ptr) {
    auto byval = cg.getNamedAttr(mlir::LLVM::LLVMDialect::getByValAttrName(), mlir::TypeAttr::get(ty));
    add(ptr, mlir::DictionaryAttr::get(cg.mlir_context(), {byval}));
}

void Builder::add_copy(Type ty, Value ptr, bool require_copy) {
    Assert(isa<mlir::LLVM::LLVMPointerType>(ptr.getType()), "Expected lvalue");
    if (require_copy) {
        auto tmp = cg.CreateAlloca(ptr.getLoc(), ty);
        cg.create<mlir::LLVM::MemcpyOp>(
            ptr.getLoc(),
            tmp,
            ptr,
            cg.CreateInt(ptr.getLoc(), i64(ty->size(target()).bytes())),
            false
        );
        ptr = tmp;
    }

    add_byval_arg(cg.C(ty).scalar(), ptr);
}


void Builder::add_pointer(Value v) {
    // No need to add 'byval' or anything like these; LLVM will just throw them
    // into stack slots if we’re out of registers either way.
    add(v);
    num_regs++;
}

void Builder::add_pointer_or_integer(Value v) {
    auto ty = v.getType();
    if (isa<mlir::LLVM::LLVMPointerType>(ty)) {
        add_pointer(v);
        return;
    }

    // Anything >128 bits is passed in memory.
    auto wd = Size::Bits(cast<mlir::IntegerType>(ty).getWidth());
    if (wd > Eightbyte * 2) {
        auto tmp = cg.CreateAlloca(v.getLoc(), wd.align(Align(Eightbyte)), Align(Eightbyte * 2));
        cg.CreateStore(v.getLoc(), tmp, v, Align(Eightbyte * 2));
        add_byval_arg(ty, tmp);
        return;
    }

    // Integers with wonky bit widths are sign extended.
    mlir::DictionaryAttr attr;
    if (wd < Size::Bits(8) or not wd.is_power_of_2()) {
        auto sext = cg.getNamedAttr(mlir::LLVM::LLVMDialect::getSExtAttrName(), mlir::UnitAttr::get(cg.mlir_context()));
        attr = mlir::DictionaryAttr::get(cg.mlir_context(), {sext});
    }

    // Don’t increment the register count if we’re out of registers since
    // an i128 is passed in memory in that case.
    u32 regs_required = wd > Eightbyte ? 2 : 1;
    if (num_regs + regs_required > MaxRegs) num_regs += regs_required;
    add(v, attr);
}

auto Builder::get_arg_attrs() -> mlir::ArrayAttr {
    return mlir::ArrayAttr::get(cg.mlir_context(), args_attrs);
}

auto Builder::get_final_args() -> ArrayRef<Value> {
    return args;
}

auto Builder::get_proc_type() -> mlir::FunctionType {
    SmallVector<mlir::Type> arg_types;
    for (auto a : args) arg_types.push_back(a.getType());
    auto ret = get_result_types();
    return mlir::FunctionType::get(cg.mlir_context(), arg_types, mlir::TypeRange{ret});
}

auto Builder::get_result_types() -> SmallVector<mlir::Type, 2> {
    if (indirect or cg.IsZeroSizedType(proc_ty->ret())) return {};

    // Classify the return type.
    auto ret_cls = ClassifyEightbytes(target(), proc_ty->ret());
    Assert(all_of(ret_cls, [](Class c) { return c == INTEGER; }));

    // And split it into word-sized chunks.
    SmallVector<mlir::Type, 2> chunks;
    Size remaining = proc_ty->ret()->size(target());
    for (;;) {
        if (remaining <= Eightbyte) {
            chunks.push_back(cg.IntTy(remaining));
            return chunks;
        }

        chunks.push_back(cg.getI64Type());
        remaining -= Eightbyte;
    }
}

auto Builder::get_result_vals(ir::CallOp call) -> SRValue {
    if (indirect) return {};

    // If this is an mrvalue, write it to memory.
    auto ret = proc_ty->ret();
    if (not target().is_returned_as_mrvalue(proc_ty->ret())) return call.getResults();

    // Write the value to memory.
    Assert(indirect_ptr);
    cg.CreateStore(call.getLoc(), indirect_ptr, call.getResult(0), ret->align(target()));
    if (call.getNumResults() == 2) {
        Assert(call.getResult(0).getType().isInteger(64));
        auto rest = cg.CreatePtrAdd(call.getLoc(), indirect_ptr, Eightbyte);
        cg.CreateStore(call.getLoc(), rest, call.getResult(1), Align(Eightbyte));
    }

    return {};
}

bool Impl::is_returned_as_mrvalue(Type ty) const {
    if (isa<SliceType, ProcType, PtrType>(ty)) return false;
    if (ty == Type::IntTy or ty == Type::BoolTy) return false;
    if (auto i = dyn_cast<IntType>(ty)) return i->bit_width() > Eightbyte;
    return true;
}

bool Impl::needs_indirect_return(Type ty) const {
    if (ty->size(*this) == Size()) return false;
    auto ret_cls = ClassifyEightbytes(*this, ty);
    return ret_cls.front() == MEMORY;
}

auto target::CreateX86_64_Linux(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target> {
    return std::make_unique<Impl>(std::move(TI));
}
