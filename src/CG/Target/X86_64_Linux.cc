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


[[maybe_unused]] constexpr u32 MaxRegs = 6;
[[maybe_unused]] constexpr Size Eightbyte = Size::Bits(64);

using enum Class;
using Classification = SmallVector<Class, 8>;

struct Impl final : Target {
    Impl(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) : Target(std::move(TI)) {}
};
}

[[maybe_unused]] static auto ClassifyEightbytes(const Target& t, Type ty) -> Classification {
    // If the size of an object is larger than eight eightbytes [...] it has class MEMORY'
    //
    // Additionally, the following rule applies aggregate that isn’t an XMM/YMM/ZMM register:
    // 'If the size of the aggregate exceeds two eightbytes [...] the whole argument is passed
    // in memory.'
    //
    // We don’t have floats at the moment, so short-circuit that here.
    auto sz = ty->memory_size(t);
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

auto target::CreateX86_64_Linux(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target> {
    return std::make_unique<Impl>(std::move(TI));
}
