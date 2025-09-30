#include <srcc/CG/ABI.hh>
#include <srcc/CG/CodeGen.hh>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

using namespace srcc;
using namespace srcc::cg;
using LLVM = mlir::LLVM::LLVMDialect;

void abi::Arg::add_byval(mlir::Type ty) {
    attrs.push_back(mlir::NamedAttribute(
        LLVM::getByValAttrName(),
        mlir::TypeAttr::get(ty)
    ));
}

void abi::Arg::add_sext(CodeGen& cg) {
    attrs.push_back(mlir::NamedAttribute(
        LLVM::getSExtAttrName(),
        mlir::UnitAttr::get(cg.mlir_context())
    ));
}

void abi::Arg::add_sret(mlir::Type ty) {
    attrs.push_back(mlir::NamedAttribute(
        LLVM::getStructRetAttrName(),
        mlir::TypeAttr::get(ty)
    ));
}

void abi::Arg::add_zext(CodeGen& cg) {
    attrs.push_back(mlir::NamedAttribute(
        LLVM::getZExtAttrName(),
        mlir::UnitAttr::get(cg.mlir_context())
    ));
}

auto abi::IRToSourceConversionContext::addr() -> Value {
    if (not indirect_ptr) indirect_ptr = codegen.CreateAlloca(loc, ty);
    return indirect_ptr;
}