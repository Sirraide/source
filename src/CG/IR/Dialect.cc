#include <mlir/Interfaces/DataLayoutInterfaces.h>
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wconversion"

// Include order matters here.
// clang-format off
#include <srcc/CG/IR/IR.hh>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <srcc/CG/IR/IREnums.cc.inc>
#include <srcc/CG/IR/IRDialect.cc.inc>
#include <srcc/CG/IR/IRInterfaces.cc.inc>

#define GET_ATTRDEF_CLASSES
#include <srcc/CG/IR/IRAttrs.cc.inc>

#define GET_TYPEDEF_CLASSES
#include <srcc/CG/IR/IRTypes.cc.inc>

#define GET_OP_CLASSES
#include <srcc/CG/IR/IROps.cc.inc>

#include <mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h>

void srcc::cg::ir::SRCCDialect::initialize() {
    addAttributes<
#       define GET_ATTRDEF_LIST
#       include <srcc/CG/IR/IRAttrs.cc.inc>
    >();

    addTypes<
#       define GET_TYPEDEF_LIST
#       include <srcc/CG/IR/IRTypes.cc.inc>
    >();

    addOperations<
#       define GET_OP_LIST
#       include <srcc/CG/IR/IROps.cc.inc>
    >();
}

void srcc::cg::ir::SRCCDialect::InitialiseContext(mlir::MLIRContext& ctx) {
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    ctx.printOpOnDiagnostic(true);
    ctx.loadDialect<
        mlir::BuiltinDialect,
        mlir::DLTIDialect,
        mlir::arith::ArithDialect,
        mlir::cf::ControlFlowDialect,
        mlir::LLVM::LLVMDialect,
        SRCCDialect
    >();
}
