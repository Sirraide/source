#ifndef SRCC_CG_IR_HH
#define SRCC_CG_IR_HH

// Include order matters here.
// clang-format off
#include <srcc/Core/Core.hh>

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include <srcc/CG/IR/IRDialect.hh.inc>
#include <srcc/CG/IR/IREnums.hh.inc>

#define GET_ATTRDEF_CLASSES
#include <srcc/CG/IR/IREnumAttrs.hh.inc>

#define GET_TYPEDEF_CLASSES
#include <srcc/CG/IR/IRTypes.hh.inc>

#define GET_OP_CLASSES
#include <srcc/CG/IR/IROps.hh.inc>

#pragma clang diagnostic pop

// clang-format on

namespace srcc::cg {
using mlir::Block;
using mlir::Operation;
using mlir::Value;
}

#endif // SRCC_CG_IR_HH
