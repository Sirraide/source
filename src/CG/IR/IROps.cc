#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include <srcc/CG/IR/IR.hh>

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

// ============================================================================
//  Printers
// ============================================================================
void AbortOp::print(mlir::OpAsmPrinter& p) {
    p << " " << stringifyAbortReason(getReasonAttr().getValue())
      << " (" << getMsg1()
      << ", " << getMsg2()
      << ")";
}

void AllocaOp::print(mlir::OpAsmPrinter& p) {
    p << " " << getBytes().getInt()
      << ", align " << getAlignment().getInt();
}

void LoadOp::print(mlir::OpAsmPrinter& p) {
    p << " " << getAddr()
      << ", " << getRes().getType()
      << ", align " << getAlignment().getInt();
}

void ProcOp::print(mlir::OpAsmPrinter& p) {
    p << " " << stringifyLinkage(getLinkage().getLinkage())
      << " " << getCc()
      << " " << getName()
      << " (" << getArgumentTypes()
      << ") -> " << getResultTypes()
      << " ";

    if (not getBody().empty())
        p.printRegion(getBody(), false);
}

void ProcRefOp::print(mlir::OpAsmPrinter& p) {
    p << " " << getProcName();
}

void RetOp::print(mlir::OpAsmPrinter& p) {
    if (getValue())
        p << " " << getValue().getType() << " " << getValue();
}

void SelectOp::print(mlir::OpAsmPrinter& p) {
    p << " " << getType()
      << ", " << getCond()
      << ", " << getThenVal()
      << ", " << getElseVal();
}

void SAddOvOp::print(mlir::OpAsmPrinter& p) { p << " " << getLhs().getType() << " " << getLhs() << ", " << getRhs(); }
void SMulOvOp::print(mlir::OpAsmPrinter& p) { p << " " << getLhs().getType() << " " << getLhs() << ", " << getRhs(); }
void SSubOvOp::print(mlir::OpAsmPrinter& p) { p << " " << getLhs().getType() << " " << getLhs() << ", " << getRhs(); }

void StoreOp::print(mlir::OpAsmPrinter& p) {
    p << " " << getAddr()
      << ", " << getValue().getType() << " " << getValue()
      << ", align " << getAlignment().getInt();
}

void TupleOp::print(mlir::OpAsmPrinter& p) {
    p << " (";
    bool first = true;
    for (auto v : getValues()) {
        if (first) first = false;
        else p << ", ";
        p << v.getType() << " " << v;
    }
    p << ")";
}

// ============================================================================
//  Parsers
// ============================================================================
mlir::ParseResult AllocaOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult AbortOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult LoadOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult ProcOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult ProcRefOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult RetOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult SelectOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult SAddOvOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult SMulOvOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult SSubOvOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult StoreOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }
mlir::ParseResult TupleOp::parse(mlir::OpAsmParser&, mlir::OperationState&) { Todo(); }

// ============================================================================
//  Folders
// ============================================================================
static auto FoldOv(
    mlir::Attribute lhs_val,
    mlir::Attribute rhs_val,
    SmallVectorImpl<mlir::OpFoldResult>& results,
    APInt (APInt::*folder)(const APInt &RHS, bool &Overflow) const
) -> mlir::LogicalResult {
    auto lhs = dyn_cast_if_present<mlir::IntegerAttr>(lhs_val);
    auto rhs = dyn_cast_if_present<mlir::IntegerAttr>(rhs_val);
    if (not lhs or not rhs) return mlir::failure();
    bool overflow = false;
    mlir::OpBuilder b(lhs.getContext());
    results.push_back(b.getIntegerAttr(lhs.getType(), (lhs.getValue().*folder)(rhs.getValue(), overflow)));
    results.push_back(b.getBoolAttr(overflow));
    return mlir::success();
}

auto ExtractOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
    auto op = getTuple().getDefiningOp<TupleOp>();
    if (not op) return nullptr;
    return op.getValues()[u32(adaptor.getIndex().getInt())];
}

auto SelectOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
    auto c = dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getCond());
    if (not c) return nullptr;
    return c.getInt() ? adaptor.getThenVal() : adaptor.getElseVal();
}

auto SAddOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::sadd_ov);
}

auto SMulOvOp::fold(FoldAdaptor adaptor, llvm::SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::smul_ov);
}

auto SSubOvOp::fold(FoldAdaptor adaptor, llvm::SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::ssub_ov);
}
