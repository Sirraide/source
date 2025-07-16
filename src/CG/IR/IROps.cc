#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include <srcc/CG/IR/IR.hh>

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

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

llvm::LogicalResult CallOp::canonicalize(CallOp op, mlir::PatternRewriter& rewriter) {
    if (op.getEnv() and isa<NilOp>(op.getEnv().getDefiningOp())) {
        rewriter.modifyOpInPlace(op, [&]{ op.getEnvMutable().clear(); });
        return mlir::success();
    }

    return mlir::failure();
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

auto SMulOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::smul_ov);
}

auto SSubOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::ssub_ov);
}
