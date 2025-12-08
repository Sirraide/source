#include <srcc/AST/Stmt.hh>
#include <mlir/Support/LLVM.h>
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include <srcc/CG/IR/IR.hh>

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

auto SRCCDialect::materializeConstant(
    mlir::OpBuilder& builder,
    mlir::Attribute value,
    mlir::Type type,
    mlir::Location loc
) -> Operation* {
    if (auto i = dyn_cast<mlir::IntegerAttr>(value); i and type.isInteger())
        return mlir::arith::ConstantOp::create(builder, loc, type, i);

    if (auto b = dyn_cast<mlir::BoolAttr>(value)) {
        Assert(type.isInteger());
        return mlir::arith::ConstantOp::create(builder, loc, type, b);
    }

    SmallString<128> s;
    llvm::raw_svector_ostream os{s};
    os << value;
    Unreachable("Don’t know how to materialise this attribute: {}", s);
}

#define COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(TYPE, ATTR, MLIRTYPE, DEBUG_STR)                        \
    mlir::Attribute ir::convertToAttribute(mlir::MLIRContext* ctx, TYPE storage) {                     \
        return ir::ATTR::get(ctx, MLIRTYPE, storage);                                                  \
    }                                                                                                  \
                                                                                                       \
    mlir::LogicalResult ir::convertFromAttribute(                                                      \
        TYPE& storage,                                                                                 \
        mlir::Attribute attr,                                                                          \
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError                                       \
    ) {                                                                                                \
        if (not isa<ATTR>(attr)) {                                                                     \
            emitError() << "invalid value for " #TYPE "; expected " #ATTR;                             \
            return mlir::failure();                                                                    \
        }                                                                                              \
                                                                                                       \
        storage = cast<ATTR>(attr).getValue();                                                         \
        return mlir::success();                                                                        \
    }                                                                                                  \
                                                                                                       \
    mlir::LogicalResult ir::readFromMlirBytecode(mlir::DialectBytecodeReader& reader, TYPE& storage) { \
        Todo();                                                                                        \
    }                                                                                                  \
                                                                                                       \
    void ir::writeToMlirBytecode(mlir::DialectBytecodeWriter& reader, TYPE storage) {                  \
        Todo();                                                                                        \
    }                                                                                                  \
                                                                                                       \
    ::mlir::Attribute ir::ATTR::parse(::mlir::AsmParser& odsParser, mlir::Type odsType) {              \
        Todo();                                                                                        \
    }                                                                                                  \
                                                                                                       \
    void ir::ATTR::print(::mlir::AsmPrinter& odsPrinter) const {                                       \
        odsPrinter << DEBUG_STR;                                                                       \
    }

#define EXPOSE_ENUM_PROPERTY(TYPE, ATTR)                                                                 \
    mlir::Attribute ir::convertToAttribute(mlir::MLIRContext* ctx, TYPE storage) {                       \
        return ir::ATTR::get(ctx, mlir::IntegerType::get(ctx, 64), storage);                             \
    }                                                                                                    \
                                                                                                         \
    mlir::LogicalResult ir::convertFromAttribute(                                                        \
        TYPE& storage,                                                                                   \
        mlir::Attribute attr,                                                                            \
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError                                         \
    ) {                                                                                                  \
        if (not isa<mlir::IntegerAttr>(attr)) {                                                          \
            emitError() << "invalid value for " #TYPE "; expected IntegerAttr";                          \
            return mlir::failure();                                                                      \
        }                                                                                                \
                                                                                                         \
        auto value = cast<mlir::IntegerAttr>(attr).getInt();                                             \
        if (value > +TYPE::$$Count) {                                                                    \
            emitError() << "invalid value for " #TYPE ": " << value;                                     \
            return mlir::failure();                                                                      \
        }                                                                                                \
                                                                                                         \
        storage = TYPE(value);                                                                           \
        return mlir::success();                                                                          \
    }                                                                                                    \
                                                                                                         \
    mlir::LogicalResult ir::readFromMlirBytecode(mlir::DialectBytecodeReader& reader, TYPE& storage) {   \
        u64 value;                                                                                       \
        if (failed(reader.readVarInt(value))) return mlir::failure();                                    \
        if (value > +TYPE::$$Count) {                                                                    \
            return mlir::failure();                                                                      \
        }                                                                                                \
        storage = TYPE(value);                                                                           \
        return mlir::success();                                                                          \
    }                                                                                                    \
                                                                                                         \
    void ir::writeToMlirBytecode(mlir::DialectBytecodeWriter& writer, TYPE storage) {                    \
        writer.writeVarInt(u64(storage));                                                                \
    }                                                                                                    \
                                                                                                         \
    ::mlir::Attribute ir::ATTR::parse(::mlir::AsmParser& odsParser, mlir::Type odsType) {                \
        Todo();                                                                                          \
    }                                                                                                    \
                                                                                                         \
    void ir::ATTR::print(::mlir::AsmPrinter& odsPrinter) const {                                         \
        odsPrinter << enchantum::to_string(getValue());                                                  \
    }

COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(TreeValue*, TreeAttr, ir::TreeType::get(ctx), "<tree>")
COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(Stmt*, StmtAttr, ir::TreeType::get(ctx), "<type>")
COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(Type, TypeAttr, ir::TypeType::get(ctx), "<stmt>")
SRCC_ENUMS_EXPOSED_TO_MLIR(EXPOSE_ENUM_PROPERTY)

// ============================================================================
//  Folders
// ============================================================================
static auto FoldOv(
    mlir::Attribute lhs_val,
    mlir::Attribute rhs_val,
    SmallVectorImpl<mlir::OpFoldResult>& results,
    APInt (APInt::*folder)(const APInt& RHS, bool& Overflow) const
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

auto SAddOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::sadd_ov);
}

auto SMulOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::smul_ov);
}

auto SSubOvOp::fold(FoldAdaptor adaptor, SmallVectorImpl<mlir::OpFoldResult>& results) -> llvm::LogicalResult {
    return FoldOv(adaptor.getLhs(), adaptor.getRhs(), results, &APInt::ssub_ov);
}

auto TypeConstantOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
    return ir::TypeAttr::get(getContext(), ir::TypeType::get(getContext()), adaptor.getValue());
}

auto TypeEqOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
    mlir::OpBuilder b{getContext()};
    if (getLhs() == getRhs()) return b.getBoolAttr(true);
    auto lhs = dyn_cast_if_present<ir::TypeAttr>(adaptor.getLhs());
    auto rhs = dyn_cast_if_present<ir::TypeAttr>(adaptor.getRhs());
    if (not lhs or not rhs) return nullptr;
    return b.getBoolAttr(lhs.getValue() == rhs.getValue());
}
