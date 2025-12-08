#ifndef SRCC_CG_IR_HH
#define SRCC_CG_IR_HH

// Include order matters here.
// clang-format off
#include <srcc/Core/Core.hh>
#include <srcc/AST/Type.hh>
#include <srcc/AST/Stmt.hh>

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshorten-64-to-32"

#include <srcc/CG/IR/IRDialect.hh.inc>
#include <srcc/CG/IR/IREnums.hh.inc>
#include <srcc/CG/IR/IRInterfaces.hh.inc>

#define GET_ATTRDEF_CLASSES
#include <srcc/CG/IR/IRAttrs.hh.inc>

#define GET_TYPEDEF_CLASSES
#include <srcc/CG/IR/IRTypes.hh.inc>

#define GET_OP_CLASSES
#include <srcc/CG/IR/IROps.hh.inc>

#pragma clang diagnostic pop

// clang-format on

#define SRCC_ENUMS_EXPOSED_TO_MLIR(X) \
    X(srcc::BuiltinMemberAccessExpr::AccessKind, BuiltinMemberAccessKindAttr)

namespace srcc::cg {
using mlir::Block;
using mlir::Operation;
using mlir::Value;

namespace ir {
auto FormatType(mlir::Type ty) -> SmallString<128>;

#define COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(TYPE, ...)                                         \
    mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader& reader, TYPE& storage); \
    void writeToMlirBytecode(mlir::DialectBytecodeWriter& reader, TYPE storage);                  \
    mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, TYPE storage);                     \
    mlir::LogicalResult convertFromAttribute(                                                     \
        TYPE& storage,                                                                            \
        mlir::Attribute attr,                                                                     \
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError                                  \
    );

COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(srcc::TreeValue*);
COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(srcc::Stmt*);
COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(srcc::Type);
SRCC_ENUMS_EXPOSED_TO_MLIR(COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE);

#undef COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE
}
}

template <typename Ty>
requires requires (const Ty& t, llvm::raw_string_ostream os) {
    os << t;
}
struct libassert::stringifier<Ty> {
    auto stringify(const Ty& val) -> std::string {
        std::string s;
        llvm::raw_string_ostream os{s};
        os << val;
        return s;
    }
};

template <>
struct libassert::stringifier<mlir::Type> {
    auto stringify(mlir::Type ty) -> std::string {
        return base::text::RenderColours(false, srcc::cg::ir::FormatType(ty).str());
    }
};

#endif // SRCC_CG_IR_HH
