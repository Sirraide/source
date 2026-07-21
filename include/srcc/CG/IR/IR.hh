#ifndef SRCC_CG_IR_HH
#define SRCC_CG_IR_HH

// Include order matters here.
// clang-format off
#include <srcc/Core/Core.hh>
#include <srcc/AST/Type.hh>
#include <srcc/AST/Stmt.hh>

namespace srcc::cg::ir {
// MLIR's property system requires everything to be hashable etc, which is something
// we DON'T want for 'Type'; we also can't use 'TypeBase*' since we disallow equality
// comparisons on it, so instead use this hack.
struct TypeWrapper {
    TypeBase* ty{};
    friend bool operator==(TypeWrapper a, TypeWrapper b) {
        return static_cast<void*>(a.ty) == static_cast<void*>(b.ty);
    }
};

inline auto hash_value(TypeWrapper value) -> llvm::hash_code {
    return llvm::hash_value(static_cast<void*>(value.ty));
}
} // namespace srcc::cg::ir

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
auto GetTypeSize(const mlir::DataLayout& dl, mlir::Type ty) -> Size;

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
COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE(srcc::cg::ir::TypeWrapper);
SRCC_ENUMS_EXPOSED_TO_MLIR(COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE);
#undef COMPILE_TIME_ONLY_PROPERTY_BOILERPLATE

/// Attribute that indicates that the value pointed to by a 'ptr'
/// parameter is undefined upon entry to a procedure; this is used
/// for out parameters.
constexpr String OutParamAttrName = "srcc.out_param";

/// Attribute that indicates that a parameter or argument is a move
/// parameter. In particular, this marks a procedure argument as moved.
constexpr String MoveParamAttrName = "srcc.move";
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
