#ifndef SRCC_IR_ENUMS_TD
#define SRCC_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "IRDialect.td"

def SRCC_AbortReason_AssertionFailed : I32EnumCase<"AssertionFailed", 0, "assert">;
def SRCC_AbortReason_ArithmeticError : I32EnumCase<"ArithmeticError", 1, "arithmetic">;
def SRCC_AbortReason_InvalidLocalRef : I32EnumCase<"InvalidLocalRef", 2, "invalid_local">;
def SRCC_AbortReason_Enum : I32Enum<"AbortReason", "Abort reason",
    [
        SRCC_AbortReason_AssertionFailed,
        SRCC_AbortReason_ArithmeticError,
        SRCC_AbortReason_InvalidLocalRef
    ]> {
    let cppNamespace = "::srcc::cg::ir";
}

#endif
