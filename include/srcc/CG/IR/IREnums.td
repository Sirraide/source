#ifndef SRCC_IR_ENUMS_TD
#define SRCC_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "IRDialect.td"

def SRCC_AbortReason_AssertionFailed : I8EnumCase<"AssertionFailed", 0, "assert">;
def SRCC_AbortReason_ArithmeticError : I8EnumCase<"ArithmeticError", 1, "arithmetic">;
def SRCC_AbortReason_InvalidLocalRef : I8EnumCase<"InvalidLocalRef", 2, "invalid_local">;
def SRCC_AbortReason_Enum : I8Enum<"AbortReason", "Abort reason", [
    SRCC_AbortReason_AssertionFailed,
    SRCC_AbortReason_ArithmeticError,
    SRCC_AbortReason_InvalidLocalRef
]> {
    let cppNamespace = "::srcc::cg::ir";
}

def SRCC_Aliasing_None : I8EnumCase<"None", 0, "base">;
def SRCC_Aliasing_Derived : I8EnumCase<"Derived", 1, "derived">;
def SRCC_Aliasing_Sibling : I8EnumCase<"Sibling", 2, "sibling">;
def SRCC_Aliasing_Enum : I8Enum<"Aliasing", "aliasing", [
    SRCC_Aliasing_None,
    SRCC_Aliasing_Derived,
    SRCC_Aliasing_Sibling
]> {
    let cppNamespace = "::srcc::cg::ir";
}

#endif
