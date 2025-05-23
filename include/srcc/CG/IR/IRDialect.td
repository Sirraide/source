#ifndef SRCC_IR_DIALECT_TD
#define SRCC_IR_DIALECT_TD

include "mlir/IR/DialectBase.td"

def SRCC_Dialect : Dialect {
    let name = "srcc";
    let summary = "Source IR";
    let cppNamespace = "::srcc::cg::ir";
    let description = [{
        IR for the Source compiler.
    }];

    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        static void InitialiseContext(mlir::MLIRContext& ctx);
    }];
}

#endif
