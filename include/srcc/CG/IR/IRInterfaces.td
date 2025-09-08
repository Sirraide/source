include "mlir/IR/Interfaces.td"

def ProcAndCallOpInterface : OpInterface<"ProcAndCallOpInterface"> {
    let cppNamespace = "::srcc::cg::ir";
    let methods = [
        InterfaceMethod<"Returns the number of arguments", "unsigned", "getNumCallArgs">,
        InterfaceMethod<"Returns the n-th argument", "::mlir::Value", "getCallArg", (ins "unsigned":$n)>,
        InterfaceMethod<"Returns the n-th argument type", "::mlir::Type", "getCallArgType", (ins "unsigned":$n)>,
        InterfaceMethod<"Returns the n-th argument's attributes", "::mlir::DictionaryAttr", "getCallArgAttrs", (ins "unsigned":$n)>
    ];
}
