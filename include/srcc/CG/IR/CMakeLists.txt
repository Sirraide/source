set(LLVM_TARGET_DEFINITIONS "IR.td")
set(MLIR_TABLEGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include/srcc/CG/IR/")

mlir_tablegen(IROps.hh.inc -gen-op-decls -dialect=srcc)
mlir_tablegen(IROps.cc.inc -gen-op-defs -dialect=srcc)
mlir_tablegen(IRTypes.hh.inc -gen-typedef-decls -typedefs-dialect=srcc)
mlir_tablegen(IRTypes.cc.inc -gen-typedef-defs -typedefs-dialect=srcc)
mlir_tablegen(IRDialect.hh.inc -gen-dialect-decls -dialect=srcc)
mlir_tablegen(IRDialect.cc.inc -gen-dialect-defs -dialect=srcc)
mlir_tablegen(IRAttrs.hh.inc -gen-attrdef-decls -attrdefs-dialect=srcc)
mlir_tablegen(IRAttrs.cc.inc -gen-attrdef-defs -attrdefs-dialect=srcc)

add_public_tablegen_target(MLIRSourceIncGen)
add_dependencies(mlir-headers MLIRSourceIncGen)

## Enums need to be in a separate file because mlir-tblgen for enums...
## just doesn’t support setting the dialect for some reason (and I am
## too lazy to implement support for it rn.)
set(LLVM_TARGET_DEFINITIONS "IREnums.td")

mlir_tablegen(IREnums.hh.inc -gen-enum-decls)
mlir_tablegen(IREnums.cc.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS "IRInterfaces.td")
mlir_tablegen(IRInterfaces.hh.inc -gen-op-interface-decls -dialect=srcc)
mlir_tablegen(IRInterfaces.cc.inc -gen-op-interface-defs -dialect=srcc)

add_public_tablegen_target(MLIRSourceEnumsIncGen)
add_dependencies(mlir-headers MLIRSourceEnumsIncGen)
