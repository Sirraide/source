#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace srcc;
using namespace srcc::cg;

auto CodeGen::emit_llvm(llvm::TargetMachine&) -> std::unique_ptr<llvm::Module> {
    Todo();
}

bool CodeGen::finalise() {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());
    if (pm.run(mlir_module).failed()) {
        ICE(Location(), "Failed to finalise IR");
        return false;
    }
    return true;
}
