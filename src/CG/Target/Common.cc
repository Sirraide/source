#include <srcc/CG/CodeGen.hh>

using namespace srcc;
using namespace srcc::cg;

/*
auto CallLowering::AssembleTypeFromRegisters(
    Type ty,
    StructLayout* register_layout,
    ArrayRef<ir::Value*> register_vals
) -> ir::Value* {
    // For things whose layout trivially matches the register representation,
    // just reuse the values directly.
    if (isa<ProcType, SliceType>(ty)) return CG.CreateAggregate(
        ty,
        CG.GetAggregateLayout(ty),
        register_vals
    );

    // Otherwise, convert the object in memory.
    auto temp = CG.CreateAlloca(
        CG.curr_proc,
        register_layout->size(),
        std::max(register_layout->align(), ty->align(CG.tu))
    );

    CG.StoreAggregate(temp, register_layout, register_vals);
    return CG.CreateLoad(ty, temp);
}
*/
