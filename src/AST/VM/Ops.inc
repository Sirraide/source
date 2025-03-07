#ifdef SRCC_VM_OP_ENUMERATORS
AShrAPInt,
AShrI8,
AShrI16,
AShrI32,
AShrI64,
Abort,
AddAPInt,
AddI8,
AddI16,
AddI32,
AddI64,
AndAPInt,
AndI8,
AndI16,
AndI32,
AndI64,
Branch,
CondBranch,
DirectCall,
EqAPInt,
EqI8,
EqI16,
EqI32,
EqI64,
LShrAPInt,
LShrI8,
LShrI16,
LShrI32,
LShrI64,
LoadAPInt,
LoadBool,
LoadClosure,
LoadI8,
LoadI16,
LoadI32,
LoadI64,
LoadPointer,
LoadSlice,
LoadType,
MemZero,
MulAPInt,
MulI8,
MulI16,
MulI32,
MulI64,
NeAPInt,
NeI8,
NeI16,
NeI32,
NeI64,
OrAPInt,
OrI8,
OrI16,
OrI32,
OrI64,
PtrAdd,
RetAPInt,
RetBool,
RetClosure,
RetI8,
RetI16,
RetI32,
RetI64,
RetPointer,
RetSlice,
RetType,
RetVoid,
SAddOvAPInt,
SAddOvI8,
SAddOvI16,
SAddOvI32,
SAddOvI64,
SDivAPInt,
SDivI8,
SDivI16,
SDivI32,
SDivI64,
SExtAPIntToAPInt,
SExtAPIntToI8,
SExtAPIntToI16,
SExtAPIntToI32,
SExtAPIntToI64,
SExtI8ToAPInt,
SExtI8ToI16,
SExtI8ToI32,
SExtI8ToI64,
SExtI16ToAPInt,
SExtI16ToI32,
SExtI16ToI64,
SExtI32ToAPInt,
SExtI32ToI64,
SExtI64ToAPInt,
SGeAPInt,
SGeI8,
SGeI16,
SGeI32,
SGeI64,
SGtAPInt,
SGtI8,
SGtI16,
SGtI32,
SGtI64,
SLeAPInt,
SLeI8,
SLeI16,
SLeI32,
SLeI64,
SLtAPInt,
SLtI8,
SLtI16,
SLtI32,
SLtI64,
SMulOvAPInt,
SMulOvI8,
SMulOvI16,
SMulOvI32,
SMulOvI64,
SRemAPInt,
SRemI8,
SRemI16,
SRemI32,
SRemI64,
SSubOvAPInt,
SSubOvI8,
SSubOvI16,
SSubOvI32,
SSubOvI64,
SelectAPInt,
SelectBool,
SelectClosure,
SelectI8,
SelectI16,
SelectI32,
SelectI64,
SelectPointer,
SelectSlice,
SelectType,
ShlAPInt,
ShlI8,
ShlI16,
ShlI32,
ShlI64,
StoreAPInt,
StoreBool,
StoreClosure,
StoreI8,
StoreI16,
StoreI32,
StoreI64,
StorePointer,
StoreSlice,
StoreType,
SubAPInt,
SubI8,
SubI16,
SubI32,
SubI64,
TruncAPIntToAPInt,
TruncAPIntToI8,
TruncAPIntToI16,
TruncAPIntToI32,
TruncAPIntToI64,
TruncI8ToAPInt,
TruncI16ToAPInt,
TruncI16ToI8,
TruncI32ToAPInt,
TruncI32ToI8,
TruncI32ToI16,
TruncI64ToAPInt,
TruncI64ToI8,
TruncI64ToI16,
TruncI64ToI32,
UDivAPInt,
UDivI8,
UDivI16,
UDivI32,
UDivI64,
UGeAPInt,
UGeI8,
UGeI16,
UGeI32,
UGeI64,
UGtAPInt,
UGtI8,
UGtI16,
UGtI32,
UGtI64,
ULeAPInt,
ULeI8,
ULeI16,
ULeI32,
ULeI64,
ULtAPInt,
ULtI8,
ULtI16,
ULtI32,
ULtI64,
URemAPInt,
URemI8,
URemI16,
URemI32,
URemI64,
Unreachable,
UnresolvedCall,
XorAPInt,
XorI8,
XorI16,
XorI32,
XorI64,
ZExtAPIntToAPInt,
ZExtAPIntToI8,
ZExtAPIntToI16,
ZExtAPIntToI32,
ZExtAPIntToI64,
ZExtI8ToAPInt,
ZExtI8ToI16,
ZExtI8ToI32,
ZExtI8ToI64,
ZExtI16ToAPInt,
ZExtI16ToI32,
ZExtI16ToI64,
ZExtI32ToAPInt,
ZExtI32ToI64,
ZExtI64ToAPInt,
#undef SRCC_VM_OP_ENUMERATORS
#endif

#ifdef SRCC_VM_OP_BUILDERS

void CreateAShr(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::AShrI8; break;
        case 16: code << Op::AShrI16; break;
        case 32: code << Op::AShrI32; break;
        case 64: code << Op::AShrI64; break;
        default: code << Op::AShrAPInt; break;
    }
    EmitOperands(args);
}

void CreateAdd(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::AddI8; break;
        case 16: code << Op::AddI16; break;
        case 32: code << Op::AddI32; break;
        case 64: code << Op::AddI64; break;
        default: code << Op::AddAPInt; break;
    }
    EmitOperands(args);
}

void CreateAnd(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::AndI8; break;
        case 16: code << Op::AndI16; break;
        case 32: code << Op::AndI32; break;
        case 64: code << Op::AndI64; break;
        default: code << Op::AndAPInt; break;
    }
    EmitOperands(args);
}

void CreateEq(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::EqI8; break;
        case 16: code << Op::EqI16; break;
        case 32: code << Op::EqI32; break;
        case 64: code << Op::EqI64; break;
        default: code << Op::EqAPInt; break;
    }
    EmitOperands(args);
}

void CreateLShr(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::LShrI8; break;
        case 16: code << Op::LShrI16; break;
        case 32: code << Op::LShrI32; break;
        case 64: code << Op::LShrI64; break;
        default: code << Op::LShrAPInt; break;
    }
    EmitOperands(args);
}

void CreateMul(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::MulI8; break;
        case 16: code << Op::MulI16; break;
        case 32: code << Op::MulI32; break;
        case 64: code << Op::MulI64; break;
        default: code << Op::MulAPInt; break;
    }
    EmitOperands(args);
}

void CreateNe(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::NeI8; break;
        case 16: code << Op::NeI16; break;
        case 32: code << Op::NeI32; break;
        case 64: code << Op::NeI64; break;
        default: code << Op::NeAPInt; break;
    }
    EmitOperands(args);
}

void CreateOr(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::OrI8; break;
        case 16: code << Op::OrI16; break;
        case 32: code << Op::OrI32; break;
        case 64: code << Op::OrI64; break;
        default: code << Op::OrAPInt; break;
    }
    EmitOperands(args);
}

void CreateSAddOv(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SAddOvI8; break;
        case 16: code << Op::SAddOvI16; break;
        case 32: code << Op::SAddOvI32; break;
        case 64: code << Op::SAddOvI64; break;
        default: code << Op::SAddOvAPInt; break;
    }
    EmitOperands(args);
}

void CreateSDiv(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SDivI8; break;
        case 16: code << Op::SDivI16; break;
        case 32: code << Op::SDivI32; break;
        case 64: code << Op::SDivI64; break;
        default: code << Op::SDivAPInt; break;
    }
    EmitOperands(args);
}

void CreateSGe(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SGeI8; break;
        case 16: code << Op::SGeI16; break;
        case 32: code << Op::SGeI32; break;
        case 64: code << Op::SGeI64; break;
        default: code << Op::SGeAPInt; break;
    }
    EmitOperands(args);
}

void CreateSGt(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SGtI8; break;
        case 16: code << Op::SGtI16; break;
        case 32: code << Op::SGtI32; break;
        case 64: code << Op::SGtI64; break;
        default: code << Op::SGtAPInt; break;
    }
    EmitOperands(args);
}

void CreateSLe(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SLeI8; break;
        case 16: code << Op::SLeI16; break;
        case 32: code << Op::SLeI32; break;
        case 64: code << Op::SLeI64; break;
        default: code << Op::SLeAPInt; break;
    }
    EmitOperands(args);
}

void CreateSLt(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SLtI8; break;
        case 16: code << Op::SLtI16; break;
        case 32: code << Op::SLtI32; break;
        case 64: code << Op::SLtI64; break;
        default: code << Op::SLtAPInt; break;
    }
    EmitOperands(args);
}

void CreateSMulOv(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SMulOvI8; break;
        case 16: code << Op::SMulOvI16; break;
        case 32: code << Op::SMulOvI32; break;
        case 64: code << Op::SMulOvI64; break;
        default: code << Op::SMulOvAPInt; break;
    }
    EmitOperands(args);
}

void CreateSRem(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SRemI8; break;
        case 16: code << Op::SRemI16; break;
        case 32: code << Op::SRemI32; break;
        case 64: code << Op::SRemI64; break;
        default: code << Op::SRemAPInt; break;
    }
    EmitOperands(args);
}

void CreateSSubOv(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SSubOvI8; break;
        case 16: code << Op::SSubOvI16; break;
        case 32: code << Op::SSubOvI32; break;
        case 64: code << Op::SSubOvI64; break;
        default: code << Op::SSubOvAPInt; break;
    }
    EmitOperands(args);
}

void CreateShl(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::ShlI8; break;
        case 16: code << Op::ShlI16; break;
        case 32: code << Op::ShlI32; break;
        case 64: code << Op::ShlI64; break;
        default: code << Op::ShlAPInt; break;
    }
    EmitOperands(args);
}

void CreateSub(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::SubI8; break;
        case 16: code << Op::SubI16; break;
        case 32: code << Op::SubI32; break;
        case 64: code << Op::SubI64; break;
        default: code << Op::SubAPInt; break;
    }
    EmitOperands(args);
}

void CreateUDiv(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::UDivI8; break;
        case 16: code << Op::UDivI16; break;
        case 32: code << Op::UDivI32; break;
        case 64: code << Op::UDivI64; break;
        default: code << Op::UDivAPInt; break;
    }
    EmitOperands(args);
}

void CreateUGe(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::UGeI8; break;
        case 16: code << Op::UGeI16; break;
        case 32: code << Op::UGeI32; break;
        case 64: code << Op::UGeI64; break;
        default: code << Op::UGeAPInt; break;
    }
    EmitOperands(args);
}

void CreateUGt(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::UGtI8; break;
        case 16: code << Op::UGtI16; break;
        case 32: code << Op::UGtI32; break;
        case 64: code << Op::UGtI64; break;
        default: code << Op::UGtAPInt; break;
    }
    EmitOperands(args);
}

void CreateULe(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::ULeI8; break;
        case 16: code << Op::ULeI16; break;
        case 32: code << Op::ULeI32; break;
        case 64: code << Op::ULeI64; break;
        default: code << Op::ULeAPInt; break;
    }
    EmitOperands(args);
}

void CreateULt(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::ULtI8; break;
        case 16: code << Op::ULtI16; break;
        case 32: code << Op::ULtI32; break;
        case 64: code << Op::ULtI64; break;
        default: code << Op::ULtAPInt; break;
    }
    EmitOperands(args);
}

void CreateURem(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::URemI8; break;
        case 16: code << Op::URemI16; break;
        case 32: code << Op::URemI32; break;
        case 64: code << Op::URemI64; break;
        default: code << Op::URemAPInt; break;
    }
    EmitOperands(args);
}

void CreateXor(ArrayRef<ir::Value*> args) {
    switch (args[0]->type()->size(c.vm.owner()).bits()) {
        case 8: code << Op::XorI8; break;
        case 16: code << Op::XorI16; break;
        case 32: code << Op::XorI32; break;
        case 64: code << Op::XorI64; break;
        default: code << Op::XorAPInt; break;
    }
    EmitOperands(args);
}
#undef SRCC_VM_OP_BUILDERS
#endif

#ifdef SRCC_VM_OP_PRINTERS
case Op::AShrAPInt:
    P("%1(ashr.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AShrI8:
    P("%1(ashr.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AShrI16:
    P("%1(ashr.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AShrI32:
    P("%1(ashr.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AShrI64:
    P("%1(ashr.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::Abort:
    P("%1(abort)");
    P(" ");
    PrintImmAbortReason();
    P(", ");
    PrintImmLocation();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AddAPInt:
    P("%1(add.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AddI8:
    P("%1(add.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AddI16:
    P("%1(add.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AddI32:
    P("%1(add.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AddI64:
    P("%1(add.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AndAPInt:
    P("%1(and.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AndI8:
    P("%1(and.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AndI16:
    P("%1(and.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AndI32:
    P("%1(and.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::AndI64:
    P("%1(and.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::Branch:
    P("%1(branch)");
    P(" ");
    PrintImmBlock();
    break;
case Op::CondBranch:
    P("%1(condbranch)");
    P(" ");
    PrintImmBlock();
    P(", ");
    PrintImmBlock();
    P(", ");
    PrintOperand();
    break;
case Op::DirectCall:
    P("%1(directcall)");
    break;
case Op::EqAPInt:
    P("%1(eq.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::EqI8:
    P("%1(eq.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::EqI16:
    P("%1(eq.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::EqI32:
    P("%1(eq.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::EqI64:
    P("%1(eq.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LShrAPInt:
    P("%1(lshr.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LShrI8:
    P("%1(lshr.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LShrI16:
    P("%1(lshr.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LShrI32:
    P("%1(lshr.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LShrI64:
    P("%1(lshr.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::LoadAPInt:
    P("%1(load.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadBool:
    P("%1(load.bool)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadClosure:
    P("%1(load.closure)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadI8:
    P("%1(load.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadI16:
    P("%1(load.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadI32:
    P("%1(load.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadI64:
    P("%1(load.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadPointer:
    P("%1(load.pointer)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadSlice:
    P("%1(load.slice)");
    P(" ");
    PrintOperand();
    break;
case Op::LoadType:
    P("%1(load.type)");
    P(" ");
    PrintOperand();
    break;
case Op::MemZero:
    P("%1(memzero)");
    P(" ");
    PrintImmI64();
    P(", ");
    PrintOperand();
    break;
case Op::MulAPInt:
    P("%1(mul.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::MulI8:
    P("%1(mul.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::MulI16:
    P("%1(mul.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::MulI32:
    P("%1(mul.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::MulI64:
    P("%1(mul.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::NeAPInt:
    P("%1(ne.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::NeI8:
    P("%1(ne.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::NeI16:
    P("%1(ne.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::NeI32:
    P("%1(ne.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::NeI64:
    P("%1(ne.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::OrAPInt:
    P("%1(or.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::OrI8:
    P("%1(or.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::OrI16:
    P("%1(or.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::OrI32:
    P("%1(or.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::OrI64:
    P("%1(or.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::PtrAdd:
    P("%1(ptradd)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::RetAPInt:
    P("%1(ret.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::RetBool:
    P("%1(ret.bool)");
    P(" ");
    PrintOperand();
    break;
case Op::RetClosure:
    P("%1(ret.closure)");
    P(" ");
    PrintOperand();
    break;
case Op::RetI8:
    P("%1(ret.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::RetI16:
    P("%1(ret.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::RetI32:
    P("%1(ret.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::RetI64:
    P("%1(ret.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::RetPointer:
    P("%1(ret.pointer)");
    P(" ");
    PrintOperand();
    break;
case Op::RetSlice:
    P("%1(ret.slice)");
    P(" ");
    PrintOperand();
    break;
case Op::RetType:
    P("%1(ret.type)");
    P(" ");
    PrintOperand();
    break;
case Op::RetVoid:
    P("%1(retvoid)");
    break;
case Op::SAddOvAPInt:
    P("%1(saddov.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SAddOvI8:
    P("%1(saddov.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SAddOvI16:
    P("%1(saddov.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SAddOvI32:
    P("%1(saddov.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SAddOvI64:
    P("%1(saddov.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SDivAPInt:
    P("%1(sdiv.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SDivI8:
    P("%1(sdiv.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SDivI16:
    P("%1(sdiv.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SDivI32:
    P("%1(sdiv.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SDivI64:
    P("%1(sdiv.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SExtAPIntToAPInt:
    P("%1(sext.apint.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtAPIntToI8:
    P("%1(sext.apint.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtAPIntToI16:
    P("%1(sext.apint.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtAPIntToI32:
    P("%1(sext.apint.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtAPIntToI64:
    P("%1(sext.apint.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI8ToAPInt:
    P("%1(sext.i8.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI8ToI16:
    P("%1(sext.i8.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI8ToI32:
    P("%1(sext.i8.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI8ToI64:
    P("%1(sext.i8.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI16ToAPInt:
    P("%1(sext.i16.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI16ToI32:
    P("%1(sext.i16.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI16ToI64:
    P("%1(sext.i16.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI32ToAPInt:
    P("%1(sext.i32.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI32ToI64:
    P("%1(sext.i32.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::SExtI64ToAPInt:
    P("%1(sext.i64.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::SGeAPInt:
    P("%1(sge.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGeI8:
    P("%1(sge.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGeI16:
    P("%1(sge.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGeI32:
    P("%1(sge.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGeI64:
    P("%1(sge.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGtAPInt:
    P("%1(sgt.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGtI8:
    P("%1(sgt.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGtI16:
    P("%1(sgt.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGtI32:
    P("%1(sgt.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SGtI64:
    P("%1(sgt.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLeAPInt:
    P("%1(sle.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLeI8:
    P("%1(sle.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLeI16:
    P("%1(sle.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLeI32:
    P("%1(sle.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLeI64:
    P("%1(sle.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLtAPInt:
    P("%1(slt.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLtI8:
    P("%1(slt.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLtI16:
    P("%1(slt.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLtI32:
    P("%1(slt.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SLtI64:
    P("%1(slt.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SMulOvAPInt:
    P("%1(smulov.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SMulOvI8:
    P("%1(smulov.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SMulOvI16:
    P("%1(smulov.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SMulOvI32:
    P("%1(smulov.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SMulOvI64:
    P("%1(smulov.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SRemAPInt:
    P("%1(srem.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SRemI8:
    P("%1(srem.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SRemI16:
    P("%1(srem.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SRemI32:
    P("%1(srem.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SRemI64:
    P("%1(srem.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SSubOvAPInt:
    P("%1(ssubov.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SSubOvI8:
    P("%1(ssubov.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SSubOvI16:
    P("%1(ssubov.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SSubOvI32:
    P("%1(ssubov.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SSubOvI64:
    P("%1(ssubov.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectAPInt:
    P("%1(select.apint)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectBool:
    P("%1(select.bool)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectClosure:
    P("%1(select.closure)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectI8:
    P("%1(select.i8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectI16:
    P("%1(select.i16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectI32:
    P("%1(select.i32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectI64:
    P("%1(select.i64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectPointer:
    P("%1(select.pointer)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectSlice:
    P("%1(select.slice)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SelectType:
    P("%1(select.type)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ShlAPInt:
    P("%1(shl.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ShlI8:
    P("%1(shl.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ShlI16:
    P("%1(shl.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ShlI32:
    P("%1(shl.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ShlI64:
    P("%1(shl.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreAPInt:
    P("%1(store.apint)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreBool:
    P("%1(store.bool)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreClosure:
    P("%1(store.closure)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreI8:
    P("%1(store.i8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreI16:
    P("%1(store.i16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreI32:
    P("%1(store.i32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreI64:
    P("%1(store.i64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StorePointer:
    P("%1(store.pointer)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreSlice:
    P("%1(store.slice)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::StoreType:
    P("%1(store.type)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SubAPInt:
    P("%1(sub.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SubI8:
    P("%1(sub.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SubI16:
    P("%1(sub.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SubI32:
    P("%1(sub.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::SubI64:
    P("%1(sub.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::TruncAPIntToAPInt:
    P("%1(trunc.apint.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncAPIntToI8:
    P("%1(trunc.apint.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncAPIntToI16:
    P("%1(trunc.apint.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncAPIntToI32:
    P("%1(trunc.apint.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncAPIntToI64:
    P("%1(trunc.apint.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI8ToAPInt:
    P("%1(trunc.i8.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI16ToAPInt:
    P("%1(trunc.i16.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI16ToI8:
    P("%1(trunc.i16.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI32ToAPInt:
    P("%1(trunc.i32.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI32ToI8:
    P("%1(trunc.i32.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI32ToI16:
    P("%1(trunc.i32.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI64ToAPInt:
    P("%1(trunc.i64.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI64ToI8:
    P("%1(trunc.i64.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI64ToI16:
    P("%1(trunc.i64.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::TruncI64ToI32:
    P("%1(trunc.i64.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::UDivAPInt:
    P("%1(udiv.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UDivI8:
    P("%1(udiv.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UDivI16:
    P("%1(udiv.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UDivI32:
    P("%1(udiv.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UDivI64:
    P("%1(udiv.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGeAPInt:
    P("%1(uge.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGeI8:
    P("%1(uge.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGeI16:
    P("%1(uge.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGeI32:
    P("%1(uge.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGeI64:
    P("%1(uge.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGtAPInt:
    P("%1(ugt.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGtI8:
    P("%1(ugt.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGtI16:
    P("%1(ugt.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGtI32:
    P("%1(ugt.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::UGtI64:
    P("%1(ugt.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULeAPInt:
    P("%1(ule.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULeI8:
    P("%1(ule.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULeI16:
    P("%1(ule.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULeI32:
    P("%1(ule.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULeI64:
    P("%1(ule.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULtAPInt:
    P("%1(ult.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULtI8:
    P("%1(ult.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULtI16:
    P("%1(ult.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULtI32:
    P("%1(ult.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ULtI64:
    P("%1(ult.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::URemAPInt:
    P("%1(urem.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::URemI8:
    P("%1(urem.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::URemI16:
    P("%1(urem.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::URemI32:
    P("%1(urem.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::URemI64:
    P("%1(urem.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::Unreachable:
    P("%1(unreachable)");
    break;
case Op::UnresolvedCall:
    P("%1(unresolvedcall)");
    break;
case Op::XorAPInt:
    P("%1(xor.ap)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::XorI8:
    P("%1(xor.8)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::XorI16:
    P("%1(xor.16)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::XorI32:
    P("%1(xor.32)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::XorI64:
    P("%1(xor.64)");
    P(" ");
    PrintOperand();
    P(", ");
    PrintOperand();
    break;
case Op::ZExtAPIntToAPInt:
    P("%1(zext.apint.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtAPIntToI8:
    P("%1(zext.apint.i8)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtAPIntToI16:
    P("%1(zext.apint.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtAPIntToI32:
    P("%1(zext.apint.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtAPIntToI64:
    P("%1(zext.apint.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI8ToAPInt:
    P("%1(zext.i8.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI8ToI16:
    P("%1(zext.i8.i16)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI8ToI32:
    P("%1(zext.i8.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI8ToI64:
    P("%1(zext.i8.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI16ToAPInt:
    P("%1(zext.i16.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI16ToI32:
    P("%1(zext.i16.i32)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI16ToI64:
    P("%1(zext.i16.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI32ToAPInt:
    P("%1(zext.i32.apint)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI32ToI64:
    P("%1(zext.i32.i64)");
    P(" ");
    PrintOperand();
    break;
case Op::ZExtI64ToAPInt:
    P("%1(zext.i64.apint)");
    P(" ");
    PrintOperand();
    break;
#undef SRCC_VM_OP_PRINTERS
#endif

