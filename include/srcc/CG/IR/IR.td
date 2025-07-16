include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

include "IRDialect.td"
include "IREnums.td"

// ============================================================================
//  Dialect
// ============================================================================
class SRCC_Op<string mnemonic, list<Trait> traits = []> :
    Op<SRCC_Dialect, mnemonic, traits>;

class SRCC_Attr<
    string name,
    string attrMnemonic,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Attribute"
> : AttrDef<SRCC_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

class SRCC_Type<string name, string type_mnemonic, list<Trait> traits = []> :
    TypeDef<SRCC_Dialect, name, traits> {
    let mnemonic = type_mnemonic;
}

// ============================================================================
//  Types
// ============================================================================
def SRCC_Void : TypeAlias<NoneType, "void">;
def SRCC_Ptr : TypeAlias<LLVMPointerType, "ptr">;
def SRCC_Tuple : TypeAlias<Builtin_Tuple, "tuple">;
def SRCC_Bool : TypeAlias<UI1, "bool">;

def SRCC_Type : AnyTypeOf<[
    Builtin_Integer,
    FunctionType,
    SRCC_Tuple,
    SRCC_Void,
    SRCC_Ptr,
    SRCC_Bool,
]>;

// ============================================================================
//  Operations
// ============================================================================
def ProcOp : SRCC_Op<"proc", [
    AffineScope, AutomaticAllocationScope, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
    let summary = "Procedure";
    let regions = (region AnyRegion:$body);
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        LinkageAttr:$linkage,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$function_type,
        UnitAttr:$variadic,
        UnitAttr:$has_indirect_return,
        UnitAttr:$has_static_chain,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "::mlir::LLVM::Linkage":$linkage,
            "::mlir::LLVM::CConv":$cc,
            "::mlir::FunctionType":$type,
            "bool":$variadic,
            "bool":$has_indirect_return,
            "bool":$has_static_chain,
            CArg<"ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs
        ), [{
            buildWithEntryBlock($_builder, $_state, name, type, attrs, type.getInputs());
            $_state.addAttribute(getCcAttrName($_state.name), ::mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
            $_state.addAttribute(getLinkageAttrName($_state.name), ::mlir::LLVM::LinkageAttr::get($_state.getContext(), linkage));
            if (variadic) $_state.addAttribute(getVariadicAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
            if (has_indirect_return) $_state.addAttribute(getHasIndirectReturnAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
            if (has_static_chain) $_state.addAttribute(getHasStaticChainAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
        }]>
    ];

    /// Most of this is copied from the func dialect’s FuncOp.
    let extraClassDeclaration = [{
        ::mlir::Block* entry() {
            if (empty()) addEntryBlock();
            return &front();
        }

        u32 getNumImplicitArgs() {
            return getHasIndirectReturn() + getHasStaticChain();
        }

        ::mlir::BlockArgument getDeclaredArg(u32 i) {
            return getArgument(i + getNumImplicitArgs());
        }

        ::mlir::BlockArgument getIndirectReturnPointer() {
            Assert(getHasIndirectReturn());
            return getArgument(0);
        }

        ::mlir::BlockArgument getStaticChainPointer() {
            Assert(getHasStaticChain());
            return getArgument(getHasIndirectReturn() + 1);
        }

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<::mlir::Type> getCallableResults() { return getFunctionType().getResults(); }

        /// Returns the argument attributes for all callable region arguments or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableArgAttrs() {
          return getArgAttrs().value_or(nullptr);
        }

        /// Returns the result attributes for all callable region results or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableResAttrs() {
          return getResAttrs().value_or(nullptr);
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "srcc"; }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];
}

def AbortOp : SRCC_Op<"abort", [Terminator]> {
    let summary = "Report a fatal error and abort the program";
    let arguments = (ins
        SRCC_AbortReasonAttr:$reason,
        SRCC_Tuple:$msg1,
        SRCC_Tuple:$msg2
    );
}

def AllocaOp : SRCC_Op<"alloca"> {
    let summary = "Allocate stack memory";
    let arguments = (ins
        Builtin_IntegerAttr:$bytes,
        Builtin_IntegerAttr:$alignment
    );

    let results = (outs SRCC_Ptr:$res);
    let builders = [
        OpBuilder<(ins
            "Size":$size,
            "Align":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                mlir::LLVM::LLVMPointerType::get($_builder.getContext()),
                $_builder.getI64IntegerAttr(size.bytes()),
                $_builder.getI64IntegerAttr(i64(alignment.value().bytes()))
            );
        }]>
    ];
}

def CallOp : SRCC_Op<"call", [AttrSizedOperandSegments]> {
    let summary = "Call a procedure";
    let arguments = (ins
        SRCC_Ptr:$addr,
        Optional<SRCC_Ptr>:$env,
        Optional<SRCC_Ptr>:$mrvalue_slot,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$proc_type,
        UnitAttr:$variadic,
        Variadic<SRCC_Type>:$args
    );

    let results = (outs Optional<SRCC_Type>:$res);
    let hasCanonicalizeMethod = 1;
    let builders = [
        OpBuilder<(ins
            "ProcRefOp":$ref,
            CArg<"::mlir::ValueRange", "{}">:$args,
            CArg<"::mlir::Value", "{}">:$env,
            CArg<"::mlir::Value", "{}">:$mrvalue_slot
        ), [{
            auto proc = ref.proc();
            build(
                $_builder,
                $_state,
                proc.getResultTypes(),
                ref,
                env,
                mrvalue_slot,
                proc.getCc(),
                proc.getFunctionType(),
                proc.getVariadic(),
                args
            );
        }]>
    ];
}

def ExtractOp : SRCC_Op<"extract", [Pure]> {
    let summary = "Extract a value from a tuple";
    let arguments = (ins SRCC_Tuple:$tuple, Builtin_IntegerAttr:$index);
    let results = (outs SRCC_Type:$res);
    let hasFolder = 1;
    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$tuple,
            "u32":$index
        ), [{
            build(
                $_builder,
                $_state,
                cast<mlir::TupleType>(tuple.getType()).getType(index),
                tuple,
                $_builder.getI64IntegerAttr(i64(index))
            );
        }]>
    ];
}

def LoadOp : SRCC_Op<"load"> {
    let summary = "Load a value from a pointer";
    let arguments = (ins
        SRCC_Ptr:$addr,
        Builtin_IntegerAttr:$alignment
    );

    let results = (outs SRCC_Type:$res);
    let builders = [
        OpBuilder<(ins
            "::mlir::Type":$type,
            "::mlir::Value":$addr,
            "Align":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                type,
                addr,
                $_builder.getI64IntegerAttr(i64(alignment.value().bytes()))
            );
        }]>
    ];
}

def NilOp : SRCC_Op<"nil", [Pure]> {
    let summary = "Create the nil value of a type";
    let results = (outs SRCC_Type:$res);
}

def ProcRefOp : SRCC_Op<"procaddr", [Pure]> {
    let summary = "Reference a procedure";
    let arguments = (ins SymbolNameAttr:$proc_name);
    let results = (outs SRCC_Type:$res);
    let builders = [
        OpBuilder<(ins
            "ProcOp":$proc
        ), [{
            build(
                $_builder,
                $_state,
                mlir::LLVM::LLVMPointerType::get($_builder.getContext()),
                proc.getSymName()
            );
        }]>
    ];

    let extraClassDeclaration = [{
        ProcOp proc() {
            return getOperation()->getParentOfType<mlir::ModuleOp>().lookupSymbol<ProcOp>(getProcName());
        }
    }];
}

def RetOp : SRCC_Op<"ret", [Terminator]> {
    let summary = "Return from a procedure";
    let arguments = (ins Optional<SRCC_Type>:$value);
}

def ReturnPointerOp : SRCC_Op<"retptr", [Pure]> {
    let summary = "Get the return pointer for a procedure";
    let results = (outs SRCC_Ptr:$pointer);
    let builders = [
        OpBuilder<(ins), [{
            build(
                $_builder,
                $_state,
                mlir::LLVM::LLVMPointerType::get($_builder.getContext())
            );
        }]>
    ];
}

def SAddOvOp : SRCC_Op<"sadd.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Add and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, I1:$overflow);
    let hasFolder = 1;
}

def SMulOvOp : SRCC_Op<"smul.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Multiply and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, I1:$overflow);
    let hasFolder = 1;
}

def SSubOvOp : SRCC_Op<"ssub.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Subtract and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, I1:$overflow);
    let hasFolder = 1;
}

def SelectOp : SRCC_Op<"select", [Pure, AllTypesMatch<["then_val", "else_val", "res"]>]> {
    let summary = "Select between two values based on a condition";
    let arguments = (ins
        SRCC_Bool:$cond,
        SRCC_Type:$then_val,
        SRCC_Type:$else_val
    );

    let results = (outs SRCC_Type:$res);
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$cond,
            "::mlir::Value":$then_val,
            "::mlir::Value":$else_val
        ), [{
            build(
                $_builder,
                $_state,
                then_val.getType(),
                cond,
                then_val,
                else_val
            );
        }]>
    ];
}

def StoreOp : SRCC_Op<"store"> {
    let summary = "Store a value to a pointer";
    let arguments = (ins
        SRCC_Ptr:$addr,
        SRCC_Type:$value,
        Builtin_IntegerAttr:$alignment
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$addr,
            "::mlir::Value":$value,
            "Align":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                addr,
                value,
                $_builder.getI64IntegerAttr(i64(alignment.value().bytes()))
            );
        }]>
    ];
}

def TupleOp : SRCC_Op<"tuple", [Pure]> {
    let summary = "Create a tuple value";
    let arguments = (ins Variadic<SRCC_Type>:$values);
    let results = (outs SRCC_Tuple:$res);
    let builders = [
        OpBuilder<(ins
            "::mlir::ValueRange":$values
        ), [{
            SmallVector<mlir::Type> types;
            for (auto v : values) types.push_back(v.getType());
            build(
                $_builder,
                $_state,
                mlir::TupleType::get($_builder.getContext(), types),
                values
            );
        }]>
    ];
}
