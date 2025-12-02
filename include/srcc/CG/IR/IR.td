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
include "IRInterfaces.td"

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
def SRCC_Ptr : TypeAlias<LLVMPointerType, "ptr">;
def SRCC_Bool : TypeAlias<I1, "bool">;

def SRCC_Type : AnyTypeOf<[
    Builtin_Integer,
    FunctionType,
    SRCC_Ptr,
    SRCC_Bool,
]>;

// ============================================================================
//  Operations
// ============================================================================
def ProcOp : SRCC_Op<"proc", [
    AffineScope, AutomaticAllocationScope, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface,
    ProcAndCallOpInterface
]> {
    let summary = "Procedure";
    let regions = (region AnyRegion:$body);
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        LinkageAttr:$linkage,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$function_type,
        UnitAttr:$variadic,
        UnitAttr:$noreturn,
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
            "bool":$noreturn,
            "::mlir::ArrayAttr":$arg_attrs,
            "::mlir::ArrayAttr":$res_attrs,
            CArg<"ArrayRef<mlir::NamedAttribute>", "{}">:$func_attrs
        ), [{
            buildWithEntryBlock($_builder, $_state, name, type, func_attrs, type.getInputs());
            $_state.addAttribute(getCcAttrName($_state.name), ::mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
            $_state.addAttribute(getLinkageAttrName($_state.name), ::mlir::LLVM::LinkageAttr::get($_state.getContext(), linkage));
            $_state.addAttribute(getArgAttrsAttrName($_state.name), arg_attrs);
            $_state.addAttribute(getResAttrsAttrName($_state.name), res_attrs);
            if (variadic) $_state.addAttribute(getVariadicAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
            if (noreturn) $_state.addAttribute(getNoreturnAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
        }]>
    ];

    /// Most of this is copied from the func dialect’s FuncOp.
    let extraClassDeclaration = [{
        ::mlir::Block* getOrCreateEntryBlock() {
            if (empty()) addEntryBlock();
            return &front();
        }

        //===------------------------------------------------------------------===//
        // ProcAndCallOpInterface
        //===------------------------------------------------------------------===//
        unsigned getNumCallArgs() {
            return getNumArguments();
        }

        ::mlir::Value getCallArg(unsigned idx) {
            if (idx < getNumArguments()) return getArgument(idx);
            return {};
        }

        ::mlir::Type getCallArgType(unsigned idx) {
            if (idx < getNumArguments()) return getArgumentTypes()[idx];
            return {};
        }

        ::mlir::DictionaryAttr getCallArgAttrs(unsigned idx) {
            if (getArgAttrs() and idx < getArgAttrs()->size())
                return cast<mlir::DictionaryAttr>(getArgAttrs().value()[idx]);
            return {};
        }

        ::mlir::DictionaryAttr getCallResultAttrs(unsigned idx) {
            if (getResAttrs() and idx < getResAttrs()->size())
                return cast<mlir::DictionaryAttr>(getResAttrs().value()[idx]);
            return {};
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
        SRCC_Ptr:$abort_info
    );
}

def CallOp : SRCC_Op<"call", [ProcAndCallOpInterface]> {
    let summary = "Call a procedure";
    let arguments = (ins
        SRCC_Ptr:$addr,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$proc_type,
        UnitAttr:$variadic,
        Variadic<SRCC_Type>:$args,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$result_attrs
    );

    let results = (outs Variadic<SRCC_Type>:$res);
    let builders = [
        OpBuilder<(ins
            "ProcRefOp":$ref,
            CArg<"::mlir::ValueRange", "{}">:$args,
            CArg<"::mlir::Value", "{}">:$env
        ), [{
            auto proc = ref.proc();
            build(
                $_builder,
                $_state,
                proc.getResultTypes(),
                ref,
                proc.getCc(),
                proc.getFunctionType(),
                proc.getVariadic(),
                args,
                {},
                {}
            );
        }]>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // ProcAndCallOpInterface
        //===------------------------------------------------------------------===//
        unsigned getNumCallArgs() {
            return unsigned(getArgs().size());
        }

        ::mlir::Value getCallArg(unsigned idx) {
            if (idx < getNumCallArgs()) return getArgs()[idx];
            return {};
        }

        ::mlir::Type getCallArgType(unsigned idx) {
            if (idx < getNumCallArgs()) return getArgs()[idx].getType();
            return {};
        }

        ::mlir::DictionaryAttr getCallArgAttrs(unsigned idx) {
            if (getArgAttrs() and idx < getArgAttrs()->size())
                return cast<mlir::DictionaryAttr>(getArgAttrs().value()[idx]);
            return {};
        }

        ::mlir::DictionaryAttr getCallResultAttrs(unsigned idx) {
            if (getResultAttrs() and idx < getResultAttrs()->size())
                return cast<mlir::DictionaryAttr>(getResultAttrs().value()[idx]);
            return {};
        }
    }];
}

def DisengageOp : SRCC_Op<"disengage"> {
    let summary = "Disengage an optional";
    let arguments = (ins SRCC_Ptr:$optional);
}

def EngageOp : SRCC_Op<"engage"> {
    let summary = "Engage an optional";
    let arguments = (ins SRCC_Ptr:$optional);
}

def FrameSlotOp : SRCC_Op<"slot", [Pure]> {
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

    let extraClassDeclaration = [{
        base::Align align() {
            return base::Align(u64(getAlignment().getInt()));
        }

        base::Size size() {
            return base::Size::Bytes(getBytes().getInt());
        }
    }];
}

def LoadOp : SRCC_Op<"load", [Pure]> {
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

def RetOp : SRCC_Op<"ret", [Pure, Terminator, ReturnLike, HasParent<"ProcOp">]> {
    let summary = "Return from a procedure";
    let arguments = (ins Variadic<SRCC_Type>:$vals);
}

def SAddOvOp : SRCC_Op<"sadd.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Add and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, SRCC_Bool:$overflow);
    let hasFolder = 1;
}

def SMulOvOp : SRCC_Op<"smul.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Multiply and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, SRCC_Bool:$overflow);
    let hasFolder = 1;
}

def SSubOvOp : SRCC_Op<"ssub.ov", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let summary = "Subtract and check for overflow";
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$res, SRCC_Bool:$overflow);
    let hasFolder = 1;
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

def UnwrapOp : SRCC_Op<"unwrap"> {
    let summary = "Unwrap an optional";
    let arguments = (ins SRCC_Ptr:$optional);
}
