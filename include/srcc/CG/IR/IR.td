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

def SRCC_Tree : SRCC_Type<"Tree", "tree", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
    let summary = "Type of a parse tree node";
    let description = "A compile-time parse tree";
    let genMnemonicAlias = 1;
}

def SRCC_Type : SRCC_Type<"Type", "type", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
    let summary = "Type of a type constant or value";
    let description = "A compile-time type value";
    let genMnemonicAlias = 1;
}

def SRCC_AnyType : AnyTypeOf<[
    Builtin_Integer,
    FunctionType,
    SRCC_Ptr,
    SRCC_Bool,
    SRCC_Tree,
    SRCC_Type,
]>;

// ============================================================================
//  Attributes
// ============================================================================
def SRCC_StmtAttr : SRCC_Attr<"Stmt", "stmt"> {
    let summary = "A statement";
    let description = "A statement";
    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::srcc::Stmt*":$value);
    let hasCustomAssemblyFormat = 1;
    let genMnemonicAlias = 1;
}

def SRCC_TreeAttr : SRCC_Attr<"Tree", "tree"> {
    let summary = "Type of a type constant or value";
    let description = "A compile-time type value";
    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::srcc::TreeValue*":$value);
    let hasCustomAssemblyFormat = 1;
    let genMnemonicAlias = 1;
}

def SRCC_TypeAttr : SRCC_Attr<"Type", "type"> {
    let summary = "Type of a type constant or value";
    let description = "A compile-time type value";
    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::srcc::Type":$value);
    let hasCustomAssemblyFormat = 1;
    let genMnemonicAlias = 1;
}

def SRCC_AbortReasonAttr : EnumAttr<SRCC_Dialect, SRCC_AbortReason_Enum, "AbortReason">;
def SRCC_BuiltinMemberAccessKindAttr : SRCC_Attr<"BuiltinMemberAccessKind", "access_kind"> {
    let summary = "Type of a type constant or value";
    let description = "A compile-time type value";
    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "::srcc::BuiltinMemberAccessExpr::AccessKind":$value);
    let hasCustomAssemblyFormat = 1;
    let genMnemonicAlias = 1;
}

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
        UnitAttr:$always_inline,
        UnitAttr:$norecurse,
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
            "bool":$always_inline,
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
            if (always_inline) $_state.addAttribute(getAlwaysInlineAttrName($_state.name), ::mlir::UnitAttr::get($_state.getContext()));
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

def AbortOp : SRCC_Op<"abort", [Terminator, NoTerminator]> {
    let summary = "Report a fatal error and abort the program";
    let regions = (region AnyRegion:$body);
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
        UnitAttr:$always_inline,
        Variadic<SRCC_AnyType>:$args,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$result_attrs
    );

    let results = (outs Variadic<SRCC_AnyType>:$res);
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
                false,
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

def EngageCopyOp : SRCC_Op<"engage.copy"> {
    let summary = "Engage/disengage an optional depending on the state of another optional";
    let arguments = (ins SRCC_Ptr:$optional, SRCC_Ptr:$copy_from);
}

// We can’t just use LLVM::LoadOp since it doesn’t support custom types.
def LoadOp : SRCC_Op<"load", [Pure]> {
    let summary = "Load a value from a pointer";
    let arguments = (ins
        SRCC_Ptr:$addr,
        Builtin_IntegerAttr:$alignment
    );

    let results = (outs SRCC_AnyType:$res);
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
    let results = (outs SRCC_AnyType:$res);
}

def ProcRefOp : SRCC_Op<"procaddr", [Pure]> {
    let summary = "Reference a procedure";
    let arguments = (ins SymbolNameAttr:$proc_name);
    let results = (outs SRCC_AnyType:$res);
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

def QuoteOp : SRCC_Op<"quote", [Pure]> {
    let summary = "A parse tree";
    let arguments = (ins Property<"Stmt*">:$tree, Variadic<SRCC_Tree>:$unquotes);
    let results = (outs SRCC_Tree);
}

def RetOp : SRCC_Op<"ret", [Pure, Terminator, ReturnLike, HasParent<"ProcOp">]> {
    let summary = "Return from a procedure";
    let arguments = (ins Variadic<SRCC_AnyType>:$vals);
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

// We can’t just use LLVM::StoreOp since it doesn’t support custom types.
def StoreOp : SRCC_Op<"store"> {
    let summary = "Store a value to a pointer";
    let arguments = (ins
        SRCC_Ptr:$addr,
        SRCC_AnyType:$value,
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

def TreeConstantOp : SRCC_Op<"tree", [Pure]> {
    let summary = "A parse tree";
    let arguments = (ins Property<"TreeValue*">:$tree);
    let results = (outs SRCC_Tree);
}

def TypeConstantOp : SRCC_Op<"type", [ConstantLike, Pure]> {
    let summary = "A type constant";
    let arguments = (ins Property<"::srcc::Type">:$value);
    let results = (outs SRCC_Type);
    let hasFolder = 1;
}

def TypeEqOp : SRCC_Op<"type.eq", [Pure]> {
    let summary = "Check if two types are equal";
    let arguments = (ins SRCC_Type:$lhs, SRCC_Type:$rhs);
    let results = (outs SRCC_Bool);
    let hasFolder = 1;
}

def TypePropertyOp : SRCC_Op<"type.prop", [Pure]> {
    let summary = "Query a compile-time known property of a type";
    let arguments = (ins SRCC_Type:$type_argument, Property<"::srcc::BuiltinMemberAccessExpr::AccessKind">:$property);
    let results = (outs Variadic<SRCC_AnyType>);
    let hasFolder = 0; // We perform the folding in codegen.
}

def UnwrapOp : SRCC_Op<"unwrap"> {
    let summary = "Unwrap an optional";
    let arguments = (ins SRCC_Ptr:$optional);
}
