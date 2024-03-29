#ifndef HLIR_OPS_TD
#define HLIR_OPS_TD

include "HLIRDialect.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"

def FuncOp : HLIR_Op<"func", [
    AffineScope, AutomaticAllocationScope, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
    let summary = "Function";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        LinkageAttr:$linkage,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$function_type,
        BoolAttr:$special_member,
        BoolAttr:$variadic,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "::mlir::LLVM::Linkage":$linkage,
            "::mlir::LLVM::CConv":$cc,
            "FunctionType":$type,
            "bool":$special_member,
            "bool":$variadic,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
        ), [{
            buildWithEntryBlock($_builder, $_state, name, type, attrs, type.getInputs());
            $_state.addAttribute(getCcAttrName($_state.name), mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
            $_state.addAttribute(getLinkageAttrName($_state.name), mlir::LLVM::LinkageAttr::get($_state.getContext(), linkage));
            $_state.addAttribute(getSpecialMemberAttrName($_state.name), mlir::BoolAttr::get($_state.getContext(), special_member));
            $_state.addAttribute(getVariadicAttrName($_state.name), mlir::BoolAttr::get($_state.getContext(), variadic));
        }]>
    ];

    /// Most of this is copied from the func dialect’s FuncOp.
    let extraClassDeclaration = [{
        /// Get the Nth explicit argument of this function.
        Value getExplicitArgument(unsigned idx) {
          if (getSpecialMember()) idx++;
          return getArgument(idx);
        }

        /// Get implicit this parameter.
        Value getImplicitThis() {
          assert (getSpecialMember() && "this parameter only exists for special member procs");
          return getArgument(0);
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
        ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

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
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "hlir"; }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def CallOp : HLIR_Op<"call", [
    CallOpInterface,
    MemRefsNormalizable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "call operation";
    let description = [{ Function call }];
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        BoolAttr:$inline_call,
        CConvAttr:$cc,
        Variadic<HLIR_Type>:$args
    );
    let results = (outs Optional<HLIR_Type>:$yield);

    let builders = [
        OpBuilder<(ins
            "::mlir::TypeRange":$results,
            "::llvm::StringRef":$callee,
            "bool":$inline_call,
            "::mlir::LLVM::CConv":$cc,
            CArg<"ValueRange", "{}">:$operands
        ), [{
          $_state.addOperands(operands);
          $_state.addAttribute(getCalleeAttrName($_state.name), FlatSymbolRefAttr::get($_state.getContext(), callee));
          $_state.addAttribute(getCcAttrName($_state.name), mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
          $_state.addAttribute(getInlineCallAttrName($_state.name), mlir::BoolAttr::get($_state.getContext(), inline_call));
          $_state.addTypes(results);
        }]>
    ];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            auto range = getODSOperandIndexAndLength(0);
            auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
            return mutableRange;
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
    }];
}

def ReturnOp : HLIR_Op<"return", [
    Pure, HasParent<"FuncOp">,
    MemRefsNormalizable, ReturnLike, Terminator,
    AttrSizedOperandSegments
]> {
    let summary = "Function return operation";
    let hasCustomAssemblyFormat = 1;
    let arguments = (ins Optional<HLIR_Type>:$yield, Variadic<HLIR_Type>:$prot);
    let hasVerifier = 0;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            CArg<"::mlir::Value", "{}">:$yield,
            CArg<"::mlir::ValueRange", "{}">:$prot
        ), [{
            if (yield) $_state.addOperands(yield);
            $_state.addOperands(prot);
            ::llvm::copy(
                ::llvm::ArrayRef<int32_t>({(yield ? 1 : 0), static_cast<int32_t>(prot.size())}),
                $_state.getOrAddProperties<Properties>().operandSegmentSizes.begin()
            );
        }]>
    ];
}

def DirectBrOp : HLIR_Op<"directbr", [
    Pure, Terminator
]> {
    let summary = "Direct branch to a block";
    let hasCustomAssemblyFormat = 1;
    let arguments = (ins Variadic<HLIR_Type>:$prot);
    let successors = (successor AnySuccessor:$dest);
    let hasVerifier = 0;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Block*":$dest,
            CArg<"::mlir::ValueRange", "{}">:$prot
        ), [{
            $_state.addSuccessors(dest);
            $_state.addOperands(prot);
        }]>
    ];
}

def StringOp : HLIR_Op<"string"> {
    let summary = "String literal";
    let arguments = (ins StrAttr:$value, IndexAttr:$index);

    let builders = [
        OpBuilder<(ins "::mlir::StringAttr":$value, "::mlir::IntegerAttr":$index), [{
            build($_builder, $_state, {}, value, index);
        }]>,

        OpBuilder<(ins "::llvm::StringRef":$value, "::llvm::APInt":$index), [{
            build(
                $_builder,
                $_state,
                {},
                value,
                index
            );
        }]>,
    ];

    let assemblyFormat = "$index `,` $value attr-dict";
    let hasVerifier = 0;
}

def ScopeOp : HLIR_Op<"scope", [AutomaticAllocationScope, OpAsmOpInterface]> {
    let summary = "Open a scope that may contain deferred code";
    let regions = (region AnyRegion:$body);
    let arguments = (ins BoolAttr:$early_yield);
    let results = (outs Optional<HLIR_Type>:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins CArg<"::mlir::Type", "{}">:$res), [{
            if (res) $_state.addTypes({res});
            auto r = $_state.addRegion();
            r->push_back(new mlir::Block);
            $_state.addAttribute("early_yield", mlir::BoolAttr::get($_state.getContext(), false));
        }]>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "hlir"; }
    }];
}

def YieldOp : HLIR_Op<"yield", [Terminator, AttrSizedOperandSegments]> {
    let summary = "Yield operation";
    let arguments = (ins
        Optional<HLIR_Type>:$yield,
        Variadic<HLIR_Type>:$prot
    );
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def DeferOp : HLIR_Op<"defer", [AutomaticAllocationScope, OpAsmOpInterface]> {
    let summary = "Defer code to be executed at end of scope";
    let regions = (region AnyRegion:$body);
    let results = (outs HLIR_TokenType:$token);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins), [{
            auto r = $_state.addRegion();
            r->push_back(new mlir::Block);
            $_state.addTypes(hlir::TokenType::get($_state.getContext()));
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the scope of this operation.
        auto getScopeOp() -> ScopeOp;

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "hlir"; }
    }];
}

def SliceDataOp : HLIR_Op<"slice_data"> {
    let summary = "Get the data pointer of a slice";
    let arguments = (ins HLIR_SliceType:$operand);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            build(
                $_builder,
                $_state,
                ReferenceType::get($_state.getContext(), operand.getType().cast<SliceType>().getElem()),
                operand
            );
        }]>
    ];
}

def SliceSizeOp : HLIR_Op<"slice_size"> {
    let summary = "Get the size of a slice";
    let arguments = (ins HLIR_SliceType:$operand);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def GlobalRefOp : HLIR_Op<"global_ref"> {
    let summary = "Get a reference to a global variable";
    let arguments = (ins SymbolRefAttr:$name);
    let results = (outs HLIR_ReferenceType:$res);
    let assemblyFormat = "$name attr-dict `:` type($res)";
    let hasVerifier = 0;
}

def NilOp : HLIR_Op<"nil"> {
    let summary = "Nil value of a given type";
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Type":$type), [{
            build(
                $_builder,
                $_state,
                type,
                {}
            );
        }]>
    ];
}

def LoadOp : HLIR_Op<"load"> {
    let summary = "Load a value from a reference";
    let arguments = (ins HLIR_ReferenceType:$operand);
    let results = (outs HLIR_Type:$type);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            build(
                $_builder,
                $_state,
                operand.getType().cast<ReferenceType>().getElem(),
                operand
            );
        }]>
    ];
}

def StoreOp : HLIR_Op<"store"> {
    let summary = "Store a value to a reference";
    let arguments = (ins
        HLIR_ReferenceType:$addr,
        HLIR_Type:$value,
        Builtin_IntegerAttr:$alignment
    );
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$addr,
            "::mlir::Value":$value,
            "std::int64_t":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                addr,
                value,
                $_builder.getI64IntegerAttr(alignment)
            );
        }]>
    ];
}

def LiteralOp : HLIR_Op<"literal"> {
    let summary = "Literal value";
    let arguments = (ins Variadic<HLIR_Type>:$args);
    let results = (outs HLIR_Type:$value);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::hlir::SliceType":$type,
            "::mlir::Value":$data,
            "::mlir::Value":$size
        ), [{
            build(
                $_builder,
                $_state,
                type,
                {data, size}
            );
        }]>
    ];
}

def ZeroinitialiserOp : HLIR_Op<"zeroinitialiser"> {
    let summary = "Zeroinitialiser";
    let arguments = (ins HLIR_ReferenceType:$operand, DefaultValuedAttr<I64Attr, "1">:$array_size);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand, CArg<"std::int64_t", "1">:$array_size), [{
            build(
                $_builder,
                $_state,
                {},
                operand,
                $_builder.getI64IntegerAttr(array_size)
            );
        }]>
    ];
}

def ArrayDecayOp : HLIR_Op<"array_decay"> {
    let summary = "Decay a reference to an array to a reference to its first element";
    let arguments = (ins HLIR_ReferenceType:$operand);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            auto type = ReferenceType::get(
                $_state.getContext(),
                operand
                    .getType()
                    .cast<ReferenceType>()
                    .getElem()
                    .cast<ArrayType>()
                    .getElem()
            );

            build(
                $_builder,
                $_state,
                type,
                operand
            );
        }]>
    ];
}

def HLIR_LocalOp : HLIR_Op<"local"> {
    let summary = "Local variable";
    let hasVerifier = 0;
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
    let results = (outs HLIR_ReferenceType:$res);
    let arguments = (ins
        Builtin_IntegerAttr:$alignment,
        DefaultValuedAttr<BoolAttr, "false">:$dtor_flag
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Type":$type,
            "std::int64_t":$alignment,
            CArg<"bool", "false">:$dtor_flag
        ), [{
            $_state.addTypes(ReferenceType::get($_state.getContext(), type));
            $_state.addAttribute("alignment", $_builder.getI64IntegerAttr(alignment));
            $_state.addAttribute("dtor_flag", mlir::BoolAttr::get($_state.getContext(), dtor_flag));
        }]>
    ];
}

def HLIR_StaticOp : HLIR_Op<"static"> {
    let summary = "Static variable";
    let hasVerifier = 0;
    let hasCustomAssemblyFormat = 1;
    let arguments = (ins
        TypeAttr:$static_type,
        SymbolNameAttr:$sym_name,
        LinkageAttr:$linkage,
        Builtin_IntegerAttr:$alignment
    );

    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "::mlir::LLVM::Linkage":$linkage,
            "::mlir::Type":$static_type,
            "std::int64_t":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                mlir::TypeAttr::get(static_type),
                StringAttr::get($_state.getContext(), name),
                mlir::LLVM::LinkageAttr::get($_state.getContext(), linkage),
                $_builder.getI64IntegerAttr(alignment)
            );
        }]>
    ];
}

def ExpIOp : HLIR_Op<
    "expi", [
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
        NoMemoryEffect
    ] # ElementwiseMappable.traits> {
    let summary = "Exponential function";
    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike:$res);
    let hasVerifier = 0;
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$lhs,
            "::mlir::Value":$rhs
        ), [{
            build(
                $_builder,
                $_state,
                lhs.getType(),
                {lhs, rhs}
            );
        }]>
    ];
}

def StructGEPOp : HLIR_Op<"structgep"> {
    let summary = "Get a pointer to a struct element by offsetting from a struct pointer";
    let arguments = (ins HLIR_ReferenceType:$struct_ref, Builtin_IntegerAttr:$idx);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$struct_ref,
            "::std::int32_t":$idx
        ), [{
            auto ty = struct_ref
                .getType().cast<ReferenceType>()
                .getElem().cast<LLVM::LLVMStructType>()
                .getBody()[idx];

            build(
                $_builder,
                $_state,
                hlir::ReferenceType::get(ty),
                struct_ref,
                $_builder.getI32IntegerAttr(idx)
            );
        }]>
    ];
}

def OffsetOp : HLIR_Op<"offset"> {
    let summary = "Offset a pointer";
    let arguments = (ins HLIR_ReferenceType:$pointer, SignlessIntegerLike:$offset);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$pointer,
            "::mlir::Value":$offset
        ), [{
            build(
                $_builder,
                $_state,
                pointer.getType(),
                {pointer, offset}
            );
        }]>
    ];
}

def ChainExtractLocalOp : HLIR_Op<"extractlocal"> {
    let summary = "Extract a local variable from a static chain area";
    let arguments = (ins HLIR_ReferenceType:$struct_ref, Builtin_IntegerAttr:$idx);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$struct_ref,
            "::std::int32_t":$idx
        ), [{
            auto ty = struct_ref
                .getType().cast<ReferenceType>()
                .getElem().cast<LLVM::LLVMStructType>()
                .getBody()[idx];

            build(
                $_builder,
                $_state,
                ty,
                struct_ref,
                $_builder.getI32IntegerAttr(idx)
            );
        }]>
    ];
}

def MakeClosureOp : HLIR_Op<"makeclosure"> {
    let summary = "Make a closure";
    let hasCustomAssemblyFormat = 1;
    let results = (outs HLIR_ClosureType);
    let arguments = (ins
        FlatSymbolRefAttr:$procedure,
        Optional<HLIR_Type>:$env
    );

    let builders = [
        OpBuilder<(ins
            "::llvm::StringRef":$procedure,
            "::mlir::Type":$proc_type,
            CArg<"::mlir::Value", "{}">:$env
        ), [{
            build(
                $_builder,
                $_state,
                proc_type,
                SymbolRefAttr::get($_state.getContext(), procedure),
                env
            );
        }]>
    ];
}

def InvokeClosureOp : HLIR_Op<"invokeclosure"> {
    let summary = "Invoke a closure";
    let hasCustomAssemblyFormat = 1;
    let results = (outs HLIR_Type);
    let arguments = (ins
        HLIR_ClosureType:$closure,
        CConv:$cc,
        Variadic<HLIR_Type>:$args
    );
}

def NewOp : HLIR_Op<"new"> {
    let summary = "Allocate an object on the heap";
    let results = (outs HLIR_ReferenceType);
    let hasCustomAssemblyFormat = 1;
}

def DeleteOp : HLIR_Op<"delete"> {
    let summary = "Free an object allocated on the heap";
    let arguments = (ins HLIR_ReferenceType:$object);
    let hasCustomAssemblyFormat = 1;
}

def ConstructOp : HLIR_Op<"construct"> {
    let summary = "Construct an object";
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let arguments = (ins
        HLIR_ReferenceType:$object,
        LocalInit:$init_kind,
        FlatSymbolRefAttr:$ctor,
        DefaultValuedAttr<I64Attr, "1">:$array_size,
        Variadic<HLIR_Type>:$args
    );

    let builders = [
        /// Constructor call.
        OpBuilder<(ins
            "::mlir::Value":$object,
            "::llvm::StringRef":$ctor,
            CArg<"::mlir::ValueRange", "{}">:$args,
            CArg<"std::int64_t", "1">:$array_size
        ), [{
            build(
                $_builder,
                $_state,
                {},
                object,
                hlir::LocalInitAttr::get($_state.getContext(), ::mlir::hlir::LocalInit::Init),
                FlatSymbolRefAttr::get($_state.getContext(), ctor),
                $_builder.getI64IntegerAttr(array_size),
                args
            );
        }]>,

        /// Zero-init or trivial copy-init.
        OpBuilder<(ins
            "::mlir::Value":$object,
            CArg<"::mlir::ValueRange", "{}">:$args,
            CArg<"std::int64_t", "1">:$array_size
        ), [{
            auto init = args.empty()
                ? ::mlir::hlir::LocalInit::Zeroinit
                : ::mlir::hlir::LocalInit::TrivialCopyInit;

            build(
                $_builder,
                $_state,
                {},
                object,
                hlir::LocalInitAttr::get($_state.getContext(), init),
                FlatSymbolRefAttr::get($_state.getContext(), ""),
                $_builder.getI64IntegerAttr(array_size),
                args
            );
        }]>
    ];

}

def DestroyOp : HLIR_Op<"destroy"> {
    let summary = "Call the destructor of a local variable";
    let arguments = (ins HLIR_ReferenceType:$object, FlatSymbolRefAttr:$dtor);
    let results = (outs HLIR_TokenType);
    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$object,
            "::llvm::StringRef":$dtor
        ), [{
            build(
                $_builder,
                $_state,
                {hlir::TokenType::get($_state.getContext())},
                object,
                SymbolRefAttr::get($_state.getContext(), dtor)
            );
        }]>
    ];
}

def UnreachableOp : HLIR_Op<"unreachable", [Terminator]> {
    let summary = "Unreachable operation";
    let hasCustomAssemblyFormat = 1;
}

def PointerEqOp : HLIR_Op<"ptreq"> {
    let summary = "Test pointer equality";
    let arguments = (ins HLIR_Type:$lhs, HLIR_Type:$rhs);
    let results = (outs HLIR_Bool:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def PointerNeOp : HLIR_Op<"ptrne"> {
    let summary = "Test pointer inequality";
    let arguments = (ins HLIR_Type:$lhs, HLIR_Type:$rhs);
    let results = (outs HLIR_Bool:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def NotOp : HLIR_Op<"not"> {
    let summary = "Logical not";
    let arguments = (ins HLIR_Bool:$operand);
    let results = (outs HLIR_Bool:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def BitCastOp : HLIR_Op<"bitcast"> {
    let summary = "Cast between two types of the same size";
    let arguments = (ins HLIR_Type:$operand);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

def ExtractOp : HLIR_Op<"extract"> {
    let summary = "Extract a value from an array";
    let arguments = (ins HLIR_Type:$operand, HLIR_Type:$index);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

class BinOpBase<string name, list<Trait> traits = []> : HLIR_Op<
    name, traits
> {
    let summary = "Arithmetic operation";
    let arguments = (ins HLIR_Type:$lhs, HLIR_Type:$rhs);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;
}

class ArithOp<string name, list<Trait> traits = []> : BinOpBase<
    name, traits # [SameOperandsAndResultType]
>;

class CmpOp<string name, list<Trait> traits = []> : BinOpBase<
    name, traits
>;


def AddOp : ArithOp<"add">;
def SubOp : ArithOp<"sub">;
def MulOp : ArithOp<"mul">;
def DivOp : ArithOp<"div">;
def RemOp : ArithOp<"rem">;
def ExpOp : ArithOp<"exp">;
def AndOp : ArithOp<"and">;
def OrOp  : ArithOp<"or">;
def XorOp : ArithOp<"xor">;
def ShlOp : ArithOp<"shl">;
def ShrOp : ArithOp<"shr">;
def SarOp : ArithOp<"sar">;
def EqOp  : CmpOp<"eq">;
def NeOp  : CmpOp<"ne">;
def LtOp  : CmpOp<"lt">;
def LeOp  : CmpOp<"le">;
def GtOp  : CmpOp<"gt">;
def GeOp  : CmpOp<"ge">;



#endif // HLIR_OPS_TD
