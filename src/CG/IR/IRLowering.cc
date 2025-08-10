#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Constants.hh>

#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

using namespace srcc;
using namespace srcc::cg;

namespace srcc::cg::lowering {
namespace LLVM = mlir::LLVM;
using mlir::failure;
using mlir::success;

struct LoweringPass final : mlir::PassWrapper<LoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
    CodeGen& cg;
    LoweringPass(CodeGen& cg) : cg(cg) {}

    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() override;
};

#define LOWERING(SRCCOp, ...)                                               \
    struct SRCCOp##Lowering final : mlir::OpConversionPattern<ir::SRCCOp> { \
        CodeGen& cg;                                                        \
        SRCCOp##Lowering(CodeGen& cg, const mlir::LLVMTypeConverter& tc)    \
            : OpConversionPattern(tc, cg.mlir_context()), cg{cg} {}         \
        auto matchAndRewrite(                                               \
            ir::SRCCOp op,                                                  \
            ir::SRCCOp##Adaptor a,                                          \
            mlir::ConversionPatternRewriter& r                              \
        ) const -> mlir::LogicalResult override {                           \
            auto replacement = CheckReplacement([&] { __VA_ARGS__ }());     \
            if (!replacement) return failure();                             \
            r.replaceOp(op, replacement.value());                           \
            return success();                                               \
        }                                                                   \
    }

template <typename Value>
auto CheckReplacement(Value v) -> std::optional<Value> {
    if constexpr (requires { !v; }) {
        if (!v) return std::nullopt;
    }

    return v;
}

template <typename Intrin>
auto LowerOverflowOp(auto op, auto& a, mlir::ConversionPatternRewriter& r) -> SmallVector<Value, 2> {
    auto intrin = r.create<Intrin>(
        op.getLoc(),
        LLVM::LLVMStructType::getLiteral(r.getContext(), {a.getLhs().getType(), r.getI1Type()}),
        a.getLhs(),
        a.getRhs()
    );

    auto val = r.create<LLVM::ExtractValueOp>(op.getLoc(), intrin, 0);
    auto overflow = r.create<LLVM::ExtractValueOp>(op.getLoc(), intrin, 1);
    return SmallVector<Value, 2>{val, overflow};
}

LOWERING(AbortOp, {
    // Just call the appropriate handler.
    String name = [&] {
        switch (a.getReason()) {
            case ir::AbortReason::ArithmeticError:
                return constants::ArithmeticFailureHandlerName;
            case ir::AbortReason::AssertionFailed:
                return constants::AssertFailureHandlerName;
            case ir::AbortReason::InvalidLocalRef:
                Unreachable("Should only be emitted during constant evaluation");
        }
        Unreachable();
    }();

    auto func = op->getParentOfType<mlir::ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(name);
    Assert(func, "Abort handler not found!");
    r.create<LLVM::CallOp>(op.getLoc(), func, a.getAbortInfo());
    return r.create<LLVM::UnreachableOp>(op.getLoc());
});

LOWERING(CallOp, {
    SmallVector<mlir::Value> args;
    SmallVector<mlir::Type> arg_types;
    auto tc = getTypeConverter<mlir::LLVMTypeConverter>();
    auto ptr = LLVM::LLVMPointerType::get(getContext());

    args.push_back(a.getAddr());
    if (a.getMrvalueSlot()) {
        args.push_back(a.getMrvalueSlot());
        arg_types.push_back(ptr);
    }

    for (auto arg : a.getArgs()) {
        args.push_back(arg);
        arg_types.push_back(arg.getType());
    }

    // Note: I *think* it’s fine to just throw the environment in there
    // even if the function we’re calling doesn’t take an environment–at
    // least on x64. For some targets, we’ll probably have to guard the
    // call and not pass an environment if it ends up being null.
    if (a.getEnv()) {
        args.push_back(a.getEnv());
        arg_types.push_back(ptr);
    }

    auto fty = LLVM::LLVMFunctionType::get(
        op.getNumResults()
            ? tc->convertType(op.getResult(0).getType())
            : LLVM::LLVMVoidType::get(getContext()),
        arg_types,
        a.getVariadic()
    );

    auto call = r.create<LLVM::CallOp>(op.getLoc(), fty, args);
    call.setCConv(op.getCc());
    return call;
});

LOWERING(ExtractOp, {
    return r.create<LLVM::ExtractValueOp>(
        op.getLoc(),
        a.getTuple(),
        unsigned(a.getIndex().getInt())
    );
});

LOWERING(FrameSlotOp, {
    return r.create<LLVM::AllocaOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(r.getContext()),
        r.getI8Type(),
        r.create<LLVM::ConstantOp>(op->getLoc(), a.getBytesAttr()),
        unsigned(a.getAlignment().getInt())
    );
});

LOWERING(LoadOp, {
    return r.create<LLVM::LoadOp>(
        op.getLoc(),
        getTypeConverter()->convertType(op.getResult().getType()),
        a.getAddr(),
        unsigned(a.getAlignment().getInt())
    );
});

LOWERING(NilOp, {
    return r.create<LLVM::ZeroOp>(
        op.getLoc(),
        getTypeConverter()->convertType(op.getResult().getType())
    );
});

LOWERING(ProcOp, {
    auto tc = getTypeConverter<mlir::LLVMTypeConverter>();
    auto old = op.getFunctionType();
    mlir::LLVMTypeConverter::SignatureConversion sc{old.getNumInputs()};

    // Add the return pointer, if need be.
    mlir::Type ret;
    if (op.getHasIndirectReturn()) {
        sc.addInputs(LLVM::LLVMPointerType::get(getContext()));
        ret = LLVM::LLVMVoidType::get(getContext());
    } else {
        ret = tc->convertType(old.getResult(0));
    }

    // Add the arguments.
    for (auto [i, arg] : enumerate(old.getInputs()))
        sc.addInputs(unsigned(i), tc->convertType(arg));

    // Add the static chain pointer.
    if (op.getHasStaticChain())
        sc.addInputs(LLVM::LLVMPointerType::get(getContext()));

    // Build the new function.
    auto llvm_fty = LLVM::LLVMFunctionType::get(
        ret,
        sc.getConvertedTypes(),
        op.getVariadic()
    );

    auto func = r.create<LLVM::LLVMFuncOp>(
        op.getLoc(),
        op.getName(),
        llvm_fty,
        op.getLinkage().getLinkage(),
        /*dsoLocal=*/false,
        op.getCc()
    );

    // And inline the body.
    r.inlineRegionBefore(op.getBody(), func.getBody(), func.end());
    if (failed(r.convertRegionTypes(&func.getBody(), *getTypeConverter(), &sc))) func = {};
    return func;
});

LOWERING(ProcRefOp, {
    return r.create<LLVM::AddressOfOp>(
        op.getLoc(),
        LLVM::LLVMPointerType::get(r.getContext()),
        a.getProcName()
    );
});

LOWERING(RetOp, {
    return r.create<LLVM::ReturnOp>(
        op.getLoc(),
        a.getValue()
    );
});

LOWERING(ReturnPointerOp, {
    return op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
});

LOWERING(SAddOvOp, { return LowerOverflowOp<LLVM::SAddWithOverflowOp>(op, a, r); });
LOWERING(SMulOvOp, { return LowerOverflowOp<LLVM::SMulWithOverflowOp>(op, a, r); });
LOWERING(SSubOvOp, { return LowerOverflowOp<LLVM::SSubWithOverflowOp>(op, a, r); });
LOWERING(StoreOp, {
    return r.create<LLVM::StoreOp>(
        op->getLoc(),
        a.getValue(),
        a.getAddr(),
        unsigned(a.getAlignment().getInt())
    );
});

LOWERING(TupleOp, {
    mlir::Value tuple = r.create<LLVM::UndefOp>(
        op.getLoc(),
        LLVM::LLVMStructType::getLiteral(
            getContext(),
            SmallVector<mlir::Type>{a.getValues().getTypes()}
        )
    );

    for (auto [i, v] : llvm::enumerate(a.getValues())) {
        tuple = r.create<LLVM::InsertValueOp>(
            op.getLoc(),
            tuple,
            v,
            i
        );
    }

    return tuple;
});

} // namespace srcc::cg::lowering

using namespace srcc::cg::lowering;

void LoweringPass::runOnOperation() {
    using namespace mlir;
    LLVMConversionTarget target{getContext()};
    LLVMTypeConverter tc{&getContext()};
    RewritePatternSet patterns{&getContext()};

    arith::populateArithToLLVMConversionPatterns(tc, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);

    // Convert none to void.
    tc.addConversion([&](NoneType) {
        return LLVM::LLVMVoidType::get(&getContext());
    });

    // Convert tuples to LLVM structs.
    //
    // FIXME: Tuple types should no longer exist here; eliminate them in
    // a separate lowering pass that also runs before constant evaluation.
    tc.addConversion([&](TupleType tt) -> LLVM::LLVMStructType {
        SmallVector<mlir::Type> args;
        if (failed(tc.convertTypes(tt.getTypes(), args))) return nullptr;
        return LLVM::LLVMStructType::getLiteral(&getContext(), args);
    });

    patterns.add< // clang-format off
        AbortOpLowering,
        CallOpLowering,
        ExtractOpLowering,
        FrameSlotOpLowering,
        LoadOpLowering,
        NilOpLowering,
        ProcOpLowering,
        ProcRefOpLowering,
        RetOpLowering,
        ReturnPointerOpLowering,
        SAddOvOpLowering,
        SMulOvOpLowering,
        SSubOvOpLowering,
        StoreOpLowering,
        TupleOpLowering
    >(cg, tc); // clang-format on

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        return signalPassFailure();
}

auto CodeGen::emit_llvm(llvm::TargetMachine& machine) -> std::unique_ptr<llvm::Module> {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(std::make_unique<LoweringPass>(*this));

    if (pm.run(mlir_module).failed()) {
        if (not tu.context().diags().has_error()) ICE(Location(), "Failed to lower module to LLVM IR");
        return nullptr;
    }

    auto m = mlir::translateModuleToLLVMIR(mlir_module, tu.llvm_context, tu.name);
    m->setTargetTriple(machine.getTargetTriple());
    m->setDataLayout(machine.createDataLayout());

    // Emit the module description if this is a module.
    if (tu.is_module) {
        SmallString<0> md;
        tu.serialise(md);
        auto mb = llvm::MemoryBuffer::getMemBuffer(md, "", false);
        llvm::embedBufferInModule(*m, mb->getMemBufferRef(), constants::ModuleDescriptionSectionName(tu.name));
    }

    return m;
}

bool CodeGen::finalise() {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());
    if (pm.run(mlir_module).failed()) {
        if (not tu.context().diags().has_error()) ICE(Location(), "Failed to finalise IR");
        return false;
    }
    return true;
}
