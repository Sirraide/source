#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Constants.hh>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <enchantum/enchantum.hpp>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#define TCase(type, name, ...) .Case<type>([&](auto&& name) { return __VA_ARGS__; })

using namespace srcc;
using namespace srcc::cg;

namespace srcc::cg::lowering {
namespace LLVM = mlir::LLVM;
using mlir::failure;
using mlir::success;
using mlir::ChangeResult;
using enum mlir::ChangeResult;

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

#define ERASE_OP(SRCCOp)                                                    \
    struct SRCCOp##Lowering final : mlir::OpConversionPattern<ir::SRCCOp> { \
        CodeGen& cg;                                                        \
        SRCCOp##Lowering(CodeGen& cg, const mlir::LLVMTypeConverter& tc)    \
            : OpConversionPattern(tc, cg.mlir_context()), cg{cg} {}         \
        auto matchAndRewrite(                                               \
            ir::SRCCOp op,                                                  \
            ir::SRCCOp##Adaptor,                                            \
            mlir::ConversionPatternRewriter& r                              \
        ) const -> mlir::LogicalResult override {                           \
            r.eraseOp(op);                                                  \
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
    auto intrin = Intrin::create(
        r,
        op.getLoc(),
        LLVM::LLVMStructType::getLiteral(r.getContext(), {a.getLhs().getType(), r.getI1Type()}),
        a.getLhs(),
        a.getRhs()
    );

    auto val = LLVM::ExtractValueOp::create(r, op.getLoc(), intrin, 0);
    auto overflow = LLVM::ExtractValueOp::create(r, op.getLoc(), intrin, 1);
    return SmallVector<Value, 2>{val, overflow};
}

static void PropagateArgAndResultAttrs(auto func_or_call, auto op) {
    func_or_call.setArgAttrsAttr(op.getArgAttrsAttr());
    if (auto attrs = op.getCallResultAttrs(0))
        func_or_call.setResAttrsAttr(mlir::ArrayAttr::get(op.getContext(), attrs));
}

LOWERING(AbortOp, {
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

    // Retrieve a declaration of the appropriate handler.
    //
    // Look up *any* symbol here, not just an 'LLVM::LLVMFuncOp' in case
    // we’re compiling the file that actually defines these symbols, in
    // which case they may not have been lowered to 'LLVMFuncOp's yet and
    // looking up a symbol of that type would fail, which would cause us
    // to declare the function an additional time and create a duplicate
    // definition error.
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto func = module.lookupSymbol(name);
    if (not func) {
        mlir::OpBuilder::InsertionGuard guard{r};
        r.setInsertionPointToEnd(&module.getBodyRegion().back());
        func = LLVM::LLVMFuncOp::create(
            r,
            r.getUnknownLoc(),
            name,
            LLVM::LLVMFunctionType::get(
                LLVM::LLVMVoidType::get(getContext()),
                {LLVM::LLVMPointerType::get(getContext())},
                false
            )
        );
    }

    Assert(op.getBody().empty(), "AbortInfoInliningPass must be run before LLVM lowering");
    LLVM::CallOp::create(r, op.getLoc(), mlir::TypeRange(), StringRef(name), a.getAbortInfo());
    return LLVM::UnreachableOp::create(r, op.getLoc());
});

LOWERING(CallOp, {
    SmallVector<mlir::Value> args;
    SmallVector<mlir::Type> arg_types;
    auto tc = getTypeConverter<mlir::LLVMTypeConverter>();

    args.push_back(a.getAddr());
    for (auto arg : a.getArgs()) args.push_back(arg);
    for (auto arg : op.getProcType().getInputs()) arg_types.push_back(arg);

    mlir::Type result;
    if (op.getNumResults() == 0) {
        result = LLVM::LLVMVoidType::get(getContext());
    } else if (op.getNumResults() == 1) {
        result = tc->convertType(op.getResult(0).getType());
    } else {
        SmallVector<mlir::Type> elems;
        if (failed(tc->convertTypes(op.getResultTypes(), elems)))
            return SmallVector<mlir::Value>();
        result = LLVM::LLVMStructType::getLiteral(getContext(), elems);
    }

    auto fty = LLVM::LLVMFunctionType::get(
        result,
        arg_types,
        a.getVariadic()
    );

    auto call = LLVM::CallOp::create(r, op.getLoc(), fty, args);
    call.setCConv(op.getCc());

    // Preserve argument and return value. attributes.
    PropagateArgAndResultAttrs(call, op);

    // Split aggregate returns.
    if (op.getNumResults() > 1) {
        SmallVector<mlir::Value> results;
        for (i64 i = 0; i < i64(op.getNumResults()); i++)
            results.push_back(LLVM::ExtractValueOp::create(r, op.getLoc(), call.getResult(), i));
        return results;
    }

    return op.getNumResults() == 0 ? SmallVector<mlir::Value>() : SmallVector{call.getResult()};
});

ERASE_OP(DisengageOp);
ERASE_OP(EngageOp);
ERASE_OP(EngageCopyOp);

LOWERING(FrameSlotOp, {
    return LLVM::AllocaOp::create(
        r,
        op->getLoc(),
        LLVM::LLVMPointerType::get(r.getContext()),
        r.getI8Type(),
        LLVM::ConstantOp::create(r, op->getLoc(), a.getBytesAttr()),
        unsigned(a.getAlignment().getInt())
    );
});

LOWERING(LoadOp, {
    return LLVM::LoadOp::create(
        r,
        op.getLoc(),
        getTypeConverter()->convertType(op.getResult().getType()),
        a.getAddr(),
        unsigned(a.getAlignment().getInt())
    );
});

LOWERING(NilOp, {
    return LLVM::ZeroOp::create(
        r,
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
    if (op.getNumResults() == 0) {
        ret = LLVM::LLVMVoidType::get(getContext());
    }  else if (op.getNumResults() == 1) {
        ret = tc->convertType(old.getResult(0));
    } else {
        SmallVector<mlir::Type> results;
        if (failed(tc->convertTypes(op.getResultTypes(), results)))
            return LLVM::LLVMFuncOp{};
        ret = LLVM::LLVMStructType::getLiteral(getContext(), results);
    }

    // Add the arguments.
    for (auto [i, arg] : enumerate(old.getInputs()))
        sc.addInputs(unsigned(i), tc->convertType(arg));

    // Build the new function.
    auto llvm_fty = LLVM::LLVMFunctionType::get(
        ret,
        sc.getConvertedTypes(),
        op.getVariadic()
    );

    auto func = LLVM::LLVMFuncOp::create(
        r,
        op.getLoc(),
        op.getName(),
        llvm_fty,
        op.getLinkage().getLinkage(),
        /*dsoLocal=*/false,
        op.getCc()
    );

    // We don’t support exception handling.
    func.setNoUnwind(true);

    // Preserve argument and return value attributes.
    PropagateArgAndResultAttrs(func, op);

    // And inline the body.
    r.inlineRegionBefore(op.getBody(), func.getBody(), func.end());
    if (failed(r.convertRegionTypes(&func.getBody(), *getTypeConverter(), &sc))) func = {};
    return func;
});

LOWERING(ProcRefOp, {
    return LLVM::AddressOfOp::create(
        r,
        op.getLoc(),
        LLVM::LLVMPointerType::get(r.getContext()),
        a.getProcName()
    );
});

LOWERING(RetOp, {
    Value res;

    if (a.getVals().size() == 1) {
        res = a.getVals().front();
    } else if (a.getVals().size() > 1) {
        res = LLVM::UndefOp::create(
            r,
            op.getLoc(),
            LLVM::LLVMStructType::getLiteral(
                getContext(),
                SmallVector<mlir::Type>{a.getVals().getTypes()}
            )
        );

        for (auto [i, v] : llvm::enumerate(a.getVals())) {
            res = LLVM::InsertValueOp::create(r,
                op.getLoc(),
                res,
                v,
                i64(i)
            );
        }
    }

    return LLVM::ReturnOp::create(r, op.getLoc(), res);
});

LOWERING(SAddOvOp, { return LowerOverflowOp<LLVM::SAddWithOverflowOp>(op, a, r); });
LOWERING(SMulOvOp, { return LowerOverflowOp<LLVM::SMulWithOverflowOp>(op, a, r); });
LOWERING(SSubOvOp, { return LowerOverflowOp<LLVM::SSubWithOverflowOp>(op, a, r); });
LOWERING(StoreOp, {
    return LLVM::StoreOp::create(
        r,
        op->getLoc(),
        a.getValue(),
        a.getAddr(),
        unsigned(a.getAlignment().getInt())
    );
});

ERASE_OP(UnwrapOp);

struct OptionalUnwrapCheckPass final : mlir::PassWrapper<OptionalUnwrapCheckPass, mlir::Pass> {
    CodeGen& cg;
    OptionalUnwrapCheckPass(CodeGen& cg) : cg(cg) {}
    bool canScheduleOn(mlir::RegisteredOperationName op) const override {
        auto name = op.getStringRef();
        return name == mlir::ModuleOp::getOperationName() or name == ir::ProcOp::getOperationName();
    }

    void runOnOperation() override;
    void CheckProcedure(ir::ProcOp proc);
};

enum struct EngagedState {
    // Optional is not engaged.
    Disengaged,

    // Optional is definitely engaged.
    Engaged,

    // We don’t know whether this is engaged.
    Unknown,
};


#define DEBUG_OPTIONAL_ANALYSIS(...)

class OptionalLatticePoint final : public mlir::dataflow::AbstractDenseLattice {
    llvm::SmallDenseMap<Value, EngagedState> state;

public:
    using AbstractDenseLattice::AbstractDenseLattice;
    bool operator==(const OptionalLatticePoint& other) const {
        return state == other.state;
    }

    // Get the state of a value at this lattice point.
    auto get(Value v) const -> EngagedState {
        auto it = state.find(v);
        if (it == state.end()) return EngagedState::Unknown;
        return it->second;
    }

    // Join two lattices.
    auto join(const AbstractDenseLattice &rhs) -> ChangeResult override {
        auto& lattice = static_cast<const OptionalLatticePoint&>(rhs);
        auto changed = NoChange;
        for (auto [value, s] : lattice.state) {
            auto it = state.find(value);
            if (it == state.end()) {
                state[value] = s;
                changed = Change;
            } else {
                auto join = it->second == s ? s : EngagedState::Unknown;
                if (join != it->second) {
                    it->getSecond() = join;
                    changed = Change;
                }
            }
        }

        DEBUG_OPTIONAL_ANALYSIS(
            llvm::dbgs() << "================================\n";
            llvm::dbgs() << "Joining: \n    * ";
            rhs.getAnchor().print(llvm::dbgs());
            llvm::dbgs() << "\n    * ";
            getAnchor().print(llvm::dbgs());
            print(llvm::dbgs());
            llvm::dbgs() << "\n================================\n";
        )

        return changed;
    }

    // Dump this lattice point.
    void print(llvm::raw_ostream& os) const override {
        for (auto [value, s] : state)
            os << "\n    +" << value << ": " << enchantum::to_string(s);
    }

    // Reset this lattice.
    auto reset() -> ChangeResult {
        if (state.empty()) return NoChange;
        state.clear();
        return Change;
    }

    // Set a value for this lattice.
    auto set(mlir::Value val, EngagedState engaged) -> ChangeResult {
        if (not state.contains(val)) {
            state[val] = engaged;
            return Change;
        }

        auto& st = state[val];
        if (st == engaged) return NoChange;
        st = engaged;
        return Change;
    }
};

struct OptionalAnalysis final : mlir::dataflow::DenseForwardDataFlowAnalysis<OptionalLatticePoint> {
    ArrayRef<ir::UnwrapOp> unwraps;
    OptionalAnalysis(mlir::DataFlowSolver &solver, ArrayRef<ir::UnwrapOp> unwraps)
        : DenseForwardDataFlowAnalysis(solver), unwraps{unwraps} {}

    auto visitOperation(
        Operation* op,
        const OptionalLatticePoint& before,
        OptionalLatticePoint* after
    ) -> mlir::LogicalResult override {
        DEBUG_OPTIONAL_ANALYSIS(
            std::println("Visiting {}", op->getName().getStringRef());
        )

        // Propagate state.
        auto changed = after->join(before);

        // Record engage/disengage.
        changed |= llvm::TypeSwitch<Operation*, ChangeResult>(op)
            TCase(ir::DisengageOp, op, after->set(op.getOptional(), EngagedState::Disengaged))
            TCase(ir::EngageOp, op, after->set(op.getOptional(), EngagedState::Engaged))
            TCase(ir::EngageCopyOp, op, after->set(op.getOptional(), after->get(op.getCopyFrom())))
            .Default(NoChange);

        // Propagate to the next operation.
        propagateIfChanged(after, changed);
        return success();
    }

    void setToEntryState(OptionalLatticePoint *lattice) override {
        auto changed = lattice->reset();
        for (auto u : unwraps) changed |= lattice->set(u.getOptional(), EngagedState::Disengaged);
        propagateIfChanged(lattice, changed);
    }
};

struct AbortInfoInliningPass final : mlir::PassWrapper<AbortInfoInliningPass, mlir::Pass> {
    CodeGen& cg;
    AbortInfoInliningPass(CodeGen& cg) : cg(cg) {}
    bool canScheduleOn(mlir::RegisteredOperationName op) const override {
        auto name = op.getStringRef();
        return name == mlir::ModuleOp::getOperationName() or name == ir::ProcOp::getOperationName();
    }

    void runOnOperation() override {
        mlir::OpBuilder b{&getContext()};
        mlir::IRRewriter r{b};
        getOperation()->walk([&](ir::AbortOp op) {
            if (op.getBody().empty()) return;
            r.inlineBlockBefore(&op.getBody().front(), op, op.getAbortInfo());
        });
    }
};
} // namespace srcc::cg::lowering

using namespace srcc::cg::lowering;

void LoweringPass::runOnOperation() {
    using namespace mlir;
    LLVMConversionTarget target{getContext()};
    LLVMTypeConverter tc{&getContext()};
    RewritePatternSet patterns{&getContext()};

    arith::populateArithToLLVMConversionPatterns(tc, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);

    patterns.add< // clang-format off
        AbortOpLowering,
        DisengageOpLowering,
        EngageOpLowering,
        EngageCopyOpLowering,
        CallOpLowering,
        FrameSlotOpLowering,
        LoadOpLowering,
        NilOpLowering,
        ProcOpLowering,
        ProcRefOpLowering,
        RetOpLowering,
        SAddOvOpLowering,
        SMulOvOpLowering,
        SSubOvOpLowering,
        StoreOpLowering,
        UnwrapOpLowering
    >(cg, tc); // clang-format on

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    // Disallow rollback because we don’t need it and disallowing it
    // is much more efficient.
    mlir::ConversionConfig config;
    config.allowPatternRollback = false;

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns), config)))
        return signalPassFailure();
}

void OptionalUnwrapCheckPass::runOnOperation() {
    using namespace mlir;
    auto op = getOperation();
    if (auto proc = dyn_cast<ir::ProcOp>(op)) return CheckProcedure(proc);
    op->walk([&](ir::ProcOp proc) { CheckProcedure(proc); });
}

void OptionalUnwrapCheckPass::CheckProcedure(ir::ProcOp proc) {
    if (proc.getBody().empty()) return;

    // Collect unwraps.
    SmallVector<ir::UnwrapOp> unwraps;
    proc->walk([&](ir::UnwrapOp op) { unwraps.push_back(op); });
    if (unwraps.empty()) return;

    // Perform the dataflow analysis on this procedure.
    mlir::DataFlowSolver solver;

    // These passes are required for the analysis to work at all.
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<mlir::dataflow::DeadCodeAnalysis>();

    // Run the analysis.
    solver.load<OptionalAnalysis>(unwraps);
    if (failed(solver.initializeAndRun(proc))) {
        cg.ICE(SLoc::Decode(proc.getLoc()), "Failed to run dataflow analysis on procedure");
        signalPassFailure();
        return;
    }

    // TODO:
    //
    // - Treat a pointer as disengaged if it is passed to a function.
    // - Handle pointer aliasing, i.e. compute what pointers are obviously based
    //   on other pointers; disengaging one of them should disengage all of them.
    // - Handle multiple IR values that represent the same pointer, e.g. we could
    //   be doing '%some_ptr + 4' twice. Engaging either should engage the other
    //   and the same applies to disengaging.

    // Determine whether the optional is engaged at each unwrap.
    for (auto unwrap : unwraps) {
        auto point = solver.getProgramPointAfter(unwrap);
        auto state = solver.lookupState<OptionalLatticePoint>(point);
        DEBUG_OPTIONAL_ANALYSIS(
            std::println("State is: {}", enchantum::to_string(state->get(unwrap.getOptional())));
        )

        if (state->get(unwrap.getOptional()) != EngagedState::Engaged) {
            cg.Error(SLoc::Decode(unwrap->getLoc()), "Optional might be nil when accessed here");
            signalPassFailure();
        }
    }
}

auto CodeGen::emit_llvm(llvm::TargetMachine& machine) -> std::unique_ptr<llvm::Module> {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(std::make_unique<AbortInfoInliningPass>(*this));
    pm.addPass(std::make_unique<LoweringPass>(*this));

    if (pm.run(mlir_module).failed()) {
        if (not tu.context().diags().has_error()) ICE(SLoc(), "Failed to lower module to LLVM IR");
        return nullptr;
    }

    auto m = mlir::translateModuleToLLVMIR(mlir_module, tu.llvm_context, tu.name);
    m->setTargetTriple(machine.getTargetTriple());
    m->setDataLayout(machine.createDataLayout());
    return m;
}

bool CodeGen::finalise_for_constant_evaluation(ir::ProcOp proc) {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(std::make_unique<OptionalUnwrapCheckPass>(*this));
    pm.addPass(std::make_unique<AbortInfoInliningPass>(*this));
    // For some reason this pass sometimes produces null values...
    // This is possibly LLVM bug 153906.
    //pm.addPass(mlir::createRemoveDeadValuesPass());
    if (pm.run(proc).succeeded()) return true;
    if (not tu.context().diags().has_error()) {
        ICE(SLoc(), "Failed to finalise IR");
        proc->dumpPretty();
    }

    return false;
}


bool CodeGen::finalise() {
    mlir::PassManager pm{&mlir};
    pm.enableVerifier(true);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());
    pm.addPass(std::make_unique<OptionalUnwrapCheckPass>(*this));
    if (pm.run(mlir_module).succeeded()) return true;
    if (not tu.context().diags().has_error()) ICE(SLoc(), "Failed to finalise IR");
    return false;
}
