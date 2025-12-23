#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <enchantum/enchantum.hpp>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>

#define TCase(type, name, ...) .Case<type>([&](auto&& name) { return __VA_ARGS__; })

using namespace srcc;
using namespace srcc::cg;

namespace srcc::cg::transforms {
using mlir::success;
using mlir::ChangeResult;
using enum mlir::ChangeResult;

/// ============================================================================
///  Optional Analysis
/// ============================================================================
#define DEBUG_OPTIONAL_ANALYSIS(...) // __VA_ARGS__

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
            os << "\n    + " << value << ": " << enchantum::to_string(s);
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
        if (not state) continue; // Not sure why this can happen but give up.
        DEBUG_OPTIONAL_ANALYSIS(
            std::println("State is: {}", enchantum::to_string(state->get(unwrap.getOptional())));
        )

        if (state->get(unwrap.getOptional()) != EngagedState::Engaged) {
            cg.Error(SLoc::Decode(unwrap->getLoc()), "Optional might be nil when accessed here");
            signalPassFailure();
        }
    }
}

/// ============================================================================
///  Critical Edge Simplification
/// ============================================================================
#define DEBUG_CRITICAL_EDGE_INLINING(...)

/// Pass that simplifies trivial critical edges (i.e. blocks that contain
/// only a conditional branch on a block argument) to facilitate dataflow
/// anlaysis.
struct TrivialCriticalEdgeInliningPass final : mlir::PassWrapper<TrivialCriticalEdgeInliningPass, mlir::Pass> {
    CodeGen& cg;
    TrivialCriticalEdgeInliningPass(CodeGen& cg) : cg(cg) {}
    bool canScheduleOn(mlir::RegisteredOperationName op) const override {
        auto name = op.getStringRef();
        return name == mlir::ModuleOp::getOperationName() or name == ir::ProcOp::getOperationName();
    }

    void runOnOperation() override {
        DEBUG_CRITICAL_EDGE_INLINING(
            llvm::dbgs() << "================== IR ======================\n";
            getOperation()->dump();
            llvm::dbgs() << "================== IR ======================\n";
        )

        mlir::OpBuilder b{&getContext()};
        mlir::IRRewriter r{b};
        getOperation()->walk([&](mlir::Block* bb) { MaybeInline(r, bb); });
    }

    void MaybeInline(mlir::IRRewriter& r, mlir::Block* bb);
};

void TrivialCriticalEdgeInliningPass::MaybeInline(mlir::IRRewriter& r, mlir::Block* bb) {
    if (not bb->mightHaveTerminator()) return;
    auto branch = dyn_cast_if_present<mlir::cf::CondBranchOp>(bb->getTerminator());
    if (
        not branch or
        bb->getNumArguments() != 1 or
        bb->empty() or
        &bb->front() != branch or
        branch.getCondition() != bb->getArgument(0)
    ) return;

    DEBUG_CRITICAL_EDGE_INLINING (
        llvm::dbgs() << "========================================\n";
        llvm::dbgs() << "Processing:\n";
        bb->print(llvm::dbgs());
        for (auto pred : bb->getPredecessors()) {
            llvm::dbgs() << "\n  -> ";
            pred->printAsOperand(llvm::dbgs());
        }
        llvm::dbgs() << "\n----------------------------------------\n";
    )

    // This is a block of the form
    //
    //   bb(i1 %x):
    //       br %x to bbX else bbY
    //
    // Inline the branch into the parents of this block and delete it.
    auto bb_arg = bb->getArgument(0);
    for (auto pred : llvm::make_early_inc_range(bb->getPredecessors())) {
        DEBUG_CRITICAL_EDGE_INLINING (
            llvm::dbgs() << "Predecessor:\n";
            pred->print(llvm::dbgs());
            llvm::dbgs() << "----------------------------------------\n";
        )

        auto term = pred->getTerminator();
        if (auto pred_br = dyn_cast<mlir::cf::BranchOp>(term)) {
            mlir::IRMapping m;
            m.map(bb_arg, pred_br.getDestOperands().front());
            r.setInsertionPointToEnd(pred);
            auto clone = branch->clone(m);

            DEBUG_CRITICAL_EDGE_INLINING (
                llvm::dbgs() << "  * Replacing: ";
                term->print(llvm::dbgs());
                llvm::dbgs() << "\n  * With: ";
                clone->print(llvm::dbgs());
            )

            r.insert(clone);
            r.eraseOp(term);

            DEBUG_CRITICAL_EDGE_INLINING (
                llvm::dbgs() << "----------------------------------------\n";
                llvm::dbgs() << "After Replacement:\n";
                pred->print(llvm::dbgs());
                llvm::dbgs() << "----------------------------------------\n";
            )
            continue;
        }

        DEBUG_CRITICAL_EDGE_INLINING (
            std::println(stderr, "REPLACED COND BR");
            llvm::dbgs() << "----------------------------------------\n";
        )

        // This is a conditional branch. Note that we can only inline this if
        // the block argument is a constant 'true' or 'false' (because then we
        // can just discard one of the branches), e.g. if we have
        //
        //   bb1:
        //       %y = ...
        //       br %y to bb5(%a) else bb2(true)
        //
        //   bb2(i1 %x):
        //       br %x to bb3(%b) else bb4(%c)
        //
        // we fold this to
        //
        //   bb1:
        //       %y = ...
        //       br %y to bb5(%a) else bb3(%b)
        //
        auto pred_br = cast<mlir::cf::CondBranchOp>(term);
        auto TryReplaceBranchDest = [&](unsigned branch_to_replace) {
            if (pred_br->getSuccessor(branch_to_replace) != bb) return;
            auto pred_args = branch_to_replace == 0
                ? pred_br.getTrueDestOperandsMutable()
                : pred_br.getFalseDestOperandsMutable();

            // Take care to map the block argument to whatever we passed to it
            // if it appears in the arguments of the branch, and also check that
            // it is indeed a constant.
            Assert(pred_args.size() == 1);
            auto block_arg = pred_args[0].get();
            auto cond_op_res = dyn_cast<mlir::OpResult>(block_arg);
            if (not cond_op_res) return;
            auto constant = dyn_cast<mlir::arith::ConstantIntOp>(cond_op_res.getOwner());
            if (not constant) return;

            // Determine what branch we’re replacing this with.
            bool is_true = cast<mlir::IntegerAttr>(constant.getValue()).getValue().getBoolValue();
            auto branch_to_take = is_true ? 0u : 1u;
            auto replacement_args = is_true ? branch.getTrueDestOperands() : branch.getFalseDestOperands();

            // Replace the successor.
            pred_br.setSuccessor(branch.getSuccessor(branch_to_take), branch_to_replace);
            pred_args.clear();
            for (auto arg : replacement_args) {
                if (arg == bb_arg) pred_args.append(block_arg);
                else pred_args.append(arg);
            }
        };

        TryReplaceBranchDest(0);
        TryReplaceBranchDest(1);
    }

    if (bb->getPredecessors().empty()) r.eraseBlock(bb);
}
}

void CodeGen::AddRequiredTransformPasses(mlir::PassManager &pm) {
    using namespace transforms;
    pm.addPass(std::make_unique<TrivialCriticalEdgeInliningPass>(*this));
    pm.addPass(mlir::createCanonicalizerPass()); // Clean up branches we generated.
    pm.addPass(std::make_unique<OptionalUnwrapCheckPass>(*this));
}