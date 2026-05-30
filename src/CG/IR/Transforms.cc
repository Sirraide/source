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

#define DEBUG_MOVE_ANALYSIS(...) /*                             \
    do {                                                        \
        if (std::getenv("DEBUG_MOVE_ANALYSIS")) { __VA_ARGS__ } \
    } while (false) */

#define DEBUG_OPTIONAL_ANALYSIS(...) // __VA_ARGS__
#define TCase(type, name, ...) .Case<type>([&](auto&& name) { return __VA_ARGS__; })

using namespace srcc;
using namespace srcc::cg;

namespace srcc::cg::transforms {
using mlir::success;
using mlir::ChangeResult;
using enum mlir::ChangeResult;

template <typename Derived>
struct ProcedurePass : mlir::PassWrapper<Derived, mlir::Pass> {
    CodeGen& cg;
    ProcedurePass(CodeGen& cg) : cg(cg) {}
    bool canScheduleOn(mlir::RegisteredOperationName op) const override {
        auto name = op.getStringRef();
        return name == mlir::ModuleOp::getOperationName() or name == ir::ProcOp::getOperationName();
    }

    void runOnOperation() override {
        using namespace mlir;
        auto op = self()->getOperation();
        if (auto proc = dyn_cast<ir::ProcOp>(op)) return self()->CheckProcedure(proc);
        op->walk([&](ir::ProcOp proc) { self()->CheckProcedure(proc); });
    }

    auto self() -> Derived* {
        return static_cast<Derived*>(this);
    }
};

/// ============================================================================
///  Use after Move Analysis
/// ============================================================================
struct UseAfterMoveCheckPass final : ProcedurePass<UseAfterMoveCheckPass> {
    using ProcedurePass::ProcedurePass;
    void CheckProcedure(ir::ProcOp proc);
};

enum class MovedState {
    /// We don’t know what this is; not used in the lattice.
    Unknown,
    /// Value is defined.
    Defined,
    /// Value is definitely moved.
    Moved,
    /// Value is moved on some control-flow paths.
    PotentiallyMoved,
};

class MoveLatticePoint final : public mlir::dataflow::AbstractDenseLattice {
    llvm::SmallDenseMap<Value, MovedState> tracked_values;
    llvm::SmallDenseMap<Value, Operation*> movers;

public:
    using AbstractDenseLattice::AbstractDenseLattice;
    using enum MovedState;

    auto defined(Value v) -> ChangeResult {
        movers.erase(v);
        return set_state(v, Defined);
    }


    auto moved(Value v, Operation* mover) -> ChangeResult {
        movers.try_emplace(v, mover);
        return set_state(v, Moved);
    }

    auto get_moving_op(Value v) const -> Operation* {
        return movers.at(v);
    }

    auto get_state(Value v) const -> MovedState {
        auto it = tracked_values.find(v);
        if (it == tracked_values.end()) return Unknown;
        return it->second;
    }

    auto join(const AbstractDenseLattice &lattice) -> ChangeResult override {
        auto& before = static_cast<const MoveLatticePoint&>(lattice);
        auto changed = NoChange;

        DEBUG_MOVE_ANALYSIS(
            llvm::dbgs() << "================================\n";
            llvm::dbgs() << "RHS: ";
            before.getAnchor().print(llvm::dbgs());
            llvm::dbgs() << '\n';
            before.print(llvm::dbgs());
            llvm::dbgs() << "THIS: ";
            getAnchor().print(llvm::dbgs());
            llvm::dbgs() << '\n';
            print(llvm::dbgs());
        );

        // Merge values into this lattice.
        for (auto [value, state] : before.tracked_values) {
            // Value is not in this lattice; just add it.
            //
            // This does mean that join(Unknown, Moved) = Moved, but in order
            // for that to happen, we must have at some point moved an Unknown
            // value (either it was Unknown and then Moved or Defined and then
            // Moved, but defining it entails deleting the original value, which
            // is itself a move), which means the program is ill-formed anyway.
            auto [it, inserted] = tracked_values.try_emplace(value, state);
            if (inserted) {
                changed = Change;
                if (state == Moved or state == PotentiallyMoved)
                     movers.insert({value, before.movers.at(value)});
            }

            // Value is in the lattice, merge them; if both are the same, do nothing;
            // otherwise, set it to PotentiallyMoved as all possible combninations result
            // in that.
            else if (it->second != state and it->second != PotentiallyMoved) {
                it->second = PotentiallyMoved;
                changed = Change;
                if (state == Moved or state == PotentiallyMoved)
                     movers.insert({value, before.movers.at(value)});
            }
        }

        DEBUG_MOVE_ANALYSIS(
            llvm::dbgs() << "JOINED:\n";
            print(llvm::dbgs());
        );

        return changed;
    }

    void print(llvm::raw_ostream& os) const override {
        for (auto [v, s] : tracked_values) {
            os << "  - ";
            v.print(os);
            os << " = " << enchantum::to_string(s) << "\n";
        }
    }

    auto reset() -> ChangeResult {
        if (tracked_values.empty()) return NoChange;
        movers.clear();
        tracked_values.clear();
        return Change;
    }

private:
    auto set_state(Value val, MovedState state) -> ChangeResult {
        auto [it, inserted] = tracked_values.try_emplace(val, state);
        if (inserted) return Change;
        if (it->second == state) return NoChange;
        it->second = state;
        return Change;
    }
};

// Check if this use of a value is a definition.
static auto UseIsDef(Value val, Operation* user) -> bool {
    if (auto s = dyn_cast<ir::StoreOp>(user)) return s.getAddr() == val;
    if (auto m = dyn_cast<mlir::LLVM::MemcpyOp>(user)) return m.getDst() == val;
    if (auto m = dyn_cast<mlir::LLVM::MemsetOp>(user)) return m.getDst() == val;
    return false;
};

// Check if this use of a value is a move.
static auto UseIsMove(Value val, Operation* user) -> bool {
    if (auto m = dyn_cast<ir::MoveOp>(user)) return m.getValue() == val;
    if (auto d = dyn_cast<ir::DeleteOp>(user)) return d.getAddr() == val;
    return false;
};

struct DeletionAnalysis final : mlir::dataflow::DenseForwardDataFlowAnalysis<MoveLatticePoint> {
    ir::ProcOp proc;
    DeletionAnalysis(mlir::DataFlowSolver &solver, ir::ProcOp proc)
        : DenseForwardDataFlowAnalysis(solver), proc{proc} {}

    auto visitOperation(
        Operation* op,
        const MoveLatticePoint& before,
        MoveLatticePoint* after
    ) -> mlir::LogicalResult override {
        auto changed = after->join(before);
        if (auto a = dyn_cast<mlir::LLVM::AllocaOp>(op)) changed |= after->defined(a);
        if (auto r = dyn_cast<ir::RetainOp>(op)) changed |= after->defined(r);
        for (auto v : op->getOperands()) {
            if (UseIsMove(v, op)) changed |= after->moved(v, op);
            else if (UseIsDef(v, op)) changed |= after->defined(v);
        }

        propagateIfChanged(after, changed);
        return success();
    }

    void setToEntryState(MoveLatticePoint *lattice) override {
        auto changed = lattice->reset();

        // Initialise all pointer parameters as defined, except for out params;
        // those should be marked as Moved instead so we drop implicit deletes
        // when assigning to them.
        for (auto [i, arg] : enumerate(proc.getArguments())) {
            if (not isa<mlir::LLVM::LLVMPointerType>(arg.getType())) continue;
            auto attrs = proc.getCallArgAttrs(u32(i));
            if (attrs and attrs.contains(ir::OutParamAttrName)) changed |= lattice->moved(arg, proc);
            else changed |= lattice->defined(arg);
        }

        propagateIfChanged(lattice, changed);
    }
};


void UseAfterMoveCheckPass::CheckProcedure(ir::ProcOp proc) {
    if (proc.getBody().empty()) return;

    // Perform the dataflow analysis on this procedure.
    mlir::DataFlowSolver solver;

    // These passes are required for the analysis to work at all.
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<mlir::dataflow::DeadCodeAnalysis>();

    // Run the analysis.
    solver.load<DeletionAnalysis>(proc);
    if (failed(solver.initializeAndRun(proc))) {
        cg.ICE(SLoc::Decode(proc.getLoc()), "Failed to run dataflow analysis on procedure");
        signalPassFailure();
        return;
    }

    // For each 'delete' operation, check if the moved value is already
    // moved by the time we get there, and remove it if so.
    SmallVector<ir::DeleteOp> ops_to_erase;
    llvm::SmallDenseMap<Value, mlir::LLVM::AllocaOp> need_delete_flag;
    llvm::SmallPtrSet<Value, 2> already_diagnosed_out_params;
    proc.getBody().walk([&](Operation* op){
        using enum MovedState;
        auto point = solver.getProgramPointBefore(op);
        auto state = solver.lookupState<MoveLatticePoint>(point);
        if (not state) return;
        auto DiagnoseIfMoved = [&](Value v) {
            auto s = state->get_state(v);
            if (s == MovedState::Moved or s == PotentiallyMoved) {
                auto mover = state->get_moving_op(v);
                bool was_deleted = isa<ir::DeleteOp>(mover);
                bool is_delete = isa<ir::DeleteOp>(op);

                // Hack: if the 'mover' is a ProcOp, then this is an uninitialised
                // out parameter.
                if (isa<ir::ProcOp>(mover)) {
                    cg.Error(
                        SLoc::Decode(op->getLoc()),
                        " '%1(out%)' parameter must be written to before it can be {}",
                        is_delete ? "deleted" : "used"
                    );
                    return;
                }

                cg.Error(
                    SLoc::Decode(op->getLoc()),
                    "{} of {} value",
                    is_delete ? "'%1(delete%)'" : "Use",
                    was_deleted ? "deleted"sv : "moved"sv
                );

                cg.Note(
                    SLoc::Decode(mover->getLoc()),
                    "{} here",
                    was_deleted ? "Deleted"sv : "Moved"sv
                );
            }
        };

        // Handle delete.
        if (auto del = dyn_cast<ir::DeleteOp>(op)) {
            auto v = del.getAddr();
            auto s = state->get_state(v);

            // Implicit delete:
            //
            //   - If the value is already Moved, drop the delete.
            //   - If the value is PotentiallyMoved, add a delete flag.
            //   - Otherwise, assert the value is defined; the compiler
            //     should not generate implicit deletes for unknown values.
            if (del.getImplicit()) {
                if (s == Moved) ops_to_erase.push_back(del);
                else if (s == PotentiallyMoved) need_delete_flag.insert({v, nullptr});
                else Assert(s == Defined, "Compiler inserted 'delete' for unknown value?");
                return;
            }

            // Explicit delete. This is like a use, except that deleting an
            // unknown value is an error.
            if (s == Unknown) {
                cg.Error(SLoc::Decode(op->getLoc()), "'%1(delete%)' of unowned pointer");
                return;
            }

            // Fall through to the regular use checking code below.
        }

        // Allow ptradd.
        if (isa<mlir::LLVM::GEPOp>(op)) return;

        // Any other operation that references a moved-from value is an error.
        for (auto v : op->getOperands())
            if (not UseIsDef(v, op))
                DiagnoseIfMoved(v);

        // If this is a return op, check that all out parameter are defined.
        if (isa<ir::RetOp>(op)) {
            for (auto [i, arg] : enumerate(proc.getArguments())) {
                auto attrs = proc.getCallArgAttrs(u32(i));
                if (
                    attrs and attrs.contains(ir::OutParamAttrName) and
                    state->get_state(arg) != Defined and
                    already_diagnosed_out_params.insert(arg).second
                ) {
                    cg.Error(
                        SLoc::Decode(arg.getLoc()),
                        "'%1(out%)' parameter is not written to on all control paths"
                    );
                }
            }
        }
    });

    // Delete implicit deletes of moved-from values. Do this before rewriting
    // deletes of potentially-moved values.
    for (auto o : ops_to_erase) o->erase();

    // For each potentially-moved value, insert an alloca at the start of the
    // function to keep track of whether it was moved at runtime; then, update
    // each definition to set it to 'false', and each move to set it to 'true'.
    if (need_delete_flag.empty()) return;
    mlir::OpBuilder b{proc.getBody()};
    auto one_64 = mlir::arith::ConstantIntOp::create(b, proc.getLoc(), 1, 64);
    auto one_8 = mlir::arith::ConstantIntOp::create(b, proc.getLoc(), 1, 8);
    auto ptr = mlir::LLVM::LLVMPointerType::get(proc->getContext());
    for (auto& [val, moved_flag] : need_delete_flag) {
        moved_flag = mlir::LLVM::AllocaOp::create(
            b,
            val.getLoc(),
            ptr,
            b.getI8Type(),
            one_64,
            1
        );

        // Initialise it to 1. This is necessary if e.g. the value is a 'move'
        // parameter, in which case we may never see a store to it before it is
        // deleted.
        ir::StoreOp::create(b, val.getLoc(), moved_flag, one_8, Align(1));
    }

    struct Entry {
        Operation* op;
        Value affected_value;
        bool live;
    };

    SmallVector<Entry> ops_to_rewrite;
    proc->walk([&](Operation* op){
        for (auto v : op->getOperands()) {
            if (not need_delete_flag.contains(v)) continue;
            if (UseIsMove(v, op)) ops_to_rewrite.emplace_back(op, v, false);
            else if (UseIsDef(v, op)) ops_to_rewrite.emplace_back(op, v, true);
        }
    });

    for (const auto& [op, affected_value, live] : ops_to_rewrite) {
        // If the op is a delete, make it conditional if the value is
        // potentially-moved here.
        auto flag = need_delete_flag.at(affected_value);
        if (auto d = dyn_cast<ir::DeleteOp>(op)) {
            auto point = solver.getProgramPointBefore(op);
            auto state = solver.lookupState<MoveLatticePoint>(point);
            if (state and state->get_state(affected_value) == MovedState::PotentiallyMoved)
                d.setDeleterFlag(flag);
        }

        // Insert *after* the op in case it is a delete (because it might
        // read from the flag).
        mlir::OpBuilder b{op};
        b.setInsertionPointAfter(op);
        auto loc = op->getLoc();
        auto val = mlir::arith::ConstantIntOp::create(b, loc, live, 8);
        ir::StoreOp::create(b, loc, flag, val, Align(1));
    }
}


/// ============================================================================
///  Optional Analysis
/// ============================================================================
struct OptionalUnwrapCheckPass final : ProcedurePass<OptionalUnwrapCheckPass> {
    using ProcedurePass::ProcedurePass;
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
    pm.addPass(std::make_unique<UseAfterMoveCheckPass>(*this));
    pm.addPass(std::make_unique<OptionalUnwrapCheckPass>(*this));
}