#include <srcc/AST/AST.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Constants.hh>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

struct CodeGen::Printer {
    CodeGen& cg;
    SmallUnrenderedString out;
    DenseMap<Operation*, i64> global_ids;
    DenseMap<mlir::BlockArgument, i64> arg_ids;
    DenseMap<Block*, i64> block_ids;
    DenseMap<Operation*, i64> inst_ids;
    i64 global = 0;
    ProcOp curr_proc{};
    bool always_include_types;

    Printer(CodeGen& cg, bool always_include_types = false) : cg{cg}, always_include_types{always_include_types} {}
    void print(mlir::ModuleOp module);
    void print_op(Operation* op);
    void print_procedure(ProcOp op);
    void print_top_level_op(Operation* op);
    auto ops(mlir::ValueRange args, bool include_types = true) -> SmallUnrenderedString;
    auto val(Value v, bool include_type = true) -> SmallUnrenderedString;

    auto Id(auto& map, auto* ptr) -> i64 {
        auto it = map.find(ptr);
        return it == map.end() ? -1 : it->second;
    }
};

static auto FormatType(mlir::Type ty) -> std::string {
    std::string tmp;
    if (isa<mlir::LLVM::LLVMPointerType>(ty)) {
        tmp += "%6(ptr%)";
    } else if (auto ui = dyn_cast<mlir::IntegerType>(ty); ui and ui.isUnsignedInteger(1)) {
        tmp += "%6(bool%)";
    } else if (auto a = dyn_cast<mlir::LLVM::LLVMArrayType>(ty)) {
        tmp += std::format("{}%1([%5({}%)]%)", FormatType(a.getElementType()), a.getNumElements());
    } else if (auto t = dyn_cast<mlir::TupleType>(ty)) {
        tmp += std::format("%1(({})%)", utils::join(t.getTypes(), ", ", "{}", FormatType));
    } else {
        llvm::raw_string_ostream os{tmp};
        tmp += "%6(";
        ty.print(os);
        tmp += "%)";
    }
    return tmp;
}

static auto IsInlineOp(Operation* op) {
    return isa<mlir::arith::ConstantIntOp, ProcRefOp, NilOp, TupleOp, mlir::LLVM::AddressOfOp, ReturnPointerOp>(op);
}

auto CodeGen::dump(bool verbose) -> SmallUnrenderedString {
    Printer p{*this, verbose};
    p.print(mlir_module);
    return std::move(p.out);
}

auto CodeGen::Printer::ops(mlir::ValueRange args, bool include_types) -> SmallUnrenderedString {
    SmallUnrenderedString tmp;
    bool first = true;
    for (auto a : args) {
        if (first) first = false;
        else tmp += ", ";
        tmp += val(a, include_types);
    }
    return tmp;
}

void CodeGen::Printer::print(mlir::ModuleOp module) {
    for (auto& op : module.getBodyRegion().front())
        print_top_level_op(&op);
}

void CodeGen::Printer::print_op(Operation* op) {
    out += "    %1(";
    if (op->getNumResults()) out += std::format("%8(%%{}%) = ", Id(inst_ids, op));
    defer { out += "%)\n"; };

    auto Target = [&](Block* dest, mlir::OperandRange args) -> SmallUnrenderedString {
        SmallUnrenderedString tmp;
        tmp += std::format("%3(bb{}%)", Id(block_ids, dest));
        if (not args.empty()) {
            tmp += "(";
            tmp += ops(args, false);
            tmp += ")";
        }
        return tmp;
    };

    /*auto IntCast = [&](StringRef name) {
        auto c = cast<ICast>(i);
        out += std::format(
            "%1({}%) {} to {}",
            name,
            DumpValue(c->args()[0]),
            c->cast_result_type()
        );
    };*/

    auto PrintArithOp = [&](StringRef mnemonic) {
        out += std::format(
            "{} {}, {}",
            mnemonic,
            val(op->getOperand(0)),
            val(op->getOperand(1), false)
        );
    };

    if (auto a = dyn_cast<AllocaOp>(op)) {
        out += std::format(
            "alloca %5({}%), align %5({}%)",
            a.getBytes().getValue(),
            a.getAlignment().getValue()
        );
        return;
    }

    if (auto s = dyn_cast<StoreOp>(op)) {
        out += std::format(
            "store {}, {}, align %5({}%)",
            val(s.getAddr(), false),
            val(s.getValue()),
            s.getAlignment().getValue()
        );
        return;
    }

    if (auto gep = dyn_cast<mlir::LLVM::GEPOp>(op)) {
        Assert(gep.getIndices().size() == 1);
        Assert(
            gep.getElemType().isSignlessInteger(8),
            "Invalid base type for gep: {}",
            FormatType(gep.getElemType())
        );

        out += std::format("ptradd {}, ", val(gep.getBase(), false));
        auto idx = gep.getIndices()[0];
        if (auto i = dyn_cast<mlir::IntegerAttr>(idx)) out += std::format("%5({}%)", i.getValue());
        else out += val(cast<Value>(idx));
        return;
    }

    if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(op)) {
        out += std::format(
            "icmp {} {}, {}",
            stringifyCmpIPredicate(cmp.getPredicate()),
            val(cmp.getLhs()),
            val(cmp.getRhs(), false)
        );
        return;
    }

    if (auto br = dyn_cast<mlir::cf::CondBranchOp>(op)) {
        out += std::format(
            "br {} to {} else {}",
            val(br.getCondition(), false),
            Target(br.getTrueDest(), br.getTrueDestOperands()),
            Target(br.getFalseDest(), br.getFalseDestOperands())
        );
        return;
    }

    if (auto br = dyn_cast<mlir::cf::BranchOp>(op)) {
        out += std::format("br {}", Target(br.getDest(), br.getDestOperands()));
        return;
    }

    if (auto l = dyn_cast<LoadOp>(op)) {
        out += std::format(
            "load {}, {}, align %5({}%)",
            FormatType(l->getResultTypes().front()),
            val(l.getAddr(), false),
            l.getAlignment().getValue()
        );
        return;
    }

    if (auto c = dyn_cast<CallOp>(op)) {
        out += "call ";
        if (c.getVariadic()) out += "variadic ";
        out += stringifyCConv(c.getCc());
        out += " ";

        if (c.getRes()) out += FormatType(c.getResultTypes().front());
        else out += "%6(void%)";
        out += " ";

        out += std::format("{}({})", val(c.getAddr(), false), ops(c.getArgs()));

        if (auto v = c.getMrvalueSlot()) out += std::format(" into {}", val(v, false));
        if (auto v = c.getEnv()) out += std::format(", env {}", val(v, false));
        return;
    }

    if (auto r = dyn_cast<RetOp>(op)) {
        out += "ret";
        if (auto v = r.getValue()) out += std::format(" {}", val(v));
        return;
    }

    if (auto a = dyn_cast<AbortOp>(op)) {
        llvm::raw_svector_ostream os(out);
        out += "abort at %5(";
        a.getLoc()->print(os, true);
        out += std::format(
            "%) %2({}%)({})",
            stringifyAbortReason(a.getReason()),
            ops(a.getOperands())
        );
        return;
    }

    if (auto m = dyn_cast<mlir::LLVM::MemcpyOp>(op)) {
        out += std::format(
            "copy{} {}, {}, {}",
            m.getIsVolatile() ? " volatile" : "",
            val(m.getDst(), false),
            val(m.getSrc(), false),
            val(m.getLen(), false)
        );
        return;
    }

    if (auto e = dyn_cast<ExtractOp>(op)) {
        out += std::format("extract {}, %5({}%)", val(e.getTuple()), e.getIndex().getValue());
        return;
    }

    if (isa<mlir::arith::AddIOp>(op)) return PrintArithOp("add");
    if (isa<mlir::arith::AndIOp>(op)) return PrintArithOp("and");
    if (isa<mlir::arith::DivSIOp>(op)) return PrintArithOp("sdiv");
    if (isa<mlir::arith::DivUIOp>(op)) return PrintArithOp("udiv");
    if (isa<mlir::arith::MulIOp>(op)) return PrintArithOp("mul");
    if (isa<mlir::arith::OrIOp>(op)) return PrintArithOp("or");
    if (isa<mlir::arith::RemSIOp>(op)) return PrintArithOp("srem");
    if (isa<mlir::arith::RemUIOp>(op)) return PrintArithOp("urem");
    if (isa<mlir::arith::ShLIOp>(op)) return PrintArithOp("shl");
    if (isa<mlir::arith::ShRSIOp>(op)) return PrintArithOp("ashr");
    if (isa<mlir::arith::ShRUIOp>(op)) return PrintArithOp("lshr");
    if (isa<mlir::arith::SubIOp>(op)) return PrintArithOp("sub");
    if (isa<mlir::arith::XOrIOp>(op)) return PrintArithOp("xor");

    if (isa<SAddOvOp>(op)) return PrintArithOp("sadd ov");
    if (isa<SSubOvOp>(op)) return PrintArithOp("ssub ov");
    if (isa<SMulOvOp>(op)) return PrintArithOp("smul ov");

    out += std::format("TODO: PRINT '{}'", op->getName().getStringRef());
}

void CodeGen::Printer::print_procedure(ProcOp proc) {
    curr_proc = proc;
    if (not out.empty()) out += "\n";

    // If this is not a declaration, number the temporaries first so we can
    // use that to print the procedure arguments.
    if (not proc.isDeclaration()) {
        i64 temp = 0;
        arg_ids.clear();
        block_ids.clear();
        inst_ids.clear();
        for (auto [id, b] : enumerate(proc.getBlocks())) {
            block_ids[&b] = i64(id);
            for (auto arg : b.getArguments())
                arg_ids[arg] = temp++;
            for (auto& i : b) {
                if (IsInlineOp(&i)) continue;
                if (i.getNumResults()) inst_ids[&i] = temp++;
            }
        }
    }

    // Print name.
    out += std::format("%1(proc%) %2({}%)", proc.getName());

    // Print args.
    if (proc.getNumArguments()) {
        if (proc.isDeclaration()) {
            auto args = proc.getArgumentTypes();
            out += std::format(" %1((%){}%1()%)", utils::join(args, "%1(,%)", "{}", FormatType));
        } else {
            out += std::format(" %1((%){}%1()%)", ops(proc.getArguments()));
        }
    }

    // Print attributes.
    if (proc.getVariadic()) out += " %1(variadic%)";
    if (proc.getHasIndirectReturn()) out += " %1(indirect%)";
    if (proc.getHasStaticChain()) out += " %1(nested%)";
    out += std::format(" %1({}%)", stringifyLinkage(proc.getLinkage().getLinkage()));
    out += std::format(" %1({}%)", stringifyCConv(proc.getCc()));

    // Print return type.
    if (proc.getNumResults() and not isa<mlir::NoneType>(proc.getResultTypes().front()))
        out += std::format(" %1(->%) {}", FormatType(proc.getResultTypes().front()));

    // Stop if there is no body.
    if (proc.isDeclaration()) {
        out += "%1(;%)\n";
        return;
    }

    // Print the procedure body.
    out += " %1({%)\n";

    // Print frame allocations.
    /*for (auto* f : proc->frame())
        out += std::format("    %8(%%{}%) %1(=%) {}\n", Id(frame_ids, f), f->allocated_type());
    if (not proc->frame().empty())
        out += "\n";*/

    // Print blocks and instructions.
    for (auto [i, b] : enumerate(proc.getBlocks())) {
        if (i == 0) {
            out += "%3(entry%)%1(:%)\n";
        } else {
            out += std::format("\n%3(bb{}%)%1(", i);
            if (b.getNumArguments()) {
                out += std::format("({})", utils::join_as(b.getArguments(), [&](mlir::BlockArgument arg) {
                    return std::format("{} %3(%%{}%)", FormatType(arg.getType()), arg_ids.at(arg));
                }));
            }
            out += ":%)\n";
        }

        for (auto& inst : b) {
            if (IsInlineOp(&inst)) continue;
            print_op(&inst);
        }

        if (not b.mightHaveTerminator())
            out += "    %1b(<<< MISSING TERMINATOR >>>%)\n";
    }
    out += "%1(}%)\n";
}

void CodeGen::Printer::print_top_level_op(Operation* op) {
    if (auto g = dyn_cast<mlir::LLVM::GlobalOp>(op)) {
        Assert(g.getConstant(), "TODO: Print non-constant globals");
        auto i = global++;
        global_ids[g] = i64(i);
        out += std::format("%3(@{}%)", i);
        if (auto init = g.getValueOrNull()) {
            out += " %1(=%) ";

            // A bit of a hack but it works.
            SmallString<128> tmp;
            llvm::raw_svector_ostream os{tmp};
            init.print(os, true);
            if (tmp.ends_with("\"")) {
                out += std::format("%3({}%)", stream{tmp.str()}.replace('%', "%%"));
            } else {
                out += std::format(
                    "%1({{%) %5({}%) %1(}}, align %) %5({}%)",
                    utils::join(tmp, "%1(,%) ", "{:02X}"),
                    g.getAlignment().value_or(1)
                );
            }
        }
        out += "\n";
        return;
    }

    if (auto p = dyn_cast<ProcOp>(op)) {
        print_procedure(p);
        return;
    }

    out += std::format("TODO: PRINT TOP LEVEL '{}'\n", op->getName().getStringRef());
}

auto CodeGen::Printer::val(Value v, bool include_type) -> SmallUnrenderedString {
    SmallUnrenderedString tmp;

    if (include_type or always_include_types) {
        tmp += FormatType(v.getType());
        tmp += " ";
    }

    if (auto b = dyn_cast<mlir::BlockArgument>(v)) {
        tmp += std::format("%3(%%{}%)", arg_ids.at(b));
        return tmp;
    }

    if (auto res = dyn_cast<mlir::OpResult>(v)) {
        auto op = res.getOwner();

        // Inline constant ops.
        if (auto c = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
            tmp += std::format("%5({}%)", c.value());
            return tmp;
        }

        if (auto p = dyn_cast<ProcRefOp>(op)) {
            tmp.clear(); // Never print the type of a proc ref.
            tmp += std::format("%2({}%)", p.proc().getName());
            return tmp;
        }

        if (auto t = dyn_cast<TupleOp>(op)) {
            tempset always_include_types = false;
            tmp += "(";
            tmp += ops(t.getValues(), false); // We’ve already printed the tuple type.
            tmp += ")";
            return tmp;
        }

        if (auto a = dyn_cast<mlir::LLVM::AddressOfOp>(op)) {
            auto g = cg.mlir_module.lookupSymbol<mlir::LLVM::GlobalOp>(a.getGlobalNameAttr());
            tmp += std::format("%3(@{}%)", global_ids.at(g));
            return tmp;
        }

        if (isa<NilOp>(op)) {
            tmp += "nil";
            return tmp;
        }

        if (isa<ReturnPointerOp>(op)) {
            tmp += "%3(retptr%)";
            return tmp;
        }

        Assert(not IsInlineOp(op), "Forgot to handle this inline op here: '{}'", op->getName().getStringRef());
        tmp += std::format("%8(%%{}", inst_ids.at(op));
        if (op->getNumResults() > 1) tmp += std::format(":{}", res.getResultNumber());
        tmp += "%)";
        return tmp;
    }

    tmp += "TODO: Print this value properly:";
    llvm::raw_svector_ostream os{tmp};
    v.print(os);
    return tmp;
}
