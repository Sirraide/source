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
    DenseMap<Operation*, i64> frame_ids;
    i64 global = 0;
    ProcOp curr_proc{};
    bool verbose;
    bool always_include_types = verbose;

    Printer(CodeGen& cg, bool verbose = false) : cg{cg}, verbose{verbose} {}
    void print(mlir::ModuleOp module);
    void print_arg_list(ProcAndCallOpInterface proc_or_call, bool types_only, bool wrap);
    void print_op(Operation* op);
    void print_procedure(ProcOp op);
    void print_top_level_op(Operation* op);
    auto ops(mlir::ValueRange args, bool include_types = true) -> SmallUnrenderedString;
    auto val(Value v, bool include_type = true) -> SmallUnrenderedString;

    auto Id(auto& map, auto* ptr) -> i64 {
        auto it = map.find(ptr);
        return it == map.end() ? -1 : it->second;
    }

    auto IsInlineOp(Operation* op) {
        return not verbose and isa< // clang-format off
            FrameSlotOp,
            NilOp,
            ProcRefOp,
            ReturnPointerOp,
            mlir::LLVM::AddressOfOp,
            mlir::arith::ConstantIntOp
        >(op); // clang-format on
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
    } else {
        llvm::raw_string_ostream os{tmp};
        tmp += "%6(";
        ty.print(os);
        tmp += "%)";
    }
    return tmp;
}

auto CodeGen::dump(bool verbose, bool generic) -> SmallUnrenderedString {
    if (generic) {
        SmallUnrenderedString s;
        llvm::raw_svector_ostream os{s};
        auto f = mlir::OpPrintingFlags().assumeVerified(true).printUniqueSSAIDs(true).enableDebugInfo(verbose, verbose);
        mlir_module.print(os, f);
        return SmallUnrenderedString(stream{s.str()}.replace('%', "%%"));
    }

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

void CodeGen::Printer::print_arg_list(ProcAndCallOpInterface proc_or_call, bool types_only, bool wrap) {
    bool is_proc = isa<ProcOp>(proc_or_call);
    auto indent = is_proc ? "    "sv : "        "sv;
    if (proc_or_call.getNumCallArgs()) {
        if (wrap) out += " %1((%)\n";
        else out += " %1((%)";

        bool first = true;
        for (unsigned i = 0; i < proc_or_call.getNumCallArgs(); i++) {
            if (wrap) out += indent;
            if (first) first = false;
            else if (not wrap) out += "%1(, %)";

            if (types_only) out += FormatType(proc_or_call.getCallArgType(i));
            else out += val(proc_or_call.getCallArg(i));

            if (auto attrs = proc_or_call.getCallArgAttrs(i)) {
                for (auto attr : attrs) {
                    using mlir::LLVM::LLVMDialect;
                    if (attr.getName() == LLVMDialect::getByValAttrName()) {
                        out += std::format(" %1(byval%) {}", FormatType(cast<mlir::TypeAttr>(attr.getValue()).getValue()));
                    } else if (attr.getName() == LLVMDialect::getZExtAttrName()) {
                        out += " %1(zeroext%)";
                    } else if (attr.getName() == LLVMDialect::getSExtAttrName()) {
                        out += " %1(signext%)";
                    } else if (attr.getName() == LLVMDialect::getStructRetAttrName()) {
                        out += std::format(" %1(sret %){}", FormatType(cast<mlir::TypeAttr>(attr.getValue()).getValue()));
                    } else if (attr.getName() == LLVMDialect::getDereferenceableAttrName()) {
                        out += std::format(" %1(dereferenceable %)%5({}%)", cast<mlir::IntegerAttr>(attr.getValue()).getInt());
                    } else {
                        out += std::format(" <DON'T KNOW HOW TO PRINT '{}'>", attr.getName());
                    }
                }
            }

            if (wrap) {
                bool last = i == proc_or_call.getNumCallArgs() - 1;
                if (not last or is_proc) out += "%1(,%)\n";
            }
        }

        out += "%1()%)";
    }
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

    auto PrintArithOp = [&](StringRef mnemonic) {
        out += std::format(
            "{} {}, {}",
            mnemonic,
            val(op->getOperand(0)),
            val(op->getOperand(1), false)
        );
    };

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

    if (auto cmp = dyn_cast<mlir::LLVM::ICmpOp>(op)) {
        out += std::format(
            "cmp {} {}, {}",
            stringifyICmpPredicate(cmp.getPredicate()),
            val(cmp.getLhs()),
            val(cmp.getRhs(), false)
        );
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

        if (c.getNumResults() == 0) {
            out += "%6(void%)";
        } else if (c.getNumResults() == 1) {
            out += FormatType(c.getResultTypes().front());
        } else {
            out += std::format(
                "%1(({})%)",
                utils::join(
                    SmallVector<mlir::Type>{c.getResultTypes()},
                    ", ",
                    "{}",
                    FormatType
                )
            );
        }

        out += " ";
        out += val(c.getAddr(), false);
        print_arg_list(c, false, c.getNumCallArgs() > 3);
        if (auto v = c.getEnv()) out += std::format(", env {}", val(v, false));
        return;
    }

    if (auto r = dyn_cast<RetOp>(op)) {
        out += std::format("ret {}", ops(r.getVals()));
        return;
    }

    if (auto a = dyn_cast<AbortOp>(op)) {
        llvm::raw_svector_ostream os(out);
        out += "abort at %5(";
        a.getLoc()->print(os, true);
        out += std::format(
            "%) %2({}%)({})",
            stringifyAbortReason(a.getReason()),
            ops(a.getAbortInfo())
        );
        return;
    }

    if (auto m = dyn_cast<mlir::LLVM::MemcpyOp>(op)) {
        out += std::format(
            "copy{} {} <- {}, {}",
            m.getIsVolatile() ? " volatile" : "",
            val(m.getDst(), false),
            val(m.getSrc(), false),
            val(m.getLen(), false)
        );
        return;
    }

    if (auto a = dyn_cast<mlir::LLVM::AddressOfOp>(op)) {
        auto g = cg.mlir_module.lookupSymbol<mlir::LLVM::GlobalOp>(a.getGlobalNameAttr());
        out += std::format("addressof %3(@{}%)", global_ids.at(g));
        return;
    }

    if (auto c = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
        out += std::format("{} %5({}%)", FormatType(c.getType()), c.value());
        return;
    }

    if (auto p = dyn_cast<ProcRefOp>(op)) {
        out += std::format("procref %2({}%)", p.proc().getName());
        return;
    }

    if (auto slot = dyn_cast<FrameSlotOp>(op)) {
        out += std::format(
            "alloca %5({}%)%1(, align%) %5({}%)",
            slot.getBytes().getValue(),
            slot.getAlignment().getValue()
        );
        return;
    }

    if (auto s = dyn_cast<mlir::arith::SelectOp>(op)) {
        out += std::format(
            "select {}, {}, {}",
            val(s.getCondition(), false),
            val(s.getTrueValue()),
            val(s.getFalseValue(), false)
        );
        return;
    }

    if (isa<ReturnPointerOp>(op)) {
        out += "%3(retptr%)";
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::ExtUIOp>(op)) {
        out += std::format("zext {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::ExtSIOp>(op)) {
        out += std::format("sext {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::TruncIOp>(op)) {
        out += std::format("trunc {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto m = dyn_cast<mlir::LLVM::MemsetOp>(op)) {
        out += std::format("set {}, {}, {}", val(m.getDst()), val(m.getVal()), val(m.getLen()));
        return;
    }

    if (isa<mlir::LLVM::UnreachableOp>(op)) {
        out += "unreachable";
        return;
    }

    if (isa<NilOp>(op)) {
        out += "nil";
        return;
    }

    if (isa<mlir::LLVM::PoisonOp>(op)) {
        out += std::format("{} poison", FormatType(op->getResult(0).getType()));
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

    Unreachable("Don’t know how to print this op: {}", op->getName().getStringRef());
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
    out += std::format("%1(proc%) %2({}%)", utils::Escape(proc.getName(), false, true));

    // Print args.
    print_arg_list(proc, proc.isDeclaration(), proc.getNumCallArgs() > 3);

    // Print attributes.
    if (proc.getVariadic()) out += " %1(variadic%)";
    if (proc.getHasStaticChain()) out += " %1(nested%)";
    out += std::format(" %1({}%)", stringifyLinkage(proc.getLinkage().getLinkage()));
    out += std::format(" %1({}%)", stringifyCConv(proc.getCc()));

    // Print return types.
    if (proc.getNumResults()) {
        out += " %1(->%) ";
        if (proc.getNumResults() == 1) {
            out += FormatType(proc.getResultTypes().front());
        } else {
            out += std::format(
                "%1(({})%)",
                utils::join(
                    SmallVector<mlir::Type>{proc.getResultTypes()},
                    ", ",
                    "{}",
                    FormatType
                )
            );
        }
    }

    // Stop if there is no body.
    if (proc.isDeclaration()) {
        out += "%1(;%)\n";
        return;
    }

    // Print the procedure body.
    out += " %1({%)\n";

    // Print frame allocations.
    if (not verbose) {
        i64 frame = 0;
        for (auto& f : proc.front()) {
            // Transformations may reorder these for some reason.
            auto slot = dyn_cast<FrameSlotOp>(&f);
            if (not slot) continue;
            auto id = frame_ids[&f] = frame++;
            out += std::format(
                "    %4(#{}%) %1(=%) %5({}%)%1(, align%) %5({}%)\n",
                id,
                slot.getBytes().getValue(),
                slot.getAlignment().getValue()
            );
        }

        if (frame) out += "\n";
    }

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
            out += std::format("%3({}%)", stream{tmp.str()}.replace('%', "%%"));
        }

        if (g.getAlignment().value_or(1) != 1)
            out += std::format("%1(, align%) %5({}%)", g.getAlignment().value_or(1));

        out += '\n';
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
    if (not v) {
        tmp += std::format("%1b(<<< NULL >>>%)");
        return tmp;
    }

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

        if (not IsInlineOp(op)) {
            tmp += std::format("%8(%%{}", inst_ids.at(op));
            if (op->getNumResults() > 1) tmp += std::format(":{}", res.getResultNumber());
            tmp += "%)";
            return tmp;
        }

        if (auto c = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
            if (c.getType().isInteger(1)) tmp += std::format("%5({}%)", c.value() != 0);
            else tmp += std::format("%5({}%)", c.value());
            return tmp;
        }

        if (auto p = dyn_cast<ProcRefOp>(op)) {
            tmp.clear(); // Never print the type of a proc ref.
            tmp += std::format("%2({}%)", utils::Escape(p.proc().getName(), false, true));
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

        if (isa<FrameSlotOp>(op)) {
            tmp += std::format("%4(#{}%)", Id(frame_ids, op));
            return tmp;
        }

        Unreachable("Unsupported inline op: '{}'", op->getName().getStringRef());
    }

    tmp += "TODO: Print this value properly:";
    llvm::raw_svector_ostream os{tmp};
    v.print(os);
    return tmp;
}
