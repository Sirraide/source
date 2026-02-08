#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/IR/IR.hh>
#include <srcc/Core/Constants.hh>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;
namespace LLVM = mlir::LLVM;

/// If this is a fixed-sized alloca, return the element count.
static auto GetConstantAllocaElemCount(LLVM::AllocaOp op) -> std::optional<u64> {
    // If the array size is not a constant, then this is a dynamic alloca.
    auto op_res = dyn_cast<mlir::OpResult>(op.getArraySize());
    if (not op_res) return std::nullopt;
    auto constant = dyn_cast<mlir::arith::ConstantOp>(op_res.getOwner());
    if (not constant) return std::nullopt;

    // Otherwise, retrieve the count.
    auto size = cast<mlir::IntegerAttr>(constant.getValue());
    return size.getValue().getZExtValue();
}

/// Check if this is a constant-size alloca.
static auto IsConstantAlloca(LLVM::AllocaOp op) {
    return GetConstantAllocaElemCount(op).has_value();
}

struct CodeGen::Printer {
    CodeGen& cg;
    SmallUnrenderedString out;
    DenseMap<Operation*, SmallString<64>> global_names;
    DenseMap<mlir::BlockArgument, i64> arg_ids;
    DenseMap<Block*, i64> block_ids;
    DenseMap<Operation*, i64> inst_ids;
    DenseMap<Operation*, i64> frame_ids;
    SmallString<64> indent;
    i64 global = 0;
    ProcOp curr_proc{};
    bool verbose;
    bool always_include_types = verbose;

    Printer(CodeGen& cg, bool verbose = false) : cg{cg}, verbose{verbose} {}
    void print(mlir::ModuleOp module);
    void print_arg_list(ProcAndCallOpInterface proc_or_call, bool types_only);
    void print_result_attrs(ProcAndCallOpInterface proc_or_call, unsigned idx);
    void print_attr(mlir::NamedAttribute attr);
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
        if (verbose) return false;
        if (auto a = dyn_cast<LLVM::AllocaOp>(op)) return IsConstantAlloca(a);
        return isa< // clang-format off
            NilOp,
            ProcRefOp,
            TreeConstantOp,
            TypeConstantOp,
            LLVM::AddressOfOp,
            mlir::arith::ConstantIntOp
        >(op); // clang-format on
    }
};

auto ir::FormatType(mlir::Type ty) -> SmallString<128> {
    SmallString<128> tmp;
    if (isa<LLVM::LLVMPointerType>(ty)) {
        tmp += "%6(ptr%)";
    } else if (isa<ir::TreeType>(ty)) {
        tmp += "%6(tree%)";
    } else if (isa<ir::TypeType>(ty)) {
        tmp += "%6(type%)";
    } else if (auto ui = dyn_cast<mlir::IntegerType>(ty); ui and ui.isUnsignedInteger(1)) {
        tmp += "%6(bool%)";
    } else if (auto a = dyn_cast<LLVM::LLVMArrayType>(ty)) {
        Format(tmp, "{}%1([%5({}%)]%)", FormatType(a.getElementType()), a.getNumElements());
    } else {
        llvm::raw_svector_ostream os{tmp};
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
        return SmallUnrenderedString(str{s.str()}.replace('%', "%%"));
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

void CodeGen::Printer::print_arg_list(ProcAndCallOpInterface proc_or_call, bool types_only) {
    // Wrap if we have at least for arguments, or if at least 2 arguments are 'ptr's
    // since we usually have a soup of 'noundef align dereferenceable' attributes
    // on those.
    bool wrap = proc_or_call.getNumCallArgs() > 3;
    if (not wrap) {
        unsigned num_ptrs = 0;
        for (unsigned i = 0; i < proc_or_call.getNumCallArgs(); i++) {
            if (
                isa<LLVM::LLVMPointerType>(proc_or_call.getCallArgType(i)) and
                proc_or_call.getCallArgAttrs(i) and
                not proc_or_call.getCallArgAttrs(i).empty()
            ) {
                num_ptrs++;
                if (num_ptrs == 2) {
                    wrap = true;
                    break;
                }
            }
        }
    }

    bool is_proc = isa<ProcOp>(proc_or_call);
    auto save_indent = indent.size();
    defer { indent.truncate(save_indent); };
    indent += "    "sv;
    if (proc_or_call.getNumCallArgs()) {
        if (is_proc) out += ' ';
        if (wrap) out += "%1((%)\n";
        else out += "%1((%)";

        bool first = true;
        for (unsigned i = 0; i < proc_or_call.getNumCallArgs(); i++) {
            if (wrap) out += indent;
            if (first) first = false;
            else if (not wrap) out += "%1(, %)";

            if (types_only) out += FormatType(proc_or_call.getCallArgType(i));
            else out += val(proc_or_call.getCallArg(i));

            if (auto attrs = proc_or_call.getCallArgAttrs(i)) {
                for (auto attr : attrs) {
                    print_attr(attr);
                }
            }

            if (wrap) {
                bool last = i == proc_or_call.getNumCallArgs() - 1;
                if (not last or is_proc) out += "%1(,%)\n";
            }
        }

        out += "%1()%)";
    } else if (not is_proc) {
        out += "%1(()%)";
    }
}

void CodeGen::Printer::print_result_attrs(ProcAndCallOpInterface proc_or_call, unsigned idx) {
    if (auto attrs = proc_or_call.getCallResultAttrs(idx)) {
        for (auto attr : attrs) {
            print_attr(attr);
        }
    }
}

void CodeGen::Printer::print_attr(mlir::NamedAttribute attr) {
    using LLVM::LLVMDialect;
    if (attr.getName() == LLVMDialect::getByValAttrName()) {
        Format(out, " %1(byval%) {}", FormatType(cast<mlir::TypeAttr>(attr.getValue()).getValue()));
    } else if (attr.getName() == LLVMDialect::getZExtAttrName()) {
        out += " %1(zeroext%)";
    } else if (attr.getName() == LLVMDialect::getSExtAttrName()) {
        out += " %1(signext%)";
    } else if (attr.getName() == LLVMDialect::getNestAttrName()) {
        out += " %1(nest%)";
    } else if (attr.getName() == LLVMDialect::getNoFreeAttrName()) {
        out += " %1(nofree%)";
    } else if (attr.getName() == LLVMDialect::getNoUndefAttrName()) {
        out += " %1(noundef%)";
    } else if (attr.getName() == LLVMDialect::getReadonlyAttrName()) {
        out += " %1(readonly%)";
    } else if (attr.getName() == LLVMDialect::getStructRetAttrName()) {
        Format(out, " %1(sret %){}", FormatType(cast<mlir::TypeAttr>(attr.getValue()).getValue()));
    } else if (attr.getName() == LLVMDialect::getDereferenceableAttrName()) {
        Format(out, " %1(dereferenceable %)%5({}%)", cast<mlir::IntegerAttr>(attr.getValue()).getInt());
    } else if (attr.getName() == LLVMDialect::getAlignAttrName()) {
        Format(out, " %1(align %)%5({}%)", cast<mlir::IntegerAttr>(attr.getValue()).getInt());
    } else {
        Format(out, " <DON'T KNOW HOW TO PRINT '{}'>", attr.getName().strref());
    }
}

void CodeGen::Printer::print_op(Operation* op) {
    out += indent;
    out += "%1(";
    if (op->getNumResults()) Format(out, "%8(%%{}%) = ", Id(inst_ids, op));
    defer { out += "%)\n"; };

    auto Target = [&](Block* dest, mlir::OperandRange args) -> SmallUnrenderedString {
        SmallUnrenderedString tmp;
        Format(tmp, "%3(bb{}%)", Id(block_ids, dest));
        if (not args.empty()) {
            tmp += "(";
            tmp += ops(args, false);
            tmp += ")";
        }
        return tmp;
    };

    auto PrintArithOp = [&](StringRef mnemonic) {
        Format(out,
            "{} {}, {}",
            mnemonic,
            val(op->getOperand(0)),
            val(op->getOperand(1), false)
        );
    };

    if (auto s = dyn_cast<ir::StoreOp>(op)) {
        Format(out,
            "store {}, {}, align %5({}%)",
            val(s.getAddr(), false),
            val(s.getValue()),
            s.getAlignment().getValue()
        );
        return;
    }

    if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
        Assert(gep.getIndices().size() == 1);
        Assert(
            gep.getElemType().isSignlessInteger(8),
            "Invalid base type for gep: {}",
            FormatType(gep.getElemType())
        );

        Format(out, "ptradd {}, ", val(gep.getBase(), false));
        auto idx = gep.getIndices()[0];
        if (auto i = dyn_cast<mlir::IntegerAttr>(idx)) Format(out, "%5({}%)", i.getValue());
        else out += val(cast<Value>(idx));
        return;
    }

    if (auto cmp = dyn_cast<LLVM::ICmpOp>(op)) {
        Format(out,
            "cmp {} {}, {}",
            stringifyICmpPredicate(cmp.getPredicate()),
            val(cmp.getLhs()),
            val(cmp.getRhs(), false)
        );
        return;
    }

    if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(op)) {
        Format(out,
            "icmp {} {}, {}",
            stringifyCmpIPredicate(cmp.getPredicate()),
            val(cmp.getLhs()),
            val(cmp.getRhs(), false)
        );
        return;
    }

    if (auto br = dyn_cast<mlir::cf::CondBranchOp>(op)) {
        Format(out,
            "br {} to {} else {}",
            val(br.getCondition(), false),
            Target(br.getTrueDest(), br.getTrueDestOperands()),
            Target(br.getFalseDest(), br.getFalseDestOperands())
        );
        return;
    }

    if (auto br = dyn_cast<mlir::cf::BranchOp>(op)) {
        Format(out, "br {}", Target(br.getDest(), br.getDestOperands()));
        return;
    }

    if (auto l = dyn_cast<ir::LoadOp>(op)) {
        Format(out,
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

        if (c.getNumResults() == 0) {
            out += " %6(void%)";
        } else if (c.getNumResults() == 1) {
            print_result_attrs(c, 0);
            out += " ";
            out += FormatType(c.getResultTypes().front());
        } else {
            Format(out,
                " %1(({})%)",
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
        print_arg_list(c, false);
        return;
    }

    if (auto r = dyn_cast<RetOp>(op)) {
        Format(out, "ret {}", ops(r.getVals()));
        return;
    }

    if (auto a = dyn_cast<AbortOp>(op)) {
        llvm::raw_svector_ostream os(out);
        out += "abort at %5(";
        a.getLoc()->print(os, true);
        Format(out,
            "%) %2({}%)({})",
            stringifyAbortReason(a.getReason()),
            ops(a.getAbortInfo())
        );

        if (not verbose) return;
        out += " {\n";
        indent += "    ";
        for (auto& o : a.getBody().front()) {
            if (IsInlineOp(&o)) continue;
            print_op(&o);
        }
        indent.truncate(indent.size() - 4);
        out += indent;
        out += "}";
        return;
    }

    if (auto m = dyn_cast<LLVM::MemcpyOp>(op)) {
        Format(out,
            "copy{} {} <- {}, {}",
            m.getIsVolatile() ? " volatile" : "",
            val(m.getDst(), false),
            val(m.getSrc(), false),
            val(m.getLen(), false)
        );
        return;
    }

    if (auto a = dyn_cast<LLVM::AddressOfOp>(op)) {
        auto g = cg.mlir_module.lookupSymbol<LLVM::GlobalOp>(a.getGlobalNameAttr());
        Format(out, "addressof %3(@{}%)", global_names.at(g));
        return;
    }

    if (auto c = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
        Format(out, "{} %5({}%)", FormatType(c.getType()), c.value());
        return;
    }

    if (auto p = dyn_cast<ProcRefOp>(op)) {
        Format(out, "procref %2({}%)", p.proc().getName());
        return;
    }

    if (auto slot = dyn_cast<LLVM::AllocaOp>(op)) {
        auto sz = GetConstantAllocaElemCount(slot);
        if (slot.getElemType().isInteger(8)) {
            if (sz.has_value()) return Format(
                out,
                "alloca %5({}%), align %5({}%)",
                *sz,
                slot.getAlignment().value_or(1)
            );

            return Format(
                out,
                "alloca {} x {}, align %5({}%)",
                FormatType(slot.getElemType()),
                val(slot.getArraySize(), false),
                slot.getAlignment().value_or(1)
            );
        }

        Assert(sz == 1, "Typed alloca should have array size 1");
        return Format(
            out,
            "alloca {}, align %5({}%)",
            FormatType(slot.getElemType()),
            slot.getAlignment().value_or(1)
        );
    }

    if (auto s = dyn_cast<mlir::arith::SelectOp>(op)) {
        Format(out,
            "select {}, {}, {}",
            val(s.getCondition(), false),
            val(s.getTrueValue()),
            val(s.getFalseValue(), false)
        );
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::ExtUIOp>(op)) {
        Format(out, "zext {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::ExtSIOp>(op)) {
        Format(out, "sext {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto ext = dyn_cast<mlir::arith::TruncIOp>(op)) {
        Format(out, "trunc {} to {}", val(ext.getIn()), FormatType(ext.getOut().getType()));
        return;
    }

    if (auto m = dyn_cast<LLVM::MemsetOp>(op)) {
        Format(out, "set {}, {}, {}", val(m.getDst()), val(m.getVal()), val(m.getLen()));
        return;
    }

    if (isa<LLVM::UnreachableOp>(op)) {
        out += "unreachable";
        return;
    }

    if (isa<NilOp>(op)) {
        out += "nil";
        return;
    }

    if (auto tree = dyn_cast<TreeConstantOp>(op)) {
        out += Format("tree {}", val(tree, false));
        return;
    }

    if (auto tree = dyn_cast<QuoteOp>(op)) {
        out += Format("tree %5({}%), {}", static_cast<void*>(tree.getTree()), ops(tree.getUnquotes(), false));
        return;
    }

    if (auto ty = dyn_cast<TypeConstantOp>(op)) {
        Format(out, "type {}", ty.getValue());
        return;
    }

    if (auto eq = dyn_cast<TypeEqOp>(op)) {
        Format(out, "type.eq {}, {}", val(eq.getLhs(), false), val(eq.getRhs(), false));
        return;
    }

    if (auto prop = dyn_cast<TypePropertyOp>(op)) {
        Format(out, "type.prop %5({}%), {}", prop.getProperty(), val(prop.getTypeArgument(), false));
        return;
    }

    if (isa<LLVM::PoisonOp>(op)) {
        Format(out, "{} poison", FormatType(op->getResult(0).getType()));
        return;
    }

    if (auto e = dyn_cast<ir::EngageOp>(op)) {
        Format(out, "engage {}", val(e.getOptional(), false));
        return;
    }

    if (auto e = dyn_cast<ir::EngageCopyOp>(op)) {
        Format(out, "engage {} <- {}", val(e.getOptional(), false), val(e.getCopyFrom(), false));
        return;
    }

    if (auto e = dyn_cast<ir::DisengageOp>(op)) {
        Format(out, "disengage {}", val(e.getOptional(), false));
        return;
    }

    if (auto e = dyn_cast<ir::UnwrapOp>(op)) {
        Format(out, "unwrap {}", val(e.getOptional(), false));
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
        auto ProcessRegion = [&](this auto& self, mlir::Region& r) -> void {
            for (auto [id, b] : enumerate(r.getBlocks())) {
                block_ids[&b] = i64(id);
                for (auto arg : b.getArguments()) {
                    if (isa<ir::AbortOp>(b.getParentOp())) continue;
                    arg_ids[arg] = temp++;
                }
                for (auto& i : b) {
                    if (IsInlineOp(&i)) continue;
                    if (i.getNumResults()) inst_ids[&i] = temp++;
                    if (auto init = dyn_cast<ir::AbortOp>(i))
                        self(init.getBody());
                }
            }
        };

        ProcessRegion(proc.getBody());
    }

    // Print name.
    Format(out, "%1(proc%) %2({}%)", utils::Escape(proc.getName(), false, true));

    // Print args.
    print_arg_list(proc, proc.isDeclaration());

    // Print attributes.
    if (proc.getVariadic()) out += " %1(variadic%)";
    if (proc.getNoreturn()) out += " %1(noreturn%)";
    if (proc.getAlwaysInline()) out += " %1(inline%)";
    if (proc.getNorecurse()) out += " %1(norecurse%)";
    Format(out, " %1({}%)", stringifyLinkage(proc.getLinkage().getLinkage()));
    Format(out, " %1({}%)", stringifyCConv(proc.getCc()));

    // Print return types.
    if (proc.getNumResults()) {
        out += " %1(->%) ";
        if (proc.getNumResults() == 1) {
            out += FormatType(proc.getResultTypes().front());
            print_result_attrs(proc, 0);
        } else {
            Format(out,
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
    tempset indent = "    ";
    out += " %1({%)\n";

    // Print frame allocations.
    if (not verbose) {
        i64 frame = 0;
        for (auto& f : proc.front()) {
            auto slot = dyn_cast<LLVM::AllocaOp>(&f);
            if (not slot) continue;
            auto sz = GetConstantAllocaElemCount(slot);
            if (not sz) continue;
            auto id = frame_ids[&f] = frame++;

            if (slot.getElemType().isInteger(8)) {
                Format(
                    out,
                    "    %4(#{}%) %1(=%) %5({}%)%1(, align%) %5({}%)\n",
                    id,
                    *sz,
                    slot.getAlignment().value_or(1)
                );
            } else {
                Assert(sz == 1, "Typed allocas should have array size 1");
                Format(
                    out,
                    "    %4(#{}%) %1(=%) {}%1(, align%) %5({}%)\n",
                    id,
                    FormatType(slot.getElemType()),
                    slot.getAlignment().value_or(1)
                );
            }
        }

        if (frame) out += "\n";
    }

    // Print blocks and instructions.
    for (auto [i, b] : enumerate(proc.getBlocks())) {
        if (i == 0) {
            out += "%3(entry%)%1(:%)\n";
        } else {
            Format(out, "\n%3(bb{}%)%1(", i);
            if (b.getNumArguments()) {
                Format(out, "({})", utils::join_as(b.getArguments(), [&](mlir::BlockArgument arg) {
                    return Format("{} %3(%%{}%)", FormatType(arg.getType()), arg_ids.at(arg));
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
    if (auto g = dyn_cast<LLVM::GlobalOp>(op)) {
        // This is a string constant.
        if (g.getSymName().starts_with(constants::CodeGenStringConstantNamePrefix)) {
            Assert(g.getConstant(), "Strings should always be constants");

            auto i = global++;
            global_names[g] = Format("{}", i64(i));
            Format(out, "%3(@{}%)", i);

            if (auto init = g.getValueOrNull()) {
                out += " %1(=%) ";

                // A bit of a hack but it works.
                SmallString<128> tmp;
                llvm::raw_svector_ostream os{tmp};
                init.print(os, true);
                Format(out, "%3({}%)", str{tmp.str()}.replace('%', "%%"));
            }

            if (g.getAlignment().value_or(1) != 1)
                Format(out, "%1(, align%) %5({}%)", g.getAlignment().value_or(1));

            out += '\n';
            return;
        }

        global_names[g] = g.getSymName();

        Format(
            out,
            "%3(@{}%) %1(= {} %5({}%), align %({}%)%)\n",
            g.getSymName(),
            stringifyLinkage(g.getLinkage()),
            cast<LLVM::LLVMArrayType>(g.getType()).getNumElements(),
            g.getAlignment().value_or(1)
        );

        return;
    }

    if (auto p = dyn_cast<ProcOp>(op)) {
        print_procedure(p);
        return;
    }

    Format(out, "TODO: PRINT TOP LEVEL '{}'\n", op->getName().getStringRef());
}

auto CodeGen::Printer::val(Value v, bool include_type) -> SmallUnrenderedString {
    SmallUnrenderedString tmp;
    if (not v) {
        Format(tmp, "%1b(<<< NULL >>>%)");
        return tmp;
    }

    if (include_type or always_include_types) {
        tmp += FormatType(v.getType());
        tmp += " ";
    }

    if (auto b = dyn_cast<mlir::BlockArgument>(v)) {
        // Special handling for the argument of an init op.
        if (isa<AbortOp>(b.getOwner()->getParentOp())) {
            tmp += "%3(#%)";
            return tmp;
        }

        Format(tmp, "%3(%%{}%)", arg_ids.at(b));
        return tmp;
    }

    if (auto res = dyn_cast<mlir::OpResult>(v)) {
        auto op = res.getOwner();

        if (not IsInlineOp(op)) {
            Format(tmp, "%8(%%{}", inst_ids.at(op));
            if (op->getNumResults() > 1) Format(tmp, ":{}", res.getResultNumber());
            tmp += "%)";
            return tmp;
        }

        if (auto c = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
            if (c.getType().isInteger(1)) Format(tmp, "%5({}%)", c.value() != 0);
            else Format(tmp, "%5({}%)", c.value());
            return tmp;
        }

        if (auto p = dyn_cast<ProcRefOp>(op)) {
            tmp.clear(); // Never print the type of a proc ref.
            Format(tmp, "%2({}%)", utils::Escape(p.proc().getName(), false, true));
            return tmp;
        }

        if (auto a = dyn_cast<LLVM::AddressOfOp>(op)) {
            auto g = cg.mlir_module.lookupSymbol<LLVM::GlobalOp>(a.getGlobalNameAttr());
            Format(tmp, "%3(@{}%)", global_names.at(g));
            return tmp;
        }

        if (isa<NilOp>(op)) {
            tmp += "nil";
            return tmp;
        }

        if (auto ty = dyn_cast<TreeConstantOp>(op)) {
            tmp += ty.getTree()->dump(&cg.context());
            return tmp;
        }

        if (auto ty = dyn_cast<TypeConstantOp>(op)) {
            Format(tmp, "{}", ty.getValue());
            return tmp;
        }

        if (isa<LLVM::AllocaOp>(op)) {
            Format(tmp, "%4(#{}%)", Id(frame_ids, op));
            return tmp;
        }

        Unreachable("Unsupported inline op: '{}'", op->getName().getStringRef());
    }

    tmp += "TODO: Print this value properly:";
    llvm::raw_svector_ostream os{tmp};
    v.print(os);
    return tmp;
}
