#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Macros.hh>

#include <clang/Basic/TargetInfo.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <memory>

using namespace srcc;
using namespace srcc::cg;
namespace arith = mlir::arith;
using Vals = mlir::ValueRange;

CodeGen::CodeGen(TranslationUnit& tu, LangOpts lang_opts, Size word_size)
    : CodeGenBase(), OpBuilder(&mlir), tu{tu}, word_size{word_size}, lang_opts{lang_opts} {
    mlir.getDiagEngine().registerHandler([&](mlir::Diagnostic& diag) {
        HandleMLIRDiagnostic(diag);
    });

    ir::SRCCDialect::InitialiseContext(mlir);
    ptr_ty = mlir::LLVM::LLVMPointerType::get(&mlir);
    bool_ty = getI1Type();
    int_ty = C(Type::IntTy);
    closure_ty = mlir::TupleType::get(&mlir, {ptr_ty, ptr_ty});
    slice_ty = mlir::TupleType::get(&mlir, {ptr_ty, getIntegerType(u32(tu.target().int_size().bits()))});
    mlir_module = mlir::ModuleOp::create(C(tu.initialiser_proc->location()), tu.name.value());

    // Declare the abort handlers.
    if (tu.abort_info_type) {
        auto ptr = PtrType::Get(tu, tu.abort_info_type);
        auto ty = ProcType::Get(tu, Type::VoidTy, {ParamTypeData{Intent::In, ptr}});
        GetOrCreateProc({}, constants::AssertFailureHandlerName, Linkage::Imported, ty);
        GetOrCreateProc({}, constants::ArithmeticFailureHandlerName, Linkage::Imported, ty);
    }
}

// ============================================================================
//  AST -> IR
// ============================================================================
auto CodeGen::C(CallingConvention l) -> mlir::LLVM::CConv {
    switch (l) {
        case CallingConvention::Source: return mlir::LLVM::CConv::Fast;
        case CallingConvention::Native: return mlir::LLVM::CConv::C;
    }
    Unreachable();
}

auto CodeGen::C(Linkage l) -> mlir::LLVM::Linkage {
    switch (l) {
        case Linkage::Internal: return mlir::LLVM::Linkage::Private;
        case Linkage::Exported: return mlir::LLVM::Linkage::External;
        case Linkage::Imported: return mlir::LLVM::Linkage::External;
        case Linkage::Reexported: return mlir::LLVM::Linkage::External;
        case Linkage::Merge: return mlir::LLVM::Linkage::LinkonceODR;
    }
    Unreachable();
}

auto CodeGen::C(Location l) -> mlir::Location {
    auto lc = l.seek_line_column(tu.context());
    if (not lc) return getUnknownLoc();
    auto f = tu.context().file_name(l.file_id);
    auto flc = mlir::FileLineColLoc::get(&mlir, f.value(), u32(lc->line), u32(lc->col));
    return mlir::OpaqueLoc::get(l.encode(), mlir::TypeID::get<Location>(), flc);
}

auto CodeGen::C(Type ty) -> mlir::Type {
    // Integer types.
    if (ty == Type::BoolTy) return getIntegerType(1);
    if (ty == Type::IntTy) return getIntegerType(u32(tu.target().int_size().bits()));
    if (auto i = dyn_cast<IntType>(ty)) return getIntegerType(u32(i->bit_width().bits()));

    // Pointer types.
    if (isa<PtrType>(ty)) return ptr_ty;

    // Aggregates need to be represented as structs for extract to work.
    if (isa<ProcType>(ty)) return closure_ty;
    if (isa<SliceType>(ty)) return slice_ty;
    if (auto r = dyn_cast<RangeType>(ty)) {
        auto e = C(r->elem());
        return mlir::TupleType::get(&mlir, {e, e});
    }

    // Structs and arrays are just turned into blobs of bytes.
    auto sz = ty->size(tu);
    if (sz == Size()) return mlir::NoneType::get(&mlir);
    return mlir::LLVM::LLVMArrayType::get(&mlir, getI8Type(), sz.bytes());
}

auto CodeGen::ConvertProcType(ProcType* ty) -> IRProcType {
    IRProcType ptype;
    SmallVector<mlir::Type> arg_types;
    mlir::Type ret = C(ty->ret());

    // Handle indirect returns.
    if (ty->ret()->is_mrvalue()) ptype.has_indirect_return = true;
    else if (IsZeroSizedType(ty->ret())) ret = mlir::NoneType::get(&mlir);

    // Add the remaining args.
    for (auto [intent, t] : ty->params()) {
        if (IsZeroSizedType(t)) continue;
        arg_types.push_back(t->pass_by_lvalue(ty->cconv(), intent) ? ptr_ty : C(t));
    }

    ptype.type = mlir::FunctionType::get(&mlir, arg_types, mlir::TypeRange{ret});
    return ptype;
}

void CodeGen::CreateAbort(Location loc, ir::AbortReason reason, Value msg1, Value msg2) {
    // The runtime defines the assertion payload as follows.
    //
    // struct AbortInfo {
    //    i8[] filename;
    //    int line;
    //    int col;
    //    i8[] msg1;
    //    i8[] msg2;
    // }
    //
    // Make sure the type exists if we need to emit an abort.
    if (not tu.abort_info_type) {
        Warn(loc, "No declaration of '__src_abort_info' found");
        create<mlir::LLVM::UnreachableOp>(C(loc));
        return;
    }

    // Reuse the abort info slot if there is one.
    if (!abort_info_slot)
        abort_info_slot = CreateAlloca(loc, tu.abort_info_type);

    // Get the file name, line, and column.
    //
    // Don’t require a valid location here as this is also called from within
    // implicitly generated code.
    auto l = C(loc);
    StructInitHelper init{*this, tu.abort_info_type, abort_info_slot};
    if (auto lc = loc.seek_line_column(context())) {
        auto name = context().file_name(loc.file_id);
        init.emit_next_field(CreateGlobalStringSlice(loc, name));
        init.emit_next_field(CreateInt(l, i64(lc->line)));
        init.emit_next_field(CreateInt(l, i64(lc->col)));
    } else {
        init.emit_next_field(CreateNil(l, tu.StrLitTy));
        init.emit_next_field(CreateInt(l, 0));
        init.emit_next_field(CreateInt(l, 0));
    }

    // Add the message parts.
    init.emit_next_field(msg1);
    init.emit_next_field(msg2);
    create<ir::AbortOp>(C(loc), reason, abort_info_slot);
}

auto CodeGen::CreateAggregate(mlir::Location loc, Value a, Value b) -> Value {
    return create<ir::TupleOp>(loc, Vals{a, b});
}

auto CodeGen::CreateAlloca(Location loc, Type ty) -> Value {
    InsertionGuard _{*this};

    // Do this stupid garbage because allowing a separate region in a function for
    // frame slots is apparently too much to ask because MLIR complains about both
    // uses not being dominated by the frame slots and about a FunctionOpInterface
    // having more than one region.
    auto it = curr_proc.entry()->begin();
    auto end = curr_proc.entry()->end();
    while (it != end and isa<ir::FrameSlotOp>(*it)) ++it;
    if (it == end) setInsertionPointToEnd(curr_proc.entry());
    else setInsertionPoint(&*it);
    return create<ir::FrameSlotOp>(C(loc), ty->size(tu), ty->align(tu));
}

void CodeGen::CreateArithFailure(Value failure_cond, Tk op, Location loc, String name) {
    If(failure_cond, [&] {
        auto op_token = CreateGlobalStringSlice(loc, Spelling(op));
        auto operation = CreateGlobalStringSlice(loc, name);
        CreateAbort(
            loc,
            ir::AbortReason::ArithmeticError,
            op_token,
            operation
        );
    });
}

template <typename Unchecked, typename Checked>
auto CodeGen::CreateBinop(
    Type,
    Value lhs,
    Value rhs,
    Location loc,
    Tk op
) -> Value {
    if (not lang_opts.overflow_checking) return createOrFold<Unchecked>(C(loc), lhs, rhs);
    SmallVector<Value> values;
    createOrFold<Checked>(values, C(loc), lhs, rhs);
    CreateArithFailure(values[1], op, loc);
    return values[0];
}

auto CodeGen::CreateBlock(ArrayRef<mlir::Type> args) -> std::unique_ptr<Block> {
    std::unique_ptr<Block> b{new Block};
    for (auto ty : args) b->addArgument(ty, getUnknownLoc());
    return b;
}

auto CodeGen::CreateBool(mlir::Location loc, bool b) -> Value {
    return create<arith::ConstantOp>(
        loc,
        getI1Type(),
        mlir::IntegerAttr::get(getI1Type(), b)
    );
}

auto CodeGen::CreateGlobalStringPtr(String data) -> Value {
    return CreateGlobalStringPtr(Align(1), data, true);
}

auto CodeGen::CreateGlobalStringPtr(Align align, String data, bool null_terminated) -> Value {
    auto& i = interned_strings[data];
    if (not i) {
        // TODO: Introduce our own Op for this and mark it as 'Pure'.
        InsertionGuard _{*this};
        setInsertionPointToStart(&mlir_module.getBodyRegion().front());
        i = create<mlir::LLVM::GlobalOp>(
            getUnknownLoc(),
            mlir::LLVM::LLVMArrayType::get(getI8Type(), data.size() + null_terminated),
            true,
            mlir::LLVM::Linkage::Private,
            std::format("__srcc_str.{}", strings++),
            getStringAttr(llvm::Twine(StringRef(data)) + (null_terminated ? "\0"sv : "")),
            align.value().bytes()
        );
    }
    return create<mlir::LLVM::AddressOfOp>(getUnknownLoc(), i);
}

auto CodeGen::CreateGlobalStringSlice(Location loc, String data) -> Value {
    return CreateAggregate(C(loc), CreateGlobalStringPtr(data), CreateInt(C(loc), i64(data.size())));
}

auto CodeGen::CreateICmp(mlir::Location loc, mlir::LLVM::ICmpPredicate pred, Value lhs, Value rhs) -> Value {
    // This is required because arith::cmpi doesn’t support pointers...
    if (isa<mlir::LLVM::LLVMPointerType>(lhs.getType()))
        return createOrFold<mlir::LLVM::ICmpOp>(loc, pred, lhs, rhs);

    auto pred_conv = [&] {
        switch (pred) {
            case mlir::LLVM::ICmpPredicate::eq: return arith::CmpIPredicate::eq;
            case mlir::LLVM::ICmpPredicate::ne: return arith::CmpIPredicate::ne;
            case mlir::LLVM::ICmpPredicate::slt: return arith::CmpIPredicate::slt;
            case mlir::LLVM::ICmpPredicate::sle: return arith::CmpIPredicate::sle;
            case mlir::LLVM::ICmpPredicate::sgt: return arith::CmpIPredicate::sgt;
            case mlir::LLVM::ICmpPredicate::sge: return arith::CmpIPredicate::sge;
            case mlir::LLVM::ICmpPredicate::ult: return arith::CmpIPredicate::ult;
            case mlir::LLVM::ICmpPredicate::ule: return arith::CmpIPredicate::ule;
            case mlir::LLVM::ICmpPredicate::ugt: return arith::CmpIPredicate::ugt;
            case mlir::LLVM::ICmpPredicate::uge: return arith::CmpIPredicate::uge;
        }
        Unreachable();
    }();

    // But we prefer to emit an arith op because it can be folded etc.
    return createOrFold<arith::CmpIOp>(loc, pred_conv, lhs, rhs);
}

auto CodeGen::CreateInt(mlir::Location loc, const APInt& value, Type ty) -> Value {
    return create<arith::ConstantOp>(loc, getIntegerAttr(C(ty), value));
}

auto CodeGen::CreateInt(mlir::Location loc, i64 value, Type ty) -> Value {
    return CreateInt(loc, value, C(ty));
}

auto CodeGen::CreateInt(mlir::Location loc, i64 value, mlir::Type ty) -> Value {
    return create<arith::ConstantOp>(loc, getIntegerAttr(ty, value));
}

auto CodeGen::CreateNil(mlir::Location loc, Type ty) -> Value {
    Assert(not ty->is_mrvalue(), "MRValues don’t have a 'nil' value");
    if (isa<ProcType>(ty)) {
        auto p = CreateNullPointer(loc);
        return CreateAggregate(loc, p, p);
    }

    if (isa<SliceType>(ty)) return CreateAggregate(
        loc,
        CreateNullPointer(loc),
        CreateNil(loc, Type::IntTy)
    );

    if (auto r = dyn_cast<RangeType>(ty)) {
        auto el = CreateNil(loc, r->elem());
        return CreateAggregate(loc, el, el);
    }

    if (ty->is_integer()) return CreateInt(loc, 0, ty);
    Assert(isa<PtrType>(ty));
    return CreateNullPointer(loc);
}

auto CodeGen::CreateNullPointer(mlir::Location loc) -> Value {
    return create<ir::NilOp>(loc, ptr_ty);
}

auto CodeGen::CreatePtrAdd(mlir::Location loc, Value addr, Size offs) -> Value {
    return createOrFold<mlir::LLVM::GEPOp>(
        loc,
        ptr_ty,
        getI8Type(),
        addr,
        mlir::LLVM::GEPArg(i32(offs.bytes())),
        mlir::LLVM::GEPNoWrapFlags::inbounds | mlir::LLVM::GEPNoWrapFlags::nusw | mlir::LLVM::GEPNoWrapFlags::nuw
    );
}

auto CodeGen::CreatePtrAdd(mlir::Location loc, Value addr, Value offs) -> Value {
    return createOrFold<mlir::LLVM::GEPOp>(
        loc,
        ptr_ty,
        getI8Type(),
        addr,
        mlir::LLVM::GEPArg(offs),
        mlir::LLVM::GEPNoWrapFlags::inbounds | mlir::LLVM::GEPNoWrapFlags::nusw | mlir::LLVM::GEPNoWrapFlags::nuw
    );
}

auto CodeGen::CreateSICast(mlir::Location loc, Value val, Type from, Type to) -> Value {
    auto from_sz = from->size(tu);
    auto to_sz = to->size(tu);
    if (from_sz == to_sz) return val;
    if (from_sz > to_sz) return createOrFold<arith::TruncIOp>(loc, C(to), val);
    if (from == Type::BoolTy) return createOrFold<arith::ExtUIOp>(loc, C(to), val);
    return createOrFold<arith::ExtSIOp>(loc, C(to), val);
}

auto CodeGen::DeclarePrintf() -> ir::ProcOp {
    if (not printf) {
        printf = GetOrCreateProc(
            Location(),
            "printf",
            Linkage::Imported,
            ProcType::Get(
                tu,
                tu.FFIIntTy,
                {{Intent::Copy, PtrType::Get(tu, tu.I8Ty)}},
                CallingConvention::Native,
                true
            )
        );
    }

    return printf.value();
}

auto CodeGen::DeclareProcedure(ProcDecl* proc) -> ir::ProcOp {
    auto& ir_proc = declared_procs[proc];
    if (ir_proc) return ir_proc;
    return GetOrCreateProc(
        proc->location(),
        MangledName(proc),
        proc->linkage,
        proc->proc_type()
    );
}

auto CodeGen::EnterBlock(std::unique_ptr<Block> bb, Vals args) -> Block* {
    auto b = bb.release();
    curr_proc.push_back(b);
    return EnterBlock(b, args);
}

auto CodeGen::EnterBlock(Block* bb, Vals args) -> Block* {
    // If there is a current block, and it is not closed, branch to the newly
    // inserted block, unless that block is the function’s entry block.
    auto curr = getInsertionBlock();
    if (curr and not curr->mightHaveTerminator())
        create<mlir::cf::BranchOp>(getUnknownLoc(), bb, args);

    // Finally, position the builder at the end of the block.
    setInsertionPointToEnd(bb);
    return bb;
}

CodeGen::EnterProcedure::EnterProcedure(CodeGen& CG, ir::ProcOp proc)
    : CG(CG), old_func(CG.curr_proc), guard{CG} {
    CG.curr_proc = proc;
    CG.EnterBlock(proc.entry());
}

auto CodeGen::GetOrCreateProc(Location loc, String name, Linkage linkage, ProcType* ty) -> ir::ProcOp {
    if (auto op = mlir_module.lookupSymbol<ir::ProcOp>(name)) return op;
    InsertionGuard _{*this};
    setInsertionPointToEnd(&mlir_module.getBodyRegion().front());
    auto [ftype, has_indirect_return] = ConvertProcType(ty);
    auto ir_proc = create<ir::ProcOp>(
        C(loc),
        name,
        C(linkage),
        C(ty->cconv()),
        ftype,
        ty->variadic(),
        has_indirect_return,
        false
    );

    // Erase the body if this is just a declaration; additionally, declarations
    // can’t have public visibility, so set it to private in that case.
    if (linkage == Linkage::Imported or linkage == Linkage::Reexported) {
        ir_proc.eraseBody();
        ir_proc.setVisibility(mlir::SymbolTable::Visibility::Private);
    }

    return ir_proc;
}

void CodeGen::HandleMLIRDiagnostic(mlir::Diagnostic& diag) {
    auto EmitDiagnostic = [&](mlir::Diagnostic& d) {
        auto level = [&] {
            switch (d.getSeverity()) {
                case mlir::DiagnosticSeverity::Note: return Diagnostic::Level::Note;
                case mlir::DiagnosticSeverity::Warning: return Diagnostic::Level::Warning;
                case mlir::DiagnosticSeverity::Error: return Diagnostic::Level::ICE;
                case mlir::DiagnosticSeverity::Remark: return Diagnostic::Level::Warning;
            }
            Unreachable();
        }();

        auto loc = Location::Decode(d.getLocation()).value_or({});
        auto msg = d.str();
        Diag(level, loc, "{}", utils::Escape(msg, false, true));
    };

    EmitDiagnostic(diag);
    for (auto& n : diag.getNotes())
        EmitDiagnostic(n);
}

auto CodeGen::If(
    Value cond,
    llvm::function_ref<Value()> emit_then,
    llvm::function_ref<Value()> emit_else
) -> Block* {
    if (not emit_else) {
        If(cond, emit_then);
        return nullptr;
    }

    auto bb_then = CreateBlock();
    auto bb_else = CreateBlock();
    auto bb_join = CreateBlock();
    create<mlir::cf::CondBranchOp>(getUnknownLoc(), cond, bb_then.get(), bb_else.get());

    // Emit the then block.
    EnterBlock(std::move(bb_then));
    auto then_val = emit_then();

    // Branch to the join block.
    if (then_val) bb_join->addArgument(then_val.getType(), getUnknownLoc());
    if (not getInsertionBlock()->mightHaveTerminator()) create<mlir::cf::BranchOp>(
        getUnknownLoc(),
        bb_join.get(),
        then_val ? Vals(then_val) : Vals()
    );

    // And emit the else block.
    EnterBlock(std::move(bb_else));
    auto else_val = emit_else();
    if (not getInsertionBlock()->mightHaveTerminator()) create<mlir::cf::BranchOp>(
        getUnknownLoc(),
        bb_join.get(),
        else_val ? Vals(else_val) : Vals()
    );

    // Finally, return the join block.
    return EnterBlock(std::move(bb_join));
}

auto CodeGen::If(Value cond, Vals args, llvm::function_ref<void()> emit_then) -> Block* {
    SmallVector<mlir::Type, 3> types;
    for (auto arg : args) types.push_back(arg.getType());

    // Create the blocks and branch to the body.
    auto body = CreateBlock(types);
    auto join = CreateBlock();
    create<mlir::cf::CondBranchOp>(
        getUnknownLoc(),
        cond,
        body.get(),
        args,
        join.get(),
        Vals()
    );

    // Emit the body and close the block.
    EnterBlock(std::move(body));
    emit_then();
    if (not getInsertionBlock()->mightHaveTerminator()) create<mlir::cf::BranchOp>(
        getUnknownLoc(),
        join.get()
    );

    // Add the join block.
    return EnterBlock(std::move(join));
}

bool CodeGen::IsZeroSizedType(Type ty) {
    return ty->size(tu) == Size();
}

bool CodeGen::LocalNeedsAlloca(LocalDecl* local) {
    if (IsZeroSizedType(local->type)) return false;
    if (local->category == Expr::SRValue) return false;
    auto p = dyn_cast<ParamDecl>(local);
    if (not p) return true;
    return not p->type->pass_by_lvalue(p->parent->cconv(), p->intent());
}

void CodeGen::Loop(llvm::function_ref<void()> emit_body) {
    auto bb_cond = EnterBlock(CreateBlock());
    emit_body();
    EnterBlock(bb_cond);
}

void CodeGen::Unless(Value cond, llvm::function_ref<void()> emit_else) {
    // Create the blocks and branch to the body.
    auto else_ = CreateBlock();
    auto join = CreateBlock();
    create<mlir::cf::CondBranchOp>(
        getUnknownLoc(),
        cond,
        join.get(),
        else_.get()
    );

    // Emit the body and close the block.
    EnterBlock(std::move(else_));
    emit_else();
    if (not getInsertionBlock()->mightHaveTerminator()) create<mlir::cf::BranchOp>(
        getUnknownLoc(),
        join.get()
    );

    // Add the join block.
    EnterBlock(std::move(join));
}

void CodeGen::While(
    llvm::function_ref<Value()> emit_cond,
    llvm::function_ref<void()> emit_body
) {
    auto bb_cond = CreateBlock();
    auto bb_body = CreateBlock();
    auto bb_end = CreateBlock();
    auto cond = bb_cond.get();

    // Emit condition.
    EnterBlock(std::move(bb_cond));
    create<mlir::cf::CondBranchOp>(getUnknownLoc(), emit_cond(), bb_body.get(), bb_end.get());

    // Emit body.
    EnterBlock(std::move(bb_body));
    emit_body();
    create<mlir::cf::BranchOp>(getUnknownLoc(), cond);

    // Continue after the loop.
    EnterBlock(std::move(bb_end));
}

// ============================================================================
//  Mangling
// ============================================================================
// Mangling codes:
//
// TODO: Yeet all of this; we don’t need name mangling because of how our modules
// work (keep a pretty name in the IR and generate some random nonsense for LLVM IR).
struct CodeGen::Mangler {
    CodeGen& CG;
    std::string name;

    explicit Mangler(CodeGen& CG, ProcDecl* proc) : CG(CG) {
        name = "_S";
        Append(proc->name);
        Append(proc->type);
    }

    void Append(StringRef s);
    void Append(Type ty);
};

void CodeGen::Mangler::Append(StringRef s) {
    Assert(not s.empty());
    if (not llvm::isAlpha(s.front())) name += "$";
    name += std::format("{}{}", s.size(), s);
}

// INVARIANT: No mangling code starts with a number (this means
// mangling codes can *end* with a number without requirding a
// separator).
void CodeGen::Mangler::Append(Type ty) {
    // FIXME: Throw out name mangling entirely and instead maintain a mapping
    // to automatically generated names in the module.
    struct Visitor {
        Mangler& M;

        void ElemTy(StringRef s, SingleElementTypeBase* t) {
            M.name += s;
            t->elem()->visit(*this);
        }

        void operator()(SliceType* sl) { ElemTy("S", sl); }
        void operator()(ArrayType* arr) { ElemTy(std::format("A{}", arr->dimension()), arr); }
        void operator()(PtrType* ref) { ElemTy("R", ref); }
        void operator()(IntType* i) { M.name += std::format("I{}", i->bit_width().bits()); }
        void operator()(BuiltinType* b) {
            switch (b->builtin_kind()) {
                case BuiltinKind::Deduced:
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable("Can’t mangle this: {}", Type{b});
                case BuiltinKind::Void: M.name += "v"; return;
                case BuiltinKind::NoReturn: M.name += "z"; return;
                case BuiltinKind::Bool: M.name += "b"; return;
                case BuiltinKind::Int: M.name += "i"; return;
            }
            Unreachable();
        }

        void operator()(ProcType* proc) {
            M.name += "F";
            proc->ret()->visit(*this);
            for (const auto& p : proc->params()) {
                M.name += [&] {
                    switch (p.intent) {
                        case Intent::Move: return "";
                        case Intent::In: return "x";
                        case Intent::Out: return "x1";
                        case Intent::Inout: return "x2";
                        case Intent::Copy: return "x3";
                    }
                    Unreachable();
                }();
                p.type->visit(*this);
            }
            M.name += "E";
        }

        void operator()(RangeType* r) {
            M.name += "q";
            r->elem()->visit(*this);
        }

        void operator()(StructType* ty) {
            if (ty->name().empty()) Todo();
            M.name += std::format("T{}{}", ty->name().size(), ty->name());
        }
    };

    ty->visit(Visitor{*this});
}

auto CodeGen::MangledName(ProcDecl* proc) -> String {
    if (proc->mangling == Mangling::None) return proc->name;

    // Maybe we’ve already cached this?
    auto it = mangled_names.find(proc);
    if (it != mangled_names.end()) return it->second;

    // Compute it.
    auto name = [&] -> std::string {
        switch (proc->mangling) {
            case Mangling::None: Unreachable();
            case Mangling::Source: return std::move(Mangler(*this, proc).name);
            case Mangling::CXX: Todo("Mangle C++ function name");
        }
        Unreachable("Invalid mangling");
    }();

    return mangled_names[proc] = tu.save(name);
}

// ============================================================================
//  Initialisation.
// ============================================================================
void CodeGen::StructInitHelper::emit_next_field(Value v) {
    Assert(i < ty->fields().size());
    auto field = ty->fields()[i++];
    auto ptr = CG.CreatePtrAdd(v.getLoc(), base, field->offset);
    CG.create<ir::StoreOp>(v.getLoc(), ptr, v, field->type->align(CG.tu));
}

void CodeGen::EmitInitialiser(Value addr, Expr* init) {
    Assert(not IsZeroSizedType(init->type), "Should have been checked before calling this");
    if (init->type->is_mrvalue()) EmitMRValue(addr, init);
    else create<ir::StoreOp>(C(init->location()), addr, Emit(init), init->type->align(tu));
}

void CodeGen::EmitMRValue(Value addr, Expr* init) { // clang-format off
    Assert(addr, "Emitting mrvalue without address?");

    // We support treating lvalues as mrvalues.
    if (init->value_category == Expr::LValue) {
        auto loc = C(init->location());
        create<mlir::LLVM::MemcpyOp>(
            loc,
            addr,
            Emit(init),
            CreateInt(loc, i64(init->type->size(tu).bytes())),
            false
        );
        return;
    }

    // Otherwise, we need to emit an mrvalue; only select kinds of expressions
    // can produce mrvalues; we need to handle each of them here.
    init->visit(utils::Overloaded{
        [&](auto*) { ICE(init->location(), "Invalid mrvalue"); },

        // Array initialisers are always mrvalues.
        [&](ArrayBroadcastExpr* e) { EmitArrayBroadcastExpr(e, addr); },
        [&](ArrayInitExpr* e) { EmitArrayInitExpr(e, addr); },

        // Blocks can be mrvalues if the last expression is an mrvalue.
        [&](BlockExpr* e) { EmitBlockExpr(e, addr); },

        // If the initialiser is a call, pass the address to it.
        [&](CallExpr* e) { EmitCallExpr(e, addr); },

        // If the initialiser is a constant expression, create a global constant for it.
        //
        // Yes, this means that this is basically an lvalue that we’re copying from;
        // the only reason the language treats it as an mrvalue is because it would
        // logically be rather weird to be able to take the address of an evaluated
        // struct literal (but not of other rvalues).
        //
        // CreateUnsafe() is fine here since mrvalues are allocated in the TU.
        [&](ConstExpr* e) {
            auto mrv = e->value->cast<eval::MRValue>();
            auto c = CreateGlobalStringPtr(
                init->type->align(tu),
                String::CreateUnsafe(static_cast<char*>(mrv.data()), mrv.size().bytes()),
                false
            );

            auto loc = C(init->location());
            create<mlir::LLVM::MemcpyOp>(
                loc,
                addr,
                c,
                CreateInt(loc, i64(init->type->size(tu).bytes())),
                false
            );
        },

        // Default initialiser here is a memset to 0.
        [&](DefaultInitExpr* e) {
            auto loc = C(init->location());
            create<mlir::LLVM::MemsetOp>(
                loc,
                addr,
                CreateInt(loc, 0, tu.I8Ty),
                CreateInt(loc, i64(e->type->size(tu).bytes())),
                false
            );
        },

        // If expressions can be mrvalues if either branch is an mrvalue.
        [&](IfExpr* e) { EmitIfExpr(e, addr); },

        // Structs literals are emitted field by field.
        [&](StructInitExpr* e) {
            auto s = e->struct_type();
            for (auto [field, val] : zip(s->fields(), e->values())) {
                if (IsZeroSizedType(field->type)) {
                    Emit(val);
                    continue;
                }

                auto offs = CreatePtrAdd(C(val->location()), addr, field->offset);
                EmitInitialiser(offs, val);
            }
        }
    });
} // clang-format on

// ============================================================================
//  CG
// ============================================================================
void CodeGen::Emit(ArrayRef<ProcDecl*> procs) {
    for (auto& p : procs) EmitProcedure(p);
}

auto CodeGen::Emit(Stmt* stmt) -> Value {
    Assert(not stmt->is_mrvalue(), "Should call EmitMRValue() instead");
    switch (stmt->kind()) {
        using K = Stmt::Kind;
#define AST_DECL_LEAF(node) \
    case K::node: Unreachable("Cannot emit " SRCC_STR(node));
#define AST_STMT_LEAF(node) \
    case K::node: return SRCC_CAT(Emit, node)(cast<node>(stmt));
#include "srcc/AST.inc"
    }

    Unreachable("Unknown statement kind");
}

auto CodeGen::EmitArrayBroadcastExpr(ArrayBroadcastExpr*) -> Value {
    Unreachable("Should only be emitted as mrvalue");
}

void CodeGen::EmitArrayBroadcast(Type elem_ty, Value addr, u64 elements, Expr* initialiser, Location loc) {
    auto l = C(loc);
    auto counter = CreateInt(l, 0);
    auto dimension = CreateInt(l, i64(elements));
    auto bb_cond = EnterBlock(CreateBlock(int_ty), counter);
    auto bb_body = CreateBlock();
    auto bb_end = CreateBlock();

    // Condition.
    auto eq = EmitArithmeticOrComparisonOperator(
        Tk::EqEq,
        Type::IntTy,
        bb_cond->getArgument(0),
        dimension,
        loc
    );
    create<mlir::cf::CondBranchOp>(l, eq, bb_end.get(), bb_body.get());

    // Initialisation.
    EnterBlock(std::move(bb_body));
    auto mul = create<arith::MulIOp>(
        l,
        bb_cond->getArgument(0),
        CreateInt(l, i64(elem_ty->array_size(tu).bytes())),
        arith::IntegerOverflowFlags::nuw // FIXME: Should this be nsw too?
    );

    auto ptr = CreatePtrAdd(l, addr, mul);
    EmitInitialiser(ptr, initialiser);

    // Increment.
    auto incr = create<arith::AddIOp>(l, bb_cond->getArgument(0), CreateInt(l, 1));
    create<mlir::cf::BranchOp>(l, bb_cond, incr.getResult());

    // Join.
    EnterBlock(std::move(bb_end));
}

void CodeGen::EmitArrayBroadcastExpr(ArrayBroadcastExpr* e, Value mrvalue_slot) {
    auto ty = cast<ArrayType>(e->type);
    Assert(ty->dimension() > 1); // For dimension = 1 we create an ArrayInitExpr instead.
    EmitArrayBroadcast(
        ty->elem(),
        mrvalue_slot,
        u64(ty->dimension()),
        e->element,
        e->location()
    );
}

auto CodeGen::EmitArrayInitExpr(ArrayInitExpr*) -> Value {
    Unreachable("Should only be emitted as mrvalue");
}

void CodeGen::EmitArrayInitExpr(ArrayInitExpr* e, Value mrvalue_slot) {
    auto ty = cast<ArrayType>(e->type);
    bool broadcast_els = u64(ty->dimension()) - e->initialisers().size();

    // Emit each initialiser.
    for (auto init : e->initialisers()) {
        EmitInitialiser(mrvalue_slot, init);
        if (init != e->initialisers().back() or broadcast_els != 0) {
            mrvalue_slot = CreatePtrAdd(
                C(init->location()),
                mrvalue_slot,
                ty->elem()->array_size(tu)
            );
        }
    }

    // And use the last initialiser to fill the rest of the array, if need be.
    if (broadcast_els) EmitArrayBroadcast(
        ty->elem(),
        mrvalue_slot,
        broadcast_els,
        e->broadcast_init(),
        e->location()
    );
}

auto CodeGen::EmitAssertExpr(AssertExpr* expr) -> Value {
    auto loc = expr->location().seek_line_column(tu.context());
    if (not loc) {
        ICE(expr->location(), "No location for assert");
        return {};
    }

    Unless(Emit(expr->cond), [&] {
        Value msg{};
        if (auto m = expr->message.get_or_null()) msg = Emit(m);
        else msg = CreateNil(C(expr->location()), tu.StrLitTy);
        auto cond_str = CreateGlobalStringSlice(expr->cond->location(), expr->cond->location().text(tu.context()));
        CreateAbort(
            expr->location(),
            ir::AbortReason::AssertionFailed,
            cond_str,
            msg
        );
    });

    return {};
}

auto CodeGen::EmitBinaryExpr(BinaryExpr* expr) -> Value {
    auto l = C(expr->location());
    switch (expr->op) {
        // Convert 'x and y' to 'if x then y else false'.
        case Tk::And: {
            return If(
                Emit(expr->lhs),
                [&] { return Emit(expr->rhs); },
                [&] { return CreateBool(l, false); }
            )->getArgument(0);
        }

        // Convert 'x or y' to 'if x then true else y'.
        case Tk::Or: {
            return If(
                Emit(expr->lhs),
                [&] { return CreateBool(l, true); },
                [&] { return Emit(expr->rhs); }
            )->getArgument(0);
        }

        // Assignment.
        case Tk::Assign: {
            auto addr = Emit(expr->lhs);
            if (IsZeroSizedType(expr->lhs->type)) Emit(expr->rhs);
            else EmitInitialiser(addr, expr->rhs);
            return addr;
        }

        // Subscripting is supported for slices and arrays.
        case Tk::LBrack: {
            Assert(
                (isa<ArrayType, SliceType>(expr->lhs->type)),
                "Unsupported type: {}",
                expr->lhs->type
            );

            auto range = Emit(expr->lhs);
            auto index = Emit(expr->rhs);
            bool is_slice = isa<SliceType>(expr->lhs->type);

            // Check that the index is in bounds.
            if (lang_opts.overflow_checking) {
                auto size = is_slice
                    ? createOrFold<ir::ExtractOp>(C(expr->lhs->location()), range, i64(1))
                    : CreateInt(l, cast<ArrayType>(expr->lhs->type)->dimension());

                CreateArithFailure(
                    CreateICmp(l, mlir::LLVM::ICmpPredicate::uge, index, size),
                    Tk::LBrack,
                    expr->location(),
                    "out of bounds access"
                );
            }

            Size elem_size = cast<SingleElementTypeBase>(expr->lhs->type)->elem()->array_size(tu);
            return CreatePtrAdd(
                l,
                is_slice ? createOrFold<ir::ExtractOp>(C(expr->lhs->location()), range, i64(0)) : range,
                createOrFold<arith::MulIOp>(l, CreateInt(l, i64(elem_size.bytes())), index)
            );
        }

        // Arithmetic assignment operators.
        case Tk::PlusEq:
        case Tk::PlusTildeEq:
        case Tk::MinusEq:
        case Tk::MinusTildeEq:
        case Tk::StarEq:
        case Tk::StarTildeEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftLeftLogicalEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq: {
            auto a = expr->lhs->type->align(tu);
            auto lvalue = Emit(expr->lhs);
            auto lhs = create<ir::LoadOp>(
                lvalue.getLoc(),
                C(expr->lhs->type),
                lvalue,
                a
            );

            auto rhs = Emit(expr->rhs);
            auto res = EmitArithmeticOrComparisonOperator(
                StripAssignment(expr->op),
                expr->lhs->type,
                lhs,
                rhs,
                expr->location()
            );

            create<ir::StoreOp>(
                C(expr->location()),
                lvalue,
                res,
                a
            );
            return lvalue;
        }

        // Anything else.
        default: return EmitArithmeticOrComparisonOperator(
            expr->op,
            expr->lhs->type,
            Emit(expr->lhs),
            Emit(expr->rhs),
            expr->location()
        );
    }
}

auto CodeGen::EmitArithmeticOrComparisonOperator(Tk op, Type type, Value lhs, Value rhs, Location ast_loc) -> Value {
    using enum OverflowBehaviour;
    auto ty = rhs.getType();
    auto loc = C(ast_loc);
    Assert(
        lhs.getType() == ty,
        "Sema should have converted these to the same type"
    );

    auto CheckDivByZero = [&] {
        auto check = CreateICmp(loc, mlir::LLVM::ICmpPredicate::eq, rhs, CreateInt(loc, 0, ty));
        CreateArithFailure(check, op, ast_loc, "division by zero");
    };

    auto CreateCheckedBinop = [&]<typename Unchecked, typename Checked>() -> Value {
        return CreateBinop<Unchecked, Checked>(type, lhs, rhs, ast_loc, op);
    };

    switch (op) {
        default: Todo("Codegen for '{}'", op);

        // 'and' and 'or' require lazy evaluation and are handled elsewhere.
        case Tk::And:
        case Tk::Or:
            Unreachable("'and' and 'or' cannot be handled here.");

        // Comparison operators.
        case Tk::ULt: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::ult, lhs, rhs);
        case Tk::UGt: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::ugt, lhs, rhs);
        case Tk::ULe: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::ule, lhs, rhs);
        case Tk::UGe: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::uge, lhs, rhs);
        case Tk::SLt: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::slt, lhs, rhs);
        case Tk::SGt: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::sgt, lhs, rhs);
        case Tk::SLe: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::sle, lhs, rhs);
        case Tk::SGe: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::sge, lhs, rhs);
        case Tk::EqEq: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::eq, lhs, rhs);
        case Tk::Neq: return CreateICmp(loc, mlir::LLVM::ICmpPredicate::ne, lhs, rhs);

        // Arithmetic operators that wrap or can’t overflow.
        case Tk::PlusTilde: return createOrFold<arith::AddIOp>(loc, lhs, rhs);
        case Tk::MinusTilde: return createOrFold<arith::SubIOp>(loc, lhs, rhs);
        case Tk::StarTilde: return createOrFold<arith::MulIOp>(loc, lhs, rhs);
        case Tk::ShiftRight: return createOrFold<arith::ShRSIOp>(loc, lhs, rhs);
        case Tk::ShiftRightLogical: return createOrFold<arith::ShRUIOp>(loc, lhs, rhs);
        case Tk::Ampersand: return createOrFold<arith::AndIOp>(loc, lhs, rhs);
        case Tk::VBar: return createOrFold<arith::OrIOp>(loc, lhs, rhs);
        case Tk::Xor: return createOrFold<arith::XOrIOp>(loc, lhs, rhs);

        // Arithmetic operators for which there is an intrinsic
        // that can perform overflow checking.
        case Tk::Plus: return CreateCheckedBinop.operator()< // clang-format off
            arith::AddIOp,
            ir::SAddOvOp
        >();

        case Tk::Minus: return CreateCheckedBinop.operator()<
            arith::SubIOp,
            ir::SSubOvOp
        >();

        case Tk::Star: return CreateCheckedBinop.operator()<
            arith::MulIOp,
            ir::SMulOvOp
        >(); // clang-format on

        // Division only requires a check for division by zero.
        case Tk::ColonSlash:
        case Tk::ColonPercent: {
            CheckDivByZero();
            return op == Tk::ColonSlash
                     ? createOrFold<arith::DivUIOp>(loc, lhs, rhs)
                     : createOrFold<arith::RemUIOp>(loc, lhs, rhs);
        }

        // Signed division additionally has to check for overflow, which
        // happens only if we divide INT_MIN by -1.
        case Tk::Slash:
        case Tk::Percent: {
            CheckDivByZero();
            auto int_min = CreateInt(loc, APInt::getSignedMinValue(u32(type->size(tu).bits())), type);
            auto minus_one = CreateInt(loc, -1, ty);
            auto check_lhs = CreateICmp(loc, mlir::LLVM::ICmpPredicate::eq, lhs, int_min);
            auto check_rhs = CreateICmp(loc, mlir::LLVM::ICmpPredicate::eq, rhs, minus_one);
            CreateArithFailure(
                createOrFold<arith::AndIOp>(loc, check_lhs, check_rhs),
                op,
                ast_loc
            );

            return op == Tk::Slash
                ? createOrFold<arith::DivSIOp>(loc, lhs, rhs)
                : createOrFold<arith::RemSIOp>(loc, lhs, rhs);
        }

        // Left shift overflows if the shift amount is equal
        // to or exceeds the bit width.
        case Tk::ShiftLeftLogical: {
            auto check = CreateICmp(loc, mlir::LLVM::ICmpPredicate::uge, rhs, CreateInt(loc, i64(type->size(tu).bits())));
            CreateArithFailure(check, op, ast_loc, "shift amount exceeds bit width");
            return createOrFold<arith::ShLIOp>(loc, lhs, rhs);
        }

        // Signed left shift additionally does not allow a sign change.
        case Tk::ShiftLeft: {
            auto check = CreateICmp(loc, mlir::LLVM::ICmpPredicate::uge, rhs, CreateInt(loc, i64(type->size(tu).bits())));
            CreateArithFailure(check, op, ast_loc, "shift amount exceeds bit width");

            // Check sign.
            auto res = createOrFold<arith::ShLIOp>(loc, lhs, rhs);
            auto sign = createOrFold<arith::ShRSIOp>(loc, lhs, CreateInt(loc, i64(type->size(tu).bits()) - 1));
            auto new_sign = createOrFold<arith::ShRSIOp>(loc, res, CreateInt(loc, i64(type->size(tu).bits()) - 1));
            auto sign_change = CreateICmp(loc, mlir::LLVM::ICmpPredicate::ne, sign, new_sign);
            CreateArithFailure(sign_change, op, ast_loc);
            return res;
        }

        // Range expressions.
        case Tk::DotDotLess: return CreateAggregate(loc, lhs, rhs);
        case Tk::DotDotEq: return CreateAggregate(
            loc,
            lhs,
            createOrFold<arith::AddIOp>(loc, rhs, CreateInt(loc, 1, rhs.getType()))
        );

        // This is lowered to a call to a compiler-generated function.
        case Tk::StarStar: Unreachable("Sema should have converted this to a call");
    }
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> Value {
    return EmitBlockExpr(expr, nullptr);
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr, Value mrvalue_slot) -> Value {
    Value ret = nullptr;
    for (auto s : expr->stmts()) {
        // Initialise variables.
        //
        // The variable may not exist yet if e.g. we started evaluating a block
        // in the middle of a function at compile-time.
        if (auto var = dyn_cast<LocalDecl>(s)) {
            Assert(var->init, "Sema should always create an initialiser for local vars");
            if (IsZeroSizedType(var->type)) {
                Emit(var->init.get());
            } else {
                if (not locals.contains(var)) EmitLocal(var);
                EmitInitialiser(locals.at(var), var->init.get());
            }
        }

        // Emitting any other declarations is a no-op. So is emitting constant
        // expressions that are unused.
        if (isa<ConstExpr, Decl>(s)) continue;

        // This is the expression we need to return from the block.
        if (s == expr->return_expr()) {
            if (s->is_mrvalue()) EmitMRValue(mrvalue_slot, cast<Expr>(s));
            else ret = Emit(s);
        }

        // This is an mrvalue expression that is not the return value; we
        // allow these here, but we need to provide stack space for them.
        else if (s->is_mrvalue()) {
            auto e = cast<Expr>(s);
            auto l = CreateAlloca(e->location(), e->type);
            EmitMRValue(l, e);
        }

        // Otherwise, this is a regular statement or expression.
        else { Emit(s); }
    }
    return ret;
}

auto CodeGen::EmitBoolLitExpr(BoolLitExpr* stmt) -> Value {
    return CreateBool(C(stmt->location()), stmt->value);
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> Value {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = DeclarePrintf();
            auto ref = create<ir::ProcRefOp>(C(expr->location()), printf);
            for (auto a : expr->args()) {
                auto loc = C(a->location());
                if (a->type == tu.StrLitTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto str_format = CreateGlobalStringPtr("%.*s");
                    auto slice = Emit(a);
                    auto data = createOrFold<ir::ExtractOp>(loc, slice, i64(0));
                    auto size = CreateSICast(loc, createOrFold<ir::ExtractOp>(loc, slice, i64(1)), Type::IntTy, tu.FFIIntTy);
                    create<ir::CallOp>(loc, ref, Vals{str_format, size, data});
                }

                else if (a->type == Type::IntTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto int_format = CreateGlobalStringPtr("%" PRId64);
                    auto val = Emit(a);
                    create<ir::CallOp>(loc, ref, Vals{int_format, val});
                }

                else if (a->type == Type::BoolTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto bool_format = CreateGlobalStringPtr("%s");
                    auto val = Emit(a);
                    auto str = createOrFold<ir::SelectOp>(loc, val, CreateGlobalStringPtr("true"), CreateGlobalStringPtr("false"));
                    create<ir::CallOp>(loc, ref, Vals{bool_format, str});
                }

                else {
                    ICE(
                        a->location(),
                        "Sorry, can’t print this type yet: {}",
                        a->type
                    );
                }
            }
            return nullptr;
        }

        case BuiltinCallExpr::Builtin::Unreachable: {
            create<mlir::LLVM::UnreachableOp>(C(expr->location()));
            EnterBlock(CreateBlock());
            return nullptr;
        }
    }

    Unreachable("Unknown builtin");
}

auto CodeGen::EmitBuiltinMemberAccessExpr(BuiltinMemberAccessExpr* expr) -> Value {
    auto l = C(expr->location());
    switch (expr->access_kind) {
        using AK = BuiltinMemberAccessExpr::AccessKind;
        case AK::SliceData: return createOrFold<ir::ExtractOp>(l, Emit(expr->operand), 0);
        case AK::SliceSize: return createOrFold<ir::ExtractOp>(l, Emit(expr->operand), 1);
        case AK::RangeStart: return createOrFold<ir::ExtractOp>(l, Emit(expr->operand), 0);
        case AK::RangeEnd: return createOrFold<ir::ExtractOp>(l, Emit(expr->operand), 1);
        case AK::TypeAlign: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->align(tu).value().bytes()));
        case AK::TypeArraySize: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->array_size(tu).bytes()));
        case AK::TypeBits: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->size(tu).bits()));
        case AK::TypeBytes: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->size(tu).bytes()));
        case AK::TypeName: return CreateGlobalStringSlice(expr->location(), tu.save(StripColours(cast<TypeExpr>(expr->operand)->value->print())));
        case AK::TypeMaxVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(l, APInt::getSignedMaxValue(u32(ty->size(tu).bits())), ty);
        }
        case AK::TypeMinVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(l, APInt::getSignedMinValue(u32(ty->size(tu).bits())), ty);
        }
    }
    Unreachable();
}

auto CodeGen::EmitCallExpr(CallExpr* expr) -> Value {
    return EmitCallExpr(expr, nullptr);
}

auto CodeGen::EmitCallExpr(CallExpr* expr, Value mrvalue_slot) -> Value {
    auto ty = cast<ProcType>(expr->callee->type);
    auto has_result = not IsZeroSizedType(ty->ret()) and not ty->ret()->is_mrvalue();

    // Callee is evaluated first.
    auto callee = Emit(expr->callee);

    // Evaluate the arguments and add them to the call.
    SmallVector<Value> args;
    for (auto arg : expr->args()) {
        auto v = Emit(arg);
        if (not IsZeroSizedType(arg->type)) args.push_back(v);
    }

    auto op = create<ir::CallOp>(
        C(expr->location()),
        has_result ? mlir::TypeRange(C(ty->ret())) : mlir::TypeRange(),
        createOrFold<ir::ExtractOp>(callee.getLoc(), callee, 0),
        createOrFold<ir::ExtractOp>(callee.getLoc(), callee, 1),
        mrvalue_slot,
        C(ty->cconv()),
        ConvertProcType(ty).type,
        ty->variadic(),
        args
    );

    // Calls are one of the very few expressions whose type can be 'noreturn', so
    // take care to handle that here; omitting this would still work, but doing so
    // allows us to throw away a lot of dead code.
    if (ty->ret() == Type::NoReturnTy) {
        create<mlir::LLVM::UnreachableOp>(C(expr->location()));
        EnterBlock(CreateBlock());
    }

    if (has_result) return op.getRes();
    return {};
}

auto CodeGen::EmitCastExpr(CastExpr* expr) -> Value {
    auto val = Emit(expr->arg);
    switch (expr->kind) {
        case CastExpr::Deref:
            return val; // This is a no-op like prefix '^'.

        case CastExpr::Integral:
            return CreateSICast(C(expr->location()), val, expr->arg->type, expr->type);

        case CastExpr::LValueToSRValue:
            Assert(expr->arg->value_category == Expr::LValue);
            if (IsZeroSizedType(expr->type)) return nullptr;
            return create<ir::LoadOp>(C(expr->location()), C(expr->type), val, expr->type->align(tu));

        case CastExpr::MaterialisePoisonValue:
            return create<mlir::LLVM::PoisonOp>(
                C(expr->location()),
                expr->value_category == Expr::SRValue
                    ? C(expr->type)
                    : ptr_ty
            );
    }

    Unreachable();
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> Value {
    return EmitValue(constant->location(), *constant->value);
}

auto CodeGen::EmitDefaultInitExpr(DefaultInitExpr* stmt) -> Value {
    if (IsZeroSizedType(stmt->type)) return nullptr;
    Assert(stmt->type->rvalue_category() == Expr::SRValue, "Emitting non-srvalue on its own?");
    return CreateNil(C(stmt->location()), stmt->type);
}

auto CodeGen::EmitEmptyStmt(EmptyStmt*) -> Value {
    return nullptr;
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> Value {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitForStmt(ForStmt* stmt) -> Value {
    SmallVector<Value> ranges;
    SmallVector<Value> args;
    SmallVector<Value> end_vals;
    SmallVector<mlir::Type> arg_types;
    auto bb_end = CreateBlock();
    auto floc = C(stmt->location());

    // Emit the ranges in order.
    for (auto r : stmt->ranges()) {
        Assert((isa<RangeType, ArrayType>(r->type)));
        ranges.push_back(Emit(r));
    }

    // Add the enumerator.
    auto* enum_var = stmt->enum_var.get_or_null();
    if (enum_var) {
        arg_types.push_back(C(enum_var->type));
        args.push_back(CreateInt(floc, 0, enum_var->type));
    }

    // Collect all loop variables. The enumerator is always at index 0.
    for (auto [r, expr] : zip(ranges, stmt->ranges())) {
        if (isa<RangeType>(expr->type)) {
            auto start = createOrFold<ir::ExtractOp>(r.getLoc(), r, 0);
            auto end = createOrFold<ir::ExtractOp>(r.getLoc(), r, 1);
            arg_types.push_back(start.getType());
            args.push_back(start);
            end_vals.push_back(end);
        } else if (auto a = dyn_cast<ArrayType>(expr->type)) {
            arg_types.push_back(ptr_ty);
            args.push_back(r);
            end_vals.push_back(CreatePtrAdd(r.getLoc(), r, a->size(tu)));
        } else {
            Unreachable("Invalid for range type: {}", expr->type);
        }
    }

    // Branch to the condition block.
    auto bb_cond = EnterBlock(CreateBlock(arg_types), args);

    // Add the loop variables to the current scope.
    auto block_args = bb_cond->getArguments().drop_front(enum_var ? 1 : 0);
    if (enum_var) locals[enum_var] = bb_cond->getArgument(0);
    for (auto [v, a] : zip(stmt->vars(), block_args)) locals[v] = a;

    // If we have multiple ranges, break if we’ve reach the end of any one of them.
    for (auto [a, e] : zip(block_args, end_vals)) {
        auto bb_cont = CreateBlock();
        auto ne = create<mlir::LLVM::ICmpOp>(floc, mlir::LLVM::ICmpPredicate::ne, a, e);
        create<mlir::cf::CondBranchOp>(floc, ne, bb_cont.get(), bb_end.get());
        EnterBlock(std::move(bb_cont));
    }

    // Body.
    Emit(stmt->body);

    // Remove the loop variables again.
    if (enum_var) locals.erase(enum_var);
    for (auto v : stmt->vars()) locals.erase(v);

    // Emit increments for all of them.
    args.clear();
    if (enum_var) {
        args.push_back(create<arith::AddIOp>( //
            floc,
            bb_cond->getArgument(0),
            CreateInt(floc, 1, enum_var->type)
        ));
    }

    for (auto [expr, a] : zip(stmt->ranges(), block_args)) {
        if (isa<RangeType>(expr->type)) {
            args.push_back(create<arith::AddIOp>(floc, a, CreateInt(floc, 1, a.getType())));
        } else if (auto arr = dyn_cast<ArrayType>(expr->type)) {
            args.push_back(CreatePtrAdd(floc, a, arr->elem()->array_size(tu)));
        } else {
            Unreachable("Invalid for range type: {}", expr->type);
        }
    }

    // Continue.
    create<mlir::cf::BranchOp>(floc, bb_cond, args);
    EnterBlock(std::move(bb_end));
    return nullptr;
}

auto CodeGen::EmitIfExpr(IfExpr* stmt) -> Value {
    auto args = If(
        Emit(stmt->cond),
        [&] { return Emit(stmt->then); },
        stmt->else_ ? [&] { return Emit(stmt->else_.get()); } : llvm::function_ref<Value()>{}
    );
    return args and args->getNumArguments() ? args->getArgument(0) : nullptr;
}

auto CodeGen::EmitIfExpr(IfExpr* stmt, Value mrvalue_slot) -> Value {
    (void) If(
        Emit(stmt->cond),
        [&] { EmitMRValue(mrvalue_slot, cast<Expr>(stmt->then));  return nullptr; },
        [&] { EmitMRValue(mrvalue_slot, cast<Expr>(stmt->else_.get())); return nullptr; }
    );
    return nullptr;
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> Value {
    return CreateInt(C(expr->location()), expr->storage.value(), expr->type);
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    if (LocalNeedsAlloca(decl)) locals[decl] = CreateAlloca(decl->location(), decl->type);
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> Value {
    if (IsZeroSizedType(expr->type)) return nullptr;
    auto l = locals.find(expr->decl);
    if (l != locals.end()) return l->second;

    Assert(bool(lang_opts.constant_eval), "Invalid local ref outside of constant evaluation?");
    auto loc = C(expr->location());
    CreateAbort(
        expr->location(),
        ir::AbortReason::InvalidLocalRef,
        CreateGlobalStringSlice(expr->location(), expr->decl->name),
        CreateNil(loc, tu.StrLitTy)
    );

    return create<mlir::LLVM::PoisonOp>(
        loc,
        expr->value_category == Expr::SRValue
            ? C(expr->type)
            : ptr_ty
    );
}

auto CodeGen::EmitLoopExpr(LoopExpr* stmt) -> Value {
    Loop([&] { if (auto b = stmt->body.get_or_null()) Emit(b); });
    return nullptr;
}

auto CodeGen::EmitMemberAccessExpr(MemberAccessExpr* expr) -> Value {
    auto base = Emit(expr->base);
    if (IsZeroSizedType(expr->type)) return base;
    return CreatePtrAdd(C(expr->location()), base, expr->field->offset);
}

auto CodeGen::EmitOverloadSetExpr(OverloadSetExpr*) -> Value {
    Unreachable("Emitting unresolved overload set?");
}

void CodeGen::EmitProcedure(ProcDecl* proc) {
    locals.clear();
    abort_info_slot = {};

    // Create the procedure.
    curr_proc = DeclareProcedure(proc);

    // If it doesn’t have a body, then we’re done.
    if (not proc->body()) return;

    // Create the entry block.
    EnterProcedure _(*this, curr_proc);

    // Emit locals.
    u32 arg_idx = 0;
    for (auto l : proc->locals) {
        if (IsZeroSizedType(l->type)) continue;
        auto p = dyn_cast<ParamDecl>(l);
        if (not p) {
            EmitLocal(l);
            continue;
        }

        if (
            (p->type->is_srvalue() and p->intent() == Intent::In) or
            p->type->pass_by_lvalue(proc->cconv(), p->intent())
        ) {
            locals[p] = curr_proc.getArgument(arg_idx++);
            continue;
        }

        auto v = locals[p] = CreateAlloca(p->location(), p->type);
        create<ir::StoreOp>(
            C(p->location()),
            v,
            curr_proc.getArgument(arg_idx++),
            p->type->align(tu)
        );
    }

    // Emit the body.
    Emit(proc->body().get());
}

auto CodeGen::EmitProcRefExpr(ProcRefExpr* expr) -> Value {
    auto op = DeclareProcedure(expr->decl);
    auto l = C(expr->location());
    return create<ir::TupleOp>(
        l,
        Vals{
            create<ir::ProcRefOp>(l, op),
            CreateNullPointer(l),
        }
    );
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> Value {
    if (curr_proc.getHasIndirectReturn()) {
        EmitInitialiser(create<ir::ReturnPointerOp>(getUnknownLoc()), expr->value.get());
        create<ir::RetOp>(C(expr->location()), Value());
        return nullptr;
    }

    auto val = expr->value.get_or_null();
    auto ret_val = val ? Emit(val) : nullptr;
    if (val and IsZeroSizedType(val->type)) ret_val = nullptr;
    create<ir::RetOp>(C(expr->location()), ret_val);
    return {};
}

auto CodeGen::EmitStrLitExpr(StrLitExpr* expr) -> Value {
    return CreateGlobalStringSlice(expr->location(), expr->value);
}

auto CodeGen::EmitStructInitExpr(StructInitExpr* e) -> Value {
    if (IsZeroSizedType(e->type)) {
        for (auto v : e->values()) Emit(v);
        return nullptr;
    }

    Unreachable("Emitting struct initialiser without memory location?");
}

auto CodeGen::EmitTypeExpr(TypeExpr* expr) -> Value {
    // These should only exist at compile time.
    ICE(expr->location(), "Can’t emit type expr");
    return nullptr;
}

auto CodeGen::EmitUnaryExpr(UnaryExpr* expr) -> Value {
    if (expr->postfix) {
        switch (expr->op) {
            default: Todo("Emit postfix '{}'", expr->op);
            case Tk::PlusPlus:
            case Tk::MinusMinus: {
                auto ptr = Emit(expr->arg);
                auto l = C(expr->location());
                auto a = expr->type->align(tu);
                auto val = create<ir::LoadOp>(ptr.getLoc(), C(expr->type), ptr, a);
                auto new_val = EmitArithmeticOrComparisonOperator(
                    expr->op == Tk::PlusPlus ? Tk::Plus : Tk::Minus,
                    expr->type,
                    val,
                    CreateInt(l, 1, expr->type),
                    expr->location()
                );

                create<ir::StoreOp>(l, ptr, new_val, a);
                return val;
            }
        }
    }

    switch (expr->op) {
        default: Todo("Emit prefix '{}'", expr->op);

        // These are both no-ops at the IR level and only affect the type
        // of the expression.
        case Tk::Ampersand:
        case Tk::Caret:
            return Emit(expr->arg);

        // Negate an integer.
        case Tk::Minus: {
            // Because of how literals are parsed, we can get into an annoying
            // corner case with this operator: e.g. if the user declares an i64
            // and attempts to initialise it with -9223372036854775808, we overflow
            // because the value 9223372036854775808 is not a valid signed integer,
            // even though -9223372036854775808 *is* valid. Be nice and special-case
            // this here.
            if (
                auto val = dyn_cast<IntLitExpr>(expr->arg);
                val and val->storage.value() - 1 == APInt::getSignedMaxValue(u32(val->type->size(tu).bits()))
            ) {
                auto copy = val->storage.value();
                copy.negate();
                return CreateInt(C(expr->location()), copy, expr->type);
            }

            // Otherwise, emit '0 - val'.
            auto a = Emit(expr->arg);
            return CreateBinop<arith::SubIOp, ir::SSubOvOp>(
                expr->arg->type,
                CreateInt(C(expr->location()), 0, expr->type),
                a,
                expr->location(),
                expr->op
            );
        }
    }
}

auto CodeGen::EmitWhileStmt(WhileStmt* stmt) -> Value {
    While(
        [&] { return Emit(stmt->cond); },
        [&] { Emit(stmt->body); }
    );

    return nullptr;
}

auto CodeGen::EmitValue(Location loc, const eval::RValue& val) -> Value { // clang-format off
    utils::Overloaded V {
        [&](bool b) -> Value { return CreateBool(C(loc), b); },
        [&](std::monostate) -> Value { return nullptr; },
        [&](Type) -> Value { Unreachable("Cannot emit type constant"); },
        [&](const APInt& value) -> Value { return CreateInt(C(loc), value, val.type()); },
        [&](eval::MRValue) -> Value { return nullptr; }, // This only happens if the value is unused.
        [&](this auto& self, const eval::Range& range) -> Value {
            return CreateAggregate(C(loc), self(range.start), self(range.end));
        }
    }; // clang-format on
    return val.visit(V);
}

auto CodeGen::emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::ProcOp {
    Assert(bool(lang_opts.constant_eval));

    // Delete any remnants of the last constant evaluation.
    if (vm_entry_point) vm_entry_point.erase();

    // Build a procedure for this statement.
    setInsertionPointToEnd(&mlir_module.getBodyRegion().front());
    auto ty = stmt->type_or_void();
    auto [ftype, has_indirect_return] = ConvertProcType(ProcType::Get(tu, ty));
    vm_entry_point = create<ir::ProcOp>(
        C(stmt->location()),
        constants::VMEntryPointName,
        C(Linkage::Internal),
        C(CallingConvention::Native),
        ftype,
        false,
        has_indirect_return,
        false
    );

    // Sema has already ensured that this is an initialiser, so throw it
    // into a return expression to handle MRValues.
    ReturnExpr re{dyn_cast<Expr>(stmt), stmt->location(), true};
    EnterProcedure _(*this, vm_entry_point);
    Emit(isa<Expr>(stmt) ? &re : stmt);
    if (not getInsertionBlock()->mightHaveTerminator()) create<ir::RetOp>(getUnknownLoc(), Value());
    return vm_entry_point;
}
