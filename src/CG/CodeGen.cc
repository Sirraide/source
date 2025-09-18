#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>
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
using Ty = mlir::Type;
namespace LLVM = mlir::LLVM;
using LLVM::LLVMDialect;

CodeGen::CodeGen(TranslationUnit& tu, LangOpts lang_opts, Size word_size)
    : CodeGenBase(),
      OpBuilder(&mlir),
      tu{tu},
      word_size{word_size},
      lang_opts{lang_opts} {
    if (libassert::is_debugger_present()) mlir.disableMultithreading();
    mlir.getDiagEngine().registerHandler([&](mlir::Diagnostic& diag) {
        HandleMLIRDiagnostic(diag);
    });

    Assert(+tu.target().triple().getArch() == +llvm::Triple::x86_64);
    Assert(tu.target().triple().isOSLinux());

    ir::SRCCDialect::InitialiseContext(mlir);
    ptr_ty = LLVM::LLVMPointerType::get(&mlir);
    bool_ty = getI1Type();
    int_ty = C(Type::IntTy);
    i128_ty = getIntegerType(128);
    mlir_module = mlir::ModuleOp::create(C(tu.initialiser_proc->location()), tu.name.value());

    // Declare the abort handlers.
    // FIXME: These should instead be declared as needed by the LLVM lowering pass.
    if (tu.abort_info_type) {
        auto ptr = PtrType::Get(tu, tu.abort_info_type);
        auto ty = ProcType::Get(tu, Type::VoidTy, {ParamTypeData{Intent::In, ptr}});
        GetOrCreateProc({}, constants::AssertFailureHandlerName, Linkage::Imported, ty, false);
        GetOrCreateProc({}, constants::ArithmeticFailureHandlerName, Linkage::Imported, ty, false);
    }
}

// ============================================================================
//  AST -> IR
// ============================================================================
auto CodeGen::C(CallingConvention l) -> LLVM::CConv {
    switch (l) {
        case CallingConvention::Source: return LLVM::CConv::Fast;
        case CallingConvention::Native: return LLVM::CConv::C;
    }
    Unreachable();
}

auto CodeGen::C(Linkage l) -> LLVM::Linkage {
    switch (l) {
        case Linkage::Internal: return LLVM::Linkage::Private;
        case Linkage::Exported: return LLVM::Linkage::External;
        case Linkage::Imported: return LLVM::Linkage::External;
        case Linkage::Reexported: return LLVM::Linkage::External;
        case Linkage::Merge: return LLVM::Linkage::LinkonceODR;
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

auto CodeGen::C(Type ty, ValueCategory vc) -> mlir::Type {
    if (vc == Expr::LValue) return ptr_ty;

    // Integer types.
    if (ty == Type::BoolTy) return IntTy(Size::Bits(1));
    if (ty == Type::IntTy) return IntTy(tu.target().int_size());
    if (auto i = dyn_cast<IntType>(ty)) return IntTy(i->bit_width());

    // Pointer types.
    if (isa<PtrType>(ty)) return ptr_ty;

    // For aggregates, call ConvertAggregateToLLVMArray() instead.
    Unreachable("C() does not support aggregate type: '{}'", ty);
}

auto CodeGen::ConvertToByteArrayType(Type ty) -> mlir::Type {
    Assert(not IsZeroSizedType(ty));
    return LLVM::LLVMArrayType::get(&mlir, getI8Type(), tu.target().preferred_size(ty).bytes());
}

auto CodeGen::ConvertProcType(ProcType* ty, bool needs_environment) -> ABICallInfo {
    return LowerProcedureSignature(
        getUnknownLoc(),
        ty,
        needs_environment,
        nullptr,
        nullptr,
        {}
    );
}

void CodeGen::CreateAbort(mlir::Location loc, ir::AbortReason reason, IRValue msg1, IRValue msg2) {
    // The runtime defines the assertion payload as follows:
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
        Warn(Location::Decode(loc), "No declaration of '__src_abort_info' found");
        create<LLVM::UnreachableOp>(loc);
        return;
    }

    // Reuse the abort info slot if there is one.
    if (!abort_info_slot)
        abort_info_slot = CreateAlloca(loc, tu.abort_info_type);

    // Get the file name, line, and column.
    //
    // Don’t require a valid location here as this is also called from within
    // implicitly generated code.
    auto l = Location::Decode(loc);
    StructInitHelper init{*this, tu.abort_info_type, abort_info_slot};
    if (auto lc = l.seek_line_column(context())) {
        auto name = context().file_name(l.file_id);
        init.emit_next_field(CreateGlobalStringSlice(loc, name));
        init.emit_next_field(CreateInt(loc, i64(lc->line)));
        init.emit_next_field(CreateInt(loc, i64(lc->col)));
    } else {
        init.emit_next_field(CreateEmptySlice(loc));
        init.emit_next_field(CreateInt(loc, 0));
        init.emit_next_field(CreateInt(loc, 0));
    }

    // Add the message parts.
    init.emit_next_field(msg1);
    init.emit_next_field(msg2);
    create<ir::AbortOp>(loc, reason, abort_info_slot);
}

auto CodeGen::CreateAlloca(mlir::Location loc, Type ty) -> Value {
    Assert(not IsZeroSizedType(ty));
    return CreateAlloca(loc, tu.target().preferred_size(ty), tu.target().preferred_align(ty));
}

auto CodeGen::CreateAlloca(mlir::Location loc, Size sz, Align a) -> Value {
    InsertionGuard _{*this};

    // Do this stupid garbage because allowing a separate region in a function for
    // frame slots is apparently too much to ask because MLIR complains about both
    // uses not being dominated by the frame slots and about a FunctionOpInterface
    // having more than one region.
    auto it = curr_proc.front().begin();
    auto end = curr_proc.front().end();
    while (it != end and isa<ir::FrameSlotOp>(*it)) ++it;
    if (it == end) setInsertionPointToEnd(&curr_proc.front());
    else setInsertionPoint(&*it);
    return create<ir::FrameSlotOp>(loc, sz, a);
}

void CodeGen::CreateArithFailure(Value failure_cond, Tk op, mlir::Location loc, String name) {
    If(loc, failure_cond, [&] {
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
    mlir::Location loc,
    Tk op
) -> Value {
    if (not lang_opts.overflow_checking) return createOrFold<Unchecked>(loc, lhs, rhs);
    SmallVector<Value> values;
    createOrFold<Checked>(values, loc, lhs, rhs);
    CreateArithFailure(values[1], op, loc);
    return values[0];
}

auto CodeGen::CreateBlock(ArrayRef<Ty> args) -> std::unique_ptr<Block> {
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

void CodeGen::CreateBuiltinAggregateStore(
    mlir::Location loc,
    Value addr,
    Type ty,
    IRValue aggregate
) {
    auto eqv = GetEquivalentStructTypeForAggregate(ty);
    auto f1 = eqv->fields()[0];
    auto f2 = eqv->fields()[1];
    CreateStore(loc, addr, aggregate.first(), f1->type->align(tu), f1->offset);
    CreateStore(loc, addr, aggregate.second(), f2->type->align(tu), f2->offset);
}

auto CodeGen::CreateEmptySlice(mlir::Location loc) -> IRValue {
    return IRValue{CreateNullPointer(loc), CreateInt(loc, 0, int_ty)};
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
        i = create<LLVM::GlobalOp>(
            getUnknownLoc(),
            LLVM::LLVMArrayType::get(getI8Type(), data.size() + null_terminated),
            true,
            LLVM::Linkage::Private,
            std::format("__srcc_str.{}", strings++),
            getStringAttr(llvm::Twine(StringRef(data)) + (null_terminated ? "\0"sv : "")),
            align.value().bytes()
        );
    }
    return create<LLVM::AddressOfOp>(getUnknownLoc(), i);
}

auto CodeGen::CreateGlobalStringSlice(mlir::Location loc, String data) -> IRValue {
    return IRValue{CreateGlobalStringPtr(data), CreateInt(loc, i64(data.size()))};
}

auto CodeGen::CreateICmp(mlir::Location loc, arith::CmpIPredicate pred, Value lhs, Value rhs) -> Value {
    Assert(not isa<LLVM::LLVMPointerType>(lhs.getType()), "Cannot compare pointers this way");

    // But we prefer to emit an arith op because it can be folded etc.
    return createOrFold<arith::CmpIOp>(loc, pred, lhs, rhs);
}

auto CodeGen::CreateInt(mlir::Location loc, const APInt& value, Type ty) -> Value {
    return create<arith::ConstantOp>(loc, getIntegerAttr(C(ty), value));
}

auto CodeGen::CreateInt(mlir::Location loc, i64 value, Type ty) -> Value {
    return CreateInt(loc, value, C(ty));
}

auto CodeGen::CreateInt(mlir::Location loc, i64 value, Ty ty) -> Value {
    return create<arith::ConstantOp>(loc, getIntegerAttr(ty, value));
}

auto CodeGen::CreateLoad(mlir::Location loc, Value addr, Type ty, Size offset) -> IRValue {
    if (auto eqv = GetEquivalentStructTypeForAggregate(ty)) {
        auto f1 = eqv->fields()[0];
        auto f2 = eqv->fields()[1];
        auto v1 = CreateLoad(loc, addr, C(f1->type), f1->type->align(tu), f1->offset + offset);
        auto v2 = CreateLoad(loc, addr, C(f2->type), f2->type->align(tu), f2->offset + offset);
        return {v1, v2};
    }

    return CreateLoad(loc, addr, C(ty), ty->align(tu), offset);
}

auto CodeGen::CreateLoad(
    mlir::Location loc,
    Value addr,
    mlir::Type type,
    Align align,
    Size offset
) -> Value {
    Assert(isa<LLVM::LLVMPointerType>(addr.getType()), "Address of load must be a pointer");

    // Adjust weird integers to a more proper size before loading them and truncate the
    // result afterwards.
    if (type.isInteger()) {
        auto pref_ty = GetPreferredIntType(type);
        if (type != pref_ty) {
            auto pref_val = create<ir::LoadOp>(loc, pref_ty, CreatePtrAdd(loc, addr, offset), align);
            return create<arith::TruncIOp>(loc, type, pref_val);
        }
    }

    return create<ir::LoadOp>(loc, type, CreatePtrAdd(loc, addr, offset), align);
}

void CodeGen::CreateMemCpy(mlir::Location loc, Value to, Value from, Type ty) {
    // For integer types and pointers, emit a load-store pair.
    if (ty->is_integer_or_bool() or isa<PtrType>(ty)) {
        auto a = ty->align(tu);
        auto v = CreateLoad(loc, from, C(ty), a);
        CreateStore(loc, to, v, a);
        return;
    }

    // For everything else, emit a memcpy.
    create<LLVM::MemcpyOp>(
        loc,
        to,
        from,
        CreateInt(loc, i64(ty->memory_size(tu).bytes())),
        false
    );
}

auto CodeGen::CreateNullPointer(mlir::Location loc) -> Value {
    return create<ir::NilOp>(loc, ptr_ty);
}

auto CodeGen::CreatePtrAdd(mlir::Location loc, Value addr, Size offs) -> Value {
    if (offs == Size()) return addr;
    return createOrFold<LLVM::GEPOp>(
        loc,
        ptr_ty,
        getI8Type(),
        addr,
        LLVM::GEPArg(i32(offs.bytes())),
        LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nusw | LLVM::GEPNoWrapFlags::nuw
    );
}

auto CodeGen::CreatePtrAdd(mlir::Location loc, Value addr, Value offs) -> Value {
    return createOrFold<LLVM::GEPOp>(
        loc,
        ptr_ty,
        getI8Type(),
        addr,
        LLVM::GEPArg(offs),
        LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nusw | LLVM::GEPNoWrapFlags::nuw
    );
}

auto CodeGen::CreateSICast(mlir::Location loc, Value val, Type from, Type to) -> Value {
    auto from_sz = from->bit_width(tu);
    auto to_sz = to->bit_width(tu);
    if (from_sz == to_sz) return val;
    if (from_sz > to_sz) return createOrFold<arith::TruncIOp>(loc, C(to), val);
    if (from == Type::BoolTy) return createOrFold<arith::ExtUIOp>(loc, C(to), val);
    return createOrFold<arith::ExtSIOp>(loc, C(to), val);
}

void CodeGen::CreateStore(mlir::Location loc, Value addr, Value val, Align align, Size offset) {
    Assert(isa<LLVM::LLVMPointerType>(addr.getType()), "Address of store must be a pointer");

    // Sign-extend weird integers to a more proper size before storing them.
    if (val.getType().isInteger()) {
        auto pref_ty = GetPreferredIntType(val.getType());
        if (val.getType() != pref_ty) val = createOrFold<arith::ExtSIOp>(loc, pref_ty, val);
    }

    create<ir::StoreOp>(loc, CreatePtrAdd(loc, addr, offset), val, align);
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
            ),
            false
        );
    }

    return printf.value();
}

auto CodeGen::DeclareProcedure(ProcDecl* proc) -> ir::ProcOp {
    auto& ir_proc = declared_procs[proc];
    if (not ir_proc) {
        ir_proc = GetOrCreateProc(
            proc->location(),
            MangledName(proc),
            proc->linkage,
            proc->proc_type(),
            proc->has_captures
        );

        proc_reverse_lookup[ir_proc] = proc;
    }
    return ir_proc;
}

auto CodeGen::EnterBlock(std::unique_ptr<Block> bb, Vals args) -> Block* {
    auto b = bb.release();
    curr_proc.push_back(b);
    return EnterBlock(b, args);
}

auto CodeGen::EnterBlock(Block* bb, Vals args) -> Block* {
    // If we’re in a procedure, there is a current block, and it is not
    // closed, branch to the newly inserted block, unless that block is
    // the function’s entry block.
    auto curr = getInsertionBlock();
    if (
        curr and
        not HasTerminator() and
        isa_and_present<ir::ProcOp>(curr->getParentOp())
    ) create<mlir::cf::BranchOp>(getUnknownLoc(), bb, args);

    // Finally, position the builder at the end of the block.
    setInsertionPointToEnd(bb);
    return bb;
}

CodeGen::EnterProcedure::EnterProcedure(CodeGen& CG, ir::ProcOp proc, ProcDecl* decl)
    : CG(CG), old_func(CG.curr_proc), old_proc(decl), guard{CG} {
    CG.curr_proc = proc;
    CG.curr_proc_decl = old_proc;
    CG.EnterBlock(proc.getOrCreateEntryBlock());
}

auto CodeGen::GetAddressOfLocal(LocalDecl* decl, Location location) -> Value {
    if (IsZeroSizedType(decl->type)) return {};

    // First, try to find the variable in the current procedure. We
    // also cache capture lookups here.
    auto l = locals.find(decl);
    if (l != locals.end()) return l->second;

    // If this is a captured variable (and we’re actually in a nested
    // procedure), we need to load its address via the static chain.
    //
    // Note: It’s possible for there to be no current proc *decl*, if
    // we’re performing constant evaluation in the middle of a nested
    // procedure.
    if (decl->captured and curr_proc_decl and decl->parent != curr_proc_decl) {
        auto env = GetStaticChainPointer(decl->parent, location);
        auto vars = decl->parent->captured_vars();
        auto it = rgs::find(vars, decl);
        Assert(it != vars.end());
        auto extra_ptr = decl->parent->has_captures; // See EmitProcRefExpr().
        auto idx = u32(rgs::distance(vars.begin(), it)) + extra_ptr;
        auto ptr = CreateLoad(
            C(location),
            env,
            ptr_ty,
            tu.target().ptr_align(),
            tu.target().ptr_size() * idx
        );

        locals[decl] = ptr;
        return ptr;
    }

    // This can fail, but only if we’re performing constant evaluation,
    // e.g. if the user writes `int x; eval x;`.
    Assert(bool(lang_opts.constant_eval), "Invalid local ref outside of constant evaluation?");
    auto loc = C(location);
    CreateAbort(
        loc,
        ir::AbortReason::InvalidLocalRef,
        CreateGlobalStringSlice(loc, decl->name.str()),
        CreateEmptySlice(loc)
    );

    EnterBlock(CreateBlock());

    // Just return a null pointer so we don’t crash.
    // TODO: Should we instead just check if we have an insert point, like, *everywhere*?
    return CreateNullPointer(getUnknownLoc());
}

auto CodeGen::GetEquivalentStructTypeForAggregate(Type ty) -> StructType* {
    if (isa<ProcType>(ty)) return tu.ClosureEquivalentStructTy;
    if (isa<SliceType>(ty)) return tu.SliceEquivalentStructTy;
    if (auto r = dyn_cast<RangeType>(ty)) return r->equivalent_struct_type();
    return nullptr;
}

auto CodeGen::GetEvalMode(Type ty) -> EvalMode {
    switch (ty->kind()) {
        case TypeBase::Kind::BuiltinType:
        case TypeBase::Kind::IntType:
        case TypeBase::Kind::ProcType:
        case TypeBase::Kind::PtrType:
        case TypeBase::Kind::SliceType:
            return EvalMode::Scalar;

        // TODO: Ranges are weird in that both eval modes make sense: memory
        // for calls and scalar for casts and for how they’re created; maybe
        // this warrants a separate eval mode (like Clang’s complex eval mode)?
        case TypeBase::Kind::RangeType:
            return EvalMode::Scalar;

        case TypeBase::Kind::ArrayType:
        case TypeBase::Kind::StructType:
            return EvalMode::Memory;
    }

    Unreachable();
}

auto CodeGen::GetEnvPtr() -> Value {
    Assert(curr_proc_decl and curr_proc_decl->has_captures);
    return curr_proc.getArguments().back();
}

auto CodeGen::GetOrCreateProc(
    Location loc,
    String name,
    Linkage linkage,
    ProcType* ty,
    bool needs_environment
) -> ir::ProcOp {
    if (auto op = mlir_module.lookupSymbol<ir::ProcOp>(name)) return op;
    InsertionGuard _{*this};
    setInsertionPointToEnd(&mlir_module.getBodyRegion().front());
    auto info = ConvertProcType(ty, needs_environment);
    auto ir_proc = create<ir::ProcOp>(
        C(loc),
        name,
        C(linkage),
        C(ty->cconv()),
        info.func,
        ty->variadic(),
        info.no_return,
        mlir::ArrayAttr::get(&mlir, info.arg_attrs),
        mlir::ArrayAttr::get(&mlir, info.result_attrs)
    );

    // Erase the body for now; additionally, declarations can’t have public
    // visibility, so set it to private in that case.
    ir_proc.eraseBody();
    ir_proc.setVisibility(mlir::SymbolTable::Visibility::Private);
    return ir_proc;
}

auto CodeGen::GetStaticChainPointer(ProcDecl* proc, Location location) -> Value {
    // Walk up the static chain until we get the procedure whose parent
    // actually contains the variable. Procedures that don’t actually
    // introduce any new captures just reuse the parent’s environment.
    auto env = GetEnvPtr();
    auto loc = C(location);

    // Start at our parent; we don’t want to insert an extra load if
    // the current procedure introduces captures since we already have
    // our own environment pointer.
    auto p = curr_proc_decl->parent.get_or_null();
    while (p and p != proc) {
        if (p->introduces_captures) env = CreateLoad(loc, env, ptr_ty, tu.target().ptr_align());
        p = p->parent.get_or_null();
    }

    // TODO: I don’t think this can fail? What could happen though is
    // that we break out of the current constant evaluation context,
    // make sure to test that at some point.
    Assert(p);
    return env;
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

        auto loc = Location::Decode(d.getLocation());
        auto msg = d.str();
        Diag(level, loc, "{}", utils::Escape(msg, false, true));
    };

    EmitDiagnostic(diag);
    for (auto& n : diag.getNotes())
        EmitDiagnostic(n);
}

bool CodeGen::HasTerminator() {
    auto curr = getInsertionBlock();
    return curr and curr->mightHaveTerminator();
}

auto CodeGen::IntTy(Size wd) -> mlir::Type {
    return getIntegerType(unsigned(wd.bits()));
}

auto CodeGen::If(
    mlir::Location loc,
    Value cond,
    llvm::function_ref<IRValue()> emit_then,
    llvm::function_ref<IRValue()> emit_else
) -> Block* {
    if (not emit_else) {
        If(loc, cond, emit_then);
        return nullptr;
    }

    auto bb_then = CreateBlock();
    auto bb_else = CreateBlock();
    auto bb_join = CreateBlock();
    create<mlir::cf::CondBranchOp>(loc, cond, bb_then.get(), bb_else.get());

    // Emit the then block.
    EnterBlock(std::move(bb_then));
    auto then_val = emit_then();

    // Branch to the join block.
    then_val.each([&](Value v){ bb_join->addArgument(v.getType(), loc); });
    if (not HasTerminator()) create<mlir::cf::BranchOp>(
        loc,
        bb_join.get(),
        then_val ? Vals(then_val) : Vals()
    );

    // And emit the else block.
    EnterBlock(std::move(bb_else));
    auto else_val = emit_else();
    if (not HasTerminator()) create<mlir::cf::BranchOp>(
        loc,
        bb_join.get(),
        else_val ? Vals(else_val) : Vals()
    );

    // Finally, return the join block.
    return EnterBlock(std::move(bb_join));
}

auto CodeGen::If(mlir::Location loc, Value cond, Vals args, llvm::function_ref<void()> emit_then) -> Block* {
    SmallVector<Ty, 3> types;
    for (auto arg : args) types.push_back(arg.getType());

    // Create the blocks and branch to the body.
    auto body = CreateBlock(types);
    auto join = CreateBlock();
    create<mlir::cf::CondBranchOp>(
        loc,
        cond,
        body.get(),
        args,
        join.get(),
        Vals()
    );

    // Emit the body and close the block.
    EnterBlock(std::move(body));
    emit_then();
    if (not HasTerminator()) create<mlir::cf::BranchOp>(
        loc,
        join.get()
    );

    // Add the join block.
    return EnterBlock(std::move(join));
}

bool CodeGen::IsZeroSizedType(Type ty) {
    return ty->memory_size(tu) == Size();
}

bool CodeGen::LocalNeedsAlloca(LocalDecl* local) {
    Assert(not isa<ParamDecl>(local), "Should not be used for parameters");
    if (IsZeroSizedType(local->type)) return false;
    if (local->category == Expr::RValue) return false;
    return true;
}

void CodeGen::Loop(llvm::function_ref<void()> emit_body) {
    auto bb_cond = EnterBlock(CreateBlock());
    emit_body();
    EnterBlock(bb_cond);
}

auto CodeGen::EmitToMemory(mlir::Location l, Expr* init) -> Value {
    if (init->is_lvalue()) return EmitScalar(init);
    auto temp = CreateAlloca(l, init->type);
    EmitRValue(temp, init);
    return temp;
}

bool CodeGen::PassByReference(Type ty, Intent i) {
    Assert(not IsZeroSizedType(ty));

    // 'inout' and 'out' parameters are always references.
    if (i == Intent::Inout or i == Intent::Out) return true;

    // Large or non-trivially copyable 'in' parameters are references.
    if (i == Intent::In) {
        if (not ty->trivially_copyable()) return true;
        return ty->bit_width(tu) > Size::Bits(128);
    }

    // Move parameters are references only if the type is not trivial
    // (because 'move' is equivalent to 'copy' otherwise); that is, for
    // trivially-copyable types, any modification of the ‘moved’ value
    // must not be reflected in the caller.
    //
    // Specifically, moving for these types is *logically* a copy, that
    // is the ‘moved’ value is not actually considered ‘moved’, and the
    // caller may continue accessing it.
    if (i == Intent::Move) {
        if (not ty->trivially_copyable()) return true;
        return false;
    }

    // Copy parameters are always passed by value; whether this is
    // accomplished by making a copy in the caller and passing a
    // pointer or whether they are passed in registers is up to the
    // target ABI and handled in a separate lowering pass.
    return false;
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
    if (not HasTerminator()) create<mlir::cf::BranchOp>(
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
        for (auto p : proc->parents_top_down()) {
            Append(p->name.str());
            name += "$";
        }

        Append(proc->name.str());
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
    if (proc->mangling == Mangling::None) return proc->name.str();

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
    CG.CreateStore(v.getLoc(), ptr, v, field->type->align(CG.tu));
}

void CodeGen::StructInitHelper::emit_next_field(IRValue v) {
    Assert(i < ty->fields().size());
    auto field = ty->fields()[i++];
    auto ptr = CG.CreatePtrAdd(v.loc(), base, field->offset);
    CG.CreateBuiltinAggregateStore(v.loc(), ptr, field->type, v);
}

void CodeGen::EmitRValue(Value addr, Expr* init) { // clang-format off
    Assert(not IsZeroSizedType(init->type), "Should have been checked before calling this");
    Assert(init->is_rvalue(), "Expected an rvalue");
    Assert(addr, "Emitting rvalue without address?");

    // Check if this is an srvalue.
    if (GetEvalMode(init->type) == EvalMode::Scalar) {
        if (init->type->is_aggregate()) {
            CreateBuiltinAggregateStore(C(init->location()), addr, init->type, Emit(init));
        } else {
            CreateStore(C(init->location()), addr, EmitScalar(init), init->type->align(tu));
        }
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

        // The initialiser might be an lvalue-to-rvalue conversion; this is used to
        // pass trivially-copyable structs by value.
        [&](CastExpr *e) {
            Assert(e->kind == CastExpr::CastKind::LValueToRValue);
            auto loc = C(e->location());
            CreateMemCpy(loc, addr, EmitScalar(e->arg), e->type);
        },

        // If the initialiser is a constant expression, create a global constant for it.
        //
        // Yes, this means that this is basically an lvalue that we’re copying from;
        // the only reason the language treats it as an mrvalue is because it would
        // logically be rather weird to be able to take the address of an evaluated
        // struct literal (but not of other rvalues).
        //
        // CreateUnsafe() is fine here since mrvalues are allocated in the TU.
        [&](ConstExpr* e) {
            auto mrv = e->value->cast<eval::MemoryValue>();
            auto c = CreateGlobalStringPtr(
                init->type->align(tu),
                String::CreateUnsafe(static_cast<char*>(mrv.data()), mrv.size().bytes()),
                false
            );

            auto loc = C(init->location());
            CreateMemCpy(loc, addr, c, init->type);
        },

        // Default initialiser here is a memset to 0.
        [&](DefaultInitExpr* e) {
            auto loc = C(init->location());
            create<LLVM::MemsetOp>(
                loc,
                addr,
                CreateInt(loc, 0, tu.I8Ty),
                CreateInt(loc, i64(e->type->memory_size(tu).bytes())),
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
                EmitRValue(offs, val);
            }
        }
    });
} // clang-format on

// ============================================================================
//  ABI
// ============================================================================
void CodeGen::ABIArg::add_byval(mlir::Type ty) {
    attrs.push_back(mlir::NamedAttribute(
        LLVMDialect::getByValAttrName(),
        mlir::TypeAttr::get(ty)
    ));
}

void CodeGen::ABIArg::add_sext(CodeGen& cg) {
    attrs.push_back(mlir::NamedAttribute(
        LLVMDialect::getSExtAttrName(),
        mlir::UnitAttr::get(cg.mlir_context())
    ));
}

void CodeGen::ABIArg::add_sret(mlir::Type ty) {
    attrs.push_back(mlir::NamedAttribute(
        LLVMDialect::getStructRetAttrName(),
        mlir::TypeAttr::get(ty)
    ));
}

void CodeGen::ABIArg::add_zext(CodeGen& cg) {
    attrs.push_back(mlir::NamedAttribute(
        LLVMDialect::getZExtAttrName(),
        mlir::UnitAttr::get(cg.mlir_context())
    ));
}

auto CodeGen::ABITypeRaisingContext::addr() -> Value {
    if (indirect_ptr) return indirect_ptr;
    indirect_ptr = cg.CreateAlloca(loc, ty);
    return indirect_ptr;
}

void CodeGen::AssertTriple() {
    auto& tt = tu.target().triple();
    Assert(
        tt.isOSLinux() and tt.getArch() == llvm::Triple::x86_64,
        "Unsupported target: {}", tt.str()
    );
}

bool CodeGen::CanUseReturnValueDirectly(Type ty) {
    if (isa<PtrType, SliceType, ProcType>(ty)) return true;
    if (ty == Type::BoolTy) return true;
    if (ty->is_integer()) {
        auto sz = ty->bit_width(tu);
        return sz <= Size::Bits(64) or sz == Size::Bits(128);
    }
    return false;
}

auto CodeGen::GetPreferredIntType(mlir::Type ty) -> mlir::Type {
    Assert(ty.isInteger());
    auto bits = Size::Bits(ty.getIntOrFloatBitWidth());
    auto preferred = tu.target().int_size(bits);
    if (preferred != bits) return getIntegerType(unsigned(preferred.bits()));
    return ty;
}

auto CodeGen::LowerByValArg(ABILoweringContext& ctx, mlir::Location l, Ptr<Expr> arg, Type t) -> ABIArgInfo {
    static constexpr Size Word = Size::Bits(64);
    ABIArgInfo info;
    auto sz = t->bit_width(tu);
    auto AddStackArg = [&] (mlir::Type arg_ty = {}) {
        if (not arg_ty) arg_ty = ConvertToByteArrayType(t);
        info.emplace_back(ptr_ty).add_byval(arg_ty);
        if (auto a = arg.get_or_null()) {
            auto addr = EmitToMemory(l, a);

            // Take care not to modify the original object if we’re passing by value.
            if (a->is_lvalue()) {
                info[0].value = CreateAlloca(l, t);
                CreateMemCpy(l, info[0].value, addr, t);
            } else {
                info[0].value = addr;
            }
        }
    };

    auto PassThrough = [&] {
        ctx.allocate();
        auto ty = C(t);
        info.emplace_back(ty);
        if (auto a = arg.get_or_null()) {
            info[0].value = EmitScalar(a);
            if (a->is_lvalue()) info[0].value = CreateLoad(l, info[0].value, ty, a->type->align(tu));
        }
    };

    // Small aggregates are passed in registers.
    if (t->is_aggregate()) {
        auto LoadWord = [&](Value addr, Size wd) -> Value {
            return CreateLoad(l, addr, IntTy(wd.as_bytes()), Align(wd.as_bytes()));
        };

        // This is passed in a single register. Structs that are this small are never
        // annotated with 'byval' for some reason.
        if (sz <= Word) {
            ctx.allocate();
            auto& a = info.emplace_back(IntTy(sz.as_bytes()));
            if (arg) a.value = LoadWord(EmitToMemory(l, arg.get()), sz);
        }

        // This is passed in two registers.
        else if (sz <= Word * 2 and ctx.allocate(2)) {
            // As an optimisation, pass closures and slices directly.
            if (isa<SliceType, ProcType>(t)) {
                auto ty = GetEquivalentStructTypeForAggregate(t);
                info.emplace_back(C(ty->fields()[0]->type));
                info.emplace_back(C(ty->fields()[1]->type));
                if (auto a = arg.get_or_null()) {
                    auto v = Emit(a);
                    if (a->is_lvalue()) v = CreateLoad(l, v.scalar(), t);
                    info[0].value = v.first();
                    info[1].value = v.second();
                }
            }

            // Other aggregates (including ranges) are more complex; just load them
            // from memory in chunks.
            else {
                // TODO: This loads padding bytes if the struct is e.g. (i32, i64); do we care?
                info.emplace_back(int_ty);
                info.emplace_back(IntTy(sz - Word));
                if (arg) {
                    auto addr = EmitToMemory(l, arg.get());
                    info[0].value = LoadWord(addr, Word);
                    info[1].value = LoadWord(CreatePtrAdd(l, addr, Word), sz - Word);
                }
            }
        }

        // This is passed on the stack.
        else { AddStackArg(); }
    }

    // For integers, it depends on the bit width.
    else if (t->is_integer_or_bool()) {
        // i65-i127 are passed in two registers.
        if (sz > Word and sz < Word * 2) {
            if (ctx.allocate(2)) {
                info.emplace_back(int_ty);
                info.emplace_back(int_ty);
                if (auto a = arg.get_or_null()) {
                    auto mem = EmitToMemory(l, a);
                    info[0].value = CreateLoad(l, mem, int_ty, t->align(tu));
                    info[1].value = CreateLoad(l, mem, int_ty, t->align(tu), Word);
                }
            } else {
                // For some reason Clang passes e.g. i127 as an i128 rather than
                // as an array of 16 bytes.
                AddStackArg(i128_ty);
            }
        }

        // i128’s ABI is apparently somewhat cursed; it is never marked as 'byval'
        // and is passed as a single value; this specifically applies only to the
        // C __i128 type and *not* _BitInt(128) for some ungodly reason; treat our
        // i128 as the former because it’s more of a builtin type.
        else if (sz == Word * 2) {
            ctx.allocate(2);
            info.emplace_back(i128_ty);
            if (auto a = arg.get_or_null())
                info[0].value = CreateLoad(l, EmitToMemory(l, a), i128_ty, t->align(tu));
        }

        // Any integers that are larger than i128 are passed on the stack.
        else if (sz > Word * 2) {
            AddStackArg();
        }

        // Lastly, any other integers are just passed through; extend them if
        // they don’t have their preferred size.
        else {
            PassThrough();
            auto ty = C(t);
            auto pref_ty = GetPreferredIntType(ty);
            if (ty != pref_ty) {
                if (t == Type::BoolTy) info[0].add_zext(*this);
                else info[0].add_sext(*this);
            }
        }
    }

    // Pointers are just passed through.
    else if (isa<PtrType>(t)) {
        PassThrough();
    }

    // Make sure that we explicitly handle all possible type kinds.
    else {
        ICE(Location::Decode(l), "Unsupported type in call lowering: {}", t);
    }

    return info;
}

auto CodeGen::LowerDirectReturn(mlir::Location l, Expr* arg) -> ABIArgInfo {
    ABILoweringContext ctx;
    return LowerByValArg(ctx, l, arg, arg->type);
}

auto CodeGen::LowerProcedureSignature(
    mlir::Location l,
    ProcType* proc,
    bool needs_environment,
    Value indirect_ptr,
    Value env_ptr,
    ArrayRef<Expr*> args
) -> ABICallInfo {
    static constexpr Size Word = Size::Bits(64);
    ABICallInfo info;
    ABILoweringContext ctx;
    auto AddArgType = [&](mlir::Type t, ArrayRef<mlir::NamedAttribute> attrs = {}) {
        info.arg_types.push_back(t);
        info.arg_attrs.push_back(getDictionaryAttr(attrs));
    };

    auto AddReturnType = [&](mlir::Type t, ArrayRef<mlir::NamedAttribute> attrs = {}) {
        info.result_types.push_back(t);
        info.result_attrs.push_back(getDictionaryAttr(attrs));
    };

    auto AddByRefArg = [&](Value v, Type t) {
        ctx.allocate();
        info.args.push_back(v);
        AddArgType(
            ptr_ty,
            getNamedAttr(
                LLVMDialect::getDereferenceableAttrName(),
                getI64IntegerAttr(i64(t->memory_size(tu).bytes()))
            )
        );
    };

    auto AddByValArg = [&](Expr* arg, Type t) {
        for (const auto& a : LowerByValArg(ctx, l, arg, t)) {
            info.args.push_back(a.value);
            AddArgType(a.ty, a.attrs);
        }
    };

    auto Arg = [&](usz i) -> Expr* {
        if (i < args.size()) return args[i];
        return nullptr;
    };

    auto ret = proc->ret();
    auto sz = ret->bit_width(tu);
    if (IsZeroSizedType(ret)) {
        if (ret == Type::NoReturnTy) info.no_return = true;
    }

    // Some types are returned via a store to a hidden argument pointer.
    else if (NeedsIndirectReturn(ret)) {
        ctx.allocate();
        info.args.push_back(indirect_ptr);
        AddArgType(
            ptr_ty,
            getNamedAttr(
                LLVMDialect::getStructRetAttrName(),
                mlir::TypeAttr::get(ConvertToByteArrayType(ret))
            )
        );
    }

    // Small aggregates are returned in registers.
    else if (ret->is_aggregate()) {
        if (sz <= Word) {
            AddReturnType(IntTy(sz.as_bytes()));
        } else if (sz <= Word * 2) {
            // TODO: This returns padding bytes if the struct is e.g. (i32, i64); do we care?
            AddReturnType(int_ty);
            AddReturnType(IntTy((sz - Word).as_bytes()));
        } else {
            Unreachable("Should never be returned directly");
        }
    }

    // i65–i127 (but *not* i128) are returned in two registers.
    else if (ret->is_integer() and sz > Word and sz < Word * 2) {
        AddReturnType(int_ty);
        AddReturnType(int_ty);
    }

    // Everything else is passed through as is.
    else {
        SmallVector<mlir::NamedAttribute, 1> attrs;

        // Extend integers that don’t have their preferred size.
        auto ty = C(ret);
        if (ty.isInteger()) {
            auto pref_ty = GetPreferredIntType(ty);
            if (ty != pref_ty) {
                if (ret == Type::BoolTy) attrs.push_back(getNamedAttr(LLVMDialect::getZExtAttrName(), getUnitAttr()));
                else attrs.push_back(getNamedAttr(LLVMDialect::getSExtAttrName(), getUnitAttr()));
            }
        }

        AddReturnType(ty, attrs);
    }

    // Evaluate the arguments and add them to the call.
    for (auto [i, param] : enumerate(proc->params())) {
        if (IsZeroSizedType(param.type)) {
            if (auto a = Arg(i)) Emit(a);
        } else if (PassByReference(param.type, param.intent)) {
            Value arg;
            if (auto a = Arg(i)) arg = EmitToMemory(l, a);
            AddByRefArg(arg, param.type);
        } else {
            AddByValArg(Arg(i), param.type);
        }
    }

    // Throw the environment pointer at the end.
    if (needs_environment) {
        AddArgType(ptr_ty, {
            getNamedAttr(LLVMDialect::getNestAttrName(), getUnitAttr()),
            getNamedAttr(LLVMDialect::getReadonlyAttrName(), getUnitAttr()),
            getNamedAttr(LLVMDialect::getNoUndefAttrName(), getUnitAttr()),
            getNamedAttr(LLVMDialect::getNoFreeAttrName(), getUnitAttr()),
        });

        if (env_ptr) info.args.push_back(env_ptr);
    }

    // Extra variadic arguments are always passed as 'copy' parameters.
    if (args.size() > proc->params().size()) {
        for (auto arg : args.drop_front(proc->params().size())) {
            Assert(not IsZeroSizedType(arg->type), "Passing zero-sized type as variadic argument?");
            for (const auto& a : LowerByValArg(ctx, l, arg, arg->type))
                info.args.push_back(a.value);
        }
    }

    info.func = mlir::FunctionType::get(&mlir, info.arg_types, info.result_types);
    return info;
}

bool CodeGen::NeedsIndirectReturn(Type ty) {
    AssertTriple();
    return ty->memory_size(tu) > Size::Bits(128);
}

auto CodeGen::WriteByValParamToMemory(ABITypeRaisingContext& vals) -> Value {
    Assert(not IsZeroSizedType(vals.type()));
    static constexpr Size Word = Size::Bits(64);
    auto sz = vals.type()->bit_width(tu);
    auto ReuseStackAddress = [&] { return vals.next(); };

    // Small aggregates are passed in registers.
    if (vals.type()->is_aggregate()) {
        auto StoreWord = [&](Value addr, Value v) {
            auto a = v.getType() == ptr_ty
                ? tu.target().ptr_align()
                : tu.target().int_align(Size::Bits(v.getType().getIntOrFloatBitWidth()));
            CreateStore(
                vals.location(),
                addr,
                v,
                a
            );
        };

        // This is passed in a single register.
        if (sz <= Word and vals.lowering().allocate()) {
            StoreWord(vals.addr(), vals.next());
            return vals.addr();
        }

        // This is passed in two registers.
        if (sz <= Word * 2 and vals.lowering().allocate(2)) {
            StoreWord(vals.addr(), vals.next());
            StoreWord(CreatePtrAdd(vals.location(), vals.addr(), Word), vals.next());
            return vals.addr();
        }

        return ReuseStackAddress();
    }

    if (vals.type()->is_integer_or_bool()) {
        // i65-i127 are passed in two registers.
        if (sz > Word and sz < Word * 2) {
            if (vals.lowering().allocate(2)) {
                auto first = vals.next();
                auto second = vals.next();
                CreateStore(vals.location(), vals.addr(), first, vals.type()->align(tu));
                CreateStore(vals.location(), vals.addr(), second, vals.type()->align(tu), Word);
                return vals.addr();
            }

            return ReuseStackAddress();
        }

        // i128 is a single register.
        if (sz == Word * 2) {
            vals.lowering().allocate(2);
            CreateStore(vals.location(), vals.addr(), vals.next(), vals.type()->align(tu));
            return vals.addr();
        }

        // Anything larger goes on the stack..
        if (sz > Word * 2) return ReuseStackAddress();
    }

    vals.lowering().allocate();
    CreateStore(vals.location(), vals.addr(), vals.next(), vals.type()->align(tu));
    return vals.addr();
}

auto CodeGen::WriteDirectReturnToMemory(ABITypeRaisingContext& vals) -> Value {
   return WriteByValParamToMemory(vals);
}

// ============================================================================
//  CG
// ============================================================================
void CodeGen::Emit(ArrayRef<ProcDecl*> procs) {
    for (auto& p : procs)
        if (p->body())
            EmitProcedure(p);
}

auto CodeGen::Emit(Stmt* stmt) -> IRValue {
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

auto CodeGen::EmitScalar(Stmt* stmt) -> Value {
    return Emit(stmt).scalar();
}

auto CodeGen::EmitArrayBroadcastExpr(ArrayBroadcastExpr*) -> IRValue {
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
        C(loc)
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
    EmitRValue(ptr, initialiser);

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

auto CodeGen::EmitArrayInitExpr(ArrayInitExpr*) -> IRValue {
    Unreachable("Should only be emitted as mrvalue");
}

void CodeGen::EmitArrayInitExpr(ArrayInitExpr* e, Value mrvalue_slot) {
    auto ty = cast<ArrayType>(e->type);
    bool broadcast_els = u64(ty->dimension()) - e->initialisers().size();

    // Emit each initialiser.
    for (auto init : e->initialisers()) {
        EmitRValue(mrvalue_slot, init);
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

auto CodeGen::EmitAssertExpr(AssertExpr* expr) -> IRValue {
    auto loc = expr->location().seek_line_column(tu.context());
    if (not loc) {
        ICE(expr->location(), "No location for assert");
        return {};
    }

    Unless(EmitScalar(expr->cond), [&] {
        IRValue msg{};
        auto l = C(expr->location());
        if (auto m = expr->message.get_or_null()) msg = Emit(m);
        else msg = CreateEmptySlice(l);
        auto cond_str = CreateGlobalStringSlice(C(expr->cond->location()), expr->cond->location().text(tu.context()));
        CreateAbort(
            l,
            ir::AbortReason::AssertionFailed,
            cond_str,
            msg
        );
    });

    return {};
}

auto CodeGen::EmitBinaryExpr(BinaryExpr* expr) -> IRValue {
    auto l = C(expr->location());
    switch (expr->op) {
        // Convert 'x and y' to 'if x then y else false'.
        case Tk::And: {
            return If(
                l,
                EmitScalar(expr->lhs),
                [&] { return EmitScalar(expr->rhs); },
                [&] { return CreateBool(l, false); }
            )->getArgument(0);
        }

        // Convert 'x or y' to 'if x then true else y'.
        case Tk::Or: {
            return If(
                l,
                EmitScalar(expr->lhs),
                [&] { return CreateBool(l, true); },
                [&] { return EmitScalar(expr->rhs); }
            )->getArgument(0);
        }

        // Assignment.
        case Tk::Assign: {
            if (IsZeroSizedType(expr->lhs->type)) {
                Emit(expr->lhs);
                Emit(expr->rhs);
                return {};
            }

            auto addr = EmitScalar(expr->lhs);
            EmitRValue(addr, expr->rhs);
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
            auto index = EmitScalar(expr->rhs);
            bool is_slice = isa<SliceType>(expr->lhs->type);

            // Check that the index is in bounds.
            if (lang_opts.overflow_checking) {
                auto size = is_slice ? range.second() : CreateInt(l, cast<ArrayType>(expr->lhs->type)->dimension());
                CreateArithFailure(
                    CreateICmp(l, arith::CmpIPredicate::uge, index, size),
                    Tk::LBrack,
                    l,
                    "out of bounds access"
                );
            }

            Size elem_size = cast<SingleElementTypeBase>(expr->lhs->type)->elem()->array_size(tu);
            return CreatePtrAdd(
                l,
                range.scalar_or_first(),
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
            auto lvalue = EmitScalar(expr->lhs);
            auto lhs = CreateLoad(
                lvalue.getLoc(),
                lvalue,
                C(expr->lhs->type),
                a
            );

            auto rhs = EmitScalar(expr->rhs);
            auto res = EmitArithmeticOrComparisonOperator(
                StripAssignment(expr->op),
                expr->lhs->type,
                lhs,
                rhs,
                C(expr->location())
            );

            CreateStore(
                C(expr->location()),
                lvalue,
                res,
                a
            );
            return lvalue;
        }

        // Range expressions.
        case Tk::DotDotLess: return {EmitScalar(expr->lhs), EmitScalar(expr->rhs)};
        case Tk::DotDotEq: {
            auto loc = C(expr->location());
            auto lhs = EmitScalar(expr->lhs);
            auto rhs = EmitScalar(expr->rhs);
            return {
                lhs,
                createOrFold<arith::AddIOp>(loc, rhs, CreateInt(loc, 1, rhs.getType())),
            };
        }

        // 'in' operator.
        case Tk::In: {
            // Currently, the RHS must be a range.
            Assert(isa<RangeType>(expr->rhs->type));
            auto loc = C(expr->location());
            auto lhs = EmitScalar(expr->lhs);
            auto rhs = Emit(expr->rhs);
            auto start_cmp = CreateICmp(loc, arith::CmpIPredicate::sge, lhs, rhs.first());
            auto end_cmp = CreateICmp(loc, arith::CmpIPredicate::sle, lhs, rhs.second());
            return create<arith::AndIOp>(loc, start_cmp, end_cmp).getResult();
        }

        // Anything else.
        default: {
            auto lhs = EmitScalar(expr->lhs);
            auto rhs = EmitScalar(expr->rhs);
            return EmitArithmeticOrComparisonOperator(
                expr->op,
                expr->lhs->type,
                lhs,
                rhs,
                C(expr->location())
            );
        }
    }
}

auto CodeGen::EmitArithmeticOrComparisonOperator(Tk op, Type type, Value lhs, Value rhs, mlir::Location loc) -> Value {
    using enum OverflowBehaviour;
    auto ty = rhs.getType();
    Assert(
        lhs.getType() == ty,
        "Sema should have converted these to the same type"
    );

    auto CheckDivByZero = [&] {
        if (not lang_opts.overflow_checking) return;
        auto check = CreateICmp(loc, arith::CmpIPredicate::eq, rhs, CreateInt(loc, 0, ty));
        CreateArithFailure(check, op, loc, "division by zero");
    };

    auto CreateCheckedBinop = [&]<typename Unchecked, typename Checked>() -> Value {
        return CreateBinop<Unchecked, Checked>(type, lhs, rhs, loc, op);
    };

    switch (op) {
        default: Todo("Codegen for '{}'", op);

        // 'and' and 'or' require lazy evaluation and are handled elsewhere.
        case Tk::And:
        case Tk::Or:
            Unreachable("'and' and 'or' cannot be handled here.");

        // Comparison operators.
        case Tk::ULt: return CreateICmp(loc, arith::CmpIPredicate::ult, lhs, rhs);
        case Tk::UGt: return CreateICmp(loc, arith::CmpIPredicate::ugt, lhs, rhs);
        case Tk::ULe: return CreateICmp(loc, arith::CmpIPredicate::ule, lhs, rhs);
        case Tk::UGe: return CreateICmp(loc, arith::CmpIPredicate::uge, lhs, rhs);
        case Tk::SLt: return CreateICmp(loc, arith::CmpIPredicate::slt, lhs, rhs);
        case Tk::SGt: return CreateICmp(loc, arith::CmpIPredicate::sgt, lhs, rhs);
        case Tk::SLe: return CreateICmp(loc, arith::CmpIPredicate::sle, lhs, rhs);
        case Tk::SGe: return CreateICmp(loc, arith::CmpIPredicate::sge, lhs, rhs);
        case Tk::EqEq: return CreateICmp(loc, arith::CmpIPredicate::eq, lhs, rhs);
        case Tk::Neq: return CreateICmp(loc, arith::CmpIPredicate::ne, lhs, rhs);

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
            if (lang_opts.overflow_checking) {
                auto int_min = CreateInt(loc, APInt::getSignedMinValue(u32(type->bit_width(tu).bits())), type);
                auto minus_one = CreateInt(loc, -1, ty);
                auto check_lhs = CreateICmp(loc, arith::CmpIPredicate::eq, lhs, int_min);
                auto check_rhs = CreateICmp(loc, arith::CmpIPredicate::eq, rhs, minus_one);
                CreateArithFailure(
                    createOrFold<arith::AndIOp>(loc, check_lhs, check_rhs),
                    op,
                    loc
                );
            }

            return op == Tk::Slash
                ? createOrFold<arith::DivSIOp>(loc, lhs, rhs)
                : createOrFold<arith::RemSIOp>(loc, lhs, rhs);
        }

        // Left shift overflows if the shift amount is equal
        // to or exceeds the bit width.
        case Tk::ShiftLeftLogical: {
            if (lang_opts.overflow_checking) {
                auto check = CreateICmp(loc, arith::CmpIPredicate::uge, rhs, CreateInt(loc, i64(type->bit_width(tu).bits())));
                CreateArithFailure(check, op, loc, "shift amount exceeds bit width");
            }
            return createOrFold<arith::ShLIOp>(loc, lhs, rhs);
        }

        // Signed left shift additionally does not allow a sign change.
        case Tk::ShiftLeft: {
            if (lang_opts.overflow_checking) {
                auto check = CreateICmp(loc, arith::CmpIPredicate::uge, rhs, CreateInt(loc, i64(type->bit_width(tu).bits())));
                CreateArithFailure(check, op, loc, "shift amount exceeds bit width");
            }

            auto res = createOrFold<arith::ShLIOp>(loc, lhs, rhs);

            // Check sign.
            if (lang_opts.overflow_checking) {
                auto sign = createOrFold<arith::ShRSIOp>(loc, lhs, CreateInt(loc, i64(type->bit_width(tu).bits()) - 1));
                auto new_sign = createOrFold<arith::ShRSIOp>(loc, res, CreateInt(loc, i64(type->bit_width(tu).bits()) - 1));
                auto sign_change = CreateICmp(loc, arith::CmpIPredicate::ne, sign, new_sign);
                CreateArithFailure(sign_change, op, loc);
            }

            return res;
        }

        // This is lowered to a call to a compiler-generated function.
        case Tk::StarStar: Unreachable("Sema should have converted this to a call");
    }
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> IRValue {
    return EmitBlockExpr(expr, nullptr);
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr, Value mrvalue_slot) -> IRValue {
    IRValue ret;
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
                EmitRValue(locals.at(var), var->init.get());
            }
        }

        // Emitting any other declarations is a no-op. So is emitting constant
        // expressions that are unused.
        if (isa<ConstExpr, Decl>(s)) continue;

        // This is the expression we need to return from the block.
        if (s == expr->return_expr()) {
            if (mrvalue_slot) EmitRValue(mrvalue_slot, cast<Expr>(s));
            else ret = Emit(s);
        }

        // This is an mrvalue expression that is not the return value; we
        // allow these here, but we need to provide stack space for them.
        else if (GetEvalMode(s->type_or_void()) == EvalMode::Memory) {
            auto e = cast<Expr>(s);
            if (IsZeroSizedType(e->type)) {
                Emit(s);
            } else {
                auto l = CreateAlloca(C(e->location()), e->type);
                EmitRValue(l, e);
            }
        }

        // Otherwise, this is a regular statement or expression.
        else { Emit(s); }
    }
    return ret;
}

auto CodeGen::EmitBoolLitExpr(BoolLitExpr* stmt) -> IRValue {
    return CreateBool(C(stmt->location()), stmt->value);
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> IRValue {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = DeclarePrintf();
            auto ref = create<ir::ProcRefOp>(C(expr->location()), printf);
            for (auto a : expr->args()) {
                auto loc = C(a->location());
                if (a->type == tu.StrLitTy) {
                    Assert(a->value_category == Expr::RValue);
                    auto str_format = CreateGlobalStringPtr("%.*s");
                    auto slice = Emit(a);
                    auto data = slice.first();
                    auto size = CreateSICast(loc, slice.second(), Type::IntTy, tu.FFIIntTy);
                    create<ir::CallOp>(loc, ref, Vals{str_format, size, data});
                }

                else if (a->type == Type::IntTy) {
                    Assert(a->value_category == Expr::RValue);
                    auto int_format = CreateGlobalStringPtr("%" PRId64);
                    auto val = EmitScalar(a);
                    create<ir::CallOp>(loc, ref, Vals{int_format, val});
                }

                else if (a->type == Type::BoolTy) {
                    Assert(a->value_category == Expr::RValue);
                    auto bool_format = CreateGlobalStringPtr("%s");
                    auto val = EmitScalar(a);
                    auto str = createOrFold<arith::SelectOp>(loc, val, CreateGlobalStringPtr("true"), CreateGlobalStringPtr("false"));
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
            return {};
        }

        case BuiltinCallExpr::Builtin::Unreachable: {
            create<LLVM::UnreachableOp>(C(expr->location()));
            EnterBlock(CreateBlock());
            return {};
        }
    }

    Unreachable("Unknown builtin");
}

auto CodeGen::EmitBuiltinMemberAccessExpr(BuiltinMemberAccessExpr* expr) -> IRValue {
    auto l = C(expr->location());
    auto GetField = [&] (unsigned i) -> IRValue {
        if (expr->operand->is_rvalue()) return Emit(expr->operand)[i];
        auto r = GetEquivalentStructTypeForAggregate(expr->operand->type);
        auto f = r->fields()[i];
        auto addr = EmitScalar(expr->operand);
        return CreateLoad(l, addr, f->type, f->offset);
    };

    switch (expr->access_kind) {
        using AK = BuiltinMemberAccessExpr::AccessKind;
        case AK::SliceData: return GetField(0);
        case AK::SliceSize: return GetField(1);
        case AK::RangeStart: return GetField(0);
        case AK::RangeEnd: return GetField(1);
        case AK::TypeAlign: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->align(tu).value().bytes()));
        case AK::TypeArraySize: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->array_size(tu).bytes()));
        case AK::TypeBits: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->bit_width(tu).bits()));
        case AK::TypeBytes: return CreateInt(l, i64(cast<TypeExpr>(expr->operand)->value->memory_size(tu).bytes()));
        case AK::TypeName: return CreateGlobalStringSlice(l, tu.save(StripColours(cast<TypeExpr>(expr->operand)->value->print())));
        case AK::TypeMaxVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(l, APInt::getSignedMaxValue(u32(ty->bit_width(tu).bits())), ty);
        }

        case AK::TypeMinVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(l, APInt::getSignedMinValue(u32(ty->bit_width(tu).bits())), ty);
        }
    }
    Unreachable();
}

auto CodeGen::EmitCallExpr(CallExpr* expr) -> IRValue {
    return EmitCallExpr(expr, nullptr);
}

auto CodeGen::EmitCallExpr(CallExpr* expr, Value mrvalue_slot) -> IRValue {
    auto proc = cast<ProcType>(expr->callee->type);
    auto l = C(expr->location());

    // Callee is evaluated first.
    auto callee = Emit(expr->callee);

    // Make sure we have a slot for the return type.
    auto ret = proc->ret();
    if (NeedsIndirectReturn(ret) and not mrvalue_slot)
        mrvalue_slot = CreateAlloca(l, ret);

    // Try to avoid passing an environment to it if possible.
    bool env_is_nil = callee.is_aggregate() and isa_and_present<ir::NilOp>(callee.second().getDefiningOp());
    bool needs_env = not env_is_nil and (
        not isa<ProcRefExpr>(expr->callee) or
        cast<ProcRefExpr>(expr->callee)->decl->has_captures
    );

    // Emit the arguments.
    auto info = LowerProcedureSignature(
        l,
        proc,
        needs_env,
        mrvalue_slot,
        needs_env ? callee.second() : nullptr,
        expr->args()
    );

    // Build the call.
    auto op = create<ir::CallOp>(
        l,
        info.result_types,
        callee.first(),
        C(proc->cconv()),
        info.func,
        proc->variadic(),
        info.args,
        mlir::ArrayAttr::get(&mlir, info.arg_attrs),
        mlir::ArrayAttr::get(&mlir, info.result_attrs)
    );

    // Calls are one of the very few expressions whose type can be 'noreturn', so
    // take care to handle that here; omitting this would still work, but doing so
    // allows us to throw away a lot of dead code.
    if (ret == Type::NoReturnTy) {
        create<LLVM::UnreachableOp>(C(expr->location()));
        EnterBlock(CreateBlock());
    }

    // Check if this operation doesn’t yield anything; this handles zero-sized and
    // indirect returns.
    if (op.getNumResults() == 0) {
        // However, if we’re returning an integer indirectly, do load it because
        // Sema expects that these are always treated as values, and we currently
        // don’t build the AST differently depending on ABI decisions.
        //
        // TODO: Should we do that though? It’d only really be a problem if we wanted
        // constant evaluation and codegen proper to have different ABIs, which would
        // almost certainly be nonsense.
        if (expr->type->is_integer()) return CreateLoad(
            l,
            mrvalue_slot,
            C(ret),
            ret->align(tu)
        );

        // Regular indirect return.
        return {};
    }

    // Simple return types can be used as-is.
    if (CanUseReturnValueDirectly(ret)) return op.getResults();

    // More complicated ones need to be written to memory.
    ABILoweringContext actx;
    ABITypeRaisingContext ctx{*this, actx, l, op.getResults(), ret, mrvalue_slot};
    auto mem = WriteDirectReturnToMemory(ctx);

    // If this is an srvalue, just load it.
    if (GetEvalMode(expr->type) == EvalMode::Scalar)
        return CreateLoad(l, mem, ret);

    // Sanity check: if this is an mrvalue, we should always construct into
    // the address that was provided to us, if there was one at all. If this
    // is somehow not the case, that indicates that this function should have
    // probably returned this type indirectly.
    Assert(not mrvalue_slot or mem == mrvalue_slot, "Should have been an indirect return");

    // This is an MRValue and shouldn’t yield anything.
    return {};
}

auto CodeGen::EmitCastExpr(CastExpr* expr) -> IRValue {
    auto val = Emit(expr->arg);
    switch (expr->kind) {
        case CastExpr::Deref:
            return val; // This is a no-op like prefix '^'.

        case CastExpr::ExplicitDiscard:
            return {};

        case CastExpr::Integral:
            return CreateSICast(C(expr->location()), val.scalar(), expr->arg->type, expr->type);

        case CastExpr::LValueToRValue: {
            Assert(expr->arg->value_category == Expr::LValue);
            if (IsZeroSizedType(expr->type)) return {};
            return CreateLoad(C(expr->location()), val.scalar(), expr->type);
        }

        case CastExpr::MaterialisePoisonValue: {
            auto op = create<LLVM::PoisonOp>(
                C(expr->location()),
                C(expr->type, expr->value_category)
            );

            return op.getRes();
        }

        case CastExpr::Range: {
            auto l = C(expr->location());
            auto to = cast<RangeType>(expr->type)->elem();
            auto from = cast<RangeType>(expr->arg->type)->elem();
            return {
                CreateSICast(l, val.first(), from, to),
                CreateSICast(l, val.second(), from, to),
            };
        }
    }

    Unreachable();
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> IRValue {
    return EmitValue(constant->location(), *constant->value);
}

auto CodeGen::EmitDefaultInitExpr(DefaultInitExpr* stmt) -> IRValue {
    auto ty = stmt->type;
    auto l = C(stmt->location());

    if (IsZeroSizedType(ty)) return {};
    Assert(GetEvalMode(ty) == EvalMode::Scalar, "Emitting non-srvalue on its own?");

    if (ty->is_integer_or_bool()) return CreateInt(l, 0, C(ty));
    if (isa<PtrType>(ty)) return CreateNullPointer(l);
    if (isa<SliceType>(ty)) return CreateEmptySlice(l);
    if (isa<ProcType>(ty)) return {CreateNullPointer(l), CreateNullPointer(l)};
    if (auto r = dyn_cast<RangeType>(ty)) {
        auto e = C(r->elem());
        return {CreateInt(l, 0, e), CreateInt(l, 0, e)};
    }

    Unreachable("Don’t know how to emit DefaultInitExpr of type '{}'", stmt->type);
}

auto CodeGen::EmitEmptyStmt(EmptyStmt*) -> IRValue {
    return {};
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> IRValue {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitForStmt(ForStmt* stmt) -> IRValue {
    SmallVector<IRValue> ranges;
    SmallVector<Value> args;
    SmallVector<Value> end_vals;
    SmallVector<Ty> arg_types;
    auto bb_end = CreateBlock();
    auto floc = C(stmt->location());

    // Emit the ranges in order.
    for (auto r : stmt->ranges()) {
        Assert((isa<ArrayType, RangeType, SliceType>(r->type)));
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
        if (isa<ArrayType>(expr->type)) {
            arg_types.push_back(ptr_ty);
            args.push_back(r.scalar());
            end_vals.push_back(CreatePtrAdd(r.loc(), r.scalar(), expr->type->memory_size(tu)));
        } else if (isa<RangeType>(expr->type)) {
            arg_types.push_back(r.first().getType());
            args.push_back(r.first());
            end_vals.push_back(r.second());
        } else if (isa<SliceType>(expr->type)) {
            auto byte_size = createOrFold<arith::MulIOp>(
                floc,
                r.second(),
                CreateInt(floc, i64(cast<SingleElementTypeBase>(expr->type)->elem()->array_size(tu).bytes()))
            );

            arg_types.push_back(ptr_ty);
            args.push_back(r.first());
            end_vals.push_back(CreatePtrAdd(r.loc(), r.first(), byte_size));
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
        auto ne =
            isa<LLVM::LLVMPointerType>(a.getType())
                ? create<LLVM::ICmpOp>(floc, LLVM::ICmpPredicate::ne, a, e)
                : CreateICmp(floc, arith::CmpIPredicate::ne, a, e);
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
        } else if (isa<ArrayType, SliceType>(expr->type)) {
            args.push_back(CreatePtrAdd(
                floc,
                a,
                cast<SingleElementTypeBase>(expr->type)->elem()->array_size(tu)
            ));
        } else {
            Unreachable("Invalid for range type: {}", expr->type);
        }
    }

    // Continue.
    create<mlir::cf::BranchOp>(floc, bb_cond, args);
    EnterBlock(std::move(bb_end));
    return {};
}

auto CodeGen::EmitIfExpr(IfExpr* stmt) -> IRValue {
    auto args = If(
        C(stmt->location()),
        EmitScalar(stmt->cond),
        [&] { return Emit(stmt->then); },
        stmt->else_ ? [&] { return Emit(stmt->else_.get()); } : llvm::function_ref<IRValue()>{}
    );

    if (not args) return {};
    return IRValue{args->getArguments()};
}

auto CodeGen::EmitIfExpr(IfExpr* stmt, Value mrvalue_slot) -> IRValue {
    If(
        C(stmt->location()),
        EmitScalar(stmt->cond),
        [&] -> IRValue { EmitRValue(mrvalue_slot, cast<Expr>(stmt->then));  return {}; },
        [&] -> IRValue { EmitRValue(mrvalue_slot, cast<Expr>(stmt->else_.get())); return {}; }
    );
    return {};
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> IRValue {
    return CreateInt(C(expr->location()), expr->storage.value(), expr->type);
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    if (LocalNeedsAlloca(decl)) locals[decl] = CreateAlloca(C(decl->location()), decl->type);
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> IRValue {
    return GetAddressOfLocal(expr->decl, expr->location());
}

auto CodeGen::EmitLoopExpr(LoopExpr* stmt) -> IRValue {
    Loop([&] { if (auto b = stmt->body.get_or_null()) Emit(b); });
    EnterBlock(CreateBlock());
    return {};
}

auto CodeGen::EmitMatchExpr(MatchExpr* expr) -> IRValue {
    Assert(llvm::any_of(expr->cases(), [](auto& c) { return not c.unreachable; }));
    auto join = CreateBlock();
    if (not IsZeroSizedType(expr->type)) {
        join->addArgument(
            C(expr->type, expr->value_category),
            C(expr->location())
        );
    }

    bool has_wildcard = false;
    for (auto [i, c] : enumerate(expr->cases())) {
        if (c.unreachable) continue;
        auto loc = C(c.loc);
        auto EmitVal = [&](Stmt* s) {
            SmallVector<Value, 2> vals;
            Emit(s).into(vals);
            create<mlir::cf::BranchOp>(loc, join.get(), vals);
        };

        if (c.cond.is_wildcard()) {
            has_wildcard = true;
            EmitVal(c.body);
            break;
        }

        If(loc, Emit(c.cond.expr()).scalar(), [&] {
            SmallVector<Value, 2> vals;
            Emit(c.body).into(vals);
            create<mlir::cf::BranchOp>(loc, join.get(), vals);
        });
    }

    if (not has_wildcard) create<LLVM::UnreachableOp>(C(expr->location()));
    IRValue vals{join->getArguments()};
    EnterBlock(std::move(join));
    return vals;
}

auto CodeGen::EmitMaterialiseTemporaryExpr(MaterialiseTemporaryExpr* stmt) -> IRValue {
    if (IsZeroSizedType(stmt->type)) {
        Emit(stmt->temporary);
        return {};
    }

    auto var = CreateAlloca(C(stmt->location()), stmt->type);
    EmitRValue(var, stmt->temporary);
    return var;
}

auto CodeGen::EmitMemberAccessExpr(MemberAccessExpr* expr) -> IRValue {
    auto base = EmitScalar(expr->base);
    if (IsZeroSizedType(expr->type)) return base;
    return CreatePtrAdd(C(expr->location()), base, expr->field->offset);
}

auto CodeGen::EmitOverloadSetExpr(OverloadSetExpr*) -> IRValue {
    Unreachable("Emitting unresolved overload set?");
}

void CodeGen::EmitProcedure(ProcDecl* proc) {
    locals.clear();
    environment_for_nested_procs = {};
    abort_info_slot = {};

    // Create the procedure.
    curr_proc = DeclareProcedure(proc);

    // If it doesn’t have a body, then we’re done.
    if (not proc->body()) return;

    // Create the entry block.
    EnterProcedure _(*this, curr_proc, proc);

    // If there is a name collision between procedures, then we will emit
    // both into a single IR function; this is very much not good, so give
    // up if that happens.
    //
    // This should never actually happen if Sema does its job properly and
    // provided we mangle nested procedures accordingly.
    if (HasTerminator()) {
        ICE(
            proc->location(),
            "An IR function with this name already exists: '{}'",
            curr_proc.getName()
        );
        return;
    }

    // Always set the visibility to public.
    //
    // tl;dr: MLIR deleting function arguments and return values causes bad things
    // to happen, and I can’t be bothered to debug that nonsense.
    //
    // The remove dead values pass or something else in MLIR is currently bugged
    // and crashes on even simple private functions; we also don’t want any MLIR
    // passes to remove unused arguments for ABI reasons (yes, a function may be
    // private, but we might end up passing it as a callback somewhere, and I’m
    // not sure it’s clever enough to figure that out).
    //
    // Furthermore, we control visibility entirely through LLVM linkage attributes,
    // so the entire MLIR visibility nonsense is categorically useless to us either
    // way.
    //
    // The test case that crashes as soon as we set this to private is:
    //
    //     program test;
    //
    //     proc x (in int exp) -> int {
    //         if exp == 0 then return 1;
    //         return 3;
    //     }
    //
    curr_proc.setVisibility(mlir::SymbolTable::Visibility::Public);

    // Lower parameters.
    //
    // We can’t rely on 'actx'’s allocation counts since that tracks ABI requirements,
    // not actual argument slots, so track them separately.
    ABILoweringContext actx;
    unsigned vals_used = 0;
    if (NeedsIndirectReturn(proc->return_type())) {
        vals_used++;
        actx.allocate();
    }

    for (auto param : proc->params()) {
        if (IsZeroSizedType(param->type)) continue;
        if (PassByReference(param->type, param->intent())) {
            locals[param] = curr_proc.getArgument(vals_used++);
            actx.allocate();
        } else {
            auto l = C(param->location());
            ABITypeRaisingContext ctx{*this, actx, l, curr_proc.getArguments().drop_front(vals_used), param->type};
            locals[param] = WriteByValParamToMemory(ctx);
            vals_used += ctx.consumed();
        }
    }

    // Declare other local variables.
    for (auto l : proc->locals) {
        if (IsZeroSizedType(l->type) or isa<ParamDecl>(l)) continue;
        EmitLocal(l);
    }

    // Emit the body.
    Emit(proc->body().get());
}

auto CodeGen::EmitProcRefExpr(ProcRefExpr* expr) -> IRValue {
    auto l = C(expr->location());
    auto op = DeclareProcedure(expr->decl);
    auto ref = create<ir::ProcRefOp>(l, op);
    if (not expr->decl->has_captures) return {ref, CreateNullPointer(l)};
    Assert(expr->decl->parent.present(), "Procedure without a parent should not have captures");

    // This procedure is directly nested within the current procedure; build
    // the environment for it.
    if (expr->decl->parent.get() == curr_proc_decl) {
        Assert(curr_proc_decl);

        // Check if we’ve already cached the computation of the environment.
        if (environment_for_nested_procs) return {ref, environment_for_nested_procs};

        // If this procedure introduces new captures, we need to build
        // a new environment.
        if (curr_proc_decl->introduces_captures) {
            // If, additionally, the environment already contained captures before
            // that, store a pointer to that environment in the new one.
            auto captures = curr_proc_decl->captured_vars();
            auto extra_ptr = curr_proc_decl->has_captures;
            auto size = tu.target().ptr_size();
            auto align = tu.target().ptr_align();
            environment_for_nested_procs = CreateAlloca(
                l,
                size * (u64(rgs::distance(captures)) + extra_ptr),
                align
            );

            if (extra_ptr) CreateStore(l, environment_for_nested_procs, GetEnvPtr(), align);
            for (auto [i, c] : enumerate(captures)) CreateStore(
                l,
                environment_for_nested_procs,
                GetAddressOfLocal(c, expr->location()),
                align,
                size * (i + extra_ptr)
            );
        }

        // Otherwise, reuse our own environment.
        else { environment_for_nested_procs = GetEnvPtr(); }
        return {ref, environment_for_nested_procs};
    }

    // Otherwise, we’re calling a sibling procedure, e.g.
    //
    //   proc f() {
    //       proc g = ...;
    //       proc h = g();
    //   }
    //
    // For this, we need to retrieve the environment pointer of the nearest
    // common ancestor of the current procedure and the one we’re calling;
    // the callee must be a direct child of that ancestor (else it would not
    // be in scope and we couldn’t call it in the first place); thus, the
    // ancestor’s environment pointer is also the environment pointer for
    // the callee. Walk the static chain to extract it.
    llvm::SmallPtrSet<ProcDecl*, 16> our_ancestors;
    for (auto p : curr_proc_decl->parents()) our_ancestors.insert(p);
    auto ancestor = our_ancestors.find(expr->decl->parent.get());
    Assert(ancestor != our_ancestors.end(), "No common ancestor!");
    return {ref, GetStaticChainPointer(*ancestor, expr->location())};
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> IRValue {
    if (HasTerminator()) return {};
    auto val = expr->value.get_or_null();
    auto ty = val ? val->type : Type::VoidTy;
    auto l = C(expr->location());

    // An indirect return is a store to a pointer.
    if (NeedsIndirectReturn(ty)) {
        EmitRValue(curr_proc.getArgument(0), expr->value.get());
        if (not HasTerminator()) create<ir::RetOp>(l, mlir::ValueRange());
        return {};
    }

    // Handle returns without a value.
    if (not val) {
        create<ir::RetOp>(l, Vals{});
        return {};
    }

    // Ignore zero-sized types.
    if (IsZeroSizedType(ty)) {
        if (val) Emit(val);
        if (not HasTerminator()) create<ir::RetOp>(l, Vals{});
        return {};
    }

    auto ret_vals = LowerDirectReturn(l, val);
    if (not HasTerminator()) create<ir::RetOp>(l, llvm::to_vector(ret_vals | vws::transform(&ABIArg::value)));
    return {};
}

auto CodeGen::EmitStrLitExpr(StrLitExpr* expr) -> IRValue {
    return CreateGlobalStringSlice(C(expr->location()), expr->value);
}

auto CodeGen::EmitStructInitExpr(StructInitExpr* e) -> IRValue {
    if (IsZeroSizedType(e->type)) {
        for (auto v : e->values()) Emit(v);
        return {};
    }

    Unreachable("Emitting struct initialiser without memory location?");
}

auto CodeGen::EmitTypeExpr(TypeExpr*) -> IRValue {
    Unreachable("Can’t emit type expr");
    return {};
}

auto CodeGen::EmitUnaryExpr(UnaryExpr* expr) -> IRValue {
    struct Increment {
        Value old_val;
        Value addr;
    };

    auto l = C(expr->location());
    auto EmitIncrement = [&] -> Increment {
        auto ptr = EmitScalar(expr->arg);
        auto a = expr->type->align(tu);
        auto val = CreateLoad(ptr.getLoc(), ptr, C(expr->type), a);
        auto new_val = EmitArithmeticOrComparisonOperator(
            expr->op == Tk::PlusPlus ? Tk::Plus : Tk::Minus,
            expr->type,
            val,
            CreateInt(l, 1, expr->type),
            l
        );

        CreateStore(l, ptr, new_val, a);
        return {.old_val = val, .addr = ptr};
    };


    if (expr->postfix) {
        switch (expr->op) {
            default: Todo("Emit postfix '{}'", expr->op);
            case Tk::PlusPlus:
            case Tk::MinusMinus:
                return EmitIncrement().old_val;
        }
    }

    switch (expr->op) {
        default: Todo("Emit prefix '{}'", expr->op);

        // These are both no-ops at the IR level.
        case Tk::Ampersand:
        case Tk::Caret:
        case Tk::Plus:
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
                val and val->storage.value() - 1 == APInt::getSignedMaxValue(u32(val->type->bit_width(tu).bits()))
            ) {
                auto copy = val->storage.value();
                copy.negate();
                return CreateInt(l, copy, expr->type);
            }

            // Otherwise, emit '0 - val'.
            auto a = EmitScalar(expr->arg);
            return CreateBinop<arith::SubIOp, ir::SSubOvOp>(
                expr->arg->type,
                CreateInt(l, 0, expr->type),
                a,
                l,
                expr->op
            );
        }

        case Tk::Not: {
            auto a = EmitScalar(expr->arg);
            return createOrFold<arith::XOrIOp>(l, a, CreateBool(l, true));
        }

        case Tk::PlusPlus:
        case Tk::MinusMinus:
            return EmitIncrement().addr;

        case Tk::Tilde: {
            auto a = EmitScalar(expr->arg);
            return createOrFold<arith::XOrIOp>(l, a, CreateInt(l, -1, expr->arg->type));
        }
    }
}

auto CodeGen::EmitWhileStmt(WhileStmt* stmt) -> IRValue {
    While(
        [&] { return EmitScalar(stmt->cond); },
        [&] { Emit(stmt->body); }
    );
    return {};
}

auto CodeGen::EmitValue(Location loc, const eval::RValue& val) -> IRValue { // clang-format off
    utils::Overloaded V {
        [&](std::monostate) -> IRValue { return {}; },
        [&](Type) -> IRValue { Unreachable("Cannot emit type constant"); },
        [&](const APInt& value) -> IRValue { return CreateInt(C(loc), value, val.type()); },
        [&](eval::MemoryValue) -> IRValue { return {}; }, // This only happens if the value is unused.
        [&](const eval::Range& r) -> IRValue {
            auto el = cast<RangeType>(val.type())->elem();
            auto l = C(loc);
            return {CreateInt(l, r.start, el), CreateInt(l, r.end, el)};
        },
    }; // clang-format on
    return val.visit(V);
}

auto CodeGen::emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::ProcOp {
    Assert(bool(lang_opts.constant_eval));

    // Delete any remnants of the last constant evaluation.
    if (vm_entry_point) vm_entry_point.erase();

    // Irrespective of what the argument type is, we return it indirectly through
    // a pointer to the first argument. This avoids having to make the constant
    // evaluator aware of ABI type rules.
    SmallVector<ParamTypeData> args;
    auto ty = stmt->type_or_void();
    auto yields_value = not IsZeroSizedType(ty);
    if (yields_value) args.push_back({Intent::Copy, PtrType::Get(tu, ty)});

    // Build a procedure for this statement.
    setInsertionPointToEnd(&mlir_module.getBodyRegion().front());
    auto info = ConvertProcType(ProcType::Get(tu, Type::VoidTy, args), false);
    auto loc = C(stmt->location());
    vm_entry_point = create<ir::ProcOp>(
        loc,
        constants::VMEntryPointName,
        C(Linkage::Internal),
        C(CallingConvention::Native),
        info.func,
        false,
        false,
        mlir::ArrayAttr::get(&mlir, info.arg_attrs),
        mlir::ArrayAttr::get(&mlir, info.result_attrs)
    );

    EnterProcedure _(*this, vm_entry_point, nullptr);
    if (yields_value) EmitRValue(vm_entry_point.getCallArg(0), cast<Expr>(stmt));
    else Emit(stmt);

    // Make sure to return from the procedure.
    if (not HasTerminator()) create<ir::RetOp>(loc, Vals());

    // Run canonicalisation etc.
    if (not finalise(vm_entry_point)) return nullptr;
    return vm_entry_point;
}

auto CodeGen::lookup(ir::ProcOp op) -> Ptr<ProcDecl> {
    auto it = proc_reverse_lookup.find(op);
    if (it != proc_reverse_lookup.end()) return it->second;
    return nullptr;
}
