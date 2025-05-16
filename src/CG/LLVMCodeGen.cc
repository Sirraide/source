#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

using namespace srcc;
using namespace srcc::cg;
namespace intrin = llvm::Intrinsic;

class cg::LLVMCodeGen : DiagsProducer<std::nullptr_t>, llvm::IRBuilder<> {
    friend DiagsProducer;

    CodeGen& cg;
    llvm::TargetMachine& machine;
    StringMap<llvm::Constant*> strings;
    std::unique_ptr<llvm::Module> llvm;

    llvm::IntegerType* const IntTy;
    llvm::IntegerType* const I1Ty;
    llvm::IntegerType* const I8Ty;
    llvm::PointerType* const PtrTy;
    llvm::IntegerType* const FFIIntTy;
    llvm::StructType* const SliceTy;
    llvm::StructType* const ClosureTy;
    llvm::StructType* const AbortHandlerArgTy;
    llvm::Type* const VoidTy;
    llvm::FunctionType* const AbortHandlerTy;
    StringMap<llvm::FunctionCallee> abort_handlers;

    struct BlockInfo {
        llvm::BasicBlock* block;
        SmallVector<llvm::PHINode*> phis;
    };

    DenseMap<ir::Block*, BlockInfo> blocks;
    DenseMap<ir::Inst*, llvm::Value*> instructions;
    DenseMap<ir::FrameSlot*, llvm::AllocaInst*> frame_slots;
    ir::Proc* curr_proc = nullptr;
    llvm::Function* curr_func = nullptr;

public:
    LLVMCodeGen(llvm::TargetMachine& target, CodeGen& cg);

    void emit(ir::Proc* proc);
    auto finalise() -> std::unique_ptr<llvm::Module>;

private:
    template <typename Ty = llvm::Type>
    auto ConvertType(Type ty, bool array_elem = false) -> Ty*;
    auto ConvertTypeForMem(Type ty) -> llvm::Type*;
    auto ConvertTypeImpl(Type ty, bool array_elem) -> llvm::Type*;
    auto ConvertProcType(ProcType* ty, bool add_env_ptr = false) -> llvm::FunctionType*;
    auto DeclareProc(ir::Proc* proc) -> llvm::FunctionCallee;
    auto Emit(ir::Inst& i) -> llvm::Value*;
    auto Emit(ir::Value* v) -> llvm::Value*;
    auto InternStringPtr(StringRef value) -> llvm::Constant*;
    auto MakeInt(i64 value) -> llvm::ConstantInt*;
};

LLVMCodeGen::LLVMCodeGen(llvm::TargetMachine& target, CodeGen& cg)
    : IRBuilder(cg.tu.llvm_context), cg{cg}, machine(target),
      llvm{std::make_unique<llvm::Module>(cg.tu.name, cg.tu.llvm_context)},
      IntTy{getInt64Ty()},
      I1Ty{getInt1Ty()},
      I8Ty{getInt8Ty()},
      PtrTy{getPtrTy()},
      FFIIntTy{cast<llvm::IntegerType>(ConvertType(cg.tu.FFIIntTy))},
      SliceTy{llvm::StructType::get(PtrTy, IntTy)},
      ClosureTy{llvm::StructType::get(PtrTy, PtrTy)},
      AbortHandlerArgTy{llvm::StructType::get(SliceTy, IntTy, IntTy, SliceTy, SliceTy)},
      VoidTy{getVoidTy()},
      AbortHandlerTy{llvm::FunctionType::get(VoidTy, {PtrTy}, false)} {
    llvm->setTargetTriple(machine.getTargetTriple());
    llvm->setDataLayout(machine.createDataLayout());

    // Create abort handlers.
    for (auto a : constants::AbortHandlers) {
        auto callee = abort_handlers[a] = llvm->getOrInsertFunction(a, AbortHandlerTy);
        auto h = cast<llvm::Function>(callee.getCallee());
        h->setDoesNotReturn();
        h->setDoesNotThrow();
        h->setCallingConv(llvm::CallingConv::Fast);
    }
}

auto CodeGen::emit_llvm(llvm::TargetMachine& target) -> std::unique_ptr<llvm::Module> {
    LLVMCodeGen llvm_cg{target, *this};
    for (auto proc : procedures()) llvm_cg.emit(proc);
    return llvm_cg.finalise();
}

auto LLVMCodeGen::finalise() -> std::unique_ptr<llvm::Module> {
    // Emit the module description if this is a module.
    if (cg.tu.is_module) {
        SmallString<0> md;
        cg.tu.serialise(md);
        auto mb = llvm::MemoryBuffer::getMemBuffer(md, "", false);
        llvm::embedBufferInModule(*llvm, mb->getMemBufferRef(), constants::ModuleDescriptionSectionName(cg.tu.name));
    }

    return std::move(llvm);
}


// ============================================================================
//  Helpers
// ============================================================================
namespace {
auto ConvertAlign(Align a) -> llvm::Align {
    return llvm::Align{a.value().bytes()};
}

auto ConvertCC(CallingConvention cc) -> llvm::CallingConv::ID {
    switch (cc) {
        case CallingConvention::Source: return llvm::CallingConv::Fast;
        case CallingConvention::Native: return llvm::CallingConv::C;
    }

    Unreachable("Unknown calling convention");
}

auto ConvertLinkage(Linkage lnk) -> llvm::GlobalValue::LinkageTypes {
    switch (lnk) {
        using L = llvm::GlobalValue::LinkageTypes;
        case Linkage::Internal: return L::PrivateLinkage;
        case Linkage::Exported: return L::ExternalLinkage;
        case Linkage::Imported: return L::ExternalLinkage;
        case Linkage::Reexported: return L::ExternalLinkage;
        case Linkage::Merge: return L::LinkOnceODRLinkage;
    }

    Unreachable("Unknown linkage");
}

auto P(llvm::Type* ty) -> std::string {
    std::string s;
    llvm::raw_string_ostream os{s};
    ty->print(os, true);
    return s;
}
}

// ============================================================================
//  Type Conversion
// ============================================================================
template <typename Ty>
auto LLVMCodeGen::ConvertType(Type ty, bool array_elem) -> Ty* {
    return cast<Ty>(ConvertTypeImpl(ty, array_elem));
}

auto LLVMCodeGen::ConvertTypeImpl(Type ty, bool array_elem) -> llvm::Type* {
    Assert(ty, "Null type in codegen");
    switch (ty->kind()) {
        case TypeBase::Kind::ProcType: Unreachable("Call ConvertTypeForMem() or ConvertProcType() instead");
        case TypeBase::Kind::SliceType: return SliceTy;
        case TypeBase::Kind::PtrType: return PtrTy;
        case TypeBase::Kind::IntType: return getIntNTy(u32(cast<IntType>(ty)->bit_width().bits()));

        case TypeBase::Kind::RangeType: {
            auto elem = ConvertTypeForMem(cast<RangeType>(ty)->elem());
            return llvm::StructType::get(llvm->getContext(), {elem, elem});
        }

        case TypeBase::Kind::ArrayType: {
            // FIXME: This doesn’t handle structs correctly at the moment.
            // FIXME: Is the above FIXME still relevant?
            auto arr = cast<ArrayType>(ty);
            auto elem = ConvertType(arr->elem(), true);
            return llvm::ArrayType::get(elem, u64(arr->dimension()));
        }

        case TypeBase::Kind::IRAggregateType: {
            SmallVector<llvm::Type*> fields;
            for (auto f : cast<IRAggregateType>(ty)->layout()->fields())
                fields.push_back(ConvertTypeForMem(f));
            return llvm::StructType::get(llvm->getContext(), fields);
        }

        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::Deduced:
                    Unreachable("Deduced type in codegen?");

                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable("Unresolved overload set type in codegen?");

                case BuiltinKind::Type:
                    Unreachable("Cannot emit 'type' type");

                case BuiltinKind::Bool: return I1Ty;
                case BuiltinKind::Int: return IntTy;

                case BuiltinKind::Void:
                case BuiltinKind::NoReturn:
                    return VoidTy;
            }

            Unreachable("Unknown builtin type");
        }

        case TypeBase::Kind::StructType: {
            auto s = cast<StructType>(ty);
            auto sz = array_elem ? s->layout()->array_size() : s->layout()->size();
            return llvm::ArrayType::get(I8Ty, u64(sz.bytes()));
        }
    }

    Unreachable("Unknown type kind");
}

auto LLVMCodeGen::ConvertTypeForMem(Type ty) -> llvm::Type* {
    // Convert procedure types to closures.
    if (isa<ProcType>(ty)) return ClosureTy;

    // Convert >128 bit integers to arrays.
    // FIXME: Should this happen in codegen, i.e. is it platform-dependent?
    if (ty->is_integer()) {
        auto sz = ty->size(cg.tu);
        if (sz > Size::Bytes(128)) return llvm::ArrayType::get(getInt8Ty(), sz.bytes());
    }

    return ConvertType(ty);
}

auto LLVMCodeGen::ConvertProcType(ProcType* ty, bool add_env_ptr) -> llvm::FunctionType* {
    auto ret = ConvertTypeForMem(ty->ret());
    DebugAssert(
        llvm::FunctionType::isValidReturnType(ret),
        "Not a valid LLVM return type: {} (from {})",
        P(ret),
        ty->ret()
    );

    SmallVector<llvm::Type*> args;
    for (const auto& p : ty->params()) {
        auto a = ConvertTypeForMem(p.type);
        args.push_back(a);
        DebugAssert(
            llvm::FunctionType::isValidArgumentType(a),
            "Not a valid LLVM argument type: {} (from {})",
            P(a),
            p.type
        );
    }

    if (add_env_ptr) args.push_back(getPtrTy());
    return llvm::FunctionType::get(ret, args, ty->variadic());
}

// ============================================================================
//  Emitting Code
// ============================================================================
auto LLVMCodeGen::DeclareProc(ir::Proc* proc) -> llvm::FunctionCallee {
    if (auto f = llvm->getFunction(proc->name())) return f;
    auto ty = proc->proc_type();
    auto callee = llvm->getOrInsertFunction(proc->name(), ConvertProcType(ty));
    auto func = cast<llvm::Function>(callee.getCallee());
    func->setCallingConv(ConvertCC(ty->cconv()));
    func->setLinkage(ConvertLinkage(proc->linkage()));
    return callee;
}

void LLVMCodeGen::emit(ir::Proc* proc) {
    curr_proc = proc;
    curr_func = cast<llvm::Function>(DeclareProc(proc).getCallee());

    // Propagate 'noreturn'.
    if (proc->proc_type()->ret() == Type::NoReturnTy)
        curr_func->setDoesNotReturn();

    // Process attributes.
    for (auto [i, a] : proc->param_attrs) {
        llvm::AttrBuilder attrs{llvm->getContext()};

        if (a.ll_sret) {
            attrs.addStructRetAttr(ConvertTypeForMem(a.ty));
            attrs.addAttribute(llvm::Attribute::DeadOnUnwind);
            attrs.addAttribute(llvm::Attribute::NoAlias);
            attrs.addAttribute(llvm::Attribute::Writable);
            attrs.addCapturesAttr(llvm::CaptureInfo::none());
            attrs.addAlignmentAttr(ConvertAlign(a.ty->align(cg.tu)));
        }

        if (a.ll_byval) {
            attrs.addByValAttr(ConvertTypeForMem(a.ty));
            attrs.addAttribute(llvm::Attribute::NoUndef);
            attrs.addAttribute(llvm::Attribute::NoAlias);
            attrs.addCapturesAttr(llvm::CaptureInfo::none());
            attrs.addAlignmentAttr(ConvertAlign(a.ty->align(cg.tu)));
        }

        if (a.ll_signext) attrs.addAttribute(llvm::Attribute::SExt);
        if (a.ll_zeroext) attrs.addAttribute(llvm::Attribute::ZExt);

        curr_func->addParamAttrs(i, attrs);
    }

    // Stop here if there is no function body.
    if (proc->empty()) return;
    blocks.clear();
    instructions.clear();

    // Create all the blocks ahead of time.
    for (auto [i, b] : enumerate(proc->blocks())) {
        auto bb = llvm::BasicBlock::Create(
            getContext(),
            b == proc->entry() ? "entry" : std::format("bb{}", i),
            curr_func
        );

        // Create a PHI for each block argument.
        SetInsertPoint(bb);
        SmallVector<llvm::PHINode*> args;
        for (auto arg : b->split_arguments())
            args.push_back(CreatePHI(ConvertType(arg->type()), 0));

        // Store the block for later.
        blocks[b] = {bb, args};
    }

    // Emit the stack slots.
    SetInsertPoint(blocks[proc->entry()].block);
    for (auto f : proc->frame()) {
        auto ty = f->allocated_type();
        auto a = CreateAlloca(ConvertTypeForMem(ty));
        a->setAlignment(std::max(a->getAlign(), ConvertAlign(f->align())));
        frame_slots[f] = a;
    }

    // Finally, emit every block.
    for (auto b : proc->blocks()) {
        SetInsertPoint(blocks[b].block);
        for (auto i : b->instructions())
            if (auto v = Emit(*i))
                instructions[i] = v;
    }
}

auto LLVMCodeGen::Emit(ir::Inst& i) -> llvm::Value* {
    switch (i.opcode()) {
        using ir::Op;

        // Special instructions.
        case Op::Abort: {
            auto& a = cast<ir::AbortInst>(i);
            SmallVector<llvm::Value*> args;

            // Get file, line, and column. Don’t require a valid location here as
            // this is also called from within implicitly generated code.
            if (auto lc = a.location().seek_line_column(cg.tu.context())) {
                auto name = cg.tu.context().file_name(a.location().file_id);
                args.push_back(InternStringPtr(name));
                args.push_back(MakeInt(i64(name.size())));
                args.push_back(MakeInt(i64(lc->line)));
                args.push_back(MakeInt(i64(lc->col)));
            } else {
                args.push_back(llvm::ConstantPointerNull::get(getPtrTy()));
                args.push_back(MakeInt(0));
                args.push_back(MakeInt(0));
                args.push_back(MakeInt(0));
            }

            for (auto arg : a.args()) args.push_back(Emit(arg));
            auto stack_val = CreateAlloca(AbortHandlerArgTy);
            Size offs;
            for (auto arg : args) {
                auto ptr = CreatePtrAdd(stack_val, getInt64(offs.bytes()));
                CreateStore(arg, ptr);
                offs += Size::Bytes(8);
            }

            auto c = CreateCall(abort_handlers.at(a.handler_name()), stack_val);
            c->setCallingConv(llvm::CallingConv::Fast);
            CreateUnreachable();
            return {};
        }

        case Op::Br: {
            auto AddArgs = [&](BlockInfo& b, ArrayRef<ir::Value*> args) {
                for (auto [phi, arg] : zip(b.phis, args))
                    phi->addIncoming(Emit(arg), GetInsertBlock());
            };

            // Emit then args.
            auto& b = cast<ir::BranchInst>(i);
            auto then = blocks.at(b.then());
            AddArgs(then, b.then_args());

            // If the branch is not conditional, we’re done.
            if (not b.is_conditional()) return CreateBr(then.block);

            // Emit else args.
            auto else_ = blocks.at(b.else_());
            AddArgs(else_, b.else_args());

            // Finally, branch.
            return CreateCondBr(Emit(b.cond()), then.block, else_.block);
        }

        case Op::Call: {
            auto& c = cast<ir::CallInst>(i);
            auto proc = c.args().front();
            auto args = c.args().drop_front(1);
            auto ty = c.proc_type();

            // Emit the callee and environment.
            auto callee = Emit(proc);
            auto env = c.environment().present() ? Emit(c.environment().get()) : nullptr;

            // Emit the arguments.
            SmallVector<llvm::Value*> llvm_args;
            for (auto arg : args) llvm_args.push_back(Emit(arg));

            // Helper to create a call.
            auto BuildCall = [&](llvm::FunctionType* ftype) {
                DebugAssert(
                    ftype->isVarArg() or ftype->params().size() == llvm_args.size(),
                    "Argument count mismatch, got {}, wanted {}",
                     llvm_args.size(),
                     ftype->params().size()
                );

                for (auto [ty, arg] : llvm::zip(ftype->params(), llvm_args)) DebugAssert(
                    ty == arg->getType(),
                    "LLVM type mismatch in call, got {}, wanted {}",
                    P(arg->getType()),
                    P(ty)
                );

                auto call =  CreateCall(ftype, callee, llvm_args);
                call->setCallingConv(ConvertCC(ty->cconv()));
                return call;
            };

            // If there is no environment, just build the call.
            if (not env) return BuildCall(ConvertProcType(c.proc_type()));

            // Otherwise, we need to add the environment, but only if it is non-null,
            // so we need to build a branch here.
            auto then = llvm::BasicBlock::Create(getContext(), "", curr_func);
            auto else_ = llvm::BasicBlock::Create(getContext(), "", curr_func);
            auto join = llvm::BasicBlock::Create(getContext(), "", curr_func);
            auto is_null = CreateICmpEQ(env, llvm::ConstantPointerNull::get(getPtrTy()));

            // Env is null.
            CreateCondBr(is_null, then, else_);
            SetInsertPoint(then);
            auto call1 = BuildCall(ConvertProcType(c.proc_type()));
            CreateBr(join);

            // Env is not null.
            SetInsertPoint(else_);
            llvm_args.push_back(env);
            auto call2 = BuildCall(ConvertProcType(c.proc_type(), true));
            CreateBr(join);

            // Merge the two values.
            SetInsertPoint(join);
            if (call1->getType()->isVoidTy()) return nullptr;
            auto phi = CreatePHI(call1->getType(), 2);
            phi->addIncoming(call1, then);
            phi->addIncoming(call2, else_);
            return phi;
        }

        case Op::Load: {
            auto& m = cast<ir::MemInst>(i);
            auto ty = ConvertType(m.memory_type()); // NOT ForMem(), because this is the RESULT type of the load.
            return CreateAlignedLoad(ty, Emit(m.ptr()), ConvertAlign(m.align()));
        }

        case Op::MemCopy: {
            auto a = ConvertAlign(i.align());
            return CreateMemCpy(Emit(i[0]), a, Emit(i[1]), a, Emit(i[2]));
        }

        case Op::MemZero: {
            auto zero = getInt8(0);
            return CreateMemSet(Emit(i[0]), zero, Emit(i[1]), ConvertAlign(i.align()));
        }

        case Op::Store: {
            auto& m = cast<ir::MemInst>(i);
            return CreateAlignedStore(Emit(m.value()), Emit(m.ptr()), ConvertAlign(m.align()));
        }

        case Op::Ret: {
            if (i.args().empty()) return CreateRetVoid();
            if (i.args().size() == 1) return CreateRet(Emit(i[0]));
            SmallVector<llvm::Value*> args;
            for (auto a : i.args())  args.push_back(Emit(a));
            return CreateAggregateRet(args.data(), u32(args.size()));
        }

        // Cast instructions.
        case Op::SExt: return CreateSExt(Emit(i[0]), ConvertType(cast<ir::ICast>(i).cast_result_type()));
        case Op::Trunc: return CreateTrunc(Emit(i[0]), ConvertType(cast<ir::ICast>(i).cast_result_type()));
        case Op::ZExt: return CreateZExt(Emit(i[0]), ConvertType(cast<ir::ICast>(i).cast_result_type()));

        // Overflow intrinsics.
        case Op::SAddOv: return CreateBinaryIntrinsic(intrin::sadd_with_overflow, Emit(i[0]), Emit(i[1]));
        case Op::SMulOv: return CreateBinaryIntrinsic(intrin::smul_with_overflow, Emit(i[0]), Emit(i[1]));
        case Op::SSubOv: return CreateBinaryIntrinsic(intrin::ssub_with_overflow, Emit(i[0]), Emit(i[1]));

        // Basic instructions.
        case Op::Add: return CreateAdd(Emit(i[0]), Emit(i[1]), "", not cg.lang_opts.overflow_checking, not cg.lang_opts.overflow_checking);
        case Op::And: return CreateAnd(Emit(i[0]), Emit(i[1]));
        case Op::AShr: return CreateAShr(Emit(i[0]), Emit(i[1]));
        case Op::ICmpEq: return CreateICmpEQ(Emit(i[0]), Emit(i[1]));
        case Op::ICmpNe: return CreateICmpNE(Emit(i[0]), Emit(i[1]));
        case Op::ICmpSGe: return CreateICmpSGE(Emit(i[0]), Emit(i[1]));
        case Op::ICmpSGt: return CreateICmpSGT(Emit(i[0]), Emit(i[1]));
        case Op::ICmpSLe: return CreateICmpSLE(Emit(i[0]), Emit(i[1]));
        case Op::ICmpSLt: return CreateICmpSLT(Emit(i[0]), Emit(i[1]));
        case Op::ICmpUGe: return CreateICmpUGE(Emit(i[0]), Emit(i[1]));
        case Op::ICmpUGt: return CreateICmpUGT(Emit(i[0]), Emit(i[1]));
        case Op::ICmpULe: return CreateICmpULE(Emit(i[0]), Emit(i[1]));
        case Op::ICmpULt: return CreateICmpULT(Emit(i[0]), Emit(i[1]));
        case Op::IMul: return CreateMul(Emit(i[0]), Emit(i[1]), "", not cg.lang_opts.overflow_checking, not cg.lang_opts.overflow_checking);
        case Op::LShr: return CreateLShr(Emit(i[0]), Emit(i[1]));
        case Op::Or: return CreateOr(Emit(i[0]), Emit(i[1]));
        case Op::PtrAdd: return CreatePtrAdd(Emit(i[0]), Emit(i[1]), "", i.inbounds);
        case Op::SDiv: return CreateSDiv(Emit(i[0]), Emit(i[1]));
        case Op::Select: return CreateSelect(Emit(i[0]), Emit(i[1]), Emit(i[2]));
        case Op::Shl: return CreateShl(Emit(i[0]), Emit(i[1]), "", not cg.lang_opts.overflow_checking, not cg.lang_opts.overflow_checking);
        case Op::SRem: return CreateSRem(Emit(i[0]), Emit(i[1]));
        case Op::Sub: return CreateSub(Emit(i[0]), Emit(i[1]), "", not cg.lang_opts.overflow_checking, not cg.lang_opts.overflow_checking);
        case Op::UDiv: return CreateUDiv(Emit(i[0]), Emit(i[1]));
        case Op::Unreachable: return CreateUnreachable();
        case Op::URem: return CreateURem(Emit(i[0]), Emit(i[1]));
        case Op::Xor: return CreateXor(Emit(i[0]), Emit(i[1]));
    }

    Unreachable("Invalid opcode: {}", +i.opcode());
}

auto LLVMCodeGen::Emit(ir::Value* v) -> llvm::Value* {
    switch (v->kind()) {
        using K = ir::Value::Kind;

        case K::Aggregate:
            Unreachable("Aggregates should never make it out of codegen");

        case K::Argument: {
            auto arg = cast<ir::Argument>(v);
            if (isa<ir::Proc>(arg->parent())) return curr_func->getArg(arg->index());
            return blocks.at(cast<ir::Block>(arg->parent())).phis[arg->index()];
        }

        case K::Block:
            return blocks.at(cast<ir::Block>(v)).block;

        case K::BuiltinConstant: {
            auto c = cast<ir::BuiltinConstant>(v);
            switch (c->id) {
                case ir::BuiltinConstantKind::True: return getTrue();
                case ir::BuiltinConstantKind::False: return getFalse();
                case ir::BuiltinConstantKind::Nil: return llvm::Constant::getNullValue(ConvertTypeForMem(c->type()));
                case ir::BuiltinConstantKind::Poison: return llvm::PoisonValue::get(ConvertType(c->type()));
            }

            Unreachable();
        }

        case K::FrameSlot: return frame_slots.at(cast<ir::FrameSlot>(v));
        case K::GlobalConstant: {
            auto g = cast<ir::GlobalConstant>(v);
            if (g->is_string()) return InternStringPtr(g->value());
            auto c = CreateGlobalString(g->value(), "", 0, nullptr, false);
            c->setAlignment(ConvertAlign(g->align()));
            return c;
        }

        case K::InstValue: {
            auto i = cast<ir::InstValue>(v);
            auto inst = instructions.at(i->inst());
            if (not i->inst()->has_multiple_results()) return inst;
            return CreateExtractValue(inst, i->index());
        }

        case K::InvalidLocalReference:
            Unreachable("Should only exist in constant evaluation");

        case K::LargeInt:
            return getInt(cast<ir::LargeInt>(v)->value());

        case K::Proc:
            return DeclareProc(cast<ir::Proc>(v)).getCallee();

        case K::SmallInt: {
            // FIXME: If type()->size() ever rounds up to 64 bits, we need to retrieve
            // the actual bit width here.
            //
            // FIXME: Or alternatively, just add i->type()->memory_size() vs i->type()->size().
            auto i = cast<ir::SmallInt>(v);
            auto x = getIntN(u32(i->type()->size(cg.tu).bits()), u64(i->value()));
            return x;
        }
    }

    Unreachable("Invalid value kind: {}", +v->kind());
}

auto LLVMCodeGen::InternStringPtr(StringRef value) -> llvm::Constant* {
    if (auto it = strings.find(value); it != strings.end()) return it->second;
    return strings[value] = CreateGlobalString(value);
}

auto LLVMCodeGen::MakeInt(i64 value) -> llvm::ConstantInt* {
    return llvm::ConstantInt::get(IntTy, u64(value));
}
