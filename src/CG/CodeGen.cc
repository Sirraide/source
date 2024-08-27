module;

#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/ConstantFold.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.codegen;
import srcc;
import srcc.token;
import srcc.constants;
using namespace srcc;
using llvm::BasicBlock;
using llvm::ConstantInt;
using llvm::IRBuilder;
using llvm::Value;
namespace Intrinsic = llvm::Intrinsic;

// ============================================================================
//  Helpers
// ============================================================================
auto CodeGen::ConvertCC(CallingConvention cc) -> llvm::CallingConv::ID {
    switch (cc) {
        case CallingConvention::Source: return llvm::CallingConv::Fast;
        case CallingConvention::Native: return llvm::CallingConv::C;
    }

    Unreachable("Unknown calling convention");
}

auto CodeGen::ConvertLinkage(Linkage lnk) -> llvm::GlobalValue::LinkageTypes {
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

void CodeGen::CreateArithFailure(Value* cond, Tk op, Location loc, StringRef name) {
    If(cond, [&] {
        // Get file, line, and column. Don’t require a valid location here as
        // this is also called from within implicitly generated code.
        Value *file, *line, *col;
        if (auto lc = loc.seek_line_column(M.context())) {
            file = GetStringSlice(M.context().file(loc.file_id)->name());
            line = MakeInt(lc->line);
            col = MakeInt(lc->col);
        } else {
            file = llvm::ConstantAggregateZero::get(SliceTy);
            line = MakeInt(0);
            col = MakeInt(0);
        }

        // Emit the failure handler.
        auto handler = DeclareArithmeticFailureHandler();
        auto op_token = GetStringSlice(Spelling(op));
        auto operation = GetStringSlice(name);
        builder.CreateCall(handler, {file, line, col, op_token, operation});
        builder.CreateUnreachable();
    });
}

auto CodeGen::DeclareAssertFailureHandler() -> llvm::FunctionCallee {
    if (not assert_failure_handler) {
        // proc __src_assert_fail (
        //     u8[] file,
        //     int line,
        //     int col,
        //     u8[] cond,
        //     u8[] message
        // ) -> noreturn
        assert_failure_handler = llvm->getOrInsertFunction(
            "__src_assert_fail",
            llvm::FunctionType::get(VoidTy, {SliceTy, IntTy, IntTy, SliceTy, SliceTy}, false)
        );

        auto f = cast<llvm::Function>(assert_failure_handler.value().getCallee());
        f->setDoesNotReturn();
        f->setDoesNotThrow();
    }

    return assert_failure_handler.value();
}

auto CodeGen::DeclareArithmeticFailureHandler() -> llvm::FunctionCallee {
    if (not overflow_handler) {
        // proc __src_int_arith_error(
        //    u8[] file,
        //    int line,
        //    int col,
        //    u8[] operator,
        //    u8[] operation,
        // )
        overflow_handler = llvm->getOrInsertFunction(
            "__src_int_arith_error",
            llvm::FunctionType::get(VoidTy, {SliceTy, IntTy, IntTy, SliceTy, SliceTy}, false)
        );

        auto f = cast<llvm::Function>(overflow_handler.value().getCallee());
        f->setDoesNotReturn();
        f->setDoesNotThrow();
    }

    return overflow_handler.value();
}

auto CodeGen::DeclareProcedure(ProcDecl* proc) -> llvm::FunctionCallee {
    auto name = MangledName(proc);
    return llvm->getOrInsertFunction(name, ConvertType<llvm::FunctionType>(proc->type));
}

auto CodeGen::DefineExp(llvm::Type* ty) -> llvm::FunctionCallee {
    if (auto it = exp_funcs.find(ty); it != exp_funcs.end())
        return it->second;

    auto func = llvm::Function::Create(
        llvm::FunctionType::get(ty, {ty, ty}, false),
        llvm::Function::PrivateLinkage,
        std::format("__srcc_exp_i{}", ty->getScalarSizeInBits()),
        llvm.get()
    );

    func->setCallingConv(llvm::CallingConv::Fast);
    func->setDoesNotThrow();
    func->setMustProgress();
    exp_funcs[ty] = func;

    // For '**', we use the same algorithm that is used during constant
    // evaluation; see EvaluationContext::EvalBinaryExpr() for an explanation
    // of how this works.
    EnterFunction _(*this, func);

    // Values that we’ll need.
    auto minus_one = ConstantInt::get(ty, u64(-1));
    auto zero = ConstantInt::get(ty, 0);
    auto one = ConstantInt::get(ty, 1);
    auto lhs = &func->arg_begin()[0];
    auto rhs = &func->arg_begin()[1];

    // x ** 0 = 1.
    If(builder.CreateICmpEQ(rhs, zero), [&] {
        builder.CreateRet(one);
    });

    // If base == 0.
    If(builder.CreateICmpEQ(lhs, zero), [&] {
        // If exp < 0, then error.
        if (M.lang_opts().OverflowChecks) {
            CreateArithFailure(
                builder.CreateICmpSLT(rhs, zero),
                Tk::StarStar,
                Location(),
                "division by zero"
            );
        } else {
            If(builder.CreateICmpSLT(rhs, zero), [&] {
                builder.CreateRet(llvm::PoisonValue::get(ty));
            });
        }

        // Otherwise, return 0.
        builder.CreateRet(zero);
    });

    // If exp < 0.
    If(builder.CreateICmpSLT(rhs, zero), [&] {
        // If base == -1, then return 1 if exp is even, -1 if odd.
        If(builder.CreateICmpEQ(lhs, minus_one), [&] {
            auto is_odd = builder.CreateTrunc(rhs, I1Ty);
            auto result = builder.CreateSelect(is_odd, minus_one, one);
            builder.CreateRet(result);
        });

        // If base == 1, then return 1, otherwise 0.
        auto cmp = builder.CreateICmpEQ(lhs, one);
        builder.CreateRet(builder.CreateSelect(cmp, one, zero));
    });

    // Handle overflow.
    auto min_value = ConstantInt::get(ty, APInt::getSignedMinValue(ty->getIntegerBitWidth()));
    auto is_min = builder.CreateICmpEQ(lhs, min_value);
    CreateArithFailure(is_min, Tk::StarStar, Location());

    // Emit the multiplication loop.
    Loop([&](BasicBlock* bb_start) {
        auto val = builder.CreatePHI(ty, 2);
        auto exp = builder.CreatePHI(ty, 2);
        val->addIncoming(lhs, bb_start);
        exp->addIncoming(rhs, bb_start);
        If(builder.CreateICmpEQ(exp, zero), [&] {
            builder.CreateRet(val);
        });

        // Computation (and overflow check).
        auto new_val = EmitArithmeticOrComparisonOperator(Tk::Star, val, lhs, Location());
        auto new_exp = builder.CreateSub(exp, one);
        val->addIncoming(new_val, builder.GetInsertBlock());
        exp->addIncoming(new_exp, builder.GetInsertBlock());
    });

    // No return here since we return in the loop.
    return func;
}

auto CodeGen::EnterBlock(BasicBlock* bb) -> BasicBlock* {
    // Add the block to the current function if we haven’t already done so.
    if (not bb->getParent()) curr_func->insert(curr_func->end(), bb);

    // If there is a current block, and it is not closed, branch to the newly
    // inserted block, unless that block is the function’s entry block.
    if (
        auto b = builder.GetInsertBlock();
        b and not b->getTerminator() and &curr_func->getEntryBlock() != bb
    ) builder.CreateBr(bb);

    // Finally, position the builder at the end of the block.
    builder.SetInsertPoint(bb);
    return bb;
}

CodeGen::EnterFunction::EnterFunction(CodeGen& CG, llvm::Function* func)
    : CG(CG), old_func(CG.curr_func), guard{CG.builder} {
    CG.curr_func = func;

    // Create the entry block if it doesn’t exist yet.
    if (func->empty()) BasicBlock::Create(CG.llvm->getContext(), "entry", func);
    CG.EnterBlock(&func->getEntryBlock());
}

auto CodeGen::GetStringPtr(StringRef s) -> llvm::Constant* {
    if (auto it = strings.find(s); it != strings.end()) return it->second;
    return strings[s] = builder.CreateGlobalStringPtr(s);
}

auto CodeGen::GetStringSlice(StringRef s) -> llvm::Constant* {
    auto ptr = GetStringPtr(s);
    auto size = ConstantInt::get(IntTy, s.size());
    return llvm::ConstantStruct::getAnon({ptr, size});
}

auto CodeGen::If(
    Value* cond,
    llvm::function_ref<Value*()> emit_then,
    llvm::function_ref<Value*()> emit_else
) -> llvm::PHINode* {
    auto bb_then = BasicBlock::Create(llvm->getContext());
    auto bb_join = BasicBlock::Create(llvm->getContext());
    auto bb_else = emit_else ? BasicBlock::Create(llvm->getContext()) : bb_join;
    builder.CreateCondBr(cond, bb_then, bb_else);

    // Emit the then block.
    EnterBlock(bb_then);
    auto then_val = emit_then();
    auto then_val_block = builder.GetInsertBlock();
    if (not builder.GetInsertBlock()->getTerminator())
        builder.CreateBr(bb_join);

    // Emit the else block if there is one.
    Value* else_val{};
    BasicBlock* else_val_block{};
    if (emit_else) {
        EnterBlock(bb_else);
        else_val = emit_else();
        else_val_block = builder.GetInsertBlock();
        if (not builder.GetInsertBlock()->getTerminator())
            builder.CreateBr(bb_join);
    }

    // Resume inserting at the join block.
    EnterBlock(bb_join);

    // Emit a PHI for the result values if they’re both present.
    if (then_val and else_val) {
        auto phi = builder.CreatePHI(then_val->getType(), 2);
        phi->addIncoming(then_val, then_val_block);
        phi->addIncoming(else_val, else_val_block);
        return phi;
    }

    return nullptr;
}

auto CodeGen::If(Value* cond, llvm::function_ref<void()> emit_then) -> BasicBlock* {
    If(cond, [&] -> Value* {
        emit_then();
        return nullptr;
    }, nullptr);

    return builder.GetInsertBlock();
}

void CodeGen::Loop(llvm::function_ref<void(BasicBlock*)> emit_body) {
    auto bb_start = builder.GetInsertBlock();
    auto bb_cond = EnterBlock(BasicBlock::Create(M.llvm_context));
    emit_body(bb_start);
    builder.CreateBr(bb_cond);
}

auto CodeGen::MakeInt(const APInt& value) -> ConstantInt* {
    return ConstantInt::get(M.llvm_context, value);
}

auto CodeGen::MakeInt(u64 integer) -> ConstantInt* {
    return ConstantInt::get(IntTy, integer, true);
}

// ============================================================================
//  Mangling
// ============================================================================
// Mangling codes:
//
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

void CodeGen::Mangler::Append(Type ty) {
    struct Visitor {
        Mangler& M;

        void ElemTy(StringRef s, SingleElementTypeBase* t) {
            M.name += s;
            t->elem()->visit(*this);
        }

        void operator()(TemplateType*) { Unreachable("Mangling dependent type?"); }
        void operator()(SliceType* sl) { ElemTy("S", sl); }
        void operator()(ArrayType* arr) { ElemTy(std::format("A{}", arr->dimension()), arr); }
        void operator()(ReferenceType* ref) { ElemTy("R", ref); }
        void operator()(IntType* i) { M.name += std::format("I{}", i->bit_width().bits()); }
        void operator()(BuiltinType* b) {
            switch (b->builtin_kind()) {
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                case BuiltinKind::Deduced:
                case BuiltinKind::Type:
                case BuiltinKind::UnresolvedOverloadSet:
                    Unreachable("Can’t mangle this: {}", b->print(M.CG.M.context().use_colours()));
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
            for (auto p : proc->params()) p->visit(*this);
            M.name += "E";
        }
    };

    ty->visit(Visitor{*this});
}

auto CodeGen::MangledName(ProcDecl* proc) -> StringRef {
    Assert(not proc->is_template(), "Mangling template?");
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

    return mangled_names[proc] = std::move(name);
}

// ============================================================================
//  Type Conversion
// ============================================================================
auto CodeGen::ConvertTypeImpl(Type ty) -> llvm::Type* {
    switch (ty->kind()) {
        case TypeBase::Kind::SliceType: return SliceTy;
        case TypeBase::Kind::ReferenceType: return PtrTy;
        case TypeBase::Kind::ProcType: return ConvertProcType(cast<ProcType>(ty).ptr());
        case TypeBase::Kind::IntType: return builder.getIntNTy(u32(cast<IntType>(ty)->bit_width().bits()));

        case TypeBase::Kind::ArrayType: {
            auto arr = cast<ArrayType>(ty);
            auto elem = ConvertType(arr->elem());
            return llvm::ArrayType::get(elem, u64(arr->dimension()));
        }

        case TypeBase::Kind::BuiltinType: {
            switch (cast<BuiltinType>(ty)->builtin_kind()) {
                case BuiltinKind::Deduced:
                case BuiltinKind::Dependent:
                case BuiltinKind::ErrorDependent:
                    Unreachable("Dependent type in codegen?");

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

        case TypeBase::Kind::TemplateType: Unreachable("TemplateType in codegen?");
    }

    Unreachable("Unknown type kind");
}

auto CodeGen::ConvertTypeForMem(Type ty) -> llvm::Type* {
    // Convert procedure types to closures.
    if (isa<ProcType>(ty)) return ClosureTy;
    return ConvertType(ty);
}

auto CodeGen::ConvertProcType(ProcType* ty) -> llvm::FunctionType* {
    // Easy case, we can do what we want here.
    // TODO: hard case: implement the C ABI.
    // if (ty->cconv() == CallingConvention::Source) {
    auto ret = ConvertType(ty->ret());
    SmallVector<llvm::Type*> args;
    for (auto p : ty->params()) args.push_back(ConvertTypeForMem(p));
    return llvm::FunctionType::get(ret, args, ty->variadic());
    //}
}

// ============================================================================
//  CG
// ============================================================================
CodeGen::CodeGen(TranslationUnit& M)
    : M{M},
      llvm{std::make_unique<llvm::Module>(M.name, M.llvm_context)},
      builder{M.llvm_context},
      IntTy{builder.getInt64Ty()},
      I1Ty{builder.getInt1Ty()},
      I8Ty{builder.getInt8Ty()},
      PtrTy{builder.getPtrTy()},
      FFIIntTy{llvm::Type::getIntNTy(M.llvm_context, 32)}, // FIXME: Get size from target.
      SliceTy{llvm::StructType::get(PtrTy, IntTy)},
      ClosureTy{llvm::StructType::get(PtrTy, PtrTy)},
      VoidTy{builder.getVoidTy()} {}

void CodeGen::Emit() {
    // Emit procedures.
    for (auto& p : M.procs) {
        locals.clear();
        EmitProcedure(p);
    }

    // Emit module description.
    if (M.is_module) {
        SmallVector<char, 0> md;
        M.serialise(md);
        auto name = constants::ModuleDescriptionSectionName(M.name);
        auto data = llvm::ConstantDataArray::get(M.llvm_context, md);
        auto var = new llvm::GlobalVariable(
            *llvm,
            data->getType(),
            true,
            llvm::GlobalValue::PrivateLinkage,
            data,
            name
        );

        var->setSection(name);
        var->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
        var->setVisibility(llvm::GlobalValue::DefaultVisibility);
    }
}

auto CodeGen::Emit(Stmt* stmt) -> Value* {
    Assert(not stmt->dependent(), "Cannot emit dependent statement");
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

auto CodeGen::EmitAssertExpr(AssertExpr* expr) -> Value* {
    auto loc = expr->location().seek_line_column(M.context());
    if (not loc) {
        ICE(expr->location(), "No location for assert");
        return {};
    }

    If(Emit(expr->cond), [&] {
        // Emit failure handler and args.
        auto failure_handler = DeclareAssertFailureHandler();
        auto file = GetStringSlice(M.context().file(expr->location().file_id)->name());
        auto line = MakeInt(loc->line);
        auto col = MakeInt(loc->col);
        auto cond_str = GetStringSlice(expr->cond->location().text(M.context()));

        // Emit the message if there is one.
        Value* msg{};
        if (auto m = expr->message.get_or_null()) msg = Emit(m);
        else msg = llvm::ConstantAggregateZero::get(SliceTy);

        // Call it.
        builder.CreateCall(failure_handler, {file, line, col, cond_str, msg});
        builder.CreateUnreachable();
    });

    return {};
}

auto CodeGen::EmitBinaryExpr(BinaryExpr* expr) -> Value* {
    switch (expr->op) {
        // Convert 'x and y' to 'if x then y else false'.
        case Tk::And: {
            return If(
                Emit(expr->lhs),
                [&] { return Emit(expr->rhs); },
                [&] { return builder.getFalse(); }
            );
        }

        // Convert 'x or y' to 'if x then true else y'.
        case Tk::Or: {
            return If(
                Emit(expr->lhs),
                [&] { return builder.getTrue(); },
                [&] { return Emit(expr->rhs); }
            );
        }

        default: return EmitArithmeticOrComparisonOperator(
            expr->op,
            Emit(expr->lhs),
            Emit(expr->rhs),
            expr->location()
        );
    }
}

auto CodeGen::EmitArithmeticOrComparisonOperator(Tk op, Value* lhs, Value* rhs, Location loc) -> Value* {
    using enum OverflowBehaviour;
    auto ty = rhs->getType();

    auto CheckDivByZero = [&] {
        auto check = builder.CreateICmpEQ(rhs, ConstantInt::get(ty, 0));
        CreateArithFailure(check, op, loc, "division by zero");
    };

    auto CreateCheckedBinop = [&](auto unchecked_op, Intrinsic::ID id) -> Value* {
        if (not M.lang_opts().OverflowChecks) return std::invoke(
            unchecked_op,
            builder,
            lhs,
            rhs,
            ""
        );

        // LLVM has intrinsics that can check for overflow.
        auto call = builder.CreateBinaryIntrinsic(id, lhs, rhs);
        auto overflow = builder.CreateExtractValue(call, 1);
        CreateArithFailure(overflow, op, loc);
        return builder.CreateExtractValue(call, 0);
    };

    switch (op) {
        default: Todo("Codegen for '{}'", op);

        // 'and' and 'or' require lazy evaluation and are handled elsewhere.
        case Tk::And:
        case Tk::Or:
            Unreachable("'and' and 'or' cannot be handled here.");

        // Comparison operators.
        case Tk::ULt: return builder.CreateICmpULT(lhs, rhs, "ult");
        case Tk::UGt: return builder.CreateICmpUGT(lhs, rhs, "ugt");
        case Tk::ULe: return builder.CreateICmpULE(lhs, rhs, "ule");
        case Tk::UGe: return builder.CreateICmpUGE(lhs, rhs, "uge");
        case Tk::SLt: return builder.CreateICmpSLT(lhs, rhs, "slt");
        case Tk::SGt: return builder.CreateICmpSGT(lhs, rhs, "sgt");
        case Tk::SLe: return builder.CreateICmpSLE(lhs, rhs, "sle");
        case Tk::SGe: return builder.CreateICmpSGE(lhs, rhs, "sge");
        case Tk::EqEq: return builder.CreateICmpEQ(lhs, rhs, "eq");
        case Tk::Neq: return builder.CreateICmpNE(lhs, rhs, "ne");

        // Arithmetic operators that wrap or can’t overflow.
        case Tk::PlusTilde: return builder.CreateAdd(lhs, rhs);
        case Tk::MinusTilde: return builder.CreateSub(lhs, rhs);
        case Tk::StarTilde: return builder.CreateMul(lhs, rhs);
        case Tk::ShiftRight: return builder.CreateAShr(lhs, rhs);
        case Tk::ShiftRightLogical: return builder.CreateLShr(lhs, rhs);
        case Tk::Ampersand: return builder.CreateAnd(lhs, rhs);
        case Tk::VBar: return builder.CreateOr(lhs, rhs);
        case Tk::Xor: return builder.CreateXor(lhs, rhs);

        // Arithmetic operators for which there is an intrinsic
        // that can perform overflow checking.
        case Tk::Plus: return CreateCheckedBinop(
            &IRBuilder<>::CreateNSWAdd,
            Intrinsic::sadd_with_overflow
        );

        case Tk::Minus: return CreateCheckedBinop(
            &IRBuilder<>::CreateNSWSub,
            Intrinsic::ssub_with_overflow
        );

        case Tk::Star: return CreateCheckedBinop(
            &IRBuilder<>::CreateNSWMul,
            Intrinsic::smul_with_overflow
        );

        // Division only requires a check for division by zero.
        case Tk::ColonSlash:
        case Tk::ColonPercent: {
            CheckDivByZero();
            return op == Tk::ColonSlash ? builder.CreateUDiv(lhs, rhs) : builder.CreateURem(lhs, rhs);
        }

        // Signed division additionally has to check for overflow, which
        // happens only if we divide INT_MIN by -1.
        case Tk::Slash:
        case Tk::Percent: {
            CheckDivByZero();
            auto int_min = ConstantInt::get(ty, APInt::getSignedMinValue(ty->getIntegerBitWidth()));
            auto minus_one = ConstantInt::get(ty, u64(-1));
            auto check_lhs = builder.CreateICmpEQ(lhs, int_min);
            auto check_rhs = builder.CreateICmpEQ(rhs, minus_one);
            CreateArithFailure(builder.CreateAnd(check_lhs, check_rhs), op, loc);
            return op == Tk::Slash ? builder.CreateSDiv(lhs, rhs) : builder.CreateSRem(lhs, rhs);
        }

        // Left shift overflows if the shift amount is equal
        // to or exceeds the bit width.
        case Tk::ShiftLeftLogical: {
            auto check = builder.CreateICmpUGE(rhs, MakeInt(lhs->getType()->getIntegerBitWidth()));
            CreateArithFailure(check, op, loc, "shift amount exceeds bit width");
            return builder.CreateShl(lhs, rhs);
        }

        // Signed left shift additionally does not allow a sign change.
        case Tk::ShiftLeft: {
            auto check = builder.CreateICmpUGE(rhs, MakeInt(lhs->getType()->getIntegerBitWidth()));
            CreateArithFailure(check, op, loc, "shift amount exceeds bit width");

            // Check sign.
            auto res = builder.CreateShl(lhs, rhs);
            auto sign = builder.CreateAShr(lhs, MakeInt(lhs->getType()->getIntegerBitWidth() - 1));
            auto new_sign = builder.CreateAShr(res, MakeInt(lhs->getType()->getIntegerBitWidth() - 1));
            auto sign_change = builder.CreateICmpNE(sign, new_sign);
            CreateArithFailure(sign_change, op, loc);
            return res;
        }

        // This is lowered to a call to a compiler-generated function.
        case Tk::StarStar: {
            auto func = DefineExp(ty);
            return builder.CreateCall(func, {lhs, rhs});
        }
    }
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> Value* {
    Value* ret = nullptr;
    for (auto s : expr->stmts()) {
        // Initialise variables.
        if (auto var = dyn_cast<LocalDecl>(s)) {
            switch (var->type->value_category()) {
                case ValueCategory::MRValue: Todo("Initialise mrvalue");
                case ValueCategory::LValue: Todo("Initialise lvalue");
                case ValueCategory::DValue: Unreachable("Dependent value in codegen?");
                case ValueCategory::SRValue: {
                    // SRValues are simply constructed and stored.
                    if (auto i = var->init.get_or_null()) builder.CreateStore(Emit(i), locals[var]);

                    // Or zero-initialised if there is no initialiser.
                    else builder.CreateStore(llvm::Constant::getNullValue(ConvertTypeForMem(var->type)), locals[var]);
                } break;
            }
        }

        // Can’t emit other declarations here.
        if (isa<Decl>(s)) continue;

        // Emit statement.
        auto val = Emit(s);
        if (s == expr->return_expr()) ret = val;
    }
    return ret;
}

auto CodeGen::EmitBoolLitExpr(BoolLitExpr* stmt) -> Value* {
    return builder.getInt1(stmt->value);
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> Value* {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = llvm->getOrInsertFunction("printf", llvm::FunctionType::get(FFIIntTy, {PtrTy}, true));
            for (auto a : expr->args()) {
                if (a->type == M.StrLitTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto str_format = GetStringPtr("%.*s");
                    auto slice = Emit(a);
                    auto data = builder.CreateExtractValue(slice, 0);
                    auto size = builder.CreateZExtOrTrunc(builder.CreateExtractValue(slice, 1), FFIIntTy);
                    builder.CreateCall(printf, {str_format, size, data});
                }

                else if (a->type == Types::IntTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto int_format = GetStringPtr("%" PRId64);
                    auto val = Emit(a);
                    builder.CreateCall(printf, {int_format, val});
                }

                else if (a->type == Types::BoolTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto bool_format = GetStringPtr("%s");
                    auto val = Emit(a);
                    auto str = builder.CreateSelect(val, GetStringPtr("true"), GetStringPtr("false"));
                    builder.CreateCall(printf, {bool_format, str});
                }

                else {
                    ICE(
                        a->location(),
                        "Sorry, can’t print this type yet: {}",
                        a->type.print(M.context().use_colours())
                    );
                }
            }
            return nullptr;
        }
    }

    Unreachable("Unknown builtin");
}

auto CodeGen::EmitCallExpr(CallExpr* expr) -> Value* {
    // FIXME: Handle C calling convention properly for small integers and structs.
    // Emit args and callee.
    SmallVector<Value*> args;
    auto proc_ty = cast<ProcType>(expr->callee->type);
    for (auto arg : expr->args()) args.push_back(Emit(arg));

    // Callee is always a closure, even if it is a static call to
    // a function with no environment.
    auto closure = Emit(expr->callee);
    auto callee = llvm::FunctionCallee(
        ConvertType<llvm::FunctionType>(proc_ty),
        builder.CreateExtractValue(closure, 0)
    );

    // Emit call.
    auto call = builder.CreateCall(callee, args);
    call->setCallingConv(ConvertCC(proc_ty->cconv()));
    return call;
}

auto CodeGen::EmitCastExpr(CastExpr* expr) -> Value* {
    auto val = Emit(expr->arg);
    switch (expr->kind) {
        case CastExpr::LValueToSRValue: {
            Assert(expr->arg->value_category == Expr::LValue);
            return builder.CreateLoad(ConvertTypeForMem(expr->type), val, "l2sr");
        }
    }

    Unreachable();
}

auto CodeGen::EmitClosure(ProcDecl* proc) -> llvm::Constant* {
    Assert(not proc->is_template(), "Requested address of template");
    auto callee = DeclareProcedure(proc);
    auto f = cast<llvm::Function>(callee.getCallee());
    return llvm::ConstantStruct::get(ClosureTy, {f, llvm::ConstantPointerNull::get(PtrTy)});
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> Value* {
    return EmitValue(*constant->value);
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> Value* {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> Value* {
    return ConstantInt::get(ConvertType(expr->type), expr->storage.value());
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    auto a = builder.CreateAlloca(ConvertTypeForMem(decl->type));
    locals[decl] = a;
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> Value* {
    return locals.at(expr->decl);
}

auto CodeGen::EmitOverloadSetExpr(OverloadSetExpr*) -> Value* {
    Unreachable("Emitting unresolved overload set?");
}

auto CodeGen::EmitParenExpr(ParenExpr*) -> Value* { Todo(); }

void CodeGen::EmitProcedure(ProcDecl* proc) {
    if (proc->is_template()) return;

    // Create the procedure.
    auto callee = DeclareProcedure(proc);
    curr_func = cast<llvm::Function>(callee.getCallee());
    curr_func->setCallingConv(ConvertCC(proc->proc_type()->cconv()));
    curr_func->setLinkage(ConvertLinkage(proc->linkage));

    // If it doesn’t have a body, then we’re done.
    if (not proc->body()) return;

    // Create the entry block.
    EnterFunction _(*this, curr_func);

    // Emit locals.
    for (auto l : proc->locals) EmitLocal(l);

    // Initialise parameters.
    for (auto [a, p] : vws::zip(curr_func->args(), proc->params()))
        builder.CreateStore(&a, locals.at(p));

    // Emit the body.
    Emit(proc->body().get());
}

auto CodeGen::EmitProcRefExpr(ProcRefExpr* expr) -> Value* {
    return EmitClosure(expr->decl);
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> Value* {
    auto val = expr->value.get_or_null();
    if (val) builder.CreateRet(Emit(val));
    else builder.CreateRetVoid();
    return {};
}

auto CodeGen::EmitSliceDataExpr(SliceDataExpr* expr) -> Value* {
    auto slice = Emit(expr->slice);
    auto stype = cast<SliceType>(expr->slice->type);

    // This supports both lvalues and rvalues because it’s easy to do so and
    // slices are small enough to where they may reasonably may end up in a
    // temporary.
    Assert(expr->value_category == Expr::SRValue or expr->value_category == Expr::LValue);
    if (expr->lvalue()) return builder.CreateLoad(ConvertType(stype->elem()), slice);
    return builder.CreateExtractValue(slice, 0);
}

auto CodeGen::EmitStrLitExpr(StrLitExpr* expr) -> Value* {
    return GetStringSlice(expr->value);
}

auto CodeGen::EmitTypeExpr(TypeExpr* expr) -> Value* {
    // These should only exist at compile time.
    ICE(expr->location(), "Can’t emit type expr");
    return nullptr;
}

auto CodeGen::EmitUnaryExpr(UnaryExpr*) -> Value* { Todo(); }

auto CodeGen::EmitValue(const eval::Value& val) -> llvm::Constant* { // clang-format off
    utils::Overloaded LValueEmitter {
        [&](String s) -> llvm::Constant* { return GetStringPtr(s); },
        [&](eval::Memory* m) -> llvm::Constant* {
            Assert(m->alive());
            Todo("Emit memory");
        }
    };

    utils::Overloaded V {
        [&](bool b) -> llvm::Constant* { return ConstantInt::get(I1Ty, b); },
        [&](ProcDecl* proc) -> llvm::Constant* { return EmitClosure(proc); },
        [&](std::monostate) -> llvm::Constant* { return nullptr; },
        [&](eval::TypeTag) -> llvm::Constant* { Unreachable("Cannot emit type constant"); },
        [&](const APInt& value) -> llvm::Constant* { return MakeInt(value); },
        [&](const eval::LValue& lval) -> llvm::Constant* { return lval.base.visit(LValueEmitter); },
        [&](this auto& self, const eval::Reference& ref) -> llvm::Constant* {
            auto base = self(ref.lvalue);
            return llvm::ConstantFoldGetElementPtr(
                ConvertType(ref.lvalue.base_type(M)),
                base,
                {},
                MakeInt(ref.offset)
            );
        },

        [&](this auto& Self, const eval::Slice& slice) -> llvm::Constant* {
            return llvm::ConstantStruct::getAnon(
                Self(slice.data),
                Self(slice.size)
            );
        }
    }; // clang-format on
    return val.visit(V);
}
