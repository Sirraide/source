#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Macros.hh>

#include <clang/Basic/TargetInfo.h>

#include <memory>

using namespace srcc;
using namespace srcc::cg;
using ir::Block;
using ir::Value;

void CodeGen::CreateArithFailure(Value* failure_cond, Tk op, Location loc, String name) {
    If(failure_cond, [&] {
        auto op_token = CreateGlobalStringSlice(Spelling(op));
        auto operation = CreateGlobalStringSlice(name);
        CreateAbort(ir::AbortReason::ArithmeticError, loc, op_token, operation);
    });
}

auto CodeGen::CreateBinop(
    Value* lhs,
    Value* rhs,
    Location loc,
    Tk op,
    auto (Builder::*build_unchecked)(Value*, Value*, bool)->Value*,
    auto (Builder::*build_overflow)(Value*, Value*)->ir::OverflowResult
) {
    if (not lang_opts.overflow_checking) return (this->*build_unchecked)(lhs, rhs, true);
    auto [val, overflow] = (this->*build_overflow)(lhs, rhs);
    CreateArithFailure(overflow, op, loc);
    return val;
}

auto CodeGen::DeclarePrintf() -> Value* {
    if (not printf) {
        printf = GetOrCreateProc(
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

auto CodeGen::DeclareProcedure(ProcDecl* proc) -> ir::Proc* {
    auto ty = proc->proc_type();
    bool noreturn = ty->ret() == Type::NoReturnTy;
    bool has_zero_sized_params = any_of(ty->params(), [&](const ParamTypeData& ty) {
        return IsZeroSizedType(ty.type);
    });

    // If the procedure returns an MRValue, we need to pass a pointer
    // to construct it into.
    bool has_indirect_return = HasIndirectReturn(ty);

    // Only rebuild the procedure type if we actually need to change
    // something.
    if (
        (ty->ret() != Type::VoidTy and IsZeroSizedType(ty->ret())) or
        has_zero_sized_params or
        has_indirect_return
    ) {
        SmallVector<ParamTypeData> sized;
        u32 ir_index = 0;

        if (has_indirect_return) {
            ir_index = 1;
            sized.push_back({Intent::In, PtrType::Get(tu, ty->ret())});
        }

        if (has_indirect_return or has_zero_sized_params) {
            auto map = std::make_unique<ArgumentMapping>();
            for (auto [i, ty] : enumerate(ty->params())) {
                if (not IsZeroSizedType(ty.type)) {
                    map->map(u32(i), ir_index++);
                    sized.push_back(ty);
                }
            }
            argument_mapping[proc] = std::move(map);
        } else {
            append_range(sized, ty->params());
        }

        // Convert zero-sized return types to void, but keep noreturn so we
        // can add the appropriate attribute when converting to LLVM IR.
        auto ret = ty->ret();
        if (has_indirect_return or (not noreturn and IsZeroSizedType(ret))) ret = Type::VoidTy;
        ty = ProcType::Get(
            tu,
            ret,
            sized,
            ty->cconv(),
            ty->variadic()
        );
    }

    // Create the procedure.
    auto name = MangledName(proc);
    auto ir_proc = GetOrCreateProc(name, proc->linkage, ty);
    ir_proc->associated_decl = proc;

    // Remember the *original* return type.
    if (has_indirect_return) ir_proc->indirect_ret_type = proc->return_type();
    return ir_proc;
}

auto CodeGen::EnterBlock(std::unique_ptr<Block> bb, ArrayRef<Value*> args) -> Block* {
    return EnterBlock(curr_proc->add(std::move(bb)), args);
}

auto CodeGen::EnterBlock(Block* bb, ArrayRef<Value*> args) -> Block* {
    // If there is a current block, and it is not closed, branch to the newly
    // inserted block, unless that block is the function’s entry block.
    if (insert_point and not insert_point->closed() and not bb->is_entry())
        CreateBr(bb, args);

    // Finally, position the builder at the end of the block.
    insert_point = bb;
    return bb;
}

CodeGen::EnterProcedure::EnterProcedure(CodeGen& CG, ir::Proc* proc)
    : CG(CG), old_func(CG.curr_proc), guard{CG} {
    CG.curr_proc = proc;

    // Create the entry block if it doesn’t exist yet.
    if (proc->empty())
        CG.EnterBlock(CG.CreateBlock());
}

auto CodeGen::GetArg(ir::Proc* proc, u32 index) -> Ptr<ir::Argument> {
    if (proc->decl()) {
        auto map = argument_mapping[proc->decl()].get();
        if (map) {
            auto mapped = map->map(index);
            if (not mapped) return {};
            index = *mapped;
        }
    }

    return proc->args()[index];
}

bool CodeGen::HasIndirectReturn(ProcType* type) {
    if (IsZeroSizedType(type->ret())) return false;
    return type->ret()->rvalue_category() == ValueCategory::MRValue;
}

auto CodeGen::If(
    Value* cond,
    llvm::function_ref<Value*()> emit_then,
    llvm::function_ref<Value*()> emit_else
) -> ArrayRef<ir::Argument*> {
    if (not emit_else) {
        If(cond, emit_then);
        return {};
    }

    auto bb_then = CreateBlock();
    auto bb_else = CreateBlock();
    CreateCondBr(cond, bb_then, bb_else);

    // Emit the then block.
    EnterBlock(std::move(bb_then));
    auto then_val = emit_then();

    // Now we can create the join block since we know how many arguments it takes.
    auto bb_join = CreateBlock(then_val ? ArrayRef{then_val->type()} : ArrayRef<Type>{});

    // Branch to the join block.
    if (not insert_point->closed())
        CreateBr(bb_join.get(), then_val ? ArrayRef{then_val} : ArrayRef<Value*>{});

    // And emit the else block.
    EnterBlock(std::move(bb_else));
    auto else_val = emit_else();
    if (not insert_point->closed())
        CreateBr(bb_join.get(), else_val ? ArrayRef{else_val} : ArrayRef<Value*>{});

    // Finally, return the join block.
    auto args = bb_join->arguments();
    EnterBlock(std::move(bb_join));
    return args;
}

auto CodeGen::If(Value* cond, ArrayRef<Value*> args, llvm::function_ref<void()> emit_then) -> Block* {
    SmallVector<Type, 3> types;
    for (auto arg : args) types.push_back(arg->type());

    // Create the blocks and branch to the body.
    auto body = CreateBlock(types);
    auto join = CreateBlock();
    CreateCondBr(cond, {body, args}, {join, {}});

    // Emit the body and close the block.
    EnterBlock(std::move(body));
    emit_then();
    if (not insert_point->closed()) CreateBr(join.get(), {});

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

void CodeGen::Unless(Value* cond, llvm::function_ref<void()> emit_else) {
    // Create the blocks and branch to the body.
    auto else_ = CreateBlock();
    auto join = CreateBlock();
    CreateCondBr(cond, {join, {}}, {else_, {}});

    // Emit the body and close the block.
    EnterBlock(std::move(else_));
    emit_else();
    if (not insert_point->closed()) CreateBr(join.get(), {});

    // Add the join block.
    EnterBlock(std::move(join));
}

void CodeGen::While(
    llvm::function_ref<Value*()> emit_cond,
    llvm::function_ref<void()> emit_body
) {
    auto bb_cond = CreateBlock();
    auto bb_body = CreateBlock();
    auto bb_end = CreateBlock();
    auto cond = bb_cond.get();

    // Emit condition.
    EnterBlock(std::move(bb_cond));
    CreateCondBr(emit_cond(), bb_body, bb_end);

    // Emit body.
    EnterBlock(std::move(bb_body));
    emit_body();
    CreateBr(cond);

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
void CodeGen::EmitInitialiser(Value* addr, Expr* init) {
    Assert(not IsZeroSizedType(addr->type()), "Should have been checked before calling this");
    if (init->type->is_srvalue()) {
        Assert(init->value_category == Expr::SRValue);
        CreateStore(Emit(init), addr);
    } else {
        EmitMRValue(addr, init);
    }
}

void CodeGen::EmitMRValue(Value* addr, Expr* init) { // clang-format off
    Assert(addr, "Emitting mrvalue without address?");

    // We support treating lvalues as mrvalues.
    if (init->value_category == Expr::LValue) {
        CreateMemCopy(addr, Emit(init), CreateInt(init->type->size(tu).bytes()));
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
            auto c = CreateGlobalConstantPtr(init->type, String::CreateUnsafe(static_cast<char*>(mrv.data()), mrv.size().bytes()));
            CreateMemCopy(addr, c, CreateInt(init->type->size(tu).bytes()));
        },

        // Default initialiser here is a memset to 0.
        [&](DefaultInitExpr* e) { CreateMemZero(addr, CreateInt(e->type->size(tu).bytes())); },

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

                auto offs = CreatePtrAdd(
                    addr,
                    CreateInt(field->offset.bytes()),
                    true
                );

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

auto CodeGen::Emit(Stmt* stmt) -> Value* {
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

auto CodeGen::EmitArrayBroadcastExpr(ArrayBroadcastExpr*) -> Value* {
    Unreachable("Should only be emitted as mrvalue");
}

void CodeGen::EmitArrayBroadcast(Type elem_ty, Value* addr, u64 elements, Expr* initialiser, Location loc) {
    auto counter = CreateInt(0);
    auto dimension = CreateInt(elements);
    auto bb_cond = EnterBlock(CreateBlock(Type::IntTy), counter);
    auto bb_body = CreateBlock();
    auto bb_end = CreateBlock();

    // Condition.
    auto eq = EmitArithmeticOrComparisonOperator(Tk::EqEq, bb_cond->arg(0), dimension, loc);
    CreateCondBr(eq, bb_end, bb_body);

    // Initialisation.
    EnterBlock(std::move(bb_body));
    auto mul = CreateIMul(bb_cond->arg(0), CreateInt(elem_ty->array_size(tu).bytes()), true);
    auto ptr = CreatePtrAdd(addr, mul, true);
    EmitInitialiser(ptr, initialiser);

    // Increment.
    auto incr = CreateAdd(bb_cond->arg(0), CreateInt(1));
    CreateBr(bb_cond, incr);

    // Join.
    EnterBlock(std::move(bb_end));
}

void CodeGen::EmitArrayBroadcastExpr(ArrayBroadcastExpr* e, Value* mrvalue_slot) {
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

auto CodeGen::EmitArrayInitExpr(ArrayInitExpr*) -> Value* {
    Unreachable("Should only be emitted as mrvalue");
}

void CodeGen::EmitArrayInitExpr(ArrayInitExpr* e, Value* mrvalue_slot) {
    auto ty = cast<ArrayType>(e->type);
    bool broadcast_els = u64(ty->dimension()) - e->initialisers().size();

    // Emit each initialiser.
    auto size = CreateInt(ty->elem()->array_size(tu).bytes());
    for (auto init : e->initialisers()) {
        EmitInitialiser(mrvalue_slot, init);
        if (init != e->initialisers().back() or broadcast_els != 0) {
            mrvalue_slot = CreatePtrAdd(
                mrvalue_slot,
                size,
                true
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

auto CodeGen::EmitAssertExpr(AssertExpr* expr) -> Value* {
    auto loc = expr->location().seek_line_column(tu.context());
    if (not loc) {
        ICE(expr->location(), "No location for assert");
        return {};
    }

    Unless(Emit(expr->cond), [&] {
        Value* msg{};
        if (auto m = expr->message.get_or_null()) msg = Emit(m);
        else msg = CreateNil(tu.StrLitTy);
        auto cond_str = CreateGlobalStringSlice(expr->cond->location().text(tu.context()));
        CreateAbort(ir::AbortReason::AssertionFailed, expr->location(), cond_str, msg);
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
                [&] { return CreateBool(false); }
            )[0];
        }

        // Convert 'x or y' to 'if x then true else y'.
        case Tk::Or: {
            return If(
                Emit(expr->lhs),
                [&] { return CreateBool(true); },
                [&] { return Emit(expr->rhs); }
            )[0];
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

            bool is_slice = isa<SliceType>(expr->lhs->type);
            auto range = Emit(expr->lhs);
            auto index = Emit(expr->rhs);

            // Check that the index is in bounds.
            if (lang_opts.overflow_checking) {
                auto size = is_slice
                              ? CreateExtractValue(range, 1)
                              : CreateInt(u64(cast<ArrayType>(expr->lhs->type)->dimension()));

                CreateArithFailure(
                    CreateICmpUGe(index, size),
                    Tk::LBrack,
                    expr->location(),
                    "out of bounds access"
                );
            }

            Size elem_size = cast<SingleElementTypeBase>(expr->lhs->type)->elem()->array_size(tu);
            return CreatePtrAdd(
                is_slice ? CreateExtractValue(range, 0) : range,
                CreateIMul(CreateInt(elem_size.bytes()), index),
                true
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
            auto lvalue = Emit(expr->lhs);
            auto lhs = CreateLoad(expr->lhs->type, lvalue);
            auto rhs = Emit(expr->rhs);
            auto res = EmitArithmeticOrComparisonOperator(
                StripAssignment(expr->op),
                lhs,
                rhs,
                expr->location()
            );
            CreateStore(res, lvalue);
            return lvalue;
        }

        // Anything else.
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
    auto ty = rhs->type();
    Assert(
        (lhs->type() == ty) or (isa<PtrType>(lhs->type()) and isa<PtrType>(rhs->type())),
        "Sema should have converted these to the same type: {}, {}",
        lhs->type(),
        ty
    );

    auto CheckDivByZero = [&] {
        auto check = CreateICmpEq(rhs, CreateInt(0, ty));
        CreateArithFailure(check, op, loc, "division by zero");
    };

    auto CreateCheckedBinop = [&]( // clang-format off
        auto (Builder::*build_unchecked)(Value*, Value*, bool)->Value*,
        auto (Builder::*build_overflow)(Value*, Value*)->ir::OverflowResult
    ) -> Value* {
        return CreateBinop(lhs, rhs, loc, op, build_unchecked, build_overflow);
    }; // clang-format on

    switch (op) {
        default: Todo("Codegen for '{}'", op);

        // 'and' and 'or' require lazy evaluation and are handled elsewhere.
        case Tk::And:
        case Tk::Or:
            Unreachable("'and' and 'or' cannot be handled here.");

        // Comparison operators.
        case Tk::ULt: return CreateICmpULt(lhs, rhs);
        case Tk::UGt: return CreateICmpUGt(lhs, rhs);
        case Tk::ULe: return CreateICmpULe(lhs, rhs);
        case Tk::UGe: return CreateICmpUGe(lhs, rhs);
        case Tk::SLt: return CreateICmpSLt(lhs, rhs);
        case Tk::SGt: return CreateICmpSGt(lhs, rhs);
        case Tk::SLe: return CreateICmpSLe(lhs, rhs);
        case Tk::SGe: return CreateICmpSGe(lhs, rhs);
        case Tk::EqEq: return CreateICmpEq(lhs, rhs);
        case Tk::Neq: return CreateICmpNe(lhs, rhs);

        // Arithmetic operators that wrap or can’t overflow.
        case Tk::PlusTilde: return CreateAdd(lhs, rhs);
        case Tk::MinusTilde: return CreateSub(lhs, rhs);
        case Tk::StarTilde: return CreateIMul(lhs, rhs);
        case Tk::ShiftRight: return CreateAShr(lhs, rhs);
        case Tk::ShiftRightLogical: return CreateLShr(lhs, rhs);
        case Tk::Ampersand: return CreateAnd(lhs, rhs);
        case Tk::VBar: return CreateOr(lhs, rhs);
        case Tk::Xor: return CreateXor(lhs, rhs);

        // Arithmetic operators for which there is an intrinsic
        // that can perform overflow checking.
        case Tk::Plus: return CreateCheckedBinop(
            &Builder::CreateAdd,
            &Builder::CreateSAddOverflow
        );

        case Tk::Minus: return CreateCheckedBinop(
            &Builder::CreateSub,
            &Builder::CreateSSubOverflow
        );

        case Tk::Star: return CreateCheckedBinop(
            &Builder::CreateIMul,
            &Builder::CreateSMulOverflow
        );

        // Division only requires a check for division by zero.
        case Tk::ColonSlash:
        case Tk::ColonPercent: {
            CheckDivByZero();
            return op == Tk::ColonSlash ? CreateUDiv(lhs, rhs) : CreateURem(lhs, rhs);
        }

        // Signed division additionally has to check for overflow, which
        // happens only if we divide INT_MIN by -1.
        case Tk::Slash:
        case Tk::Percent: {
            CheckDivByZero();
            auto int_min = CreateInt(APInt::getSignedMinValue(u32(ty->size(tu).bits())), ty);
            auto minus_one = CreateInt(u64(-1), ty);
            auto check_lhs = CreateICmpEq(lhs, int_min);
            auto check_rhs = CreateICmpEq(rhs, minus_one);
            CreateArithFailure(CreateAnd(check_lhs, check_rhs), op, loc);
            return op == Tk::Slash ? CreateSDiv(lhs, rhs) : CreateSRem(lhs, rhs);
        }

        // Left shift overflows if the shift amount is equal
        // to or exceeds the bit width.
        case Tk::ShiftLeftLogical: {
            auto check = CreateICmpUGe(rhs, CreateInt(ty->size(tu).bits()));
            CreateArithFailure(check, op, loc, "shift amount exceeds bit width");
            return CreateShl(lhs, rhs);
        }

        // Signed left shift additionally does not allow a sign change.
        case Tk::ShiftLeft: {
            auto check = CreateICmpUGe(rhs, CreateInt(ty->size(tu).bits()));
            CreateArithFailure(check, op, loc, "shift amount exceeds bit width");

            // Check sign.
            auto res = CreateShl(lhs, rhs);
            auto sign = CreateAShr(lhs, CreateInt(ty->size(tu).bits() - 1));
            auto new_sign = CreateAShr(res, CreateInt(ty->size(tu).bits() - 1));
            auto sign_change = CreateICmpNe(sign, new_sign);
            CreateArithFailure(sign_change, op, loc);
            return res;
        }

        // Range expressions.
        case Tk::DotDotLess: return CreateRange(lhs, rhs);
        case Tk::DotDotEq: return CreateRange(lhs, CreateAdd(rhs, CreateInt(1, rhs->type())));

        // This is lowered to a call to a compiler-generated function.
        case Tk::StarStar: Unreachable("Sema should have converted this to a call");
    }
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> Value* {
    return EmitBlockExpr(expr, nullptr);
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr, Value* mrvalue_slot) -> Value* {
    Value* ret = nullptr;
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
            auto l = CreateAlloca(curr_proc, e->type);
            EmitMRValue(l, e);
        }

        // Otherwise, this is a regular statement or expression.
        else { Emit(s); }
    }
    return ret;
}

auto CodeGen::EmitBoolLitExpr(BoolLitExpr* stmt) -> Value* {
    return CreateBool(stmt->value);
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> Value* {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = DeclarePrintf();
            for (auto a : expr->args()) {
                if (a->type == tu.StrLitTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto str_format = CreateGlobalStringSlice("%.*s")->data;
                    auto slice = Emit(a);
                    auto data = CreateExtractValue(slice, 0);
                    auto size = CreateSICast(CreateExtractValue(slice, 1), tu.FFIIntTy);
                    CreateCall(printf, {str_format, size, data});
                }

                else if (a->type == Type::IntTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto int_format = CreateGlobalStringSlice("%" PRId64)->data;
                    auto val = Emit(a);
                    CreateCall(printf, {int_format, val});
                }

                else if (a->type == Type::BoolTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto bool_format = CreateGlobalStringSlice("%s")->data;
                    auto val = Emit(a);
                    auto str = CreateSelect(val, CreateGlobalStringSlice("true")->data, CreateGlobalStringSlice("false")->data);
                    CreateCall(printf, {bool_format, str});
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
            CreateUnreachable();
            EnterBlock(CreateBlock());
            return nullptr;
        }
    }

    Unreachable("Unknown builtin");
}

auto CodeGen::EmitBuiltinMemberAccessExpr(BuiltinMemberAccessExpr* expr) -> Value* {
    switch (expr->access_kind) {
        using AK = BuiltinMemberAccessExpr::AccessKind;

        // These support both lvalues and srvalues.
        case AK::SliceData: {
            auto slice = Emit(expr->operand);
            auto stype = cast<SliceType>(expr->operand->type);
            if (expr->operand->lvalue()) return CreateLoad(PtrType::Get(tu, stype->elem()), slice);
            return CreateExtractValue(slice, 0);
        }

        case AK::SliceSize: {
            auto slice = Emit(expr->operand);
            if (expr->operand->lvalue()) {
                auto ptr_sz = CreateInt(tu.target().ptr_size().bytes(), Type::IntTy);
                return CreateLoad(Type::IntTy, CreatePtrAdd(slice, ptr_sz, true));
            }
            return CreateExtractValue(slice, 1);
        }

        case AK::RangeStart: {
            auto range = Emit(expr->operand);
            auto elem = cast<RangeType>(expr->operand->type)->elem();
            if (expr->operand->lvalue()) return CreateLoad(elem, range);
            return CreateExtractValue(range, 0);
        }

        case AK::RangeEnd: {
            auto range = Emit(expr->operand);
            auto elem = cast<RangeType>(expr->operand->type)->elem();
            if (expr->operand->lvalue()) {
                auto offs = CreateInt(elem->array_size(tu).bytes(), Type::IntTy);
                return CreateLoad(Type::IntTy, CreatePtrAdd(range, offs, true));
            }
            return CreateExtractValue(range, 1);
        }

        case AK::TypeAlign: return CreateInt(cast<TypeExpr>(expr->operand)->value->align(tu).value().bytes());
        case AK::TypeArraySize: return CreateInt(cast<TypeExpr>(expr->operand)->value->array_size(tu).bytes());
        case AK::TypeBits: return CreateInt(cast<TypeExpr>(expr->operand)->value->size(tu).bits());
        case AK::TypeBytes: return CreateInt(cast<TypeExpr>(expr->operand)->value->size(tu).bytes());
        case AK::TypeName: return CreateGlobalStringSlice(tu.save(StripColours(cast<TypeExpr>(expr->operand)->value->print())));
        case AK::TypeMaxVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(APInt::getSignedMaxValue(u32(ty->size(tu).bits())), ty);
        }
        case AK::TypeMinVal: {
            auto ty = cast<TypeExpr>(expr->operand)->value;
            return CreateInt(APInt::getSignedMinValue(u32(ty->size(tu).bits())), ty);
        }
    }
    Unreachable();
}

auto CodeGen::EmitCallExpr(CallExpr* expr) -> Value* {
    return EmitCallExpr(expr, nullptr);
}

auto CodeGen::EmitCallExpr(CallExpr* expr, Value* mrvalue_slot) -> Value* {
    auto ty = cast<ProcType>(expr->callee->type);
    Assert(HasIndirectReturn(ty) == bool(mrvalue_slot));

    // Callee is evaluated first.
    auto callee = Emit(expr->callee);

    // Add the return value pointer first if there is one.
    SmallVector<Value*> args;
    if (mrvalue_slot) args.push_back(mrvalue_slot);

    // Evaluate the arguments and add them to the call.
    //
    // Don’t add zero-sized arguments, but take care to still emit
    // them since they may have side effects.
    for (auto arg : expr->args()) {
        auto v = Emit(arg);
        if (not IsZeroSizedType(arg->type)) args.push_back(v);
    }

    // This handles calling convention lowering etc.
    return CreateCall(callee, args);
}

auto CodeGen::EmitCastExpr(CastExpr* expr) -> Value* {
    auto val = Emit(expr->arg);
    switch (expr->kind) {
        case CastExpr::Deref:
            return val; // This is a no-op like prefix '^'.

        case CastExpr::Integral:
            return CreateSICast(val, expr->type);

        case CastExpr::LValueToSRValue:
            Assert(expr->arg->value_category == Expr::LValue);
            if (IsZeroSizedType(expr->type)) return nullptr;
            return CreateLoad(expr->type, val);

        case CastExpr::MaterialisePoisonValue:
            return CreatePoison(
                expr->value_category == Expr::SRValue
                    ? expr->type
                    : PtrType::Get(tu, Type::VoidTy)
            );
    }

    Unreachable();
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> Value* {
    return EmitValue(*constant->value);
}

auto CodeGen::EmitDefaultInitExpr(DefaultInitExpr* stmt) -> Value* {
    if (IsZeroSizedType(stmt->type)) return nullptr;
    Assert(stmt->type->rvalue_category() == Expr::SRValue, "Emitting non-srvalue on its own?");
    return CreateNil(stmt->type);
}

auto CodeGen::EmitEmptyStmt(EmptyStmt*) -> Value* {
    return nullptr;
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> Value* {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitForStmt(ForStmt* stmt) -> Value* {
    SmallVector<Value*> ranges;
    SmallVector<Type> arg_types;
    SmallVector<Value*> args;
    SmallVector<Value*> end_vals;
    auto bb_end = CreateBlock();

    // Emit the ranges in order.
    for (auto r : stmt->ranges()) {
        Assert((isa<RangeType, ArrayType>(r->type)));
        ranges.push_back(Emit(r));
    }

    // Add the enumerator.
    auto* enum_var = stmt->enum_var.get_or_null();
    if (enum_var) {
        arg_types.push_back(enum_var->type);
        args.push_back(CreateInt(0, enum_var->type));
    }

    // Collect all loop variables. The enumerator is always at index 0.
    for (auto [r, expr] : zip(ranges, stmt->ranges())) {
        if (isa<RangeType>(expr->type)) {
            auto start = CreateExtractValue(r, 0);
            auto end = CreateExtractValue(r, 1);
            arg_types.push_back(start->type());
            args.push_back(start);
            end_vals.push_back(end);
        } else if (auto a = dyn_cast<ArrayType>(expr->type)) {
            arg_types.push_back(PtrType::Get(tu, a->elem()));
            args.push_back(r);
            end_vals.push_back(CreatePtrAdd(r, CreateInt(a->size(tu).bytes()), true));
        } else {
            Unreachable("Invalid for range type: {}", expr->type);
        }
    }

    // Branch to the condition block.
    auto bb_cond = EnterBlock(CreateBlock(arg_types), args);

    // Add the loop variables to the current scope.
    auto block_args = bb_cond->arguments().drop_front(enum_var ? 1 : 0);
    if (enum_var) locals[enum_var] = bb_cond->arg(0);
    for (auto [v, a] : zip(stmt->vars(), block_args)) locals[v] = a;

    // If we have multiple ranges, break if we’ve reach the end of any one of them.
    for (auto [a, e] : zip(block_args, end_vals)) {
        auto bb_cont = CreateBlock();
        auto ne = CreateICmpNe(a, e);
        CreateCondBr(ne, bb_cont, bb_end);
        EnterBlock(std::move(bb_cont));
    }

    // Body.
    Emit(stmt->body);

    // Remove the loop variables again.
    if (enum_var) locals.erase(enum_var);
    for (auto v : stmt->vars()) locals.erase(v);

    // Emit increments for all of them.
    args.clear();
    if (enum_var) args.push_back(CreateAdd(bb_cond->arg(0), CreateInt(1, enum_var->type)));
    for (auto [expr, a] : zip(stmt->ranges(), block_args)) {
        if (isa<RangeType>(expr->type)) {
            args.push_back(CreateAdd(a, CreateInt(1, a->type())));
        } else if (auto arr = dyn_cast<ArrayType>(expr->type)) {
            auto sz = CreateInt(arr->elem()->array_size(tu).bytes());
            args.push_back(CreatePtrAdd(a, sz, true));
        } else {
            Unreachable("Invalid for range type: {}", expr->type);
        }
    }

    // Continue.
    CreateBr(bb_cond, args);
    EnterBlock(std::move(bb_end));
    return nullptr;
}

auto CodeGen::EmitIfExpr(IfExpr* stmt) -> Value* {
    auto args = If(
        Emit(stmt->cond),
        [&] { return Emit(stmt->then); },
        stmt->else_ ? [&] { return Emit(stmt->else_.get()); } : llvm::function_ref<Value*()>{}
    );
    return args.empty() ? nullptr : args.front();
}

auto CodeGen::EmitIfExpr(IfExpr* stmt, Value* mrvalue_slot) -> Value* {
    (void) If(
        Emit(stmt->cond),
        [&] { EmitMRValue(mrvalue_slot, cast<Expr>(stmt->then));  return nullptr; },
        [&] { EmitMRValue(mrvalue_slot, cast<Expr>(stmt->else_.get())); return nullptr; }
    );
    return nullptr;
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> Value* {
    return CreateInt(expr->storage.value(), expr->type);
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    if (LocalNeedsAlloca(decl)) locals[decl] = CreateAlloca(curr_proc, decl->type);
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> Value* {
    if (IsZeroSizedType(expr->type)) return nullptr;
    auto l = locals.find(expr->decl);
    if (l != locals.end()) return l->second;
    Assert(bool(lang_opts.constant_eval), "Invalid local ref outside of constant evaluation?");
    return CreateInvalidLocalReference(expr);
}

auto CodeGen::EmitLoopExpr(LoopExpr* stmt) -> Value* {
    Loop([&] { if (auto b = stmt->body.get_or_null()) Emit(b); });
    return nullptr;
}

auto CodeGen::EmitMemberAccessExpr(MemberAccessExpr* expr) -> Value* {
    auto base = Emit(expr->base);
    if (IsZeroSizedType(expr->type)) return base;
    return CreatePtrAdd(base, CreateInt(expr->field->offset.bytes()), true);
}

auto CodeGen::EmitOverloadSetExpr(OverloadSetExpr*) -> Value* {
    Unreachable("Emitting unresolved overload set?");
}

void CodeGen::EmitProcedure(ProcDecl* proc) {
    locals.clear();

    // Create the procedure.
    curr_proc = DeclareProcedure(proc);

    // If it doesn’t have a body, then we’re done.
    if (not proc->body()) return;

    // Create the entry block.
    EnterProcedure _(*this, curr_proc);

    // Emit locals.
    for (auto l : proc->locals) EmitLocal(l);

    // Initialise parameters.
    for (auto [i, p] : enumerate(proc->params())) {
        auto arg = GetArg(curr_proc, u32(i)).get_or_null();
        if (not arg) continue;
        if (not LocalNeedsAlloca(p)) locals[p] = arg;
        else CreateStore(arg, locals.at(p));
    }

    // Emit the body.
    Emit(proc->body().get());
}

auto CodeGen::EmitProcRefExpr(ProcRefExpr* expr) -> Value* {
    return DeclareProcedure(expr->decl);
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> Value* {
    if (curr_proc->indirect_ret_type) {
        auto slot = curr_proc->args().front();
        EmitInitialiser(slot, expr->value.get());
        CreateReturn();
        return nullptr;
    }

    auto val = expr->value.get_or_null();
    auto ret_val = val ? Emit(val) : nullptr;
    if (val and IsZeroSizedType(val->type)) ret_val = nullptr;
    CreateReturn(ret_val);
    return {};
}

auto CodeGen::EmitStrLitExpr(StrLitExpr* expr) -> Value* {
    return CreateGlobalStringSlice(expr->value);
}

auto CodeGen::EmitStructInitExpr(StructInitExpr* e) -> Value* {
    if (IsZeroSizedType(e->type)) {
        for (auto v : e->values()) Emit(v);
        return nullptr;
    }

    Unreachable("Emitting struct initialiser without memory location?");
}

auto CodeGen::EmitTypeExpr(TypeExpr* expr) -> Value* {
    // These should only exist at compile time.
    ICE(expr->location(), "Can’t emit type expr");
    return nullptr;
}

auto CodeGen::EmitUnaryExpr(UnaryExpr* expr) -> Value* {
    if (expr->postfix) {
        switch (expr->op) {
            default: Todo("Emit postfix '{}'", expr->op);
            case Tk::PlusPlus:
            case Tk::MinusMinus: {
                auto ptr = Emit(expr->arg);
                auto val = CreateLoad(expr->type, ptr);
                auto new_val = EmitArithmeticOrComparisonOperator(
                    expr->op == Tk::PlusPlus ? Tk::Plus : Tk::Minus,
                    val,
                    CreateInt(1, expr->type),
                    expr->location()
                );

                CreateStore(new_val, ptr);
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
            auto a = Emit(expr->arg);

            // Because of how literals are parsed, we can get into an annoying
            // corner case with this operator: e.g. if the user declares an i64
            // and attempts to initialise it with -9223372036854775808, we overflow
            // because the value 9223372036854775808 is not a valid signed integer,
            // even though -9223372036854775808 *is* valid. Be nice and special-case
            // this here.
            if (
                auto val = a->as_int(tu);
                val and *val - 1 == APInt::getSignedMaxValue(u32(a->type()->size(tu).bits()))
            ) {
                val->negate();
                return CreateInt(std::move(*val), expr->type);
            }

            // Otherwise, emit '0 - val'.
            return CreateBinop(
                CreateInt(0, expr->type),
                a,
                expr->location(),
                expr->op,
                &Builder::CreateSub,
                &Builder::CreateSSubOverflow
            );
        }
    }
}

auto CodeGen::EmitWhileStmt(WhileStmt* stmt) -> Value* {
    While(
        [&] { return Emit(stmt->cond); },
        [&] { Emit(stmt->body); }
    );

    return nullptr;
}

auto CodeGen::EmitValue(const eval::RValue& val) -> Value* { // clang-format off
    utils::Overloaded V {
        [&](bool b) -> Value* { return CreateBool(b); },
        [&](std::monostate) -> Value* { return nullptr; },
        [&](Type) -> Value* { Unreachable("Cannot emit type constant"); },
        [&](const APInt& value) -> Value* { return CreateInt(value, val.type()); },
        [&](eval::MRValue) -> Value* { return nullptr; }, // This only happens if the value is unused.
        [&](this auto& self, const eval::Range& range) -> Value* {
            return CreateRange(self(range.start), self(range.end));
        }
    }; // clang-format on
    return val.visit(V);
}

auto CodeGen::emit_stmt_as_proc_for_vm(Stmt* stmt) -> ir::Proc* {
    Assert(bool(lang_opts.constant_eval));

    // Create a return value parameter if need be.
    auto ty = stmt->type_or_void();
    bool has_indirect_return = ty->rvalue_category() == Expr::MRValue;
    ParamTypeData arg{Intent::In, PtrType::Get(tu, ty)};
    ArrayRef<ParamTypeData> args;
    if (has_indirect_return) {
        ty = Type::VoidTy;
        args = arg;
    }

    // Create a new procedure for this statement.
    auto proc = GetOrCreateProc(
        constants::VMEntryPointName,
        Linkage::Internal,
        ProcType::Get(tu, ty, args)
    );

    // Inform the procedure about whether we’re returning indirectly.
    if (has_indirect_return) proc->indirect_ret_type = Type::VoidTy;

    // Sema has already ensured that this is an initialiser, so throw it
    // into a return expression to handle MRValues.
    ReturnExpr re{dyn_cast<Expr>(stmt), stmt->location(), true};
    EnterProcedure _(*this, proc);
    Emit(isa<Expr>(stmt) ? &re : stmt);
    if (not insert_point->closed()) CreateReturn();
    return proc;
}
