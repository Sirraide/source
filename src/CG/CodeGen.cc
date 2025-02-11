#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Macros.hh>

#include <memory>

using namespace srcc;
using namespace srcc::cg;
using ir::Value;
using ir::Block;

void CodeGen::CreateArithFailure(Value* failure_cond, Tk op, Location loc, String name) {
    If(failure_cond, [&] {
        auto op_token = CreateString(Spelling(op));
        auto operation = CreateString(name);
        CreateAbort(constants::ArithmeticFailureHandlerName, loc, op_token, operation);
    });
}

auto CodeGen::DeclarePrintf() -> Value* {
    if (not printf) {
        printf = GetOrCreateProc(
            "printf",
            Linkage::Imported,
            ProcType::Get(
                tu,
                tu.FFIIntTy,
                {{Intent::Copy, ReferenceType::Get(tu, tu.I8Ty)}},
                CallingConvention::Native,
                true
            )
        );
    }

    return printf.value();
}


auto CodeGen::DeclareProcedure(ProcDecl* proc) -> ir::Proc* {
    auto name = MangledName(proc);
    return GetOrCreateProc(name, proc->linkage, proc->proc_type());
}

auto CodeGen::DefineExp(Type ty) -> ir::Proc* {
    // Check if we’ve emitted this before.
    auto name = std::format("__srcc_exp_i{}", ty->size(tu).bits());
    if (auto existing = GetExistingProc(name).get_or_null()) return existing;

    // If not, create it now.
    auto proc = GetOrCreateProc(
        tu.save(name),
        Linkage::Merge,
        ProcType::Get(
            tu,
            ty,
            {{Intent::In, ty}, {Intent::In, ty}}
        )
    );

    EnterProcedure _(*this, proc);

    // Values that we’ll need.
    auto minus_one = CreateInt(-1, ty);
    auto zero = CreateInt(0, ty);
    auto one = CreateInt(1, ty);
    auto args = proc->args(*this);
    auto lhs = args[0];
    auto rhs = args[1];

    // x ** 0 = 1.
    If(CreateICmpEq(rhs, zero), [&] {
        CreateReturn(one);
    });

    // If base == 0.
    If(CreateICmpEq(lhs, zero), [&] {
        // If exp < 0, then error.
        if (tu.lang_opts().overflow_checking) {
            CreateArithFailure(
                CreateICmpSLt(rhs, zero),
                Tk::StarStar,
                Location(),
                "attempting to raise 0 to a negative power"
            );
        } else {
            If(CreateICmpSLt(rhs, zero), [&] {
                CreateReturn(CreatePoison(ty));
            });
        }

        // Otherwise, return 0.
        CreateReturn(zero);
    });

    // If exp < 0.
    If(CreateICmpSLt(rhs, zero), [&] {
        // If base == -1, then return 1 if exp is even, -1 if odd.
        If(CreateICmpEq(lhs, minus_one), [&] {
            auto is_odd = CreateSICast(rhs, IntType::Get(tu, Size::Bits(1)));
            auto result = CreateSelect(is_odd, minus_one, one);
            CreateReturn(result);
        });

        // If base == 1, then return 1, otherwise 0.
        auto cmp = CreateICmpEq(lhs, one);
        CreateReturn(CreateSelect(cmp, one, zero));
    });

    // Handle overflow.
    auto min_value = CreateInt(APInt::getSignedMinValue(unsigned(ty->size(tu).bits())), ty);
    auto is_min = CreateICmpEq(lhs, min_value);
    CreateArithFailure(is_min, Tk::StarStar, Location());

    // Emit the multiplication loop.
    Loop({lhs, rhs}, [&] -> SmallVector<Value*> {
        auto val = insert_point->arg(0);
        auto exp = insert_point->arg(1);
        If(CreateICmpEq(exp, zero), [&] { CreateReturn(val); });

        // Computation (and overflow check).
        auto new_val = EmitArithmeticOrComparisonOperator(Tk::Star, val, lhs, Location());
        auto new_exp = CreateSub(exp, one);
        return {new_val, new_exp};
    });

    // No return here since we return in the loop.
    return proc;
}

auto CodeGen::EnterBlock(std::unique_ptr<Block> bb, ArrayRef<Value*> args) -> Block* {
    return EnterBlock(curr_proc->add(std::move(bb)), args);
}

auto CodeGen::EnterBlock(Block* bb, ArrayRef<Value*> args) -> Block* {
    // If there is a current block, and it is not closed, branch to the newly
    // inserted block, unless that block is the function’s entry block.
    if (insert_point and not insert_point->closed() and curr_proc->entry() != bb)
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

auto CodeGen::If(
    Value* cond,
    llvm::function_ref<Value*()> emit_then,
    llvm::function_ref<Value*()> emit_else
) -> ArrayRef<ir::Argument*> {
    Assert(emit_then and emit_else, "Both branches must return a value for this overload");
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

bool LocalNeedsAlloca(LocalDecl* local) {
    auto p = dyn_cast<ParamDecl>(local);
    if (not p) return true;
    return not p->is_rvalue_in_parameter() and not p->type->pass_by_lvalue(p->parent->cconv(), p->intent());
}

void CodeGen::Loop(
    ArrayRef<Value*> block_args,
    llvm::function_ref<SmallVector<Value*>()> emit_body
) {
    auto bb_cond = EnterBlock(CreateBlock(block_args), block_args);
    auto vals = emit_body();

    Assert(
        insert_point->closed() or block_args.size() == vals.size(),
        "Mismatched argument count in branch to loop condition"
    );

    if (not insert_point->closed()) CreateBr(bb_cond, vals);
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

    // Emit condition.
    CreateBr(bb_cond.get(), {});
    EnterBlock(std::move(bb_cond));
    CreateCondBr(emit_cond(), bb_body, bb_end);

    // Emit body.
    EnterBlock(std::move(bb_body));
    emit_body();
    CreateBr(bb_cond.get(), {});

    // Continue after the loop.
    EnterBlock(std::move(bb_end));
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

// INVARIANT: No mangling code starts with a number (this means
// mangling codes can *end* with a number without requirding a
// separator).
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

        void operator()(StructType* ty) {
            Todo();
        }
    };

    ty->visit(Visitor{*this});
}

auto CodeGen::MangledName(ProcDecl* proc) -> String {
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

    return mangled_names[proc] = tu.save(name);
}

// ============================================================================
//  Initialisation.
// ============================================================================
void CodeGen::PerformVariableInitialisation(Value* addr, Expr* init) {
    switch (init->type->value_category()) {
        case Expr::DValue: Unreachable("Dependent initialiser in codegen?");
        case Expr::LValue: Unreachable("Initialisation from lvalue?");

        // Emit + store.
        case Expr::SRValue: {
            CreateStore(Emit(init), addr);
            return;
        }

        case Expr::MRValue: {
            // Default initialiser here is a memset to 0.
            if (isa<DefaultInitExpr>(init)) {
                CreateMemZero(addr, CreateInt(init->type->size(tu).bytes()));
                return;
            }

            // Structs are otherwise constructed field by field.
            if (auto lit = dyn_cast<StructInitExpr>(init)) {
                auto s = lit->struct_type();
                for (auto [field, val] : zip(s->fields(), lit->values())) {
                    auto offs = CreatePtrAdd(
                        addr,
                        CreateInt(field->offset.bytes()),
                        true
                    );
                    PerformVariableInitialisation(offs, val);
                }
                return;
            }

            // Anything else here is nonsense.
            ICE(init->location(), "Unsupported initialiser");
            return;
        }
    }

    Unreachable();
}

// ============================================================================
//  CG
// ============================================================================
void CodeGen::Emit(ArrayRef<ProcDecl*> procs) {
    for (auto& p : procs) EmitProcedure(p);
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
    auto loc = expr->location().seek_line_column(tu.context());
    if (not loc) {
        ICE(expr->location(), "No location for assert");
        return {};
    }

    Unless(Emit(expr->cond), [&] {
        Value* msg{};
        if (auto m = expr->message.get_or_null()) msg = Emit(m);
        else msg = CreateNil(tu.StrLitTy);
        auto cond_str = CreateString(expr->cond->location().text(tu.context()));
        CreateAbort(constants::AssertFailureHandlerName, expr->location(), cond_str, msg);
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
            PerformVariableInitialisation(addr, expr->rhs);
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
            if (tu.lang_opts().overflow_checking) {
                auto size = is_slice
                              ? CreateExtractValue(range, 1)
                              : CreateInt(cast<ArrayType>(expr->lhs->type)->dimension());

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
    Assert(lhs->type() == rhs->type(), "Sema should have converted these to the same type");
    auto ty = rhs->type();

    auto CheckDivByZero = [&] {
        auto check = CreateICmpEq(rhs, CreateInt(0, ty));
        CreateArithFailure(check, op, loc, "division by zero");
    };

    auto CreateCheckedBinop = [&](
        auto (Builder::*build_unchecked)(Value*, Value*, bool) -> Value*,
        auto (Builder::*build_overflow)(Value*, Value*) -> ir::OverflowResult
    ) -> Value* {
        if (not tu.lang_opts().overflow_checking) (this->*build_unchecked)(lhs, rhs, true);
        auto [val, overflow] = (this->*build_overflow)(lhs, rhs);
        CreateArithFailure(overflow, op, loc);
        return val;
    };

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
            auto minus_one = CreateInt(-1, ty);
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

        // This is lowered to a call to a compiler-generated function.
        case Tk::StarStar: {
            auto func = DefineExp(ty);
            return CreateCall(func, {lhs, rhs});
        }
    }
}

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> Value* {
    Value* ret = nullptr;
    for (auto s : expr->stmts()) {
        // Initialise variables.
        if (auto var = dyn_cast<LocalDecl>(s)) {
            Assert(var->init, "Sema should always create an initialiser for local vars");
            PerformVariableInitialisation(locals.at(var), var->init.get());
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
    return CreateBool(stmt->value);
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> Value* {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = DeclarePrintf();
            for (auto a : expr->args()) {
                if (a->type == tu.StrLitTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto str_format = CreateString("%.*s")->data;
                    auto slice = Emit(a);
                    auto data = CreateExtractValue(slice, 0);
                    auto size = CreateSICast(CreateExtractValue(slice, 1), tu.FFIIntTy);
                    CreateCall(printf, {str_format, size, data});
                }

                else if (a->type == Types::IntTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto int_format = CreateString("%" PRId64)->data;
                    auto val = Emit(a);
                    CreateCall(printf, {int_format, val});
                }

                else if (a->type == Types::BoolTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto bool_format = CreateString("%s")->data;
                    auto val = Emit(a);
                    auto str = CreateSelect(val, CreateString("true")->data, CreateString("false")->data);
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
            if (expr->lvalue()) return CreateLoad(ReferenceType::Get(tu, stype->elem()), slice);
            return CreateExtractValue(slice, 0);
        }

        case AK::SliceSize: {
            auto slice = Emit(expr->operand);
            if (expr->lvalue()) return CreateLoad(Types::IntTy, CreatePtrAdd(slice, CreateInt(8, Types::IntTy), true)); // FIXME: Magic number.
            return CreateExtractValue(slice, 1);
        }

        case AK::TypeAlign: return CreateInt(i64(cast<TypeExpr>(expr->operand)->value->align(tu).value()));
        case AK::TypeArraySize: return CreateInt(cast<TypeExpr>(expr->operand)->value->array_size(tu).bytes());
        case AK::TypeBits: return CreateInt(cast<TypeExpr>(expr->operand)->value->size(tu).bits());
        case AK::TypeBytes: return CreateInt(cast<TypeExpr>(expr->operand)->value->size(tu).bytes());
        case AK::TypeName: return CreateString(tu.save(StripColours(cast<TypeExpr>(expr->operand)->value->print())));
    }
}

auto CodeGen::EmitCallExpr(CallExpr* expr) -> Value* {
    // Callee is evaluated first.
    auto callee = Emit(expr->callee);

    // Then the arguments.
    SmallVector<Value*> args;
    for (auto arg : expr->args()) args.push_back(Emit(arg));

    // This handles calling convention lowering etc.
    return CreateCall(callee, args);
}

auto CodeGen::EmitCastExpr(CastExpr* expr) -> Value* {
    auto val = Emit(expr->arg);
    switch (expr->kind) {
        case CastExpr::LValueToSRValue:
            Assert(expr->arg->value_category == Expr::LValue);
            return CreateLoad(expr->type, val);

        case CastExpr::Integral:
            return CreateSICast(val, expr->type);
    }

    Unreachable();
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> Value* {
    return EmitValue(*constant->value);
}

auto CodeGen::EmitDefaultInitExpr(DefaultInitExpr* stmt) -> Value* {
    Assert(stmt->type->value_category() == Expr::SRValue, "Emitting non-srvalue on its own?");
    return CreateNil(stmt->type);
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> Value* {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitIfExpr(IfExpr* stmt) -> Value* {
    auto args = If(
        Emit(stmt->cond),
        [&] { return Emit(stmt->then); },
        stmt->else_ ? [&] { return Emit(stmt->else_.get()); } : llvm::function_ref<Value*()>{}
    );
    return args.empty() ? nullptr : args.front();
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> Value* {
    return CreateInt(expr->storage.value(), expr->type);
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    if (LocalNeedsAlloca(decl)) locals[decl] = CreateAlloca(decl->type);
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> Value* {
    return locals.at(expr->decl);
}

auto CodeGen::EmitMemberAccessExpr(MemberAccessExpr* expr) -> Value* {
    Todo();
}

auto CodeGen::EmitOverloadSetExpr(OverloadSetExpr*) -> Value* {
    Unreachable("Emitting unresolved overload set?");
}

auto CodeGen::EmitParenExpr(ParenExpr* expr) -> Value* {
    return Emit(expr->expr);
}

void CodeGen::EmitProcedure(ProcDecl* proc) {
    if (proc->is_template()) return;
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
    for (auto [a, p] : zip(curr_proc->args(*this), proc->params())) {
        if (not LocalNeedsAlloca(p)) locals[p] = a;
        else CreateStore(a, locals.at(p));
    }

    // Emit the body.
    Emit(proc->body().get());
}

auto CodeGen::EmitProcRefExpr(ProcRefExpr* expr) -> Value* {
    return DeclareProcedure(expr->decl);
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> Value* {
    auto val = expr->value.get_or_null();
    CreateReturn(val ? Emit(val) : nullptr);
    return {};
}

auto CodeGen::EmitStaticIfExpr(StaticIfExpr*) -> Value* {
    Unreachable();
}

auto CodeGen::EmitStrLitExpr(StrLitExpr* expr) -> Value* {
    return CreateString(expr->value);
}

auto CodeGen::EmitStructInitExpr(StructInitExpr*) -> Value* {
    Unreachable("Emitting struct initialiser without memory location?");
}

auto CodeGen::EmitTypeExpr(TypeExpr* expr) -> Value* {
    // These should only exist at compile time.
    ICE(expr->location(), "Can’t emit type expr");
    return nullptr;
}

auto CodeGen::EmitUnaryExpr(UnaryExpr*) -> Value* { Todo(); }

auto CodeGen::EmitWhileStmt(WhileStmt* stmt) -> Value* {
    While(
        [&] { return Emit(stmt->cond); },
        [&] { Emit(stmt->body); }
    );

    return nullptr;
}

auto CodeGen::EmitValue(const eval::Value& val) -> Value* { // clang-format off
    utils::Overloaded LValueEmitter {
        [&](String s) -> Value* { return CreateString(s); },
        [&](eval::Memory* m) -> Value* {
            Assert(m->alive());
            Todo("Emit memory");
        }
    };

    utils::Overloaded V {
        [&](bool b) -> Value* { return CreateBool(b); },
        [&](ProcDecl* proc) -> Value* { return DeclareProcedure(proc); },
        [&](std::monostate) -> Value* { return nullptr; },
        [&](Type) -> Value* { Unreachable("Cannot emit type constant"); },
        [&](const APInt& value) -> Value* { return CreateInt(value, val.type()); },
        [&](const eval::LValue& lval) -> Value* { return lval.base.visit(LValueEmitter); },
        [&](this auto& self, const eval::Reference& ref) -> Value* {
            auto base = self(ref.lvalue);
            return CreatePtrAdd(
                base,
                CreateInt(ref.offset * u64(ref.lvalue.type->array_size(tu).bytes()), Types::IntTy),
                true
            );
        },

        [&](this auto& Self, const eval::Slice& slice) -> Value* {
            return CreateSlice(
                Self(slice.data),
                CreateInt(slice.size, Types::IntTy)
            );
        }
    }; // clang-format on
    return val.visit(V);
}
