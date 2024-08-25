module;

#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/ConstantFold.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.codegen;
import srcc;
import srcc.token;
import srcc.constants;
using namespace srcc;
using llvm::Value;

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


auto CodeGen::DeclareProcedure(ProcDecl* proc) -> llvm::FunctionCallee {
    auto name = MangledName(proc);
    return llvm->getOrInsertFunction(name, ConvertType<llvm::FunctionType>(proc->type));
}

auto CodeGen::GetStringPtr(StringRef s) -> llvm::Constant* {
    if (auto it = strings.find(s); it != strings.end()) return it->second;
    return strings[s] = builder.CreateGlobalStringPtr(s);
}

auto CodeGen::GetStringSlice(StringRef s) -> llvm::Constant* {
    auto ptr = GetStringPtr(s);
    auto size = llvm::ConstantInt::get(IntTy, s.size());
    return llvm::ConstantStruct::getAnon({ptr, size});
}

auto CodeGen::MakeInt(const APInt& value) -> llvm::ConstantInt* {
    return llvm::ConstantInt::get(M.llvm_context, value);
}

auto CodeGen::MakeInt(u64 integer) -> llvm::ConstantInt* {
    return llvm::ConstantInt::get(IntTy, integer, true);
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

    // Emit condition and branch.
    auto cond = Emit(expr->cond);
    auto err = llvm::BasicBlock::Create(llvm->getContext(), "assert.fail");
    auto join = llvm::BasicBlock::Create(llvm->getContext(), "assert.join");
    builder.CreateCondBr(cond, err, join);
    curr_func->insert(curr_func->end(), err);

    // Emit failure handler and args.
    auto failure_handler = DeclareAssertFailureHandler();
    auto file = GetStringSlice(M.context().file(expr->location().file_id)->name());
    auto line = MakeInt(loc->line);
    auto col = MakeInt(loc->col);
    auto cond_str = GetStringSlice(expr->cond->location().text(M.context()));

    // Emit the message if there is one.
    Value* msg{};
    builder.SetInsertPoint(err);
    if (auto m = expr->message.get_or_null()) msg = Emit(m);
    else msg = llvm::ConstantAggregateZero::get(SliceTy);

    // Call it.
    builder.CreateCall(failure_handler, {file, line, col, cond_str, msg});

    // Terminate the block and insert the join block.
    builder.CreateUnreachable();
    curr_func->insert(curr_func->end(), join);
    builder.SetInsertPoint(join);
    return {};
}

auto CodeGen::EmitBinaryExpr(BinaryExpr* expr) -> Value* {
    using enum OverflowBehaviour;

    auto lhs = Emit(expr->lhs);
    auto rhs = Emit(expr->rhs);

    switch (expr->op) {
        default: Todo("Codegen for '{}'", expr->op);

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
    return llvm::ConstantInt::get(ConvertType(expr->type), expr->storage.value());
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
    auto entry = llvm::BasicBlock::Create(M.llvm_context, "entry", curr_func);
    builder.SetInsertPoint(entry);

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
        [&](bool b) -> llvm::Constant* { return llvm::ConstantInt::get(I1Ty, b); },
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
