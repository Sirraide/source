module;

#include <llvm/IR/ConstantFold.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <ranges>
#include <srcc/Macros.hh>

module srcc.codegen;
import srcc;
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

auto CodeGen::ConvertProcType(ProcType* ty) -> llvm::FunctionType* {
    // Easy case, we can do what we want here.
    // TODO: hard case: implement the C ABI.
    // if (ty->cconv() == CallingConvention::Source) {
    auto ret = ConvertType(ty->ret());
    SmallVector<llvm::Type*> args;
    for (auto p : ty->params()) args.push_back(ConvertType(p));
    return llvm::FunctionType::get(ret, args, ty->variadic());
    //}
}

auto CodeGen::GetString(StringRef s) -> llvm::Constant* {
    if (auto it = strings.find(s); it != strings.end()) return it->second;
    return strings[s] = builder.CreateGlobalStringPtr(s);
}

auto CodeGen::MakeInt(const APInt& value) -> llvm::ConstantInt* {
    return llvm::ConstantInt::get(M.llvm_context, value);
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

auto CodeGen::EmitBlockExpr(BlockExpr* expr) -> Value* {
    Value* ret = nullptr;
    for (auto s : expr->stmts()) {
        // Can’t emit declarations here.
        if (isa<Decl>(s)) continue;
        auto val = Emit(s);
        if (s == expr->return_expr()) ret = val;
    }
    return ret;
}

auto CodeGen::EmitBuiltinCallExpr(BuiltinCallExpr* expr) -> Value* {
    switch (expr->builtin) {
        case BuiltinCallExpr::Builtin::Print: {
            auto printf = llvm->getOrInsertFunction("printf", llvm::FunctionType::get(FFIIntTy, {PtrTy}, true));
            auto str_format = GetString("%.*s");
            auto int_format = GetString("%" PRId64);
            for (auto a : expr->args()) {
                if (a->type == M.StrLitTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto slice = Emit(a);
                    auto data = builder.CreateExtractValue(slice, 0);
                    auto size = builder.CreateZExtOrTrunc(builder.CreateExtractValue(slice, 1), FFIIntTy);
                    builder.CreateCall(printf, {str_format, size, data});
                }

                else if (a->type == Types::IntTy) {
                    Assert(a->value_category == Expr::SRValue);
                    auto val = Emit(a);
                    builder.CreateCall(printf, {int_format, val});
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
    auto callee = llvm::FunctionCallee(ConvertType<llvm::FunctionType>(proc_ty), Emit(expr->callee));
    for (auto arg : expr->args()) args.push_back(Emit(arg));

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
            return builder.CreateLoad(ConvertType(expr->type), val, "l2sr");
        }
    }

    Unreachable();
}

auto CodeGen::EmitConstExpr(ConstExpr* constant) -> llvm::Constant* {
    return EmitValue(*constant->value);
}

auto CodeGen::EmitEvalExpr(EvalExpr*) -> Value* {
    Unreachable("Should have been evaluated");
}

auto CodeGen::EmitIntLitExpr(IntLitExpr* expr) -> llvm::Constant* {
    return llvm::ConstantInt::get(ConvertType(expr->type), expr->storage.value());
}

void CodeGen::EmitLocal(LocalDecl* decl) {
    auto a = builder.CreateAlloca(ConvertType(decl->type));
    locals[decl] = a;
}

auto CodeGen::EmitLocalRefExpr(LocalRefExpr* expr) -> Value* {
    return locals.at(expr->decl);
}

auto CodeGen::EmitProcAddress(ProcDecl* proc) -> llvm::Constant* {
    Assert(not proc->is_template(), "Requested address of template");
    auto callee = llvm->getOrInsertFunction(proc->name, ConvertType<llvm::FunctionType>(proc->type));
    return cast<llvm::Function>(callee.getCallee());
}

void CodeGen::EmitProcedure(ProcDecl* proc) {
    if (proc->is_template()) return;

    // Create the procedure.
    auto callee = llvm->getOrInsertFunction(proc->name, ConvertType<llvm::FunctionType>(proc->type));
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
    return EmitProcAddress(expr->decl);
}

auto CodeGen::EmitReturnExpr(ReturnExpr* expr) -> llvm::Value* {
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
    auto ptr = GetString(expr->value.value());
    auto size = llvm::ConstantInt::get(IntTy, expr->value.size());
    return llvm::ConstantStruct::getAnon({ptr, size});
}

auto CodeGen::EmitValue(const eval::Value& val) -> llvm::Constant* { // clang-format off
    utils::Overloaded LValueEmitter {
        [&](String s) -> llvm::Constant* { return GetString(s); },
        [&](eval::Memory* m) -> llvm::Constant* {
            Assert(m->alive());
            Todo("Emit memory");
        }
    };

    utils::Overloaded V {
        [&](ProcDecl* proc) -> llvm::Constant* { return EmitProcAddress(proc); },
        [&](std::monostate) -> llvm::Constant* { return nullptr; },
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
