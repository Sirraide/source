#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>

#include <llvm/ADT/BitVector.h>

#include <srcc/CG/ABI.hh>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include "TargetDefs.hh"

// FIXME: Get rid of all of this code once LLVM’s ABI lowering library is available upstream.

using namespace srcc;
using namespace srcc::cg;
using mlir::Value;
using mlir::LLVM::LLVMDialect;

namespace {
enum class Class {
    INTEGER,
    SSE,
    SSEUP,
    X87,
    X87UP,
    COMPLEX_X87,
    NO_CLASS,
    MEMORY,
};


[[maybe_unused]] constexpr u32 MaxRegs = 6;
[[maybe_unused]] constexpr Size Eightbyte = Size::Bits(64);

using enum Class;
using Classification = SmallVector<Class, 8>;

/// Context used to convert an entire argument list.
class LoweringContext {
    LIBBASE_MOVE_ONLY(LoweringContext);
    static constexpr unsigned MaxRegs = 6;

    // There is no correlation between this value and how many IR arguments
    // a procedure has; do *not* attempt to use this for anything other than
    // tracking ABI requirements.
    unsigned regs = 0;

public:
    LoweringContext() = default;

    /// Attempt to allocate 'n' argument registers.
    bool allocate(unsigned n = 1) {
        if (regs + n > MaxRegs) return false;
        regs += n;
        return true;
    }
};

struct ABIImpl final : abi::ABI {
    bool can_use_return_value_directly(
        CodeGen& cg,
        Type ty
    ) const override;

    auto lower_direct_return(
        CodeGen& cg,
        mlir::Location l,
        Expr* arg
    ) const -> abi::ArgInfo override;

    void lower_parameters(CodeGen& cg, ProcData& pdata) const override;

    auto lower_proc_type(
        CodeGen& cg,
        ProcType* ty,
        bool needs_environment
    ) const -> abi::CallInfo override;

    auto lower_procedure_signature(
        CodeGen& cg,
        mlir::Location l,
        ProcType* proc,
        bool needs_environment,
        mlir::Value indirect_ptr,
        mlir::Value env_ptr,
        ArrayRef<Expr*> args
    ) const -> abi::CallInfo override;

    bool needs_indirect_return(CodeGen& cg, Type ty) const override;
    bool pass_in_parameter_by_reference(CodeGen& cg, Type ty) const override;
    auto write_call_results_to_mem(abi::IRToSourceConversionContext& ctx) const -> mlir::Value override;

    /// Lower a single argument that is passed by value.
    auto LowerByValArgOrReturn(
        CodeGen& cg,
        LoweringContext& ctx,
        mlir::Location l,
        Ptr<Expr> arg,
        Type t
    ) const -> abi::ArgInfo;

    /// Take a bundle of IR arguments that represent a value that has been passed
    /// in one or more registers, write it to a memory address, and return that
    /// address; if this value was actually passed on the stack, the stack address
    /// is returned directly. Otherwise, a new variable is allocated via the 'vals'
    /// object.
    auto WriteByValParamOrReturnToMemory(
        LoweringContext& lowering,
        abi::IRToSourceConversionContext& ctx
    ) const -> mlir::Value;
};

struct Impl final : Target {
    Impl(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI)
        : Target(std::move(TI), std::make_unique<ABIImpl>()) {}
};
}

[[maybe_unused]] static auto ClassifyEightbytes(const Target& t, Type ty) -> Classification {
    // If the size of an object is larger than eight eightbytes [...] it has class MEMORY'
    //
    // Additionally, the following rule applies aggregate that isn’t an XMM/YMM/ZMM register:
    // 'If the size of the aggregate exceeds two eightbytes [...] the whole argument is passed
    // in memory.'
    //
    // We don’t have floats at the moment, so short-circuit that here.
    auto sz = ty->memory_size(t);
    if (sz > Eightbyte * 2) return {MEMORY};

    // Integer types and pointers are INTEGER. >64 bit integers are split into
    // 8-byte chunks, each of which are INTEGER.
    if (ty == Type::BoolTy or ty == Type::IntTy or isa<PtrType>(ty)) return {INTEGER};
    if (auto i = dyn_cast<IntType>(ty)) {
        if (i->bit_width() <= Eightbyte) return {INTEGER};
        if (i->bit_width() <= Eightbyte * 2) return {INTEGER, INTEGER};
        return {MEMORY};
    }

    // For arrays, since every element has the same type, we only need to classify them once.
    if (auto a = dyn_cast<ArrayType>(ty)) {
        // Shortcut if this is an array of memory elements.
        auto cls = ClassifyEightbytes(t, a->elem());
        if (cls.front() == MEMORY) return {MEMORY};

        // If this is a one-element array, it collapses to the containing class.
        if (a->dimension() == 1) return cls;

        // Because of how arrays work, there is no way that this isn’t passed in memory
        // if it’s larger than 128 bytes (the only objects larger than that that can be
        // passed in registers are YMM and ZMM registers; specifically the requirement
        // is that only the first eightbyte has class SSE and all other eightbytes class
        // SSEUP; since the nature of an array means that we will have multiple eightbytes
        // with class SSE, this rule does not apply here).
        //
        // Note: This check is technically redundant at the moment, but it won’t be once
        // we actually support e.g. __m256 because then the same check further up will have
        // to be removed.
        if (sz > Eightbyte * 2) return {MEMORY};

        // The only other option therefore is to split the array into word-sized chunks.
        if (sz > Eightbyte) return {INTEGER, INTEGER};
        return {INTEGER};
    }

    // For structs, we’re supposed to classify the fields recursively.
    //
    // However, this step, too, can be short-circuited because we don’t
    // support floats yet. (In conclusion, I’m *really* not looking forward
    // to adding support for floats—at least not YMM/XMM/ZMM registers...)
    if (sz > Eightbyte) return {INTEGER, INTEGER};
    return {INTEGER};
}

static auto GetElemTyAttrsForPointee(
    CodeGen& cg,
    Type pointee
) -> std::array<mlir::NamedAttribute, 2> {
    return {
        cg.getNamedAttr(
            LLVMDialect::getDereferenceableAttrName(),
            cg.getI64IntegerAttr(i64(pointee->memory_size(cg.translation_unit()).bytes()))
        ),
        cg.getNamedAttr(
            LLVMDialect::getAlignAttrName(),
            cg.getI64IntegerAttr(i64(pointee->align(cg.translation_unit()).value().bytes()))
        ),
    };
};

auto target::CreateX86_64_Linux(
    llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI
) -> std::unique_ptr<Target> {
    return std::make_unique<Impl>(std::move(TI));
}

auto ABIImpl::LowerByValArgOrReturn(
    CodeGen& cg,
    LoweringContext& ctx,
    mlir::Location l,
    Ptr<Expr> arg,
    Type t
) const -> abi::ArgInfo {
    static constexpr Size Word = Size::Bits(64);
    abi::ArgInfo info;
    auto sz = t->bit_width(cg.translation_unit());
    auto AddStackArg = [&] (mlir::Type arg_ty = {}) {
        if (not arg_ty) arg_ty = cg.ConvertToByteArrayType(t);
        info.emplace_back(cg.get_ptr_ty()).add_byval(arg_ty);
        if (auto a = arg.get_or_null()) {
            auto addr = cg.EmitToMemory(l, a);

            // Take care not to modify the original object if we’re passing by value.
            if (a->is_lvalue()) {
                info[0].value = cg.CreateAlloca(l, t);
                cg.CreateMemCpy(l, info[0].value, addr, t);
            } else {
                info[0].value = addr;
            }
        }
    };

    auto PassThrough = [&] {
        ctx.allocate();
        auto ty = cg.C(t);
        info.emplace_back(ty);
        if (auto a = arg.get_or_null()) {
            info[0].value = cg.EmitScalar(a);
            if (a->is_lvalue()) {
                info[0].value = cg.CreateLoad(
                    l,
                    info[0].value,
                    ty,
                    a->type->align(cg.translation_unit())
                );
            }
        }
    };

    // Transparent optionals behave like their underlying type; non-transparent
    // optionals are aggregates and are handled below.
    if (auto opt = dyn_cast<OptionalType>(t); opt and opt->has_transparent_layout())
        t = opt->elem();

    // Small aggregates are passed in registers.
    if (t->is_aggregate()) {
        auto LoadWord = [&](Value addr, Size wd) -> Value {
            return cg.CreateLoad(
                l,
                addr,
                cg.IntTy(wd.as_bytes()),
                Align(wd.as_bytes())
            );
        };

        // This is passed in a single register. Structs that are this small are never
        // annotated with 'byval' for some reason.
        if (sz <= Word) {
            ctx.allocate();
            auto& a = info.emplace_back(cg.IntTy(sz.as_bytes()));
            if (arg) a.value = LoadWord(cg.EmitToMemory(l, arg.get()), sz);
        }

        // This is passed in two registers.
        else if (sz <= Word * 2 and ctx.allocate(2)) {
            // As an optimisation, pass closures and slices directly.
            if (isa<SliceType, ProcType>(t)) {
                auto ty = cg.GetEquivalentRecordTypeForAggregate(t);
                info.emplace_back(cg.C(ty->layout().fields()[0]->type));
                info.emplace_back(cg.C(ty->layout().fields()[1]->type));
                if (auto a = arg.get_or_null()) {
                    auto v = cg.Emit(a);
                    if (a->is_lvalue()) v = cg.CreateLoad(l, v.scalar(), t);
                    info[0].value = v.first();
                    info[1].value = v.second();
                }
            }

            // Other aggregates (including ranges) are more complex; just load them
            // from memory in chunks.
            else {
                // TODO: This loads padding bytes if the struct is e.g.
                // (i32, i64); do we care?
                info.emplace_back(cg.getI64Type());
                info.emplace_back(cg.IntTy(sz - Word));
                if (arg) {
                    auto addr = cg.EmitToMemory(l, arg.get());
                    info[0].value = LoadWord(addr, Word);
                    info[1].value = LoadWord(cg.CreatePtrAdd(l, addr, Word), sz - Word);
                }
            }
        }

        // This is passed on the stack.
        else { AddStackArg(); }
    }

    // For integers, it depends on the bit width.
    else if (t->is_integer_or_bool() or isa<EnumType>(t)) {
        // i65-i127 are passed in two registers.
        if (sz > Word and sz < Word * 2) {
            if (ctx.allocate(2)) {
                info.emplace_back(cg.getI64Type());
                info.emplace_back(cg.getI64Type());
                if (auto a = arg.get_or_null()) {
                    auto mem = cg.EmitToMemory(l, a);
                    auto align = t->align(cg.translation_unit());
                    info[0].value = cg.CreateLoad(l, mem, cg.getI64Type(), align);
                    info[1].value = cg.CreateLoad(l, mem, cg.getI64Type(), align, Word);
                }
            } else {
                // For some reason Clang passes e.g. i127 as an i128 rather than
                // as an array of 16 bytes.
                AddStackArg(cg.getIntegerType(128));
            }
        }

        // i128’s ABI is apparently somewhat cursed; it is never marked as 'byval'
        // and is passed as a single value; this specifically applies only to the
        // C __i128 type and *not* _BitInt(128) for some ungodly reason; treat our
        // i128 as the former because it’s more of a builtin type.
        else if (sz == Word * 2) {
            auto i128_ty = cg.getIntegerType(128);
            ctx.allocate(2);
            info.emplace_back(i128_ty);
            if (auto a = arg.get_or_null()) {
                info[0].value = cg.CreateLoad(
                    l,
                    cg.EmitToMemory(l, a),
                    i128_ty,
                    t->align(cg.translation_unit())
                );
            }
        }

        // Any integers that are larger than i128 are passed on the stack.
        else if (sz > Word * 2) {
            AddStackArg();
        }

        // Lastly, any other integers are just passed through; extend them if
        // they don’t have their preferred size.
        else {
            PassThrough();
            auto ty = cg.C(t);
            auto pref_ty = cg.GetPreferredIntType(ty);
            if (ty != pref_ty) {
                if (t == Type::BoolTy) info[0].add_zext(cg);
                else info[0].add_sext(cg);
            }
        }
    }

    // Pointers are just passed through, and so are compile-time values.
    else if (auto ptr_ty = dyn_cast<PtrType>(t)) {
        PassThrough();
        append_range(info[0].attrs, GetElemTyAttrsForPointee(cg, ptr_ty->elem()));
        if (ptr_ty->is_immutable()) info[0].attrs.push_back(cg.getNamedAttr(
            LLVMDialect::getReadonlyAttrName(),
            cg.getUnitAttr()
        ));
    }

    // So are compile-time values.
    else if (t == Type::TreeTy or t == Type::TypeTy) {
        // We don’t really bother emitting alignment/dereferencable information for
        // these types since the IR is always interpreted (and not optimised) anyway.
        PassThrough();
    }

    // Make sure that we explicitly handle all possible type kinds.
    else {
        cg.ICE(SLoc::Decode(l), "Unsupported type in call lowering: {}", t);
    }

    // Make everything 'noundef'
    for (auto& i : info) i.attrs.emplace_back(cg.getNamedAttr(
        LLVMDialect::getNoUndefAttrName(),
        cg.getUnitAttr()
    ));

    return info;
}

auto ABIImpl::WriteByValParamOrReturnToMemory(
    LoweringContext& lowering,
    abi::IRToSourceConversionContext& vals
) const -> Value {
    static constexpr Size Word = Size::Bits(64);
    auto& cg = vals.cg();
    auto& tu = cg.translation_unit();
    Assert(not cg.IsZeroSizedType(vals.type()));
    auto sz = vals.type()->bit_width(tu);
    auto ReuseStackAddress = [&] { return vals.next(); };
    auto t = vals.type();

    // Transparent optionals behave like their underlying type; non-transparent
    // optionals are aggregates and are handled below.
    if (auto opt = dyn_cast<OptionalType>(t); opt and opt->has_transparent_layout())
        t = opt->elem();

    // Small aggregates are passed in registers.
    if (t->is_aggregate()) {
        auto StoreWord = [&](Value addr, Value v) {
            auto a = v.getType() == cg.get_ptr_ty()
                ? tu.target().ptr_align()
                : tu.target().int_align(Size::Bits(v.getType().getIntOrFloatBitWidth()));
            cg.CreateStore(
                vals.location(),
                addr,
                v,
                a
            );
        };

        // This is passed in a single register.
        if (sz <= Word and lowering.allocate()) {
            StoreWord(vals.addr(), vals.next());
            return vals.addr();
        }

        // This is passed in two registers.
        if (sz <= Word * 2 and lowering.allocate(2)) {
            StoreWord(vals.addr(), vals.next());
            StoreWord(cg.CreatePtrAdd(vals.location(), vals.addr(), Word), vals.next());
            return vals.addr();
        }

        return ReuseStackAddress();
    }

    if (t->is_integer_or_bool() or isa<EnumType>(t)) {
        // i65-i127 are passed in two registers.
        if (sz > Word and sz < Word * 2) {
            if (lowering.allocate(2)) {
                auto first = vals.next();
                auto second = vals.next();
                cg.CreateStore(
                    vals.location(),
                    vals.addr(),
                    first,
                    t->align(tu)
                );

                cg.CreateStore(
                    vals.location(),
                    vals.addr(),
                    second,
                    t->align(tu),
                    Word
                );

                return vals.addr();
            }

            return ReuseStackAddress();
        }

        // i128 is a single register.
        if (sz == Word * 2) {
            lowering.allocate(2);
            cg.CreateStore(
                vals.location(),
                vals.addr(),
                vals.next(),
                t->align(tu)
            );

            return vals.addr();
        }

        // Anything larger goes on the stack..
        if (sz > Word * 2) return ReuseStackAddress();
    }

    lowering.allocate();
    cg.CreateStore(
        vals.location(),
        vals.addr(),
        vals.next(),
        t->align(tu)
    );

    return vals.addr();
}

bool ABIImpl::can_use_return_value_directly(
    CodeGen& cg,
    Type ty
) const {
    if (auto opt = dyn_cast<OptionalType>(ty); opt and opt->has_transparent_layout())
        return can_use_return_value_directly(cg, opt->elem());

    if (isa<PtrType, SliceType, ProcType>(ty)) return true;
    if (ty == Type::BoolTy or ty == Type::TreeTy or ty == Type::TypeTy) return true;
    if (ty->is_integer()) {
        auto sz = ty->bit_width(cg.translation_unit());
        return sz <= Size::Bits(64) or sz == Size::Bits(128);
    }
    return false;
}

auto ABIImpl::lower_direct_return(
    CodeGen& cg,
    mlir::Location l,
    Expr* arg
) const -> cg::abi::ArgInfo {
    LoweringContext ctx;
    return LowerByValArgOrReturn(cg, ctx, l, arg, arg->type);
}

auto ABIImpl::lower_procedure_signature(
    CodeGen& cg,
    mlir::Location l,
    ProcType* proc,
    bool needs_environment,
    Value indirect_ptr,
    Value env_ptr,
    ArrayRef<Expr*> args
) const -> abi::CallInfo {
    static constexpr Size Word = Size::Bits(64);
    abi::CallInfo info;
    LoweringContext ctx;
    auto AddArgType = [&](mlir::Type t, ArrayRef<mlir::NamedAttribute> attrs = {}) {
        info.arg_types.push_back(t);
        info.arg_attrs.push_back(cg.getDictionaryAttr(attrs));
    };

    auto AddReturnType = [&](mlir::Type t, ArrayRef<mlir::NamedAttribute> attrs = {}) {
        info.result_types.push_back(t);
        info.result_attrs.push_back(cg.getDictionaryAttr(attrs));
    };

    auto AddByRefArg = [&](Value v, Type t) {
        ctx.allocate();
        info.args.push_back(v);
        AddArgType(cg.get_ptr_ty(), GetElemTyAttrsForPointee(cg, t));
    };

    auto AddByValArg = [&](Expr* arg, Type t) {
        for (const auto& a : LowerByValArgOrReturn(cg, ctx, l, arg, t)) {
            info.args.push_back(a.value);
            AddArgType(a.ty, a.attrs);
        }
    };

    auto Arg = [&](usz i) -> Expr* {
        if (i < args.size()) return args[i];
        return nullptr;
    };

    auto ret = proc->ret();
    if (auto opt = dyn_cast<OptionalType>(ret); opt and opt->has_transparent_layout())
        ret = opt->elem();

    auto sz = ret->bit_width(cg.translation_unit());
    if (cg.IsZeroSizedType(ret)) {
        if (ret == Type::NoReturnTy) info.no_return = true;
    }

    // Some types are returned via a store to a hidden argument pointer.
    else if (needs_indirect_return(cg, ret)) {
        ctx.allocate();
        info.args.push_back(indirect_ptr);
        AddArgType(
            cg.get_ptr_ty(),
            cg.getNamedAttr(
                LLVMDialect::getStructRetAttrName(),
                mlir::TypeAttr::get(cg.ConvertToByteArrayType(ret))
            )
        );
    }

    // Small aggregates are returned in registers.
    else if (ret->is_aggregate()) {
        if (sz <= Word) {
            AddReturnType(cg.IntTy(sz.as_bytes()));
        } else if (sz <= Word * 2) {
            // TODO: This returns padding bytes if the struct is e.g. (i32, i64); do we care?
            AddReturnType(cg.getI64Type());
            AddReturnType(cg.IntTy((sz - Word).as_bytes()));
        } else {
            Unreachable("Should never be returned directly");
        }
    }

    // i65–i127 (but *not* i128) are returned in two registers.
    else if (ret->is_integer() and sz > Word and sz < Word * 2) {
        AddReturnType(cg.getI64Type());
        AddReturnType(cg.getI64Type());
    }

    // Everything else is passed through as is.
    else {
        SmallVector<mlir::NamedAttribute, 1> attrs;

        // Extend integers that don’t have their preferred size.
        auto ty = cg.C(ret);
        if (ty.isInteger()) {
            auto pref_ty = cg.GetPreferredIntType(ty);
            if (ty != pref_ty) {
                if (ret == Type::BoolTy) attrs.push_back(cg.getNamedAttr(LLVMDialect::getZExtAttrName(), cg.getUnitAttr()));
                else attrs.push_back(cg.getNamedAttr(LLVMDialect::getSExtAttrName(), cg.getUnitAttr()));
            }
        }

        AddReturnType(ty, attrs);
    }

    // Evaluate the arguments and add them to the call.
    for (auto [i, param] : enumerate(proc->params())) {
        if (cg.IsZeroSizedType(param.type)) {
            if (auto a = Arg(i)) cg.Emit(a);
        } else if (cg.PassByReference(param.type, param.intent)) {
            Value arg;
            if (auto a = Arg(i)) arg = cg.EmitToMemory(l, a);
            AddByRefArg(arg, param.type);
        } else {
            AddByValArg(Arg(i), param.type);
        }
    }

    // Throw the environment pointer at the end.
    if (needs_environment) {
        AddArgType(cg.get_ptr_ty(), {
            cg.getNamedAttr(LLVMDialect::getNestAttrName(), cg.getUnitAttr()),
            cg.getNamedAttr(LLVMDialect::getReadonlyAttrName(), cg.getUnitAttr()),
            cg.getNamedAttr(LLVMDialect::getNoUndefAttrName(), cg.getUnitAttr()),
            cg.getNamedAttr(LLVMDialect::getNoFreeAttrName(), cg.getUnitAttr()),
        });

        if (env_ptr) info.args.push_back(env_ptr);
    }

    // Extra variadic arguments are always passed as 'copy' parameters.
    if (args.size() > proc->params().size()) {
        for (auto arg : args.drop_front(proc->params().size())) {
            Assert(not cg.IsZeroSizedType(arg->type), "Passing zero-sized type as variadic argument?");
            for (const auto& a : LowerByValArgOrReturn(cg, ctx, l, arg, arg->type))
                info.args.push_back(a.value);
        }
    }

    info.func = mlir::FunctionType::get(cg.mlir_context(), info.arg_types, info.result_types);
    return info;
}

void ABIImpl::lower_parameters(CodeGen& cg, ProcData& pdata) const {
    Assert(pdata.decl, "Not an AST-level procedure!");

    // We can’t rely on 'lowering'’s allocation counts since that tracks
    // ABI requirements, not actual argument slots, so track the latter
    // separately.
    LoweringContext lowering;
    unsigned vals_used = 0;
    if (needs_indirect_return(cg, pdata.decl->return_type())) {
        vals_used++;
        lowering.allocate();
    }

    for (auto param : pdata.decl->params()) {
        if (cg.IsZeroSizedType(param->type)) continue;
        if (cg.PassByReference(param->type, param->intent())) {
            pdata.locals[param] = pdata.proc.getArgument(vals_used++);
            lowering.allocate();
        } else {
            auto l = cg.C(param->location());
            abi::IRToSourceConversionContext ctx{cg, l, pdata.proc.getArguments().drop_front(vals_used), param->type};
            pdata.locals[param] = WriteByValParamOrReturnToMemory(lowering, ctx);
            vals_used += ctx.consumed();
        }
    }
}

auto ABIImpl::lower_proc_type(
    CodeGen& cg,
    ProcType* ty,
    bool needs_environment
) const -> abi::CallInfo {
    return lower_procedure_signature(
        cg,
        cg.getUnknownLoc(),
        ty,
        needs_environment,
        nullptr,
        nullptr,
        {}
    );
}

bool ABIImpl::needs_indirect_return(CodeGen& cg, Type ty) const {
    return ty->memory_size(cg.translation_unit()) > Size::Bits(128);
}

bool ABIImpl::pass_in_parameter_by_reference(CodeGen& cg, Type ty) const {
    return ty->memory_size(cg.translation_unit()) > Size::Bits(128);
}

auto ABIImpl::write_call_results_to_mem(
    abi::IRToSourceConversionContext& ctx
) const -> mlir::Value {
    LoweringContext lowering;
    return WriteByValParamOrReturnToMemory(lowering, ctx);
}