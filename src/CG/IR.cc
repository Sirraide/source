#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>

using namespace srcc;
using namespace srcc::cg;
using namespace srcc::cg::ir;

auto Managed::operator new(usz sz, Builder& b) -> void* {
    return b.Allocate(sz, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

Builder::Builder(TranslationUnit& tu) : tu{tu} {}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateImpl(Args&&... args) -> Inst* {
    Assert(insert_point, "No insert point");
    auto i = new (*this) InstTy(*this, LIBBASE_FWD(args)...);
    insert_point->insts.push_back(i);
    return i;
}

template <std::derived_from<Inst> InstTy, typename... Args>
auto Builder::CreateSpecialGetVal(Type val_ty, Args&&... args) -> Value* {
    return new (*this) InstValue(CreateImpl<InstTy>(LIBBASE_FWD(args)...), val_ty, 0);
}

auto Builder::Create(Op op, ArrayRef<Value*> vals) -> Inst* {
    return CreateImpl<Inst>(op, vals);
}

auto Builder::CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> Value* {
    return new (*this) InstValue(Create(op, vals), ty, 0);
}

void Builder::CreateAbort(String handler, Location loc, Value* msg1, Value* msg2)  {
    CreateImpl<AbortInst>(handler, loc, ArrayRef{msg1, msg2});
}

auto Builder::CreateAdd(Value* a, Value* b, bool nowrap) -> Value* {
    auto add = CreateAndGetVal(Op::Add, a->type(), {a, b});
    add->nowrap = nowrap;
    return add;
}

auto Builder::CreateAlloca(Type ty) -> Value* {
    return CreateSpecialGetVal<Alloca>(ReferenceType::Get(tu, ty), ty);
}

auto Builder::CreateAnd(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::And, a->type(), {a, b});
}

auto Builder::CreateAShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::AShr, a->type(), {a, b});
}

auto Builder::CreateBlock(ArrayRef<Value*> args) -> std::unique_ptr<Block> {
    SmallVector<Type, 3> types;
    for (auto arg : args) types.push_back(arg->type());
    return CreateBlock(types);
}

auto Builder::CreateBlock(ArrayRef<Type> args) -> std::unique_ptr<Block> {
    auto b = std::unique_ptr<Block>(new Block);
    append_range(b->arg_types, args);
    for (auto [i, ty] : enumerate(b->arg_types))
        b->arg_vals.push_back(new (*this) Argument(b.get(), u32(i), ty));
    return b;
}

auto Builder::CreateBlock(Proc* proc, ArrayRef<Type> args) -> Block* {
    return proc->add(CreateBlock(args));
}

auto Builder::CreateBool(bool value) -> Value* {
    return &(value ? true_val : false_val);
}

void Builder::CreateBr(Block* dest, ArrayRef<Value*> args) {
    CreateImpl<BranchInst>(nullptr, BranchTarget{dest, args}, BranchTarget{});
}

auto Builder::CreateCall(Value* callee, ArrayRef<Value*> args) -> Value* {
    SmallVector operands{callee};
    append_range(operands, args);
    return CreateAndGetVal(Op::Call, callee->type(), operands);
}

void Builder::CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block) {
    CreateImpl<BranchInst>(cond, then_block, else_block);
}

auto Builder::CreateExtractValue(Value* aggregate, u32 idx) -> Value* {
    if (auto s = dyn_cast<Slice>(aggregate)) {
        if (idx == 0) return s->data;
        if (idx == 1) return s->size;
        Unreachable("Invalid index for slice");
    }

    if (auto s = dyn_cast<SliceType>(aggregate->type().ptr())) {
        if (idx == 0) return new (*this) Extract(aggregate, 0, ReferenceType::Get(tu, s->elem()));
        if (idx == 1) return new (*this) Extract(aggregate, 1, Types::IntTy);
        Unreachable("Invalid index for slice type");
    }

    Todo("Extract a value from this type: {}", aggregate->type());
}

auto Builder::CreateInt(APInt val, Type type) -> Value* {
    if (type->size(tu) <= Size::Bits(64)) return CreateInt(i64(val.getZExtValue()), type);
    auto m = std::unique_ptr<LargeInt>(new LargeInt(std::move(val), type));
    large_ints.push_back(std::move(m));
    return large_ints.back().get();
}

auto Builder::CreateInt(i64 val, Type type) -> Value* {
    auto& i = small_ints[val];
    if (not i) i = new (*this) SmallInt(val, type);
    return i;
}

auto Builder::CreateInvalidLocalReference(LocalRefExpr* ref) -> Value* {
    // The actual type of this is a reference type since local refs are lvalues.
    auto ref_ty = ReferenceType::Get(tu, ref->type);
    return new (*this) InvalidLocalReference(ref, ref_ty);
}

auto Builder::CreateIMul(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::IMul, a->type(), {a, b});
    i->nowrap = nowrap;
    return i;
}

auto Builder::CreateICmpEq(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpEq, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpNe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpNe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpULt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpULe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpUGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpUGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSLt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSLe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLe, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGt, Types::BoolTy, {a, b});
}

auto Builder::CreateICmpSGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGe, Types::BoolTy, {a, b});
}

auto Builder::CreateLoad(Type ty, Value* ptr) -> Value* {
    return CreateSpecialGetVal<MemInst>(ty, Op::Load, ty, ty->align(tu), ptr);
}

auto Builder::CreateLShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::LShr, a->type(), {a, b});
}

void Builder::CreateMemZero(Value* addr, Value* bytes) {
    Create(Op::MemZero, {addr, bytes});
}

auto Builder::CreateNil(Type ty) -> Value* {
    return new (*this) BuiltinConstant(BuiltinConstantKind::Nil, ty);
}

auto Builder::CreateOr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Or, a->type(), {a, b});
}

auto Builder::CreatePoison(Type ty) -> Value* {
    return new (*this) BuiltinConstant(BuiltinConstantKind::Poison, ty);
}

auto Builder::CreatePtrAdd(Value* ptr, Value* offs, bool inbounds) -> Value* {
    Assert(isa<ReferenceType>(ptr->type()), "First argument to ptradd must be a pointer");
    auto i = CreateAndGetVal(Op::PtrAdd, ptr->type(), {ptr, offs});
    i->inbounds = inbounds;
    return i;
}

void Builder::CreateReturn(Value* val) {
    Create(Op::Ret, val ? ArrayRef{val} : ArrayRef<Value*>{});
}

auto Builder::CreateSAddOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SAddOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
}

auto Builder::CreateSDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SDiv, a->type(), {a, b});
}

auto Builder::CreateSelect(Value* cond, Value* then_val, Value* else_val) -> Value* {
    return CreateAndGetVal(Op::Select, then_val->type(), {cond, then_val, else_val});
}

auto Builder::CreateShl(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Shl, a->type(), {a, b});
}

auto Builder::CreateSICast(Value* i, Type to_type) -> Value* {
    auto from = i->type()->size(tu).bits();
    auto to = to_type->size(tu).bits();
    if (from == to) return i;
    if (from > to) return CreateSpecialGetVal<ICast>(to_type, Op::Trunc, to_type, i);
    return CreateSpecialGetVal<ICast>(to_type, Op::ZExt, to_type, i);
}

auto Builder::CreateSlice(Value* data, Value* size) -> Slice* {
    return new (*this) Slice(SliceType::Get(tu, cast<ReferenceType>(data->type())->elem()), data, size);
}

auto Builder::CreateSMulOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SMulOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
}

auto Builder::CreateSRem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SRem, a->type(), {a, b});
}

auto Builder::CreateSSubOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SSubOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Types::BoolTy, 1)};
}

void Builder::CreateStore(Value* val, Value* ptr) {
    CreateImpl<MemInst>(Op::Store, val->type(), val->type()->align(tu), ptr, val);
}

auto Builder::CreateString(String s) -> Slice* {
    auto data = new (*this) StringData(s, ReferenceType::Get(tu, tu.StrLitTy->elem()));
    auto size = CreateInt(i64(s.size()), Types::IntTy);
    return new (*this) Slice(tu.StrLitTy, data, size);
}

auto Builder::CreateSub(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::Sub, a->type(), {a, b});
    i->nowrap = nowrap;
    return i;
}

auto Builder::CreateUDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::UDiv, a->type(), {a, b});
}

void Builder::CreateUnreachable() {
    CreateAndGetVal(Op::Unreachable, Types::VoidTy, {});
}

auto Builder::CreateURem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::URem, a->type(), {a, b});
}

auto Builder::CreateXor(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::Xor, a->type(), {a, b});
}

auto Builder::GetOrCreateProc(String s, Linkage link, ProcType* ty) -> Proc* {
    auto& proc = procs[s.value()];
    if (proc) return proc.get();
    proc = std::unique_ptr<Proc>(new Proc(s, ty, link));
    return proc.get();
}

auto Builder::GetOrCreateProc(ProcDecl* proc, String mangled_name) -> Proc* {
    auto ir_proc = GetOrCreateProc(mangled_name, proc->linkage, proc->proc_type());
    ir_proc->associated_decl = proc;
    return ir_proc;
}

auto Builder::GetExistingProc(StringRef name) -> Ptr<Proc> {
    auto proc = procs.find(name);
    return proc == procs.end() ? nullptr : proc->second.get();
}

auto Builder::Result(Inst* i, Type ty, u32 idx) -> InstValue* {
    return new (*this) InstValue(i, ty, idx);
}

bool Block::closed() const {
    if (insts.empty()) return false;
    switch (insts.back()->opcode()) {
        case Op::Abort:
        case Op::Br:
        case Op::Ret:
        case Op::Unreachable:
            return true;

        default:
            return false;
    }
}

BranchInst::BranchInst(Builder& b, Value* cond, BranchTarget then, BranchTarget else_)
    : Inst(b, Op::Br, [&] {
          SmallVector<Value*, 16> args{};
          if (cond) args.push_back(cond);
          append_range(args, then.args);
          append_range(args, else_.args);
          return args;
      }()),
      then_args_num(u32(then.args.size())), then_block(then.dest), else_block(else_.dest) {}

Inst::Inst(Builder& b, Op op, ArrayRef<Value*> args) : arguments{args.copy(b.alloc)}, op{op} {}

bool Inst::has_multiple_results() const {
    switch (op) {
        case Op::SAddOv:
        case Op::SMulOv:
        case Op::SSubOv:
            return true;
        default:
            return false;
    }
}

auto Proc::add(std::unique_ptr<Block> b) -> Block* {
    b->p = this;
    body.push_back(std::move(b));
    return body.back().get();
}

auto Proc::args(Builder& b) -> ArrayRef<Argument*> {
    if (not arguments.empty() or ty->params().empty()) return arguments;
    for (auto [i, p] : enumerate(ty->params())) {
        Type param_ty = p.type->pass_by_lvalue(ty->cconv(), p.intent)
                          ? ReferenceType::Get(b.tu, p.type)
                          : p.type;
        arguments.push_back(new (b) Argument(this, u32(i), param_ty));
    }
    return arguments;
}

/// ====================================================================
///  Printing
/// ====================================================================
namespace {
class Printer {
public:
    TranslationUnit& tu;
    SmallUnrenderedString out;
    DenseMap<Argument*, i64> arg_ids;
    DenseMap<Block*, i64> block_ids;
    DenseMap<Inst*, i64> inst_ids;
    Proc* curr_proc{};

    Printer(Builder& b);
    void DumpInst(Inst* i);
    auto DumpValue(Value* b) -> SmallUnrenderedString;
    bool ReturnsValue(Inst* i);

    auto Id(auto& map, auto* ptr) -> i64 {
        auto it = map.find(ptr);
        return it == map.end() ? -1 : it->second;
    }
};
}

auto Builder::Dump() -> SmallUnrenderedString {
    return std::move(Printer{*this}.out);
}

Printer::Printer(Builder& b) : tu{b.tu} {
    for (auto* proc : b.procedures()) {
        curr_proc = proc;
        if (not out.empty()) out += "\n";
        out += proc->type()->print(proc->name(), true);

        // Stop if there is no body.
        if (proc->empty()) {
            out += "%1(;)\n";
            continue;
        }

        // Number all blocks and instructions.
        i64 temp = 0;
        arg_ids.clear();
        block_ids.clear();
        inst_ids.clear();

        for (auto* arg : proc->args(b)) arg_ids[arg] = temp++;
        for (const auto& [id, b] : vws::enumerate(proc->blocks())) {
            block_ids[b] = id;
            for (auto* arg : b->arguments())
                arg_ids[arg] = temp++;
            for (auto* i : b->instructions())
                if (ReturnsValue(i))
                    inst_ids[i] = temp++;
        }

        // Print the procedure body.
        out += " %1({)\n";
        for (const auto& [i, b] : enumerate(proc->blocks())) {
            out += i == 0 ? "%3(entry)%1(" : std::format("\n%3(bb{})%1(", i);
            if (not b->arguments().empty()) {
                out += std::format("({}\033)", utils::join_as(b->arguments(), [&](Argument* arg) {
                    return std::format("{} %3(\033%{})", b->argument_types()[arg->index()], arg_ids.at(arg));
                }));
            }
            out += ":)\n";
            for (auto* inst : b->instructions())
                DumpInst(inst);
        }
        out += "%1(})\n";
    }
}

void Printer::DumpInst(Inst* i) {
    out += "    %1(";
    if (ReturnsValue(i)) out += std::format("%8(\033%{}) = ", Id(inst_ids, i));
    defer { out += ")\n"; };

    auto Target = [&](Block* dest, ArrayRef<Value*> args) {
        out += std::format("%3(bb{})", Id(block_ids, dest));
        if (not args.empty()) {
            out += std::format(
                "({}\033)",
                utils::join_as(args, [&](Value* v) { return DumpValue(v); })
            );
        }
    };

    auto IntCast = [&](StringRef name) {
        auto c = cast<ICast>(i);
        out += std::format(
            "%1({}) {} to {}",
            name,
            DumpValue(c->args()[0]),
            c->cast_result_type()
        );
    };

    auto Simple = [&](StringRef name) {
        out += std::format(
            "%1({}){}{}",
            name,
            i->args().empty() ? ""sv : " "sv,
            utils::join_as(i->args(), [&](Value* v) { return DumpValue(v); } )
        );
    };

    switch (i->opcode()) {
        // Special instructions.
        case Op::Abort: {
            auto a = cast<AbortInst>(i);

            String file;
            i64 line, col;
            if (auto lc = a->location().seek_line_column(tu.context())) {
                file = tu.context().file_name(a->location().file_id);
                line = i64(lc->line);
                col = i64(lc->col);
            } else {
                file = "<builtin>";
                line = 0;
                col = 0;
            }

            out += std::format(
                "%1(abort) at %6(<{}:{}:{}>) %2({})({}\033)",
                file,
                line,
                col,
                a->handler_name(),
                utils::join_as(a->args(), [&](Value* v) { return DumpValue(v); } )
            );
        } break;

        case Op::Alloca: {
            auto a = cast<Alloca>(i);
            out += std::format("%1(alloca) {}", a->allocated_type());
        } break;

        case Op::Br: {
            auto b = cast<BranchInst>(i);
            if (b->is_conditional()) {
                out += std::format("%1(br) {} to ", DumpValue(b->cond()));
                Target(b->then(), b->then_args());
                out += " else ";
                Target(b->else_(), b->else_args());
            } else {
                out += std::format("%1(br) ");
                Target(b->then(), b->then_args());
            }
        } break;

        case Op::Call: {
            auto proc = i->args().front();
            auto args = i->args().drop_front(1);
            auto ret = cast<ProcType>(proc->type())->ret();
            if (args.empty()) {
                out += std::format("%1(call) {} {}", ret, DumpValue(proc));
            } else {
                out += std::format(
                    "%1(call) {} {}({}\033)",
                    ret,
                    DumpValue(proc),
                    utils::join_as(args, [&](Value* v) { return DumpValue(v); } )
                );
            }
        } break;

        case Op::Load: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(load) {}, {}, align %5({})",
                m->memory_type(),
                DumpValue(m->args()[0]),
                m->align()
            );
        } break;

        case Op::Store: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(store) {} to {}, {}, align %5({})",
                m->memory_type(),
                DumpValue(m->args()[0]),
                DumpValue(m->args()[1]),
                m->align()
            );
        } break;

        // Integer conversions.
        case Op::SExt: IntCast("sext"); break;
        case Op::Trunc: IntCast("trunc"); break;
        case Op::ZExt: IntCast("zext"); break;

        // Generic instructions.
        case Op::Add: Simple("add"); break;
        case Op::And: Simple("and"); break;
        case Op::AShr: Simple("ashr"); break;
        case Op::ICmpEq: Simple("icmp eq"); break;
        case Op::ICmpNe: Simple("icmp ne"); break;
        case Op::ICmpULt: Simple("icmp ult"); break;
        case Op::ICmpULe: Simple("icmp ule"); break;
        case Op::ICmpUGe: Simple("icmp uge"); break;
        case Op::ICmpUGt: Simple("icmp ugt"); break;
        case Op::ICmpSLt: Simple("icmp slt"); break;
        case Op::ICmpSLe: Simple("icmp sle"); break;
        case Op::ICmpSGe: Simple("icmp sge"); break;
        case Op::ICmpSGt: Simple("icmp sgt"); break;
        case Op::IMul: Simple("imul"); break;
        case Op::LShr: Simple("lshr"); break;
        case Op::MemZero: Simple("mem zero"); break;
        case Op::Or: Simple("or"); break;
        case Op::PtrAdd: Simple("ptradd"); break;
        case Op::Ret: Simple("ret"); break;
        case Op::SAddOv: Simple("sadd ov"); break;
        case Op::SDiv: Simple("sdiv"); break;
        case Op::Select: Simple("select"); break;
        case Op::Shl: Simple("shl"); break;
        case Op::SMulOv: Simple("smul ov"); break;
        case Op::SRem: Simple("srem"); break;
        case Op::Sub: Simple("sub"); break;
        case Op::SSubOv: Simple("ssub ov"); break;
        case Op::UDiv: Simple("udiv"); break;
        case Op::Unreachable: Simple("unreachable"); break;
        case Op::URem: Simple("urem"); break;
        case Op::Xor: Simple("xor"); break;
    }
}

bool Printer::ReturnsValue(Inst* i) {
    switch (i->opcode()) {
        case Op::Abort:
        case Op::Br:
        case Op::MemZero:
        case Op::Ret:
        case Op::Store:
        case Op::Unreachable:
            return false;

        case Op::Call: {
            auto ret = cast<ProcType>(i->args().front()->type())->ret();
            return ret != Types::VoidTy and ret != Types::NoReturnTy;
        }

        default:
            return true;
    }
}

auto Printer::DumpValue(Value* v) -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (v->kind()) {
        case Value::Kind::Argument: {
            auto arg = cast<Argument>(v);
            out += std::format("%{}(\033%{})", isa<Proc>(arg->parent()) ? '4' : '3', Id(arg_ids, arg));
        } break;

        case Value::Kind::Block:
            out += std::format("%3(bb{})", Id(block_ids, cast<Block>(v)));
            break;

        case Value::Kind::BuiltinConstant: {
            auto c = cast<BuiltinConstant>(v);
            switch (c->id) {
                case BuiltinConstantKind::True: out += "true"; break;
                case BuiltinConstantKind::False: out += "false"; break;
                case BuiltinConstantKind::Nil: out += "nil"; break;
                case BuiltinConstantKind::Poison: out += "poison"; break;
            }
        } break;

        case Value::Kind::Extract: {
            auto e = cast<Extract>(v);
            out += std::format("{}[%5({})]", DumpValue(e->aggregate()), e->index());
        } break;

        case Value::Kind::InstValue: {
            auto i = cast<InstValue>(v);
            if (i->inst()->has_multiple_results()) out += std::format("%8(\033%{}:{})", Id(inst_ids, i->inst()), i->index());
            else out += std::format("%8(\033%{})", Id(inst_ids, i->inst()));
        } break;

        case Value::Kind::InvalidLocalReference: {
            auto i = cast<InvalidLocalReference>(v);
            auto d = i->referenced_local()->decl;
            out += std::format(
                "<invalid access to %2({})::%8({})>",
                d->parent->name,
                d->name
            );
        } break;

        case Value::Kind::LargeInt: {
            auto l = cast<LargeInt>(v);
            out += std::format("{} %5({})", l->type(), l->value());
        } break;

        case Value::Kind::Proc: {
            auto p = cast<Proc>(v);
            out += std::format("%2({})", p->name());
        } break;

        case Value::Kind::Slice: {
            auto s = cast<Slice>(v);
            if (auto sz = dyn_cast<SmallInt>(s->size); sz and isa<StringData>(s->data)) {
                auto str = cast<StringData>(s->data);
                out += std::format("s%3(\"\002{}\003\")", utils::Escape(str->value().take(usz(sz->value())), true));
            } else {
                out += std::format("{} (%5({}), %5({})\033)", s->type(), DumpValue(s->data), DumpValue(s->size));
            }
        } break;

        case Value::Kind::SmallInt: {
            auto s = cast<SmallInt>(v);
            out += std::format("{} %5({})", v->type(), s->value());
        } break;

        case Value::Kind::StringData: {
            auto str = cast<StringData>(v);
            out += std::format("&%3(\"\002{}\003\")", utils::Escape(str->value(), true));
        } break;
    }
    return out;
}
