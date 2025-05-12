#include <srcc/AST/AST.hh>
#include <srcc/CG/IR.hh>
#include <srcc/Core/Constants.hh>

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

auto Builder::CreateAndGetVal(Op op, Type ty, ArrayRef<Value*> vals) -> InstValue* {
    return new (*this) InstValue(Create(op, vals), ty, 0);
}

void Builder::CreateAbort(AbortReason reason, Location loc, Value* msg1, Value* msg2) {
    CreateImpl<AbortInst>(reason, loc, ArrayRef{msg1, msg2});
}

auto Builder::CreateAdd(Value* a, Value* b, bool nowrap) -> Value* {
    auto add = CreateAndGetVal(Op::Add, a->type(), {a, b});
    add->inst()->nowrap = nowrap;
    return add;
}

auto Builder::CreateAlloca(Proc* parent, Type ty) -> Value* {
    Assert(parent, "Alloca without parent procedure?");
    auto a = new (*this) FrameSlot(tu, parent, ty);
    parent->frame_info.push_back(a);
    return a;
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
    return CreateAndGetVal(Op::Call, cast<ProcType>(callee->type())->ret(), operands);
}

void Builder::CreateCondBr(Value* cond, BranchTarget then_block, BranchTarget else_block) {
    Assert(cond->type() == Type::BoolTy, "Branch condition must be a bool");
    CreateImpl<BranchInst>(cond, then_block, else_block);
}

auto Builder::CreateExtractValue(Value* aggregate, u32 idx) -> Value* {
    if (auto s = dyn_cast<Slice>(aggregate)) {
        if (idx == 0) return s->data;
        if (idx == 1) return s->size;
        Unreachable("Invalid index for slice");
    }

    if (auto s = dyn_cast<Range>(aggregate)) {
        if (idx == 0) return s->start;
        if (idx == 1) return s->end;
        Unreachable("Invalid index for range");
    }

    if (auto s = dyn_cast<SliceType>(aggregate->type().ptr())) {
        if (idx == 0) return new (*this) Extract(aggregate, 0, PtrType::Get(tu, s->elem()));
        if (idx == 1) return new (*this) Extract(aggregate, 1, Type::IntTy);
        Unreachable("Invalid index for slice type");
    }

    if (auto s = dyn_cast<RangeType>(aggregate->type().ptr())) {
        if (idx == 0) return new (*this) Extract(aggregate, 0, s->elem());
        if (idx == 1) return new (*this) Extract(aggregate, 1, s->elem());
        Unreachable("Invalid index for slice type");
    }

    Todo("Extract a value from this type: {}", aggregate->type());
}

auto Builder::CreateInt(APInt val, Type type) -> Value* {
    if (type->size(tu) <= Size::Bits(64)) return CreateInt(val.getZExtValue(), type);
    auto m = std::unique_ptr<LargeInt>(new LargeInt(std::move(val), type));
    large_ints.push_back(std::move(m));
    return large_ints.back().get();
}

auto Builder::CreateInt(u64 val, Type type) -> Value* {
    return new (*this) SmallInt(val, type);
}

auto Builder::CreateInvalidLocalReference(LocalRefExpr* ref) -> Value* {
    // The actual type of this is a reference type since local refs are lvalues.
    auto ref_ty = PtrType::Get(tu, ref->type);
    return new (*this) InvalidLocalReference(ref, ref_ty);
}

auto Builder::CreateIMul(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::IMul, a->type(), {a, b});
    i->inst()->nowrap = nowrap;
    return i;
}

auto Builder::CreateICmpEq(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpEq, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpNe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpNe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpULt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpULe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpULe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpUGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpUGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpUGe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSLt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSLe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSLe, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSGt(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGt, Type::BoolTy, {a, b});
}

auto Builder::CreateICmpSGe(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::ICmpSGe, Type::BoolTy, {a, b});
}

auto Builder::CreateLoad(Type ty, Value* ptr) -> Value* {
    return CreateSpecialGetVal<MemInst>(ty, Op::Load, ty, ty->align(tu), ptr);
}

auto Builder::CreateLShr(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::LShr, a->type(), {a, b});
}

void Builder::CreateMemCopy(Value* to, Value* from, Value* bytes) {
    Create(Op::MemCopy, {to, from, bytes});
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
    Assert(isa<PtrType>(ptr->type()), "First argument to ptradd must be a pointer");

    // This is really common in struct initialisers, so optimise for it.
    if (auto lit = dyn_cast<SmallInt>(offs); lit and lit->value() == 0)
        return ptr;

    auto i = CreateAndGetVal(Op::PtrAdd, ptr->type(), {ptr, offs});
    i->inst()->inbounds = inbounds;
    return i;
}

auto Builder::CreateRange(Value* start, Value* end) -> Value* {
    return new (*this) Range(RangeType::Get(tu, start->type()), start, end);
}

void Builder::CreateReturn(Value* val) {
    Create(Op::Ret, val ? ArrayRef{val} : ArrayRef<Value*>{});
}

auto Builder::CreateSAddOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SAddOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
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
    if (i->type() == to_type) return i;
    if (from > to) return CreateSpecialGetVal<ICast>(to_type, Op::Trunc, to_type, i);
    return CreateSpecialGetVal<ICast>(to_type, Op::SExt, to_type, i);
}

auto Builder::CreateSlice(Value* data, Value* size) -> Slice* {
    return new (*this) Slice(SliceType::Get(tu, cast<PtrType>(data->type())->elem()), data, size);
}

auto Builder::CreateSMulOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SMulOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
}

auto Builder::CreateSRem(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::SRem, a->type(), {a, b});
}

auto Builder::CreateSSubOverflow(Value* a, Value* b) -> OverflowResult {
    auto i = Create(Op::SSubOv, {a, b});
    return {Result(i, a->type(), 0), Result(i, Type::BoolTy, 1)};
}

void Builder::CreateStore(Value* val, Value* ptr) {
    CreateImpl<MemInst>(Op::Store, val->type(), val->type()->align(tu), ptr, val);
}

auto Builder::CreateGlobalConstantPtr(Type ty, String s) -> GlobalConstant* {
    auto g = new (*this) GlobalConstant(s, PtrType::Get(tu, ty), ty->align(tu));
    global_consts.push_back(g);
    return g;
}

auto Builder::CreateGlobalStringSlice(String s) -> Slice* {
    auto data = CreateGlobalConstantPtr(tu.I8Ty, s);
    auto size = CreateInt(s.size(), Type::IntTy);
    data->string = true;
    return new (*this) Slice(tu.StrLitTy, data, size);
}

auto Builder::CreateSub(Value* a, Value* b, bool nowrap) -> Value* {
    auto i = CreateAndGetVal(Op::Sub, a->type(), {a, b});
    i->inst()->nowrap = nowrap;
    return i;
}

auto Builder::CreateUDiv(Value* a, Value* b) -> Value* {
    return CreateAndGetVal(Op::UDiv, a->type(), {a, b});
}

void Builder::CreateUnreachable() {
    CreateAndGetVal(Op::Unreachable, Type::VoidTy, {});
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

    // Initialise arguments.
    for (auto [i, p] : enumerate(ty->params())) {
        Type param_ty = p.type->pass_by_lvalue(ty->cconv(), p.intent)
                          ? PtrType::Get(tu, p.type)
                          : p.type;
        proc->arguments.push_back(new (*this) Argument(proc.get(), u32(i), param_ty));
    }

    return proc.get();
}

auto Builder::GetExistingProc(StringRef name) -> Ptr<Proc> {
    auto proc = procs.find(name);
    return proc == procs.end() ? nullptr : proc->second.get();
}

auto Builder::Result(Inst* i, Type ty, u32 idx) -> InstValue* {
    return new (*this) InstValue(i, ty, idx);
}

auto AbortInst::handler_name() const -> String {
    switch (abort_reason()) {
        case AbortReason::AssertionFailed: return constants::AssertFailureHandlerName;
        case AbortReason::ArithmeticError: return constants::ArithmeticFailureHandlerName;
    }
    Unreachable();
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

bool Block::is_entry() const {
    return p and p->entry() == this;
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

Inst::Inst(Builder& b, Op op, ArrayRef<Value*> args) : arguments{args.copy(b.tu.allocator())}, op{op} {}

bool Inst::has_multiple_results() const {
    return result_types().size() > 1;
}

auto Inst::result_types() const -> SmallVector<Type, 2> {
    using V = SmallVector<Type, 2>;
    switch (op) {
        case Op::Abort:
        case Op::Br:
        case Op::MemCopy:
        case Op::MemZero:
        case Op::Ret:
        case Op::Store:
        case Op::Unreachable:
            return {};

        case Op::Call: {
            auto t = cast<ProcType>(arguments[0]->type())->ret();
            return t == Type::VoidTy or t == Type::NoReturnTy ? V{} : V{t};
        }

        case Op::Load:
            return {cast<MemInst>(this)->memory_type()};

        case Op::Select:
            return {arguments[1]->type()};

        case Op::SExt:
        case Op::Trunc:
        case Op::ZExt:
            return {cast<ICast>(this)->cast_result_type()};

        case Op::SAddOv:
        case Op::SMulOv:
        case Op::SSubOv:
            return {arguments[0]->type(), Type::BoolTy};

        case Op::Add:
        case Op::And:
        case Op::AShr:
        case Op::IMul:
        case Op::LShr:
        case Op::Or:
        case Op::PtrAdd:
        case Op::SDiv:
        case Op::Shl:
        case Op::SRem:
        case Op::Sub:
        case Op::UDiv:
        case Op::URem:
        case Op::Xor:
            return {arguments[0]->type()};

        case Op::ICmpEq:
        case Op::ICmpNe:
        case Op::ICmpSGe:
        case Op::ICmpSGt:
        case Op::ICmpSLe:
        case Op::ICmpSLt:
        case Op::ICmpUGe:
        case Op::ICmpUGt:
        case Op::ICmpULe:
        case Op::ICmpULt:
            return {Type::BoolTy};
    }

    Unreachable();
}

auto Proc::add(std::unique_ptr<Block> b) -> Block* {
    b->p = this;
    body.push_back(std::move(b));
    return body.back().get();
}

auto Value::as_int(TranslationUnit& tu) -> std::optional<APInt> {
    if (not isa<SmallInt, LargeInt>(this)) return std::nullopt;
    auto wd = u32(type()->size(tu).bits());
    APInt val{wd, 0};
    if (auto s = dyn_cast<SmallInt>(this)) val = s->value();
    else val = cast<LargeInt>(this)->value();
    return val;
}

/// ====================================================================
///  Printing
/// ====================================================================
namespace {
class Printer {
public:
    TranslationUnit& tu;
    SmallUnrenderedString out;
    DenseMap<GlobalConstant*, i64> global_ids;
    DenseMap<Argument*, i64> arg_ids;
    DenseMap<Block*, i64> block_ids;
    DenseMap<Inst*, i64> inst_ids;
    DenseMap<FrameSlot*, i64> frame_ids;
    Proc* curr_proc{};

    Printer(TranslationUnit& tu) : tu{tu} {}
    void Dump(Builder& b);
    void DumpInst(Inst* i);
    void DumpProc(Proc* p);
    auto DumpValue(Value* b) -> SmallUnrenderedString;
    bool ReturnsValue(Inst* i);

    auto Id(auto& map, auto* ptr) -> i64 {
        auto it = map.find(ptr);
        return it == map.end() ? -1 : it->second;
    }
};
} // namespace

auto Builder::Dump() -> SmallUnrenderedString {
    Printer p{tu};
    p.Dump(*this);
    return std::move(p.out);
}

void Printer::Dump(Builder& b) {
    for (auto [i, g] : enumerate(b.global_constants())) {
        global_ids[g] = i64(i);
        out += std::format("%3(@{}%) %1(=%) ", i);
        if (g->is_string()) out += std::format("%3(\"{}\"%)", utils::Escape(g->value(), true, true));
        else {
            out += "%1({%) %5(";
            out += utils::join(g->value(), "%1(,%) ", "{:02X}");
            out += "%) %1(}, align %)";
            out += std::format("%5({}%)", g->align());
        }
        out += "\n";
    }

    for (auto* proc : b.procedures())
        DumpProc(proc);
}

void Printer::DumpInst(Inst* i) {
    out += "    %1(";
    if (ReturnsValue(i)) out += std::format("%8(%%{}%) = ", Id(inst_ids, i));
    defer { out += "%)\n"; };

    auto Target = [&](Block* dest, ArrayRef<Value*> args) {
        out += std::format("%3(bb{}%)", Id(block_ids, dest));
        if (not args.empty()) {
            out += std::format(
                "({})",
                utils::join_as(args, [&](Value* v) { return DumpValue(v); })
            );
        }
    };

    auto IntCast = [&](StringRef name) {
        auto c = cast<ICast>(i);
        out += std::format(
            "%1({}%) {} to {}",
            name,
            DumpValue(c->args()[0]),
            c->cast_result_type()
        );
    };

    auto Simple = [&](StringRef name) {
        out += std::format(
            "%1({}%){}{}",
            name,
            i->args().empty() ? ""sv : " "sv,
            utils::join_as(i->args(), [&](Value* v) { return DumpValue(v); })
        );
    };

    switch (i->opcode()) {
        // Special instructions.
        case Op::Abort: {
            auto a = cast<AbortInst>(i);
            auto [file, line, col] = a->location().info_or_builtin(tu.context());
            out += std::format(
                "%1(abort%) at %6(<{}:{}:{}>%) %2({}%)({})",
                file,
                line,
                col,
                a->handler_name(),
                utils::join_as(a->args(), [&](Value* v) { return DumpValue(v); })
            );
        } break;

        case Op::Br: {
            auto b = cast<BranchInst>(i);
            if (b->is_conditional()) {
                out += std::format("%1(br%) {} to ", DumpValue(b->cond()));
                Target(b->then(), b->then_args());
                out += " else ";
                Target(b->else_(), b->else_args());
            } else {
                out += std::format("%1(br%) ");
                Target(b->then(), b->then_args());
            }
        } break;

        case Op::Call: {
            auto proc = i->args().front();
            auto args = i->args().drop_front(1);
            auto ret = cast<ProcType>(proc->type())->ret();
            if (args.empty()) {
                out += std::format("%1(call%) {} {}", ret, DumpValue(proc));
            } else {
                out += std::format(
                    "%1(call%) {} {}({})",
                    ret,
                    DumpValue(proc),
                    utils::join_as(args, [&](Value* v) { return DumpValue(v); })
                );
            }
        } break;

        case Op::Load: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(load%) {}, {}, align %5({}%)",
                m->memory_type(),
                DumpValue(m->args()[0]),
                m->align()
            );
        } break;

        case Op::Store: {
            auto m = cast<MemInst>(i);
            out += std::format(
                "%1(store%) {} to {}, {}, align %5({}%)",
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
        case Op::MemCopy: Simple("copy"); break;
        case Op::MemZero: Simple("zero"); break;
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

void Printer::DumpProc(Proc* proc) {
    curr_proc = proc;
    if (not out.empty()) out += "\n";
    out += proc->type()->print(proc->name(), true);

    // Stop if there is no body.
    if (proc->empty()) {
        out += "%1(;%)\n";
        return;
    }

    // Number all blocks and instructions.
    i64 temp = 0;
    arg_ids.clear();
    block_ids.clear();
    inst_ids.clear();

    for (auto* arg : proc->args()) arg_ids[arg] = temp++;
    for (auto* arg : proc->frame()) frame_ids[arg] = temp++;
    for (const auto& [id, b] : vws::enumerate(proc->blocks())) {
        block_ids[b] = id;
        for (auto* arg : b->arguments())
            arg_ids[arg] = temp++;
        for (auto* i : b->instructions())
            if (ReturnsValue(i))
                inst_ids[i] = temp++;
    }

    // Print the procedure body.
    out += " %1({%)\n";

    // Print frame allocations.
    for (auto* f : proc->frame())
        out += std::format("    %8(%%{}%) %1(=%) {}\n", Id(frame_ids, f), f->allocated_type());
    if (not proc->frame().empty())
        out += "\n";

    // Print blocks and instructions.
    for (const auto& [i, b] : enumerate(proc->blocks())) {
        out += i == 0 ? "%3(entry%)%1(" : std::format("\n%3(bb{}%)%1(", i);
        if (not b->arguments().empty()) {
            out += std::format("({})", utils::join_as(b->arguments(), [&](Argument* arg) {
                return std::format("{} %3(%%{}%)", b->argument_types()[arg->index()], arg_ids.at(arg));
            }));
        }
        out += ":%)\n";
        for (auto* inst : b->instructions())
            DumpInst(inst);
    }
    out += "%1(}%)\n";
}

bool Printer::ReturnsValue(Inst* i) {
    return not i->result_types().empty();
}

auto Printer::DumpValue(Value* v) -> SmallUnrenderedString {
    SmallUnrenderedString out;
    switch (v->kind()) {
        case Value::Kind::Argument: {
            auto arg = cast<Argument>(v);
            out += std::format("%{}(%%{}%)", isa<Proc>(arg->parent()) ? '4' : '3', Id(arg_ids, arg));
        } break;

        case Value::Kind::Block:
            out += std::format("%3(bb{}%)", Id(block_ids, cast<Block>(v)));
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
            out += std::format("{}[%5({}%)]", DumpValue(e->aggregate()), e->index());
        } break;

        case Value::Kind::FrameSlot: {
            auto f = cast<FrameSlot>(v);
            out += std::format("%8(%%{}%)", Id(frame_ids, f));
        } break;

        case Value::Kind::GlobalConstant: {
            auto c = cast<GlobalConstant>(v);
            out += std::format("%3(@{}%)", Id(global_ids, c));
        } break;

        case Value::Kind::InstValue: {
            auto i = cast<InstValue>(v);
            if (i->inst()->has_multiple_results()) out += std::format("%8(%%{}:{}%)", Id(inst_ids, i->inst()), i->index());
            else out += std::format("%8(%%{}%)", Id(inst_ids, i->inst()));
        } break;

        case Value::Kind::InvalidLocalReference: {
            auto i = cast<InvalidLocalReference>(v);
            auto d = i->referenced_local()->decl;
            out += std::format(
                "<invalid access to %2({}%)::%8({}%)>",
                d->parent->name,
                d->name
            );
        } break;

        case Value::Kind::LargeInt: {
            auto l = cast<LargeInt>(v);
            out += std::format("{} %5({}%)", l->type(), l->value());
        } break;

        case Value::Kind::Proc: {
            auto p = cast<Proc>(v);
            out += std::format("%2({}%)", p->name());
        } break;

        case Value::Kind::Range: {
            auto s = cast<Range>(v);
            out += std::format("%1(({}, {})%)", DumpValue(s->start), DumpValue(s->end));
        } break;

        case Value::Kind::Slice: {
            auto s = cast<Slice>(v);
            if (
                auto sz = dyn_cast<SmallInt>(s->size);
                sz and
                isa<GlobalConstant>(s->data) and
                cast<GlobalConstant>(s->data)->is_string()
            ) {
                auto str = cast<GlobalConstant>(s->data);
                out += std::format("s%3(\"{}\"%)", utils::Escape(str->value().take(usz(sz->value())), true, true));
            } else {
                out += std::format("{} (%5({}%), %5({}%))", s->type(), DumpValue(s->data), DumpValue(s->size));
            }
        } break;

        case Value::Kind::SmallInt: {
            auto s = cast<SmallInt>(v);
            out += std::format(
                "{} %5({}%)",
                v->type(),
                APInt(u32(v->type()->size(tu).bits()), u64(s->value()))
            );
        } break;
    }
    return out;
}

void Inst::dump(TranslationUnit& tu) {
    Printer p{tu};
    p.DumpInst(this);
    p.out += "\n";
    std::print("{}", text::RenderColours(true, p.out.str()));
}

void Proc::dump(TranslationUnit& tu) {
    Printer p{tu};
    p.DumpProc(this);
    p.out += "\n";
    std::print("{}", text::RenderColours(true, p.out.str()));
}
