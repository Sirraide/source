#include <srcc/AST/AST.hh>

#include "VMInternal.hh"

using namespace srcc;
using namespace srcc::eval;

#define INTERP_LOOP_BEGIN for (;;) {
#define INTERP_LOOP_END }
#define OP(op) case Op::op:

#define BUILTIN_INTEGER_OP(name, size) case Op::name##size: { \
    auto lhs = SmallIntegerOperand(); \
    auto lhs = SmallIntegerOperand();\
}

namespace {
struct StackFrame {
    u64 stack_base;
    ByteReader code;

    StackFrame(u64 stack_base, ByteReader code) : stack_base(stack_base), code(code) {}
};
}

/*auto VM::ExecuteProcedure(const VMProc& entry_point, bool complain) -> std::optional<Value> {
    SmallVector<std::byte, 1024> stack;
    std::vector<StackFrame> frames;
    const u64 max_steps = owner().context().eval_steps ?: std::numeric_limits<u64>::max();
    u64 steps = 0;
    ByteReader* code{};

    auto PushFrame = [&](const VMProc& proc) {
        frames.emplace_back(stack.size(), ByteReader{proc.code});

        // Allocate space on the stack.
        auto new_stack_size = Size::Bytes(stack.size()).align(proc.frame_align) + proc.frame_size;
        stack.resize(u64(new_stack_size.bytes()));

        // Update the ip.
        code = &frames.back().code;
    };

    auto PopFrame = [&] {
        Assert(frames.size() > 0, "Stack underflow");

        // Resize the stack.
        stack.resize(frames.back().stack_base);
        frames.pop_back();

        // Reset the ip.
        code = &frames.back().code;
    };

    auto BuiltinIntegerOp = [&]<std::integral IntTy> (IntTy Operation(IntTy, IntTy)) {
        auto lhs = IntTy(SmallIntegerOperand());
        auto rhs = IntTy(SmallIntegerOperand());
        auto res = Operation(lhs, rhs);
    };

    // Setup.
    PushFrame(entry_point);

    // Start of the interpreter loop.
    //
    // We use a bunch of macros for this to facilitate changing the implementation
    // paradigm (e.g. switch-case vs computed gotos).
    INTERP_LOOP_BEGIN

    // Get the next instruction.
    Op op = code->read<Op>();

    Todo("Interpret op: {}", +op);

    INTERP_LOOP_END
}*/


