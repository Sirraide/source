#ifndef SRCC_AST_VM_INTERNAL_HH
#define SRCC_AST_VM_INTERNAL_HH

#include <srcc/AST/Eval.hh>

namespace srcc::eval {
void PrintByteCode(const ByteBuffer& code);

// First byte of an instruction operand.
enum class OpValue : u8 {
    /// The literal integer '0'.
    Lit0,

    /// The literal integer '1'.
    Lit1,

    /// The literal integer '2'.
    Lit2,

    /// Small integer constant (<= 64 bits).
    ///
    /// Followed by an inline u64 value.
    /// TODO: Inline value should be a VarInt.
    SmallInt,

    /// Value added to inline frame offsets.
    InlineOffs,

    /// Value used to indicate a large frame offset follows.
    ///
    /// Followed by a ByteOffset.
    /// TODO: Should the ByteOffset be stored as a VarInt?
    LargeOffs = 255,

    /// Maximum inline frame offset, inclusive.
    MaxInlineOffs = LargeOffs - 3 - 1,
};

enum class FrameOffset : u32;
enum class CodeOffset : u32;

#define INTEGER_OP(name) \
    name,                \
        name##I8 = name, \
        name##I16,       \
        name##I32,       \
        name##I64,       \
        name##APInt

#define TYPED_OP(name) \
    INTEGER_OP(name),  \
        name##Bool,    \
        name##Closure, \
        name##Ptr,     \
        name##Slice,   \
        name##Type

#define INTEGER_EXT(X, i)       \
    X(I8ToI16, i##8, i##16)     \
    X(I8ToI32, i##8, i##32)     \
    X(I8ToI64, i##8, i##64)     \
    X(I8ToAPInt, i##8, APInt)   \
    X(I16ToI32, i##16, i##32)   \
    X(I16ToI64, i##16, i##64)   \
    X(I16ToAPInt, i##16, APInt) \
    X(I32ToI64, i##32, i##64)   \
    X(I32ToAPInt, i##32, APInt) \
    X(I64ToAPInt, i##64, APInt) \
    X(APIntToAPInt, APInt, APInt)

#define INTEGER_TRUNC(X)           \
    X(TruncI8ToAPInt, i8, APInt)   \
    X(TruncI16ToI8, i16, i8)       \
    X(TruncI16ToAPInt, i16, APInt) \
    X(TruncI32ToI8, i32, i8)       \
    X(TruncI32ToI16, i32, i16)     \
    X(TruncI32ToAPInt, i32, APInt) \
    X(TruncI64ToI8, i64, i8)       \
    X(TruncI64ToI16, i64, i16)     \
    X(TruncI64ToI32, i64, i32)     \
    X(TruncI64ToAPInt, i32, APInt) \
    X(TruncAPIntToI8, APInt, i8)   \
    X(TruncAPIntToI16, APInt, i16) \
    X(TruncAPIntToI32, APInt, i32) \
    X(TruncAPIntToI64, APInt, i64) \
    X(TruncAPIntToAPInt, APInt, APInt)

// ============================================================================
//  Operations
// ============================================================================
/// Bytecode operations.
///
/// Each opcode is a single byte. The function of an opcode is
/// described by a comment above it, and a list of its operands
/// is given at the end of the comment. The operands may either
/// be source-level types (e.g. u8[]), or compiler-internal types
/// (e.g. Location).
///
/// When a vm procedure is executed, it reserves space for its
/// arguments, basic block arguments, and any temporaries produced
/// by its instructions. Instead of pushing and popping values off
/// a stack, each instruction that produces a value is mapped to
/// a slot in the frame.
///
/// If a subsequent instruction then references a value produced
/// by a prior instruction, it can simply access that instruction’s
/// slot in the frame. This allows us to deal with the fact that
/// an instruction’s values may be referenced multiple times throughout
/// the function.
///
/// An ‘I’ denotes immediate operands, and an ‘F’ denotes operands
/// that reference frame slots. A declaration of the form ‘X, Y -> Z’
/// means that the instruction takes two operands of type X and Y,
/// and its result is a value of type Z.
///
/// Immediate arguments always precede frame arguments.
enum class Op : u8 {
    /// Abort evaluation due to a constraint violation (e.g. a failed
    /// assertion or integer overflow).
    ///
    /// F: u8[], u8[]
    /// I: AbortReason, Location
    Abort,

    /// Integer addition.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Add),

    /// Integer bitwise AND.
    ///
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(And),

    /// Integer arithmetic shift right.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(AShr),

    /// Unconditional branch.
    ///
    /// I: BlockAddr
    Branch,

/// Integer conversions.
///
/// F: iX -> iY
#define X(name, ...) SExt##name,
    INTEGER_EXT(X, i)
#undef X
#define X(name, ...) ZExt##name,
        INTEGER_EXT(X, i)
#undef X

    /// Integer comparison.
    ///
    /// F: iX, iX -> bool
    INTEGER_OP(CmpEq),
    INTEGER_OP(CmpNe),
    INTEGER_OP(CmpSLt),
    INTEGER_OP(CmpULt),
    INTEGER_OP(CmpSGt),
    INTEGER_OP(CmpUGt),
    INTEGER_OP(CmpSLe),
    INTEGER_OP(CmpULe),
    INTEGER_OP(CmpSGe),
    INTEGER_OP(CmpUGe),

    /// Conditional branch.
    ///
    /// F: bool
    /// I: BlockAddr, BlockAddr
    CondBranch,

    /// Direct call to a procedure.
    ///
    /// The arguments are determine by the procedure type and
    /// remain on the stack after the call.
    ///
    /// F: [any...] -> [any...]
    /// I: Pointer
    DirectCall,

    /// Load a value.
    ///
    /// F: Pointer -> any
    TYPED_OP(Load),

    /// Integer logical shift right.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(LShr),

    /// Set memory to 0.
    ///
    /// F: Pointer
    /// I: i64
    MemZero,

    /// Integer multiplication.
    INTEGER_OP(Mul),

    /// Integer bitwise OR.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Or),

    /// Pointer addition.
    ///
    /// F: Pointer, i64 -> Pointer
    PtrAdd,

    /// Return a value.
    ///
    /// F: any
    TYPED_OP(Ret),

    /// Return from a function without a return value.
    RetVoid,

    /// Integer addition with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SAddOv),

    /// Signed integer division.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(SDiv),

    /// Select between two values.
    ///
    /// F: bool, any, any -> any
    Select,

    /// Integer left shift.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Shl),

    /// Signed multiplication with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SMulOv),

    /// Signed integer remainder.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(SRem),

    /// Signed subtraction with overflow checking.
    ///
    /// F: iX, iX -> iX, bool
    INTEGER_OP(SSubOv),

    /// Store a value.
    ///
    /// F: Pointer, any
    TYPED_OP(Store),

    /// Integer subtraction.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Sub),

/// Integer truncation.
///
/// F: iX -> iY
#define X(name, ...) name,
    INTEGER_TRUNC(X)
#undef X

    /// Unsigned integer division.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(UDiv),

    /// Unreachable instruction.
    Unreachable,

    /// Call to a procedure that has not been compiled yet.
    ///
    /// F: [any...] -> [any...]
    /// I: String
    UnresolvedCall,

    /// Unsigned integer remainder.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(URem),

    /// Integer bitwise XOR.
    ///
    /// F: iX, iX -> iX
    INTEGER_OP(Xor),
};

#undef TYPED_OP
#undef INTEGER_OP
#undef INTEGER_EXT
#undef INTEGER_TRUNC
} // namespace srcc::eval

class [[nodiscard]] srcc::eval::VMProc {
public:
    ByteBuffer code;
    Size frame_size;
    Align frame_align;
};

#endif // SRCC_AST_VM_INTERNAL_HH
