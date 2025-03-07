#ifndef SRCC_AST_VM_INTERNAL_HH
#define SRCC_AST_VM_INTERNAL_HH

#include <srcc/AST/Eval.hh>

namespace srcc::eval {
void PrintByteCode(const TranslationUnit& tu, const ByteBuffer& code);

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
/// Immediate arguments always precede frame arguments.
enum class Op : u8 {
#define SRCC_VM_OP_ENUMERATORS
#include "Ops.inc"
};
} // namespace srcc::eval

class [[nodiscard]] srcc::eval::VMProc {
public:
    ByteBuffer code;
    Size frame_size;
    Align frame_align;
};

#endif // SRCC_AST_VM_INTERNAL_HH
