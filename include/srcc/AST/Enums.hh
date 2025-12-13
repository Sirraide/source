#ifndef SRCC_AST_ENUMS_HH
#define SRCC_AST_ENUMS_HH

#include <srcc/Core/Utils.hh>

#include <base/Assert.hh>

#include <format>

namespace srcc {
class TranslationUnit;
enum class BuiltinKind : u8;
enum class CallingConvention : u8;
enum class ExprKind : u8;
enum class Intent : u8;
enum class Linkage : u8;
enum class Mangling : u8;
enum class ValueCategory : u8;
enum class OverflowBehaviour : u8;
enum class ScopeKind : u8;
enum class ParamPassingMode : u8;
} // namespace srcc

/// Parameter intents.
enum class srcc::Intent : base::u8 {
    /// The parameter is moved into the callee, who takes ownership
    /// of it. This may be lowered to a copy for small types, or to
    /// pass-by-reference for larger ones.
    ///
    /// This is the default if no intent is specified and cannot be
    /// written explicitly.
    Move,

    /// The parameter is passed readonly.
    In,

    /// The parameter is used to return a value.
    Out,

    /// The parameter is used to return a value, but the callee
    /// may also inspect it before that.
    Inout,

    /// The parameter is copied and behaves like a local variable in
    /// the callee. This is the only valid intent for procedures that
    /// use the C++ ABI.
    Copy,
};

/// Parameter passing modes.
///
/// These are what intents are lowered to.
enum class srcc::ParamPassingMode : base::u8 {
    /// Pass an lvalue of the same type.
    ///
    /// This is used in cases where the callee wants to modify a
    /// value in the caller’s stack frame.
    ///
    /// Sema will ensure that this is an lvalue.
    SameTypeLValue,

    /// Pass an rvalue to the callee.
    ///
    /// Depending on the calling convention, target, and size of the
    /// type, this may entail passing the value indirectly by storing
    /// it to memory and passing a pointer instead.
    ///
    /// Sema will ensure that this is an srvalue or mrvalue.
    RValue,

    /// Pass an lvalue if we already have one, and materialise a
    /// temporary and pass it otherwise.
    ///
    /// This mode cannot be used for SRValue types.
    ///
    /// Sema will ensure that this is an lvalue.
    AnyLValue,
};

/// Builtin types.
enum class srcc::BuiltinKind : base::u8 {
    Void,
    NoReturn,
    Bool,
    Int,
    Deduced,
    Tree,
    Type,
    UnresolvedOverloadSet,
    Nil,
};

/// Linkage of a global entity.
enum class srcc::Linkage : base::u8 {
    Internal,   ///< Not accessible outside this module.
    Exported,   ///< Exported from the module.
    Imported,   ///< Imported from another module.
    Reexported, ///< Imported and exported.
    Merge,      ///< Merge definitions of this entity across modules.
};

/// Mangling scheme of a global entity.
enum class srcc::Mangling : base::u8 {
    None,   ///< Do not mangle this at all.
    Source, ///< Mangling this using our language rules.
    CXX,    ///< Use C++’s name mangling rules.
};

/// Calling convention of a function.
enum class srcc::CallingConvention : base::u8 {
    /// Native calling convention for C and C++ interop.
    ///
    /// Currently, this is also the default calling convention for
    /// simplicity. We could use fastcc, but there are some issues
    /// w/ that; in particular, Clang doesn’t support it, so we can’t
    /// call functions in modules from C++ properly if we use that
    /// calling convention, and it also means that users would have
    /// to care about the calling convention when writing function
    /// types if we had two different ones that were in common use.
    Native,
};

enum class srcc::ValueCategory : base::u8 {
    /// A value that doesn’t have identity; there are two types
    /// of rvalues:
    ///
    /// Scalar rvalues, i.e. an rvalue available as one or more SSA
    /// variables. This is everything that we can construct w/o
    /// needing a memory location.
    ///
    /// Memory rvalues, really a set of instructions to initialise
    /// a memory location; evaluating this usually entails writing
    /// values into memory and doesn’t actually produce a value.
    ///
    /// The former roughly correspond to scalar prvalues in C++,
    /// the latter to prvalues of class type; note that these are
    /// *not* xvalues; we don’t have xvalues.
    ///
    /// We used to have an actual distinction between MRValue and
    /// SRValue throughout the compiler, but really, only codegen
    /// cares about this and at that point the distinction is also
    /// ABI-dependent, so it was removed from the AST.
    RValue,

    /// An object on the stack or heap that stores a value and has
    /// a memory location that can be references.
    ///
    /// These roughly correspond to glvalues in C++.
    LValue,

    // NOTE: Update TypeAndValueCategory if new categories require
    // us to use more than 2 bits to store this.
};

enum class srcc::OverflowBehaviour : base::u8 {
    /// Abort the program.
    Trap,

    /// Wrap around like normal.
    Wrap,
};

enum class srcc::ScopeKind : base::u8 {
    /// A block expression.
    Block,

    /// A block expression that is also the body of a procedure definition.
    Procedure,

    /// A struct declaration.
    Struct,
};

template <>
struct std::formatter<srcc::Intent> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(srcc::Intent i, FormatContext& ctx) const {
        auto s = [i] -> std::string_view {
            switch (i) {
                using enum srcc::Intent;
                case Move: return "move";
                case In: return "in";
                case Out: return "out";
                case Inout: return "inout";
                case Copy: return "copy";
            }
            return "<invalid intent>";
        }();
        return std::formatter<std::string_view>::format(s, ctx);
    }
};

#endif // SRCC_AST_ENUMS_HH
