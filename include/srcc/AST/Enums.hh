#ifndef SRCC_AST_ENUMS_HH
#define SRCC_AST_ENUMS_HH

#include <srcc/Core/Utils.hh>

#include <base/Assert.hh>

#include <format>

namespace srcc {
class TranslationUnit;
enum class BuiltinKind : u8;
enum class CallingConvention : u8;
enum class Dependence : u8;
enum class ExprKind : u8;
enum class Intent : u8;
enum class Linkage : u8;
enum class Mangling : u8;
enum class ValueCategory : u8;
enum class OverflowBehaviour : u8;
enum class ScopeKind : u8;
} // namespace srcc

/// Expression dependence.
///
/// An expression is *dependent* if it references a template
/// parameter or contains a dependent subexpression; this has
/// two main ramifications.
///
/// We may not be able to fully analyse the expression until
/// we instantiate it; this means we can’t check its type (or
/// use its value if it’s a constant expression) until then.
///
/// We have to create a copy of the expression to instantiate
/// any dependent subexpressions, even if the expression itself
/// is neither type nor value dependent.
///
/// Lastly, because error recovery is pretty similar to dependence,
/// i.e. we don’t want to try and typecheck an expression if we
/// know that determining its type caused an error, we also model
/// it as dependence. Unlike other forms of dependence, error
/// dependence is never resolved once set.
enum class srcc::Dependence : base::u8 {
    /// This expression is not dependent.
    None = 0,

    /// This expression is dependent in some way.
    ///
    /// If only this bit is set, then this expression contains
    /// dependent code that needs to be instantiated during
    /// template instantiation.
    Instantiation = 1,

    /// The value of this expression is dependent.
    ValueDependent = 1 << 1,
    Value = ValueDependent | Instantiation,

    /// The type of this expression is dependent.
    TypeDependent = 1 << 2,
    Type = TypeDependent | Instantiation,

    /// Combination of value and type dependence.
    ValueAndType = Value | Type,

    /// This expression contains an error.
    ErrorDependent = 1 << 7,
    Error = ErrorDependent | Instantiation,
};

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

/// Builtin types.
enum class srcc::BuiltinKind : base::u8 {
    Void,
    Dependent,
    ErrorDependent,
    NoReturn,
    Bool,
    Int,
    Deduced,
    Type,
    UnresolvedOverloadSet,
};

namespace srcc {
constexpr auto operator|(Dependence a, Dependence b) -> Dependence { return Dependence(u8(a) | u8(b)); }
constexpr auto operator|=(Dependence& a, Dependence b) -> Dependence& { return a = a | b; }
constexpr bool operator&(Dependence a, Dependence b) { return Dependence(u8(a) & u8(b)) != Dependence::None; }
} // namespace srcc

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
    Source, ///< Default calling convention.
    Native, ///< Native calling convention for C and C++ interop.
};

enum class srcc::ValueCategory : base::u8 {
    /// Scalar rvalue, i.e. an rvalue that is passed in registers
    /// and available as an SSA variable. This is everything that
    /// we can construct w/o needing a memory location.
    ///
    /// These roughly correspond to scalar prvalues in C++.
    SRValue,

    /// Class rvalues, i.e. an rvalue (usually of class type) that
    /// needs a memory location to construct into.
    ///
    /// These roughly correspond to prvalues of class type in C++;
    /// note that these are *not* xvalues; we don’t have xvalues.
    MRValue,

    /// An object on the stack or heap that stores a value and has
    /// a memory location that can be references.
    ///
    /// These roughly correspond to glvalues in C++.
    LValue,

    /// Dependent, don’t know yet.
    DValue,
};

enum class srcc::OverflowBehaviour : base::u8 {
    /// Abort the program.
    Trap,

    /// Wrap around like normal.
    Wrap,
};

enum class srcc::ScopeKind : base::u8 {
    Block,
    Procedure,
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
