#ifndef SRCC_CG_ABI_HH
#define SRCC_CG_ABI_HH

#include <srcc/Core/Utils.hh>
#include <srcc/AST/Stmt.hh>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

namespace srcc::cg {
class CodeGen;
class ProcData;
}

namespace srcc::cg::abi {
/// ABI lowering information about a function argument list.
struct CallInfo {
    SmallVector<mlir::Type> result_types;
    SmallVector<mlir::Type> arg_types;
    SmallVector<mlir::Attribute> result_attrs;
    SmallVector<mlir::Attribute> arg_attrs;
    SmallVector<mlir::Value> args;
    mlir::FunctionType func;
    bool no_return = false;
};

/// A single IR-level argument.
struct Arg {
    LIBBASE_MOVE_ONLY(Arg);

public:
    mlir::Type ty;
    mlir::Value value = nullptr; ///< Only populated if we’re lowering a call.
    SmallVector<mlir::NamedAttribute, 1> attrs{};

    Arg(mlir::Type ty): ty(ty) {}

    /// Add a 'byval' attribute.
    void add_byval(mlir::Type ty);

    /// Add an 'signext' attribute.
    void add_sext(CodeGen& cg);

    /// Add an 'sret' attribute.
    void add_sret(mlir::Type ty);

    /// Add an 'zeroext' attribute.
    void add_zext(CodeGen& cg);
};

/// Context used to convert a bundle of IR arguments back to a Source type.
///
/// One of these is created for each AST-level argument or return type; an
/// instance of this should not be reused.
class IRToSourceConversionContext {
    CodeGen& codegen;
    mlir::Location loc;
    mlir::ValueRange range;
    Type ty;
    mlir::Value indirect_ptr = {};
    unsigned i = 0;

public:
    /// Create a new context.
    ///
    /// \param cg The CodeGen instance.
    /// \param loc The location of the thing we’re creating.
    /// \param r The input values.
    /// \param ty The type we’re creating.
    /// \param addr The memory location to write to.
    explicit IRToSourceConversionContext(
        CodeGen& cg,
        mlir::Location loc,
        mlir::ValueRange r,
        Type ty,
        mlir::Value addr = nullptr
    ) : codegen(cg), loc(loc), range(r), ty(ty), indirect_ptr(addr) {}

    /// Get the CodeGen instance.
    [[nodiscard]] auto cg() -> CodeGen& { return codegen; }

    /// Get or create address into which to store the value, if any.
    [[nodiscard]] auto addr() -> mlir::Value;

    /// Get the number of IR arguments that were consumed.
    [[nodiscard]] auto consumed() -> unsigned { return i; }

    /// Get the location of the value that this is initialising.
    [[nodiscard]] auto location() -> mlir::Location { return loc; }

    /// Get the next value and consume it.
    [[nodiscard]] auto next() -> mlir::Value {
        Assert(i < range.size());
        return range[i++];
    }

    /// Get the type that we’re creating.
    [[nodiscard]] auto type() -> Type { return ty; }
};

/// ABI lowering information about an argument that is passed by value.
using ArgInfo = SmallVector<Arg, 2>;

/// Target-specific ABI hooks.
class ABI {
public:
    virtual ~ABI() = default;

    /// Whether a value of this type can be used as-is when returned from a function.
    [[nodiscard]] virtual bool can_use_return_value_directly(
        CodeGen& cg,
        Type ty
    ) const = 0;

    /// Lower a direct return value.
    [[nodiscard]] virtual auto lower_direct_return(
        CodeGen& cg,
        mlir::Location l,
        Expr* arg
    ) const -> ArgInfo = 0;

    /// Lower the parameters to a procedure and create local
    /// variables for them.
    virtual void lower_parameters(CodeGen& cg, ProcData& pdata) const = 0;

    /// Lower a procedure type.
    [[nodiscard]] virtual auto lower_proc_type(
        CodeGen& cg,
        ProcType* ty,
        bool needs_environment
    ) const -> CallInfo = 0;

    /// Perform ABI lowering for a call or argument list.
    [[nodiscard]] virtual auto lower_procedure_signature(
        CodeGen& cg,
        mlir::Location l,
        ProcType* proc,
        bool needs_environment,
        mlir::Value indirect_ptr,
        mlir::Value env_ptr,
        ArrayRef<Expr*> args
    ) const -> CallInfo = 0;

    /// Whether a type must be returned indirectly via a pointer.
    [[nodiscard]] virtual bool needs_indirect_return(
        CodeGen& cg,
        Type ty
    ) const = 0;

    /// Whether an 'in' parameter of this type should be passed by
    /// reference; this should *only* perform ABI-specific checks
    /// (e.g. does this fit in a register); any other conditions
    /// that would require passing by reference (such as the value
    /// not being trivially copyable) will have already been checked.
    [[nodiscard]] virtual bool pass_in_parameter_by_reference(
        CodeGen& cg,
        Type ty
    ) const = 0;

    /// Write the results of a call operation to memory.
    [[nodiscard]] virtual auto write_call_results_to_mem(
        abi::IRToSourceConversionContext& ctx
    ) const -> mlir::Value = 0;
};
}

#endif