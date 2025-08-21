#ifndef SRCC_CG_TARGET_TARGET_HH
#define SRCC_CG_TARGET_TARGET_HH

#include <srcc/AST/Type.hh>
#include <srcc/Core/Core.hh>

#include <clang/Basic/TargetInfo.h>

namespace mlir {
class TypeRange;
}
namespace llvm {
class BitVector;
}
namespace srcc {
class Target;
}

namespace srcc::cg {
class CallBuilder;
class CodeGen;
}

namespace mlir {
class ArrayAttr;
class Value;
}

// TODO: Move these to a local header in lib/CG/Target
namespace srcc::target {
auto CreateX86_64_Linux(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target>;
}

class srcc::cg::CallBuilder {
    LIBBASE_IMMOVABLE(CallBuilder);

protected:
    CodeGen& cg;
    CallBuilder(CodeGen& cg) : cg{cg} {}

public:
    virtual ~CallBuilder() = default;

    /// Add a by-value argument.
    ///
    /// If 'require_copy' is true, then we need to emit a stack copy to avoid
    /// modifying the original if we end up passing this by pointer. If it is
    /// 'false', the original value can be used directly.
    virtual void add_argument(Type param_ty, mlir::Value v, bool require_copy) = 0;

    /// Add a pointer or integer argument. This is NOT to be used for 'bool'!
    virtual void add_pointer_or_integer(mlir::Value v) = 0;

    /// Add a pointer argument.
    virtual void add_pointer(mlir::Value v) = 0;

    /// Get the argument attributes.
    [[nodiscard]] virtual auto get_arg_attrs() -> mlir::ArrayAttr = 0;

    /// Get the final call arguments.
    [[nodiscard]] virtual auto get_final_args() -> ArrayRef<mlir::Value> = 0;

    /// Get the current target.
    [[nodiscard]] auto target() -> const Target&;
};

class srcc::Target {
    LIBBASE_IMMOVABLE(Target);

    llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI;

protected:
    Target(llvm::IntrusiveRefCntPtr<clang::TargetInfo>);

public:
    virtual ~Target();
    static auto Create(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target>;

    /// Get a helper to build a function call.
    ///
    /// \param ty The type of the call.
    /// \param indirect_ptr The indirect return pointer, if any.
    [[nodiscard]] virtual auto get_call_builder(
        cg::CodeGen& cg,
        ProcType* ty,
        mlir::Value indirect_ptr
    ) const -> std::unique_ptr<cg::CallBuilder> = 0;

    /// Get the underlying Clang target.
    [[nodiscard]] auto clang() const -> const clang::TargetInfo& { return *TI; }

    /// Get the alignment and size of a closure on this target.
    [[nodiscard]] auto closure_align() const -> Align { return ptr_align(); }
    [[nodiscard]] auto closure_size() const -> Size { return 2 * ptr_size(); }

    /// Get the alignment of the builtin 'int' type.
    [[nodiscard]] auto int_align() const -> Align { return ptr_align(); }

    /// Get the alignment of an integer type.
    [[nodiscard]] auto int_align(const IntType* ty) const -> Align { return int_align(ty->bit_width()); }
    [[nodiscard]] auto int_align(Size width) const -> Align {
        return Align(TI->getBitIntAlign(u32(width.bits())) / 8);
    }

    /// Get the in-memory size of the builtin 'int' type.
    [[nodiscard]] auto int_size() const -> Size { return ptr_size(); }

    /// Get the in-memory size of an integer type.
    [[nodiscard]] auto int_size(const IntType* ty) const -> Size { return int_size(ty->bit_width()); }
    [[nodiscard]] auto int_size(Size width) const -> Size {
        return Size::Bits(TI->getBitIntWidth(u32(width.bits())));
    }

    /// Whether this type must be returned indirectly via memory by passing an
    /// extra pointer argument to the function.
    virtual bool needs_indirect_return(Type ty) const = 0;

    /// Get the pointer alignment.
    [[nodiscard]] auto ptr_align() const -> Align { return Align(TI->PointerAlign / 8); }

    /// Get the pointer sice.
    [[nodiscard]] auto ptr_size() const -> Size { return Size::Bits(TI->PointerWidth); }

    /// Get the alignment of slices on this target.
    [[nodiscard]] auto slice_align() const -> Align { return std::max(ptr_align(), int_align()); }

    /// Get the size of slices on this target.
    [[nodiscard]] auto slice_size() const -> Size { return ptr_size().align(int_align()) + int_size(); }

    /// Get the target triple.
    [[nodiscard]] auto triple() const -> const llvm::Triple& {
        return TI->getTriple();
    }
};

#endif // SRCC_CG_TARGET_TARGET_HH
