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
class CodeGen;
}

namespace srcc::cg::ir {
class CallOp;
}

namespace mlir {
class ArrayAttr;
class FunctionType;
class Type;
class Value;
}

// TODO: Move these to a local header in lib/CG/Target
namespace srcc::target {
auto CreateX86_64_Linux(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target>;
}

class srcc::Target {
    LIBBASE_IMMOVABLE(Target);

    llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI;

protected:
    Target(llvm::IntrusiveRefCntPtr<clang::TargetInfo>);

public:
    virtual ~Target();
    static auto Create(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target>;

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
        if (width == Size::Bits(128)) return Align(Size::Bits(TI->getInt128Align()));
        return Align(TI->getBitIntAlign(u32(width.bits())) / 8);
    }

    /// Get the in-memory size of the builtin 'int' type.
    [[nodiscard]] auto int_size() const -> Size { return ptr_size(); }

    /// Get the in-memory size of an integer type.
    [[nodiscard]] auto int_size(const IntType* ty) const -> Size { return int_size(ty->bit_width()); }
    [[nodiscard]] auto int_size(Size width) const -> Size {
        return Size::Bits(TI->getBitIntWidth(u32(width.bits())));
    }

    /// Get the preferred alignment of this type.
    [[nodiscard]] auto preferred_align(Type ty) const -> Align {
        if (ty->is_integer()) return int_align(ty->size(*this));
        return ty->align(*this);
    }

    /// Get the preferred size of this type.
    [[nodiscard]] auto preferred_size(Type ty) const -> Size {
        if (ty->is_integer()) return int_size(ty->size(*this));
        return ty->size(*this);
    }

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
