#ifndef SRCC_INTERNAL_CG_TARGET_TARGET_DEFS_HH
#define SRCC_INTERNAL_CG_TARGET_TARGET_DEFS_HH

#include <srcc/CG/Target/Target.hh>

namespace srcc::cg::target {
using ClangTarget = llvm::IntrusiveRefCntPtr<clang::TargetInfo>;
using SourceTarget = std::unique_ptr<Target>;

// Target creation hooks.
auto CreateX86_64_Linux(ClangTarget TI) -> SourceTarget;
}

#endif