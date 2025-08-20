#include <srcc/CG/Target/Target.hh>

using namespace srcc;

namespace {
struct Impl final : Target {
    Impl(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) : Target(std::move(TI)) {}
};
}

auto target::CreateX86_64_Linux(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target> {
    return std::make_unique<Impl>(std::move(TI));
}

