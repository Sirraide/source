#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>
#include "TargetDefs.hh"

using namespace srcc;
using namespace srcc::cg;

Target::Target(
    llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI,
    std::unique_ptr<cg::abi::ABI> target_abi
) : TI{std::move(TI)}, target_abi{std::move(target_abi)} {}

Target::~Target() = default;

auto Target::Create(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target> {
    const auto& triple = TI->getTriple();
    if (triple.getArch() == llvm::Triple::x86_64 and triple.isOSLinux())
        return target::CreateX86_64_Linux(std::move(TI));

    Fatal("Unsupported target '{}'", triple.str());
}

