#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

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

auto Target::create_machine(int opt_level) const -> std::unique_ptr<llvm::TargetMachine> {
    // Get the target.
    std::string error;
    auto* target = llvm::TargetRegistry::lookupTarget(triple(), error);
    if (not error.empty() or not target) Fatal(
        "Failed to lookup target triple '{}': {}",
        triple().getTriple(),
        error
    );

    // Get feature flags.
    SmallString<128> features;
    if (opt_level == 4) {
        StringMap<bool> feature_map = llvm::sys::getHostCPUFeatures();
        for (auto& [feature, enabled] : feature_map)
            if (enabled)
                Format(features, "+{},", feature.str());
    }

    // User-specified features are applied last.
    // for (auto& [feature, enabled] : target_features)
    //    Format(features, "{}{},", enabled ? '+' : '-', feature.str());
    // if (not features.empty()) features.pop_back();

    // Get CPU.
    std::string cpu;
    if (opt_level == 4) cpu = llvm::sys::getHostCPUName();
    if (cpu.empty()) cpu = "generic";

    // Target options.
    llvm::TargetOptions opts;

    // Get opt level.
    llvm::CodeGenOptLevel opt;
    switch (opt_level) {
        case 0: opt = llvm::CodeGenOptLevel::None; break;
        case 1: opt = llvm::CodeGenOptLevel::Less; break;
        case 2: opt = llvm::CodeGenOptLevel::Default; break;
        default: opt = llvm::CodeGenOptLevel::Aggressive; break;
    }

    // Create machine.
    std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
        triple(),
        cpu,               // Target CPU
        features,          // Features.
        opts,              // Options.
        llvm::Reloc::PIC_, // Relocation model.
        std::nullopt,      // Code model.
        opt                // Opt level.
    )};

    return machine;
}

