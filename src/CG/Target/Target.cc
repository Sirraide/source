#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>


using namespace srcc;

Target::Target(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) : TI(std::move(TI)) {}
Target::~Target() = default;

auto Target::Create(llvm::IntrusiveRefCntPtr<clang::TargetInfo> TI) -> std::unique_ptr<Target> {
    const auto& triple = TI->getTriple();
    if (triple.getArch() == llvm::Triple::x86_64 and triple.isOSLinux())
        return target::CreateX86_64_Linux(std::move(TI));

    Fatal("Unsupported target '{}'", triple.str());
}

/*
auto cg::CallBuilder::target() -> const Target& {
    return cg.translation_unit().target();
}
*/
