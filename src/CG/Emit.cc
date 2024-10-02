module;
#include <base/Assert.hh>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ArchiveWriter.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <srcc/Macros.hh>
module srcc.codegen;
import srcc;
using namespace srcc;

// I’m at the end of my patience with the myriad of overly verbose
// ways in which error handling is done in this goddamn library.
class OpenFile {
    std::error_code ec;
    llvm::raw_fd_ostream stream;

public:
    OpenFile(StringRef path) : stream{path, ec} {
        if (ec) Fatal("Could not open file '{}' for writing: {}", path, ec.message());
    }

    auto operator->() -> llvm::raw_fd_ostream* { return &stream; }
    operator llvm::raw_fd_ostream&() { return stream; }
};


// Largely copied and adapted from Clang.
void CodeGen::OptimiseModule(llvm::TargetMachine& machine, TranslationUnit&, llvm::Module& m) {
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::ModulePassManager MPM;

    llvm::PipelineTuningOptions PTO;
    PTO.MergeFunctions = +machine.getOptLevel() >= 2;
    llvm::PassBuilder PB(&machine, PTO);

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    MPM.addPass(llvm::VerifierPass());
    MPM.run(m, MAM);
}

int CodeGen::EmitModuleOrProgram(llvm::TargetMachine& machine, TranslationUnit& tu, llvm::Module& m) {
    // Create a temporary path for the object file.
    std::string object_file_path = File::TempPath(".o");

    // Open it for writing.
    OpenFile os{object_file_path};

    // Use the legacy pass manager for this as the new one doesn’t
    // support codegen yet...
    llvm::legacy::PassManager pm;
    if (machine.addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::ObjectFile))
        Fatal("Unable to add passes to emit object file");

    // Emit the object file and flush the stream.
    pm.run(m);
    os->close();

    // Yeet the object file when we’re done.
    llvm::FileRemover remover(object_file_path);

    // Link the damn thing. Clang is better than us at figuring this out.
    // TODO: Add a --linker=foo option.
    std::string linker = [] -> std::string {
#ifndef _WIN32
        // 'ld' is terrible, so try 'mold' or 'lld' first.
        if (auto mold = llvm::sys::findProgramByName("mold")) return *mold;
        if (auto lld = llvm::sys::findProgramByName("lld")) return *lld;
        return "";
#else
        return "";
#endif
    }();

    // Derive the file name from the program name; for modules, always
    // use '.mod' instead of e.g. '.a' because linking against a module
    // isn’t enough to make it usable: you also have to run the module
    // initialiser.
    std::string out_name{tu.name.sv()};
    if (tu.is_module) out_name = std::format("{}.mod", out_name);
    else if (machine.getTargetTriple().isOSWindows()) out_name += ".exe";

    // If we’re compiling a module, create a static archive.
    // TODO: And a shared library.
    if (tu.is_module) {
        // We could avoid writing to a temporary file in this case (but
        // we’d have to add files anyway if we’re combining modules into
        // a collection; would it be worth it?).
        OpenFile archive((tu.context().module_path() / out_name).string());
        auto obj = llvm::NewArchiveMember::getFile(object_file_path, true);
        Assert(obj, "Failed to read the file we just wrote?");
        auto err = writeArchiveToStream(
            archive,
            obj.get(),
            llvm::SymtabWritingMode::NormalSymtab,
            llvm::object::Archive::getDefaultKindForTriple(const_cast<llvm::Triple&>(machine.getTargetTriple())),
            true,
            false
        );

        // Yes, this is truthy on failure for some ungodly reason.
        if (not err) Fatal("Failed to write archive: {}", utils::FormatError(err));
        return 0;
    }

    // Collect args.
    SmallVector<std::string> clang_link_args;
    clang_link_args.push_back(SOURCE_CLANG_EXE);
    clang_link_args.push_back(object_file_path);
    clang_link_args.push_back("-o");
    clang_link_args.push_back(std::move(out_name));
    if (not linker.empty()) clang_link_args.push_back(std::format("-fuse-ld={}", linker));

    SmallVector<StringRef> args_ref;
    for (auto& arg : clang_link_args) args_ref.push_back(arg);

    // We could run the linker without waiting for it, but that defeats
    // the purpose of making the number of jobs configurable, so block
    // until it’s done.
    return llvm::sys::ExecuteAndWait(SOURCE_CLANG_EXE, args_ref);
}

