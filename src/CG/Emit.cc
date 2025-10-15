#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Constants.hh>

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

#include <base/Assert.hh>

#include <print>

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
void cg::CodeGen::optimise(llvm::TargetMachine& machine, TranslationUnit&, llvm::Module& m) {
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

int cg::CodeGen::write_to_file(
    llvm::TargetMachine& machine,
    TranslationUnit& tu,
    llvm::Module& m,
    ArrayRef<std::string> lib_paths,
    ArrayRef<std::string> link_libs,
    ArrayRef<std::string> additional_objects,
    StringRef program_file_name_override
) {
    // Create a temporary path for the object file.
    std::string object_file_path = fs::TempPath(".o");

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

    // If we’re compiling a module, create a static archive.
    // TODO: And a shared library.
    if (tu.is_module) {
        // Derive the file name from the module name; always use '.mod' instead
        // of e.g. '.a' because linking against a module isn’t enough to make it
        // usable: you also have to run the module initialiser.
        auto out_name = std::format("{}.{}", tu.name.sv(), constants::ModuleFileExtension);
        auto desc_name = std::format("{}.{}", tu.name.sv(), constants::ModuleDescriptionFileExtension);

        // We could avoid writing to a temporary file in this case (but
        // we’d have to add files anyway if we’re combining modules into
        // a collection; would it be worth it?).
        OpenFile archive((tu.context().module_path() / out_name).string());

        // Collect members.
        SmallVector<llvm::NewArchiveMember> members;
        auto us = llvm::NewArchiveMember::getFile(object_file_path, true);
        Assert(us, "Failed to read the file we just wrote?");
        us->MemberName = tu.name;
        members.push_back(std::move(*us));
        for (auto& obj_path : additional_objects) {
            auto obj = llvm::NewArchiveMember::getFile(obj_path, true);
            if (auto e = obj.takeError()) Fatal("Failed to read object file '{}': {}", obj_path, utils::FormatError(e));
            members.push_back(std::move(*obj));
        }

        // Emit the archive.
        auto err = writeArchiveToStream(
            archive,
            members,
            llvm::SymtabWritingMode::NormalSymtab,
            llvm::object::Archive::getDefaultKindForTriple(machine.getTargetTriple()),
            true,
            false
        );

        // Yes, this is truthy on failure for some ungodly reason.
        if (err) Fatal("Failed to write archive: {}", utils::FormatError(err));

        // Emit the module description.
        auto desc = tu.serialise();
        auto res = File::Write(desc.data(), desc.size(), tu.context().module_path() / desc_name);
        if (not res) Fatal("Failed to write module description: {}", res.error());
        return 0;
    }

    // Determine the file name; For programs, we allow overriding this with a
    // user-defined output file name.
    std::string out_name{tu.name.sv()};
    if (not program_file_name_override.empty()) out_name = program_file_name_override;
    else if (machine.getTargetTriple().isOSWindows()) out_name += ".exe";

    // Collect args.
    SmallVector<std::string> clang_link_args;
    clang_link_args.push_back(SOURCE_CLANG_EXE);
    clang_link_args.push_back(object_file_path);
    clang_link_args.push_back("-o");
    clang_link_args.push_back(std::move(out_name));
    if (not linker.empty()) clang_link_args.push_back(std::format("-fuse-ld={}", linker));

    SmallVector<StringRef> args_ref;
    for (auto& arg : clang_link_args) args_ref.push_back(arg);
    for (auto& obj : additional_objects) args_ref.push_back(obj);
    for (auto& obj : lib_paths) {
        args_ref.push_back("-L");
        args_ref.push_back(obj);
    }

    for (auto& obj : link_libs) {
        args_ref.push_back("-l");
        args_ref.push_back(obj);
    }

    for (auto& [_, import] : tu.linkage_imports)
        if (auto src_mod = dyn_cast<ImportedSourceModuleDecl>(import))
            args_ref.push_back(src_mod->mod_path);

    // We could run the linker without waiting for it, but that defeats
    // the purpose of making the number of jobs configurable, so block
    // until it’s done.
    int code = llvm::sys::ExecuteAndWait(SOURCE_CLANG_EXE, args_ref);
    if (code != 0) std::println(
        "Linker invocation: {}",
        utils::join(utils::quote_escaped(args_ref), " ")
    );
    return code;
}


