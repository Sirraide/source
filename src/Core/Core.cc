#include <srcc/Core/Core.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

#include <base/FS.hh>

#include <generator>
#include <mutex>
#include <random>
#include <thread>

#ifdef __linux__
#    include <unistd.h>
#endif

using namespace srcc;

// ============================================================================
//  Context
// ============================================================================
Context::Context() {
    static std::once_flag init;
    std::call_once(init, [] {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        const char* args[]{
            "srcc",
            "-x86-asm-syntax=intel",
            nullptr,
        };

        llvm::cl::ParseCommandLineOptions(2, args, "", &llvm::errs(), nullptr);

        // Open the current process as a library.
        std::string err_msg;
        Assert(
            not llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr, &err_msg),
            "Failed to load libc: {}",
            err_msg
        );
    });
}

auto Context::create_target_machine() const -> std::unique_ptr<llvm::TargetMachine> {
    // No need to acquire a lock since we don’t access any shared
    // shared state (except opt_level, which is never written to).
    auto triple = llvm::sys::getDefaultTargetTriple();

    // Get the target.
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (not error.empty() or not target) Fatal(
        "Failed to lookup target triple '{}': {}",
        triple,
        error
    );

    // Get feature flags.
    std::string features;
    if (opt_level == 4) {
        StringMap<bool> feature_map = llvm::sys::getHostCPUFeatures();
        for (auto& [feature, enabled] : feature_map)
            if (enabled)
                features += std::format("+{},", feature.str());
    }

    // User-specified features are applied last.
    // for (auto& [feature, enabled] : target_features)
    //    features += std::format("{}{},", enabled ? '+' : '-', feature.str());
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
        llvm::Triple(triple),
        cpu,               // Target CPU
        features,          // Features.
        opts,              // Options.
        llvm::Reloc::PIC_, // Relocation model.
        std::nullopt,      // Code model.
        opt                // Opt level.
    )};

    return machine;
}

auto Context::create_virtual_file(std::unique_ptr<llvm::MemoryBuffer> data) -> const File& {
    auto f = new File(*this, "<virtual>", "<virtual>", std::move(data), u16(all_files.size()));
    all_files.emplace_back(f);
    return *f;
}

auto Context::diags() const -> DiagnosticsEngine& {
    Assert(diags_engine, "Diagnostics engine not set!");
    return *diags_engine;
}

auto Context::file(File::Id idx) const -> const File* {
    if (usz(idx) >= all_files.size()) return nullptr;
    return all_files[usz(idx)].get();
}

auto Context::file_name(File::Id id) const -> String {
    auto* f = file(id);
    return use_short_filenames ? f->short_name() : f->name();
}

auto Context::get_file(fs::PathRef path) -> const File& {
    return try_get_file(path).value();
}

void Context::_initialise_context_(fs::Path module_path, int optimisation_level) {
    module_dir = std::move(module_path);
    opt_level = optimisation_level;
}

auto Context::module_path() const -> fs::PathRef {
    return module_dir;
}

void Context::set_diags(llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags) {
    if (diags_engine) diags_engine->flush();
    diags_engine = std::move(diags);
}

auto Context::try_get_file(fs::PathRef path) -> Result<const File&> {
    std::error_code ec;
    auto can = canonical(path, ec);
    if (ec) return Error(
        "Failed to canonicalise file path '{}': {}",
        path,
        ec.message()
    );

    if (auto it = files_by_path.find(can); it != files_by_path.end())
        return *it->second;

    static constexpr usz MaxFiles = std::numeric_limits<u16>::max();
    Assert(
        all_files.size() < MaxFiles,
        "Sorry, that’s too many files for us! (max is {})",
        MaxFiles
    );

    auto mem = Try(File::LoadFileData(can));
    auto f = new File(*this, can, String::Save(saver, path.string()), std::move(mem), u16(all_files.size()));
    all_files.emplace_back(f);
    files_by_path[std::move(can)] = f;
    return *f;
}


// ============================================================================
//  File
// ============================================================================
auto srcc::File::Write(const void* data, usz size, fs::PathRef file) -> std::expected<void, std::string> {
    auto err = llvm::writeToOutput(absolute(file).string(), [=](llvm::raw_ostream& os) {
        os.write(static_cast<const char*>(data), size);
        return llvm::Error::success();
    });

    if (err) return Error("Failed to write to file '{}': {}", file, utils::FormatError(err));
    return {};
}

void srcc::File::WriteOrDie(void* data, usz size, fs::PathRef file) {
    if (not Write(data, size, file)) Fatal(
        "Failed to write to file '{}': {}",
        file,
        std::strerror(errno)
    );
}

srcc::File::File(
    Context& ctx,
    fs::Path path,
    String name,
    std::unique_ptr<llvm::MemoryBuffer> contents,
    u16 id
) : ctx(ctx),
    file_path(std::move(path)),
    file_name(name),
    buffer(std::move(contents)),
    id(id) {
    short_file_name = String::CreateUnsafe(str{name}.take_back_until_any("/\\").text());
}

auto srcc::File::LoadFileData(fs::PathRef path) -> Result<std::unique_ptr<llvm::MemoryBuffer>> {
    auto buf = llvm::MemoryBuffer::getFile(
        path.string(),
        true,
        false
    );

    if (auto ec = buf.getError()) return Error(
        "Could not load file '{}': {}",
        path,
        ec.message()
    );

    // Construct the file data.
    return std::move(*buf);
}
