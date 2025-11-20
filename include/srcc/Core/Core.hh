#ifndef SRCC_CORE_HH
#define SRCC_CORE_HH

#include <srcc/Core/Location.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

#include <base/FS.hh>

#include <expected>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>

namespace srcc {
class Context;
class Driver;
class DiagnosticsEngine;
class File;
struct LangOpts;
using FileId = i32;
} // namespace srcc

class srcc::Context {
    friend Driver;

    /// Module dir.
    fs::Path module_dir;

    /// Diagnostics engine.
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags_engine;

    /// Target.
    llvm::Triple target_triple;

    /// Files loaded by the context.
    std::vector<std::unique_ptr<File>> all_files;
    std::unordered_map<fs::Path, File*> files_by_path; // FIXME: use inode number instead.

    /// For saving strings.
    llvm::BumpPtrAllocator alloc;
    llvm::StringSaver saver{alloc};

public:
    /// Constant evaluator steps.
    u64 eval_steps = 1 << 20;

    /// Optimisation level.
    int opt_level = 0;

    /// Whether to use coloured output.
    bool use_colours = true;

    /// Whether to use short filenames.
    bool use_short_filenames = false;

    /// Create a new context with default options.
    explicit Context();

    /// Create a machine for this target.
    [[nodiscard]] auto create_target_machine() const -> std::unique_ptr<llvm::TargetMachine>;

    /// Create a virtual file.
    [[nodiscard]] auto create_virtual_file(
        std::unique_ptr<llvm::MemoryBuffer> data,
        fs::PathRef name = "<virtual>"
    ) -> const File&;

    [[nodiscard]] auto create_virtual_file(
        StringRef data,
        fs::PathRef name = "<virtual>"
    ) -> const File&;

    /// Get diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine&;

    /// Get a file by index. Returns nullptr if the index is out of bounds.
    [[nodiscard]] auto file(FileId idx) const -> const File*;

    /// Get all files
    [[nodiscard]] auto files() const {
        return all_files | vws::transform([](auto& f) { return f.get(); });
    }

    /// Get the appropriate filename for a file id.
    [[nodiscard]] auto file_name(FileId id) const -> String;

    /// Get a file from disk.
    ///
    /// This will load the file the first time it is requested.
    [[nodiscard]] auto get_file(fs::PathRef path) -> const File&;

    /// DO NOT CALL THIS. This is only here for the driver.
    void _initialise_context_(fs::Path module_path, int opt_level);

    /// The directory in which modules should be placed.
    [[nodiscard]] auto module_path() const -> fs::PathRef;

    /// Set the diagnostics engine.
    void set_diags(llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags);

    /// Get the target triple.
    [[nodiscard]] auto triple() const -> const llvm::Triple& { return target_triple; }

    /// Attempt to get a file from disk.
    ///
    /// This will load the file the first time it is requested.
    [[nodiscard]] auto try_get_file(fs::PathRef path) -> Result<const File&>;
};

struct srcc::LangOpts {
    /// Enable overflow checking. When disabled, overflow is
    /// undefined behaviour instead.
    bool overflow_checking : 1 = true;

    /// We’re compiling for constant evaluation.
    bool constant_eval : 1 = false;

    /// We’re compiling without a runtime; usually, this means
    /// that we’re actually compiling the runtime.
    bool no_runtime: 1 = false;
};

/// A file in the context.
class srcc::File {
    SRCC_IMMOVABLE(File);

public:
    using Id = FileId;

private:
    /// Context handle.
    Context& ctx;

    /// The absolute file path.
    fs::Path file_path;

    /// The name of the file as specified on the command line.
    String file_name;
    String short_file_name;

    /// The contents of the file.
    std::unique_ptr<llvm::MemoryBuffer> buffer;

    /// The id of the file.
    const Id id;

public:
    /// Get an iterator to the beginning of the file.
    [[nodiscard]] auto begin() const { return buffer->getBufferStart(); }

    /// Get the owning context.
    [[nodiscard]] auto context() const -> Context& { return ctx; }

    /// Get the file contents.
    [[nodiscard]] auto contents() const -> String {
        return String::CreateUnsafe(StringRef(data(), usz(size())));
    }

    /// Get the file data.
    [[nodiscard]] auto data() const -> const char* { return buffer->getBufferStart(); }

    /// Get an iterator to the end of the file.
    [[nodiscard]] auto end() const { return buffer->getBufferEnd(); }

    /// Get the id of this file.
    [[nodiscard]] auto file_id() const -> Id { return id; }

    /// Get the short file name.
    [[nodiscard]] auto name() const -> String { return file_name; }

    /// Get the file path.
    [[nodiscard]] auto path() const -> fs::PathRef { return file_path; }

    /// Get the short file name.
    [[nodiscard]] auto short_name() const -> String { return short_file_name; }

    /// Get the size of the file.
    [[nodiscard]] auto size() const -> isz { return isz(buffer->getBufferSize()); }

    /// Write to a file on disk.
    [[nodiscard]] static auto Write(
        const void* data,
        usz size,
        fs::PathRef file
    ) -> std::expected<void, std::string>;

    /// Write to a file on disk and terminate on error.
    static void WriteOrDie(void* data, usz size, fs::PathRef file);

    /// Two files are equal if they’re the same file.
    [[nodiscard]] friend auto operator==(const File& a, const File& b) -> bool {
        return a.file_id() == b.file_id();
    }

private:
    /// The context is the only thing that can create files.
    friend Context;

    /// Construct a file from a name and source.
    explicit File(
        Context& ctx,
        fs::Path path,
        String name,
        std::unique_ptr<llvm::MemoryBuffer> contents,
        u16 id
    );

    /// Load a file from disk.
    static auto LoadFileData(fs::PathRef path) -> Result<std::unique_ptr<llvm::MemoryBuffer>>;
};

#endif // SRCC_CORE_HH
