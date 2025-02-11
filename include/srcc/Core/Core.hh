#ifndef SRCC_CORE_HH
#define SRCC_CORE_HH

#include <srcc/Core/Location.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
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
class DiagnosticsEngine;
class File;
struct LangOpts;
} // namespace srcc

/// All members of 'Context' are thread-safe.
/// FIXME: Move all the members up into this and remove Impl.
class srcc::Context {
    SRCC_DECLARE_HIDDEN_IMPL(Context);

public:
    /// Create a new context with default options.
    explicit Context();

    /// Create a machine for this target.
    [[nodiscard]] auto create_target_machine() const -> std::unique_ptr<llvm::TargetMachine>;

    /// Get diagnostics engine.
    [[nodiscard]] auto diags() const -> DiagnosticsEngine&;

    /// Enable or disable coloured output.
    void enable_colours(bool enable);

    /// Enable or disable short filenames.
    void enable_short_filenames(bool enable);

    /// Get the number of steps the constant evaluator can run for.
    [[nodiscard]] auto eval_steps() const -> u64;

    /// Get a file by index. Returns nullptr if the index is out of bounds.
    [[nodiscard]] auto file(usz idx) const -> const File*;

    /// Get the appropriate filename for a file id.
    [[nodiscard]] auto file_name(i32 id) const -> String;

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

    /// Set the maximum step count for the constant evaluator.
    void set_eval_steps(u64 steps);

    /// Whether to enable coloured output.
    [[nodiscard]] bool use_colours() const;

    /// Whether to enable short filenames.
    [[nodiscard]] bool use_short_filenames() const;
};

struct srcc::LangOpts {
    // Enable overflow checking. When disabled, overflow is
    // undefined behaviour instead.
    bool overflow_checking : 1 = true;
};

/// A file in the context.
class srcc::File {
    SRCC_IMMOVABLE(File);

    /// Context handle.
    Context& ctx;

    /// The absolute file path.
    fs::Path file_path;

    /// The name of the file as specified on the command line.
    String file_name;
    String short_file_name;

    /// The contents of the file.
    std::unique_ptr<llvm::MemoryBuffer> contents;

    /// The id of the file.
    const i32 id;

public:
    /// Get an iterator to the beginning of the file.
    [[nodiscard]] auto begin() const { return contents->getBufferStart(); }

    /// Get the owning context.
    [[nodiscard]] auto context() const -> Context& { return ctx; }

    /// Get the file data.
    [[nodiscard]] auto data() const -> const char* { return contents->getBufferStart(); }

    /// Get an iterator to the end of the file.
    [[nodiscard]] auto end() const { return contents->getBufferEnd(); }

    /// Get the id of this file.
    [[nodiscard]] auto file_id() const { return id; }

    /// Get the short file name.
    [[nodiscard]] auto name() const -> String { return file_name; }

    /// Get the file path.
    [[nodiscard]] auto path() const -> fs::PathRef { return file_path; }

    /// Get the short file name.
    [[nodiscard]] auto short_name() const -> String { return short_file_name; }

    /// Get the size of the file.
    [[nodiscard]] auto size() const -> isz { return isz(contents->getBufferSize()); }

    /// Get a temporary file path.
    [[nodiscard]] static auto TempPath(StringRef extension) -> fs::Path;

    /// Write to a file on disk.
    [[nodiscard]] static auto Write(
        const void* data,
        usz size,
        fs::PathRef file
    ) -> std::expected<void, std::string>;

    /// Write to a file on disk and terminate on error.
    static void WriteOrDie(void* data, usz size, fs::PathRef file);

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
    static auto LoadFileData(fs::PathRef path) -> std::unique_ptr<llvm::MemoryBuffer>;
};

#endif // SRCC_CORE_HH
