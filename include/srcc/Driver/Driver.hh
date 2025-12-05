#ifndef SRCC_DRIVER_HH
#define SRCC_DRIVER_HH

#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

namespace srcc {
class Driver;
enum struct Action : u8;
}

enum struct srcc::Action : srcc::u8 {
    /// Does what you’d expect: compile all input files to
    /// executables and modules and save them to disk.
    Compile,

    /// Dump the exports of the module being compiled.
    DumpExports,

    /// Dump the contents of the module.
    DumpModule,

    /// Run codegen and dump our codegen IR.
    DumpIR,

    /// Print tokens.
    DumpTokens,

    /// Run the input through the constant evaluator, as
    /// though the entire file were wrapped in an eval {}
    /// block.
    Eval,

    /// As 'Eval', but also dump the resulting IR.
    EvalDumpIR,

    /// Lex tokens only and exit.
    Lex,

    /// Emit LLVM IR.
    EmitLLVM,

    /// Parse only and exit.
    Parse,

    /// Run sema only and exit.
    Sema,
};

class srcc::Driver : DiagsProducer {
public:
    struct Options {
        /// Target triple.
        llvm::Triple triple;

        /// The path to a directory where modules should be stored.
        std::string module_output_path;

        /// Output file name override. Only valid for programs.
        std::string output_file_name;

        /// Directories to search for modules.
        std::span<std::string> module_search_paths;

        /// Directories to add to the C/C++ header search path.
        std::span<std::string> clang_include_paths;

        /// Directories to add to the native library search path.
        std::span<std::string> lib_paths;

        /// Libraries to link against.
        std::span<std::string> link_libs;

        /// Additional objects to link in.
        ///
        /// Unlike 'link_libs', these are directly passed to the linker
        /// as-is rather than via '-l'.
        std::span<std::string> link_objects;

        /// The action to perform.
        Action action;

        /// Language options.
        LangOpts lang_opts;

        /// How many steps the constant evaluator can run for before
        /// we give up.
        u64 eval_steps;

        /// How many errors are printed before we stop printing them
        /// altogether. Set to 0 to disable.
        u32 error_limit;

        /// Optimisation level.
        u8 opt_level;

        /// Whether to print the AST as part of the job. If the action
        /// specified as Parse or Sema, this only prints the parse tree
        /// or AST, respectively.
        bool print_ast : 1;

        /// Whether to run in verify-diagnostics mode.
        bool verify : 1;

        /// Whether to use colours in the output.
        bool colours : 1;

        /// Whether to use short filenames (e.g. 'foo.src' instead of
        /// 'path/to/foo.src').
        bool short_filenames : 1;

        /// IR dumping flags.
        bool ir_generic : 1 = false;
        bool ir_no_finalise : 1 = false;
        bool ir_verbose : 1 = false;
    };

private:
    Options opts;
    SmallVector<fs::Path> files;
    Context ctx;
    bool compiled = false;

public:
    /// Create a new driver.
    explicit Driver(Options opts) : opts{std::move(opts)} {}

    /// Add a file to the list of files to compile.
    void add_file(std::string_view file_path);

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& {  return ctx.diags(); }

    /// Run compile jobs.
    ///
    /// \return 0 on success, non-zero on failure.
    int run_job();

private:
    template <typename... Args>
    void Diag(Diagnostic::Level level, SLoc loc, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(level, loc, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    int Error(std::format_string<Args...> fmt, Args&&... args) {
        Diag(Diagnostic::Level::Error, SLoc(), fmt, std::forward<Args>(args)...);
        return 1;
    }

    /// Parse a file and return the parsed module.
    auto ParseFile(fs::PathRef path, bool verify) -> ParsedModule::Ptr;
};

#endif // SRCC_DRIVER_HH
