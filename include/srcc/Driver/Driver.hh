#ifndef SRCC_DRIVER_HH
#define SRCC_DRIVER_HH

#include <srcc/Macros.hh>
#include <srcc/Core/Utils.hh>

namespace srcc {
class Driver;
enum struct Action : u8;
}

enum struct srcc::Action : srcc::u8 {
    /// Does what you’d expect: compile all input files to
    /// executables and modules and save them to disk.
    Compile,

    /// Dump the contents of the module.
    DumpModule,

    /// Print tokens.
    DumpTokens,

    /// Run the input through the constant evaluator, as
    /// though the entire file were wrapped in an eval {}
    /// block.
    Eval,

    /// Lex tokens only and exit.
    Lex,

    /// Emit LLVM IR.
    EmitLLVM,

    /// Parse only and exit.
    Parse,

    /// Run sema only and exit.
    Sema,
};

class srcc::Driver {
    SRCC_DECLARE_HIDDEN_IMPL(Driver);

public:
    struct Options {
        /// The path to a directory where modules should be stored.
        std::string module_output_path;

        /// Output file name override. Only valid for programs.
        std::string output_file_name;

        /// Directories to search for modules.
        std::vector<std::string> module_search_paths;

        /// Additional objects to link in.
        std::vector<std::string> link_objects;

        /// The action to perform.
        Action action;

        /// How many steps the constant evaluator can run for before
        /// we give up.
        u64 eval_steps;

        /// How many errors are printed before we stop printing them
        /// altogether. Set to 0 to disable.
        u32 error_limit;

        /// Number of threads to use for compilation.
        u32 num_threads;

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

        /// Whether to perform overflow checking.
        bool overflow_checking : 1;

        /// Whether to implicitly import the runtime module.
        bool import_runtime : 1;
    };

    /// Create a new driver.
    explicit Driver(Options opts);

    /// Add a file to the list of files to compile.
    void add_file(std::string_view file_path);

    /// Run compile jobs.
    ///
    /// \return 0 on success, non-zero on failure.
    int run_job();
};

#endif // SRCC_DRIVER_HH
