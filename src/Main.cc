#include <clopts.hh>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Signals.h>
#include <thread>

#ifdef __linux__
#    include <csignal>
#endif

import srcc;
import srcc.utils;
import srcc.driver;

using namespace srcc;

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,

    // General options.
    option<"--colour", "Enable or disable coloured output (default: auto)", values<"auto", "always", "never">>,
    option<"--error-limit", "Limit how many errors are printed; passing 0 removes the limit", std::int64_t>,
    option<"--mo", "Path to a directory where compiled modules will be placed (default: '.')">,
    option<"-o", "Override the default output file name">,
    multiple<option<"--link-object", "Link a compiled object file into every TU that is part of this compilation">>,
    multiple<experimental::short_option<"-M", "Path to a directory that should be searched for compiled modules">>,
    experimental::short_option<"-j", "Number of threads to use for compilation", std::int64_t>,
    experimental::short_option<"-O", "Optimisation level", values<0, 1, 2, 3, 4>>,

    // General flags.
    flag<"--ast", "Dump the parse tree / AST">,
    flag<"--dump-module", "Dump the contents of a module or C++ header that we can import">,
    flag<"--lex", "Lex tokens only (but do not print them) and exit">,
    flag<"--tokens", "Print tokens and exit">,
    flag<"--parse", "Parse only and exit">,
    flag<"--sema", "Run sema only and exit">,
    flag<"--verify", "Run in verify-diagnostics mode">,
    flag<"--eval", "Run the entire input through the constant evaluator">,
    flag<"--llvm", "Emit LLVM IR">,
    flag<"--noruntime", "Do not automatically import the runtime module">,

    // Features.
    // TODO: Consider: short_option<"-f, "Enable or disable a feature", values<"overflow-checks">> or
    // something in that vein.
    flag<"-fno-overflow-checks">,

    help<>
>; // clang-format on
}

#ifdef __linux__
std::atomic_bool colours_enabled;
void InitSignalHandlers() {
    signal(SIGSEGV, [](int) {
        auto msg = colours_enabled.load(std::memory_order_relaxed)
                     ? "\033[1;35mInternal Compiler Error: \033[m\033[1mSegmentation fault\033[m"sv
                     : "Internal Compiler Error: Segmentation fault";

        llvm::errs() << msg << "\n";
        llvm::sys::PrintStackTrace(llvm::errs());
        _Exit(1);
    });
}
#endif

int main(int argc, char** argv) {
    llvm::sys::DisableSystemDialogsOnCrash();
    auto opts = ::detail::options::parse(argc, argv);

    // Enable colours.
    auto colour_opt = opts.get_or<"--colour">("auto");
    bool use_colour = colour_opt == "never"  ? false
                    : colour_opt == "always" ? true
                                             : isatty(fileno(stderr)) && isatty(fileno(stdout)); // FIXME: Cross-platform


#ifdef __linux__
    if (use_colour) colours_enabled.store(true, std::memory_order_relaxed);
    InitSignalHandlers();
#endif

    // Figure out what we want to do.
    auto action = opts.get<"--eval">()        ? Action::Eval
                : opts.get<"--dump-module">() ? Action::DumpModule
                : opts.get<"--lex">()         ? Action::Lex
                : opts.get<"--llvm">()        ? Action::EmitLLVM
                : opts.get<"--parse">()       ? Action::Parse
                : opts.get<"--sema">()        ? Action::Sema
                : opts.get<"--tokens">()      ? Action::DumpTokens
                                              : Action::Compile;

    // Collect module search paths.
    std::vector<std::string> module_search_paths{
        opts.get<"-M">().begin(),
        opts.get<"-M">().end(),
    };

    module_search_paths.push_back(SOURCE_PROJECT_DIR_NAME "/modules");

    // Create driver.
    Driver driver{{
        .module_output_path = opts.get_or<"--mo">("."),
        .output_file_name = opts.get_or<"-o">(""),
        .module_search_paths = std::move(module_search_paths),

        .link_objects = std::vector<std::string>{
            opts.get<"--link-object">().begin(),
            opts.get<"--link-object">().end(),
        },

        .action = action,
        .error_limit = u32(opts.get_or<"--error-limit">(20)),
        .num_threads = u32(opts.get_or<"-j">(std::thread::hardware_concurrency())),
        .opt_level = u8(opts.get_or<"-O">(0)),
        .print_ast = opts.get<"--ast">(),
        .verify = opts.get<"--verify">(),
        .colours = use_colour,
        .overflow_checking = not opts.get<"-fno-overflow-checks">(),
        .import_runtime = not opts.get<"--noruntime">()
    }};

    // Add files.
    driver.add_file(*opts.get<"file">());

    // Dew it.
    return driver.run_job();
}
