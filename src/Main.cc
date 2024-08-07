#include <clopts.hh>
#include <thread>

import srcc;
import srcc.utils;
import srcc.driver;

using namespace srcc;

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,
    option<"--colour", "Enable or disable coloured output (default: auto)", values<"auto", "always", "never">>,
    option<"--error-limit", "Limit how many errors are printed; passing 0 removes the limit", std::int64_t>,
    experimental::short_option<"-j", "Number of threads to use for compilation", std::int64_t>,
    flag<"--ast", "Dump the parse tree / AST">,
    flag<"--lex", "Lex tokens only (but do not print them) and exit">,
    flag<"--tokens", "Print tokens and exit">,
    flag<"--parse", "Parse only and exit">,
    flag<"--sema", "Run sema only and exit">,
    flag<"--verify", "Run in verify-diagnostics mode">,
    flag<"--eval", "Run the entire input through the constant evaluator">,
    help<>
>; // clang-format on
}

int main(int argc, char** argv) {
    auto opts = ::detail::options::parse(argc, argv);

    // Enable colours.
    auto colour_opt = opts.get_or<"--colour">("auto");
    bool use_colour = colour_opt == "never"  ? false
                    : colour_opt == "always" ? true
                                             : isatty(fileno(stderr)) && isatty(fileno(stdout)); // FIXME: Cross-platform

    // Figure out what we want to do.
    auto action = opts.get<"--eval">()   ? Action::Eval
                : opts.get<"--tokens">() ? Action::DumpTokens
                : opts.get<"--lex">()    ? Action::Lex
                : opts.get<"--parse">()  ? Action::Parse
                : opts.get<"--sema">()   ? Action::Sema
                                         : Action::Compile;

    // Create driver.
    Driver driver{{
        .action = action,
        .error_limit = u32(opts.get_or<"--error-limit">(20)),
        .num_threads = u32(opts.get_or<"-j">(std::thread::hardware_concurrency())),
        .print_ast = opts.get<"--ast">(),
        .verify = opts.get<"--verify">(),
        .colours = use_colour,
    }};

    // Add files.
    driver.add_file(*opts.get<"file">());

    // Dew it.
    return driver.run_job();
}
