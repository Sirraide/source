#include <../out/libs/clopts/include/clopts.hh>

import srcc;
import srcc.utils;
import srcc.driver;

using namespace srcc;

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,
    option<"--colour", "Enable or disable coloured output (default: auto)", values<"auto", "always", "never">>,
    flag<"--parse-tree", "Dump the parse tree">,
    flag<"--ast", "Dump the AST">,
    help<>
>; // clang-format on
}

int main(int argc, char** argv) {
    auto opts = ::detail::options::parse(argc, argv);
    Driver driver;

    // Enable colours.
    auto colour_opt = opts.get_or<"--colour">("auto");
    bool use_colour = colour_opt == "never"  ? false
                    : colour_opt == "always" ? true
                                             : isatty(fileno(stderr)) && isatty(fileno(stdout)); // FIXME: Cross-platform
    driver.enable_colours(use_colour);

    // Add files.
    driver.add_file(*opts.get<"file">());

    // Figure out what we want to do.
    auto action = opts.get<"--parse-tree">() ? Action::Parse
                : opts.get<"--ast">()        ? Action::AST
                                             : Action::Compile;

    // Dew it.
    return driver.run_job(action);
}
