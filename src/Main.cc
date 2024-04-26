#include <../out/libs/clopts/include/clopts.hh>

import srcc;
import srcc.utils;
import srcc.driver;

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,
    option<"--colour", "Enable or disable coloured output (default: auto)", values<"auto", "always", "never">>,
    help<>
>; // clang-format on
}

int main(int argc, char** argv) {
    auto opts = detail::options::parse(argc, argv);
    srcc::Driver driver;

    // Enable colours.
    auto colour_opt = opts.get_or<"--colour">("auto");
    bool use_colour = colour_opt == "never"  ? false
                    : colour_opt == "always" ? true
                                             : isatty(fileno(stderr)) && isatty(fileno(stdout)); // FIXME: Cross-platform
    driver.enable_colours(use_colour);

    // Add files.
    driver.add_file(*opts.get<"file">());

    // Dew it.
    return driver.compile();
}
