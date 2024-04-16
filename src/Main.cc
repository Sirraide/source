#include <clopts.hh>

import srcc;
import srcc.utils;
import srcc.driver;

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">
>; // clang-format on
}

int main(int argc, char** argv) {
    auto opts = detail::options::parse(argc, argv);
    srcc::Driver driver;
    driver.add_file(*opts.get<"file">());
    return driver.compile();
}
