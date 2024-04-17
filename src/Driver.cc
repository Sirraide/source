// ============================================================================
//                                   DRIVER
//
// This file implements the Source compiler driver. The driver is the main
// entry point for the compiler and is responsible for creating the compiler
// context, dispatching work amongst threads and managing the compilation
// process.
//
// Some invariants:
//
//   - All public members of 'Driver' are thread-safe and must lock the
//     driver mutex upon entry, unless they only access 'impl->ctx' or
//     atomic variables.
//
//   - Members of 'Driver' MUST NOT be called by other members.
//
//   - No member of 'Driver::Impl' may lock the driver mutex.
//
// ============================================================================
module;

#include <algorithm>
#include <filesystem>
#include <fmt/std.h>
#include <future>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ThreadPool.h>
#include <mutex>
#include <ranges>
#include <srcc/Macros.hh>
#include <string>
#include <unordered_set>

module srcc.driver;
import srcc;
import srcc.frontend.parser;

using namespace srcc;

// ============================================================================
//  Internals
// ============================================================================
struct Driver::Impl {
    SmallVector<File::Path> files;
    llvm::DefaultThreadPool thread_pool;
    Context ctx;
    std::mutex mutex;
    bool compiled = false;

    int compile();

    auto ParseFile(const File::Path& path) -> ParsedModule*;
};

int Driver::Impl::compile() {
    Assert(not compiled, "Can only call compile() once per Driver instance!");
    compiled = true;

    // Duplicate TUs would create horrible linker errors.
    // FIXME: Use inode instead?
    std::unordered_set<File::Path> file_uniquer;
    for (const auto& f : files)
        if (not file_uniquer.insert(canonical(f)).second)
            Diag::Fatal("Duplicate file name in command-line: '{}'", canonical(f));

    // Parse files in parallel.
    SmallVector<std::shared_future<ParsedModule*>> futures;
    for (const auto& f : files) futures.push_back(thread_pool.async([&] { return ParseFile(f); }));

    // Wait for all modules to be parsed.
    thread_pool.wait();

    // Collect modules.
    SmallVector<std::unique_ptr<ParsedModule>> parsed_modules;
    for (auto& f : futures) {
        auto ptr = f.get();
        if (not ptr) return 1;
        parsed_modules.emplace_back(ptr);
    }

    // Dump them for now.
    for (const auto& m : parsed_modules) m->dump();
    return 42;
}

auto Driver::Impl::ParseFile(const File::Path& path) -> ParsedModule* {
    auto& f = ctx.get_file(path);
    if (auto res = Parser::Parse(f)) return res->release();
    return nullptr;
}

// ============================================================================
//  API
// ============================================================================
SRCC_DEFINE_HIDDEN_IMPL(Driver);
Driver::Driver() : impl(new Impl) {}

void Driver::add_file(std::string_view file_path) {
    std::unique_lock _{impl->mutex};
    impl->files.push_back(File::Path(file_path));
}

int Driver::compile() {
    std::unique_lock _{impl->mutex};
    return impl->compile();
}

void Driver::enable_colours(bool enable) {
    impl->ctx.enable_colours(enable);
}
