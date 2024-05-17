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
import srcc.frontend.sema;
import srcc.codegen;

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

    int run_job(Action action);

    auto ParseFile(const File::Path& path) -> ParsedModule*;
};

int Driver::Impl::run_job(Action action) {
    Assert(not compiled, "Can only call compile() once per Driver instance!");
    compiled = true;

    // Create diags engine.
    ctx.set_diags(StreamingDiagnosticsEngine::Create(ctx));

    // Duplicate TUs would create horrible linker errors.
    // FIXME: Use inode instead?
    std::unordered_set<File::Path> file_uniquer;
    for (const auto& f : files) {
        auto can = canonical(f);
        if (not file_uniquer.insert(can).second) Fatal(
            "Duplicate file name in command-line: '{}'",
            can
        );
    }

    // Parse files in parallel.
    SmallVector<std::shared_future<ParsedModule*>> futures;
    for (const auto& f : files) futures.push_back(thread_pool.async([&] { return ParseFile(f); }));

    // Wait for all modules to be parsed.
    thread_pool.wait();

    // Collect modules.
    SmallVector<std::unique_ptr<ParsedModule>> parsed_modules;
    for (auto& f : futures) {
        auto ptr = f.get();
        parsed_modules.emplace_back(ptr);
    }

    // Dump modules if we should only do parsing.
    if (action == Action::Parse) {
        for (auto& m : parsed_modules) m->dump();
        return 0;
    }

    // Combine parsed modules that belong to the same module.
    // TODO: topological sort, group, and schedule.
    auto module = Sema::Translate(parsed_modules);
    if (action == Action::AST) {
        module->dump();
        return 0;
    }

    // Don’t try and codegen if there was an error.
    if (ctx.diags().has_error()) return 1;
    auto ir_module = CodeGen::Emit(*module);
    ir_module->dump();
    std::exit(42);
}

auto Driver::Impl::ParseFile(const File::Path& path) -> ParsedModule* {
    auto& f = ctx.get_file(path);
    return Parser::Parse(f).release();
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

int Driver::run_job(Action action) {
    std::unique_lock _{impl->mutex};
    return impl->run_job(action);
}

void Driver::enable_colours(bool enable) {
    impl->ctx.enable_colours(enable);
}
