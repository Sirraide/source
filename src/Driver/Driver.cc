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
#include <print>
#include <future>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ThreadPool.h>
#include <mutex>
#include <ranges>
#include <srcc/Macros.hh>
#include <string>
#include <unordered_set>

module srcc.driver;
import srcc;
import srcc.frontend.verifier;
import srcc.frontend.parser;
import srcc.frontend.sema;
import srcc.codegen;

using namespace srcc;

// ============================================================================
//  Internals
// ============================================================================
class DriverThreadPool {
    std::optional<llvm::StdThreadPool> thread_pool;

public:
    DriverThreadPool(u32 num_threads) {
        if (num_threads != 1) thread_pool.emplace(llvm::ThreadPoolStrategy(num_threads));
    }

    /// Run a task asynchronously.
    template <typename Task>
    auto run(Task task) -> std::shared_future<decltype(task())>;

    /// Wait for all tasks to finish.
    void wait();
};

struct Driver::Impl : DiagsProducer<> {
    Options opts;
    DriverThreadPool thread_pool;
    SmallVector<File::Path> files;
    Context ctx;
    std::mutex mutex;
    bool compiled = false;

    Impl(Options opts)
        : opts(opts),
          thread_pool(opts.num_threads) {}

    int run_job();

    template <typename... Args>
    void Diag(Diagnostic::Level level, Location loc, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(level, loc, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    int Error(std::format_string<Args...> fmt, Args&&... args) {
        Diag(Diagnostic::Level::Error, Location(), fmt, std::forward<Args>(args)...);
        return 1;
    }

    /// Parse a file and return the parsed module.
    auto ParseFile(const File::Path& path, bool lex_only, bool verify) -> ParsedModule*;
};

int Driver::Impl::run_job() {
    Assert(not compiled, "Can only call compile() once per Driver instance!");
    compiled = true;

    // Always create a regular diags engine first for driver diags.
    ctx.set_diags(StreamingDiagnosticsEngine::Create(ctx));

    // Disable colours in verify mode.
    ctx.enable_colours(opts.colours and not opts.verify);

    // Verifier can only run in sema/parse/lex mode.
    if (
        opts.verify and
        opts.action != Action::Parse and
        opts.action != Action::Sema and
        opts.action != Action::Lex
    ) return Error("--verify requires one of: --lex, --parse, --sema");

    // Duplicate TUs would create horrible linker errors.
    // FIXME: Use inode instead?
    std::unordered_set<File::Path> file_uniquer;
    for (const auto& f : files) {
        if (not fs::exists(f)) {
            Error("File '{}' does not exist", f);
            continue;
        }

        auto can = canonical(f);
        if (not file_uniquer.insert(can).second) Fatal(
            "Duplicate file name in command-line: '{}'",
            can
        );
    }

    // Stop if there was a file we couldn’t find.
    if (ctx.diags().has_error()) return 1;

    // Replace driver diags with the actual diags engine.
    if (opts.verify) ctx.set_diags(VerifyDiagnosticsEngine::Create(ctx));

    // Parse files in parallel.
    SmallVector<std::shared_future<ParsedModule*>> futures;
    for (const auto& f : files) futures.push_back(thread_pool.run([&] {
        return ParseFile(f, opts.action == Action::Lex, opts.verify);
    }));

    // Wait for all modules to be parsed.
    thread_pool.wait();

    // Collect modules.
    SmallVector<std::unique_ptr<ParsedModule>> parsed_modules;
    for (auto& f : futures) {
        auto ptr = f.get();
        parsed_modules.emplace_back(ptr);
    }

    // Run the verifier.
    const auto Verify = [&] {
        auto& engine = static_cast<VerifyDiagnosticsEngine&>(ctx.diags());
        return engine.verify() ? 0 : 1;
    };

    // Dump modules if we should only do parsing.
    if (opts.action == Action::Parse or opts.action == Action::Lex) {
        if (opts.verify) return Verify();
        if (opts.print_ast)
            for (auto& m : parsed_modules)
                m->dump();
        return ctx.diags().has_error();
    }

    if (ctx.diags().has_error()) {
        if (opts.verify) return Verify();
        return 1;
    }

    // Combine parsed modules that belong to the same module.
    // TODO: topological sort, group, and schedule.
    auto module = Sema::Translate(parsed_modules);
    if (opts.action == Action::Sema) {
        if (opts.verify) return Verify();
        if (opts.print_ast) module->dump();
        return ctx.diags().has_error();
    }

    // Don’t try and codegen if there was an error.
    Assert(not opts.verify, "Cannot verify codegen");
    if (ctx.diags().has_error()) return 1;
    auto ir_module = CodeGen::Emit(*module);
    ir_module->dump();
    std::exit(42);
}

auto Driver::Impl::ParseFile(const File::Path& path, bool lex_only, bool verify) -> ParsedModule* {
    auto engine = verify ? static_cast<VerifyDiagnosticsEngine*>(&ctx.diags()) : nullptr;
    auto& f = ctx.get_file(path);
    return Parser::Parse(f, lex_only, verify ? engine->comment_token_callback() : nullptr).release();
}

template <typename Task>
auto DriverThreadPool::run(Task task) -> std::shared_future<decltype(task())> {
    // If we’re using more than one thread, run this normally.
    if (thread_pool.has_value()) return thread_pool->async(std::move(task));

    // Otherwise, run it on the main thread.
    using RetVal = decltype(task());
    std::promise<RetVal> promise;
    promise.set_value(task());
    return promise.get_future();
}

void DriverThreadPool::wait() {
    // No-op if we’re in single-threaded mode.
    if (thread_pool.has_value()) thread_pool->wait();
}

// ============================================================================
//  API
// ============================================================================
SRCC_DEFINE_HIDDEN_IMPL(Driver);
Driver::Driver(Options opts) : impl(new Impl{opts}) {}

void Driver::add_file(std::string_view file_path) {
    std::unique_lock _{impl->mutex};
    impl->files.push_back(File::Path(file_path));
}

int Driver::run_job() {
    std::unique_lock _{impl->mutex};
    return impl->run_job();
}
