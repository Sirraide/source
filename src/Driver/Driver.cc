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
#include <srcc/CG/CodeGen.hh>
#include <srcc/Core/Core.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Driver/Driver.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/ThreadPool.h>

#include <algorithm>
#include <filesystem>
#include <future>
#include <mutex>
#include <print>
#include <ranges>
#include <unordered_set>

namespace srcc {
class ParsedModule;
}
using namespace srcc;

void Driver::add_file(std::string_view file_path) {
    files.push_back(fs::Path(file_path));
}

int Driver::run_job() {
    Assert(not compiled, "Can only call compile() once per Driver instance!");
    compiled = true;
    auto a = opts.action;
    ctx._initialise_context_(opts.module_output_path, opts.opt_level);

    // Always create a regular diags engine first for driver diags.
    ctx.set_diags(StreamingDiagnosticsEngine::Create(ctx, opts.error_limit));

    /// Print pending diagnostics on exit.
    defer { ctx.diags().flush(); };

    // Disable colours in verify mode.
    ctx.use_colours = opts.colours and not opts.verify;

    // Verifier can only run in sema/parse/lex mode.
    if (
        opts.verify and
        a != Action::Parse and
        a != Action::Sema and
        a != Action::Lex and
        a != Action::DumpTokens
    ) return Error("--verify requires one of: --lex, --parse, --sema, --tokens");

    // AST dump requires parse/sema mode.
    if (opts.print_ast and a != Action::Parse and a != Action::Sema)
        return Error("--ast requires --parse or --sema");

    // Create lang opts.
    LangOpts lang_opts;
    lang_opts.overflow_checking = opts.overflow_checking;

    // Forward options to context.
    ctx.eval_steps = opts.eval_steps;
    ctx.use_short_filenames = opts.short_filenames;

    // Handle this first; it only supports 1 file.
    if (a == Action::DumpModule) {
        if (files.size() != 1) return Error("'%3(--dump-module%)' requires exactly one file");
    }

    // Otherwise, check for duplicate TUs as they would create
    // horrible linker errors.
    // FIXME: Use inode instead?
    else {
        std::unordered_set<fs::Path> file_uniquer;
        for (const auto& f : files) {
            if (not fs::File::Exists(f)) {
                Error("File '{}' does not exist", f);
                continue;
            }

            auto can = canonical(f);
            if (not file_uniquer.insert(can).second) Fatal(
                "Duplicate file name in command-line: '{}'",
                can
            );
        }
    }

    // Stop if there was a file we couldn’t find.
    if (ctx.diags().has_error()) return 1;

    // Replace driver diags with the actual diags engine.
    if (opts.verify) ctx.set_diags(VerifyDiagnosticsEngine::Create(ctx));

    // Dump a module.
    if (a == Action::DumpModule) {
        llvm::BumpPtrAllocator alloc;
        llvm::StringSaver saver{alloc};
        ModuleLoader loader{ctx, opts.module_search_paths};
        auto mod = loader.load(
            "module",
            String::Save(saver, files.front().string()),
            Location(),
            files.front().string().starts_with("<")
        );

        if (not mod) return 1;
        if (auto tu = mod.value().dyn_cast<TranslationUnit*>()) tu->dump();
        else Todo();
        return 0;
    }

    // Run the verifier.
    const auto Verify = [&] {
        auto& engine = static_cast<VerifyDiagnosticsEngine&>(ctx.diags());
        return engine.verify() ? 0 : 1;
    };

    // We only allow one file if we’re only lexing.
    if (a == Action::Lex or a == Action::DumpTokens) {
        if (files.size() != 1) return Error("Lexing supports one file");
        auto engine = opts.verify ? static_cast<VerifyDiagnosticsEngine*>(&ctx.diags()) : nullptr;
        auto [alloc, stream] = Parser::ReadTokens(
            ctx.get_file(*files.begin()),
            opts.verify ? engine->comment_token_callback() : nullptr
        );

        if (opts.verify) return Verify();
        if (a == Action::DumpTokens) {
            for (auto tok : stream) {
                auto lc = tok.location.seek_line_column(ctx);
                if (not lc) std::print("<invalid srcloc>\n");
                else {
                    std::print(
                        "{}:{}-{}: ",
                        lc->line,
                        lc->col,
                        lc->col + tok.location.len - 1
                    );

                    std::println("{}", tok.spelling(ctx));
                }
            }
        }

        return ctx.diags().has_error();
    }

    // Parse files in parallel.
    SmallVector<ParsedModule::Ptr> parsed_modules;
    for (const auto& f : files) parsed_modules.push_back(ParseFile(f, opts.verify));

    // Dump parse tree.
    if (a == Action::Parse) {
        ctx.diags().flush();
        if (opts.verify) return Verify();
        if (opts.print_ast)
            for (auto& m : parsed_modules)
                m->dump();
        return ctx.diags().has_error();
    }

    // Stop if there was an error.
    if (ctx.diags().has_error()) return 1;

    // Load the runtime.
    ModuleLoader loader{ctx, opts.module_search_paths};
    StringMap<ImportHandle> imported;
    if (opts.import_runtime) {
        auto rt = loader.load(
            "__src_runtime",
            "__src_runtime",
            parsed_modules.front()->program_or_module_loc,
            false
        );

        if (not rt) return 1;
        imported.try_emplace("__src_runtime", std::move(rt.value()));
    }

    // And all imported modules we need.
    for (auto& m : parsed_modules) {
        for (auto& i : m->imports) {
            auto h = loader.load(i.import_name, i.linkage_name, i.loc, i.linkage_name.starts_with('<'));
            if (not h) return 1;
            imported.try_emplace(i.import_name.sv(), std::move(h.value()));
        }
    }

    // Drop the loader’s refcounts.
    loader.release_all();

    // Combine parsed modules that belong to the same module.
    // TODO: topological sort, group, and schedule.
    // TODO: Maybe dispatch to multiple processes? It would simplify things if
    // a single compiler invocation only had to deal with 1 TU, and ~~loading
    // module descriptions multiple times might not be a bottleneck (but C++
    // headers probably would be...).~~ Irrelevant. We need to import them into
    // the module that uses them anyway.
    auto tu = Sema::Translate(lang_opts, parsed_modules, std::move(imported));
    if (a == Action::Sema) {
        ctx.diags().flush();
        if (opts.verify) return Verify();
        if (opts.print_ast) tu->dump();
        return ctx.diags().has_error();
    }

    // Run the constant evaluator.
    if (a == Action::Eval) {
        // TODO: Static initialisation.
        if (ctx.diags().has_error()) return 1;
        auto res = tu->vm.eval(tu->file_scope_block);
        return res.has_value() ? 0 : 1;
    }

    // A module’s file name is kind of important, so don’t allow changing it.
    if (tu->is_module and not opts.output_file_name.empty()) return Error(
        "The '-o' option can only be used with programs, not modules"
    );

    // Create a machine for this target.
    auto machine = ctx.create_target_machine();

    // Don’t try and codegen if there was an error.
    Assert(not opts.verify, "Cannot verify codegen");
    if (ctx.diags().has_error()) return 1;

    // Run codegen.
    cg::CodeGen cg{*tu, tu->lang_opts(), Size::Bits(64)};
    for (auto p : tu->procs) cg.emit(p);
    if (ctx.diags().has_error()) return 1;

    // Dump IR.
    if (a == Action::DumpIR) {
        std::print("{}", text::RenderColours(opts.colours, cg.dump().str()));
        return 0;
    }

    // Run LLVM lowering.
    auto ir_module = cg.emit_llvm(*machine);
    if (ctx.diags().has_error()) return 1;

    // Run the optimiser before potentially dumping the module.
    if (opts.opt_level) cg.optimise(*machine, *tu, *ir_module);

    // Emit LLVM IR.
    if (a == Action::EmitLLVM) {
        ir_module->print(llvm::outs(), nullptr);
        return 0;
    }

    // Finally, emit the module.
    return cg.write_to_file(
        *machine,
        *tu,
        *ir_module,
        opts.link_objects,
        opts.output_file_name
    );
}

auto Driver::ParseFile(fs::PathRef path, bool verify) -> ParsedModule::Ptr {
    auto engine = verify ? static_cast<VerifyDiagnosticsEngine*>(&ctx.diags()) : nullptr;
    auto& f = ctx.get_file(path);
    return Parser::Parse(f, verify ? engine->comment_token_callback() : nullptr);
}
