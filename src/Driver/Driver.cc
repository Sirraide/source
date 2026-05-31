#include <srcc/CG/CodeGen.hh>
#include <srcc/CG/Target/Target.hh>
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
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Timer.h>

#include <filesystem>
#include <print>
#include <unordered_set>

namespace srcc {
class ParsedModule;
}
using namespace srcc;

auto Driver::PrepareJob() -> int {
    Assert(not compiled, "Can only call compile() once per Driver instance!");
    compiled = true;
    auto a = opts.action;
    ctx._initialise_context_(opts.module_output_path, opts.opt_level);

    // Always create a regular diags engine first for driver diags.
    driver_diags = StreamingDiagnosticsEngine::Create(ctx, opts.error_limit);
    ctx.set_diags(driver_diags);

    // Print pending diagnostics on exit.
    defer { ctx.diags().flush(); };

    // Load shared libraries.
    for (const auto& p : opts.eval_libs) {
        auto res = ctx.load_shared_library(p);
        if (not res) return Error("Could not load shared library '{}': {}", p.string(), res.error());
    }

    // Check if the target is supported.
    if (not opts.triple.isOSLinux() or opts.triple.getArch() != llvm::Triple::x86_64) {
        return Error(
            "Unsupported target triple '{}'. Only x86_64 Linux is supported at the moment.",
            opts.triple.getTriple()
        );
    }

    // Disable colours in verify mode.
    ctx.use_colours = opts.colours and not opts.verify;

    // Verifier can only run in sema/parse/lex mode.
    if (
        opts.verify and
        a != Action::CodeGen and
        a != Action::Parse and
        a != Action::Sema and
        a != Action::Lex and
        a != Action::DumpTokens and
        a != Action::DumpIR
    ) return Error("--verify requires one of: --cg, --lex, --parse, --sema, --ir, --tokens");

    // AST dump requires parse/sema mode.
    if (opts.print_ast and a != Action::Parse and a != Action::Sema)
        return Error("--ast requires --parse or --sema");

    // IR dumping flags are only valid if we’re dumping IR.
    if (a != Action::DumpIR) {
        if (opts.ir_generic) return Error("--ir-generic requires --ir");
        if (opts.ir_no_finalise) return Error("--ir-no-finalise requires --ir");
        if (opts.ir_verbose) return Error("--ir-verbose requires --ir");
    }

    // Forward options to context.
    ctx.eval_steps = opts.eval_steps;
    ctx.use_short_filenames = opts.short_filenames;
    ctx.target_triple = opts.triple;

    // Check for duplicate TUs as they would create horrible linker errors.
    // FIXME: Use inode instead?
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

    // Stop if there was a file we couldn’t find.
    if (ctx.diags().has_error()) return 1;
    return 0;
}

void Driver::add_file(fs::Path file_path) {
    files.push_back(std::move(file_path));
}

int Driver::dump_module(StringRef import_string) {
    Assert(opts.action == Action::DumpModule);
    if (auto res = PrepareJob(); res != 0)
        return res;

    static constexpr String ImportName = "__srcc_dummy_import__";
    auto import_str = std::format(
        "program __srcc_dummy__; import {} as {};",
        import_string,
        ImportName
    );

    auto mod = Parser::Parse(ctx.create_virtual_file(import_str), nullptr);
    if (not mod) return 1;

    SmallVector<ParsedModule::Ptr> modules;
    modules.push_back(std::move(mod));
    auto tu = Sema::Translate(
        opts.lang_opts,
        std::move(modules),
        opts.module_search_paths,
        opts.clang_include_paths,
        opts.clang_options
    );

    if (not tu or ctx.diags().has_error()) return 1;
    tu->logical_imports.at(ImportName)->dump(ctx.use_colours);
    return 0;
}

int Driver::run_job() {
    if (auto res = PrepareJob(); res != 0)
        return res;

    Assert(
        opts.action != Action::DumpModule,
        "Call dump_module() instead of run_job() to dump a module"
    );

    defer { driver_diags->flush(); };
    defer { ctx.diags().flush(); };

    // Timing setup.
    if (opts.time_trace_path.has_value()) llvm::timeTraceProfilerInitialize(0, "srcc");

    // Replace context diags with the actual diags engine.
    if (opts.verify) ctx.set_diags(VerifyDiagnosticsEngine::Create(ctx));

    // Run the verifier.
    const auto Verify = [&] {
        ctx.diags().flush();
        auto& engine = static_cast<VerifyDiagnosticsEngine&>(ctx.diags());
        return engine.verify() ? 0 : 1;
    };

    // Check if we failed.
    const auto CanContinue = [&] (StringRef phase) {
        if (not ctx.diags().has_error()) return true;

        // If we're in verify mode, report that we couldn't get to the step
        // we want to verify due to a prior error.
        if (opts.verify) Error(
            "Could not run '--verify' as compilation failed during {}",
            phase
        );

        return false;
    };

    // We only allow one file if we’re only lexing.
    auto a = opts.action;
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
                        u64(lc->col) + (tok.location.measure_token_length(ctx).value_or(1) ?: 1) - 1
                    );

                    std::println("{}", tok.location.text(ctx));
                }
            }
        }

        return ctx.diags().has_error();
    }

    // Parse files.
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
    if (not CanContinue("parsing")) return 1;

    // Combine parsed modules that belong to the same module.
    // TODO: topological sort, group, and schedule.
    // TODO: Maybe dispatch to multiple processes? It would simplify things if
    // a single compiler invocation only had to deal with 1 TU, and ~~loading
    // module descriptions multiple times might not be a bottleneck (but C++
    // headers probably would be...).~~ Irrelevant. We need to import them into
    // the module that uses them anyway.
    auto tu = Sema::Translate(
        opts.lang_opts,
        std::move(parsed_modules),
        opts.module_search_paths,
        opts.clang_include_paths,
        opts.clang_options
    );

    if (a == Action::Sema) {
        ctx.diags().flush();
        if (opts.verify) return Verify();
        if (opts.print_ast) tu->dump();
        return ctx.diags().has_error();
    }

    // Run the constant evaluator.
    if (a == Action::Eval or a == Action::EvalDumpIR) {
        // TODO: Static initialisation.
        if (not CanContinue("sema")) return 1;
        auto res = tu->vm.eval(nullptr, tu->file_scope_block, true, a == Action::EvalDumpIR, true);
        return res.has_value() ? 0 : 1;
    }

    // A module’s file name is kind of important, so don’t allow changing it.
    if (tu->is_module and not opts.output_file_name.empty()) return Error(
        "The '-o' option can only be used with programs, not modules"
    );

    // Create a machine for this target.
    auto machine = tu->target().create_machine(opts.opt_level);

    // Don’t try and codegen if there was an error.
    if (not CanContinue("sema")) return 1;

    // Run codegen.
    cg::CodeGen cg{*tu, tu->lang_opts()};
    cg.emit_as_needed(tu->procs);
    if (opts.action == Action::CodeGen) {
        if (opts.verify) return Verify();
        return ctx.diags().has_error();
    }

    // Don’t run the finaliser if codegen failed.
    if (not CanContinue("codegen")) return 1;

    // Run finalisation.
    bool finalise_ok = opts.ir_no_finalise or cg.finalise(not opts.ir_no_verify);
    if (opts.verify) return Verify();
    if (not CanContinue("IR finalisation")) return 1;

    // Dump exports.
    if (a == Action::DumpExports) {
        if (not tu->is_module) return Error("--exports cannot be used with a 'program'");
        for (auto e : tu->exports.sorted_decls()) e->dump(ctx.use_colours);
        return 0;
    }

    // Dump IR.
    if (a == Action::DumpIR) {
        auto s = cg.dump(opts.ir_verbose, opts.ir_generic);
        std::print("{}", text::RenderColours(opts.colours, s.str()));
        return finalise_ok ? 0 : 1;
    }

    // Give up if we’re actually trying to do anything with this
    // if there was an error.
    if (not finalise_ok) return 1;

    // Run LLVM lowering.
    auto ir_module = cg.emit_llvm(*machine);
    if (not CanContinue("LLVM lowering")) return 1;

    // Always run the optimiser before potentially dumping the module.
    //
    // We do this even at -O0 since there are some mandatory passes (e.g.
    // to process 'inline' functions).
    cg.optimise(*machine, *tu, *ir_module);

    // Emit LLVM IR.
    if (a == Action::EmitLLVM) {
        ir_module->print(llvm::outs(), nullptr);
        return 0;
    }

    // Emit the module.
    int res = cg.write_to_file(
        *machine,
        *tu,
        *ir_module,
        opts.lib_paths,
        opts.link_libs,
        opts.link_objects,
        opts.output_file_name
    );

    // Write the timing report if requested.
    if (opts.time_trace_path.has_value()) {
        SmallString<0> str;
        llvm::raw_svector_ostream os{str};
        llvm::timeTraceProfilerWrite(os);
        File::WriteOrDie(str.data(), str.size(), *opts.time_trace_path);
    }

    return res;
}

auto Driver::ParseFile(fs::PathRef path, bool verify) -> ParsedModule::Ptr {
    auto engine = verify ? static_cast<VerifyDiagnosticsEngine*>(&ctx.diags()) : nullptr;
    auto& f = ctx.get_file(path);
    return Parser::Parse(f, verify ? engine->comment_token_callback() : nullptr);
}
