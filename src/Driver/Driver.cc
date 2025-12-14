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

#include <filesystem>
#include <print>
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

    // Run the verifier.
    const auto Verify = [&] {
        ctx.diags().flush();
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
                        u64(lc->col) + (tok.location.measure_token_length(ctx).value_or(1) ?: 1) - 1
                    );

                    std::println("{}", tok.location.text(ctx));
                }
            }
        }

        return ctx.diags().has_error();
    }

    // Dump a module.
    if (a == Action::DumpModule) {
        static constexpr String ImportName = "__srcc_dummy_import__";
        auto import_str = std::format(
            "program __srcc_dummy__; import {} as {};",
            files.front().string(),
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
            opts.clang_include_paths
        );

        if (not tu or ctx.diags().has_error()) return 1;
        tu->logical_imports.at(ImportName)->dump(ctx.use_colours);
        return 0;
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
    if (ctx.diags().has_error()) return 1;

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
        opts.clang_include_paths
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
        if (ctx.diags().has_error()) return 1;
        auto res = tu->vm.eval(tu->file_scope_block, true, a == Action::EvalDumpIR);
        return res.has_value() ? 0 : 1;
    }

    // A module’s file name is kind of important, so don’t allow changing it.
    if (tu->is_module and not opts.output_file_name.empty()) return Error(
        "The '-o' option can only be used with programs, not modules"
    );

    // Create a machine for this target.
    auto machine = tu->target().create_machine(opts.opt_level);

    // Don’t try and codegen if there was an error.
    if (ctx.diags().has_error()) {
        if (opts.verify) ICE(SLoc(), "Could not run verifier on codegen due to Sema error");
        return 1;
    }

    // Run codegen.
    cg::CodeGen cg{*tu, tu->lang_opts()};
    cg.emit_as_needed(tu->procs);
    if (opts.action == Action::CodeGen) {
        if (opts.verify) return Verify();
        return ctx.diags().has_error();
    }

    // Don’t run the finaliser if codegen failed.
    if (ctx.diags().has_error()) {
        if (opts.verify) {
            ctx.set_diags(StreamingDiagnosticsEngine::Create(ctx, opts.error_limit));
            ICE(SLoc(), "Could not run --verify on finalised IR due to codegen error");
            Note(SLoc(), "Pass --cg if you want to test unfinalised IR");
        }
        return 1;
    }

    // Run finalisation.
    bool finalise_ok = opts.ir_no_finalise or cg.finalise();
    if (opts.verify) return Verify();
    if (ctx.diags().has_error()) return 1;

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
        opts.lib_paths,
        opts.link_libs,
        opts.link_objects,
        opts.output_file_name
    );
}

auto Driver::ParseFile(fs::PathRef path, bool verify) -> ParsedModule::Ptr {
    auto engine = verify ? static_cast<VerifyDiagnosticsEngine*>(&ctx.diags()) : nullptr;
    auto& f = ctx.get_file(path);
    return Parser::Parse(f, verify ? engine->comment_token_callback() : nullptr);
}
