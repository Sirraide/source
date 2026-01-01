#include <srcc/Core/Utils.hh>
#include <srcc/Driver/Driver.hh>

#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Signals.h>
#include <llvm/TargetParser/Host.h>

#include <base/Clopts.hh>

#ifdef __linux__
#    include <csignal>
#endif

using namespace srcc;
using namespace base::cmd;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,

    // General options.
    option<"--colour", "Enable or disable coloured output (default: auto)", values<"auto", "always", "never">>,
    option<"--error-limit", "Limit how many errors are printed; passing 0 removes the limit", std::int64_t>,
    option<"--mo", "Path to a directory where compiled modules will be placed (default: '.')">,
    option<"-o", "Override the default output file name">,
    option<"--eval-steps", "Maximum number of evaluation steps before compile-time evaluation results in an error", std::int64_t>,
    option<"--target", "Target triple to compile for">,
    multiple<option<"--link-object", "Link a compiled object file into every TU that is part of this compilation">>,
    multiple<short_option<"-M", "Path to a directory that should be searched for compiled modules">>,
    multiple<short_option<"-I", "Add a directory to the C/C++ header search path">>,
    multiple<short_option<"-L", "Add a directory to the library search path">>,
    multiple<short_option<"-l", "Link against a library">>,
    short_option<"-O", "Optimisation level", values<0, 1, 2, 3, 4>>,

    // Flags that determine what action to take.
    flag<"--cg", "Run codegen but do not emit anything. See also --ir, --llvm.">,
    flag<"--eval", "Run the entire input through the constant evaluator">,
    flag<"--eval-dump-ir", "As --eval, but also dump the IR used for evaluation">,
    flag<"--exports", "Dump the module description to stdout">,
    flag<"--dump-module", "Dump the contents of a module or C++ header that we can import">,
    flag<"--ir", "Run codegen and emit IR. See also --cg, --llvm.">,
    flag<"--lex", "Lex tokens only (but do not print them) and exit">,
    flag<"--llvm", "Run codegen and emit LLVM IR. See also --cg, --ir.">,
    flag<"--parse", "Parse only and exit">,
    flag<"--sema", "Run sema only and exit">,
    flag<"--tokens", "Print tokens and exit">,
    mutually_exclusive<
        "--cg", "--eval", "--eval-dump-ir", "--exports", "--dump-module",
        "--ir", "--lex", "--llvm", "--parse", "--sema", "--tokens"
    >,

    // General flags.
    flag<"--ast", "Dump the parse tree / AST">,
    flag<"--verify", "Run in verify-diagnostics mode">,
    flag<"--ir-generic", "Use the generic MLIR assembly format">,
    flag<"--ir-no-finalise", "Don’t finalise the IR">,
    flag<"--ir-no-verify", "Don’t verify the IR">,
    flag<"--ir-verbose", "Always print the type of a value and other details">,
    flag<"--noruntime", "Do not automatically import the runtime module">,
    flag<"--short-filenames", "Use the filename only instead of the full path in diagnostics">,

    // Features.
    // TODO: Consider: short_option<"-f, "Enable or disable a feature", values<"overflow-checks">> or
    // something in that vein.
    flag<"-fno-overflow-checks", "Disable overflow checking">,
    flag<"-fstringify-asserts", "Stringify assert conditions if possible">,
    flag<"-fgc-procs", "Strip procedures that are never called">,

    // Internal flags.
    flag<"-Xpreamble", "Enable or disable the preamble", {.hidden = true, .default_value = true}>,

    help<>
>; // clang-format on

// Disable leak detection for now because MLIR seems to ‘leak’ a
// small amount of memory (probably some persistent data structures
// that are allocated once and never freed by design).
extern "C" [[gnu::used]] const char* __asan_default_options() { return "detect_leaks=0"; }

// Global colour flag used in signal handlers etc.
static std::atomic_bool colours_enabled;

#ifdef __linux__
static void InitSignalHandlers() {
    std::signal(SIGSEGV, [](int) {
        auto msg = colours_enabled.load(std::memory_order_relaxed)
            ? "\033[1;35mInternal Compiler Error: \033[m\033[1mSegmentation fault\033[m"sv
            : "Internal Compiler Error: Segmentation fault";

        llvm::errs() << msg << "\n";
        llvm::sys::PrintStackTrace(llvm::errs());
        std::_Exit(1);
    });

    std::signal(SIGABRT, [](int) {
        auto msg = colours_enabled.load(std::memory_order_relaxed)
            ? "\033[1;35mInternal Compiler Error: \033[m\033[1mAborted\033[m"sv
            : "Internal Compiler Error: Aborted";

        llvm::errs() << msg << "\n";
        llvm::sys::PrintStackTrace(llvm::errs());
        std::_Exit(1);
    });
}
#endif

static auto ParseArgs(int argc, char** argv) -> options::optvals_type {
    Context ctx;
    auto diags = StreamingDiagnosticsEngine::Create(ctx);
    auto opts = options::parse(argc, argv, [&](std::string msg) {
        diags->report(Diagnostic(Diagnostic::Level::Error, SLoc(), msg));
        return false;
    });

    if (diags->has_error()) {
        diags->flush();
        std::exit(1);
    }

    return opts;
}

int main(int argc, char** argv) {
    llvm::sys::DisableSystemDialogsOnCrash();
    libassert::enable_virtual_terminal_processing_if_needed();
    auto opts = ParseArgs(argc, argv);

    // Enable colours.
    auto colour_opt = opts.get<"--colour">("auto");
    bool use_colour =
          colour_opt == "never"  ? false
        : colour_opt == "always" ? true
        :                          libassert::isatty(STDERR_FILENO) and libassert::isatty(STDOUT_FILENO);

    if (use_colour) colours_enabled.store(true, std::memory_order_relaxed);

#ifdef __linux__
    InitSignalHandlers();
#endif

    // Set assert handler.
    libassert::set_failure_handler([](const libassert::assertion_info& info){
        auto ice = colours_enabled.load(std::memory_order_relaxed)
            ? "\033[1;35mInternal Compiler Error\033[m"sv
            : "Internal Compiler Error";

        // Print this first since formatting the stack trace can take a while.
        std::print(stderr, "{}: (generating stacktrace...)", ice);

        // Get the trace.
        auto message = info.to_string(
            libassert::terminal_width(STDERR_FILENO),
            colours_enabled.load(std::memory_order_relaxed)
                ? libassert::get_color_scheme()
                : libassert::color_scheme::blank
        );

        std::print(stderr, "\r{}: {}", ice, message);
        std::_Exit(1);
    });

    // Disable this because they take too long and it’s fucking annoying.
    llvm::sys::Process::PreventCoreFiles();

    // Figure out what we want to do.
    auto action = opts.get<"--cg">()           ? Action::CodeGen
                : opts.get<"--eval-dump-ir">() ? Action::EvalDumpIR
                : opts.get<"--eval">()         ? Action::Eval
                : opts.get<"--exports">()      ? Action::DumpExports
                : opts.get<"--dump-module">()  ? Action::DumpModule
                : opts.get<"--ir">()           ? Action::DumpIR
                : opts.get<"--lex">()          ? Action::Lex
                : opts.get<"--llvm">()         ? Action::EmitLLVM
                : opts.get<"--parse">()        ? Action::Parse
                : opts.get<"--sema">()         ? Action::Sema
                : opts.get<"--tokens">()       ? Action::DumpTokens
                                               : Action::Compile;

    // Collect module search paths.
    std::vector<std::string> module_search_paths{
        opts.get<"-M">().begin(),
        opts.get<"-M">().end(),
    };

    module_search_paths.push_back(SOURCE_PROJECT_DIR_NAME "/modules");

    // Determine the target triple.
    llvm::Triple triple;
    if (auto tgt = opts.get<"--target">()) {
        triple = llvm::Triple(llvm::Triple::normalize(*tgt));
    } else {
        triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    }

    // GC procedures by default unless --ir or --llvm is passed.
    bool gc_procs = action != Action::DumpIR and action != Action::EmitLLVM;
    if (opts.get<"-fgc-procs">()) gc_procs = true;

    // TODO:
    //  - Move lang opts to be TU-specific in case the TU wants
    //    to alter them using e.g. pragmas.
    //
    //  - Move eval steps into lang opts.
    //
    //  - Add a pragma to control eval steps.
    //
    //  - eval steps = 0 to disable the check entirely. (Note:
    //    ‘disabling’ constant evaluation is impossible since
    //    it’s required by the language in some places even if
    //    `eval` is not used.)

    // Create driver.
    Driver driver{{// clang-format off
        .triple = std::move(triple),
        .module_output_path = opts.get<"--mo">("."),
        .output_file_name = opts.get<"-o">(""),
        .module_search_paths = module_search_paths,
        .clang_include_paths = opts.get<"-I">(),
        .lib_paths = opts.get<"-L">(),
        .link_libs = opts.get<"-l">(),
        .link_objects = opts.get<"--link-object">(),
        .action = action,
        .lang_opts = {
            .overflow_checking = not opts.get<"-fno-overflow-checks">(),
            .no_runtime = opts.get<"--noruntime">(),
            .no_preamble = not opts.get<"-Xpreamble">(),
            .stringify_asserts = opts.get<"-fstringify-asserts">(),
            .gc_procs = gc_procs,
        },
        .eval_steps = u64(opts.get<"--eval-steps">(1 << 20)),
        .error_limit = u32(opts.get<"--error-limit">(20)),
        .opt_level = u8(opts.get<"-O">(0)),
        .print_ast = opts.get<"--ast">(),
        .verify = opts.get<"--verify">(),
        .colours = use_colour,
        .short_filenames = opts.get<"--short-filenames">(),
        .ir_generic = opts.get<"--ir-generic">(),
        .ir_no_finalise = opts.get<"--ir-no-finalise">(),
        .ir_no_verify = opts.get<"--ir-no-verify">(),
        .ir_verbose = opts.get<"--ir-verbose">(),
    }}; // clang-format on

    // Add files.
    driver.add_file(*opts.get<"file">());

    // Dew it.
    return driver.run_job();
}
