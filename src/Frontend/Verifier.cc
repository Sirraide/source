module;

#include <cctype>
#include <fmt/format.h>
#include <fmt/std.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/STLExtras.h>
#include <srcc/Macros.hh>
#include <string_view>

module srcc.frontend.verifier;
import srcc.utils;
using namespace srcc;
using namespace std::literals;

VerifyDiagnosticsEngine::VerifyDiagnosticsEngine(const Context& ctx) : DiagnosticsEngine(ctx) {
    // Colours are disabled in verify mode, so save whether
    // we *actually* want to use colours (in the output of
    // the verifier).
    enable_colours = ctx.use_colours();
}

/// Decode location so we don’t have to do that every time we check a
/// diagnostic. We can’t do anything if the source location is not valid.
auto VerifyDiagnosticsEngine::DecodeLocation(Location loc) -> DecodedLocation {
    auto file = ctx.file(loc.file_id);
    auto decoded = loc.seek_line_column(ctx);
    Assert(file and decoded.has_value(), "Verify-diagnostics-mode requires valid source locations.");
    return {file, decoded->line};
}

/// This is called by the lexer when a comment token is encountered.
void VerifyDiagnosticsEngine::HandleCommentToken(const Token& tok) {
    // Skip slashes and initial whitespace.
    auto comment = StringRef(tok.text);
    comment = comment.drop_while([](char c) { return c == '/'; });
    comment = comment.drop_while([](char c) { return std::isspace(c); });

    // If the comment doesn’t start with 'expected-', ignore it.
    if (!comment.starts_with("expected-")) return;

    // Handle 'expected-no-diagnostics'.
    comment = comment.drop_front("expected-"sv.size());
    if (comment.trim() == "no-diagnostics") {
        expects_none = true;
        return;
    }

    // Parse the diagnostic type. 'ICE' is not supported, but used here
    // as a failure state so we don’t have to bother with optionals.
    auto type = comment.take_while([](char c) { return std::isalnum(c); });
    auto level = llvm::StringSwitch<Diagnostic::Level>(type)
                     .Case("error", Diagnostic::Level::Error)
                     .Case("warning", Diagnostic::Level::Warning)
                     .Case("note", Diagnostic::Level::Note)
                     .Default(Diagnostic::Level::ICE);

    // Didn’t find a valid diagnostic type.
    if (level == Diagnostic::Level::ICE) return;

    // Next, we expect a colon.
    comment = comment.drop_front(type.size());
    if (not comment.starts_with(':')) return;

    // The rest of the comment is the diagnostic text.
    comment = comment.drop_front(1).trim();
    expected_diags.emplace_back(level, std::string{comment}, DecodeLocation(tok.location));
}


void VerifyDiagnosticsEngine::report_impl(Diagnostic&& diag) {
    seen_diags.emplace_back(std::move(diag), DecodeLocation(diag.where));
}

bool VerifyDiagnosticsEngine::verify() {
    using enum utils::Colour;
    utils::Colours C{enable_colours};
    bool ok = true;

    // If we expected no diagnostics, complain if we were instructed to check for any.
    //
    // This diagnostics engine only supports being run in a terminal and is intended to
    // be used for testing; don’t bother with another diagnostics engine here; just print
    // to stderr.
    if (expects_none and not expected_diags.empty()) {
        ok = false;
        fmt::print(
            stderr,
            "{}{}Error: {}{}Cannot specify both 'expected-no-diagnostics' and expected diagnostics.{}\n",
            C(Bold),
            C(Red),
            C(Reset),
            C(Bold),
            C(Reset)
        );
    }

    // Check that we have seen every diagnostic that we expect.
    for (auto& expected : expected_diags) {
        auto it = rgs::find_if(seen_diags, [&](const SeenDiagnostic& sd) {
            if (sd.loc != expected.loc) return false;
            if (sd.diag.level != expected.level) return false;
            return sd.diag.msg.contains(expected.text);
        });

        if (it != seen_diags.end()) {
            expected.seen = true;
            utils::erase_unordered(seen_diags, it);
        }
    }

    // Erase all diagnostics that were seen.
    llvm::erase_if(expected_diags, [](const ExpectedDiagnostic& ed) { return ed.seen; });

    // Complain about every diagnostic that remains.
    if (not expected_diags.empty()) {
        ok = false;
        fmt::print(stderr, "{}Expected diagnostics that were not seen:{}\n", C(Bold), C(Reset));
        for (const auto& expected : expected_diags) fmt::print(
            stderr,
            "  {}{}:{} {}{}: {}{}\n",
            C(Bold),
            expected.loc.file->path(),
            expected.loc.line,
            Diagnostic::Colour(C, expected.level),
            Diagnostic::Name(expected.level),
            C(Reset),
            expected.text
        );
    }

    // And about every diagnostic that we didn’t expect.
    if (not seen_diags.empty()) {
        ok = false;
        fmt::print(stderr, "{}Unexpected diagnostics:{}\n", C(Bold), C(Reset));
        for (const auto& seen : seen_diags) fmt::print(
            stderr,
            "  {}{}:{} {}{}: {}{}\n",
            C(Bold),
            seen.loc.file->path(),
            seen.loc.line,
            Diagnostic::Colour(C, seen.diag.level),
            Diagnostic::Name(seen.diag.level),
            C(Reset),
            seen.diag.msg
        );
    }

    // Verification succeeds if we have seen all expected diagnostics.
    return ok;
}
