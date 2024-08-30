module;

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <print>
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

    // Nested engine for reporting errors during the comment
    // parsing process.
    diags_reporter = StreamingDiagnosticsEngine::Create(ctx);
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
    stream comment{tok.text.sv()};
    comment.drop_while('/');
    comment.trim_front();

    // If the comment doesn’t start with 'expected-', ignore it.
    if (!comment.starts_with("expected-")) return;

    // Handle 'expected-no-diagnostics'.
    comment.drop("expected-"sv.size());
    if (comment.trim() == "no-diagnostics") {
        expects_none = true;
        return;
    }

    // Parse the diagnostic type. 'ICE' is not supported, but used here
    // as a failure state so we don’t have to bother with optionals.
    auto type = comment.take_while(llvm::isAlnum);
    auto level = llvm::StringSwitch<Diagnostic::Level>(type)
                     .Case("error", Diagnostic::Level::Error)
                     .Case("warning", Diagnostic::Level::Warning)
                     .Case("note", Diagnostic::Level::Note)
                     .Default(Diagnostic::Level::ICE);

    // Didn’t find a valid diagnostic type.
    if (level == Diagnostic::Level::ICE) return;

    // Parse line offsets.
    auto where = DecodeLocation(tok.location);
    if (comment.trim_front().starts_with('@')) {
        comment.drop().trim_front();

        // Offset may be relative.
        bool negative = comment.starts_with("-");
        bool relative = negative or comment.starts_with("+");
        if (relative) comment.drop().trim_front();

        // Parse the offset.
        auto offset = Parse<u64>(comment.take_while(llvm::isDigit));
        if (not offset.has_value()) return Error(
            tok.location,
            "Invalid line offset in expected diagnostic: '{}'\n",
            offset.error()
        );

        auto offs = offset.value();
        if (relative) {
            if (negative) {
                where.line -= offs;
            } else where.line += offs;
        } else {
            where.line = offs;
        }

        // Sanity check.
        if (where.line < 1 or where.line > usz(where.file->size())) return Error(
            tok.location,
            "Invalid computed line offset in expected diagnostic: '{}'\n",
            offset.error()
        );
    }

    // Next, the optional count.
    u32 count = 1;
    if (comment.trim_front().starts_with_any("123456789")) {
        auto res = Parse<u32>(comment.take_while(llvm::isDigit));
        if (not res.has_value()) return Error(
            tok.location,
            "Invalid count in expected diagnostic: '{}'\n",
            res.error()
        );
        count = res.value();
    }

    // Next, we expect a colon.
    if (not comment.trim_front().starts_with(':')) return;

    // The rest of the comment is the diagnostic text.
    comment.drop().trim_front();
    expected_diags.emplace_back(level, std::string{comment.text()}, where, count);
}

void VerifyDiagnosticsEngine::report_impl(Diagnostic&& diag) {
    // Remove line-wrap formatting codes.
    std::erase_if(diag.msg, [](char c) { return c == '\v' or c == '\r'; });
    rgs::replace(diag.msg, '\f', ' ');
    diag.msg = text::RenderColours(false, diag.msg);
    seen_diags.emplace_back(std::move(diag), DecodeLocation(diag.where));
}

bool VerifyDiagnosticsEngine::verify() {
    bool ok = true;

    // If we expected no diagnostics, complain if we were instructed to check for any.
    //
    // This diagnostics engine only supports being run in a terminal and is intended to
    // be used for testing; don’t bother with another diagnostics engine here; just print
    // to stderr.
    if (expects_none and not expected_diags.empty()) {
        ok = false;
        print(
            stderr,
            "%b(%1(Error:) Cannot specify both 'expected-no-diagnostics' and expected diagnostics.)\n"
        );
    }

    // Conversely, also complain if we saw nothing at all.
    if (not expects_none and expected_diags.empty()) {
        ok = false;
        print(
            stderr,
            "%b(%1(Error:) Expected at least one 'expected-' directive. Use "
            "'expected-no-diagnostics' if no diagnostics are expected.)\n"
        );
    }

    // Check that we have seen every diagnostic that we expect.
    for (auto& expected : expected_diags) {
        while (expected.count != 0) {
            auto it = rgs::find_if(seen_diags, [&](const SeenDiagnostic& sd) {
                if (sd.loc != expected.loc) return false;
                if (sd.diag.level != expected.level) return false;
                return sd.diag.msg.contains(expected.text);
            });

            if (it == seen_diags.end()) break;
            expected.count--;
            utils::erase_unordered(seen_diags, it);
        }
    }

    // Erase all diagnostics that were seen.
    llvm::erase_if(expected_diags, [](const ExpectedDiagnostic& ed) { return ed.count == 0; });

    // Complain about every diagnostic that remains.
    if (not expected_diags.empty()) {
        ok = false;
        print(stderr, "%b(Expected diagnostics that were not seen:)\n");
        for (const auto& expected : expected_diags) print(
            stderr,
            "  %b({}:{} %{}({}:)) {}\n",
            expected.loc.file->path(),
            expected.loc.line,
            Diagnostic::Colour(expected.level),
            Diagnostic::Name(expected.level),
            expected.text
        );
    }

    // And about every diagnostic that we didn’t expect.
    if (not seen_diags.empty()) {
        ok = false;
        print(stderr, "%b(Unexpected diagnostics:)\n");
        for (const auto& seen : seen_diags) print(
            stderr,
            "  %b({}:{} %{}({}:)) {}\n",
            seen.loc.file->path(),
            seen.loc.line,
            Diagnostic::Colour(seen.diag.level),
            Diagnostic::Name(seen.diag.level),
            seen.diag.msg
        );
    }

    // Verification succeeds if we have seen all expected diagnostics.
    return ok;
}
