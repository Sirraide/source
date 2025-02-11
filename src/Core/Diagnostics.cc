#include <srcc/Core/Core.hh>
#include <srcc/Core/Diagnostics.hh>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Process.h>

#include <base/Stream.hh>

using namespace srcc;

// ============================================================================
//  Diagnostic Formatting
// ============================================================================
/// Get the colour of a diagnostic.
static auto Colour(Diagnostic::Level kind) -> char {
    using Level = Diagnostic::Level;
    switch (kind) {
        case Level::ICE: return '5';
        case Level::Warning: return '3';
        case Level::Note: return '2';
        case Level::Error: return '1';
    }
    Unreachable();
}

/// Get the name of a diagnostic.
static auto Name(Diagnostic::Level kind) -> std::string_view {
    using Level = Diagnostic::Level;
    switch (kind) {
        case Level::ICE: return "Internal Compiler Error";
        case Level::Error: return "Error";
        case Level::Warning: return "Warning";
        case Level::Note: return "Note";
    }
    Unreachable();
}

auto TakeColumns(u32stream& s, usz n) -> std::pair<std::u32string, usz> {
    static constexpr usz TabSize = 4;
    static constexpr std::u32string_view Tab = U"    ";

    std::u32string buf;
    usz sz = 0;

    while (not s.empty() and sz < n) {
        // Handle ANSI escape sequences, tabs, and non-printable chars.
        if (s.starts_with('\033')) buf += s.take_until('m');
        else if (s.starts_with(U'\t')) {
            sz += TabSize;
            buf += Tab;
        } else if (*s.front() > char32_t(31)) {
            sz += c32(*s.front()).width();
        }
        buf += s.take();
    }

    return {std::move(buf), sz};
}

auto TextWidth(std::u32string_view data) -> usz {
    u32stream text{data};
    auto [_, sz] = TakeColumns(text, std::numeric_limits<usz>::max());
    return sz;
}

auto FormatDiagnostic(
    const Context& ctx,
    const Diagnostic& diag,
    Opt<Location> previous_loc
) -> std::string {
    std::string out;

    // Print any extra data that should come after the source line.
    auto PrintExtraData = [&] {
        if (diag.extra.empty()) return;

        // Adding a newline at the end here forces an extra empty line to
        // be emitted when we print the diagnostic; that looks better if
        // we have extra data, but it’s weird if we don’t, which is why
        // we add the linebreak after the main diagnostic text here.
        //
        // Make sure the extra data is at the start of a line so that \r
        // is handled properly.
        // TODO: Just turn \r into \n mid-line.
        out += std::format("\n{}\n", diag.extra);
    };

    // If the location is invalid, either because the specified file does not
    // exists, its position is out of bounds or 0, or its length is 0, then we
    // skip printing the location.
    auto l = diag.where.seek(ctx);
    if (not l.has_value()) {
        // Print the message.
        out += std::format(
            "%b(%{}({}:) {})",
            Colour(diag.level),
            Name(diag.level),
            diag.msg
        );

        // Even if the location is invalid, print the file name if we can.
        if (auto f = ctx.file(diag.where.file_id))
            out += std::format("\n  in %b(\002{}\003:<invalid location>)\n\n", f->name());

        PrintExtraData();
        return out;
    }

    // If the location is valid, get the line, line number, and column number.
    const auto [line, col, line_start, line_end] = *l;
    auto col_offs = col - 1;

    // Split the line into everything before the range, the range itself,
    // and everything after.
    std::string before(line_start, usz(col_offs));
    std::string range(line_start + col_offs, std::min<u64>(diag.where.len, u64(line_end - (line_start + col_offs))));
    auto after = line_start + col_offs + diag.where.len > line_end
                   ? std::string{}
                   : std::string(line_start + col_offs + diag.where.len, line_end);

    // Replace tabs with spaces. We need to do this *after* splitting
    // because this invalidates the offsets. Also escape any parentheses
    // in the source text.
    utils::ReplaceAll(before, "\t", "    ");
    utils::ReplaceAll(range, "\t", "    ");
    utils::ReplaceAll(after, "\t", "    ");
    auto before_wd = TextWidth(text::ToUTF32(before));
    auto range_wd = TextWidth(text::ToUTF32(range));

    // TODO: Explore this idea:
    //
    //   Error at foo.src:1:1
    //    1 | code goes here
    //                  ~~~~ Error message goes here
    //

    // Print the diagnostic name and message.
    out += std::format(
        "%b(%{}({}:) {})\n",
        Colour(diag.level),
        Name(diag.level),
        diag.msg
    );

    // Print the location if it is in the same file as the previous
    // diagnostic or if there are extra locations.
    if (
        not diag.extra_locations.empty() or
        not previous_loc.has_value() or
        previous_loc.value().file_id != diag.where.file_id
    ) {
        auto PrintLocation = [&](Location loc, LocInfoShort l) {
            const auto& file = *ctx.file(loc.file_id);
            out += std::format(
                "at %b4(\002{}\003):{}:{}\n",
                file.name(),
                l.line,
                l.col
            );
        };

        // Print main location.
        out += "  ";
        PrintLocation(diag.where, l->short_info());

        // Print extra locations.
        for (const auto& [note, extra] : diag.extra_locations) {
            auto extra_lc = extra.seek_line_column(ctx);
            if (not extra_lc) continue;

            out += "  ";
            if (not note.empty()) {
                out += note;
                out += ' ';
            }

            PrintLocation(extra, *extra_lc);
        }

        out += "\n";
    }

    // Print the line up to the start of the location, the range in the right
    // colour, and the rest of the line.
    // TODO: Proper underlines: \033[1;58:5:1;4:3m
    out += std::format("%b({} |) \002{}\003", line, before);
    out += std::format("%b8(\002{}\003)", range);
    out += std::format("\002{}\003\n", after);

    // Determine the number of digits in the line number.
    const auto digits = std::to_string(line).size();

    // Underline the range. For that, we first pad the line based on the number
    // of digits in the line number and append more spaces to line us up with
    // the range.
    for (usz i = 0, end = digits + before_wd + " | "sv.size(); i < end; i++)
        out += " ";

    // Finally, underline the range.
    out += std::format(
        "%{}b({})",
        Colour(diag.level),
        std::string(range_wd, '~')
    );

    // And print any extra data.
    PrintExtraData();
    return out;
}

auto RenderDiagnostics(
    const Context& ctx,
    ArrayRef<Diagnostic> backlog,
    usz cols
) -> std::string {
    bool use_colours = ctx.use_colours();

    // Subtract 2 here for the leading character and the space after it.
    auto cols_rem = cols - 2;

    // Print an extra line after a line that was broken into multiple lines if
    // another line would immediately follow; this keeps track of that.
    bool prev_was_multiple = false;

    // Print all diagnostics that we have queued up as a group.
    bool first_line = true;
    std::string buffer;
    Opt<Location> previous_loc;
    for (auto [di, diag] : enumerate(backlog)) {
        // Render the diagnostic text.
        auto out = text::RenderColours(use_colours, FormatDiagnostic(ctx, diag, previous_loc));

        // Then, indent everything properly.
        auto lines = stream{out}.split("\n");
        auto count = rgs::distance(lines);
        for (auto [i, line] : lines | vws::enumerate) {
            auto EmitLeading = [&](bool last_line_segment, bool segment_empty = false) {
                StringRef leading;
                if (
                    last_line_segment and
                    di == backlog.size() - 1 and
                    i == count - 1
                ) {
                    // Edge case: print nothing if this is the only line.
                    if (first_line) return;
                    leading = "╰"sv;
                } else {
                    leading = first_line ? "╭"sv : "│"sv;
                    first_line = false;
                }

                buffer += leading;
                if (not segment_empty) buffer += " ";
            };

            // A '\r' at the start of the line prints an empty line only if
            // the previous line was not empty.
            bool add_line = line.consume('\r');
            if (add_line and not buffer.ends_with("│\n")) {
                EmitLeading(false, true);
                buffer += "\n";
            }

            // Print an extra empty line after a line that was broken into pieces.
            if (prev_was_multiple and not line.empty() and not add_line) {
                EmitLeading(false, true);
                buffer += "\n";
            }

            // Always clear this flag.
            prev_was_multiple = false;

            // The text might need some post processing: within a line,
            // a vertical tab can be used to set a tab stop to which the
            // line is indented if broken into multiple pieces; form feeds
            // can additionally be used to specify where a line break can
            // happen, in which case no other line breaks are allowed.
            if (line.contains_any("\v\f")) {
                // Convert to utf32 so we can iterate chars more easily.
                auto utf32 = text::ToUTF32(line.text());

                // Check if everything fits in a single line. size() may be
                // an overestimate because it contains ANSI escape code etc.
                // but if that already fits, then we don’t need to check
                // anything else.
                if (utf32.size() < cols_rem or TextWidth(utf32) < cols_rem) {
                    SmallString<80> s;
                    EmitLeading(true);
                    for (auto c : line.text()) {
                        if (c == '\f') s.push_back(' ');
                        else if (c != '\v') s.push_back(c);
                    }
                    buffer += stream{s.str()}.trim().text();
                    buffer += "\n";
                }

                // Otherwise, we need to break the line.
                else {
                    u32stream s{utf32};

                    // Determine hanging indentation. This is the position of
                    // the first '\v', or all leading whitespace if there is
                    // none.
                    usz hang;
                    if (s.contains(U'\v')) {
                        auto start = s.take_until(U'\v');
                        hang = TextWidth(start);

                        // Emit everything up to the tab stop.
                        s.drop();
                        EmitLeading(false);
                        buffer += text::ToUTF8(start);
                    } else {
                        // We don’t use tabs in diagnostics, so just check for spaces.
                        auto ws = s.take_while(' ');
                        hang = ws.size();
                        EmitLeading(false);
                        buffer += text::ToUTF8(ws);
                    }

                    // Emit the rest of the line.
                    std::string hang_indent(hang, ' ');
                    auto EmitRestOfLine = [&](u32stream rest_of_line, auto parts) {
                        buffer += text::ToUTF8(rest_of_line.trim().text());
                        buffer += "\n";

                        // Emit the remaining parts.
                        auto total = rgs::distance(parts);
                        for (auto [j, part] : parts | vws::enumerate) {
                            EmitLeading(j == total - 1, part.empty());
                            if (not part.empty()) {
                                buffer += hang_indent;
                                buffer += text::ToUTF8(u32stream{std::u32string_view{part.data(), part.size()}}.trim().text());
                            }
                            buffer += "\n";
                        }

                        prev_was_multiple = true;
                    };

                    // Next, if we have form feeds, split the text along them
                    // and indent every part based on the hanging indentation.
                    if (s.contains(U'\f')) {
                        auto first = s.take_until(U'\f');
                        EmitRestOfLine(first, s.drop().split(U"\f"));
                    }

                    // Otherwise, just wrap the test at screen width.
                    else {
                        auto chunk_size = cols_rem - hang;
                        auto first = TakeColumns(s, chunk_size).first;
                        SmallVector<std::u32string, 8> chunks;
                        while (not s.empty()) {
                            auto [chunk, _] = TakeColumns(s, chunk_size);
                            chunks.push_back(std::move(chunk));
                        }

                        EmitRestOfLine(first, chunks);
                    }
                }
            }

            // If neither is present, emit the line literally.
            else {
                EmitLeading(true, line.empty());
                buffer += line.text();
                buffer += "\n";
            }
        }

        previous_loc = diag.where;
    }

    return buffer;
}

// ============================================================================
//  Diagnostic
// ============================================================================
Diagnostic::Diagnostic(Level lvl, Location where, std::string msg, std::string extra)
    : level(lvl),
      where(where),
      msg(std::move(msg)),
      extra(std::move(extra)) {}

// ============================================================================
//  Streaming Diagnostics Engine
// ============================================================================
void DiagnosticsEngine::report(Diagnostic&& diag) {
    if (diag.level == Diagnostic::Level::Error or diag.level == Diagnostic::Level::ICE)
        error_flag.store(true, std::memory_order_relaxed);
    report_impl(std::move(diag));
}

StreamingDiagnosticsEngine::StreamingDiagnosticsEngine(
    const Context& ctx,
    u32 error_limit,
    llvm::raw_ostream& output_stream
) : DiagnosticsEngine(ctx), stream(output_stream), error_limit(error_limit) {}

StreamingDiagnosticsEngine::~StreamingDiagnosticsEngine() {
    Assert(backlog.empty(), "Diagnostics not flushed?");
}

auto StreamingDiagnosticsEngine::Create(
    const Context& ctx,
    u32 error_limit,
    llvm::raw_ostream& output_stream
) -> Ptr {
    return llvm::IntrusiveRefCntPtr(new StreamingDiagnosticsEngine(ctx, error_limit, output_stream));
}

void StreamingDiagnosticsEngine::EmitDiagnostics() {
    if (backlog.empty()) return;
    if (printed) stream << "\n";
    printed++;
    stream << RenderDiagnostics(ctx, backlog, cols());
    backlog.clear();
}

void StreamingDiagnosticsEngine::add_extra_location_impl(Location extra, std::string note) {
    if (backlog.empty()) return;
    backlog.back().extra_locations.emplace_back(std::move(note), extra);
}

void StreamingDiagnosticsEngine::add_remark(std::string msg) {
    if (backlog.empty()) return;
    backlog.back().extra += std::move(msg);
}

u32 StreamingDiagnosticsEngine::cols() {
    return std::max<u32>({llvm::sys::Process::StandardErrColumns(), llvm::sys::Process::StandardOutColumns(), 80});
}

void StreamingDiagnosticsEngine::flush() {
    EmitDiagnostics();
    stream.flush();
}

void StreamingDiagnosticsEngine::report_impl(Diagnostic&& diag) {
    // Give up if we’ve printed too many errors.
    if (error_limit and printed >= error_limit) {
        if (printed == error_limit) {
            printed++;
            EmitDiagnostics();

            stream << text::RenderColours(
                ctx.use_colours(),
                std::format(
                    "\n%b(%{}(Error:) Too many errors emitted (> {}\033). Not showing any more errors.)\n",
                    Colour(Diagnostic::Level::Error),
                    printed - 1
                )
            );

            stream << text::RenderColours(
                ctx.use_colours(),
                std::format(
                    "%b(%{}(Note:) Use '--error-limit <limit>' to show more errors.)\n",
                    Colour(Diagnostic::Level::Note)
                )
            );
        }

        return;
    }

    // If this not a note, emit the backlog.
    if (diag.level != Diagnostic::Level::Note) EmitDiagnostics();
    backlog.push_back(std::move(diag));
}

// ============================================================================
//  Verify Diagnostics Engine
// ============================================================================
VerifyDiagnosticsEngine::VerifyDiagnosticsEngine(const Context& ctx) : DiagnosticsEngine(ctx) {
    // Colours are disabled in verify mode, so save whether
    // we *actually* want to use colours (in the output of
    // the verifier).
    enable_colours = ctx.use_colours();

    // Nested engine for reporting errors.
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

    // Parse line offsets. Note that '@*' indicates that any
    // location is allowed.
    Opt<DecodedLocation> diag_loc = DecodeLocation(tok.location);
    if (comment.trim_front().consume('@')) {
        if (comment.trim_front().consume('*')) diag_loc = {};
        else {
            // Offset may be relative.
            bool negative = comment.starts_with("-");
            bool relative = negative or comment.starts_with("+");
            if (relative) comment.drop().trim_front();

            // Parse the offset.
            auto offset = Parse<i64>(comment.take_while(llvm::isDigit));
            if (not offset.has_value()) return Error(
                tok.location,
                "Invalid line offset in expected diagnostic: '{}'\n",
                offset.error()
            );

            auto where = diag_loc.value();
            auto offs = offset.value();
            if (relative) {
                if (negative) {
                    where.line -= offs;
                } else where.line += offs;
            } else {
                where.line = offs;
            }

            // Sanity check.
            if (where.line < 1 or where.line > where.file->size()) return Error(
                tok.location,
                "Invalid computed line offset in expected diagnostic: '{}'\n",
                offset.error()
            );

            diag_loc = where;
        }
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
    expected_diags.emplace_back(level, std::string{comment.text()}, diag_loc, count);
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

    // Sort expected diagnostics such that any diags without a location
    // are placed at the end of the vector.
    rgs::partition(expected_diags, [](const ExpectedDiagnostic& ed) { return ed.loc.has_value(); });

    // Check that we have seen every diagnostic that we expect.
    for (auto& expected : expected_diags) {
        while (expected.count != 0) {
            auto it = rgs::find_if(seen_diags, [&](const SeenDiagnostic& sd) {
                if (expected.loc and sd.loc != expected.loc) return false;
                if (sd.diag.level != expected.level) return false;
                return sd.diag.msg.contains(expected.text);
            });

            if (it == seen_diags.end()) break;
            expected.count--;
            seen_diags.erase(it);
        }
    }

    // Erase all diagnostics that were seen.
    llvm::erase_if(expected_diags, [](const ExpectedDiagnostic& ed) { return ed.count == 0; });

    // Complain about every diagnostic that remains.
    if (not expected_diags.empty()) {
        ok = false;
        print(stderr, "%b(Expected diagnostics that were not seen:)\n");
        for (const auto& expected : expected_diags) {
            if (expected.loc) {
                print(
                    stderr,
                    "  %b({}:{})",
                    expected.loc.value().file->path(),
                    expected.loc.value().line
                );
            } else {
                print(stderr, "  %b(anywhere)");
            }

            print(
                stderr,
                " %b(%{}({}:)) {}\n",
                Colour(expected.level),
                Name(expected.level),
                expected.text
            );
        }
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
            Colour(seen.diag.level),
            Name(seen.diag.level),
            seen.diag.msg
        );
    }

    // Verification succeeds if we have seen all expected diagnostics.
    return ok;
}
