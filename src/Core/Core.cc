module;

#include <filesystem>
#include <generator>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Unicode.h>
#include <mutex>
#include <random>
#include <ranges>
#include <srcc/Macros.hh>
#include <thread>

#ifdef __linux__
#    include <unistd.h>
#endif

module srcc;
import base.text;
using namespace srcc;
using namespace base;

// ============================================================================
//  Context
// ============================================================================
struct Context::Impl {
    llvm::LLVMContext llvm;

    /// Diagnostics engine.
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags_engine;

    /// Mutex used by API functions that may mutate the context.
    std::recursive_mutex context_mutex;

    /// Files loaded by the context.
    std::vector<std::unique_ptr<File>> files;
    std::unordered_map<File::Path, File*> files_by_path; // FIXME: use inode number instead.

    /// Mutex used for printing diagnostics.
    mutable std::recursive_mutex diags_mutex;

    /// Whether there was an error.
    mutable std::atomic<bool> errored = false;

    /// Whether to use coloured output.
    std::atomic<bool> enable_colours = true;
};

SRCC_DEFINE_HIDDEN_IMPL(Context);
Context::Context() : impl(new Impl) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    });
}

auto Context::diags() const -> DiagnosticsEngine& {
    Assert(impl->diags_engine, "Diagnostics engine not set!");
    return *impl->diags_engine;
}

void Context::enable_colours(bool enable) {
    impl->enable_colours.store(enable, std::memory_order_release);
}

auto Context::file(usz idx) const -> const File* {
    std::unique_lock _{impl->context_mutex};

    if (idx >= impl->files.size()) return nullptr;
    return impl->files[idx].get();
}

auto Context::get_file(const File::Path& path) -> const File& {
    std::unique_lock _{impl->context_mutex};

    auto can = canonical(path);
    if (auto it = impl->files_by_path.find(can); it != impl->files_by_path.end())
        return *it->second;

    static constexpr usz MaxFiles = std::numeric_limits<u16>::max();
    Assert(
        impl->files.size() < MaxFiles,
        "Sorry, that’s too many files for us! (max is {})",
        MaxFiles
    );

    auto mem = File::LoadFileData(can);
    auto f = new File(*this, can, path.string(), std::move(mem), u16(impl->files.size()));
    impl->files.emplace_back(f);
    impl->files_by_path[std::move(can)] = f;
    return *f;
}

void Context::set_diags(llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags) {
    if (impl->diags_engine) impl->diags_engine->flush();
    impl->diags_engine = std::move(diags);
}

bool Context::use_colours() const {
    return impl->enable_colours.load(std::memory_order_acquire);
}

// ============================================================================
//  File
// ============================================================================
auto File::TempPath(StringRef extension) -> Path {
    std::mt19937 rd(std::random_device{}());

    // Get the temporary directory.
    auto tmp_dir = std::filesystem::temp_directory_path();

    // Use the pid on Linux, and another random number on Windows.
#ifdef __linux__
    auto pid = std::to_string(u32(getpid()));
#else
    auto pid = std::to_string(rd());
#endif

    // Get the current time and tid.
    auto now = chr::system_clock::now().time_since_epoch().count();
    auto tid = std::to_string(u32(std::hash<std::thread::id>{}(std::this_thread::get_id())));

    // And some random letters too.
    // Do NOT use `char` for this because it’s signed on some systems (including mine),
    // which completely breaks the modulo operation below... Thanks a lot, C.
    std::array<u8, 8> rand{};
    rgs::generate(rand, [&] { return rd() % 26 + 'a'; });

    // Create a unique file name.
    auto tmp_name = std::format(
        "{}.{}.{}.{}",
        pid,
        tid,
        now,
        std::string_view{reinterpret_cast<char*>(rand.data()), rand.size()}
    );

    // Append it to the temporary directory.
    auto f = tmp_dir / tmp_name;
    if (not extension.empty()) {
        if (not extension.starts_with(".")) f += '.';
        f += extension;
    }
    return f;
}

auto File::Write(const void* data, usz size, const Path& file) -> std::expected<void, std::string> {
    auto err = llvm::writeToOutput(absolute(file).string(), [=](llvm::raw_ostream& os) {
        os.write(static_cast<const char*>(data), size);
        return llvm::Error::success();
    });

    std::string text;
    llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase& e) {
        text += std::format("Failed to write to file '{}': {}", file, e.message());
    });
    return std::unexpected(text);
}

void File::WriteOrDie(void* data, usz size, const Path& file) {
    if (not Write(data, size, file)) Fatal(
        "Failed to write to file '{}': {}",
        file,
        std::strerror(errno)
    );
}

File::File(
    Context& ctx,
    Path path,
    std::string name,
    std::unique_ptr<llvm::MemoryBuffer> contents,
    u16 id
) : ctx(ctx),
    file_path(std::move(path)),
    file_name(std::move(name)),
    contents(std::move(contents)),
    id(id) {}

auto File::LoadFileData(const Path& path) -> std::unique_ptr<llvm::MemoryBuffer> {
    auto buf = llvm::MemoryBuffer::getFile(
        path.string(),
        true,
        false
    );

    if (auto ec = buf.getError()) Fatal(
        "Could not load file '{}': {}",
        path,
        ec.message()
    );

    // Construct the file data.
    return std::move(*buf);
}

// ============================================================================
//  Location
// ============================================================================
bool Location::seekable(const Context& ctx) const {
    auto* f = ctx.file(file_id);
    if (not f) return false;
    return pos + len <= f->size() + 1 and is_valid();
}

/// Seek to a source location. The location must be valid.
auto Location::seek(const Context& ctx) const -> std::optional<LocInfo> {
    if (not seekable(ctx)) return std::nullopt;
    LocInfo info{};

    // Get the file that the location is in.
    const auto* f = ctx.file(file_id);

    // Seek back to the start of the line.
    const char* const data = f->data();
    info.line_start = data + pos;
    while (info.line_start > data and *info.line_start != '\n') info.line_start--;
    if (*info.line_start == '\n') info.line_start++;

    // Seek forward to the end of the line.
    const char* const end = data + f->size();
    info.line_end = data + pos;
    while (info.line_end < end and *info.line_end != '\n') info.line_end++;

    // Determine the line and column number.
    info.line = 1;
    info.col = 1;
    for (const char* d = data; d < data + pos; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 1;
        } else {
            info.col++;
        }
    }

    // Done!
    return info;
}

/// TODO: Lexer should create map that counts where in a file the lines start so
/// we can do binary search on that instead of iterating over the entire file.
auto Location::seek_line_column(const Context& ctx) const -> std::optional<LocInfoShort> {
    if (not seekable(ctx)) return std::nullopt;
    LocInfoShort info{};

    // Get the file that the location is in.
    const auto* f = ctx.file(file_id);

    // Seek back to the start of the line.
    const char* const data = f->data();

    // Determine the line and column number.
    info.line = 1;
    info.col = 1;
    for (const char* d = data; d < data + pos; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 1;
        } else {
            info.col++;
        }
    }

    // Done!
    return info;
}

auto Location::text(const Context& ctx) const -> String {
    if (not seekable(ctx)) return "";
    auto* f = ctx.file(file_id);
    return String::CreateUnsafe(StringRef{f->data(), usz(f->size())}.substr(pos, len));
}

// ============================================================================
//  Diagnostics
// ============================================================================
StreamingDiagnosticsEngine::~StreamingDiagnosticsEngine() {
    Assert(backlog.empty(), "Diagnostics not flushed?");
}

auto EscapeParens(StringRef str) -> std::string {
    std::string s{str};
    utils::ReplaceAll(s, ")", "\033)");
    return s;
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
            Diagnostic::Colour(diag.level),
            Diagnostic::Name(diag.level),
            diag.msg
        );

        // Even if the location is invalid, print the file name if we can.
        if (auto f = ctx.file(diag.where.file_id))
            out += std::format("\n  in %b({}:<invalid location>)\n\n", EscapeParens(f->name()));

        PrintExtraData();
        return out;
    }

    // If the location is valid, get the line, line number, and column number.
    const auto [line, col, line_start, line_end] = *l;
    auto col_offs = col - 1;

    // Split the line into everything before the range, the range itself,
    // and everything after.
    std::string before(line_start, col_offs);
    std::string range(line_start + col_offs, std::min<u64>(diag.where.len, u64(line_end - (line_start + col_offs))));
    auto after = line_start + col_offs + diag.where.len > line_end
                   ? std::string{}
                   : std::string(line_start + col_offs + diag.where.len, line_end);

    // Replace tabs with spaces. We need to do this *after* splitting
    // because this invalidates the offsets. Also escape any parentheses
    // in the source text.
    before = EscapeParens(before);
    range = EscapeParens(range);
    after = EscapeParens(after);
    utils::ReplaceAll(before, "\t", "    ");
    utils::ReplaceAll(range, "\t", "    ");
    utils::ReplaceAll(after, "\t", "    ");

    // TODO: Explore this idea:
    //
    //   Error at foo.src:1:1
    //    1 | code goes here
    //                  ~~~~ Error message goes here
    //

    // Print the diagnostic name and message.
    out += std::format(
        "%b(%{}({}:) {})\n",
        Diagnostic::Colour(diag.level),
        Diagnostic::Name(diag.level),
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
                "at %b4({}):{}:{}\n",
                EscapeParens(file.name()),
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
    out += std::format("%b({} |) {}", line, before);
    out += std::format("%b8({})", range);
    out += std::format("{}\n", after);

    // Determine the number of digits in the line number.
    const auto digits = std::to_string(line).size();

    // Underline the range. For that, we first pad the line based on the number
    // of digits in the line number and append more spaces to line us up with
    // the range.
    for (usz i = 0, end = digits + TextWidth(text::ToUTF32(before)) + sizeof(" | ") - 1; i < end; i++)
        out += " ";

    // Finally, underline the range.
    out += std::format(
        "%{}b({})",
        Diagnostic::Colour(diag.level),
        std::string(TextWidth(text::ToUTF32(range)), '~')
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
                        auto ws = s.take_while_or_empty(' ');
                        hang = ws.size();
                        EmitLeading(false);
                        buffer += text::ToUTF8(ws);
                    }

                    // Emit the rest of the line.
                    std::string hang_indent(hang, ' ');
                    auto EmitRestOfLine = [&](std::u32string_view rest_of_line, auto parts) {
                        buffer += text::ToUTF8(rest_of_line);
                        buffer += "\n";

                        // Emit the remaining parts.
                        auto total = rgs::distance(parts);
                        for (auto [j, part] : parts | vws::enumerate) {
                            EmitLeading(j == total - 1, part.empty());
                            if (not part.empty()) {
                                buffer += hang_indent;
                                buffer += text::ToUTF8(std::u32string_view{part.data(), part.size()});
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

void StreamingDiagnosticsEngine::EmitDiagnostics() {
    using enum utils::Colour;
    utils::Colours C{ctx.use_colours()};
    if (backlog.empty()) return;
    if (printed) stream << "\n";
    printed++;
    stream << RenderDiagnostics(ctx, backlog, cols());
    stream << C(Reset);
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

            stream << text::RenderColours(ctx.use_colours(), std::format(
                "\n%b(%{}(Error:) Too many errors emitted (> {}\033). Not showing any more errors.)\n",
                Diagnostic::Colour(Diagnostic::Level::Error),
                printed - 1
            ));

            stream << text::RenderColours(ctx.use_colours(), std::format(
                "%b(%{}(Note:) Use '--error-limit <limit>' to show more errors.)\n",
                Diagnostic::Colour(Diagnostic::Level::Note)
            ));
        }

        return;
    }

    // If this not a note, emit the backlog.
    if (diag.level != Diagnostic::Level::Note) EmitDiagnostics();
    backlog.push_back(std::move(diag));
}
