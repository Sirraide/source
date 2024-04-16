module;

#include <filesystem>
#include <fmt/std.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Unicode.h>
#include <mutex>
#include <random>
#include <srcc/Macros.hh>
#include <thread>

module srcc;

using namespace srcc;

// ============================================================================
//  Context
// ============================================================================
struct Context::Impl {
    llvm::LLVMContext llvm;

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
Context::Context() : impl(new Impl) {}

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
    if (impl->files.size() >= MaxFiles) Diag::ICE(
        "Sorry, that’s too many files for us! (max is {})",
        MaxFiles
    );

    auto mem = File::LoadFileData(can);
    auto f = new File(*this, can, path.string(), std::move(mem), u16(impl->files.size()));
    impl->files.emplace_back(f);
    impl->files_by_path[std::move(can)] = f;
    return *f;
}

auto Context::has_error() const -> bool {
    return impl->errored.load(std::memory_order_acquire);
}

auto Context::lock_diags_mutex() const -> std::unique_lock<std::recursive_mutex> {
    return std::unique_lock{impl->diags_mutex};
}

void Context::set_error() const {
    impl->errored.store(true, std::memory_order_release);
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
#ifndef _WIN32
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
    auto tmp_name = fmt::format(
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

auto File::Write(const void* data, usz size, const Path& file) -> Result<> {
    auto err = llvm::writeToOutput(absolute(file).string(), [=](llvm::raw_ostream& os) {
        os.write(static_cast<const char*>(data), size);
        return llvm::Error::success();
    });

    return Result<>::Error(std::move(err), [&](const llvm::ErrorInfoBase& e) {
        return fmt::format("Failed to write to file '{}': {}", file, e.message());
    });
}

void File::WriteOrDie(void* data, usz size, const Path& file) {
    if (not Write(data, size, file))
        Diag::Fatal("Failed to write to file '{}': {}", file, std::strerror(errno));
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

    if (auto ec = buf.getError()) Diag::Fatal(
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
    return pos + len <= f->size() and is_valid();
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

auto Location::text(const Context& ctx) const -> StringRef {
    if (not seekable(ctx)) return "";
    auto* f = ctx.file(file_id);
    return StringRef{f->data(), usz(f->size())}.substr(pos, len);
}

// ============================================================================
//  Diagnostics
// ============================================================================
std::atomic EnableAssertColours{bool(isatty(fileno(stderr)))};
std::atomic<llvm::raw_ostream*> DiagsStream{&llvm::errs()};

/// Get the colour of a diagnostic.
constexpr auto Colour(utils::Colours C, Diag::Kind kind) -> std::string_view {
    using Kind = Diag::Kind;
    using enum utils::Colour;
    switch (kind) {
        case Kind::ICE: return C(Magenta);
        case Kind::Warning: return C(Yellow);
        case Kind::Note: return C(Green);

        case Kind::Fatal:
        case Kind::Error:
            return C(Red);

        default:
            return C(None);
    }
}

/// Get the name of a diagnostic.
constexpr std::string_view Name(Diag::Kind kind) {
    using Kind = Diag::Kind;
    switch (kind) {
        case Kind::ICE: return "Internal Compiler Error";
        case Kind::Fatal: return "Fatal Error";
        case Kind::Error: return "Error";
        case Kind::Warning: return "Warning";
        case Kind::Note: return "Note";
        default: return "Diagnostic";
    }
}

/// Remove project directory from filename.
constexpr auto NormaliseFilename(std::string_view filename) -> std::string_view {
    if (auto pos = filename.find(SOURCE_PROJECT_DIR_NAME); pos != std::string_view::npos) {
        static constexpr std::string_view name{SOURCE_PROJECT_DIR_NAME};
        filename.remove_prefix(pos + name.size() + 1);
    }
    return filename;
}

void Diag::EnableColours(bool enable) {
    EnableAssertColours.store(enable);
}

void Diag::SetDiagsStream(llvm::raw_ostream& stream) {
    DiagsStream.store(&stream);
}

void Diag::HandleFatalErrors() {
    // Abort on ICE.
    if (kind == Kind::ICE)
        LIBASSERT_PANIC("Internal Compiler Error"); // Separate line for breakpoint.

    // Exit on a fatal error. Never print a backtrace here as fatal
    // errors are due to the underlying system misbehaving, so a stack
    // trace won’t help because it’s not our fault.
    if (kind == Kind::Fatal)
        std::exit(FatalExitCode); // Separate line for breakpoint.
}

// Print a diagnostic with no (valid) location info.
void Diag::PrintDiagWithoutLocation(utils::Colours C) {
    using enum utils::Colour;
    auto& stream = *DiagsStream.load();

    // Print error location, if present.
    if (sloc.line() != 0 and kind != Kind::Fatal) {
        stream << fmt::format(
            "{}{}:{}:{}: ",
            C(Bold),
            NormaliseFilename(sloc.file_name()),
            sloc.line(),
            sloc.column()

        );
    }

    // Print the message.
    stream << fmt::format("{}{}{}: {}", C(Bold), Colour(C, kind), Name(kind), C(Reset));
    stream << fmt::format("{}{}{}\n", C(Bold), msg, C(Reset));
    HandleFatalErrors();
}

void Diag::print() {
    utils::Colours C(ctx ? ctx->use_colours() : EnableAssertColours.load(std::memory_order_relaxed));
    using enum utils::Colour;
    auto& stream = *DiagsStream.load();

    // If this diagnostic is suppressed, do nothing.
    if (kind == Kind::None) return;
    defer {
        // If the diagnostic is an error, set the error flag.
        if (kind == Kind::Error and ctx)
            ctx->set_error(); /// Separate line so we can put a breakpoint here.

        // Don’t print the same diagnostic twice.
        kind = Kind::None;

        // Reset the colour when we’re done.
        stream << fmt::format("{}", C(Reset));
    };

    // If there is no context, then there is also no location info.
    if (not ctx) {
        PrintDiagWithoutLocation(C);
        return;
    }

    /// Make sure we don’t interleave diagnostics.
    auto _ = ctx ? ctx->lock_diags_mutex() : std::unique_lock<std::recursive_mutex>{};

    /// Make sure that diagnostics don’t clump together, but also don’t insert
    /// an ugly empty line before the first diagnostic.
    if (ctx->has_error() and kind != Kind::Note) stream << "\n";

    /// If the location is invalid, either because the specified file does not
    /// exists, its position is out of bounds or 0, or its length is 0, then we
    /// skip printing the location.
    auto l = where.seek(*ctx);
    if (not l.has_value()) {
        /// Even if the location is invalid, print the file name if we can.
        if (auto f = ctx->file(where.file_id))
            stream << fmt::format("{}{}: ", C(Bold), f->path());

        /// Print the message.
        PrintDiagWithoutLocation(C);
        return;
    }

    /// If the location is valid, get the line, line number, and column number.
    const auto [line, col, line_start, line_end] = *l;
    auto col_offs = col - 1;

    /// Split the line into everything before the range, the range itself,
    /// and everything after.
    std::string before(line_start, col_offs);
    std::string range(line_start + col_offs, std::min<u64>(where.len, u64(line_end - (line_start + col_offs))));
    auto after = line_start + col_offs + where.len > line_end
                   ? std::string{}
                   : std::string(line_start + col_offs + where.len, line_end);

    /// Replace tabs with spaces. We need to do this *after* splitting
    /// because this invalidates the offsets.
    utils::ReplaceAll(before, "\t", "    ");
    utils::ReplaceAll(range, "\t", "    ");
    utils::ReplaceAll(after, "\t", "    ");

    /// Print the file name, line number, and column number.
    const auto& file = *ctx->file(where.file_id);
    stream << fmt::format("{}{}:{}:{}: ", C(Bold), file.name(), line, col);

    /// Print the diagnostic name and message.
    stream << fmt::format("{}{}: ", Colour(C, kind), Name(kind));
    stream << fmt::format("{}{}\n", C(Reset), msg);

    /// Print the line up to the start of the location, the range in the right
    /// colour, and the rest of the line.
    stream << fmt::format(" {} | {}", line, before);
    stream << fmt::format("{}{}{}{}", C(Bold), Colour(C, kind), range, C(Reset));
    stream << fmt::format("{}\n", after);

    /// Determine the number of digits in the line number.
    const auto digits = utils::NumberWidth(line);

    /// LLVM’s columnWidthUTF8() function returns -1 for non-printable characters
    /// for some ungodly reason, so guard against that.
    static const auto ColumnWidth = [](StringRef text) {
        auto wd = llvm::sys::unicode::columnWidthUTF8(text);
        return wd < 0 ? 0 : usz(wd);
    };

    /// Underline the range. For that, we first pad the line based on the number
    /// of digits in the line number and append more spaces to line us up with
    /// the range.
    for (usz i = 0, end = digits + ColumnWidth(before) + sizeof("  | ") - 1; i < end; i++)
        stream << " ";

    /// Finally, underline the range.
    stream << fmt::format("{}{}", C(Bold), Colour(C, kind));
    for (usz i = 0, end = ColumnWidth(range); i < end; i++) stream << "~";
    stream << "\n";

    /// Handle fatal errors.
    HandleFatalErrors();
}
