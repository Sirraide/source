#include <srcc/Core/Core.hh>
#include <srcc/Core/Location.hh>

#include <mlir/IR/Location.h>

using namespace srcc;

/// Get the line+column position of a character in a source file; if the
/// location points to a '\n' character, treat it as part of the previous
/// line.
///
/// TODO: Lexer should create map that counts where in a file the lines start so
/// we can do binary search on that instead of iterating over the entire file.
static void SeekLineColumn(LocInfoShort& info, const srcc::File& f, const char* ptr) {
    info.line = 1;
    info.col = 1;
    info.file = &f;
    for (const char *d = f.data(); d < ptr; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 1;
        } else {
            info.col++;
        }
    }
}

auto SLoc::Decode(mlir::Location loc) -> SLoc {
    if (
        auto o = dyn_cast<mlir::OpaqueLoc>(loc);
        o and o.getUnderlyingTypeID() == mlir::TypeID::get<SLoc>()
    ) {
        return Decode(o.getUnderlyingLocation());
    }

    return SLoc();
}

auto SLoc::after(const Context& ctx) const -> SLoc {
    if (file(ctx) == SLoc(ptr + 1).file(ctx)) return SLoc(ptr + 1);
    return *this;
}

auto SLoc::file(const Context& ctx) const -> const File* {
    for (auto f : ctx.files())
        if (std::less_equal{}(f->begin(), ptr) and std::less{}(ptr, f->end()))
            return f;
    return nullptr;
}

auto SLoc::format(const Context& ctx, bool include_file_name) const -> std::string {
    auto f = seek_line_column(ctx);
    if (not f) return "<invalid>";
    if (include_file_name) return std::format("{}:{}:{}", f->file->name(), f->line, f->col);
    return std::format("<{}:{}>", f->line, f->col);
}

auto SLoc::text(const Context& ctx) const -> String {
    auto len = measure_token_length(ctx);
    if (not len) return "";
    return String::CreateUnsafe(ptr, *len);
}

auto SLoc::seek_line_column(const Context& ctx) const -> std::optional<LocInfoShort> {
    auto f = file(ctx);
    if (not f) return std::nullopt;
    LocInfoShort info{};
    SeekLineColumn(info, *f, ptr);
    return info;
}

auto SLoc::seek(const Context& ctx) const -> std::optional<LocInfo> {
    auto f = file(ctx);
    if (not f) return std::nullopt;
    LocInfo info{};
    SeekLineColumn(info, *f, ptr);
    auto len = measure_token_length(ctx);

    // Get everything before the range.
    str s{f->data(), usz(ptr - f->data())};
    info.before = String::CreateUnsafe(s.take_back_until_any("\r\n").text());

    // If the position is directly on a '\n', then we treat this as being
    // at the very end of the previous line, so stop here.
    if (ptr[*len] == '\n' or ptr[*len] == '\r') return info;

    // Next, get everything in and after the range.
    s = str{f->contents().value()};
    s.drop(usz(ptr - f->data()));
    info.range = String::CreateUnsafe(s.take(*len).text());
    info.after = String::CreateUnsafe(s.take_until('\n').text());

    // Done!
    return info;
}

auto SRange::text(const Context& ctx) const -> String {
    auto fb = begin.file(ctx);
    auto fe = end.file(ctx);
    auto len = end.measure_token_length(ctx);
    if (not fb or fb != fe or not len) return "";
    return String::CreateUnsafe(begin.ptr, usz(end.ptr - begin.ptr) + *len);
}
