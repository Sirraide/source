#include <srcc/Core/Core.hh>
#include <srcc/Core/Location.hh>

using namespace srcc;

/// Get the line+column position of a character in a source file; if the
/// location points to a '\n' character, treat it as part of the previous
/// line.
static void SeekLineColumn(LocInfoShort& info, const char* data, u32 pos) {
    info.line = 1;
    info.col = 1;
    for (const char *d = data, *end = data + pos; d < end; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 1;
        } else {
            info.col++;
        }
    }
}

Location::Location(Location a, Location b) {
    if (a.file_id != b.file_id) return;
    if (not a.is_valid() or not b.is_valid()) return;
    pos = std::min<u32>(a.pos, b.pos);
    len = u16(std::max<u32>(a.pos + a.len, b.pos + b.len) - pos);
    file_id = a.file_id;
}

auto Location::after() const -> Location {
    Location l = {pos + len, 1, file_id};
    return l.is_valid() ? l : *this;
}

[[nodiscard]] auto Location::contract_left(isz amount) const -> Location {
    if (amount > len) return {};
    Location l = *this;
    l.len = u16(l.len - amount);
    return l;
}

[[nodiscard]] auto Location::contract_right(isz amount) const -> Location {
    if (amount > len) return {};
    Location l = *this;
    l.pos = u32(l.pos + u32(amount));
    l.len = u16(l.len - amount);
    return l;
}

auto Location::info_or_builtin(const Context& ctx) const -> std::tuple<String, i64, i64> {
    String file = "<builtin>";
    i64 line{}, col{};
    if (auto lc = seek_line_column(ctx)) {
        file = ctx.file_name(file_id);
        line = i64(lc->line);
        col = i64(lc->col);
    }
    return {file, line, col};
}

bool Location::seekable(const Context& ctx) const {
    auto* f = ctx.file(file_id);
    if (not f) return false;
    return pos + len <= f->size() + 1 and is_valid();
}

auto Location::seek(const Context& ctx) const -> std::optional<LocInfo> {
    if (not seekable(ctx)) return std::nullopt;
    LocInfo info{};
    const auto* f = ctx.file(file_id);
    const char* const data = f->data();
    SeekLineColumn(info, data, pos);

    // Get everything before the range.
    stream s{data, pos};
    info.before = String::CreateUnsafe(s.take_back_until_any("\r\n"));

    // If the position is directly on a '\n', then we treat this as being
    // at the very end of the previous line, so stop here.
    if (data[pos] == '\n' or data[pos] == '\r') return info;

    // Next, get everything in and after the range.
    s = stream{f->contents()};
    s.drop(pos);
    info.range = String::CreateUnsafe(s.take(len));
    info.after = String::CreateUnsafe(s.take_until('\n'));

    // Done!
    return info;
}

/// TODO: Lexer should create map that counts where in a file the lines start so
/// we can do binary search on that instead of iterating over the entire file.
auto Location::seek_line_column(const Context& ctx) const -> std::optional<LocInfoShort> {
    if (not seekable(ctx)) return std::nullopt;
    LocInfoShort info{};
    SeekLineColumn(info, ctx.file(file_id)->data(), pos);
    return info;
}

auto Location::text(const Context& ctx) const -> String {
    if (not seekable(ctx)) return "";
    auto* f = ctx.file(file_id);
    return String::CreateUnsafe(StringRef{f->data(), usz(f->size())}.substr(pos, len));
}

[[nodiscard]] auto Location::operator<<(isz amount) const -> Location {
    Location l = *this;
    if (not is_valid()) return l;
    l.pos = std::min(pos, u32(pos - u32(amount)));
    return l;
}

[[nodiscard]] auto Location::operator>>(isz amount) const -> Location {
    Location l = *this;
    l.pos = std::max(pos, u32(pos + u32(amount)));
    return l;
}

[[nodiscard]] auto Location::operator<<=(isz amount) const -> Location {
    Location l = *this << amount;
    l.len = std::max(l.len, u16(l.len + amount));
    return l;
}

[[nodiscard]] auto Location::operator>>=(isz amount) const -> Location {
    Location l = *this;
    l.len = std::max(l.len, u16(l.len + amount));
    return l;
}
