#include <srcc/Core/Core.hh>
#include <srcc/Core/Location.hh>

using namespace srcc;

/// Create a new location that spans two locations.
Location::Location(Location a, Location b) {
    if (a.file_id != b.file_id) return;
    if (not a.is_valid() or not b.is_valid()) return;
    pos = std::min<u32>(a.pos, b.pos);
    len = u16(std::max<u32>(a.pos + a.len, b.pos + b.len) - pos);
}

/// Contract a source location to the left.
[[nodiscard]] auto Location::contract_left(isz amount) const -> Location {
    if (amount > len) return {};
    Location l = *this;
    l.len = u16(l.len - amount);
    return l;
}

/// Contract a source location to the right.
[[nodiscard]] auto Location::contract_right(isz amount) const -> Location {
    if (amount > len) return {};
    Location l = *this;
    l.pos = u32(l.pos + u32(amount));
    l.len = u16(l.len - amount);
    return l;
}

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

/// Shift a source location to the left.
[[nodiscard]] auto Location::operator<<(isz amount) const -> Location {
    Location l = *this;
    if (not is_valid()) return l;
    l.pos = std::min(pos, u32(pos - u32(amount)));
    return l;
}

/// Shift a source location to the right.
[[nodiscard]] auto Location::operator>>(isz amount) const -> Location {
    Location l = *this;
    l.pos = std::max(pos, u32(pos + u32(amount)));
    return l;
}

/// Extend a source location to the left.
[[nodiscard]] auto Location::operator<<=(isz amount) const -> Location {
    Location l = *this << amount;
    l.len = std::max(l.len, u16(l.len + amount));
    return l;
}

/// Extend a source location to the right.
[[nodiscard]] auto Location::operator>>=(isz amount) const -> Location {
    Location l = *this;
    l.len = std::max(l.len, u16(l.len + amount));
    return l;
}
