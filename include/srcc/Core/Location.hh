#ifndef SRCC_CORE_LOCATION_HH
#define SRCC_CORE_LOCATION_HH

#include <srcc/Core/Utils.hh>

namespace srcc {
class Context;
struct Location;
struct LocInfo;
struct LocInfoShort;
}

/// A short decoded source location.
struct srcc::LocInfoShort {
    usz line;
    usz col;
};

/// A decoded source location.
struct srcc::LocInfo {
    usz line;
    usz col;
    const char* line_start;
    const char* line_end;

    auto short_info() const -> LocInfoShort { return {line, col}; }
};

/// A source range in a file.
struct srcc::Location {
    u32 pos{};
    u16 len{};
    u16 file_id{};

    constexpr Location() = default;
    Location(u32 pos, u16 len, u16 file_id)
        : pos(pos), len(len), file_id(file_id) {}

    /// Create a new location that spans two locations.
    Location(Location a, Location b);

    /// Shift a source location to the left.
    [[nodiscard]] auto operator<<(isz amount) const -> Location;

    /// Shift a source location to the right.
    [[nodiscard]] auto operator>>(isz amount) const -> Location;

    /// Extend a source location to the left.
    [[nodiscard]] auto operator<<=(isz amount) const -> Location;

    /// Extend a source location to the right.
    [[nodiscard]] auto operator>>=(isz amount) const -> Location;

    /// Contract a source location to the left.
    [[nodiscard]] auto contract_left(isz amount) const -> Location;

    /// Contract a source location to the right.
    [[nodiscard]] auto contract_right(isz amount) const -> Location;

    /// Compare two source locations for equality.
    [[nodiscard]] bool operator==(const Location& other) const = default;

    /// Encode a location as a 64-bit number.
    [[nodiscard]] u64 encode() const { return std::bit_cast<u64>(*this); }

    [[nodiscard]] bool is_valid() const { return len != 0; }

    /// Seek to a source location.
    [[nodiscard]] auto seek(const Context& ctx) const -> std::optional<LocInfo>;

    /// Seek to a source location, but only return the line and column.
    [[nodiscard]] auto seek_line_column(const Context& ctx) const -> std::optional<LocInfoShort>;

    /// Get the text pointed to by this source location.
    ///
    /// This returns a StringRef instead of a String because the returned
    /// range is almost certainly not null-terminated.
    [[nodiscard]] auto text(const Context& ctx) const -> String;

    /// Decode a source location from a 64-bit number.
    static auto Decode(u64 loc) -> Location {
        return std::bit_cast<Location>(loc);
    }

private:
    [[nodiscard]] bool seekable(const Context& ctx) const;
};

template <>
struct std::formatter<srcc::Location> : std::formatter<std::string> {
    template <typename FormatContext>
    auto format(srcc::Location l, FormatContext& ctx) const {
        auto str = std::format("[{}:{}, {}]", l.pos, l.len, l.file_id);
        return std::formatter<std::string>::format(str, ctx);
    }
};

#endif //SRCC_CORE_LOCATION_HH
