#ifndef SRCC_CORE_LOCATION_HH
#define SRCC_CORE_LOCATION_HH

#include <srcc/Core/Utils.hh>

namespace srcc {
class Context;
struct Location;
struct LocInfo;
struct LocInfoShort;
}

namespace mlir {
class Location;
}

/// A short decoded source location.
struct srcc::LocInfoShort {
    i64 line;
    i64 col;
};

/// A decoded source location.
struct srcc::LocInfo : LocInfoShort {
    String before, range, after;
};

/// A source range in a file.
struct srcc::Location {
    using Encoded = uptr;

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

    /// Get the location of the character right after this location.
    [[nodiscard]] auto after() const -> Location;

    /// Contract a source location to the left.
    [[nodiscard]] auto contract_left(isz amount) const -> Location;

    /// Contract a source location to the right.
    [[nodiscard]] auto contract_right(isz amount) const -> Location;

    /// Compare two source locations for equality.
    [[nodiscard]] bool operator==(const Location& other) const = default;

    /// Encode a location as a 64-bit number.
    [[nodiscard]] auto encode() const -> Encoded { return std::bit_cast<Encoded>(*this); }

    /// Get file line and column or return <builtin:0:0> if invalid.
    [[nodiscard]] auto info_or_builtin(const Context& ctx) const -> std::tuple<String, i64, i64>;

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
    static auto Decode(Encoded loc) -> Location {
        return std::bit_cast<Location>(loc);
    }

    /// Try to decode an MLIR location.
    static auto Decode(mlir::Location loc) -> Location;

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
