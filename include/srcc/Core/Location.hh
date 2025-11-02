#ifndef SRCC_CORE_LOCATION_HH
#define SRCC_CORE_LOCATION_HH

#include <srcc/Core/Utils.hh>

namespace srcc {
class Context;
struct SLoc;
struct SRange;
struct LocInfo;
struct LocInfoShort;
class File;
}

namespace mlir {
class Location;
}

/// A short decoded source location.
struct srcc::LocInfoShort {
    i64 line;
    i64 col;
    const File* file;
};

/// A decoded source location.
struct srcc::LocInfo : LocInfoShort {
    String before, range, after;
};

/// A source range in a file.
struct srcc::SLoc {
    friend SRange;
    using Encoded = uptr;

private:
    /// The actual location data is just a pointer; we can figure out what
    /// file this is in by comparing it against the start/end pointers of
    /// each file in the context.
    const char* ptr = nullptr;

public:
    constexpr SLoc() = default;
    explicit SLoc(const char* ptr) : ptr(ptr) {}

    /// Get the location of the character right after this location.
    [[nodiscard]] auto after(const Context& ctx) const -> SLoc;

    /// Encode a location as a 64-bit number.
    [[nodiscard]] auto encode() const -> Encoded { return std::bit_cast<Encoded>(*this); }

    /// Get the file to which this location belongs. Returns nullptr
    /// if this location is invalid.
    [[nodiscard]] auto file(const Context& ctx) const -> const File*;

    /// Format this location to a string.
    [[nodiscard]] auto format(const Context& ctx, bool include_file_name) const -> std::string;

    /// Check if this is a valid source location.
    [[nodiscard]] bool is_valid() const { return ptr != nullptr; }

    /// Measure the length of the token that this points to.
    [[nodiscard]] auto measure_token_length(const Context& ctx) const -> std::optional<u64>;

    /// Get the underlying pointer.
    [[nodiscard]] auto pointer() const -> const char* { return ptr; }

    /// Seek to a source location, retrieving only line and column information.
    [[nodiscard]] auto seek_line_column(const Context& ctx) const -> std::optional<LocInfoShort>;

    /// Seek to a source range.
    [[nodiscard]] auto seek(const Context& ctx) const -> std::optional<LocInfo>;

    /// Get the text pointed to by this source location.
    [[nodiscard]] auto text(const Context& ctx) const -> String;

    /// Compare locations.
    [[nodiscard]] friend auto operator<=>(SLoc a, SLoc b) = default;

    /// Decode a source location from a 64-bit number.
    [[nodiscard]] static auto Decode(Encoded loc) -> SLoc {
        return std::bit_cast<SLoc>(loc);
    }

    /// Try to decode an MLIR location.
    [[nodiscard]] static auto Decode(mlir::Location loc) -> SLoc;

private:
    [[nodiscard]] bool seekable(const Context& ctx) const;
};

struct srcc::SRange {
    SLoc begin{};
    SLoc end{};

    constexpr SRange() = default;
    SRange(SLoc b, SLoc e) : begin(b), end(e) {
        if (begin > end) std::swap(begin, end);
    }

    /// Get the text pointed to by this source range.
    [[nodiscard]] auto text(const Context& ctx) const -> String;
};

#endif //SRCC_CORE_LOCATION_HH
