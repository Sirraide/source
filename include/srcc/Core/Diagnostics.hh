#ifndef SRCC_CORE_DIAGNOSTICS_HH
#define SRCC_CORE_DIAGNOSTICS_HH

#include <srcc/Core/Core.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Core/Location.hh>
#include <srcc/Core/Token.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/IntrusiveRefCntPtr.h>

namespace srcc {
class Diagnostic;
class DiagnosticsEngine;
class StreamingDiagnosticsEngine;
class VerifyDiagnosticsEngine;
class DiagsProducer;
}

/// A diagnostic.
///
/// This holds the data associated with a diagnostic, i.e. the source
/// location, level, and message.
class srcc::Diagnostic {
public:
    /// Diagnostic severity.
    enum struct Level : u8 {
        Ignored, ///< Do not emit this.
        Note,    ///< Informational note.
        Warning, ///< Warning, but no hard error.
        Error,   ///< Hard error. Program is ill-formed.
        ICE,     ///< Internal compiler error. Usually used for things we don’t support yet.
    };

    Level level = Level::Ignored;
    Location where;

    /// Main diagnostic message.
    std::string msg;

    /// Extra data to print after the location.
    std::string extra;

    /// Extra locations to print.
    SmallVector<std::pair<std::string, Location>, 0> extra_locations;

    /// Create an empty diagnostic.
    Diagnostic() = default;

    /// Create a diagnostic.
    Diagnostic(Level lvl, Location where, std::string msg, std::string extra = "");

    /// Create a diagnostic with a format string and arguments.
    template <typename... Args>
    Diagnostic(
        Level lvl,
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) : Diagnostic{lvl, where, std::format(fmt, std::forward<Args>(args)...)} {}

    /// Render diagnostics to text.
    ///
    /// \p render_colours If false, keep formatting codes in the
    ///    output rather than rendering it.
    static auto Render(
        const Context& ctx,
        ArrayRef<Diagnostic> diagnostics,
        usz cols,
        bool render_colours = true
    ) -> std::string;
};

/// Mixin to provide helpers for creating errors.
class srcc::DiagsProducer {
    struct Falsy {
        // Allow conversion to false.
        constexpr /* implicit */ operator bool() const { return false; }

        // Allow conversion to any nullable type.
        template <typename ty>
        requires std::convertible_to<std::nullptr_t, ty>
        constexpr /* implicit */ operator ty() const { return nullptr; }
    };

public:
    template <typename... Args>
    static auto CreateError(
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Diagnostic {
        return Diagnostic(Diagnostic::Level::Error, where, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto CreateICE(
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Diagnostic {
        return Diagnostic(Diagnostic::Level::ICE, where, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto CreateNote(
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Diagnostic {
        return Diagnostic(Diagnostic::Level::Note, where, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto CreateWarning(
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Diagnostic {
        return Diagnostic(Diagnostic::Level::Warning, where, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto Error(
        this auto& This,
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Falsy {
        This.diags().report(DiagsProducer::CreateError(where, fmt, std::forward<Args>(args)...));
        return Falsy();
    }

    template <typename... Args>
    auto ICE(
        this auto& This,
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) -> Falsy {
        This.diags().report(DiagsProducer::CreateICE(where, fmt, std::forward<Args>(args)...));
        return Falsy();
    }

    template <typename... Args>
    void Note(
        this auto& This,
        Location loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        This.diags().report(DiagsProducer::CreateNote(loc, fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void Warn(
        this auto& This,
        Location loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        This.diags().report(DiagsProducer::CreateWarning(loc, fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void Remark(
        this auto& This,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        This.diags().add_remark(std::format(fmt, std::forward<Args>(args)...));
    }
};

/// This class handles dispatching diagnostics. Objects of this
/// type are NOT thread-safe. Create a separate one for each thread.
class srcc::DiagnosticsEngine : public llvm::RefCountedBase<DiagnosticsEngine> {
    SRCC_IMMOVABLE(DiagnosticsEngine);

protected:
    /// The context that owns this engine.
    const Context& ctx;
    std::atomic<bool> error_flag = false;

public:
    using Ptr = llvm::IntrusiveRefCntPtr<DiagnosticsEngine>;
    virtual ~DiagnosticsEngine() = default;
    explicit DiagnosticsEngine(const Context& ctx) : ctx(ctx) {}

    /// Add additional location information to a diagnostic.
    template <typename... Args>
    void add_extra_location(Location extra, std::format_string<Args...> fmt, Args&&... args) {
        add_extra_location_impl(extra, std::format(fmt, std::forward<Args>(args)...));
    }

    /// Add a remark to a diagnostic as extra information.
    virtual void add_remark(std::string msg) {}

    /// How many columns the output we’re printing to has. Returns
    /// 0 if there is no column limit.
    virtual u32 cols() = 0;

    /// Issue a diagnostic.
    template <typename... Args>
    void diag(
        Diagnostic::Level lvl,
        Location where,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        report(Diagnostic{lvl, where, std::format(fmt, std::forward<Args>(args)...)});
    }

    /// Emit pending diagnostics.
    virtual void flush() {}

    /// Check whether any diagnostics have been issued.
    [[nodiscard]] bool has_error() const { return error_flag.load(std::memory_order_relaxed); }

    /// Issue a diagnostic.
    void report(Diagnostic&& diag);

protected:
    /// Override this to implement the actual reporting.
    virtual void report_impl(Diagnostic&& diag) = 0;
    virtual void add_extra_location_impl(Location, std::string) {}
};

/// Diagnostics engine that outputs to a stream.
class srcc::StreamingDiagnosticsEngine final : public DiagnosticsEngine {
    llvm::raw_ostream& stream;

    /// Used to limit how many errors we print before giving up.
    u32 error_limit;
    u32 printed = 0;

    /// Backlog of diagnostics so we can group notes with errors/warnings.
    SmallVector<Diagnostic, 20> backlog;

    StreamingDiagnosticsEngine(const Context& ctx, u32 error_limit, llvm::raw_ostream& output_stream);
    ~StreamingDiagnosticsEngine() override;

public:
    /// Create a new diagnostic engine.
    [[nodiscard]] static auto Create(
        const Context& ctx,
        u32 error_limit = 0,
        llvm::raw_ostream& output_stream = llvm::errs()
    ) -> Ptr;

    void add_remark(std::string msg) override;
    u32 cols() override;
    void flush() override;

private:
    void add_extra_location_impl(Location, std::string) override;
    void report_impl(Diagnostic&&) override;
    void EmitDiagnostics();
};

class srcc::VerifyDiagnosticsEngine final : public DiagnosticsEngine
    , DiagsProducer
    , public text::ColourFormatter {
public:
    /// Type of the callback used to handle comment tokens.
    using CommentTokenCallback = std::function<void(const Token&)>;

private:
    struct DecodedLocation {
        const File* file;
        i64 line;
        constexpr auto operator<=>(const DecodedLocation&) const = default;
    };

    struct ExpectedDiagnostic {
        Diagnostic::Level level;
        std::string text;
        std::optional<DecodedLocation> loc;
        u32 count;
    };

    struct SeenDiagnostic {
        Diagnostic diag;
        std::optional<DecodedLocation> loc;
    };

    SmallVector<SeenDiagnostic, 0> seen_diags;
    SmallVector<ExpectedDiagnostic, 0> expected_diags;
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine> diags_reporter;
    bool expects_none = false;
    bool enable_colours;

public:
    explicit VerifyDiagnosticsEngine(const Context& ctx);

    /// Print everything in one line.
    u32 cols() override { return 0; }

    /// Get the diagnostics engine for reporting errors during the verification
    /// and comment parsing steps.
    auto diags() const -> DiagnosticsEngine& { return *diags_reporter; }

    /// Get the comment callback that should be called by the lexer
    /// when it encounters a comment token.
    auto comment_token_callback() -> CommentTokenCallback {
        return [this](const Token& tok) { HandleCommentToken(tok); };
    }

    void report_impl(Diagnostic&& diags) override;

    bool use_colour() const { return enable_colours; }

    /// Run the verification step.
    ///
    /// \return True on success.
    bool verify();

    /// Create a new diagnostic engine.
    [[nodiscard]] static auto Create(const Context& ctx) -> Ptr {
        return llvm::IntrusiveRefCntPtr(new VerifyDiagnosticsEngine(ctx));
    }

private:
    friend DiagsProducer;

    // For reporting errors during the verification and comment parsing steps.
    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        diags_reporter->diag(lvl, where, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Error(Location loc, std::format_string<Args...> fmt, Args&&... args) {
        Diag(Diagnostic::Level::Error, loc, fmt, std::forward<Args>(args)...);
    }

    auto DecodeLocation(Location loc) -> Opt<DecodedLocation>;
    void HandleCommentToken(const Token& tok);
    void ParseMagicComment(str comment, Location loc);
};

#endif // SRCC_CORE_DIAGNOSTICS_HH
