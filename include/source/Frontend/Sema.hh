#ifndef SOURCE_FRONTEND_SEMA_HH
#define SOURCE_FRONTEND_SEMA_HH

#include <source/Frontend/AST.hh>
#include <source/Support/Buffer.hh>

namespace src {
template <typename T>
struct make_formattable {
    using type = T;
};

template <typename T>
requires std::is_reference_v<T>
struct make_formattable<T> {
    using type = make_formattable<std::remove_reference_t<T>>::type;
};

template <std::derived_from<Expr> T>
struct make_formattable<T*> {
    using type = std::string;
};

template <>
struct make_formattable<Type> {
    using type = std::string;
};

template <>
struct make_formattable<const Type> {
    using type = std::string;
};

template <typename T>
using make_formattable_t = make_formattable<T>::type;

class Sema {
    Module* mod{};
    ProcDecl* curr_proc{};
    BlockExpr* curr_scope{};

    /// Loops that we’re currently analysing.
    SmallVector<Loop*, 10> loop_stack;

    /// The defer expression whose contents we are currently analysing.
    DeferExpr* curr_defer{};

    /// Expressions that require unwinding once everything else has been checked.
    struct Unwind {
        BlockExpr* in_scope;
        UnwindExpr* expr;

        /// Unused if this is a goto.
        BlockExpr* to_scope{};
    };

    /// Conversion sequence to convert from one type to another.
    ///
    /// Each conversion in the sequence entails performing one
    /// of the following actions; at most one CastExpr may be
    /// created for each conversion sequence.
    ///
    ///     - Creating a CastExpr.
    ///     - Creating a ConstExpr with a certain value.
    ///     - Calling a constructor.
    struct ConversionSequence {
        struct BuildCast {
            CastKind kind;
            Type to;
        };

        struct BuildConstExpr {
            /// Empty so we don’t store 27 EvalResults for a large
            /// conversion sequence; since we can only construct
            /// one ConstExpr anyway, a ConversionSequence only
            /// needs to store a single EvalResult.
        };

        struct CallConstructor {
            ProcDecl* ctor;
            Buffer<Expr*> args;
        };

        /// Convert an overload set to a DeclRefExpr to a procedure.
        struct OverloadSetToProc {
            ProcDecl* proc;
        };

        using Entry = std::variant<BuildCast, BuildConstExpr, CallConstructor, OverloadSetToProc>;
        SmallVector<Entry> entries;
        std::optional<EvalResult> constant;
        int score{};

        static void ApplyCast(Sema& s, Expr*& e, CastKind kind, Type to);
        static void ApplyConstExpr(Sema& s, Expr*& e, EvalResult res);
        static void ApplyConstructor(Sema& s, Expr*& e, ProcDecl* ctor, ArrayRef<Expr*> args);
        static void ApplyOverloadSetToProc(Sema& s, Expr*& e, ProcDecl* proc);
    };

    /// Helper for checking conversions.
    template <bool perform_conversion>
    struct ConversionContext {
        Sema& S;
        ConversionSequence* seq;
        Expr** e;
        int score{};
        bool has_expr{};

        ConversionContext(Sema& s, ConversionSequence& seq, Expr** e = nullptr)
        requires (not perform_conversion)
            : S(s), seq(&seq), e(e) {
            if (e) has_expr = true;
        }

        ConversionContext(Sema& s, Expr*& e)
        requires perform_conversion
            : S(s), seq(nullptr), e(&e) {
            has_expr = true;
        }

        /// Get the current expression; this can only be valid if the conversion
        /// sequence is empty or if the last entry is a ConstExpr; otherwise, the
        /// expression has already been converted to something else; note that there
        /// may also be no expression at all in some cases if we’re just checking
        /// whether two types are convertible with one another.
        readonly(Expr*, expr, return has_expr ? *e : nullptr);

        /// Whether the current expression is an lvalue.
        readonly(bool, is_lvalue, return expr and expr->is_lvalue);

        /// Emit a cast.
        Type cast(CastKind k, Type to);

        /// Emit lvalue-to-rvalue conversion.
        void lvalue_to_rvalue();

        /// Emit a conversion from an overload set to a procedure.
        Type overload_set_to_proc(ProcDecl* proc);

        /// Replace the expression with a constant.
        Type replace_with_constant(EvalResult&& res);

        /// Attempt to evaluate this as a constant expression.
        bool try_evaluate(EvalResult& out);
    };

    /// Overload candidate.
    struct Candidate {
        enum struct Status {
            Viable,
            ArgumentCountMismatch,
            ArgumentTypeMismatch,
            NoViableArgOverload,
        };

        enum { InvalidScore = -1 };

        ProcDecl* proc;
        SmallVector<ConversionSequence> arg_convs;
        Status s = Status::Viable;
        int score = InvalidScore;

        readonly_const(ProcType*, type, return cast<ProcType>(proc->type));
        usz mismatch_index{};
    };

    /// Result of overload resolution.
    struct OverloadResolutionResult {
        /// We couldn’t find any viable overloads.
        struct ResolutionFailure {
            Sema& S;
            Location where;
            SmallVector<Candidate> overloads;
            ArrayRef<Expr*> args;
            bool suppress_diagnostics = false;

            /// Suppress diagnostics for this resolution failure.
            void suppress() { suppress_diagnostics = true; }

            ResolutionFailure(Sema& s, Location w, SmallVector<Candidate> o, ArrayRef<Expr*> a)
                : S(s), where(w), overloads(std::move(o)), args(a) {}

            ResolutionFailure(const ResolutionFailure&) = delete;
            ResolutionFailure& operator=(const ResolutionFailure&) = delete;
            ResolutionFailure(ResolutionFailure&& other) : S(other.S) {
                *this = std::move(other);
            }

            ResolutionFailure& operator=(ResolutionFailure&& other) {
                if (this == std::addressof(other)) return *this;
                S = other.S;
                where = other.where;
                overloads = std::move(other.overloads);
                args = other.args;
                suppress_diagnostics = std::exchange(other.suppress_diagnostics, true);
                return *this;
            }

            /// Emit diagnostics for overload resolution failure.
            ~ResolutionFailure();
        };

        /// We found a single viable overload, but there was some other problem
        /// with it not relating to overloading that causes the program to be
        /// ill-formed (e.g. parameter intent problems). Alternatively, we didn’t
        /// even get to overload resolution because there was something wrong with
        /// e.g. the arguments.
        ///
        /// This always indicates an ill-formed program. Do not attempt to recover
        /// from this.
        using IllFormed = std::nullptr_t;

        /// We found a single viable overload.
        using Ok = ProcDecl*;

        /// The result.
        using Data = std::variant<ResolutionFailure, IllFormed, Ok>;
        Data result;

        /// Construct a new result.
        OverloadResolutionResult(IllFormed) : result(nullptr) {}
        OverloadResolutionResult(Ok ok) : result(ok) {}
        OverloadResolutionResult(ResolutionFailure f) : result(std::move(f)) {}

        /// Whether this makes the program ill-formed.
        bool ill_formed() const { return std::holds_alternative<IllFormed>(result); }

        /// Whether this is a resolution failure.
        bool failure() const { return std::holds_alternative<ResolutionFailure>(result); }

        /// Get the procedure that was resolved.
        auto resolved() const -> ProcDecl* { return std::get<Ok>(result); }

        /// Check if this is at all valid.
        explicit operator bool() const { return not ill_formed() and not failure(); }
    };

    /// Machinery for tracking the state of lvalues.
    class LValueState {
        using Path = SmallVector<u32, 4>;
        struct State {
            /// Whether the lvalue itself is an active optional. Always true
            /// if this is not an optional.
            bool active_optional : 1;

            /// List of field paths that are active, by field index.
            SmallVector<Path, 1> active_fields;
        };

        struct Moves {
            /// Location of move of the entire variable, if any.
            Location loc{};

            /// Locations of moves of subobjects.
            std::map<Path, Location> subobjects{};
        };

    public:
        /// Guard for entering a scope.
        class ScopeGuard {
            friend LValueState;
            Sema& S;
            ScopeGuard* previous;

            /// Entities of optional type whose active state has changed in
            /// this scope, as well as the value of that state in the previous
            /// scope.
            ///
            /// The root of an entry is always a local variable.
            DenseMap<LocalDecl*, State> changes;

        public:
            ScopeGuard(Sema& S);
            ~ScopeGuard();
        };

        /// Guard for temporarily activating and optional. This always
        /// resets the active state to what it was before the guard,
        /// irrespective of whether it is changed after the guard was
        /// created.
        struct OptionalActivationGuard {
            friend LValueState;
            Sema& S;
            Expr* expr;

        public:
            OptionalActivationGuard(Sema& S, Expr* expr);
            ~OptionalActivationGuard();
        };

    private:
        Sema* S;

        /// All entities that are currently tracked, and whether they
        /// are active. Key is the root object of the entity, which
        /// is always a local variable (this is because tracking only
        /// makes sense for lvalues since rvalues don’t survive long
        /// enough to be tracked in the first place).
        DenseMap<LocalDecl*, State> tracked;

        /// Location of last move for variables; used only for diagnostics.
        DenseMap<LocalDecl*, Moves> last_moves;

        /// Current guard.
        ScopeGuard* guard{};

        /// Used to implement Activate() and Deactivate().
        void ChangeOptionalState(Expr* e, auto cb);

        /// Get the path to an entity of optional type.
        auto GetObjectPath(MemberAccessExpr* e) -> std::pair<LocalDecl*, Path>;

    public:
        LValueState(Sema* S) : S(S) {}

        /// Mark an optional as active until the end of the current scope.
        ///
        /// \param e The expression to activate. May be null.
        void ActivateOptional(Expr* e);

        /// Mark an optional as inactive until the end of the current scope.
        ///
        /// \param e The expression to deactivate. May be null.
        void DeactivateOptional(Expr* e);

        /// If this is an active entity of optional type, return the optional
        /// type, else return null.
        auto GetActiveOptionalType(Expr* e) -> OptionalType*;

        /// Test if an expression checks whether an entity of optional type
        /// is nil, and if so, return a pointer to that entity.
        auto MatchOptionalNilTest(Expr* test) -> Expr*;

        /// Mark an object as definitely moved from.
        ///
        /// \param e The expression to mark as moved from. May be null.
        void SetDefinitelyMoved(Expr* e);
    } LValueState{this};

    /// Expressions that still need to be checked for unwinding.
    SmallVector<Unwind> unwind_entries;

    /// Expressions that need to know what the current full expression is.
    SmallVector<Expr*> needs_link_to_full_expr{};

    /// Expressions eligible for `.x` access.
    SmallVector<Expr*> with_stack;

    /// In some cases, name lookup has to consider different scopes other
    /// than just the scope of the identifier—such as when an enum inherits
    /// from an enum declared in a different scope.
    class DeclContext {
        /// Currently always an enum.
        EnumType* scope;

    public:
        class [[nodiscard]] Guard {
            Sema& s;

        public:
            template <typename... Args>
            Guard(Sema* s, Args&&... args) : s(*s) {
                s->decl_contexts.push_back(DeclContext{std::forward<Args>(args)...});
            }

            ~Guard() {
                s.decl_contexts.pop_back();
            }
        };

        struct Entry {
            Expr* expr{};
            Type type{Type::Unknown}; ///< DeclRefExprs may need to receive a different type.
            explicit operator bool() { return expr != nullptr; }
        };

        DeclContext(EnumType* e) : scope(e) {}

        /// Find an entry in this context.
        auto find(Sema& s, String name) -> Entry;
    };

    /// Active declaration contexts.
    SmallVector<DeclContext> decl_contexts;

    /// Enums that are currently being analysed.
    ///
    /// DeclRefExprs that reference enum members within an enum have
    /// their type set to the underlying type of the enum instead of
    /// the enum type.
    SmallVector<EnumType*> open_enums;

    /// Get the current Source context.
    readonly(Context*, ctx, return mod->context);

    /// Number of anonymous procedures.
    usz lambda_counter = 0;

    /// Whether we’re currently the direct child of a block.
    bool at_block_level = false;

    /// Whether to print unsupported C++ imports.
    bool debug_cxx = false;

    /// Whether we’re in an unevaluated context.
    bool unevaluated = false;

public:
    Sema(Module* mod) : mod(mod) {}

    /// Use Context::has_error to check for errors.
    static void Analyse(Module* mod, bool debug_cxx = false) {
        Sema s{mod};
        s.debug_cxx = debug_cxx;
        s.AnalyseModule();
    }

    /// Analyse the given expression and issue an error if it is not a type.
    bool AnalyseAsType(Type& e, bool diag_if_not_type = true);

private:
    bool Analyse(Expr*& e);

    template <std::derived_from<Expr> Expression>
    requires (not std::same_as<Expression, Expr>)
    bool Analyse(Expression*& e) {
        Expr *expr = e;
        if (not Analyse(expr)) return false;
        e = cast<Expression>(expr);
        return true;
    }

    template <bool allow_undefined>
    bool AnalyseDeclRefExpr(Expr*& e);

    void AnalyseExplicitCast(Expr*& e, bool is_hard);
    bool AnalyseInvoke(Expr*& e, bool direct_child_of_block = false);
    bool AnalyseInvokeBuiltin(Expr*& e);
    void AnalyseModule();
    void AnalyseProcedure(ProcDecl* proc);
    bool AnalyseProcedureType(ProcDecl* proc);
    void AnalyseRecord(RecordType* r);
    bool AnalyseVariableInitialisation(Expr* e, ConstructExpr*& ctor, Type& type, SmallVectorImpl<Expr*>& init_args);

    /// Apply a conversion sequence to an expression.
    void ApplyConversionSequence(Expr*& e, std::same_as<ConversionSequence> auto&& seq);

    /// Determine whether a parameter should be passed by value and check its type.
    /// \return False on error.
    bool ClassifyParameter(ParamInfo* info);

    /// Get a constructor for a type and a set of arguments.
    ///
    /// \param loc Location to use for errors.
    /// \param ty The type to construct.
    /// \param args Arguments with which to construct a \p ty.
    /// \param target Expression that is marked as active if this creates an active optional.
    /// \param raw Disregard initialisers at the top level (used for raw literals).
    /// \return The constructor for `into`.
    auto Construct(
        Location loc,
        Type ty,
        MutableArrayRef<Expr*> args,
        Expr* target = nullptr,
        bool raw = false
    ) -> ConstructExpr*;

    /// Convert an expression to a type, inserting implicit conversions
    /// as needed. This  *will* perform lvalue-to-rvalue conversion if
    /// the type conversion requires it and also in any case unless \p
    /// lvalue is true.
    bool Convert(Expr*& e, Type to, bool lvalue = false);

    /// Implements Convert() and TryConvert().
    template <bool perform_conversion>
    bool ConvertImpl(
        ConversionContext<perform_conversion>& ctx,
        Type from, /// Required for recursive calls in non-conversion mode.
        Type to
    );

    /// Ensure that an expression is valid as the condition of an if expression,
    /// while loop, etc.
    bool EnsureCondition(Expr*& e);

    /// Create an implicit dereference, but do not overwrite the original expression.
    [[nodiscard]] auto CreateImplicitDereference(Expr* e, isz depth) -> Expr*;

    /// Create a diagnostic at a location.
    template <typename... Args>
    Diag EmitDiag(Diag::Kind k, Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        if (k == Diag::Kind::Error) {
            if (e->sema.errored) return Diag();
            e->sema.set_errored();
        }

        return Diag(mod->context, k, e->location, fmt, MakeFormattable(std::forward<Args>(args))...);
    }

    /// Create a diagnostic and mark an expression as errored.
    template <typename... Args>
    Diag EmitDiag(Diag::Kind k, Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        return Diag(mod->context, k, loc, fmt, MakeFormattable(std::forward<Args>(args))...);
    }

    /// Emit an error.
    template <typename... Args>
    Diag EmitError(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        return EmitDiag(Diag::Kind::Error, loc, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    Diag EmitError(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        return EmitDiag(Diag::Kind::Error, e, fmt, std::forward<Args>(args)...);
    }

    /// Same as EmitError(), but returns false for convenience.
    template <typename... Args>
    bool Error(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitError(loc, fmt, std::forward<Args>(args)...);
        return false;
    }

    /// Same as EmitError(), but returns false for convenience.
    template <typename... Args>
    bool Error(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitError(e, fmt, std::forward<Args>(args)...);
        return false;
    }

    /// Evaluate a constant expression.
    bool Evaluate(Expr* e, EvalResult& out, bool must_succeed = true);

    /// Evaluate an integral constant expression and replace it with the result.
    ///
    /// \param e The expression to replace.
    /// \param must_succeed Whether to issue an error if evaluation fails.
    /// \param integer_or_bool The desired type this must have. If the value is
    ///        not implicitly convertible to this type, evaluation fails. If unset
    ///        any integral or boolean type is accepted.
    /// \return Whether evaluation succeeded.
    bool EvaluateAsIntegerInPlace(
        Expr*& e,
        bool must_succeed = true,
        std::optional<Type> integer_or_bool = std::nullopt
    );

    /// Evaluate a boolean constant expression and replace it with the result.
    bool EvaluateAsBoolInPlace(Expr*& e, bool must_succeed = true);

    /// Evaluate a constant expression as an overload set. This must always
    /// yield an overload set, so if this fails, it’s an ICE. Only call this
    /// if the type of the expression is actually OverloadSet.
    auto EvaluateAsOverloadSet(Expr* e) -> OverloadSetExpr*;

    /// Perform any final operations (after type conversion) required to
    /// pass an expression an an argument to a call.
    ///
    /// \param arg The argument to finalise.
    /// \param param The parameter to which the argument is being passed, or
    ///        nullptr if this is a variadic argument.
    /// \return False on error.
    bool FinaliseInvokeArgument(Expr*& arg, const ParamInfo* param);

    /// Dereference a reference, yielding an lvalue.
    ///
    /// This automatically handles dereferencing both references that
    /// are themselves lvalues and rvalues.
    void InsertImplicitDereference(Expr*& e, isz depth);

    /// Perform lvalue-to-rvalue conversion.
    ///
    /// Notably, this does *not* change the type of the expression; unlike
    /// in C++, expressions of reference type can be rvalues or lvalues.
    void InsertLValueToRValueConversion(Expr*& e);

    /// Check if this is an 'in' parameter.
    bool IsInParameter(Expr* e);

    template <bool in_array = false>
    bool MakeDeclType(Type& e);

    template <typename T>
    auto MakeFormattable(T&& t) -> make_formattable_t<T> {
        using Type = std::remove_cvref_t<T>;
        if constexpr (std::is_pointer_v<Type> and std::derived_from<std::remove_pointer_t<Type>, Expr>) {
            return std::forward<T>(t)->type.str(mod->context->use_colours, true);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<Type>, src::Type>) {
            return std::forward<T>(t).str(mod->context->use_colours, true);
        } else {
            return std::forward<T>(t);
        }
    }

    /// Materialise a temporary value.
    ///
    /// This is the opposite of lvalue-to-rvalue conversion: it takes an rvalue and
    /// converts it to a temporary. Note that nothing happens if the expression is
    /// already a MaterialiseTemporaryExpr of the same type.
    bool MaterialiseTemporary(Expr*& e, Type type);

    /// Materialise a temporary value if it is an rvalue.
    ///
    /// Lvalues are unaffected by this.
    void MaterialiseTemporaryIfRValue(Expr*& e);

    /// Create a Note().
    template <typename... Args>
    void Note(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitDiag(Diag::Kind::Note, loc, fmt, std::forward<Args>(args)...);
    }

    /// Create a Note().
    template <typename... Args>
    void Note(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitDiag(Diag::Kind::Note, e, fmt, std::forward<Args>(args)...);
    }

    /// Resolve overload set.
    ///
    /// \param where Location for issuing diagnostics.
    /// \param overloads The overloads to resolve.
    /// \param args The arguments to the call.
    /// \return The resolved overload, or nullptr if resolution failed.
    auto PerformOverloadResolution(
        Location where,
        ArrayRef<ProcDecl*> overloads,
        MutableArrayRef<Expr*> args
    ) -> OverloadResolutionResult;

    /// Search an enum for an enumerator
    auto SearchEnumScope(EnumType* enum_type, String name) -> DeclContext::Entry;

    /// Like Convert(), but does not perform the conversion, and does not
    /// issue any diagnostics on conversion failure.
    ///
    /// This  *will* perform lvalue-to-rvalue conversion if
    /// the type conversion requires it and also in any case unless \p
    /// lvalue is true.
    bool TryConvert(ConversionSequence& out, Expr* e, Type to, bool lvalue = false);

    /// Strip references and optionals (if they’re active) from the expression
    /// to yield the underlying value.
    [[nodiscard]] Expr* Unwrap(Expr* e, bool keep_lvalues = false);

    /// Unwrap an expression and replace it with the unwrapped expression.
    void UnwrapInPlace(Expr*& e, bool keep_lvalues = false);

    /// Get the unwrapped type of an expression.
    Type UnwrappedType(Expr* e);

    /// Unwinder.
    ///
    /// See comments in Sema.cc for a detailed explanation of all of this.
    ///
    /// Used for stack unwinding as part of direct branches (goto, break
    /// continue, return).
    ///
    /// If this is a small vector, store unwound expressions in it. If it
    /// is an expression, instead emit an error and mark that expression as
    /// errored.
    using UnwindContext = llvm::PointerUnion<SmallVectorImpl<Expr*>*, Expr*>;
    bool UnwindLocal(UnwindContext ctx, BlockExpr* S, Expr* FE, Expr* To);
    auto Unwind(UnwindContext ctx, BlockExpr* S, Expr* E, BlockExpr* To) -> Expr*;
    void UnwindUpTo(BlockExpr* parent, BlockExpr* to, UnwindExpr* uw);
    void ValidateDirectBr(GotoExpr* g, BlockExpr* source);
};
} // namespace src

#endif // SOURCE_FRONTEND_SEMA_HH
