#ifndef SRCC_FRONTEND_SEMA_HH
#define SRCC_FRONTEND_SEMA_HH

#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/TinyPtrVector.h>

#include <base/Macros.hh>

namespace srcc {
class ModuleLoader;
class Sema;
}

namespace srcc {
class TemplateInstantiator;
}

class srcc::Sema : public DiagsProducer {
    SRCC_IMMOVABLE(Sema);

    class Importer;
    class ImmediateInitContext;
    class OverloadInitContext;
    class TentativeInitContext;

public:
    class ASTReader;
    class ASTWriter;

private:
    friend DiagsProducer;
    friend eval::Eval;
    friend TemplateInstantiator;
    friend Importer;

    /// RAII Object to push and pop a scope.
    class [[nodiscard]] EnterScope {
    public:
        Sema& S;

    private:
        Scope* scope = nullptr;

    public:
        EnterScope(Sema& S, ScopeKind kind = ScopeKind::Block, bool should_enter = true);
        EnterScope(Sema& S, bool should_enter);
        EnterScope(Sema& S, Scope* scope);
        ~EnterScope();

        /// Not copyable since copying scopes is nonsense.
        EnterScope(const EnterScope&) = delete;
        const EnterScope& operator=(const EnterScope&) = delete;

        /// However, we may want to pass these to functions in some cases
        /// to open a new scope or indicate that they do so.
        EnterScope(EnterScope&& other) : S{other.S}, scope{std::exchange(other.scope, nullptr)} {}
        EnterScope& operator=(EnterScope&& other) {
            Assert(&S == &other.S, "Cannot move scope between semas");
            scope = std::exchange(other.scope, nullptr);
            return *this;
        }

        /// Get the scope.
        auto get() const -> Scope* {
            Assert(scope, "Accessing scope after it has been moved");
            return scope;
        }
    };

    struct ProcScopeInfo {
        SRCC_IMMOVABLE(ProcScopeInfo);
        ProcDecl* proc;
        SmallVector<LocalDecl*> locals;
        LoopToken loop_depth = LoopToken(0);
        bool current_loop_has_break = false;
        const EnterScope es;

        ProcScopeInfo(Sema& S, ProcDecl* proc)
            : proc{proc}, es{S, proc->scope} {}
    };

    class [[nodiscard]] EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);
        ProcScopeInfo info;

    public:
        explicit EnterProcedure(Sema& S, ProcDecl* proc);
        ~EnterProcedure() { info.es.S.proc_stack.pop_back(); }
    };

    class [[nodiscard]] EnterLoop {
        SRCC_IMMOVABLE(EnterLoop);
        Sema& S;
        bool save_current_loop_has_break;

    public:
        explicit EnterLoop(Sema& S);
        ~EnterLoop();

        [[nodiscard]] auto token() -> LoopToken;
    };

    /// A (possibly empty) sequence of conversions applied to a type.
    class ConversionSequence;

    /// A conversion from one type to another.
    class Conversion {
        LIBBASE_MOVE_ONLY(Conversion);

    public:
        struct RecordInitData {
            RecordType* ty;
            std::vector<ConversionSequence> field_convs;
        };

        struct ArrayInitData {
            ArrayType* ty;
            std::vector<ConversionSequence> elem_convs{};

            /// This needs to be a flag so as to distinguish i[2](1) from i[2](1, 0);
            /// the former has a broadcast initialiser but the latter doesn’t.
            bool has_broadcast_initialiser = false;

            /// Get the element to broadcast into the rest of the array, after all initialisers
            /// proper have been evaluated.
            [[nodiscard]] auto broadcast_initialiser() const -> const ConversionSequence* {
                if (not has_broadcast_initialiser) return nullptr;
                return &elem_convs.back();
            }
        };

        struct ArrayBroadcastData {
            ArrayType* type;
            std::unique_ptr<ConversionSequence> seq;
        };

        struct SliceFromPtrAndSizeData {
            SliceType* slice;
            std::unique_ptr<ConversionSequence> ptr;
            std::unique_ptr<ConversionSequence> size;
        };

        enum struct Kind : u8 {
            ArrayBroadcast,
            ArrayDecay,
            ArrayInit,
            DefaultInit,
            ExpandTuple,
            IntegralCast,
            LValueToRValue,
            MaterialisePoison,
            MaterialiseTemporary,
            NilToOptional,
            OptionalUnwrap,
            OptionalWrap,
            RangeCast,
            RecordInit,
            SelectOverload,
            SliceFromArray,
            SliceFromPtrAndSize,
            StripParens,
            StrLitToCStr,
            TupleToFirstElement,
        };

        Kind kind;
        Variant< // clang-format off
            TypeAndValueCategory,
            RecordInitData,
            ArrayInitData,
            ArrayBroadcastData,
            SliceFromPtrAndSizeData,
            u32
        > data; // clang-format on

    private:
        Conversion(Kind kind) : kind{kind} {}
        Conversion(Kind kind, Type ty, ValueCategory val = Expr::RValue) : kind{kind}, data{TypeAndValueCategory(ty, val)} {}
        Conversion(Kind kind, u32 index) : kind{kind}, data{index} {}
        Conversion(RecordInitData conversions) : kind{Kind::RecordInit}, data{std::move(conversions)} {}
        Conversion(ArrayInitData data) : kind{Kind::ArrayInit}, data{std::move(data)} {}
        Conversion(ArrayBroadcastData data) : kind{Kind::ArrayBroadcast}, data{std::move(data)} {}
        Conversion(SliceFromPtrAndSizeData data) : kind{Kind::SliceFromPtrAndSize}, data{std::move(data)} {}

    public:
        ~Conversion();
        static auto ArrayBroadcast(ArrayBroadcastData data) -> Conversion { return Conversion{std::move(data)}; }
        static auto ArrayDecay(Type ty) -> Conversion { return Conversion{Kind::ArrayDecay, ty}; }
        static auto ArrayInit(ArrayInitData data) -> Conversion { return Conversion{std::move(data)}; }
        static auto DefaultInit(Type ty) -> Conversion { return Conversion{Kind::DefaultInit, ty}; }
        static auto ExpandTuple() -> Conversion { return Conversion{Kind::ExpandTuple}; }
        static auto IntegralCast(Type ty) -> Conversion { return Conversion{Kind::IntegralCast, ty}; }
        static auto LValueToRValue() -> Conversion { return Conversion{Kind::LValueToRValue}; }
        static auto MaterialiseTemporary() -> Conversion { return Conversion{Kind::MaterialiseTemporary}; }
        static auto NilToOptional(Type ty) -> Conversion { return Conversion{Kind::NilToOptional, ty}; }
        static auto OptionalUnwrap(Type ty) -> Conversion { return Conversion{Kind::OptionalUnwrap, ty}; }
        static auto OptionalWrap(Type ty) -> Conversion { return Conversion{Kind::OptionalWrap, ty}; }
        static auto Poison(Type ty, ValueCategory val) -> Conversion { return Conversion{Kind::MaterialisePoison, ty, val}; }
        static auto RangeCast(Type ty) -> Conversion { return Conversion{Kind::RangeCast, ty}; }
        static auto RecordInit(RecordInitData conversions) -> Conversion { return Conversion{std::move(conversions)}; }
        static auto SelectOverload(u32 index) -> Conversion { return Conversion{Kind::SelectOverload, index}; }
        static auto SliceFromArray() -> Conversion { return Conversion{Kind::SliceFromArray}; }
        static auto SliceFromPtrAndSize(SliceFromPtrAndSizeData data) -> Conversion { return Conversion{std::move(data)}; }
        static auto StripParens() -> Conversion { return Conversion{Kind::StripParens}; }
        static auto StrLitToCStr() -> Conversion { return Conversion{Kind::StrLitToCStr}; }
        static auto TupleToFirstElement() -> Conversion { return Conversion{Kind::TupleToFirstElement}; }

        Type type() const { return data.get<TypeAndValueCategory>().type(); }
        auto value_category() const -> ValueCategory {
            return data.get<TypeAndValueCategory>().value_category();
        }
    };

    class ConversionSequence {
        LIBBASE_MOVE_ONLY(ConversionSequence);

    public:
        SmallVector<Conversion, 1> conversions;
        ConversionSequence() = default;
        void add(Conversion conv) { conversions.push_back(std::move(conv)); }
        u32 badness();
    };

    class TentativeConversionContext {
        LIBBASE_IMMOVABLE(TentativeConversionContext);
        ConversionSequence& seq;
        const usz num_conversions;
        bool committed = false, rolled_back = false;

    public:
        TentativeConversionContext(ConversionSequence& seq)
            : seq{seq}, num_conversions{seq.conversions.size()} {}

        ~TentativeConversionContext() {
            Assert(
                committed or rolled_back,
                "Must call either commit() or rollback()"
            );
        }

        void commit() { committed = true; }
        void rollback() {
            rolled_back = true;
            seq.conversions.truncate(num_conversions);
        }
    };

    using DiagsVector = SmallVector<Diagnostic, 2>;

    template <typename T>
    struct ValueOrDiagsVector : std::expected<T, DiagsVector> {
        using Base = std::expected<T, DiagsVector>;
        using Base::Base;
        ValueOrDiagsVector(DiagsVector d) : Base(std::unexpected(std::move(d))) {}
        ValueOrDiagsVector(Diagnostic d) : Base(std::unexpected(DiagsVector{std::move(d)})) {}
    };

    /// Either an empty result or a list of diagnostics.
    using MaybeDiags = ValueOrDiagsVector<void>;

    /// Result of building a conversion sequence.
    using ConversionSequenceOrDiags = ValueOrDiagsVector<ConversionSequence>;

    /// The result of name lookup.
    struct LookupResult {
        enum struct Reason : u8 {
            /// Lookup was successful.
            Success,

            /// Lookup was ambiguous. This need not be an error if we’re
            /// looking up e.g. function overloads.
            Ambiguous,

            /// Lookup references a declaration that could not be exported;
            /// this is a hard error.
            FailedToImport,

            /// The name was not found.
            NotFound,

            /// One of the path segments did not name a scope.
            NonScopeInPath,
        };

        /// The decl(s) that were found, if any. May be completely empty
        /// if lookup failed.
        llvm::TinyPtrVector<Decl*> decls;

        /// The name we failed to look up, if any. Will be unset
        /// if the lookup was successful.
        DeclName name;

        /// Reason for failure.
        Reason result;

        LookupResult(DeclName name) : name{name}, result{Reason::NotFound} {}
        LookupResult(ArrayRef<Decl*> decls, DeclName name, Reason result) : decls{decls}, name{name}, result{result} {}
        LookupResult(Decl* decl, DeclName name, Reason result) : name{name}, result{result} {
            if (decl) decls.push_back(decl);
        }

        /// Check if this lookup result is a success.
        [[nodiscard]] auto successful() const -> bool { return result == Reason::Success; }
        [[nodiscard]] explicit operator bool() const { return successful(); }

        static auto Ambiguous(DeclName name, ArrayRef<Decl*> decls) {
            Assert(not decls.empty());
            return LookupResult{decls, name, Reason::Ambiguous};
        }

        static auto FailedToImport(DeclName name) { return LookupResult{{}, name, Reason::FailedToImport}; }
        static auto NonScopeInPath(DeclName name, Decl* decl = nullptr) { return LookupResult{decl, name, Reason::NonScopeInPath}; }
        static auto NotFound(DeclName name) { return LookupResult{name}; }
        static auto Success(Decl* decl) {
            Assert(decl);
            return LookupResult{decl, {}, Reason::Success};
        }
    };

    /// A successful template substitution.
    struct TemplateSubstitution : public FoldingSetNode {
        ALLOCATE_IN_TU(TemplateSubstitution);
        FoldingSetNodeIDRef hash;

        /// The substituted procedure type.
        ProcType* type;

        /// Procedure scope to use for instantiation.
        Scope* scope;

        /// The instantiated procedure, if we have already instantiated it.
        ProcDecl* instantiation = nullptr;

        /// Whether the constraint for this substitution is satisfied.
        bool constraint_satisfied = true;

        TemplateSubstitution(
            FoldingSetNodeIDRef hash,
            ProcType* type,
            Scope* scope
        ) : hash{hash}, type{type}, scope{scope} {}

        void Profile(FoldingSetNodeID& id) { id = hash; }
    };

    /// The result of substituting a procedure type before template instantiation.
    class SubstitutionResult {
        LIBBASE_MOVE_ONLY(SubstitutionResult);

    public:
        /// Deduction failed entirely for a TDD.
        struct DeductionFailed {
            /// The parameter that we failed to deduce.
            String param;

            /// The index of the parameter in which we tried to
            /// perform deduction.
            u32 param_index;
        };

        /// The type of a TDD decl was deduced to be two different types
        /// in two different parameters.
        struct DeductionAmbiguous {
            /// The name of the parameter that we failed to deduce.
            String param;

            /// The two arguments that caused the ambiguity.
            u32 first, second;

            /// The two types that we deduced each time.
            Type first_type, second_type;
        };

        /// The constraint on the template was not satisfied.
        struct ConstraintNotSatisfied {};

        /// There was a hard error which has already been reported.
        struct Error {};

        /// What happened.
        using DataType = Variant< // clang-format off
            TemplateSubstitution*,
            DeductionFailed,
            DeductionAmbiguous,
            ConstraintNotSatisfied,
            Error
        >;

        DataType data = Error{}; // clang-format on

        /// Create an invalid substitution.
        SubstitutionResult() = default;
        SubstitutionResult(std::nullptr_t) {};

        /// Create a successful substitution.
        SubstitutionResult(TemplateSubstitution* subst) : data{subst} {
            Assert(subst);
            Assert(subst->constraint_satisfied);
        }

        /// Create an invalid substitution.
        SubstitutionResult(DeductionFailed failed) : data{failed} {}
        SubstitutionResult(DeductionAmbiguous ambiguous) : data{ambiguous} {}
        SubstitutionResult(ConstraintNotSatisfied not_satisfied) : data{not_satisfied} {}

        /// Check if this was successful and return the substitution if so.
        auto success() const -> TemplateSubstitution* {
            auto s = data.get_if<TemplateSubstitution*>();
            return s ? *s : nullptr;
        }
    };

    /// Overload resolution candidate.
    struct Candidate {
        LIBBASE_MOVE_ONLY(Candidate);

    public:
        // Viable candidate.
        struct Viable {
            // The conversion sequences that need to be applied to each
            // argument if this overload does get selected.
            SmallVector<ConversionSequence, 4> conversions;

            // How 'bad' is this overload, i.e. how many conversions are
            // required to make it work.
            u32 badness = 0;
        };

        // Candidate for which the argument count didn’t match the parameter
        // count. This is also used if we don’t have enough template parameters,
        // in which case NO SubstitutionInfo object will be created.
        struct ArgumentCountMismatch {};

        // There was an error during deduction, but we do have a SubstitutionInfo object.
        struct DeductionError {};

        // We failed to initialise a parameter.
        struct ParamInitFailed {
            DiagsVector diags;
            u32 param_index;
        };

        // The procedure (template) that this candidate represents.
        Decl* decl;

        // Substitution for this template if this is one. This may not
        // exist in some error cases.
        SubstitutionResult subst{};

        // Whether this candidate is still viable, or why not.
        using Status = Variant< // clang-format off
            Viable,
            ArgumentCountMismatch,
            ParamInitFailed,
            DeductionError
        >; // clang-format on
        Status status = Viable{};

        Candidate(Decl* p) : decl{p} {
            Assert((isa<ProcDecl, ProcTemplateDecl>(p)));
        }

        auto badness() const -> u32 { return status.get<Viable>().badness; }
        auto param_count() const -> usz;
        auto param_loc(usz index) const -> SLoc;
        auto proc_type() const -> ProcType*;
        auto type_for_diagnostic() const -> SmallUnrenderedString;
        bool has_c_varargs() const;
        bool has_valid_proc_type() const;
        bool is_template() const { return isa<ProcTemplateDecl>(decl); }
        bool is_variadic() const;
        auto non_variadic_params() const -> u32;
        bool viable() const { return status.is<Viable>(); }
    };

    /// Pattern matching.
    class MatchContext {
        LIBBASE_IMMOVABLE(MatchContext);

    protected:
        Sema& S;
        MatchContext(Sema& s) : S{s} {}

    public:
        virtual ~MatchContext() = default;

        struct AddResult {
            enum struct Kind {
                Ok,
                Exhaustive,  ///< This pattern makes the match exhaustive.
                InvalidType, ///< We don’t know what to do w/ this (e.g. if someone passes "foo" to an integer match).
                Subsumed,    ///< Subsumed by an earlier pattern.
            } kind{};
            ArrayRef<SLoc> locations{};
        };

        [[nodiscard]] virtual auto add_constant_pattern(const eval::RValue& pattern, SLoc loc) -> AddResult = 0;
        [[nodiscard]] virtual auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr> = 0;
        [[nodiscard]] virtual auto preprocess(Expr* pattern) -> Ptr<Expr> { return pattern; }
        virtual void note_missing(SLoc match_loc) = 0;

        static auto Ok() -> AddResult { return {AddResult::Kind::Ok}; }
        static auto Exhaustive(bool exhaustive = true) -> AddResult { return {exhaustive ? AddResult::Kind::Exhaustive : AddResult::Kind::Ok}; }
        static auto InvalidType() -> AddResult { return {AddResult::Kind::InvalidType}; }
        static auto Subsumed(ArrayRef<SLoc> locations) -> AddResult {
            return {AddResult::Kind::Subsumed, locations};
        }
    };

    class BoolMatchContext : public MatchContext {
        SLoc true_loc;
        SLoc false_loc;

    public:
        BoolMatchContext(Sema& s) : MatchContext{s} {}
        [[nodiscard]] auto add_constant_pattern(const eval::RValue& pattern, SLoc loc) -> AddResult override;
        [[nodiscard]] auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr> override;
        void note_missing(SLoc match_loc) override;
    };

    class IntMatchContext : public MatchContext {
        /// Range of values that is satisfied.
        struct Range {
            APInt start;
            APInt end; // Inclusive.
            SmallVector<SLoc> locations;

            Range(APInt start, APInt end, SLoc loc)
                : start{std::move(start)}, end{std::move(end)}, locations{loc} {
                Assert(this->start.sle(this->end));
            }

            /// Check if this range is adjacent to another range.
            [[nodiscard]] bool adjacent(const Range& r) {
                APInt one{start.getBitWidth(), 1};
                return r.start.ssub_sat(one).sle(end) and r.end.sadd_sat(one).sge(start);
            }

            /// Merge another range into this one.
            void merge(const Range& r) {
                start = llvm::APIntOps::smin(start, r.start);
                end = llvm::APIntOps::smax(end, r.end);
                locations.append(r.locations);
            }

            /// Check if this range overlaps another range.
            [[nodiscard]] bool overlaps(const Range& r) {
                return r.start.sle(end) and r.end.sge(start);
            }

            /// Check if this range fully includes another range.
            bool subsumes(const Range& r) {
                return r.start.sge(start) and r.end.sle(end);
            }
        };

        SmallVector<Range> ranges;
        APInt max;
        APInt min;
        Type ty;

    public:
        IntMatchContext(Sema& s, Type ty);
        [[nodiscard]] auto add_constant_pattern(const eval::RValue& pattern, SLoc loc) -> AddResult override;
        [[nodiscard]] auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr> override;
        [[nodiscard]] auto preprocess(Expr* pattern) -> Ptr<Expr> override;
        void note_missing(SLoc match_loc) override;

    private:
        [[nodiscard]] auto add_range(Range r) -> AddResult;
    };

    class OptionalMatchContext : public MatchContext {
        [[maybe_unused]] OptionalType* optional;
        SLoc nil_loc;
        std::unique_ptr<MatchContext> inner;
        bool inner_exhaustive = false;

    public:
        OptionalMatchContext(Sema& s, OptionalType* optional, std::unique_ptr<MatchContext> inner);
        [[nodiscard]] auto add_constant_pattern(const eval::RValue& pattern, SLoc loc) -> AddResult override;
        [[nodiscard]] auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr> override;
        [[nodiscard]] auto preprocess(Expr* pattern) -> Ptr<Expr> override;
        void note_missing(SLoc match_loc) override;
    };

    /// Hint as to what lookup is trying to find.
    ///
    /// Note that no guarantee is made that successful lookup actually
    /// found the thing we want to find. E.g. if 'Type' is passed, a
    /// ‘successful’ lookup may still have found a function.
    ///
    /// This is mainly intended as a disambiguation hint for scopes and
    /// C++ names as there may be declarations of types and functions with
    /// the same name in C++ code (e.g. 'stat').
    enum struct LookupHint {
        Any,   ///< Any kind of name.
        Scope, ///< Something that can be the LHS of '::'.
        Type,  ///< We’re looking for a type.
    };

    /// Context that indicates where an injection is being performed; depending
    /// on it, we maye allow injecting multiple statements or different kinds
    /// of statements.
    using InjectionContext = llvm::PointerUnion<
        Stmt**,                 // Inject exactly 1 statement.
        SmallVectorImpl<Stmt*>* // Inject any number of statements.
    >;

    Context& ctx;
    TranslationUnit::Ptr tu;
    ArrayRef<std::string> search_paths;
    ArrayRef<std::string> clang_include_paths;
    usz assert_stringifiers = 0;
    usz generated_cxx_macro_decls = 0;

    /// Modules that need to be translated.
    SmallVector<ParsedModule::Ptr> modules_to_translate;

    /// Additional modules that were generated during translation (e.g. for
    /// the preamble or '#inject'ions).
    SmallVector<ParsedModule::Ptr> extra_modules;

    /// Stack of active procedures.
    SmallVector<ProcScopeInfo*> proc_stack;

    /// Stack of active scopes.
    SmallVector<Scope*> scope_stack;

    /// Template deduction information for each template.
    DenseMap<ParsedProcDecl*, TemplateParamDeductionInfo> parsed_template_deduction_infos;

    /// C++ decls that have already been imported (or that
    /// already failed to import before).
    DenseMap<clang::Decl*, Ptr<Decl>> imported_decls;

    /// C++ records that have already been imported (or that
    /// already failed to import before).
    DenseMap<clang::RecordDecl*, std::optional<Type>> imported_records;

    /// C++ macros that have already been imported (or that
    /// already failed to import before).
    DenseMap<clang::MacroInfo*, Ptr<Decl>> imported_macros;

    /// C++ TUs that we own.
    SmallVector<std::unique_ptr<clang::ASTUnit>> clang_ast_units;

    /// Cached template substitutions.
    DenseMap<ProcTemplateDecl*, FoldingSet<TemplateSubstitution>> template_substitutions;

    /// Map from instantiations to their substitutions.
    DenseMap<ProcDecl*, usz> template_substitution_indices;

    /// Next mangling number to use for procedures.
    ManglingNumber next_proc_mangling_number = ManglingNumber(1);

    /// Whether were are inside of an eval.
    bool inside_eval = false;

    /// Whether translation has started.
    bool started_translating = false;

    /// We disallow passing zero-sized structs to native procedures (or returning
    /// them from them), because C doesn’t really have zero-sized types; however,
    /// we might see a declaration of such a procedure before the type in question
    /// is complete, at which point we don’t know its size yet, for such types, we
    /// instead diagnose this at end of translation.
    struct DeferredNativeProcArgOrReturn {
        StructType* type;
        SLoc loc;
        bool is_return;
    };

    SmallVector<DeferredNativeProcArgOrReturn> incomplete_structs_in_native_proc_type;

    /// Struct declarations that are yet to be completed.
    DenseMap<StructType*, ParsedStructDecl*> pending_struct_definitions;

    /// Stack of struct definitions we’re currently completing; this is used to report cycles.
    SmallVector<StructType*> struct_translation_stack;

    Sema(Context& ctx);
    ~Sema();

public:
    /// Analyse a set of parsed modules and combine them into a single module.
    ///
    /// @return The combined module, or `nullptr` if there was an error.
    [[nodiscard]] static auto Translate(
        const LangOpts& opts,
        SmallVector<ParsedModule::Ptr> modules,
        ArrayRef<std::string> module_search_paths,
        ArrayRef<std::string> clang_include_paths
    ) -> TranslationUnit::Ptr;

    /// Get the context.
    auto context() const -> Context& { return ctx; }

    /// Get the diagnostics engine.
    auto diags() const -> DiagnosticsEngine& { return ctx.diags(); }

    /// Get the current procedure.
    auto curr_proc() -> ProcScopeInfo& {
        Assert(not proc_stack.empty(), "Procedure stack underflow");
        return *proc_stack.back();
    }

    /// Get the current scope.
    auto curr_scope() -> Scope* { return scope_stack.back(); }

    /// Get the global scope.
    auto global_scope() -> Scope* { return tu->global_scope(); }

private:
    /// Add a declaration to a scope.
    ///
    /// This correctly handles redeclarations and declarations
    /// with an empty name.
    void AddDeclToScope(Scope* scope, Decl* d);

    /// Add an object to the with stack.
    void AddEntryToWithStack(Scope* scope, LocalDecl* object, SLoc with);

    /// Add an initialiser to a variable declaration.
    void AddInitialiserToDecl(LocalDecl* d, Ptr<Expr> init);

    /// Called after a parsed module is added. Returns the same module
    /// for convenience. If 'translate' is true, the module’s contents
    /// will be appended to the end of the TU.
    auto AddParsedModule(ParsedModule::Ptr p, bool translate = false) -> ParsedModule*;

    /// Apply a conversion to an expression or list of expressions.
    void ApplyConversion(SmallVectorImpl<Expr*>& exprs, const Conversion& conv, SLoc loc);
    [[nodiscard]] auto ApplySimpleConversion(Expr* arg, const Conversion& conv, SLoc loc) -> Expr*;

    /// Apply a conversion sequence to an expression.
    [[nodiscard]] auto ApplyConversionSequence(
        ArrayRef<Expr*> exprs,
        const ConversionSequence& seq,
        SLoc loc
    ) -> Expr*;

    /// Build an initialiser for an aggregate type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildAggregateInitialiser(
        ConversionSequence& seq,
        RecordType* s,
        ArrayRef<Expr*> args,
        SLoc loc
    ) -> MaybeDiags;

    /// Build an initialiser for an array type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildArrayInitialiser(
        ConversionSequence& seq,
        ArrayType* a,
        ArrayRef<Expr*> args,
        SLoc loc
    ) -> MaybeDiags;

    /// Build an initialiser for a slice type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildSliceInitialiser(
        ConversionSequence& seq,
        SliceType* a,
        ArrayRef<Expr*> args,
        SLoc loc
    ) -> MaybeDiags;

    /// Build a conversion sequence that can be applied to a list
    /// of arguments to create an expression that can initialise
    /// a variable of type \p var_type.
    ///
    /// The value is *usually* an rvalue, but it may be an lvalue
    /// if we’re e.g. copying structs since that will be lowered
    /// to a memcpy.
    ///
    /// During parameter initialisation, pass the parameter intent
    /// if there is one; in all other cases, leave it as 'Move'
    /// because that is the default for assignment.
    ///
    /// The expression returned from this is suitable for initialising
    /// a variable of the given type, i.e. codegen knows what to do with
    /// it as the argument to PerformVariableInitialisation().
    ///
    /// \param var_type The type of the variable to be initialised.
    /// \param args The initialisation arguments; can be empty for default
    ///    initialisation.
    ///
    /// \param init_loc Location to report diagnostics at.
    /// \param want_lvalue If do, try to produce an lvalue.
    ///
    /// \return Whether initialisation was successful.
    auto BuildConversionSequence(
        Type var_type,
        ArrayRef<Expr*> args,
        SLoc init_loc,
        bool want_lvalue = false
    ) -> ConversionSequenceOrDiags;

    /// Overload of BuildInitialiser() that builds the initialiser immediately.
    auto BuildInitialiser(
        Type var_type,
        ArrayRef<Expr*> args,
        SLoc loc,
        bool want_lvalue = false
    ) -> Ptr<Expr>;

    /// Check that a type is valid for a record field.
    [[nodiscard]] bool CheckFieldType(Type ty, SLoc loc);

    /// Check additional constraints on a call that need to happen after overload resolution.
    bool CheckIntents(ProcType* ty, ArrayRef<Expr*> args);

    /// Check that a collection of patterns is exhaustive, and return 'true' if so.
    bool CheckMatchExhaustive(
        SLoc match_loc,
        Expr* control_expr,
        Type ty,
        MutableArrayRef<MatchCase> cases
    );

    bool CheckMatchExhaustiveImpl(
        MatchContext& ctx,
        SLoc match_loc,
        Expr* control_expr,
        Type ty,
        MutableArrayRef<MatchCase> cases
    );

    /// Check if the declaration of an overloaded operator is well-formed.
    bool CheckOverloadedOperator(ProcDecl* d, bool builtin_operator);

    /// Check that a type is valid for a variable declaration.
    [[nodiscard]] bool CheckVariableType(Type ty, SLoc loc);

    /// Attempt to complete a struct type.
    bool CompleteDefinition(StructType* ty);

    /// Determine the common type and value category of a set of expressions and,
    /// if there is one, ensure they all have the same type and value category.
    auto ComputeCommonTypeAndValueCategory(MutableArrayRef<Expr*> exprs) -> TypeAndValueCategory;

    /// Create a reference to a declaration.
    [[nodiscard]] auto CreateReference(Decl* d, SLoc loc) -> Ptr<Expr>;

    /// Add a variable to the current scope and procedure.
    void DeclareLocal(LocalDecl* d);

    /// Perform template deduction.
    auto DeduceType(ParsedStmt* parsed_type, Type input_type) -> Type;

    /// Diagnose that we’re using a zero-sized type in a native procedure signature.
    void DiagnoseZeroSizedTypeInNativeProc(Type ty, SLoc use, bool is_return);

    /// Evaluate a statement.
    auto Evaluate(Stmt* e, bool complain = true) -> std::optional<eval::RValue>;

    /// Evaluate a statement, returning an expression that caches the result on success
    /// and nullptr on failure. The returned expression need not be a ConstExpr.
    auto EvaluateIntoExpr(Stmt* e, SLoc loc) -> Ptr<Expr>;

    /// Extract the scope that is the body of a declaration, if it has one.
    auto GetScopeFromDecl(Decl* d) -> Ptr<Scope>;

    /// Import a declaration from a C++ AST.
    auto ImportCXXDecl(ImportedClangModuleDecl* clang_module, CXXDecl* decl) -> Ptr<Decl>;

    /// Import a C++ header.
    auto ImportCXXHeaders(
        String logical_name,
        ArrayRef<String> header_names,
        SLoc import_loc
    ) -> Ptr<ImportedClangModuleDecl>;

    /// Inject a parse tree into the program.
    bool InjectTree(Expr* injected, Type desired_type, InjectionContext ctx);

    /// Instantiate a procedure template.
    auto InstantiateTemplate(
        ProcTemplateDecl* pattern,
        TemplateSubstitution& info,
        SLoc inst_loc
    ) -> ProcDecl*;

    /// Check if an integer literal can be stored in a given type.
    bool IntegerLiteralFitsInType(const APInt& i, Type ty, bool negated);

    /// Check that we have a complete type.
    [[nodiscard]] bool RequireCompleteType(Type ty);

    /// Check if an operator that takes a sequence of argument types must be overloaded.
    bool IsUserDefinedOverloadedOperator(Tk op, ArrayRef<Type> argument_types);

    /// Check whether this parsed type is the builtin 'var' type.
    bool IsBuiltinVarType(ParsedStmt* stmt);

    /// Check if a type is zero-sized or incomplete.
    bool IsZeroSizedOrIncomplete(Type ty);

    /// Load a native header or Source module from the system include path.
    void LoadModule(
        String logical_name,
        ArrayRef<String> linkage_names,
        SLoc import_loc,
        bool is_open,
        bool is_cxx_header
    );

    /// Load a Source module.
    auto LoadModuleFromArchive(
        String logical_name,
        String linkage_name,
        SLoc import_loc
    ) -> Ptr<ImportedSourceModuleDecl>;

    /// Use LookUpName() instead.
    auto LookUpCXXName(
        ImportedClangModuleDecl* clang_module,
        ArrayRef<DeclName> names,
        LookupHint hint
    ) -> LookupResult;

    auto LookUpCXXNameImpl(
        ImportedClangModuleDecl* clang_module,
        ArrayRef<DeclName> names,
        LookupHint hint
    ) -> LookupResult;

    /// Use LookUpName() instead.
    auto LookUpQualifiedName(
        Scope* in_scope,
        ArrayRef<DeclName> names,
        LookupHint hint
    ) -> LookupResult;

    /// Perform unqualified name lookup.
    auto LookUpUnqualifiedName(
        Scope* in_scope,
        DeclName name,
        LookupHint hint,
        bool this_scope_only
    ) -> LookupResult;

    /// Look up a name in a scope.
    ///
    /// Name lookup differs between unqualified and qualified names: for
    /// unqualified names, we look up the name in the scope it was encountered
    /// in, and all of its parent scope.
    ///
    /// For qualified name lookup, we start by performing unqualified lookup
    /// for the first name in the path, except that names of imported modules
    /// are also considered if all else fails. The remaining path segments are
    /// then looked up in the scope of the declaration found by the previous
    /// segment only.
    ///
    /// \param in_scope The scope to start searching in.
    /// \param names The path to look up.
    /// \param loc The location of the lookup.
    /// \param hint Hint as to what we’re looking for.
    /// \param complain Emit a diagnostic if lookup fails.
    auto LookUpName(
        Scope* in_scope,
        ArrayRef<DeclName> names,
        SLoc loc,
        LookupHint hint = LookupHint::Any,
        bool complain = true
    ) -> LookupResult;

    /// Convert an lvalue to an srvalue.
    [[nodiscard]] auto LValueToRValue(Expr* expr) -> Expr*;

    /// Ensure that an expression is a condition (e.g. for 'if', 'assert', etc.)
    [[nodiscard]] bool MakeCondition(Expr*& e, StringRef op);

    /// Wrap the result of constant evaluation in a ConstExpr for caching, if doing
    /// so is beneficial (e.g. we don’t wrap integer literals).
    [[nodiscard]] auto MakeConstExpr(
        Stmt* evaluated_stmt,
        eval::RValue val,
        SLoc loc
    ) -> Expr*;

    /// Create a local variable and add it to the current scope and procedure.
    [[nodiscard]] auto MakeLocal(
        Type ty,
        ValueCategory vc,
        String name,
        SLoc loc
    ) -> LocalDecl*;

    /// Ensure that an expression is an rvalue of the given type.
    template <typename Callback>
    [[nodiscard]] bool MakeRValue(Type ty, Expr*& e, Callback EmitDiag);

    /// Ensure that an expression is an rvalue of the given type. This is
    /// mainly used for expressions involving operators.
    [[nodiscard]] bool MakeRValue(Type ty, Expr*& e, StringRef elem_name, StringRef op);

    /// Materialise a temporary value.
    [[nodiscard]] auto MaterialiseTemporary(Expr* expr) -> Expr*;

    /// Materialise a temporary value and create a variable to store it;
    /// returns a reference to the variable. If 'expr' already is a variable,
    /// no new variable is created.
    [[nodiscard]] auto MaterialiseVariable(Expr* expr) -> LocalDecl*;

    /// Mark that a number of match cases are unreachable.
    ///
    /// \param it The last case that can still be matched.
    /// \param cases The *entire* list of match cases.
    void MarkUnreachableAfter(auto it, MutableArrayRef<MatchCase> cases);

    /// Parse C++ code into a new TU.
    auto ParseCXX(StringRef code) -> std::unique_ptr<clang::ASTUnit>;

    /// Resolve an overload set.
    auto PerformOverloadResolution(
        OverloadSetExpr* overload_set,
        ArrayRef<Expr*> args,
        SLoc call_loc
    ) -> std::pair<ProcDecl*, SmallVector<Expr*>>;

    /// Deserialise an AST.
    auto ReadAST(ImportedSourceModuleDecl* module_decl, const File& f) -> Result<>;

    /// Issue an error about lookup failure.
    void ReportLookupFailure(const LookupResult& result, SLoc loc);

    /// Issue an error about overload resolution failure.
    void ReportOverloadResolutionFailure(
        MutableArrayRef<Candidate> candidates,
        ArrayRef<Expr*> call_args,
        SLoc call_loc,
        u32 final_badness
    );

    /// Whether a procedure requires a mangling number.
    bool RequiresManglingNumber(const ParsedProcAttrs& attrs);

    /// Substitute types in a procedure template.
    auto SubstituteTemplate(
        ProcTemplateDecl* proc_template,
        ArrayRef<TypeLoc> input_types
    ) -> SubstitutionResult;

    /// Try to perform variable initialisation and return the result if
    /// if it succeeds, and nullptr on failure, but don’t emit a diagnostic
    /// either way.
    auto TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr>;

    /// Unwrap an optional value.
    auto UnwrapOptional(Expr* opt, SLoc loc) -> Expr*;

    /// Unwrap (multi-level) pointers and optionals.
    auto UnwrapPointersAndOptionals(Expr* e) -> Ptr<Expr>;

    /// Building AST nodes.
    auto BuildAssertExpr(Expr* cond, Ptr<Expr> msg, bool is_compile_time, SLoc loc, SRange cond_range) -> Ptr<Expr>;
    auto BuildArrayType(TypeLoc base, Expr* size) -> Type;
    auto BuildArrayType(TypeLoc base, i64 size, SLoc loc) -> Type;
    auto BuildBinaryExpr(Tk op, Expr* lhs, Expr* rhs, SLoc loc) -> Ptr<Expr>;
    auto BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, SLoc loc) -> BlockExpr*;
    auto BuildBuiltinCallExpr(BuiltinCallExpr::Builtin builtin, ArrayRef<Expr*> args, SLoc call_loc) -> Ptr<BuiltinCallExpr>;
    auto BuildBuiltinMemberAccessExpr(BuiltinMemberAccessExpr::AccessKind ak, Expr* operand, SLoc loc) -> Ptr<BuiltinMemberAccessExpr>;
    auto BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, SLoc loc) -> Ptr<Expr>;
    auto BuildCompleteStructType(String name, RecordLayout* layout, SLoc decl_loc) -> StructType*;
    auto BuildDeclRefExpr(ArrayRef<DeclName> names, SLoc loc, Type desired_type = {}) -> Ptr<Expr>;
    auto BuildEvalExpr(Stmt* arg, SLoc loc) -> Ptr<Expr>;
    auto BuildExplicitCast(Type to, Expr* arg, SLoc loc, bool is_hard_cast) -> Ptr<Expr>;
    auto BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, SLoc loc) -> Ptr<IfExpr>;
    auto BuildMatchExpr(Ptr<Expr> control_expr, Type ty, MutableArrayRef<MatchCase> cases, SLoc loc) -> Ptr<Expr>;
    auto BuildMemberAccessExpr(Expr* base, FieldDecl* field, SLoc loc) -> Ptr<Expr>;
    auto BuildParamDecl(ProcScopeInfo& proc, const ParamTypeData* param, u32 index, bool with_param, String name, SLoc loc) -> ParamDecl*;
    auto BuildProcDeclInitial(
        Scope* proc_scope,
        ProcType* ty,
        DeclName name,
        SLoc loc,
        ParsedProcAttrs attrs,
        InheritedProcedureProperties props,
        ProcTemplateDecl* pattern = nullptr
    ) -> ProcDecl*;

    auto BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr>;
    auto BuildReturnExpr(Ptr<Expr> value, SLoc loc, bool implicit) -> ReturnExpr*;
    auto BuildSliceType(Type base, SLoc loc) -> Type;
    auto BuildStaticIfExpr(Expr* cond, ParsedStmt* then, Ptr<ParsedStmt> else_, SLoc loc) -> Ptr<Stmt>;
    auto BuildTupleType(ArrayRef<TypeLoc> types) -> Type;
    auto BuildTypeExpr(Type ty, SLoc loc) -> TypeExpr*;
    auto BuildUnaryExpr(Tk op, Expr* operand, bool postfix, SLoc loc) -> Ptr<Expr>;

    /// Entry point.
    void Translate(bool load_runtime);

    /// Statements.
#define PARSE_TREE_LEAF_EXPR(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Ptr<Stmt>;
#define PARSE_TREE_LEAF_DECL(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Decl*;
#define PARSE_TREE_LEAF_STMT(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Ptr<Stmt>;
#define PARSE_TREE_LEAF_TYPE(Name)
#include "srcc/ParseTree.inc"

    auto TranslateExpr(ParsedStmt* parsed, Type desired_type = Type()) -> Ptr<Expr>;

    /// Declarations.
    auto TranslateEntireDecl(Decl* decl, ParsedDecl* parsed) -> Ptr<Decl>;
    auto TranslateEnumDeclInitial(ParsedEnumDecl* parsed) -> Ptr<TypeDecl>;
    void TranslateEnumerators(EnumType* e, ParsedEnumDecl* parsed);
    auto TranslateDeclInitial(ParsedDecl* parsed) -> std::optional<Ptr<Decl>>;
    auto TranslateProc(ProcDecl* decl, Ptr<ParsedStmt> body, ArrayRef<ParsedVarDecl*> decls) -> ProcDecl*;
    auto TranslateProcBody(ProcDecl* decl, ParsedStmt* body, ArrayRef<ParsedVarDecl*> decls) -> Ptr<Stmt>;
    auto TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<Decl>;
    auto TranslateStmt(ParsedStmt* parsed, Type desired_type = Type()) -> Ptr<Stmt>;
    bool TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed, Type desired_type = Type());
    auto TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl>;

    /// Types.
    auto TranslateArrayType(ParsedBinaryExpr* parsed) -> Type;
    auto TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type;
    auto TranslateIntType(ParsedIntType* parsed) -> Type;
    auto TranslateNamedType(ParsedDeclRefExpr* parsed) -> Type;
    auto TranslateRangeType(ParsedRangeType* parsed) -> Type;
    auto TranslateSliceType(ParsedSliceType* parsed) -> Type;
    auto TranslateTemplateType(ParsedTemplateType* parsed) -> Type;
    auto TranslateType(ParsedStmt* stmt, Type fallback = Type()) -> Type;
    auto TranslateTypeofType(ParsedTypeofType* parsed) -> Type;
    auto TranslateOptionalType(ParsedOptionalType* stmt) -> Type;
    auto TranslatePtrType(ParsedPtrType* stmt) -> Type;
    auto TranslateProcType(ParsedProcType* parsed, ArrayRef<Type> deduced_var_parameters = {}) -> Type;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, SLoc where, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }
};

#endif // SRCC_FRONTEND_SEMA_HH
