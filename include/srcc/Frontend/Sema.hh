#ifndef SRCC_FRONTEND_SEMA_HH
#define SRCC_FRONTEND_SEMA_HH

#include <srcc/AST/AST.hh>
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

    friend DiagsProducer;
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

        /// Pop the scope if it is still active.
        ~EnterScope() {
            if (scope) S.scope_stack.pop_back();
        }

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

        enum struct Kind : u8 {
            ArrayBroadcast,
            ArrayInit,
            ExpandTuple,
            DefaultInit,
            IntegralCast,
            LValueToRValue,
            MaterialisePoison,
            MaterialiseTemporary,
            RangeCast,
            SelectOverload,
            SliceFromArray,
            StripParens,
            StrLitToCStr,
            TupleToFirstElement,
            RecordInit,
        };

        Kind kind;
        Variant< // clang-format off
            TypeAndValueCategory,
            RecordInitData,
            ArrayInitData,
            ArrayBroadcastData,
            u32
        > data; // clang-format on

    private:
        Conversion(Kind kind) : kind{kind} {}
        Conversion(Kind kind, Type ty, ValueCategory val = Expr::RValue) : kind{kind}, data{TypeAndValueCategory(ty, val)} {}
        Conversion(Kind kind, u32 index) : kind{kind}, data{index} {}
        Conversion(RecordInitData conversions) : kind{Kind::RecordInit}, data{std::move(conversions)} {}
        Conversion(ArrayInitData data) : kind{Kind::ArrayInit}, data{std::move(data)} {}
        Conversion(ArrayBroadcastData data) : kind{Kind::ArrayBroadcast}, data{std::move(data)} {}

    public:
        ~Conversion();
        static auto ArrayBroadcast(ArrayBroadcastData data) -> Conversion { return Conversion{std::move(data)}; }
        static auto ArrayInit(ArrayInitData data) -> Conversion { return Conversion{std::move(data)}; }
        static auto DefaultInit(Type ty) -> Conversion { return Conversion{Kind::DefaultInit, ty}; }
        static auto ExpandTuple() -> Conversion { return Conversion{Kind::ExpandTuple}; }
        static auto IntegralCast(Type ty) -> Conversion { return Conversion{Kind::IntegralCast, ty}; }
        static auto LValueToRValue() -> Conversion { return Conversion{Kind::LValueToRValue}; }
        static auto MaterialiseTemporary() -> Conversion { return Conversion{Kind::MaterialiseTemporary}; }
        static auto Poison(Type ty, ValueCategory val) -> Conversion { return Conversion{Kind::MaterialisePoison, ty, val}; }
        static auto RangeCast(Type ty) -> Conversion { return Conversion{Kind::RangeCast, ty}; }
        static auto RecordInit(RecordInitData conversions) -> Conversion { return Conversion{std::move(conversions)}; }
        static auto SelectOverload(u32 index) -> Conversion { return Conversion{Kind::SelectOverload, index}; }
        static auto SliceFromArray() -> Conversion { return Conversion{Kind::SliceFromArray}; }
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

        static auto Ambiguous(DeclName name, ArrayRef<Decl*> decls) { return LookupResult{decls, name, Reason::Ambiguous}; }
        static auto FailedToImport() { return LookupResult{{}, {}, Reason::FailedToImport}; }
        static auto NonScopeInPath(DeclName name, Decl* decl = nullptr) { return LookupResult{decl, name, Reason::NonScopeInPath}; }
        static auto NotFound(DeclName name) { return LookupResult{name}; }
        static auto Success(Decl* decl) { return LookupResult{decl, {}, Reason::Success}; }
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

        /// There was a hard error which has already been reported.
        struct Error {};

        /// What happened.
        using DataType = Variant< // clang-format off
            TemplateSubstitution*,
            DeductionFailed,
            DeductionAmbiguous,
            Error
        >;

        DataType data = Error{}; // clang-format on

        SubstitutionResult() = default;

        template <std::convertible_to<DataType> T>
        SubstitutionResult(T&& v): data{std::forward<T>(v)} {}

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
        auto param_loc(usz index) const -> Location;
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
        struct AddResult {
            enum struct Kind {
                Ok,
                Exhaustive,  ///< This pattern makes the match exhaustive.
                InvalidType, ///< We don’t know what to do w/ this (e.g. if someone passes "foo" to an integer match).
                Subsumed,    ///< Subsumed by an earlier pattern.
            } kind{};
            ArrayRef<Location> locations{};
        };

        static auto Ok() -> AddResult { return {AddResult::Kind::Ok}; }
        static auto Exhaustive() -> AddResult { return {AddResult::Kind::Exhaustive}; }
        static auto InvalidType() -> AddResult { return {AddResult::Kind::InvalidType}; }
        static auto Subsumed(ArrayRef<Location> locations) -> AddResult {
            return {AddResult::Kind::Subsumed, locations};
        }
    };

    class BoolMatchContext : public MatchContext {
        Location true_loc;
        Location false_loc;

    public:
        BoolMatchContext(Sema& s) : MatchContext{s} {}
        [[nodiscard]] auto add_constant_pattern(const eval::RValue& pattern, Location loc) -> AddResult;
        [[nodiscard]] auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr>;
        [[nodiscard]] auto preprocess(Expr* pattern) -> Ptr<Expr>;
        void note_missing(Location match_loc);
    };

    class IntMatchContext : public MatchContext {
        /// Range of values that is satisfied.
        struct Range {
            APInt start;
            APInt end; // Inclusive.
            SmallVector<Location> locations;

            Range(APInt start, APInt end, Location loc)
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
        [[nodiscard]] auto add_constant_pattern(const eval::RValue& pattern, Location loc) -> AddResult;
        [[nodiscard]] auto build_comparison(Expr* control_expr, Expr* pattern_expr) -> Ptr<Expr>;
        [[nodiscard]] auto preprocess(Expr* pattern) -> Ptr<Expr>;
        void note_missing(Location match_loc);

    private:
        [[nodiscard]] auto add_range(Range r) -> AddResult;
    };

    Context& ctx;
    TranslationUnit::Ptr tu;
    SmallVector<ParsedModule::Ptr> parsed_modules;
    ArrayRef<std::string> search_paths;
    ArrayRef<std::string> clang_include_paths;

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

    /// C++ TUs that we own.
    SmallVector<std::unique_ptr<clang::ASTUnit>> clang_ast_units;

    /// Cached template substitutions.
    DenseMap<ProcTemplateDecl*, FoldingSet<TemplateSubstitution>> template_substitutions;

    /// Map from instantiations to their substitutions.
    DenseMap<ProcDecl*, usz> template_substitution_indices;

    /// We disallow passing zero-sized structs to native procedures (or returning
    /// them from them), because C doesn’t really have zero-sized types; however,
    /// we might see a declaration of such a procedure before the type in question
    /// is complete, at which point we don’t know its size yet, for such types, we
    /// instead diagnose this at end of translation.
    struct DeferredNativeProcArgOrReturn {
        StructType* type;
        Location loc;
        bool is_return;
    };

    SmallVector<DeferredNativeProcArgOrReturn> incomplete_structs_in_native_proc_type;

    /// Whether we’re currently parsing imported declarations.
    bool importing_module = false;

    Sema(Context& ctx);
    ~Sema();

public:
    /// Analyse a set of parsed modules and combine them into a single module.
    ///
    /// @return The combined module, or `nullptr` if there was an error.
    [[nodiscard]] static auto Translate(
        const LangOpts& opts,
        ParsedModule::Ptr preamble,
        SmallVector<ParsedModule::Ptr> modules,
        ArrayRef<std::string> module_search_paths,
        ArrayRef<std::string> clang_include_paths,
        bool load_runtime
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

    /// Apply a conversion to an expression or list of expressions.
    void ApplyConversion(SmallVectorImpl<Expr*>& exprs, const Conversion& conv, Location loc);
    [[nodiscard]] auto ApplySimpleConversion(Expr* arg, const Conversion& conv, Location loc) -> Expr*;

    /// Apply a conversion sequence to an expression.
    [[nodiscard]] auto ApplyConversionSequence(
        ArrayRef<Expr*> exprs,
        const ConversionSequence& seq,
        Location loc
    ) -> Expr*;

    /// Build an initialiser for an aggregate type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildAggregateInitialiser(
        ConversionSequence& seq,
        RecordType* s,
        ArrayRef<Expr*> args,
        Location loc
    ) -> MaybeDiags;

    /// Build an initialiser for an array type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildArrayInitialiser(
        ConversionSequence& seq,
        ArrayType* a,
        ArrayRef<Expr*> args,
        Location loc
    ) -> MaybeDiags;

    /// Build an initialiser for a slice type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildSliceInitialiser(
        ConversionSequence& seq,
        SliceType* a,
        ArrayRef<Expr*> args,
        Location loc
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
        Location init_loc,
        bool want_lvalue = false
    ) -> ConversionSequenceOrDiags;

    /// Overload of BuildInitialiser() that builds the initialiser immediately.
    auto BuildInitialiser(
        Type var_type,
        ArrayRef<Expr*> args,
        Location loc,
        bool want_lvalue = false
    ) -> Ptr<Expr>;

    /// Check that a type is valid for a record field.
    [[nodiscard]] bool CheckFieldType(Type ty, Location loc);

    /// Check additional constraints on a call that need to happen after overload resolution.
    bool CheckIntents(ProcType* ty, ArrayRef<Expr*> args);

    /// Check that a collection of patterns is exhaustive, and return 'true' if so.
    template <typename MContext>
    bool CheckMatchExhaustive(
        MContext& ctx,
        Location match_loc,
        Expr* control_expr,
        Type ty,
        MutableArrayRef<MatchCase> cases
    );

    /// Check if the declaration of an overloaded operator is well-formed.
    bool CheckOverloadedOperator(ProcDecl* d, bool builtin_operator);

    /// Check that a type is valid for a variable declaration.
    [[nodiscard]] bool CheckVariableType(Type ty, Location loc);

    /// Determine the common type and value category of a set of expressions and,
    /// if there is one, ensure they all have the same type and value category.
    auto ComputeCommonTypeAndValueCategory(MutableArrayRef<Expr*> exprs) -> TypeAndValueCategory;

    /// Create a reference to a declaration.
    [[nodiscard]] auto CreateReference(Decl* d, Location loc) -> Ptr<Expr>;

    /// Add a variable to the current scope and procedure.
    void DeclareLocal(LocalDecl* d);

    /// Perform template deduction.
    auto DeduceType(ParsedStmt* parsed_type, Type input_type) -> Type;

    /// Diagnose that we’re using a zero-sized type in a native procedure signature.
    void DiagnoseZeroSizedTypeInNativeProc(Type ty, Location use, bool is_return);

    /// Evaluate a statement, returning an expression that caches the result on success
    /// and nullptr on failure. The returned expression need not be a ConstExpr.
    auto Evaluate(Stmt* e, Location loc) -> Ptr<Expr>;

    /// Evaluate a statement as an integer or bool value.
    auto EvaluateAsIntOrBool(Stmt* s) -> std::optional<eval::RValue>;

    /// Extract the scope that is the body of a declaration, if it has one.
    auto GetScopeFromDecl(Decl* d) -> Ptr<Scope>;

    /// Import a declaration from a C++ AST.
    auto ImportCXXDecl(clang::ASTUnit& ast, CXXDecl* decl) -> Ptr<Decl>;

    /// Import a C++ header.
    auto ImportCXXHeaders(
        String logical_name,
        ArrayRef<String> header_names,
        Location import_loc
    ) -> Ptr<ImportedClangModuleDecl>;

    /// Instantiate a procedure template.
    auto InstantiateTemplate(
        ProcTemplateDecl* pattern,
        TemplateSubstitution& info,
        Location inst_loc
    ) -> ProcDecl*;

    /// Check if an integer literal can be stored in a given type.
    bool IntegerLiteralFitsInType(const APInt& i, Type ty, bool negated);

    /// Check that we have a complete type.
    [[nodiscard]] bool IsCompleteType(Type ty, bool null_type_is_complete = true);

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
        Location import_loc,
        bool is_open,
        bool is_cxx_header
    );

    /// Load a Source module.
    auto LoadModuleFromArchive(
        String logical_name,
        String linkage_name,
        Location import_loc
    ) -> Ptr<ImportedSourceModuleDecl>;

    /// Use LookUpName() instead.
    auto LookUpCXXName(clang::ASTUnit* ast, ArrayRef<DeclName> names) -> LookupResult;

    /// Use LookUpName() instead.
    auto LookUpQualifiedName(Scope* in_scope, ArrayRef<DeclName> names) -> LookupResult;

    /// Perform unqualified name lookup.
    auto LookUpUnqualifiedName(
        Scope* in_scope,
        DeclName name,
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
    /// \param complain Emit a diagnostic if lookup fails.
    auto LookUpName(
        Scope* in_scope,
        ArrayRef<DeclName> names,
        Location loc,
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
        Location loc
    ) -> Expr*;

    /// Create a local variable and add it to the current scope and procedure.
    [[nodiscard]] auto MakeLocal(
        Type ty,
        ValueCategory vc,
        String name,
        Location loc
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
    [[nodiscard]] auto MaterialiseVariable(Expr* expr) -> Expr*;

    /// Mark that a number of match cases are unreachable.
    ///
    /// \param it The last case that can still be matched.
    /// \param cases The *entire* list of match cases.
    void MarkUnreachableAfter(auto it, MutableArrayRef<MatchCase> cases);

    /// Resolve an overload set.
    auto PerformOverloadResolution(
        OverloadSetExpr* overload_set,
        ArrayRef<Expr*> args,
        Location call_loc
    ) -> std::pair<ProcDecl*, SmallVector<Expr*>>;

    /// Issue an error about lookup failure.
    void ReportLookupFailure(const LookupResult& result, Location loc);

    /// Issue an error about overload resolution failure.
    void ReportOverloadResolutionFailure(
        MutableArrayRef<Candidate> candidates,
        ArrayRef<Expr*> call_args,
        Location call_loc,
        u32 final_badness
    );

    /// Substitute types in a procedure template.
    auto SubstituteTemplate(
        ProcTemplateDecl* proc_template,
        ArrayRef<TypeLoc> input_types
    ) -> SubstitutionResult;

    /// Try to perform variable initialisation and return the result if
    /// if it succeeds, and nullptr on failure, but don’t emit a diagnostic
    /// either way.
    auto TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr>;

    /// Building AST nodes.
    auto BuildAssertExpr(Expr* cond, Ptr<Expr> msg, bool is_compile_time, Location loc) -> Ptr<Expr>;
    auto BuildArrayType(TypeLoc base, Expr* size) -> Type;
    auto BuildArrayType(TypeLoc base, i64 size, Location loc) -> Type;
    auto BuildBinaryExpr(Tk op, Expr* lhs, Expr* rhs, Location loc) -> Ptr<Expr>;
    auto BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, Location loc) -> BlockExpr*;
    auto BuildBuiltinCallExpr(BuiltinCallExpr::Builtin builtin, ArrayRef<Expr*> args, Location call_loc) -> Ptr<BuiltinCallExpr>;
    auto BuildBuiltinMemberAccessExpr(BuiltinMemberAccessExpr::AccessKind ak, Expr* operand, Location loc) -> Ptr<BuiltinMemberAccessExpr>;
    auto BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr>;
    auto BuildDeclRefExpr(ArrayRef<DeclName> names, Location loc) -> Ptr<Expr>;
    auto BuildEvalExpr(Stmt* arg, Location loc) -> Ptr<Expr>;
    auto BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, Location loc) -> Ptr<IfExpr>;
    auto BuildMatchExpr(Ptr<Expr> control_expr, Type ty, MutableArrayRef<MatchCase> cases, Location loc) -> Ptr<Expr>;
    auto BuildParamDecl(ProcScopeInfo& proc, const ParamTypeData* param, u32 index, bool with_param, String name, Location loc) -> ParamDecl*;
    auto BuildProcDeclInitial(Scope* proc_scope, ProcType* ty, DeclName name, Location loc, ParsedProcAttrs attrs, ProcTemplateDecl* pattern = nullptr) -> ProcDecl*;
    auto BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr>;
    auto BuildReturnExpr(Ptr<Expr> value, Location loc, bool implicit) -> ReturnExpr*;
    auto BuildSliceType(Type base, Location loc) -> Type;
    auto BuildStaticIfExpr(Expr* cond, ParsedStmt* then, Ptr<ParsedStmt> else_, Location loc) -> Ptr<Stmt>;
    auto BuildTupleType(ArrayRef<TypeLoc> types) -> Type;
    auto BuildTypeExpr(Type ty, Location loc) -> TypeExpr*;
    auto BuildUnaryExpr(Tk op, Expr* operand, bool postfix, Location loc) -> Ptr<Expr>;
    auto BuildWhileStmt(Expr* cond, Stmt* body, Location loc) -> Ptr<WhileStmt>;

    /// Entry point.
    void Translate(bool have_preamble, bool load_runtime);

    /// Statements.
#define PARSE_TREE_LEAF_EXPR(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Ptr<Stmt>;
#define PARSE_TREE_LEAF_DECL(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Decl*;
#define PARSE_TREE_LEAF_STMT(Name) auto Translate##Name(Parsed##Name* parsed, Type desired_type) -> Ptr<Stmt>;
#define PARSE_TREE_LEAF_TYPE(Name)
#include "srcc/ParseTree.inc"

    auto TranslateExpr(ParsedStmt* parsed, Type desired_type = Type()) -> Ptr<Expr>;

    /// Declarations.
    auto TranslateEntireDecl(Decl* decl, ParsedDecl* parsed) -> Ptr<Decl>;
    auto TranslateDeclInitial(ParsedDecl* parsed) -> std::optional<Ptr<Decl>>;
    auto TranslateProc(ProcDecl* decl, Ptr<ParsedStmt> body, ArrayRef<ParsedVarDecl*> decls) -> ProcDecl*;
    auto TranslateProcBody(ProcDecl* decl, ParsedStmt* body, ArrayRef<ParsedVarDecl*> decls) -> Ptr<Stmt>;
    auto TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<Decl>;
    auto TranslateStmt(ParsedStmt* parsed, Type desired_type = Type()) -> Ptr<Stmt>;
    auto TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed, Type desired_type = Type()) -> void;
    auto TranslateStruct(TypeDecl* decl, ParsedStructDecl* parsed) -> Ptr<TypeDecl>;
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
    auto TranslatePtrType(ParsedPtrType* stmt) -> Type;
    auto TranslateProcType(ParsedProcType* parsed, ArrayRef<Type> deduced_var_parameters = {}) -> Type;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }
};

#endif // SRCC_FRONTEND_SEMA_HH
