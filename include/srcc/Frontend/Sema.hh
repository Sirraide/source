#ifndef SRCC_FRONTEND_SEMA_HH
#define SRCC_FRONTEND_SEMA_HH

#include <srcc/AST/AST.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Macros.hh>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TinyPtrVector.h>

#include <ranges>

namespace srcc {
class ModuleLoader;
class Sema;
}

namespace srcc {
class TemplateInstantiator;
}

/// Structure that is used to collect imports for a compilation job; the
/// loader does *not* own any modules; they are refcounted instead.
class srcc::ModuleLoader : public DefaultDiagsProducer<> {
    friend DefaultDiagsProducer;
    Context& ctx;
    StringMap<ImportHandle> modules;
    SmallVector<std::string> module_search_paths;

public:
    ModuleLoader(Context& ctx, ArrayRef<std::string> module_search_paths)
        : ctx{ctx}, module_search_paths{module_search_paths.begin(), module_search_paths.end()} {}

    /// Load a C++ header from the system include path.
    ///
    /// \param logical_name The name given to this in source.
    /// \param linkage_name The actual name of this.
    /// \param import_loc Where this was imported from.
    /// \param is_cxx_header Whether this is a C++ header name.
    auto load(
        String logical_name,
        String linkage_name,
        Location import_loc,
        bool is_cxx_header
    ) -> Opt<ImportHandle>;

    /// Release all handles held by the module loader; they will be deleted
    /// when the refcount reaches 0.
    void release_all() { modules.clear(); }

private:
    auto ImportCXXHeader(StringRef name, Location import_loc) -> Opt<ImportHandle>;
    auto LoadModuleFromArchive(StringRef name, Location import_loc) -> Opt<ImportHandle>;
};

class srcc::Sema : DiagsProducer<std::nullptr_t> {
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
        EnterScope(Sema& S, ScopeKind kind = ScopeKind::Block);
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

        ProcScopeInfo(Sema& S, ProcDecl* proc) : proc{proc}, es{S, proc->scope} {}
    };

    class [[nodiscard]] EnterProcedure {
        SRCC_IMMOVABLE(EnterProcedure);
        ProcScopeInfo info;

    public:
        explicit EnterProcedure(Sema& S, ProcDecl* proc);
        ~EnterProcedure() { info.es.S.proc_stack.pop_back(); }
    };

    /// A conversion from one type to another.
    struct Conversion {
        enum struct Kind : u8 {
            LValueToSRValue,
            IntegralCast,
            SelectOverload
        };

        Kind kind;
        union {
            Type ty;
            u16 index{};
        };

        Conversion(Kind kind) : kind{kind} {}
        Conversion(Kind kind, Type ty) : kind{kind}, ty{ty} {}
        Conversion(Kind kind, u16 index) : kind{kind}, index{index} {}

        static auto IntegralCast(Type ty) -> Conversion { return Conversion{Kind::IntegralCast, ty}; }
        static auto LValueToSRValue() -> Conversion { return Conversion{Kind::LValueToSRValue}; }
        static auto SelectOverload(u16 index) -> Conversion { return Conversion{Kind::SelectOverload, index}; }
    };

    /// A (possibly empty) sequence of conversions applied to a type.
    using ConversionSequence = SmallVector<Conversion, 1>;

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
        String name;

        /// Reason for failure.
        Reason result;

        LookupResult(String name) : name{name}, result{Reason::NotFound} {}
        LookupResult(ArrayRef<Decl*> decls, String name, Reason result) : decls{decls}, name{name}, result{result} {}
        LookupResult(Decl* decl, String name, Reason result) : name{name}, result{result} {
            if (decl) decls.push_back(decl);
        }

        /// Check if this lookup result is a success.
        [[nodiscard]] auto successful() const -> bool { return result == Reason::Success; }
        [[nodiscard]] explicit operator bool() const { return successful(); }

        static auto Ambiguous(String name, ArrayRef<Decl*> decls) { return LookupResult{decls, name, Reason::Ambiguous}; }
        static auto FailedToImport() { return LookupResult{{}, "", Reason::FailedToImport}; }
        static auto NonScopeInPath(String name, Decl* decl = nullptr) { return LookupResult{decl, name, Reason::NonScopeInPath}; }
        static auto NotFound(String name) { return LookupResult{name}; }
        static auto Success(Decl* decl) { return LookupResult{decl, "", Reason::Success}; }
    };

    /// The result of substituting a procedure type before template instantiation.
    struct SubstitutionInfo {
        LIBBASE_IMMOVABLE(SubstitutionInfo);

        /// The template that was substituted.
        ProcTemplateDecl* pattern;

        /// Canonical arguments provided by the user.
        SmallVector<Type> input_types;

        /// Substitution was successful.
        struct Success {
            /// The substituted procedure type.
            ProcType* type;

            /// Procedure scope to use for instantiation.
            Scope* scope;

            /// The instantiated procedure, if we have already instantiated it.
            ProcDecl* instantiation = nullptr;

            Success(
                ProcType* type,
                Scope* scope
            ) : type{type}, scope{scope} {}
        };

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

            /// The two parameters that caused the ambiguity.
            u32 first, second;

            /// The two types that we deduced each time.
            Type first_type, second_type;
        };

        /// There was a hard error which has already been reported.
        struct Error {};

        /// What happened.
        Variant< // clang-format off
            Success,
            DeductionFailed,
            DeductionAmbiguous,
            Error
        > data = Error{}; // clang-format on

        SubstitutionInfo() = default;
        SubstitutionInfo(ProcTemplateDecl* pattern, SmallVector<Type> input_types)
            : pattern{pattern}, input_types{std::move(input_types)} {}

        auto success() -> Success* { return data.get_if<Success>(); }
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

        // An lvalue was required by the parameter intent, but an
        // rvalue found.
        struct LValueIntentMismatch {
            // Index of the argument which caused the mismatch.
            u32 mismatch_index;
        };

        // One of the arguments is an overload set, and we failed
        // to match any of the overloads against a parameter. This
        // has information about why each of the overloads didn’t
        // match. Because this only performs a subset of overload
        // resolution, it only needs to deal with a subset of failure
        // reasons.
        struct NestedResolutionFailure {
            // Index of the argument which contains this overload set.
            u32 mismatch_index;
        };

        // An lvalue parameter requires a value of the same type to be
        // passed, but that wasn’t the case.
        struct SameTypeLValueRequired {
            // Index of the argument which caused the mismatch.
            u32 mismatch_index;
        };

        // Candidate for which there was something wrong with the arguments;
        // used for generic argument-related failures.
        struct TypeMismatch {
            // Which argument rendered it not viable.
            u32 mismatch_index;
        };

        // The procedure (template) that this candidate represents.
        Decl* decl;

        // Substitution for this template if this is one. This may not
        // exist in some error cases.
        SubstitutionInfo* subst = nullptr;

        // Whether this candidate is still viable, or why not.
        using Status = Variant< // clang-format off
            Viable,
            ArgumentCountMismatch,
            DeductionError,
            LValueIntentMismatch,
            NestedResolutionFailure,
            SameTypeLValueRequired,
            TypeMismatch
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
        bool has_valid_proc_type() const;
        bool is_template() const { return isa<ProcTemplateDecl>(decl); }
        bool is_variadic() const;
        bool viable() const { return status.is<Viable>(); }
    };

    Context& ctx;
    TranslationUnit::Ptr M;
    ArrayRef<ParsedModule::Ptr> parsed_modules;

    /// Stack of active procedures.
    SmallVector<ProcScopeInfo*> proc_stack;

    /// Stack of active scopes.
    SmallVector<Scope*> scope_stack;

    /// Template deduction information for each template.
    DenseMap<ParsedProcDecl*, TemplateParamDeductionInfo> parsed_template_deduction_infos;

    /// C++ decls that have already been imported (or that
    /// already failed to import before).
    DenseMap<clang::Decl*, Ptr<Decl>> imported_decls;

    /// Cached template substitutions.
    DenseMap<ProcTemplateDecl*, SmallVector<std::unique_ptr<SubstitutionInfo>>> template_substitutions;

    /// Map from instantiations to their substitutions.
    DenseMap<ProcDecl*, usz> template_substitution_indices;

    explicit Sema(Context& ctx) : ctx(ctx) {}

public:
    /// Analyse a set of parsed modules and combine them into a single module.
    ///
    /// @return The combined module, or `nullptr` if there was an error.
    [[nodiscard]] static auto Translate(
        const LangOpts& opts,
        ArrayRef<ParsedModule::Ptr> modules,
        StringMap<ImportHandle> imported_modules
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
    auto global_scope() -> Scope* { return scope_stack.front(); }

private:
    /// Add a declaration to a scope.
    ///
    /// This correctly handles redeclarations and declarations
    /// with an empty name.
    void AddDeclToScope(Scope* scope, Decl* d);

    /// Build an initialiser for an aggregate type.
    ///
    /// This should not be called directly; call BuildInitialiser() instead.
    auto BuildAggregateInitialiser(StructType* s, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr>;

    /// Build an rvalue that can initialise a variable using a
    /// list of arguments.
    ///
    /// During parameter initialisation, pass the parameter intent
    /// if there is one; in all other cases, leave it as 'Move'
    /// because that is the default for assignment.
    ///
    /// The expression that results from the conversion is an rvalue
    /// suitable for initialising a variable of the given type iff it
    /// is an srvalue, or an expression that, when provided with a
    /// memory location, evaluates an mrvalue into that location.
    ///
    /// \param ctx The initialisation context; this determines whether
    ///   any required conversions are applied immediately, or deferred
    ///   until a later point in time.
    ///
    /// \param var_type The type of the variable to be initialised.
    /// \param arg The argument to initialise it with.
    /// \param intent If the variable is a parameter, its intent.
    ///
    /// \param cc If this is an argument to a call, the calling convention
    ///    of the called procedure.
    ///
    /// \param in_call Whether this is argument passing.
    ///
    /// \return Whether initialisation was successful.
    template <typename InitContext>
    bool BuildInitialiser(
        InitContext& ctx,
        Type var_type,
        Expr* arg,
        Intent intent = Intent::Move,
        CallingConvention cc = CallingConvention::Source,
        bool in_call = false
    );

    /// Build and initialiser immediately, returning the
    /// computed initialiser or nullptr on error.
    auto BuildInitialiser(
        Type var_type,
        Expr* arg,
        Intent intent,
        CallingConvention cc,
        Location loc = {}
    ) -> Ptr<Expr>;

    /// Build an initialiser immediately. 'args' can be empty to
    /// attempt to build an empty initialiser.
    auto BuildInitialiser(Type var_type, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr>;

    /// Check that a type is valid for a variable declaration.
    [[nodiscard]] Type AdjustVariableType(Type ty, Location loc);

    /// Apply a conversion to an expression.
    [[nodiscard]] auto ApplyConversion(Expr* e, Conversion conv) -> Expr*;

    /// Apply a conversion sequence to an expression.
    [[nodiscard]] auto ApplyConversionSequence(Expr* e, ConversionSequence& seq) -> Expr*;

    /// Create a reference to a declaration.
    [[nodiscard]] auto CreateReference(Decl* d, Location loc) -> Ptr<Expr>;

    /// Add a variable to the current scope and procedure.
    void DeclareLocal(LocalDecl* d);

    /// Perform template deduction.
    auto DeduceType(
        ParsedStmt* parsed_type,
        u32 parsed_type_index,
        ArrayRef<TypeLoc> input_types
    ) -> Type;

    /// Extract the scope that is the body of a declaration, if it has one.
    auto GetScopeFromDecl(Decl* d) -> Ptr<Scope>;

    /// Ensure that an expression is an srvalue of the given type. This is
    /// mainly used for expressions involving operators.
    bool MakeSRValue(Type ty, Expr*& e, StringRef elem_name, StringRef op);

    /// Import a declaration from a C++ AST.
    auto ImportCXXDecl(clang::ASTUnit& ast, CXXDecl* decl) -> Ptr<Decl>;

    /// Instantiate a procedure template.
    auto InstantiateTemplate(SubstitutionInfo& info, Location inst_loc) -> ProcDecl*;

    /// Check if an integer literal can be stored in a given type.
    bool IntegerFitsInType(const APInt& i, Type ty);

    /// Use LookUpName() instead.
    auto LookUpCXXName(clang::ASTUnit* ast, ArrayRef<String> names) -> LookupResult;

    /// Use LookUpName() instead.
    auto LookUpQualifiedName(Scope* in_scope, ArrayRef<String> names) -> LookupResult;

    /// Perform unqualified name lookup.
    auto LookUpUnqualifiedName(
        Scope* in_scope,
        String name,
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
        ArrayRef<String> names,
        Location loc,
        bool complain = true
    ) -> LookupResult;

    /// Convert an lvalue to an srvalue.
    [[nodiscard]] auto LValueToSRValue(Expr* expr) -> Expr*;

    /// Materialise a temporary value.
    [[nodiscard]] auto MaterialiseTemporary(Expr* expr) -> Expr*;

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
        ArrayRef<Candidate> candidates,
        ArrayRef<Expr*> call_args,
        Location call_loc,
        u32 final_badness
    );

    /// Substitute types in a procedure template.
    auto SubstituteTemplate(
        ProcTemplateDecl* proc_template,
        ArrayRef<TypeLoc> input_types
    ) -> SubstitutionInfo&;

    /// Try to perform variable initialisation and return the result if
    /// if it succeeds, and nullptr on failure, but don’t emit a diagnostic
    /// either way.
    auto TryBuildInitialiser(Type var_type, Expr* arg) -> Ptr<Expr>;

    /// Building AST nodes; called after translation and template instantiation.
    auto BuildAssertExpr(Expr* cond, Ptr<Expr> msg, Location loc) -> Ptr<AssertExpr>;
    auto BuildBinaryExpr(Tk op, Expr* lhs, Expr* rhs, Location loc) -> Ptr<BinaryExpr>;
    auto BuildBlockExpr(Scope* scope, ArrayRef<Stmt*> stmts, Location loc) -> BlockExpr*;
    auto BuildBuiltinCallExpr(BuiltinCallExpr::Builtin builtin, ArrayRef<Expr*> args, Location call_loc) -> Ptr<BuiltinCallExpr>;
    auto BuildBuiltinMemberAccessExpr(BuiltinMemberAccessExpr::AccessKind ak, Expr* operand, Location loc) -> Ptr<BuiltinMemberAccessExpr>;
    auto BuildCallExpr(Expr* callee_expr, ArrayRef<Expr*> args, Location loc) -> Ptr<Expr>;
    auto BuildEvalExpr(Stmt* arg, Location loc) -> Ptr<Expr>;
    auto BuildIfExpr(Expr* cond, Stmt* then, Ptr<Stmt> else_, Location loc) -> Ptr<IfExpr>;
    auto BuildParamDecl(ProcScopeInfo& proc, const ParamTypeData* param, u32 index, bool with_param, String name, Location loc) -> ParamDecl*;
    auto BuildProcDeclInitial(Scope* proc_scope, ProcType* ty, String name, Location loc, ParsedProcAttrs attrs) -> ProcDecl*;
    auto BuildProcBody(ProcDecl* proc, Expr* body) -> Ptr<Expr>;
    auto BuildReturnExpr(Ptr<Expr> value, Location loc, bool implicit) -> ReturnExpr*;
    auto BuildStaticIfExpr(Expr* cond, ParsedStmt* then, Ptr<ParsedStmt> else_, Location loc) -> Ptr<Stmt>;
    auto BuildTypeExpr(Type ty, Location loc) -> TypeExpr*;
    auto BuildUnaryExpr(Tk op, Expr* operand, bool postfix, Location loc) -> Ptr<UnaryExpr>;
    auto BuildWhileStmt(Expr* cond, Stmt* body, Location loc) -> Ptr<WhileStmt>;

    /// Entry point.
    void Translate();

    /// Statements.
#define PARSE_TREE_LEAF_EXPR(Name) auto Translate##Name(Parsed##Name* parsed)->Ptr<Stmt>;
#define PARSE_TREE_LEAF_DECL(Name) auto Translate##Name(Parsed##Name* parsed)->Decl*;
#define PARSE_TREE_LEAF_STMT(Name) auto Translate##Name(Parsed##Name* parsed)->Ptr<Stmt>;
#define PARSE_TREE_LEAF_TYPE(Name)
#include "srcc/ParseTree.inc"

    auto TranslateExpr(ParsedStmt* parsed) -> Ptr<Expr>;

    /// Declarations.
    auto TranslateEntireDecl(Decl* decl, ParsedDecl* parsed) -> Ptr<Decl>;
    auto TranslateDeclInitial(ParsedDecl* parsed) -> std::optional<Ptr<Decl>>;
    auto TranslateProc(ProcDecl* decl, Ptr<ParsedStmt> body, ArrayRef<ParsedLocalDecl*> decls) -> ProcDecl*;
    auto TranslateProcBody(ProcDecl* decl, ParsedStmt* body, ArrayRef<ParsedLocalDecl*> decls) -> Ptr<Stmt>;
    auto TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<Decl>;
    auto TranslateStmt(ParsedStmt* parsed) -> Ptr<Stmt>;
    auto TranslateStmts(SmallVectorImpl<Stmt*>& stmts, ArrayRef<ParsedStmt*> parsed) -> void;
    auto TranslateStruct(TypeDecl* decl, ParsedStructDecl* parsed) -> Ptr<TypeDecl>;
    auto TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl>;

    /// Types.
    auto TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type;
    auto TranslateIntType(ParsedIntType* parsed) -> Type;
    auto TranslateNamedType(ParsedDeclRefExpr* parsed) -> Type;
    auto TranslateSliceType(ParsedSliceType* parsed) -> Type;
    auto TranslateTemplateType(ParsedTemplateType* parsed) -> Type;
    auto TranslateType(ParsedStmt* stmt, Type fallback = Type()) -> Type;
    auto TranslateProcType(ParsedProcType* parsed) -> Type;

    template <typename... Args>
    void Diag(Diagnostic::Level lvl, Location where, std::format_string<Args...> fmt, Args&&... args) {
        ctx.diags().diag(lvl, where, fmt, std::forward<Args>(args)...);
    }
};

#endif // SRCC_FRONTEND_SEMA_HH
