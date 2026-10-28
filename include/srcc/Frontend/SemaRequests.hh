#ifndef SRCC_FRONTEND_SEMA_REQUESTS_HH
#define SRCC_FRONTEND_SEMA_REQUESTS_HH

#include <srcc/AST/Stmt.hh>
#include <srcc/Frontend/ParseTree.hh>

#include <llvm/ADT/Hashing.h>

#include <deque>
#include <functional>
#include <memory>

namespace srcc::sema {
/// Maximum number of simultaneously active semantic requests.
inline constexpr usz MaxRequestDepth = 128;

/// Translation state for one entry into a procedure, linked to its lexical
/// parent. Requests retains these contexts after the entry's scope guard exits
/// so deferred work can restore its environment and register captures.
struct ProcedureContext {
    ProcDecl* proc;
    ProcedureContext* parent;
    SmallVector<LocalDecl*> locals;
    LoopToken loop_depth = LoopToken(0);
    bool current_loop_has_break = false;

    ProcedureContext(ProcDecl* proc, ProcedureContext* parent) : proc{proc}, parent{parent} {}
};

/// One quote expansion, linked to the expansion containing its injection site.
/// Quoted tokens retain their original locations; visibility comparisons lift
/// them to the injection site when crossing expansion boundaries. Requests owns
/// these records so saved request contexts can reference them safely.
struct ExpansionContext {
    ExpansionContext* parent;
    SLoc site;
    usz depth;
};

/// Lexical environment restored before resolving a declaration or body: its
/// scope, procedure, evaluation mode, and quote expansion. Scope and procedure
/// chains are shared through parent links rather than copied into each request.
/// Variable visibility is determined separately using source locations.
struct RequestContext {
    Scope* scope;
    ProcedureContext* procedure;
    bool inside_eval;
    ExpansionContext* expansion;
};

/// Deferred translation of a parsed declaration in one particular scope.
/// Specialisations sharing a parse tree have separate records. The result may
/// be published before translation finishes.
struct DeclarationRequest {
    ParsedDecl* parsed;
    Decl* result = nullptr;
    RequestContext context;
    bool exported = false;
};

/// One compile-time parameter in a substituted source signature.
struct CompileTimeParameter {
    /// Position among all source parameters, including runtime parameters.
    u32 index;

    /// Substituted parameter type and conversion constraints.
    ParamTypeData type;
};

/// A substituted template signature that stores compile-time parameters
/// along with the runtime procedure type.
class ProcTemplateSignature final :
    TrailingObjects<ProcTemplateSignature, CompileTimeParameter> {
    friend TrailingObjects;

    u32 num_compile_time_params;

    ProcTemplateSignature(ProcType* runtime, ArrayRef<CompileTimeParameter> compile_time);

public:
    /// Runtime signature; updated when the procedure body deduces its result.
    ProcType* runtime_type;

    /// Create a new template signature.
    static auto Create(
        TranslationUnit& tu,
        ProcType* runtime,
        ArrayRef<CompileTimeParameter> compile_time
    ) -> ProcTemplateSignature*;

    /// Compile-time parameters in source order.
    auto compile_time_params() const -> ArrayRef<CompileTimeParameter> {
        return getTrailingObjects(num_compile_time_params);
    }

    /// Number of source parameters, including compile-time parameters.
    auto param_count() const -> u32 {
        return runtime_type->param_count() + num_compile_time_params;
    }

    /// Get the type of a parameter by its source index.
    auto parameter(u32 source_index) const -> const ParamTypeData&;

    /// Print the source signature without manufacturing a mixed ProcType.
    auto print(ProcTemplateDecl* pattern) const -> SmallUnrenderedString;

    /// Map a runtime parameter index back to its source parameter index.
    auto source_index(u32 runtime_index) const -> u32;
};

/// Source and lexical environment needed to translate a procedure body.
/// Ordinary procedures retain their parsed declaration; generated procedures
/// own a body builder instead. Both kinds use the saved context.
struct ProcedureSource {
    using BodyBuilder = std::move_only_function<Ptr<Expr>(ProcDecl*)>;
    Variant<ParsedProcDecl*, BodyBuilder> body;
    RequestContext context;
    Ptr<ProcTemplateSignature> template_signature{};
};

/// A procedure reference that may need its callee's enclosing environment.
/// Recursive references can be formed before capture information is available;
/// retaining the caller and location lets that information propagate later.
struct ProcedureReference {
    ProcDecl* caller;
    SLoc loc;
};

/// One active semantic dependency and its requesting location.
struct Request {
    /// Translate a registered declaration.
    struct Declaration {
        DeclarationRequest* subject;
        bool operator==(const Declaration&) const = default;
    };

    /// Compute an enumerator's value.
    struct EnumeratorValue {
        EnumeratorDecl* subject;
        bool operator==(const EnumeratorValue&) const = default;
    };

    /// Translate or generate a procedure body and deduce its return type.
    struct ProcedureBody {
        ProcDecl* subject;
        bool operator==(const ProcedureBody&) const = default;
    };

    /// Complete record construction properties and declare generated initialisers.
    struct RecordInitialisers {
        RecordType* subject;
        bool operator==(const RecordInitialisers& other) const {
            return static_cast<const void*>(subject) == static_cast<const void*>(other.subject);
        }
    };

    /// Complete a struct's fields, physical layout, and member declarations.
    struct StructLayout {
        StructType* subject;
        bool operator==(const StructLayout& other) const {
            return static_cast<const void*>(subject) == static_cast<const void*>(other.subject);
        }
    };

    /// Request completion state.
    enum struct State : u8 { Evaluating, Complete, Failed };

    /// Request data.
    using Subject = Variant<
        Declaration,
        EnumeratorValue,
        ProcedureBody,
        RecordInitialisers,
        StructLayout
    >;

    Subject subject;
    SLoc loc;
};
} // namespace srcc::sema

/// Hashing and equality for semantic request cache keys.
template <>
struct llvm::DenseMapInfo<srcc::sema::Request::Subject> {
    using Request = srcc::sema::Request;

    static unsigned getHashValue(const Request::Subject& subject) {
        return subject.visit([&](auto request) {
            return unsigned(llvm::hash_combine(
                subject.index(),
                static_cast<const void*>(request.subject)
            ));
        });
    }

    static bool isEqual(const Request::Subject& lhs, const Request::Subject& rhs) {
        return lhs == rhs;
    }
};

namespace srcc::sema {
/// Request evaluator state. This tracks scheduling, source information, and
/// the lexical contexts needed to complete requests on demand.
struct Requests {
    /// Status of each requested subject.
    DenseMap<Request::Subject, Request::State> results;

    /// Active requests.
    SmallVector<Request> stack;

    /// Substitutions may recurse while evaluating signature expressions.
    usz template_substitution_depth = 0;

    /// Active initialiser overload resolutions and their argument types. This
    /// detects conversion cycles that do not re-enter a cached semantic request.
    SmallVector<std::pair<RecordType*, SmallVector<Type>>> resolving_initialisers;

    /// Translation contexts for each procedure.
    SmallVector<std::unique_ptr<ProcedureContext>> procedure_contexts;

    /// Declaration requests, keyed by scope and parse tree so
    /// separate template specialisations have different declaration
    /// translations.
    DenseMap<std::pair<Scope*, ParsedDecl*>, std::unique_ptr<DeclarationRequest>> declarations;

    /// Non-owning index of registered declarations in each scope. Lookup uses
    /// it to resolve matching declarations that have not yet been translated.
    DenseMap<Scope*, SmallVector<DeclarationRequest*>> scope_declarations;

    /// Maps semantic declarations back to their request records, providing
    /// lexical context for layout, template substitution, and visibility checks.
    DenseMap<Decl*, DeclarationRequest*> declaration_sources;

    /// Sources and lexical contexts for procedure definitions requested on
    /// demand. Each source holds a parsed declaration or owns a builder to
    /// generate the body.
    DenseMap<ProcDecl*, ProcedureSource> procedure_sources;

    /// References to nested procedures whose captures may still be discovered.
    DenseMap<ProcDecl*, SmallVector<ProcedureReference>> capture_users;

    /// Generated bodies and user initialisers awaiting completion, grouped by
    /// their lexical procedure. Signature and body translation can have different
    /// contexts for the same procedure; they share this worklist and retain their
    /// own source contexts.
    DenseMap<ProcDecl*, SmallVector<ProcDecl*>> pending_procedure_bodies;

    /// Enumerators whose implicit increments overflowed, including subsequent
    /// implicit values in that sequence. Suppresses repeated overflow diagnostics.
    llvm::DenseSet<EnumeratorDecl*> overflowing_enumerators;

    /// Enums whose values are currently being translated. References to their
    /// enumerators use the underlying integer type within those definitions.
    SmallVector<EnumType*> enum_value_stack;

    /// Order in which source files first appear during statement registration.
    /// Visibility checks use this order when comparing locations across files.
    DenseMap<const File*, usz> file_order;

    /// Owns quote expansion records with stable addresses for saved contexts
    /// and links between nested expansions.
    std::deque<ExpansionContext> expansions;

    /// Innermost active quote expansion, or null outside injected code. Restored
    /// when entering a saved request context or leaving an expansion.
    ExpansionContext* expansion = nullptr;
};
} // namespace srcc::sema

#endif
