#ifndef SRCC_FRONTEND_SEMA_REQUESTS_HH
#define SRCC_FRONTEND_SEMA_REQUESTS_HH

#include <srcc/AST/Stmt.hh>
#include <srcc/Frontend/ParseTree.hh>

#include <llvm/ADT/PointerUnion.h>

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
/// be published before translation finishes, and mangling numbers are reserved
/// at registration so they do not depend on request evaluation order.
struct DeclarationRequest {
    ParsedDecl* parsed;
    Decl* result = nullptr;
    RequestContext context;
    ManglingNumber mangling = ManglingNumber::None;
    bool exported = false;
};

/// Source and lexical environment needed to translate a procedure body.
/// Ordinary procedures retain their parsed declaration; generated procedures
/// own a body builder instead. Both kinds use the saved context.
struct ProcedureSource {
    using BodyBuilder = std::move_only_function<Ptr<Expr>(ProcDecl*)>;
    Variant<ParsedProcDecl*, BodyBuilder> body;
    RequestContext context;
};

/// Identity of a cached semantic result. The pointer's type selects the request:
/// declaration translation, enumerator value, procedure body, or struct layout.
using RequestSubject = llvm::PointerUnion<
    DeclarationRequest*,
    EnumeratorDecl*,
    ProcDecl*,
    StructType*
>;

/// Cached request status.
enum struct RequestState : u8 { Evaluating, Complete, Failed };

/// One active dependency, including the requesting location used in cycle notes.
struct RequestFrame {
    RequestSubject subject;
    SLoc loc;
};

/// Request evaluator state. This tracks scheduling, source information, and
/// the lexical contexts needed to complete requests on demand.
struct Requests {
    /// Status of each requested subject.
    DenseMap<RequestSubject, RequestState> results;

    /// Active requests.
    SmallVector<RequestFrame> stack;

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

    /// Generated bodies awaiting completion, grouped by their lexical procedure.
    /// Signature and body translation can have different contexts for the same
    /// procedure; they share this worklist and retain their own source contexts.
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
