// =================================================================================
// SemaRequests.cc - Request Handlers
//
// Semantic requests compute one result; completed results live in the AST; the
// request table caches success and failure. Re-entering an active request is an
// error.
//
// For example, this resolves the signature and then requests the body of 'f':
//
//     #assert f() == 42;
//     proc f -> int = 42;
//
// A return-type inference cycle instead re-enters an active body request:
//
//     proc f = g();
//     proc g = f();
//
// Record layout and initialiser readiness are separate requests. For example:
//
//     struct A { value: B; }
//     struct B { value: int; init = A::(B::(42,),).value; }
//
// Completing A's physical layout must not infer B's initialiser yet: that body
// needs A's layout. Initialiser readiness can subsequently infer return types
// and register generated signatures. Generated bodies use ProcedureSource body
// builders and the same cached procedure-body requests as parsed procedures.
// Raw construction requires only the target's layout; fields initialise normally.
//
// Providers are separate Sema members, dispatched by Request(). AST emission
// retains source order even when a later variable's type is needed earlier, e.g.:
//
//     proc outer () {
//         var result = later();
//         var value: int = side_effect();
//         proc later -> int = typeof(value)(42);
//     }
//
// 'value' is visible inside 'later', but its initialiser still runs after the
// call. Each template specialisation has its own scope and declaration requests.
//
// Compile-time parameters extend the substitution key with converted argument
// values. Their bindings are published while translating the source signature:
//
//     proc f ($t: type, $n: int, value: t[n]) -> t[n] = value;
//
// Here t and n are constant declarations available to later parameter types,
// constraints, and body requests. Only their argument values make it into the
// runtime signature.
// =================================================================================
#include <srcc/AST/Eval.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Frontend/Sema.hh>

using namespace srcc;
using namespace srcc::sema;

#define TRY(...) ({                      \
    auto _res = (__VA_ARGS__);           \
    if (not _res) return utils::Falsy(); \
    *_res;                               \
})

ProcTemplateSignature::ProcTemplateSignature(
    ProcType* runtime,
    ArrayRef<CompileTimeParameter> compile_time
) : num_compile_time_params{u32(compile_time.size())}, runtime_type{runtime} {
    std::uninitialized_copy(compile_time.begin(), compile_time.end(), getTrailingObjects());
}

auto ProcTemplateSignature::Create(
    TranslationUnit& tu,
    ProcType* runtime,
    ArrayRef<CompileTimeParameter> compile_time
) -> ProcTemplateSignature* {
    auto size = totalSizeToAlloc<CompileTimeParameter>(compile_time.size());
    auto memory = tu.allocate(size, alignof(ProcTemplateSignature));
    return ::new (memory) ProcTemplateSignature(runtime, compile_time);
}

auto ProcTemplateSignature::parameter(u32 index) const -> const ParamTypeData& {
    Assert(index < param_count());
    u32 runtime_index = index;
    for (auto& param : compile_time_params()) {
        if (param.index == index) return param.type;
        if (param.index > index) break;
        --runtime_index;
    }
    return runtime_type->params()[runtime_index];
}

auto ProcTemplateSignature::print(ProcTemplateDecl* pattern) const -> SmallUnrenderedString {
    SmallUnrenderedString out;
    out += "%1(proc%) (";
    for (u32 i = 0; i < param_count(); ++i) {
        if (i) out += ", ";
        auto& param = parameter(i);
        auto parsed = pattern->pattern->params()[i];
        if (param.exact) out += "%1(exact%) ";
        if (param.intent != Intent::Move) Format(out, "%1({}%) ", param.intent);
        if (parsed->compile_time) Format(out, "%3(${}%): ", parsed->name);
        out += param.type->print();
        if (param.variadic) out += "...";
    }
    if (runtime_type->has_c_varargs()) out += param_count() ? ", ..." : "...";
    Format(out, ") -> {}", runtime_type->ret());
    return out;
}

auto ProcTemplateSignature::source_index(u32 index) const -> u32 {
    Assert(index < runtime_type->param_count());
    for (auto& param : compile_time_params()) {
        if (param.index > index) break;
        ++index;
    }
    return index;
}

Sema::EnterRequestContext::EnterRequestContext(
    Sema& S,
    RequestContext context
) : S{S},
    scopes{std::move(S.scope_stack)},
    procedures{std::move(S.proc_stack)},
    inside_eval{std::exchange(S.inside_eval, context.inside_eval)},
    expansion{std::exchange(S.request_data.expansion, context.expansion)} {
    // Rebuild scope stack.
    S.scope_stack.clear();
    for (auto scope = context.scope; scope; scope = scope->parent()) S.scope_stack.push_back(scope);
    rgs::reverse(S.scope_stack);

    // Rebuild procedure stack.
    S.proc_stack.clear();
    for (auto proc = context.procedure; proc; proc = proc->parent) S.proc_stack.push_back(proc);
    rgs::reverse(S.proc_stack);
}

Sema::EnterRequestContext::~EnterRequestContext() {
    S.scope_stack = std::move(scopes);
    S.proc_stack = std::move(procedures);
    S.inside_eval = inside_eval;
    S.request_data.expansion = expansion;
}

auto Sema::GetDeclarationRequest(Decl* decl) -> Ptr<DeclarationRequest> {
    return request_data.declaration_sources.lookup(decl);
}

auto Sema::GetDeclarationRequest(ParsedDecl* decl) -> Ptr<DeclarationRequest> {
    auto request = request_data.declarations.find({curr_scope(), decl});
    if (request == request_data.declarations.end()) return nullptr;
    return request->second.get();
}

auto Sema::GetRequestContext() -> RequestContext {
    return {curr_scope(), &curr_proc(), inside_eval, request_data.expansion};
}

auto Sema::GetRequestResult(RequestSubject subject) -> std::optional<bool> {
    auto it = request_data.results.find(subject);
    if (
        it == request_data.results.end() or
        it->second == Request::State::Evaluating
    ) return std::nullopt;
    return it->second == Request::State::Complete;
}

bool Sema::IsVariableVisible(const DeclarationRequest& source, SLoc use) {
    auto declaration = source.parsed->loc;
    auto declaration_expansion = source.context.expansion;
    auto use_expansion = request_data.expansion;

    // Compare locations in their nearest shared expansion context.
    while (declaration_expansion != use_expansion) {
        auto declaration_depth = declaration_expansion ? declaration_expansion->depth : 0;
        auto use_depth = use_expansion ? use_expansion->depth : 0;

        if (declaration_depth >= use_depth) {
            declaration = declaration_expansion->site;
            declaration_expansion = declaration_expansion->parent;
        }

        if (use_depth >= declaration_depth) {
            use = use_expansion->site;
            use_expansion = use_expansion->parent;
        }
    }

    // Determine whether the use of the variable precedes the declaration.
    Assert(declaration.is_valid() and use.is_valid(), "Variables should have valid locations");
    auto declaration_file = declaration.file(ctx);
    auto use_file = use.file(ctx);
    Assert(declaration_file and use_file);
    if (declaration_file == use_file) return declaration <= use;
    auto first = request_data.file_order.find(declaration_file);
    auto second = request_data.file_order.find(use_file);
    Assert(first != request_data.file_order.end() and second != request_data.file_order.end());
    return first->second < second->second;
}

bool Sema::RegisterCapture(ProcDecl* owner, Decl* captured, SLoc loc) {
    struct Capture {
        ProcDecl* caller;
        ProcDecl* owner;
        Decl* captured;
        SLoc loc;
    };

    // Captures that still need to be validated.
    SmallVector<Capture> pending{{curr_proc().proc, owner, captured, loc}};
    bool all_ok = true;

    // Keep going until we've validated all captures.
    while (not pending.empty()) {
        auto capture = pending.pop_back_val();

        // Validate the whole lexical chain before changing capture state.
        bool curr_ok = true;
        for (
            ProcDecl* proc = capture.caller;
            proc != capture.owner;
            proc = proc->parent.get() // Unwrap here as we must eventually reach the owner.
        ) {
            // Ignore non-special procedures.
            if (proc->special_proc == SpecialProcedure::None) continue;

            // This is a special procedure that has a capture, this is ill-formed.
            capture.caller->contains_error = proc->contains_error = true;

            // A recursive reference can expose a forbidden capture after the
            // caller's body request completed. Invalidate that cached result too.
            if (GetRequestResult(Request::ProcedureBody{capture.caller}) == true)
                request_data.results[Request::ProcedureBody{capture.caller}] = Request::State::Failed;

            if (GetRequestResult(Request::ProcedureBody{proc}) == true)
                request_data.results[Request::ProcedureBody{proc}] = Request::State::Failed;

            // Diagnose the capture.
            if (isa<LocalDecl>(capture.captured)) {
                Error(
                    capture.loc,
                    "Variable '{}' cannot be captured inside '%1({}%)'",
                    capture.captured->name,
                    proc->name
                );
                Note(capture.captured->location(), "Variable declared here");
            } else {
                Error(
                    capture.loc,
                    "Procedure '{}' cannot be referenced inside '%1({}%)' because it captures enclosing variables",
                    capture.captured->name,
                    proc->name
                );
                Note(capture.captured->location(), "Procedure declared here");
            }

            curr_ok = all_ok = false;
            break;
        }

        // Move on to the next capture if we issued a diagnostic.
        if (not curr_ok) continue;

        // Go up the call chain and capture the variable recursively.
        for (
            ProcDecl* proc = capture.caller;
            proc != capture.owner;
            proc = proc->parent.get()
        ) {
            if (proc->has_captures) continue;
            proc->has_captures = true;

            // Mark that we’ve computed that this procedure requires captures.
            auto users = request_data.capture_users.find(proc);
            if (users == request_data.capture_users.end()) continue;

            // Each procedure that creates a closure for this procedure must itself
            // be checked to determine whether the capture is valid.
            for (auto use : users->second)
                pending.push_back({use.caller, proc->parent.get(), proc, use.loc});
        }
    }

    return all_ok;
}

bool Sema::Request(RequestSubject subject, SLoc loc) {
    // Check if the request has already been fulfilled.
    if (auto result = GetRequestResult(subject)) return *result;

    // If we’re currently resolving it, report an error.
    //
    // FIXME: We should think of a nicer way to format these, similarly
    // to what we do for overload resolution diagnostics.
    if (request_data.results.contains(subject)) {
        Error(loc, "Cyclic semantic dependency");
        bool in_cycle = false;
        for (auto frame : request_data.stack) {
            in_cycle |= frame.subject == subject;
            if (not in_cycle) continue;
            frame.subject.visit(utils::Overloaded{
                [&](Request::Declaration r) {
                    Note(frame.loc, "While resolving declaration '{}'", r.subject->parsed->name);
                },
                [&](Request::EnumeratorValue r) {
                    Note(frame.loc, "While computing value of enumerator '{}'", r.subject->name);
                },
                [&](Request::ProcedureBody r) {
                    Note(frame.loc, "While translating body of '{}'", r.subject->name);
                },
                [&](Request::RecordInitialisers r) {
                    Note(frame.loc, "While completing initialisers of '{}'", r.subject);
                },
                [&](Request::StructLayout r) {
                    Note(frame.loc, "While computing layout of '{}'", r.subject->decl()->name);
                },
            });
        }
        return false;
    }

    // Take care not to overflow the stack.
    if (request_data.stack.size() >= sema::MaxRequestDepth) {
        request_data.results[subject] = Request::State::Failed;
        return Error(loc, "Exceeded maximum semantic request nesting depth ({})", sema::MaxRequestDepth);
    }

    // Mark that we’ve started processing this request.
    request_data.results[subject] = Request::State::Evaluating;
    request_data.stack.push_back({subject, loc});
    defer { request_data.stack.pop_back(); };

    // Actually process the request.
    bool success = subject.visit(utils::Overloaded{
        [&](Request::Declaration r) { return RequestDeclarationImpl(*r.subject); },
        [&](Request::EnumeratorValue r) { return RequestEnumeratorImpl(r.subject); },
        [&](Request::ProcedureBody r) { return RequestProcedureBodyImpl(r.subject); },
        [&](Request::RecordInitialisers r) { return RequestRecordInitialisersImpl(r.subject); },
        [&](Request::StructLayout r) { return RequestStructLayoutImpl(r.subject); },
    });

    // Set its result.
    request_data.results[subject] = success ? Request::State::Complete : Request::State::Failed;
    return success;
}

bool Sema::RequestCompleteType(Type ty, SLoc loc) {
    // Structs can be made complete on demand.
    // FIXME: Tuples as well.
    if (auto s = dyn_cast<StructType>(ty))
        return RequestStructLayout(s);
    if (not ty->is_complete())
        return Error(loc, "Invalid use of incomplete type '{}'", ty);
    return true;
}

auto Sema::RequestDeclaration(DeclarationRequest& request) -> Ptr<Decl> {
    Request(Request::Declaration{&request}, request.parsed->loc);
    return request.result;
}

bool Sema::RequestDeclarationImpl(DeclarationRequest& request) {
    EnterRequestContext context{*this, request.context};

    // Perform initial translation; for variables, which are order-dependent,
    // we perform full translation instead.
    if (not request.result) {
        if (auto var = dyn_cast<ParsedVarDecl>(request.parsed)) {
            request.result = TranslateVarDecl(var, {});
        } else {
            auto result = TranslateDeclInitial(request.parsed);
            if (result) request.result = result->get_or_null();
        }
    }

    if (not request.result) return false;
    request_data.declaration_sources[request.result] = &request;
    if (request.exported) {
        if (isa<ProcTemplateDecl>(request.result)) {
            request.result->set_invalid();
            return Error(request.parsed->loc, "Exporting procedure templates is not supported yet");
        }

        AddDeclToScope(&tu->exports, request.result);
        if (auto object = dyn_cast<ObjectDecl>(request.result)) object->linkage = Linkage::Exported;
    }

    return request.result->valid();
}

bool Sema::RequestEnumerator(EnumeratorDecl* enumerator) {
    // Check if we already know its value.
    if (auto result = GetRequestResult(Request::EnumeratorValue{enumerator})) return *result;
    if (enumerator->value) return true;

    // If not, request it and save it in the AST.
    bool ok = Request(Request::EnumeratorValue{enumerator}, enumerator->location());
    if (not ok and not enumerator->value) {
        enumerator->value = tu->store_int(
            APInt{unsigned(enumerator->parent->bit_width(*tu).bits()), 0}
        );
    }
    return ok;
}

bool Sema::RequestEnumeratorImpl(EnumeratorDecl* enumerator) {
    auto e = enumerator->parent;
    auto source = GetDeclarationRequest(e->decl());
    Assert(source, "Computing an enumerator that we did not parse");

    // Enter the scope of the enum.
    EnterRequestContext context{*this, source->context};
    EnterScope scope{*this, e->scope()};

    // Mark that we’re entering the definition of this enumerator; this
    // tells Sema to set its type to the underlying type of the enum.
    request_data.enum_value_stack.push_back(e);
    defer { request_data.enum_value_stack.pop_back(); };

    // Translate the value if we parsed one.
    auto parsed = cast<ParsedEnumDecl>(source->parsed);
    auto index = enumerator->index;
    auto& p = parsed->enumerators()[index];
    bool ok = true;
    if (auto init = p.value.get_or_null()) {
        auto value = TranslateExpr(init);
        if (value) value = BuildInitialiser(e->elem(), value.get(), init->loc);
        if (value) {
            if (auto evaluated = RequestEvaluation(value.get())) {
                enumerator->value = tu->store_int(evaluated->cast<APInt>());
                return true;
            }
        }
        ok = false;
    }

    // If not, compute the next value; this may require recursively requesting
    // the preceding enumerators.
    APInt value{unsigned(e->bit_width(*tu).bits()), 0};
    if (index) {
        auto previous = e->enumerator(index - 1);
        ok &= RequestEnumerator(previous);
        value = previous->value->value();

        // If the computation of the previous enumerator overflowed,
        // this one logically must too.
        if (request_data.overflowing_enumerators.contains(previous)) {
            request_data.overflowing_enumerators.insert(enumerator);
        }

        // If not, compute the next value by adding 1.
        else {
            bool overflow = false;
            auto next = value.uadd_ov(APInt{value.getBitWidth(), 1}, overflow);
            if (overflow) {
                Error(
                    enumerator->location(),
                    "Value '%5({}%)' of enumerator does not fit in underlying type '{}'",
                    value.zext(value.getBitWidth() + 2) + 1, e->elem()
                );

                // Remember that this overflowed to prevent duplicate diagnostics.
                request_data.overflowing_enumerators.insert(enumerator);
                ok = false;
            } else {
                value = std::move(next);
            }
        }
    }

    // Store the value in the AST.
    enumerator->value = tu->store_int(value);
    return ok;
}

void Sema::RequestEnumerators(EnumType* e) {
    auto source = GetDeclarationRequest(e->decl());
    Assert(source, "Completing an enum that we did not parse");
    auto parsed = cast<ParsedEnumDecl>(source->parsed);

    // Resolving a value can introduce declarations through an injection. Use
    // indices rather than retaining an iterator into the scope's declaration array.
    for (usz index = 0; index < parsed->enumerators().size(); ++index)
        RequestEnumerator(e->enumerator(u32(index)));

    e->finalise();
}

auto Sema::RequestEvaluation(Stmt* e, bool complain) -> std::optional<eval::RValue> {
    return tu->vm.eval(this, e, complain);
}

auto Sema::RequestEvaluationIntoExpr(Stmt* s, SLoc loc) -> Ptr<Expr> {
    auto value = RequestEvaluation(s);
    if (not value.has_value()) return nullptr;
    return MakeConstExpr(s, std::move(*value), loc);
}

auto Sema::RequestLookup(
    Ptr<Scope> start_scope,
    ArrayRef<DeclNameLoc> names,
    SLoc loc,
    LookupHint hint
) -> LookupResult {
    Assert(not names.empty());

    // If we have no scope and a single name, then this is unqualified lookup.
    if (start_scope.invalid() and names.size() == 1)
        return RequestUnqualifiedLookup(names.front(), hint);

    // If we don't have a scope, perform unqualified lookup of the first name
    // first. If we only have a single name, instead use the current scope.
    Scope* in_scope = nullptr;
    if (start_scope.invalid()) {
        Assert(names.size() > 1);

        // The first segment is looked up using unqualified lookup, but don’t
        // complain immediately if we can’t find it because we also need to
        // check module names. The first segment is also allowed to be empty,
        // which means we’re looking up a name in the global scope.
        auto first = names.consume_front();
        Assert(not first.name.empty());
        switch (auto res = RequestUnqualifiedLookup(first, LookupHint::Scope); res.result) {
            using enum LookupResult::Reason;
            case Success: {
                auto scope = GetScopeFromDecl(res.decls.front());
                if (scope.invalid()) return LookupResult::NonScopeInPath(first, res.decls.front());
                in_scope = scope.get();
            } break;

            // Can’t perform scope access on an ambiguous lookup.
            case Ambiguous:
                return res;

            // Failed imports can only happen with C++ declarations, which are handled
            // elsewhere; we should *never* fail to import something from one of our
            // own modules.
            case FailedToImport:
                Unreachable("Should never happen");

            // Unqualified lookup should never complain about this.
            case NonScopeInPath:
                Unreachable("Non-scope error in unqualified lookup?");

            // Search module names here.
            //
            // This way, we don’t have to try and represent modules in the AST,
            // and we also don’t have to deal with what happens if unqualified
            // lookup finds a module name, because a module name alone is useless
            // if it’s not on the lhs of `::`.
            case NotFound: {
                auto it = tu->logical_imports.find(first.name.str());
                if (it == tu->logical_imports.end() or not it->getValue()) return res;
                if (auto s = dyn_cast<ImportedSourceModuleDecl>(it->second)) {
                    in_scope = &s->exports;
                    break;
                }

                // We found an imported C++ header; do a C++ lookup.
                auto hdr = dyn_cast<ImportedClangModuleDecl>(it->second);
                return LookUpCXXName(hdr, names, hint);
            } break;
        }
    } else {
        in_scope = start_scope.get();
    }

    // For all elements but the last, we have to look up scopes.
    Assert(in_scope);
    for (auto name : names.drop_back()) {
        // Perform lookup.
        LookupResult res{llvm::to_vector(in_scope->decls_with_name(name.name)), name};
        if (not res.successful()) return res;

        // The declaration must reference a scope.
        auto scope = GetScopeFromDecl(res.decls.front());
        if (scope.invalid()) {
            res.result = LookupResult::Reason::NonScopeInPath;
            return res;
        }

        // Keep going down the path.
        in_scope = scope.get();
    }

    // Finally, look up the last name in the scope.
    return RequestScopeLookup(in_scope, names.back(), hint);
}

bool Sema::RequestParameterTypes(ProcDecl* proc) {
    bool ok = true;
    for (auto param : proc->params()) ok &= CheckVariableType(param->type, param->location());
    if (not ok) proc->set_invalid();
    return ok;
}

bool Sema::RequestPendingProcedureBodies(ProcDecl* owner) {
    if (not owner) owner = curr_proc().proc;
    bool ok = true;

    // Detach each batch before invoking providers. Recursive drains see only
    // newly registered work and cannot request an already-active body again.
    for (;;) {
        auto pending = request_data.pending_procedure_bodies.find(owner);
        if (pending == request_data.pending_procedure_bodies.end()) break;
        auto procedures = std::move(pending->second);
        request_data.pending_procedure_bodies.erase(pending);
        for (auto proc : procedures) ok &= RequestProcedureBody(proc, proc->location());
    }

    return ok;
}

bool Sema::RequestProcedureBody(ProcDecl* proc, SLoc loc) {
    if (auto result = GetRequestResult(Request::ProcedureBody{proc})) return *result;
    if (proc->body()) return proc->valid() and not proc->contains_error;
    if (not proc->will_have_body) return proc->valid();
    return Request(Request::ProcedureBody{proc}, loc);
}

bool Sema::RequestProcedureBodyImpl(ProcDecl* proc) {
    auto it = request_data.procedure_sources.find(proc);
    if (it == request_data.procedure_sources.end())
        return Error(proc->location(), "Procedure body is not available yet");

    // The builder must stay alive if completing this body rehashes the map.
    auto source = std::move(it->second);
    EnterRequestContext context{*this, source.context};
    EnterProcedure procedure{*this, proc};
    RequestParameterTypes(proc);

    // Generate or translate a parsed body.
    Ptr<Expr> body;
    if (auto build = source.body.get_if<ProcedureSource::BodyBuilder>()) {
        body = (*build)(proc);
    } else {
        auto parsed = source.body.get<ParsedProcDecl*>();
        if (
            proc->is_instantiation() and
            proc->is_overloaded_operator() and
            not CheckOverloadedOperator(proc, parsed->type->attrs.builtin_operator, source.template_signature)
        ) {
            proc->set_invalid();
            return false;
        }

        ActivateWithParameters(GetParamSpecs(parsed->params()));
        auto ret = proc->return_type();
        auto desired = ret != Type::DeducedTy ? Opt<Type>(ret) : std::nullopt;
        if (not desired and proc->special_proc == SpecialProcedure::Init) desired = proc->props.associated_type;
        body = TranslateExpr(parsed->body.get(), desired);
    }

    // Finish building the body,
    body = BuildProcBody(proc, body);
    if (not RequestPendingProcedureBodies()) proc->set_invalid();
    if (body) proc->finalise(body, curr_proc().locals);
    else proc->set_invalid();

    // Return type deduction may have changed the procedure type; update our
    // template signature if we are a specialisation.
    if (source.template_signature) source.template_signature->runtime_type = proc->proc_type();
    return proc->valid() and not proc->contains_error and bool(proc->body());
}

auto Sema::RequestReference(Decl* d, SLoc loc) -> Ptr<Expr> { // clang-format off
    if (not d->valid()) return nullptr;
    return d->visit(utils::Overloaded{
        [&](FieldDecl*) -> Ptr<Expr> { Unreachable(); },
        [&](ModuleDecl*) -> Ptr<Expr> { Unreachable(); },
        [&](ProcDecl* proc) -> Ptr<Expr> {
            // If this is a nested procedure, mark it as used by the current
            // procedure so we can check that creating a closure for it here
            // is valid once we know what its captures are.
            if (
                proc->parent and
                not proc->name.is_init() and
                proc->special_proc == SpecialProcedure::None and
                not proc->has_captures and
                curr_proc().proc != proc->parent.get()
            ) request_data.capture_users[proc].push_back({curr_proc().proc, loc});

            // Request its body if the return type is deduced.
            if (
                proc->return_type() == Type::DeducedTy and
                not RequestProcedureBody(proc, loc)
            ) return nullptr;

            // If it has captures, check that we can create a closure for it.
            if (
                proc->has_captures and
                not RegisterCapture(proc->parent.get(), proc, loc)
            ) return nullptr;

            // Ok, create the closure.
            return new (*tu) ProcRefExpr(proc, loc);
        },
        [&](ProcTemplateDecl*) -> Ptr<Expr> { return OverloadSetExpr::Create(*tu, d, loc); },
        [&](CompileTimeParamDecl* param) -> Ptr<Expr> {
            auto constant = new (*tu) ConstExpr(*tu, *param->value->value, loc);
            return constant;
        },
        [&](TypeDecl* td) -> Ptr<Expr> { return new (*tu) TypeExpr(td->type, loc); },
        [&](GlobalDecl* g) -> Ptr<Expr> { return new (*tu) GlobalRefExpr(g, loc); },
        [&](CXXMacroExpansionDecl* md) -> Ptr<Expr> { return md->value; },
        [&](EnumeratorDecl* e) -> Ptr<Expr> {
            // Do NOT check if the entire enum is complete here as enumerators are allowed to
            // depend on preceding enumerators!
            if (not RequestEnumerator(e)) return nullptr;

            // Within the definition of an enum, the type of its enumerators is the underlying
            // type of the enum rather than the enum type.
            Type ty = is_contained(request_data.enum_value_stack, e->parent) ? e->parent->elem() : e->parent;
            return MakeConstExpr(
                e,
                eval::RValue(e->value->value(), ty),
                loc
            );
        },
        [&](LocalDecl* local) -> Ptr<Expr> {
            // Check if this variable is declared in a parent procedure and captured it
            // if so; do *not* capture zero-sized variables however since they’ll be deleted
            // entirely anyway.
            if (
                curr_proc().proc != local->parent and
                local->type->memory_size(*tu) != Size()
            ) {
                if (not RegisterCapture(local->parent, local, loc)) return nullptr;
                local->captured = true;
                local->parent->introduces_captures = true;
            }

            return new (*tu) LocalRefExpr(
                local,
                local->category,
                loc
            );
        },
        [&](WithFieldRefDecl* with) -> Ptr<Expr> {
            return BuildMemberAccessExpr(
                with->base,
                with->referenced_field,
                loc
            );
        },
        [&](WithBuiltinFieldRefDecl* with) -> Ptr<Expr> {
            return BuildBuiltinMemberAccessExpr(
                with->referenced_field,
                with->base,
                loc
            );
        },
    });
} // clang-format on

auto Sema::RequestReferenceOrOverloadSet(SLoc loc, LookupResult& res) -> Ptr<Expr> {
    if (res.successful()) return RequestReference(res.decls.front(), loc);
    Assert(res.result == LookupResult::Reason::Ambiguous);
    return OverloadSetExpr::Create(*tu, res.decls, loc);
}

void Sema::RequestRegisterDeclarations(ArrayRef<ParsedStmt*> statements) {
    for (auto statement : statements) {
        Assert(statement->loc.is_valid(), "Registered statements require source locations");
        request_data.file_order.try_emplace(statement->loc.file(ctx), request_data.file_order.size());

        // Skip this if it isn’t a declaration.
        auto parsed = dyn_cast<ParsedDecl>(statement);
        if (not parsed) continue;

        // Unwrap 'export'.
        bool exported = false;
        if (auto exp = dyn_cast<ParsedExportDecl>(parsed)) {
            parsed = exp->decl;
            exported = true;
        }

        // If we’ve already registered a request, skip this.
        if (GetDeclarationRequest(parsed)) continue;

        // Create a new request for it.
        auto request = std::make_unique<DeclarationRequest>();
        request->parsed = parsed;
        request->exported = exported;
        request->context = GetRequestContext();
        request_data.scope_declarations[curr_scope()].push_back(request.get());
        request_data.declarations[{curr_scope(), parsed}] = std::move(request);
    }

    // Publish type identities before requesting any signature or layout.
    for (auto statement : statements) {
        if (auto exp = dyn_cast<ParsedExportDecl>(statement)) statement = exp->decl;
        if (auto structure = dyn_cast<ParsedStructDecl>(statement)) {
            auto request = GetDeclarationRequest(structure);
            if (not request->result) {
                request->result = TranslateStructDeclInitial(structure).get_or_null();
                request_data.declaration_sources[request->result] = request.get();
            }
        }
    }
}

auto Sema::RequestScopeLookup(
    Scope* in_scope,
    DeclNameLoc name,
    LookupHint hint
) -> LookupResult {
    // Search declarations that have been registered for translation in this
    // scope. Requests can append declarations, so use indices here.
    for (usz index = 0;; ++index) {
        auto pending = request_data.scope_declarations.find(in_scope);
        if (
            pending == request_data.scope_declarations.end() or
            index >= pending->second.size()
        ) break;

        auto request = pending->second[index];
        if (request->parsed->name != name.name) continue;

        // Skip variables that are not in scope here.
        if (
            auto var = dyn_cast<ParsedVarDecl>(request->parsed);
            var and
            not IsVariableVisible(*request, name.loc)
        ) continue;

        // Skip variables that are still being translated.
        if (
            isa<ParsedVarDecl>(request->parsed) and
            request->result and
            request_data.results.lookup(Request::Declaration{request}) == Request::State::Evaluating
        ) continue;

        // Request this for translation.
        if (not RequestDeclaration(*request))
            return LookupResult::NotFound(name);
    }

    // Next, search decls that have been requested.
    SmallVector<Decl*> decls;
    for (auto decl : in_scope->decls_with_name(name.name)) {
        if (auto source = GetDeclarationRequest(decl)) {
            if (
                auto var = dyn_cast<ParsedVarDecl>(source->parsed);
                var and
                not IsVariableVisible(*source.get(), name.loc)
            ) continue;
        }
        decls.push_back(decl);
    }

    return LookupResult(decls, name);
}

bool Sema::RequestRecordInitialisers(RecordType* record, SLoc loc) {
    record = cast<RecordType>(record->canonical);
    if (auto result = GetRequestResult(Request::RecordInitialisers{record})) return *result;

    // Request the layout; note that it may already have been computed
    // if this is an imported type.
    if (auto structure = dyn_cast<StructType>(record)) {
        if (not RequestStructLayout(structure)) return false;
        if (not GetDeclarationRequest(structure->decl())) return true;
    }

    return Request(Request::RecordInitialisers{record}, loc);
}

bool Sema::RequestRecordInitialisersImpl(RecordType* record) {
    Assert(record->is_canonical());

    // If this is a struct, make sure it is declared.
    auto structure = dyn_cast<StructType>(record);
    auto source = structure ? GetDeclarationRequest(structure->decl()) : nullptr;

    // Enter the record scope.
    EnterRequestContext context{*this, source ? source->context : GetRequestContext()};
    EnterScope scope{*this, record->scope()};
    auto loc = structure ? structure->decl()->location() : request_data.stack.back().loc;

    // Whether all fields have a default initialiser.
    bool all_default = true;

    // Whether a field has a default initialiser that must be called.
    bool calls_fields = false;

    // Whether there are any initialisers, user-declared or not.
    bool has_initialisers = not record->initialisers().empty();

    // Whether we have an empty initialiser.
    bool has_empty = false;

    // Determine if we need to generate an initialiser procedure, and whether
    // we can even do so in the first place.
    for (auto field : record->scope()->fields()) {
        auto base = field->type->strip_arrays();

        // If the field is itself a record, we need to know if it has initialisers.
        if (
            auto nested = dyn_cast<RecordType>(base);
            nested and not RequestRecordInitialisers(nested, loc)
        ) return false;

        // Update properties.
        all_default &= field->type->can_init_from_no_args();
        calls_fields |= not field->type->can_zero_init();
    }

    // If we have user-declared initialisers, check if any of them can be used as
    // a default initialiser. Include variadic templates that accept an empty argument
    // list, but exclude initialisers that return a different type.
    if (has_initialisers) {
        // Is there a user-declared initialiser that accepts 0 arguments?
        bool accepts_empty = rgs::any_of(record->initialisers(), [](Decl* decl) {
            if (not decl->valid()) return false;
            if (auto proc = dyn_cast<ProcDecl>(decl)) return Callee(proc).argument_count_matches_parameters(0);
            return Callee(cast<ProcTemplateDecl>(decl)).argument_count_matches_parameters(0);
        });

        if (accepts_empty) {
            // There is; attempt to resolve it.
            auto args = TupleExpr::Create(*tu, Type::CallArgListTy, {}, {}, loc);
            bool hard_error = false;
            auto call = ResolveInitialiser(record, args, loc, &hard_error);

            // If the return type of that initialiser is the record type, we can
            // use it as a default initialiser.
            if (call) has_empty = call->proc->return_type() == record;
            else if (hard_error) {
                for (auto& diagnostic : call.error()) ReportDiag(std::move(diagnostic));
                return false;
            }
        }
    }

    // There is no user-declared initialiser; generate initialisers if needed/possible.
    else if (all_default and calls_fields) {
        DefineSpecialProcedures(
            record,
            /*props=*/{},
            /*parsed_deleter=*/nullptr,
            /*parsed_copy=*/nullptr,
            loc,
            /*generate_initialisers=*/true
        );

        has_initialisers = has_empty = true;
    }

    // Finally, update the record bits.
    record->update_initialisation(*tu, has_initialisers, has_empty);
    return true;
}

bool Sema::RequestStructLayout(StructType* s) {
    if (auto result = GetRequestResult(Request::StructLayout{s})) return *result;
    if (s->is_complete()) return true;
    return Request(Request::StructLayout{s}, s->decl()->location());
}

bool Sema::RequestStructLayoutImpl(StructType* s) {
    auto source = GetDeclarationRequest(s->decl());
    Assert(source, "Completing a struct that we did not parse");
    EnterRequestContext context{*this, source->context};
    auto parsed = cast<ParsedStructDecl>(source->parsed);

    // Translate the fields and build the layout.
    EnterScope _{*this, s->scope()};
    RecordLayout::Builder lb{*tu};
    bool ok = true;
    for (auto [i, f] : enumerate(parsed->fields())) {
        auto ty = TranslateType(f->type);
        if (ty and not CheckFieldType(*ty, f->loc)) ty = std::nullopt;
        ok &= ty.has_value();
        auto field = new (*tu) FieldDecl(
            s,
            ty.value_or(Type::VoidTy),
            f->name.str(),
            u32(i),
            f->loc
        );

        lb.add_field(ty.value_or(Type::VoidTy));
        AddDeclToScope(s->scope(), field);
    }

    // Build the layout.
    auto [layout, props] = lb.build(not parsed->initialisers().empty());
    s->finalise(layout);

    // Declare the initialisers.
    auto Cast = [](ParsedProcDecl* p) -> ParsedStmt* { return p; };
    auto initialisers = llvm::to_vector(parsed->initialisers() | vws::transform(Cast));
    RequestRegisterDeclarations(initialisers);

    for (auto init : parsed->initialisers()) {
        auto decl = RequestDeclaration(*GetDeclarationRequest(init).get());
        if (not decl) {
            ok = false;
            continue;
        }

        if (auto proc = dyn_cast<ProcDecl>(decl.get())) {
            request_data.pending_procedure_bodies[source->context.procedure->proc].push_back(proc);
        }
    }

    if (not ok) s->decl()->set_invalid();

    // Define the special procedures, excluding the initialisers.
    DefineSpecialProcedures(
        s,
        props,
        parsed->deleter(),
        parsed->copy_proc(),
        s->decl()->location()
    );

    return ok;
}

auto Sema::RequestTemplateInstantiation(
    ProcTemplateDecl* pattern,
    TemplateSubstitution& info,
    SLoc inst_loc
) -> Ptr<ProcDecl> {
    auto& instantiations = tu->template_instantiations[pattern];
    if (not is_contained(instantiations, info.specialisation))
        instantiations.push_back(info.specialisation);

    // Check if we already know the signature of this.
    auto body = request_data.results.find(Request::ProcedureBody{info.specialisation});
    if (
        body != request_data.results.end() and
        body->second == Request::State::Evaluating and
        info.specialisation->return_type() != Type::DeducedTy
    ) return info.specialisation;

    // If not, the return type is deduced; request the body.
    if (not RequestProcedureBody(info.specialisation, inst_loc)) return {};
    Assert(Type(info.signature->runtime_type) == info.specialisation->proc_type());
    return info.specialisation;
}

auto Sema::RequestTemplateSubstitution(
    Candidate& candidate,
    const CandidateArgumentLists::List& source_args,
    bool is_associated_call,
    bool& hard_error
) -> SubstitutionResult {
    ArrayRef<Expr*> arguments = source_args.values;
    auto proc_template = candidate.callee.cast<ProcTemplateDecl>();
    if (request_data.template_substitution_depth >= MaxRequestDepth) {
        Error(
            proc_template->location(),
            "Exceeded maximum template substitution nesting depth ({})",
            MaxRequestDepth
        );
        return {};
    }

    ++request_data.template_substitution_depth;
    defer { --request_data.template_substitution_depth; };

    SmallVector<TypeLoc> input_types;
    for (auto arg : arguments) input_types.emplace_back(arg->type, arg->location());
    auto source = GetDeclarationRequest(proc_template);
    EnterRequestContext context{*this, source ? source->context : GetRequestContext()};

    // Perform deduction.
    //
    // We need to do this *before* caching, else e.g. 'proc f ($T a, T b)' will
    // be instantiated twice if called with 'f(i32, i16)' and 'f(i32, i8)', which
    // is outright wrong.
    using Deduced = std::pair<u32, TypeLoc>;
    TreeMap<String, Deduced> deduced;

    // First, handle all deduction sites.
    //
    // Note that we might not have any if this is a variadic function with
    // no actual template parameters.
    auto param_types = proc_template->pattern->type->param_types();
    auto it = template_deduction_infos.find(proc_template);
    if (it != template_deduction_infos.end()) {
        for (const auto& [name, indices] : it->second) {
            for (auto i : indices) {
                auto parsed = param_types[i];

                // There might be no corresponding argument if we didn’t
                // pass any variadic arguments to this. In that case, use
                // what ever we deduced earlier, or 'void' if this is the
                // only place where this parameter is deduced.
                if (input_types.size() == i) {
                    Assert(proc_template->has_variadic_param);
                    deduced.try_emplace(name, Deduced{u32(i), {Type::VoidTy, parsed.type->loc}});
                    continue;
                }

                // Otherwise, deduce the parameter from its corresponding argument(s).
                auto ty = DeduceType(parsed.type, input_types[i].ty);
                if (not ty) {
                    return SubstitutionResult::DeductionFailed{
                        name,
                        i
                    };
                }

                if (parsed.variadic) {
                    Assert(proc_template->has_variadic_param);

                    // We already deduced 'ty' from the first variadic argument, so
                    // start at the one after it, if there is one, and make sure we
                    // get the same type for each argument.
                    for (u32 j = u32(param_types.size()), e = u32(input_types.size()); j < e; j++) {
                        auto next = DeduceType(parsed.type, input_types[j].ty);
                        if (not next) {
                            return SubstitutionResult::DeductionFailed{
                                name,
                                j,
                            };
                        }

                        if (next != ty) return SubstitutionResult::DeductionAmbiguous{
                            name,
                            u32(i),
                            u32(j),
                            *ty,
                            *next,
                        };
                    }
                }

                // If the type has not been deduced yet, remember it.
                auto [it, inserted] = deduced.try_emplace(name, Deduced{u32(i), {*ty, parsed.type->loc}});
                if (inserted) continue;

                // Otherwise, check that the deduction result is the same.
                if (it->second.second.ty != ty) {
                    return SubstitutionResult::DeductionAmbiguous{
                        name,
                        it->second.first,
                        u32(i),
                        it->second.second.ty,
                        *ty,
                    };
                }
            }
        }
    }

    // Next, deduce any 'var' parameters.
    SmallVector<Type> deduced_var_parameters;
    for (auto [i, p] : enumerate(param_types)) {
        if (not IsBuiltinVarType(p.type)) continue;
        if (not p.variadic) deduced_var_parameters.push_back(input_types[i].ty);
        else {
            Assert(i == param_types.size() - 1);

            // Build a tuple consisting of the remaining arguments.
            // TODO: Maybe a failure to build the type here should not be a hard error.
            auto types = ArrayRef(input_types).drop_front(i);
            auto names = llvm::to_vector(vws::repeat (String(), types.size()));
            auto ty = TRY(BuildTupleType(types, names, proc_template->location()));
            deduced_var_parameters.push_back(ty);
        }
    }

    // Profile deduced argument types.
    llvm::FoldingSetNodeID id;
    for (const auto& [_, d] : deduced) id.AddPointer(d.second.ty->canonical);
    for (auto ty : deduced_var_parameters) id.AddPointer(ty->canonical);

    // If we don’t have any compile-time parameters, check to see if we’ve
    // substituted this it before.
    Ptr<TemplateSubstitution> cached;
    auto FindCached = [&] {
        llvm::FoldingSetInsertToken unused;
        cached = template_substitutions[proc_template].lookup(id, unused);
    };

    // There are no constants; return this if we’ve cached it before.
    bool has_constants = rgs::any_of(param_types, &ParsedParameter::compile_time);
    if (not has_constants) {
        FindCached();
        if (cached) return cached.get();
    }

    // Otherwise, enter the procedure scope and rebuild the type.
    EnterScope scope{*this, EnterScope::Procedure, proc_template->props.associated_type};

    // Declare all template parameters.
    for (auto [name, d] : deduced) AddDeclToScope(
        scope.get(),
        new (*tu) TemplateTypeParamDecl(name, d.second)
    );

    SubstitutionResult result;
    bool failed_compile_time_arg = false;
    SmallVector<String> constant_arguments;

    // Compile-time parameters are available to later parameters, e.g.
    //
    //     proc f ($t: type, $n: int, data: t[n])
    //
    // is valid. This function checks that the argument is actually convertible
    // to the parameter type and a constant, along with adding it to the procedure
    // scope.
    auto BindParameter = [&](u32 index, ParamTypeData& param) {
        // Non-compile-time parameters can be ignored here.
        if (param_types[index].compile_time) {
            // Fail substitution.
            auto Fail = [&](DiagsVector diags) {
                result.data = SubstitutionResult::ArgumentNotConstant{std::move(diags), index};
                failed_compile_time_arg = true;
                return false;
            };

            // Build an initialiser for the parameter.
            auto arg = arguments[index];
            if (not CheckVariableType(param.type, arg->location())) {
                hard_error = true;
                return false;
            }

            auto conversion = BuildParameterConversionSequence(
                param,
                arg,
                arg->location(),
                is_associated_call and source_args.source_indices[index] == 0
            );

            if (not conversion) return Fail(std::move(conversion.error()));

            // Save the ranking for later.
            candidate.status.get<Candidate::Viable>().ranking.set(
                source_args.source_indices[index],
                conversion->rank()
            );

            // Apply the conversion.
            auto converted = ApplyConversionSequence(arg, *conversion, arg->location());
            if (not converted) return false;

            // Evaluate the argument.
            auto value = RequestEvaluation(converted.get(), false);
            auto name = proc_template->pattern->params()[index]->name;
            if (not value) return Fail(DiagsVector{CreateError(
                arg->location(),
                "Argument to constant parameter '{}' is not a compile-time constant",
                name
            )});

            if (auto profile = value->profile(id); not profile) return Fail(DiagsVector{CreateError(
                arg->location(),
                "Argument to constant parameter '{}' {}",
                name,
                profile.error()
            )});

            constant_arguments.push_back(tu->save(
                value->isa<eval::TreeValue*>()
                    ? "quote{...}"
                    : text::RenderColours(false, value->print(&ctx).str())
            ));
            auto constant = new (*tu) ConstExpr(*tu, std::move(*value), arg->location());

            // Declare it so later parts of the signature can use it.
            auto parsed = proc_template->pattern->params()[index];
            AddDeclToScope(
                scope.get(),
                new (*tu) CompileTimeParamDecl({parsed->name, parsed->loc}, constant)
            );

            // Handle 'with' as well here.
            if (parsed->with_loc.is_valid()) AddEntryToWithStack(
                scope.get(),
                Save(constant),
                parsed->with_loc
            );
        }

        // If this is now cached, avoid rebuilding the signature; returning
        // 'false' here will cause BuildProcSignature() to bail out early.
        if (has_constants and index + 1 == param_types.size()) FindCached();
        return not cached;
    };

    // Build the declaration of the template specialisation.
    auto proc = BuildProcSignature(
        proc_template->pattern,
        proc_template->props,
        deduced_var_parameters,
        proc_template,
        BindParameter
    );

    // We might get a cache hit after resolving compile-time parameters.
    if (cached) {
        result.data = cached.get();
        return result;
    }

    // A compile-time argument was ill-formed.
    if (failed_compile_time_arg) return result;

    // Mark that we’re done substituting.
    for (auto d : scope.get()->decls())
        if (auto param = dyn_cast<TemplateTypeParamDecl>(d))
            param->in_substitution = false;

    // There was some error building the declaration; give up.
    if (not proc) return {};

    // Stringify the argument values since they’re used in the symbol name.
    SmallString<256> display_arguments;
    auto AddArgument = [&](auto value) {
        display_arguments += display_arguments.empty() ? "<" : ", ";
        Format(display_arguments, "{}", value);
    };

    // Print type template parameters in deduction order.
    SmallVector<std::pair<u32, Type>> type_display_arguments;
    for (auto [_, deduction] : deduced)
        type_display_arguments.emplace_back(deduction.first, deduction.second.ty);

    rgs::stable_sort(type_display_arguments, {}, &std::pair<u32, Type>::first);
    for (auto [_, type] : type_display_arguments)
        AddArgument(text::RenderColours(false, type->canonical->print().str()));

    // Print deduced 'var' parameters next.
    for (auto type : deduced_var_parameters)
        AddArgument(text::RenderColours(false, type->canonical->print().str()));

    // Lastly, add compile-time parameters.
    for (auto value : constant_arguments) AddArgument(value);
    if (not display_arguments.empty()) display_arguments += ">";
    proc->template_arguments = tu->save(display_arguments.str());

    // Check the constraint.
    if (auto where = proc_template->pattern->where.get_or_null()) {
        EnterProcedure parameters{*this, proc.get()};
        auto constraint = TRY(TranslateExpr(where));
        if (not MakeCondition(constraint, "where")) return {};
        auto value = RequestEvaluation(constraint);
        if (not value.has_value()) return {};
        if (not value->cast<APInt>().getBoolValue())
            return SubstitutionResult::ConstraintNotSatisfied();
    }

    // Substitution and constraint checking succeeded; store the signature and
    // specialisation for later.
    auto info = new (*tu) TemplateSubstitution(
        id.Intern(tu->allocator()),
        request_data.procedure_sources.find(proc.get())->second.template_signature.get(),
        proc.get()
    );

    template_substitutions[proc_template].insert(info);
    result.data = info;
    return result;
}

auto Sema::RequestUnqualifiedLookup(
    DeclNameLoc name,
    LookupHint hint
) -> LookupResult {
    Assert(not name.name.empty());
    auto in_scope = curr_scope();
    while (in_scope) {
        auto res = RequestScopeLookup(in_scope, name, hint);
        if (res.result != LookupResult::Reason::NotFound) return res;
        else in_scope = in_scope->parent();
    }

    // If we couldn’t find it, try to find it in the open modules.
    if (not tu->open_modules.empty()) {
        Assert(tu->open_modules.size() == 1, "TODO: Lookup involving multiple open modules");
        auto mod = tu->open_modules.front();
        if (auto s = dyn_cast<ImportedSourceModuleDecl>(mod))
            return RequestScopeLookup(&s->exports, name, hint);
        return LookUpCXXName(cast<ImportedClangModuleDecl>(mod), name, hint);
    }

    return LookupResult::NotFound(name);
}
