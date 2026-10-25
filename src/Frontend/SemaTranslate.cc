#include <srcc/Core/Constants.hh>
#include <srcc/Frontend/Parser.hh>
#include <srcc/Frontend/Sema.hh>

using namespace srcc;
using namespace srcc::sema;

#define TRY(...) ({                      \
    auto _res = (__VA_ARGS__);           \
    if (not _res) return utils::Falsy(); \
    *_res;                               \
})

// ============================================================================
//  Translation Driver
// ============================================================================
Sema::~Sema() = default;
Sema::Sema(Context& ctx) : ctx(ctx) {
    using namespace llvm::vfs;
    ImportVFS = llvm::makeIntrusiveRefCnt<OverlayFileSystem>(getRealFileSystem());
    PCHVFS = llvm::makeIntrusiveRefCnt<InMemoryFileSystem>();
    ImportVFS->pushOverlay(PCHVFS);
}

auto Sema::AddParsedModule(ParsedModule::Ptr p, bool translate) -> ParsedModule* {
    if (p == nullptr) return nullptr;

    tu->add_allocator(std::move(p->alloc));
    tu->add_integer_storage(std::move(p->integers));
    tu->add_quoted_tokens(std::move(p->quoted_tokens));

    if (translate) {
        Assert(not started_translating, "Cannot enqueue module during translation");
        return modules_to_translate.emplace_back(std::move(p)).get();
    }

    return extra_modules.emplace_back(std::move(p)).get();
}

auto Sema::Translate(
    const LangOpts& opts,
    SmallVector<ParsedModule::Ptr> modules,
    ArrayRef<std::string> module_search_paths,
    ArrayRef<std::string> clang_include_paths,
    ArrayRef<std::string> clang_options
) -> TranslationUnit::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};

    // Create the TU.
    S.tu = TranslationUnit::Create(
        first->context(),
        opts,
        first->name,
        first->is_module
    );

    // Take ownership of the modules.
    for (auto& m : modules) S.AddParsedModule(std::move(m), true);
    S.search_paths = module_search_paths;
    S.clang_include_paths = clang_include_paths;
    S.clang_options = clang_options;

    // Translate it.
    S.Translate(not opts.no_runtime);
    return std::move(S.tu);
}

void Sema::Translate(bool load_runtime) {
    if (ctx.diags().has_error()) return;
    Assert(not modules_to_translate.empty(), "Need at least 1 module");
    Assert(not started_translating, "Translate() called twice?");
    started_translating = true;

    // Set up scope stacks.
    tu->initialiser_proc->scope = global_scope();
    EnterProcedure _{*this, tu->initialiser_proc};

    // Initialise FFI types.
    auto DeclareBuiltinType = [&](String name, Type type) {
        auto decl = AliasType::Create(*tu, type, name, SLoc())->decl();
        AddDeclToScope(global_scope(), decl);
    };

    DeclareBuiltinType("__srcc_ffi_char", tu->FFICharTy);
    DeclareBuiltinType("__srcc_ffi_wchar", tu->FFIWCharTy);
    DeclareBuiltinType("__srcc_ffi_short", tu->FFIShortTy);
    DeclareBuiltinType("__srcc_ffi_int", tu->FFIIntTy);
    DeclareBuiltinType("__srcc_ffi_long", tu->FFILongTy);
    DeclareBuiltinType("__srcc_ffi_longlong", tu->FFILongLongTy);

    // Translate the preamble first since the runtime and other modules rely
    // on it always being available.
    SmallVector<Stmt*> top_level_stmts;
    if (not tu->lang_opts().no_preamble) {
        auto preamble = AddParsedModule(
            Parser::Parse(
                ctx.create_virtual_file(srcc::sema::preamble::PreambleSource, "__srcc_preamble.src"),
                /*comment_callback=*/{},
                /*is_internal_file=*/true
            )
        );

        if (not preamble or ctx.diags().has_error()) return;
        if (not TranslateStmts(top_level_stmts, preamble->top_level)) {
            ICE(tu->initialiser_proc->location(), "Failed to translate preamble");
            return;
        }
    }

    // Load the runtime.
    if (load_runtime) LoadModule(
        constants::RuntimeModuleName,
        constants::RuntimeModuleName,
        modules_to_translate.front()->program_or_module_loc,
        false,
        false
    );

    // And process other imports.
    for (auto& m : modules_to_translate) {
        for (auto& i : m->imports) {
            LoadModule(
                i.import_name,
                i.linkage_names,
                i.loc,
                i.is_open_import,
                i.is_header_import
            );
        }
    }

    // Bail if we couldn’t load a module.
    if (ctx.diags().has_error()) return;
    llvm::TimeTraceScope _{"[SRCC] Semantic Analysis"};

    // Collect all statements and translate them.
    SmallVector<ParsedStmt*> statements;
    for (auto& p : modules_to_translate) append_range(statements, p->top_level);
    TranslateStmts(top_level_stmts, statements);
    tu->file_scope_block = BlockExpr::Create(*tu, global_scope(), top_level_stmts, SLoc{});

    // File scope block should never be dependent.
    tu->initialiser_proc->finalise(
        BuildProcBody(tu->initialiser_proc, tu->file_scope_block),
        curr_proc().locals
    );

    // Perform any checks that require translation to be complete.
    for (usz i = 0; i < incomplete_structs_in_native_proc_type.size(); ++i) {
        auto [s, loc, is_return] = incomplete_structs_in_native_proc_type[i];
        if (not RequestStructLayout(s)) continue;
        if (s->layout().size() == Size()) DiagnoseZeroSizedTypeInNativeProc(s, loc, is_return);
    }

    // Sanity check.
    Assert(proc_stack.size() == 1);
    Assert(scope_stack.size() == 1);
}

/// Like TranslateStmt(), but checks that the argument is an expression.
auto Sema::TranslateExpr(ParsedStmt* parsed, Opt<Type> desired_type) -> Ptr<Expr> {
    auto stmt = TranslateStmt(parsed, desired_type);
    if (stmt.invalid()) return nullptr;
    if (not isa<Expr>(stmt.get())) return Error(parsed->loc, "Expected expression");
    return cast<Expr>(stmt.get());
}

/// Dispatch to translate a statement.
auto Sema::TranslateStmt(ParsedStmt* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
#define PARSE_TREE_LEAF_TYPE(node)             \
    case K::node: {                            \
        auto ty = TRY(TranslateType(parsed));  \
        return BuildTypeExpr(ty, parsed->loc); \
    }
#define PARSE_TREE_LEAF_NODE(node) \
    case K::node: return LIBBASE_CAT(Translate, node)(cast<LIBBASE_CAT(Parsed, node)>(parsed), desired_type);
#include "srcc/ParseTree.inc"
    }

    Unreachable("Invalid parsed statement kind: {}", +parsed->kind());
}

bool Sema::TranslateStmts(
    SmallVectorImpl<Stmt*>& stmts,
    ArrayRef<ParsedStmt*> parsed,
    Opt<Type> desired_type
) {
    bool ok = true;

    // Register each declaration before we start resolving them.
    RequestRegisterDeclarations(parsed);

    // Request the type/signature of every order-independent declaration.
    for (auto statement : parsed) {
        if (auto exp = dyn_cast<ParsedExportDecl>(statement)) statement = exp->decl;
        if (auto decl = dyn_cast<ParsedDecl>(statement); decl and not isa<ParsedVarDecl>(decl))
            ok &= bool(RequestDeclaration(*GetDeclarationRequest(decl).get()));
    }

    // Having collected out-of-order symbols, now translate all statements for real.
    for (auto p : parsed) {
        if (auto d = dyn_cast<ParsedDecl>(p)) {
            if (auto exp = dyn_cast<ParsedExportDecl>(d)) d = exp->decl;
            auto decl = RequestDeclaration(*GetDeclarationRequest(d).get());
            if (decl and not isa<ParsedVarDecl>(d)) decl = TranslateEntireDecl(decl.get(), d);
            if (decl) stmts.push_back(decl.get());
            if (not decl or not decl.get()->valid()) ok = false;
            continue;
        }

        // Injections here may inject multiple statements.
        if (auto inject = dyn_cast<ParsedInjectExpr>(p)) {
            auto injected = TranslateExpr(inject->injected);
            if (auto i = injected.get_or_null()) ok &= InjectTree(i, desired_type, &stmts);
            else ok = false;
            continue;
        }

        auto stmt = TranslateStmt(p, p == parsed.back() ? desired_type : std::nullopt);
        if (stmt.present()) stmts.push_back(stmt.get());
        else ok = false;
    }

    ok &= RequestPendingProcedureBodies();
    return ok;
}

// ============================================================================
//  Translation of Individual Statements
// ============================================================================
auto Sema::TranslateAssertExpr(ParsedAssertExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    auto cond = TRY(TranslateExpr(parsed->cond));
    Ptr<Expr> msg;
    if (auto m = parsed->message.get_or_null()) msg = TRY(TranslateExpr(m));
    return BuildAssertExpr(cond, msg, parsed->is_compile_time, parsed->loc, parsed->cond_range);
}

auto Sema::TranslateBinaryExpr(ParsedBinaryExpr* expr, Opt<Type> desired_type) -> Ptr<Stmt> {
    auto lhs = TRY(TranslateExpr(expr->lhs, desired_type));
    auto rhs = TRY(TranslateExpr(expr->rhs, desired_type));
    return BuildBinaryExpr(expr->op, lhs, rhs, expr->loc);
}

auto Sema::TranslateBlockExpr(ParsedBlockExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    EnterScope _{*this, parsed->should_push_scope};
    SmallVector<Stmt*> stmts;
    if (not TranslateStmts(stmts, parsed->stmts(), desired_type)) return {};
    return BuildBlockExpr(curr_scope(), stmts, parsed->loc);
}

auto Sema::TranslateBoolLitExpr(ParsedBoolLitExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    return new (*tu) BoolLitExpr(parsed->value, parsed->loc);
}

auto Sema::TranslateBreakContinueExpr(ParsedBreakContinueExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    if (curr_proc().loop_depth == LoopToken(0)) return Error(
        parsed->loc,
        "'%1({}%)' outside loop",
        parsed->is_continue ? "continue"sv : "break"sv
    );

    if (not parsed->is_continue) curr_proc().current_loop_has_break = true;
    return new (*tu) BreakContinueExpr(
        parsed->is_continue,
        curr_proc().loop_depth,
        parsed->loc
    );
}

auto Sema::TranslateCallExpr(ParsedCallExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    Expr* callee{};
    Opt<TupleExpr::ExprAndName> object_param;
    auto TranslateArgs = [&] -> Ptr<TupleExpr> {
        auto e = TranslateTupleExpr(object_param, parsed->args, Type::CallArgListTy);
        return cast_if_present<TupleExpr>(e.get_or_null());
    };

    // The callee may be a builtin.
    if (auto dre = dyn_cast<ParsedDeclRefExpr>(parsed->callee); dre and dre->is_single_ident()) {
        auto bk = BuiltinCallExpr::Parse(dre->names().front().name.str());
        if (bk.has_value()) {
            // 'Dump' does not take an expression, but rather any node.
            if (bk.value() == BuiltinCallExpr::Builtin::Dump) {
                for (auto a : parsed->args->elems()) {
                    ctx.diags().flush();
                    std::println(stderr, "== Parse Tree ==");
                    a.expr()->dump(ctx.use_colours);
                    std::println(stderr, "== AST ==");
                    auto res = TranslateStmt(a.expr());
                    if (auto s = res.get_or_null()) {
                        // FIXME: Introduce some printing mode that also dumps e.g. the TypeDecl
                        // of a TypeExpr if it is a struct.
                        if (auto te = dyn_cast<TypeExpr>(s)) {
                            if (auto rd = dyn_cast<StructType>(te->value)) {
                                rd->decl()->dump(ctx.use_colours);
                                continue;
                            }
                        }

                        s->dump(ctx.use_colours);
                    }
                }

                // Return a void expression.
                return BuildInitialiser(Type::VoidTy, {}, parsed->loc);
            }

            auto args = TRY(TranslateArgs());

            // Allow this once we can declare all builtins in the preamble, which
            // requires compile-time parameters, which requires the new parameter
            // syntax refactor.
            if (args->is_named()) return ICE(
                parsed->loc,
                "Named arguments to builtins are currently not supported"
            );

            return BuildBuiltinCallExpr(bk.value(), args->values(), parsed->loc);
        }
    }

    // The callee may also be an associated procedure, so if this is a member
    // access expression, we need to handle it explicitly here.
    if (auto ma = dyn_cast<ParsedMemberExpr>(parsed->callee)) {
        auto access = TRY(TranslateMemberAccess(ma, true));
        if (access.is_associated_proc_ref()) {
            object_param.emplace(access.base, DeclNameLoc());
            callee = access.callee;
        } else {
            callee = access.base;
        }
    }

    // Otherwise, it’s a regular expression.
    else callee = TRY(TranslateExpr(parsed->callee));

    // Finally, build the call.
    Assert(callee);
    auto args = TRY(TranslateArgs());
    return BuildCallExpr(callee, std::move(args), parsed->loc, object_param.has_value());
}

/// Translate a parsed name to a reference to the declaration it references.
auto Sema::TranslateDeclRefExpr(ParsedDeclRefExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    Assert(not parsed->empty(), "DRE is empty?");

    // Handle the root expression if we have one. This must resolve to a type
    // that contains a scope (such as a struct or enum type).
    Scope* root = nullptr;
    if (auto r = parsed->root().get_or_null()) {
        auto ty = TRY(TranslateType(r));

        // This must be some type that has a scope.
        if (auto e = dyn_cast<EnumType>(ty)) root = e->scope();
        else if (auto s = dyn_cast<StructType>(ty)) root = s->scope();
        else return Error(parsed->loc, "Cannot perform scope access on type '{}'", ty);
    }

    return BuildDeclRefExpr(
        parsed->initial_scope(),
        root,
        parsed->names(),
        parsed->loc,
        desired_type
    );
}

auto Sema::TranslateCopyExpr(ParsedCopyExpr* c, Opt<Type>) -> Ptr<Stmt> {
    auto arg = TRY(TranslateExpr(c->arg));

    // A trivially-copyable type need not be copied explicitly.
    if (arg->type->move_is_copy()) {
        if (not curr_proc().proc->is_instantiation()) Warn(
            c->loc,
            "Redundant explicit '%1(copy%)' of trivially-copyable type '{}'",
            arg->type
        );

        return arg;
    }

    // Copying a temporary is a no-op.
    if (arg->is_rvalue()) {
        if (not curr_proc().proc->is_instantiation()) Warn(
            c->loc,
            "Redundant '%1(copy%)' of temporary value",
            arg->type
        );

        return arg;
    }

    // If the type is not copyable, this is an error.
    if (not arg->type->can_copy()) {
        Error(c->loc, "Type '{}' is not copyable", arg->type);
        Remark(
            "If you want to be able to copy it, implement '%1(copy%)' for\f'{}'",
            arg->type->strip_arrays_and_optionals()
        );
        return arg;
    }

    return new (*tu) CastExpr(arg->type, CastExpr::LValueCopy, arg, c->loc);
}

/// Perform initial processing of a decl so it can be used by the rest
/// of the code. This only handles order-independent decls.
auto Sema::TranslateDeclInitial(ParsedDecl* d) -> std::optional<Ptr<Decl>> {
    // Unwrap exports.
    if (auto exp = dyn_cast<ParsedExportDecl>(d)) {
        auto decl = TranslateDeclInitial(exp->decl);
        if (decl and decl->present()) {
            AddDeclToScope(&tu->exports, decl->get());

            // If this declaration has linkage, adjust it now.
            if (auto object = dyn_cast<ObjectDecl>(decl->get()))
                object->linkage //
                    = object->linkage == Linkage::Imported
                        ? Linkage::Reexported
                        : Linkage::Exported;
        }
        return decl;
    }

    if (auto proc = dyn_cast<ParsedProcDecl>(d)) return TranslateProcDeclInitial(proc);
    if (auto s = dyn_cast<ParsedStructDecl>(d)) return TranslateStructDeclInitial(s);
    if (auto e = dyn_cast<ParsedEnumDecl>(d)) return TranslateEnumDeclInitial(e);
    return std::nullopt;
}

auto Sema::TranslateDeleteExpr(ParsedDeleteExpr* expr, Opt<Type>) -> Ptr<Stmt> {
    auto arg = TRY(TranslateExpr(expr->expr));
    return MaybeBuildDeleteExpr(arg, false, expr->loc);
}

auto Sema::TranslateDeferStmt(ParsedDeferStmt* stmt, Opt<Type>) -> Ptr<Stmt> {
    auto body = TRY(TranslateStmt(stmt->body));
    return new (*tu) DeferStmt(body, stmt->loc);
}

/// Translate the body of a declaration.
auto Sema::TranslateEntireDecl(Decl* d, ParsedDecl* parsed) -> Ptr<Decl> {
    // Unwrap exports; pass along the actual parsed decl for this.
    if (auto exp = dyn_cast<ParsedExportDecl>(parsed))
        return TranslateEntireDecl(d, exp->decl);

    // Ignore this if there was a problem w/ the procedure type. Also,
    // if this is a template, there is nothing more to be done here.
    if (auto proc = dyn_cast<ParsedProcDecl>(parsed)) {
        if (not d) return nullptr;
        if (isa<ProcTemplateDecl>(d)) return d;
        auto procedure = cast<ProcDecl>(d);
        if (procedure->will_have_body) RequestProcedureBody(procedure, proc->loc);
        else {
            RequestParameterTypes(procedure);
            RequestPendingProcedureBodies(procedure);
        }

        return procedure;
    }

    // Complete struct declarations.
    if (isa<ParsedStructDecl>(parsed)) {
        if (not d) return nullptr;
        auto record = cast<StructType>(cast<TypeDecl>(d)->type);
        RequestRecordInitialisers(record, d->location());
        return d;
    }

    // Complete enum definitions.
    if (isa<ParsedEnumDecl>(parsed)) {
        if (not d) return nullptr;
        auto ty = cast<EnumType>(cast<TypeDecl>(d)->type);
        if (not ty->is_complete()) RequestEnumerators(ty);
        return d;
    }

    // No special handling for anything else.
    auto res = TranslateStmt(parsed);
    if (res.invalid()) return nullptr;
    return cast<Decl>(res.get());
}

auto Sema::TranslateEnumDecl(ParsedEnumDecl* e, Opt<Type>) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

auto Sema::TranslateEnumDeclInitial(ParsedEnumDecl* e) -> Ptr<TypeDecl> {
    Type underlying_type = Type::IntTy;
    if (auto parsed_ty = e->underlying_type.get_or_null()) {
        if (auto ty = TranslateType(parsed_ty)) {
            if (ty->is_integer_or_bool()) underlying_type = *ty;
            else Error(e->loc, "Underlying type of enum must be an integer type, was '{}'", *ty);
        }
    }

    // Create and declare the enum type.
    auto enum_type = new (*tu) EnumType(
        *tu,
        tu->create_scope<BlockScope>(curr_scope()),
        e->name,
        underlying_type,
        e->loc
    );

    AddDeclToScope(curr_scope(), enum_type->decl());

    // Declare the enumerators, but don’t compute their values yet.
    EnterScope scope{*this, enum_type->scope()};
    for (auto [index, enumerator] : enumerate(e->enumerators())) {
        auto decl = new (*tu) EnumeratorDecl(
            enum_type,
            u32(index),
            enumerator.name,
            enumerator.loc
        );

        AddDeclToScope(scope.get(), decl);
    }

    return enum_type->decl();
}

auto Sema::TranslateExportDecl(ParsedExportDecl*, Opt<Type>) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

auto Sema::TranslateEmptyStmt(ParsedEmptyStmt* parsed, Opt<Type>) -> Ptr<Stmt> {
    return new (*tu) EmptyStmt(parsed->loc);
}

auto Sema::TranslateEvalExpr(ParsedEvalExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    EnterScope _{*this};
    Stmt* arg{};

    {
        tempset inside_eval = true;
        arg = TRY(TranslateStmt(parsed->expr));
    }

    return BuildEvalExpr(arg, parsed->loc);
}

auto Sema::TranslateFieldDecl(ParsedFieldDecl*, Opt<Type>) -> Decl* {
    Unreachable("Handled as part of StructDecl translation");
}

auto Sema::TranslateForStmt(ParsedForStmt* parsed, Opt<Type>) -> Ptr<Stmt> {
    EnterScope _{*this};
    EnterLoop loop{*this};

    // The number of variables must be less than or equal to the number of ranges.
    if (parsed->vars().size() > parsed->ranges().size()) return Error(
        parsed->loc,
        "'%1(for%)' loop declares more variables than it has ranges ({} vs {})",
        parsed->vars().size(),
        parsed->ranges().size()
    );

    // The types of the variables depend on the ranges, so translate them first.
    SmallVector<Expr*> ranges;
    for (auto r : parsed->ranges())
        if (auto e = TranslateExpr(r).get_or_null())
            ranges.push_back(e);

    // Give up if something went wrong.
    if (ranges.size() != parsed->ranges().size()) return {};

    // Make sure the ranges are something we can actually iterate over.
    for (auto& r : ranges) {
        if (isa<RangeType, SliceType>(r->type)) {
            r = LValueToRValue(r);
        } else if (isa<ArrayType>(r->type)) {
            r = MaterialiseTemporary(r);
        } else {
            return Error(
                r->location(),
                "Invalid type '{}' for range of '%1(for%)' loop",
                r->type
            );
        }
    }

    // Declare the enumerator variable if there is one.
    Ptr<LocalDecl> enum_var;
    if (parsed->has_enumerator()) {
        enum_var = MakeLocal(
            Type::IntTy,
            Expr::RValue,
            parsed->enum_name,
            parsed->enum_loc
        );
    }

    // Create the loop variables; they have no initialisers since they’re really
    // just values bound to the elements of the ranges.
    SmallVector<LocalDecl*> vars;
    for (auto [v, r] : zip(parsed->vars(), ranges)) {
        auto MakeVar = [&](Type ty, ValueCategory cat) {
            auto var = MakeLocal(
                ty,
                cat,
                v.first,
                v.second
            );

            vars.push_back(var);
        };

        r->type->visit(utils::Overloaded{
            [&](auto*) { Unreachable(); },
            [&](ArrayType* ty) { MakeVar(ty->elem(), Expr::LValue(r->is_immutable_lvalue())); },
            [&](SliceType* ty) { MakeVar(ty->elem(), Expr::LValue(ty->is_immutable())); },
            [&](RangeType* ty) { MakeVar(ty->elem(), Expr::RValue); },
        });
    }

    // Now that we have the variables, translate the loop body.
    auto body = TRY(TranslateStmt(parsed->body));
    return ForStmt::Create(*tu, loop.token(), enum_var, vars, ranges, body, parsed->loc);
}

auto Sema::TranslateIfExpr(ParsedIfExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    EnterScope _{*this, not parsed->is_static};
    auto cond = TRY(TranslateExpr(parsed->cond));
    if (parsed->is_static) return BuildStaticIfExpr(cond, parsed->then, parsed->else_, parsed->loc);
    auto then = TRY(TranslateStmt(parsed->then, desired_type));
    Ptr else_ = parsed->else_ ? TRY(TranslateStmt(parsed->else_.get(), desired_type)) : nullptr;
    return BuildIfExpr(cond, then, else_, parsed->loc);
}

auto Sema::TranslateInjectExpr(ParsedInjectExpr* parsed, Opt<Type> ty) -> Ptr<Stmt> {
    auto injected = TRY(TranslateExpr(parsed->injected));
    Stmt* ptr{};
    if (not InjectTree(injected, ty, &ptr)) return {};
    return ptr;
}

auto Sema::TranslateIntLitExpr(ParsedIntLitExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    // Determine the type of this.
    //
    // If we have a desired type, and the value fits in that type,
    // then the type of the literal is that type. Otherwise, if the
    // value fits in an 'int', its type is 'int'. If not, the type
    // is the smallest power of two large enough to store the value.
    auto val = parsed->storage.value();
    Type ty = TRY([&] -> Opt<Type> {
        if (desired_type and desired_type->is_integer() and IntegerLiteralFitsInType(val, *desired_type, false))
            return desired_type;
        if (IntegerLiteralFitsInType(val, Type::IntTy, false))
            return Type::IntTy;

        auto bits = Size::Bits(llvm::PowerOf2Ceil(val.getActiveBits()));
        if (bits <= IntType::MaxBits)
            return IntType::Get(*tu, bits);

        // Print and colour the type names manually here since we can’t
        // even create a type this large properly...
        Error(parsed->loc, "Sorry, we can’t compile a number that big :(");
        Note(
            parsed->loc,
            "The maximum supported integer type is {}, "
            "which is smaller than an %6(i{:i}%), which would "
            "be required to store a value of {}",
            IntType::Get(*tu, IntType::MaxBits),
            bits,
            parsed->storage.str(false) // Parsed literals are unsigned.
        );
        return {};
    }());

    auto desired_bits = ty->bit_width(*tu);
    auto stored_bits = Size::Bits(val.getBitWidth());
    auto storage = desired_bits == stored_bits
        ? parsed->storage
        : tu->store_int(val.zextOrTrunc(unsigned(desired_bits.bits())));

    return new (*tu) IntLitExpr(
        ty,
        storage,
        parsed->loc
    );
}

auto Sema::TranslateLoopExpr(ParsedLoopExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    EnterScope _{*this};
    EnterLoop loop{*this};
    Ptr<Stmt> body;
    if (auto b = parsed->body.get_or_null()) body = TRY(TranslateStmt(b));
    return new (*tu) LoopExpr(
        loop.token(),
        body,
        curr_proc().current_loop_has_break ? Type::VoidTy : Type::NoReturnTy,
        parsed->loc
    );
}

auto Sema::TranslateMatchExpr(ParsedMatchExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    EnterScope _{*this};
    SmallVector<MatchCase> cases;
    bool ok = true;

    // Give up here if translating the controlling expression fails since
    // the semantics of a 'match' w/ and w/o a controlling expression are
    // completely different.
    Ptr<Expr> control_expr = parsed->control_expr()
        ? TRY(TranslateExpr(parsed->control_expr().get()))
        : nullptr;

    // Translate the type if there is one.
    auto ty = parsed->declared_type()
        ? TranslateType(parsed->declared_type().get(), Type::VoidTy)
        : Type::DeducedTy;

    for (auto c : parsed->cases()) {
        auto body = TranslateStmt(c.body, desired_type);
        if (not body) ok = false;

        // The wildcard pattern isn’t an expression and requires special handling.
        if (
            auto dre = dyn_cast<ParsedDeclRefExpr>(c.cond);
            dre and
            dre->is_single_ident() and
            dre->names().front().name.str() == "_"
        ) {
            if (body) cases.emplace_back(MatchCase::Pattern::Wildcard(), body.get(), dre->loc);
            continue;
        }

        auto cond = TranslateExpr(c.cond);
        if (cond and body) {
            cases.emplace_back(
                cond.get(),
                body.get(),
                c.cond->loc
            );
        }
    }

    if (not ok) return {};
    return BuildMatchExpr(control_expr, ty, cases, parsed->loc);
}

auto Sema::TranslateMemberAccess(
    ParsedMemberExpr* parsed,
    bool is_call
) -> std::optional<MemberAccess> {
    // Translate the base access.
    auto base = TRY(TranslateExpr(parsed->base));
    auto ty = base->type->strip_pointers_and_optionals();
    auto TryAssociatedLookup = [&](auto FallbackError) -> std::optional<MemberAccess> {
        auto res = RequestUnqualifiedLookup(
            DeclNameLoc{parsed->member, parsed->loc},
            LookupHint::Any
        );

        if (not res.successful_or_ambiguous())
            return std::invoke(FallbackError);

        // Filter out all declarations whose associated type doesn't match.
        llvm::erase_if(res.decls, [&](Decl* d) {
            auto proc = dyn_cast<ProcDecl>(d);
            return not proc or proc->props.associated_type != ty;
        });

        // If this leaves us with nothing, fail.
        if (res.decls.empty())
            return std::invoke(FallbackError);

        Expr* callee = TRY(RequestReferenceOrOverloadSet(parsed->loc, res));
        return MemberAccess::AssociatedProcRef(base, callee);
    };

    // Helper to call 'TryAssociatedLookup' w/ a lambda that issues a diagnostic.
#   define TRY_ASSOCIATED(...) TryAssociatedLookup([&] { return Error(__VA_ARGS__); });

    // Struct member access.
    if (auto s = dyn_cast<RecordType>(ty)) {
        if (not s->is_complete()) return TRY_ASSOCIATED(
            parsed->loc,
            "Member access on incomplete type '{}'",
            ty
        );

        auto field = RequestScopeLookup(
            s->scope(),
            DeclNameLoc{parsed->member, parsed->loc},
            LookupHint::Any
        );

        switch (field.result) {
            using enum LookupResult::Reason;
            case Success: break;
            case Ambiguous:
            case FailedToImport:
            case NonScopeInPath:
                Unreachable();

            case NotFound: return TRY_ASSOCIATED(
                parsed->loc,
                "Struct '{}' has no member named '{}'",
                ty,
                parsed->member
            );
        }

        auto access = TRY(BuildMemberAccessExpr(
            base,
            cast<FieldDecl>(field.decls.front()),
            parsed->loc
        ));

        return MemberAccess::Simple(access);
    }

    // Member access on builtin types.
    auto members = BuiltinMemberAccessExpr::GetAllBuiltinMembersOf(ty);
    if (members.empty()) return TRY_ASSOCIATED(
        parsed->loc,
        "Cannot perform member access on type '{}'",
        ty
    );

    auto it = rgs::find(members, parsed->member, &BuiltinMemberAccessExpr::BuiltinMember::name);
    if (it == members.end()) return TRY_ASSOCIATED(
        parsed->loc,
        "'{}' has no member named '{}'",
        ty,
        parsed->member
    );

    auto access = TRY(BuildBuiltinMemberAccessExpr(it->kind, base, parsed->loc));
    return MemberAccess::Simple(access);
#   undef TRY_ASSOCIATED
}

auto Sema::TranslateMemberExpr(ParsedMemberExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    auto access = TRY(TranslateMemberAccess(parsed, false));
    Assert(not access.is_associated_proc_ref());
    return access.base;
}

auto Sema::TranslateNilExpr(ParsedNilExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    return new (*tu) NilExpr(parsed->loc);
}

auto Sema::TranslateQuoteExpr(ParsedQuoteExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    // Translate all unquotes.
    SmallVector<Expr*> unquotes;
    for (auto u : parsed->unquotes()) {
        EnterScope _{*this}; // Prevent anything from leaking out of the unquote.
        auto tree = TRY(TranslateExpr(u->arg));
        if (not MakeRValue(Type::TreeTy, tree, [&]{
            Error(
                tree->location(),
                "Cannot inject value of type '{}'",
                tree->type
            );
        })) return {};

        unquotes.push_back(tree);
    }

    return QuoteExpr::Create(*tu, parsed, unquotes, parsed->loc);
}

auto Sema::TranslateRawTupleExpr(ParsedRawTupleExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    auto type = parsed->type ? TranslateType(parsed->type.get()) : desired_type;
    if (not type or type == Type::CallArgListTy) {
        if (not parsed->type) Error(parsed->loc, "Type of raw tuple cannot be inferred");
        return {};
    }

    if (not CheckVariableType(*type, parsed->loc)) return {};
    auto tuple = TRY(TranslateTupleExpr(std::nullopt, parsed->tuple, Type::CallArgListTy));
    return BuildRawInitialiser(*type, tuple, parsed->loc);
}

auto Sema::TranslateThisExpr(ParsedThisExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    // Find the associated type.
    auto associated_type = curr_scope()->associated_type();
    if (not associated_type) return Error(
        parsed->loc,
        "'%1({}%)' cannot be used outside of a member function",
        parsed->spelling()
    );

    // If this is 'This', then it is that type.
    if (parsed->is_type)
        return new (*tu) TypeExpr(*associated_type, parsed->loc);

    // Otherwise, perform name lookup for 'this'.
    return BuildDeclRefExpr(DeclNameLoc{String("this"), parsed->loc}, parsed->loc);
}

auto Sema::TranslateTupleExpr(ParsedTupleExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    // In a type context, a tuple expression is treated as a tuple type.
    if (desired_type == Type::TypeTy) {
        auto type = TRY(TranslateType(parsed));
        return BuildTypeExpr(type, parsed->loc);
    }

    if (desired_type and isa<RecordType>(*desired_type)) {
        auto args = TRY(TranslateTupleExpr(std::nullopt, parsed, Type::CallArgListTy));
        return BuildInitialiser(*desired_type, args, parsed->loc);
    }

    return TranslateTupleExpr(std::nullopt, parsed, desired_type);
}

auto Sema::TranslateTupleExpr(
    Opt<TupleExpr::ExprAndName> object_param,
    ParsedTupleExpr* parsed,
    Opt<Type> desired_type
) -> Ptr<Expr> {
    SmallVector<Expr*> exprs;
    SmallVector<DeclNameLoc> names;
    if (object_param.has_value()) {
        exprs.push_back(object_param->expr);
        names.push_back(object_param->name);
    }

    // Pass along the desired type if this is a single parenthesised expression.
    auto desired_elem_ty = desired_type;
    if (
        desired_type == Type::CallArgListTy or
        object_param.has_value() or
        not parsed->is_paren_expr()
    ) desired_elem_ty = std::nullopt;

    // Translate the elements.
    bool ok = true;
    for (auto elem : parsed->elems()) {
        auto e = TranslateExpr(elem.expr(), desired_elem_ty).get_or_null();
        if (not e) {
            ok = false;
            continue;
        }

        // Not a splat.
        if (elem.splat() == ParsedTupleElem::Splat::None) {
            exprs.push_back(e);
            names.push_back(elem.name);
            continue;
        }

        // Handle splats.
        bool named = elem.splat() == ParsedTupleElem::Splat::Named;

        // Literal tuples can be unpacked as is. Don’t allow doing so for
        // tuples of struct type as the struct members may not match the
        // tuple elements if we’re calling a constructor.
        if (auto te = dyn_cast<TupleExpr>(e); te and te->is_tuple()) {
            for (auto [expr, name] : te->elems()) {
                exprs.push_back(expr);
                if (named) names.push_back(name);
                else names.push_back(DeclNameLoc());
            }
            continue;
        }

        // We can only splat aggregates; disallow builtin aggregates
        // (e.g. closures, slices), since splatting them rarely makes
        // sense.
        if (not isa<ArrayType, RecordType>(e->type)) {
            Error(e->location(), "Cannot splat expression of type '{}'", e->type);
            ok = false;
            continue;
        }

        // Records generally need a member access here.
        if (auto r = dyn_cast<RecordType>(e->type)) {
            auto saved = Save(e);
            for (auto f : r->scope()->fields()) {
                auto ma = BuildMemberAccessExpr(saved, f, e->location());
                if (not ma) {
                    ok = false;
                    continue;
                }

                exprs.push_back(ma.get());
                if (named) names.push_back({f->name, f->location()});
                else names.push_back(DeclNameLoc());
            }
            continue;
        }

        // Check that this isn’t too large; else the compiler will hang trying
        // to splat a 1-billion-element array.
        auto a = cast<ArrayType>(e->type);
        auto dim = a->dimension();
        auto limit = i64(tu->lang_opts().array_splat_limit);
        if (limit != 0 and dim > limit) {
            Error(e->location(), "Array dimension {} exceeds maximum splat size of {}", dim, limit);
            Note(e->location(), "Pass '-farray-splat-limit=<value>' to increase the limit");
            continue;
        }

        // Add each array element.
        auto saved = Save(e);
        for (i64 i = 0; i < dim; i++) {
            auto idx = IntLitExpr::Create(*tu, Type::IntTy, i, e->location());
            auto el = BuildBinaryExpr(Tk::LBrack, saved, idx, e->location());
            if (not el) ok = false;
            else {
                exprs.push_back(el.get());
                names.push_back(DeclNameLoc());
            }
        }
    }

    if (not ok) return {};
    return BuildTuple(exprs, desired_type, names, not parsed->has_trailing_comma, parsed->loc);
}

auto Sema::TranslateVarDecl(ParsedVarDecl* parsed, Opt<Type>) -> Decl* {
    if (parsed->is_static) Todo();
    Assert(not parsed->with_loc.is_valid(), "'with' should currently only be parsed on param decls");

    auto ty = TranslateType(parsed->type);
    auto vc = Expr::LValue(parsed->immutable);

    // If we're at the global scope, make this a global variable.
    AnyVarDecl decl;
    if (curr_scope() == global_scope()) {
        auto g = new (*tu) GlobalDecl(
            tu.get(),
            nullptr,
            ty.value_or(Type::VoidTy),
            parsed->immutable,
            parsed->name,
            Linkage::Internal,
            Mangling::Source,
            parsed->loc
        );

        g->mangling_number = next_global_mangling_number++;
        AddDeclToScope(curr_scope(), g);
        decl = g;
    } else {
        decl = MakeLocal(
            ty.value_or(Type::VoidTy),
            vc,
            parsed->name.str(),
            parsed->loc
        );
    }

    // Complete the request for this declaration.
    if (auto request = GetDeclarationRequest(parsed)) {
        request->result = decl.decl();
        request_data.declaration_sources[decl.decl()] = request.get();
    }

    // Don’t even bother with the initialiser if the type is ill-formed.
    if (not ty) return decl.set_invalid();

    // Translate the initialiser.
    Ptr<Expr> init;
    if (auto val = parsed->init.get_or_null()) {
        init = TranslateExpr(val, decl.type() != Type::DeducedTy ? Opt<Type>(decl.type()) : std::nullopt);
        if (init.invalid()) return decl.set_invalid();
    }

    // And attempt to add an initialiser irrespective of whether we
    // parsed one; this will check if default initialisation is valid
    // if need be.
    AddInitialiserToDecl(decl, init);
    return decl.decl();
}

auto Sema::TranslateProcDecl(ParsedProcDecl*, Opt<Type>) -> Decl* {
    Unreachable("Should not be translated in TranslateStmt()");
}

/// Perform initial type checking on a procedure, enough to enable calls
/// to it to be translated, but without touching its body, if there is one.
auto Sema::TranslateProcDeclInitial(ParsedProcDecl* parsed) -> Ptr<Decl> {
    // Diagnose invalid combinations of attributes.
    auto& attrs = parsed->type->attrs;

    // 'extern' implies 'nomangle' and disallows 'inline'.
    if (attrs.extern_) {
        if (attrs.nomangle) Error(
            parsed->loc,
            "'%1(extern%)' already implies '%1(nomangle%)'"
        );

        if (attrs.inline_) Error(
            parsed->loc,
            "'%1(extern%)' procedure cannot be '%1(inline%)'"
        );

        attrs.nomangle = true;
        attrs.inline_ = false;
    }

    // So does 'native'.
    else if (attrs.native) {
        if (attrs.nomangle) Error(
            parsed->loc,
            "'%1(native%)' already implies '%1(nomangle%)'"
        );

        attrs.nomangle = true;
    }

    // '__srcc_builtin_op__' is obviously only allowed on operators.
    if (attrs.builtin_operator and not parsed->name.is_operator_name()) {
        attrs.builtin_operator = false;
        Error(parsed->loc, "invalid use of '%1(__srcc_builtin_op__%)'");
    }

    // Variadic templates cannot have C varargs.
    bool has_variadic_param = parsed->type->has_variadic_param();
    if (has_variadic_param and attrs.c_varargs) {
        attrs.c_varargs = false;
        Error(parsed->loc, "Variadic function cannot be '%1(varargs%)'");
    }

    // Check if this is a template.
    TemplateParamDeductionInfo deduction_info;
    auto IsTemplate = [&] {
        for (auto [i, p] : enumerate(parsed->params())) {
            p->traverse([&](ParsedTemplateType* t) {
                deduction_info[t->name].insert(u32(i));
            });
        }


        if (not deduction_info.empty()) return true;

        // A function with a 'var' parameter is also a template.
        return rgs::any_of(parsed->type->param_types(), [&](auto& p) {
            return IsBuiltinVarType(p.type);
        });
    };

    // Determine inherited properties.
    InheritedProcedureProperties props;
    props.always_inline = attrs.inline_;
    if (parsed->name.is_init()) props.associated_type = curr_scope()->associated_type();

    // Determine its mangling number.
    //
    // This number is appended to the mangled name of a procedure and is
    // used to avoid mangled name collisions. Consider:
    //
    //     {
    //         proc a {}
    //     }
    //     {
    //         proc a {}
    //     }
    //
    // This is not ill-formed since the procedures are in different scopes,
    // and thus, no possible overload set is ever going to contain both of
    // them. Here, we need to make sure that they end up with different names
    // in the IR.
    if (auto request = GetDeclarationRequest(parsed)) {
        props.mangling_number = request->mangling;
    } else {
        props.mangling_number = RequiresManglingNumber(attrs)
            ? next_proc_mangling_number++
            : ManglingNumber::None;
    }

    // TODO: Check if the signature contains a compile-time only type (e.g. 'tree')/
    // TODO: When translating composite types, compute whether they’re compile-time only.
    props.is_compile_time_only = inside_eval;

    // Resolve the associated type.
    if (auto a = parsed->associated_type.get_or_null()) {
        props.associated_type = TranslateType(a);

        // We auto-dereference pointers an optionals, so it's a bit problematic if
        // the associated type is one of those.
        if (isa_and_nonnull<PtrType, OptionalType>(props.associated_type)) {
            SmallString<32> quals;
            Type ty = *props.associated_type;
            for (;;) {
                if (isa<PtrType>(ty)) quals += "^";
                else if (isa<OptionalType>(ty)) quals += "?";
                else break;
                ty = cast<SingleElementTypeBase>(ty)->elem();
            }

            Error(
                a->loc,
                "{} type '{}' cannot be used as an associated type",
                isa<PtrType>(*props.associated_type) ? "Pointer" : "Optional",
                *props.associated_type
            );

            Remark("Write '{}%1(::%)' instead and then use '%1(This{}%)' as the parameter type", ty, quals);
        }
    }

    // If this is a template, we can’t do much right now.
    Decl* decl{};
    if (IsTemplate()) {
        auto temp = ProcTemplateDecl::Create(
            *tu,
            parsed,
            curr_scope() == global_scope() ? nullptr : curr_proc().proc,
            props,
            has_variadic_param
        );

        if (parsed->name.is_init()) {
            temp->parent = cast<StructType>(*props.associated_type)->parent;
        }

        decl = temp;
        if (auto request = GetDeclarationRequest(parsed))
            request_data.declaration_sources[temp] = request.get();
        if (not deduction_info.empty())
            template_deduction_infos.try_emplace(temp, std::move(deduction_info));
    }

    // Otherwise, convert its signature now.
    else {
        Assert(parsed->where.invalid(), "TODO: Constraint on non-template");
        EnterScope scope{*this, EnterScope::Procedure, props.associated_type};
        auto proc = BuildProcSignature(parsed, props);
        if (not proc) return nullptr;
        decl = proc.get();

        // If this is an operator, check that we can overload it.
        if (
            proc->is_overloaded_operator() and
            not CheckOverloadedOperator(proc.get(), attrs.builtin_operator)
        ) proc->set_invalid();
    }

    // Currently, only the last parameter is allowed to be variadic.
    if (has_variadic_param) {
        auto params_except_last = parsed->type->param_types().drop_back();
        auto it = rgs::find_if(params_except_last, &ParsedParameter::variadic);
        if (it != params_except_last.end()) {
            decl->set_invalid();
            Error(
                parsed->params()[usz(it - params_except_last.begin())]->loc,
                "Only the last parameter can be variadic"
            );
        }
    }

    // Variadic parameters may not be 'inout' or 'out' since passing
    // things by reference gets complicated if they’re somehow also
    // supposed to be inside a slice or tuple.
    for (auto p : parsed->type->param_types()) {
        if (p.variadic and (p.intent == Intent::Inout or p.intent == Intent::Out)) {
            Error(p.type->loc, "Variadic parameter cannot have intent '%1({}%)'", p.intent);
            decl->set_invalid();
        }
    }

    AddDeclToScope(curr_scope(), decl);
    return decl;
}

auto Sema::TranslateStructDecl(ParsedStructDecl*, Opt<Type>) -> Decl* {
    Unreachable("Should not be translated normally");
}

auto Sema::TranslateStructDeclInitial(ParsedStructDecl* parsed) -> Ptr<TypeDecl> {
    auto sc = tu->create_scope<RecordScope>(curr_scope());
    auto ty = StructType::Create(
        *tu,
        sc,
        parsed->name.str(),
        parsed->loc
    );

    AddDeclToScope(curr_scope(), ty->decl());
    ty->parent = curr_scope() == global_scope() ? nullptr : curr_proc().proc;
    return ty->decl();
}

/// Translate a string literal.
auto Sema::TranslateStrLitExpr(ParsedStrLitExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    return StrLitExpr::Create(*tu, parsed->value, parsed->loc);
}

/// Translate a return expression.
auto Sema::TranslateReturnExpr(ParsedReturnExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    Ptr<Expr> ret_val;

    // Translate the value.
    if (parsed->value.present()) {
        auto proc = curr_proc().proc;
        auto ret = proc->return_type();
        auto desired = ret != Type::DeducedTy ? Opt<Type>(ret) : std::nullopt;

        // Inside of an initialiser, the desired return type, if unset, is the
        // associated type of the procedure (i.e. the enclosing struct).
        if (not desired and proc->special_proc == SpecialProcedure::Init) desired = proc->props.associated_type;
        ret_val = TranslateExpr(parsed->value.get(), desired);
        if (not ret_val) return {};
    }

    return BuildReturnExpr(ret_val.get_or_null(), parsed->loc, false);
}

auto Sema::TranslateUnaryExpr(ParsedUnaryExpr* parsed, Opt<Type> desired_type) -> Ptr<Stmt> {
    auto arg = TRY(TranslateExpr(parsed->arg, desired_type));
    return BuildUnaryExpr(parsed->op, arg, parsed->postfix, parsed->loc);
}

auto Sema::TranslateUnquoteExpr(ParsedUnquoteExpr*, Opt<Type>) -> Ptr<Stmt> {
    Unreachable("Never translated as an expression");
}

auto Sema::TranslateWhileStmt(ParsedWhileStmt* parsed, Opt<Type>) -> Ptr<Stmt> {
    EnterScope _{*this};
    EnterLoop loop{*this};
    auto cond = TRY(TranslateExpr(parsed->cond));
    auto body = TRY(TranslateStmt(parsed->body));
    if (not MakeCondition(cond, "while")) return {};
    return new (*tu) WhileStmt(loop.token(), cond, body, parsed->loc);
}

auto Sema::TranslateWithExpr(ParsedWithExpr* parsed, Opt<Type>) -> Ptr<Stmt> {
    EnterScope _{*this};
    auto expr = Save(TRY(TranslateExpr(parsed->expr)));
    AddEntryToWithStack(curr_scope(), expr, parsed->loc);
    auto body = TRY(TranslateStmt(parsed->body));
    return new (*tu) WithExpr(expr, body, parsed->loc);
}

// ============================================================================
//  Translation of Types
// ============================================================================
auto Sema::TranslateArrayType(ParsedBinaryExpr* parsed) -> Opt<Type> {
    Assert(parsed->op == Tk::LBrack);
    auto elem = TRY(TranslateType(parsed->lhs));
    auto size = TRY(TranslateExpr(parsed->rhs));
    return BuildArrayType({elem, parsed->loc}, size);
}

auto Sema::TranslateBuiltinType(ParsedBuiltinType* parsed) -> Type {
    return parsed->ty;
}

auto Sema::TranslateIntType(ParsedIntType* parsed) -> Type {
    if (parsed->bit_width > IntType::MaxBits) {
        Error(parsed->loc, "The maximum integer type is %6(i{:i}%)", IntType::MaxBits);
        return IntType::Get(*tu, IntType::MaxBits);
    }
    return IntType::Get(*tu, parsed->bit_width);
}

auto Sema::TranslateNamedType(ParsedDeclRefExpr* parsed) -> Opt<Type> {
    Assert(not parsed->empty(), "DRE is empty?");
    auto res = RequestLookup(nullptr, parsed->names(), parsed->loc, LookupHint::Type);
    if (not res.successful()) {
        ReportLookupFailure(std::move(res));
        return {};
    }

    // Template type.
    if (auto ttd = dyn_cast<TemplateTypeParamDecl>(res.decls.front()))
        return ttd->arg_type();

    // Type decl (struct or type alias).
    if (auto s = dyn_cast<TypeDecl>(res.decls.front()))
        return s->type;

    Error(parsed->loc, "'{}' does not name a type", utils::join(parsed->names(), "::"));
    Note(res.decls.front()->location(), "Declared here");
    return {};
}

auto Sema::TranslateOptionalType(ParsedOptionalType* parsed) -> Opt<Type> {
    auto elem = TRY(TranslateType(parsed->elem));
    if (not CheckVariableType(elem, parsed->loc)) return {};
    if (elem == Type::NilTy) return Error(parsed->loc, "Element type of optional cannot be '%1(nil%)'");
    return OptionalType::Get(*tu, elem);
}

auto Sema::TranslateProcType(
    ParsedProcType* parsed,
    bool allow_immutable_params,
    ArrayRef<Type> deduced_var_parameters,
    llvm::function_ref<ProcDecl*(ProcType*)> declare
) -> Opt<Type> {
    // Sanity check.
    //
    // We use u32s for indices here and there, so ensure that this is small
    // enough. For now, only allow up to 65535 parameters because that’s
    // more than anyone should need.
    if (parsed->param_types().size() > std::numeric_limits<u16>::max()) {
        return Error(
            parsed->loc,
            "Sorry, that’s too many parameters (max is {})",
            std::numeric_limits<u16>::max()
        );
    }

    SmallVector<ParamTypeData, 10> params;
    u32 var_params = 0;
    for (auto a : parsed->param_types()) {
        // Binding mutability belongs to a declaration, not a procedure type.
        bool has_error = a.immutable and not allow_immutable_params;
        if (has_error) Error(a.type->loc, "'%1(val%)' is only allowed in procedure declarations");
        auto ty_res = TranslateType(a.type);
        if (not ty_res) {
            has_error = true;
            ty_res = Type::VoidTy;
        }

        // If this parameter’s type is 'var', then substitute whatever we
        // deduced for it.
        Type ty = *ty_res;
        bool is_var_param = ty == Type::DeducedTy;
        if (is_var_param) {
            Assert(var_params < deduced_var_parameters.size());
            ty = deduced_var_parameters[var_params++];
        }

        // Check this here, but don’t do anything about it; this is only
        // harmful if we emit LLVM IR for it, and we won’t be getting there
        // anyway because of this error.
        //
        // see DeferredNativeProcArgOrReturn.
        if (parsed->attrs.native and IsZeroSizedOrIncomplete(ty))
            DiagnoseZeroSizedTypeInNativeProc(ty, a.type->loc, false);

        // If this is a variadic parameter, convert it to a slice.
        //
        // Do *not* do this if this is a 'var...' parameter since we pass
        // those as a tuple; this will have already been handled during
        // substitution.
        if (a.variadic and not is_var_param) {
            auto ty_res = BuildSliceType(ty, a.immutable, a.type->loc);
            if (ty_res) {
                ty = *ty_res;
            } else {
                has_error = true;
                ty = Type::VoidTy;
            }
        }

        if (not ProhibitDeducedTypes(ty, a.type->loc)) {
            has_error = true;
            ty = Type::VoidTy;
        }

        params.emplace_back(a.intent, ty, a.variadic);
        params.back().contains_error = has_error;
    }

    auto type = ProcType::Get(
        *tu,
        {Type::DeducedTy, Expr::RValue},
        params,
        CallingConvention::Native,
        parsed->attrs.c_varargs
    );

    // If we have a procedure here and not just a type, declare it now.
    std::optional<EnterProcedure> proc_scope;
    if (declare) {
        auto proc = declare(type);
        proc_scope.emplace(*this, proc);
    }

    auto ret = TranslateType(parsed->ret_type, Type::VoidTy);
    if (
        parsed->attrs.native and
        ret != Type::VoidTy and
        ret != Type::NoReturnTy and
        IsZeroSizedOrIncomplete(ret)
    ) DiagnoseZeroSizedTypeInNativeProc(ret, parsed->ret_type->loc, true);

    return ProcType::AdjustRet(*tu, type, ret);
}

auto Sema::TranslatePtrType(ParsedPtrType* stmt) -> Opt<Type> {
    auto immutable = dyn_cast<ParsedValueType>(stmt->elem);
    auto ty = TRY(TranslateType(immutable ? immutable->elem : stmt->elem));
    if (not ProhibitDeducedTypes(ty, stmt->loc)) return {};
    return PtrType::Get(*tu, ty, immutable != nullptr);
}

auto Sema::TranslateRangeType(ParsedRangeType* parsed) -> Opt<Type> {
    auto ty = TRY(TranslateType(parsed->elem));

    // Only ranges of integers are supported.
    if (not ty->is_integer()) return Error(
        parsed->loc,
        "Range element type must be an integer, but was '%1({}%)'",
        ty
    );

    return RangeType::Get(*tu, ty);
}

auto Sema::TranslateSliceType(ParsedSliceType* parsed) -> Opt<Type> {
    auto immutable = dyn_cast<ParsedValueType>(parsed->elem);
    auto ty = TRY(TranslateType(immutable ? immutable->elem : parsed->elem));
    return BuildSliceType(ty, immutable != nullptr, parsed->loc);
}

auto Sema::TranslateTemplateType(ParsedTemplateType* parsed) -> Opt<Type> {
    auto res = RequestScopeLookup(curr_scope(), {parsed->name, parsed->loc}, LookupHint::Type);
    if (not res) return Error(parsed->loc, "Deduced template type cannot occur here");
    auto ty = cast<TemplateTypeParamDecl>(res.decls.front());
    if (not ty->in_substitution) return Error(parsed->loc, "Deduced template type cannot occur here");
    return ty->arg_type();
}

auto Sema::TranslateTypeofType(ParsedTypeofType* parsed) -> Opt<Type> {
    return TRY(TranslateExpr(parsed->arg))->type;
}

auto Sema::TranslateValueType(ParsedValueType* parsed) -> Opt<Type> {
    auto ty = TRY(TranslateType(parsed->elem));

    // 'val' is parsed as a type, but it isn’t really a type, but rather a
    // property of variables and specifically pointer types; any contexts that
    // admit 'val' must handle it separately; if we encounter it anywhere else
    // then that’s just an error. Pretend it isn't there.
    Error(parsed->loc, "'%1(val%)' is only allowed in variable declarations or pointer types");
    return ty;
}

auto Sema::TranslateType(ParsedStmt* parsed, Type fallback) -> Type {
    return TranslateType(parsed).value_or(fallback);
}

auto Sema::TranslateType(ParsedStmt* parsed) -> Opt<Type> {
    switch (parsed->kind()) {
        using K = ParsedStmt::Kind;
#       define PARSE_TREE_LEAF_TYPE(n) case K::n: return Translate##n(cast<Parsed##n>(parsed));
#       include "srcc/ParseTree.inc"
        case K::DeclRefExpr: return TranslateNamedType(cast<ParsedDeclRefExpr>(parsed));
        case K::NilExpr: return Type::NilTy;

        // Array types are parsed as subscript expressions.
        case K::BinaryExpr: {
            auto b = cast<ParsedBinaryExpr>(parsed);
            if (b->op != Tk::LBrack) goto default_;
            return TranslateArrayType(b);
        }

        // Tuples can be treated as types.
        case K::TupleExpr: {
            SmallVector<TypeLoc> types;
            auto t = cast<ParsedTupleExpr>(parsed);
            bool ok = true;
            for (auto e : t->elems()) {
                if (auto ty = TranslateType(e.expr())) types.emplace_back(*ty, e.expr()->loc);
                else ok = false;
            }

            // Bail if there was an error.
            if (not ok) return {};

            // If this is a simple parenthesised expression, with no name, just return the 1st element type.
            if (types.size() == 1 and t->is_paren_expr()) return types.front().ty;

            // Otherwise, build a tuople.
            auto names = llvm::to_vector(vws::transform(t->elems(), [&](auto& e) { return e.name.name.str(); }));
            if (ok) return BuildTupleType(types, names, parsed->loc);
            return {};
        }

        // Injections may yield a type.
        case K::InjectExpr: {
            auto injected = TRY(TranslateInjectExpr(cast<ParsedInjectExpr>(parsed), std::nullopt));
            if (injected->type_or_void() != Type::TypeTy) goto default_;
            auto ty = RequestEvaluation(injected);
            if (ty.has_value()) return ty->cast<Type>();
            goto default_;
        }

        // If we don’t know what this is, try to evaluate it as a type.
        default:
        default_: {
            auto e = TRY(TranslateExpr(parsed, Type::TypeTy));
            auto val = TRY(RequestEvaluation(e));
            if (val.isa<Type>()) return val.cast<Type>();
            return Error(parsed->loc, "Expected type");
        }
    }
}
