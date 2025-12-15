#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/DeclarationName.h>
#include <clang/AST/RecordLayout.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/MacroArgs.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Parse/Parser.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Tooling/Tooling.h>

#include <base/Assert.hh>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>

using namespace srcc;

class Sema::Importer {
    Sema& S;
    ImportedClangModuleDecl* clang_module;

public:
    explicit Importer(Sema& S, ImportedClangModuleDecl* clang_module) : S(S), clang_module(clang_module) {}
    auto AST() -> clang::ASTContext& { return clang_module->clang_ast.getASTContext(); }
    auto ImportDecl(clang::Decl* D) -> Ptr<Decl>;
    auto ImportRecord(clang::RecordDecl* RD) -> std::optional<Type>;
    auto ImportFunction(clang::FunctionDecl* D) -> Ptr<ProcDecl>;
    auto ImportType(const clang::Type* T) -> std::optional<Type>;
    auto ImportType(clang::QualType T) { return ImportType(T.getTypePtr()); }
    auto ImportSourceLocation(clang::SourceLocation sloc) -> SLoc;
    auto ImportValue(const clang::APValue& val, clang::QualType Ty) -> std::optional<eval::RValue>;
};

auto Sema::Importer::ImportDecl(clang::Decl* D) -> Ptr<Decl> {
    D = D->getCanonicalDecl();

    // If we have attempted to find this before, do not do so again.
    if (auto it = S.imported_decls.find(D); it != S.imported_decls.end())
        return it->second;

    // Otherwise, try to create it now. First, mark that we’ve already
    // tried doing this.
    S.imported_decls[D] = nullptr;

    // Ignore invalid ones.
    if (D->isInvalidDecl()) return {};
    switch (D->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::Enum:
            // TODO
            break;

        case K::Function: {
            auto f = ImportFunction(cast<clang::FunctionDecl>(D));
            if (not f) break;
            return f;
        }

        case K::Namespace:
            // TODO
            break;

        case K::Record:
        case K::CXXRecord: {
            auto ty = ImportRecord(cast<clang::RecordDecl>(D));
            if (ty) return cast<StructType>(ty.value())->decl();
        } break;

        case K::Typedef: {
            auto td = cast<clang::TypedefDecl>(D);
            auto clang_ty = AST().getTypedefType(
                clang::ElaboratedTypeKeyword::None,
                std::nullopt,
                td
            );

            if (clang_ty->isRecordType()) {
                auto ty = ImportType(clang_ty);
                if (ty) return cast<StructType>(ty.value())->decl();
            }
        } break;

        case K::Using:
            // TODO
            break;

        case K::Var:
            // TODO
            break;
    }

    if (auto n = dyn_cast<clang::NamedDecl>(D)) {
        S.Error(
            ImportSourceLocation(n->getLocation()),
            "Importing declaration of '{}' from C++ is not supported",
            n->getQualifiedNameAsString()
        );
    } else {
        S.Error(
            ImportSourceLocation(D->getBeginLoc()),
            "Importing this C++ declaration is not supported"
        );
    }

    return {};
}

auto Sema::Importer::ImportFunction(clang::FunctionDecl* D) -> Ptr<ProcDecl> {
    D = D->getDefinition() ?: D->getCanonicalDecl();
    if (isa<clang::CXXMethodDecl>(D)) return {};
    auto FPT = D->getType()->getAs<clang::FunctionProtoType>();
    Assert(FPT, "No prototype in C++?");

    // If the return type hasn’t been deduced yet, we can’t import it.
    if (D->getReturnType()->getAs<clang::AutoType>()) return {};

    // Don’t import immediate or inline functions for now.
    if (D->isImmediateFunction() or D->isInlineSpecified()) return {};

    // Don’t import functions with internal linkage, or anything
    // attached to a module.
    if (D->getLinkageInternal() != clang::Linkage::External) return {};
    if (D->getOwningModule()) return {};

    // Do not import language builtins.
    if (auto ID = D->getBuiltinID()) {
        // Note: Clang treats C standard library functions (e.g. 'puts') as
        // builtins as well, but those count as ‘library builtins’.
        if (not AST().BuiltinInfo.isPredefinedLibFunction(ID))
            return {};
    }

    // Import the type.
    auto T = ImportType(FPT);
    if (not T) return {};
    if (not isa<ProcType>(*T)) return {};

    // Create the procedure.
    auto PD = ProcDecl::Create(
        *S.tu,
        clang_module,
        cast<ProcType>(T.value().ptr()),
        S.tu->save(D->getNameAsString()),
        Linkage::Imported,
        D->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        ManglingNumber::None,
        ImportSourceLocation(D->getNameInfo().getBeginLoc())
    );

    // Create param decls.
    SmallVector<LocalDecl*> Params;
    for (auto [I, P] : enumerate(D->parameters())) {
        Params.push_back(new (*S.tu) ParamDecl(
            &PD->param_types()[I],
            Expr::LValue,
            S.tu->save(P->getName()),
            PD,
            u32(I),
            false,
            ImportSourceLocation(P->getLocation())
        ));
    }

    PD->finalise(nullptr, Params);
    return PD;
}

auto Sema::Importer::ImportRecord(clang::RecordDecl* RD) -> std::optional<Type> {
    Assert(RD);
    auto it = S.imported_records.find(RD);
    if (it != S.imported_records.end()) return it->second;

    // Create the cache entry now so we fail fast next time if we can’t import
    // this; don’t hold on to a reference to the cache entry here since we’re
    // about to import more types, which might invalidate it.
    S.imported_records[RD] = std::nullopt;

    // Skip unions and incomplete types.
    if (not RD or not RD->isCompleteDefinition() or RD->isUnion()) return std::nullopt;

    // Import the fields.
    auto& RL = AST().getASTRecordLayout(RD);
    SmallVector<FieldDecl*> Fields;
    for (auto [I, F] : enumerate(RD->fields())) {
        if (F->isBitField()) return std::nullopt;
        if (F->getMaxAlignment() != 0) return std::nullopt;
        if (F->hasInClassInitializer()) return std::nullopt;
        auto FTY = ImportType(F->getType());
        if (not FTY) return std::nullopt;
        Fields.push_back(new (*S.tu) FieldDecl(
            FTY.value(),
            Size::Bits(RL.getFieldOffset(unsigned(I))),
            S.tu->save(F->getName()),
            ImportSourceLocation(F->getLocation())
        ));
    }

    // Validate other properties of this type.
    if (auto CXX = dyn_cast<clang::CXXRecordDecl>(RD)) {
        if (not CXX->isCLike()) return std::nullopt;
    }

    // Determine the name of this type.
    StringRef Name;
    if (RD->hasNameForLinkage()) {
        if (RD->getDeclName().isIdentifier()) {
            Name = RD->getName();
        } else if (auto TD = RD->getTypedefNameForAnonDecl()) {
            Name = TD->getName();
        }
    }

    // Determine if this contains a pointer.
    bool contains_pointer = any_of(Fields, [&](auto *fd) {
        return fd->type->is_or_contains_pointer();
    });

    // Build the layout.
    auto rl = RecordLayout::Create(
        *S.tu,
        Fields,
        Size::Bytes(RL.getSize().getQuantity()),
        Size::Bytes(RL.getSize().getQuantity()),
        Align(RL.getAlignment().getQuantity()),
        RecordLayout::Bits::Trivial(contains_pointer)
    );

    auto Struct = S.BuildCompleteStructType(
        S.tu->save(Name),
        rl,
        ImportSourceLocation(RD->getLocation())
    );

    S.imported_records[RD] = Struct;
    return Struct;
}

auto Sema::Importer::ImportType(const clang::Type* T) -> std::optional<Type> {
    // FIXME: C++ pointers should be imported as nullable pointers once
    // we support optionals.

    // Handle known type sugar first.
    if (
        auto TD = T->getAs<clang::TypedefType>();
        TD and TD->getDecl()->getName() == "size_t" and
        T->getCanonicalTypeUnqualified() == AST().getSizeType()
    ) return Type::IntTy;

    // Only handle canonical types from here on.
    T = T->getCanonicalTypeUnqualified().getTypePtr();
    switch (T->getTypeClass()) {
        using K = clang::Type::TypeClass;
        default: return std::nullopt;
        case K::Builtin: {
            switch (cast<clang::BuiltinType>(T)->getKind()) {
                using K = clang::BuiltinType::Kind;
                default: return std::nullopt;
                case K::Void: return Type::VoidTy;
                case K::Bool: return Type::BoolTy;

                case K::SChar:
                case K::UChar:
                case K::Char_S:
                case K::Char_U:
                    return S.tu->FFICharTy;

                case K::WChar_S:
                case K::WChar_U:
                    return S.tu->FFIWCharTy;

                case K::Short:
                case K::UShort:
                    return S.tu->FFIShortTy;

                case K::Int:
                case K::UInt:
                    return S.tu->FFIIntTy;

                case K::Long:
                case K::ULong:
                    return S.tu->FFILongTy;

                case K::LongLong:
                case K::ULongLong:
                    return S.tu->FFILongLongTy;
            }
        }

        case K::LValueReference:
        case K::RValueReference:
        case K::Pointer: {
            auto Elem = ImportType(T->getPointeeType());
            if (not Elem) return std::nullopt;
            return PtrType::Get(*S.tu, *Elem);
        }

        case K::BitInt: {
            auto B = cast<clang::BitIntType>(T);
            return IntType::Get(*S.tu, Size::Bits(B->getNumBits()));
        }

        case K::ConstantArray: {
            auto C = cast<clang::ConstantArrayType>(T);
            auto Elem = ImportType(C->getElementType());
            if (not Elem) return std::nullopt;
            return ArrayType::Get(*S.tu, *Elem, i64(C->getSize().getZExtValue()));
        }

        case K::FunctionProto: {
            auto FPT = cast<clang::FunctionProtoType>(T);
            if (FPT->getCallConv() != clang::CallingConv::CC_C) return std::nullopt;

            auto Ret = FPT->getExtInfo().getNoReturn() ? Type::NoReturnTy : ImportType(FPT->getReturnType());
            if (not Ret) return std::nullopt;

            SmallVector<ParamTypeData, 6> Params;
            for (auto P : FPT->param_types()) {
                auto Ty = ImportType(P);
                if (not Ty) return std::nullopt;
                Params.emplace_back(Intent::Copy, *Ty);
            }

            return ProcType::Get(
                *S.tu,
                *Ret,
                Params,
                CallingConvention::Native,
                FPT->isVariadic()
            );
        }

        case K::Record: {
            auto RD = T->getAsRecordDecl();
            if (not RD) return std::nullopt;
            return ImportRecord(RD);
        }
    }
}

auto Sema::Importer::ImportSourceLocation(clang::SourceLocation sloc) -> SLoc {
    if (not sloc.isValid()) return {};
    auto& sm = AST().getSourceManager();
    auto f = S.ctx.try_get_file(sm.getFilename(sloc).str());
    if (not f.has_value()) return {};
    return SLoc(f.value()->data() + sm.getFileOffset(sloc));
}

auto Sema::Importer::ImportValue(
    const clang::APValue& val,
    clang::QualType ty
) -> std::optional<eval::RValue> {
    switch (val.getKind()) {
        case clang::APValue::Int: {
            auto int_ty = ImportType(ty);
            if (not int_ty or not int_ty.value()->is_integer_or_bool()) return std::nullopt;
            return eval::RValue(val.getInt(), *int_ty);
        }

        case clang::APValue::None:
        case clang::APValue::Indeterminate:
        case clang::APValue::Float:
        case clang::APValue::FixedPoint:
        case clang::APValue::ComplexInt:
        case clang::APValue::ComplexFloat:
        case clang::APValue::LValue:
        case clang::APValue::Vector:
        case clang::APValue::Array:
        case clang::APValue::Struct:
        case clang::APValue::Union:
        case clang::APValue::MemberPointer:
        case clang::APValue::AddrLabelDiff:
            return std::nullopt;
    }

    Unreachable();
}

auto Sema::ImportCXXDecl(ImportedClangModuleDecl* clang_module, CXXDecl* decl) -> Ptr<Decl> {
    Importer importer(*this, clang_module);
    auto d = importer.ImportDecl(decl);
    imported_decls[decl] = d;
    return d;
}

auto Sema::ParseCXX(StringRef code) -> std::unique_ptr<clang::ASTUnit> {
    std::vector<std::string> args{
        "-x",
        "c++",
        "-Xclang",
        "-triple",
        "-Xclang",
        tu->target().triple().getTriple(),
        "-std=c++2c",
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",
        "-fcolor-diagnostics",
        "-fsyntax-only",
    };

    for (const auto& p : clang_include_paths) {
        args.push_back("-I");
        args.push_back(p);
    }

    return clang::tooling::buildASTFromCodeWithArgs(
        code,
        args,
        "__srcc.imports.cc",
        SOURCE_CLANG_EXE
    );
}

auto Sema::ImportCXXHeaders(
    String logical_name,
    ArrayRef<String> header_names,
    SLoc import_loc
) -> Ptr<ImportedClangModuleDecl> {
    auto AST = ParseCXX(utils::join(header_names, "", "#include {}\n"));
    if (not AST) {
        Error(import_loc, "Header import failed");
        return {};
    }

    clang_ast_units.push_back(std::move(AST));
    return ImportedClangModuleDecl::Create(
        *tu,
        *clang_ast_units.back(),
        logical_name,
        header_names,
        import_loc
    );
}

namespace {
enum class Kind {
    Nothing,
    Unsupported,
    Function,
    Type,
};
}

auto Sema::LookUpCXXNameImpl(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclName> names,
    LookupHint hint
) -> LookupResult {
    auto ast = &clang_module->clang_ast;
    auto& clang_sema = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = clang_sema.getPreprocessor();

    // Look up all scopes in the path.
    clang::DeclContext* ctx = ast_ctx.getTranslationUnitDecl();
    for (auto n : names.drop_back()) {
        auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(n.str())));
        if (not res.isSingleResult()) return LookupResult::NonScopeInPath(n);
        auto new_ctx = dyn_cast<clang::DeclContext>(res.front());
        if (not new_ctx) return LookupResult::NonScopeInPath(n);
        ctx = new_ctx;
    }

    // Look up the last segment in the scope.
    // TODO: Support operators.
    auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(names.back().str())));

    // Figure out what we found.
    auto kind = Kind::Nothing;
    auto Merge = [&](Kind k) {
        using enum Kind;
        Assert(k != Nothing, "'Nothing' should only be used as an initial state");
        Assert(kind != Unsupported, "Should not process any more decls after reaching 'Unsupported'");
        Assert(k != Unsupported, "Should just early return instead of merging 'Unsupported'");
        if (k == kind) {} // Do nothing. They’re the same anyway.
        else if (kind == Nothing) kind = k;
        else if (kind == Type and hint == LookupHint::Type) {} // Prefer types if types were requested.
        else if (kind == Function or k == Function) kind = Function;
        else {
            Assert(k == Type and kind == Type);
            kind = Type;
        }
    };

    for (auto d : res) {
        if (d->isInvalidDecl()) continue;
        if (isa<clang::FunctionDecl>(d)) Merge(Kind::Function);
        else if (isa<clang::TypedefDecl, clang::RecordDecl>(d)) Merge(Kind::Type);
        else {
            kind = Kind::Unsupported;
            break;
        }
    }

    // And import the declarations we care about.
    SmallVector<Decl*> converted;
    auto ImportFiltered = [&](auto filter) {
        for (auto d : res) {
            if (d->isInvalidDecl() or not std::invoke(filter, d)) continue;
            auto decl = ImportCXXDecl(clang_module, d);
            if (not decl) return LookupResult::FailedToImport(names.back());
            if (not is_contained(converted, decl.get())) converted.push_back(decl.get());
        }

        if (converted.empty()) return LookupResult::NotFound(names.back());
        if (converted.size() == 1) return LookupResult::Success(converted.front());
        return LookupResult::Ambiguous(names.back(), converted);
    };

    switch (kind) {
        case Kind::Nothing:
            return LookupResult::NotFound(names.back());

        case Kind::Unsupported:
            return LookupResult::FailedToImport(names.back());

        case Kind::Function:
            return ImportFiltered([](auto* d){ return isa<clang::FunctionDecl>(d); });

        case Kind::Type:
            return ImportFiltered([](auto* d){ return isa<clang::TypedefDecl, clang::RecordDecl>(d); });
    }

    Unreachable();
}

auto Sema::LookUpCXXName(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclName> names,
    LookupHint hint
) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto DoNameLookup = [&] {
        return LookUpCXXNameImpl(clang_module, names, hint);
    };

    // If this is anything other than a single identifier, it can’t be a macro.
    if (names.size() != 1 or not names.front().is_str())
        return DoNameLookup();

    auto ast = &clang_module->clang_ast;
    auto& clang_sema = ast->getSema();
    auto& pp = clang_sema.getPreprocessor();
    auto id = pp.getIdentifierInfo(names.front().str());
    auto *mi = pp.getMacroInfo(id);
    if (not mi) return DoNameLookup();

    // Refuse to import ‘builtin’ macros (i.e. __LINE__, __COUNTER__, and friends)
    // as well as function-like macros.
    if (mi->isBuiltinMacro() or mi->isFunctionLike())
        return LookupResult::FailedToImport(names.back());

    // If we have attempted to find this before, do not do so again.
    if (auto it = imported_macros.find(mi); it != imported_macros.end()) {
        if (it->second.present()) return LookupResult::Success(it->second.get());
        return LookupResult::FailedToImport(names.back());
    }

    // Cache that we attempted to import this.
    imported_macros[mi] = nullptr;

    // Enter the macro definition as a new file.
    auto& sm = clang_sema.getSourceManager();
    auto clang_sloc = sm.getLocForStartOfFile(sm.getMainFileID());
    auto fid = sm.createFileID(
        llvm::MemoryBuffer::getMemBuffer(
            std::string(names.front().str()), // Null-terminate this.
            "<macro expansion>",
            true
        )
    );

    if (pp.EnterSourceFile(fid, nullptr, clang_sloc))
        return LookupResult::FailedToImport(names.back());

    // Ask the preprocessor to expand the macro.
    std::vector<clang::Token> toks;

    // Manual implementation of LexTokensUntilEOF that also keeps the
    // EOF token, because 'StringifyArgument' expects an EOF-terminated
    // token list, because of course it does.
    for (;;) {
        auto& tok = toks.emplace_back();
        pp.Lex(tok);
        if (tok.is(clang::tok::eof)) break;
        if (tok.is(clang::tok::unknown)) return LookupResult::FailedToImport(names.back());
    }

    // Macro expanded to nothing.
    if (toks.size() == 1) return LookupResult::FailedToImport(names.back());

    // If it is a single identifier, just look up that name.
    if (
        toks.size() == 2 and
        toks.front().is(clang::tok::identifier)
    ) return LookUpCXXNameImpl(
        clang_module,
        DeclName(tu->save(toks.front().getIdentifierInfo()->getName())),
        hint
    );

    // Otherwise, we need to parse this thing.
    //
    // For now, we attempt to parse and evaluate it as a constant expression in
    // a new TU; this *does* mean that we can’t reference any symbols declared in
    // the actual imported TU, but this should be fine for most macros that aren’t
    // just a single identifier.
    //
    // FIXME: Can we convert the old TU into a PCH or sth like it and then
    // import it when we parse the new TU? That way, we don’t have to modify the
    // old TU, but we should still have access to everything in it.
    auto expansion = clang::MacroArgs::StringifyArgument(
        toks.data(),
        pp,
        false,
        clang_sloc,
        clang_sloc
    );

    // Wrap the expansion in a function; take care to strip the quotes around the string literal.
    auto name = Format("__srcc_expanded_macro_{}", generated_cxx_macro_decls);
    auto code = Format(
        "extern \"C\" decltype(auto) {} = [] {{ return {}; }} ();",
        name,
        StringRef(expansion.getLiteralData(), expansion.getLength()).drop_front().drop_back()
    );

    auto macro_tu = ParseCXX(code);
    if (not macro_tu or macro_tu->getSema().hasUncompilableErrorOccurred())
        return LookupResult::FailedToImport(names.back());

    // Take care to retrieve the II in the preprocessor of the new TU.
    clang::DeclarationName expanded_name{macro_tu->getPreprocessor().getIdentifierInfo(name)};
    auto res = macro_tu->getASTContext().getTranslationUnitDecl()->lookup(expanded_name);
    if (not res.isSingleResult() or res.front()->isInvalidDecl())
        return LookupResult::FailedToImport(names.back());

    clang::APValue init_val;
    SmallVector<clang::PartialDiagnosticAt> diags;
    auto var = cast<clang::VarDecl>(res.front());

    // Import the source location of the macro.
    SLoc macro_loc;
    {
        Importer i{*this, clang_module};
        macro_loc = i.ImportSourceLocation(mi->getDefinitionLoc());
    }

    // The importer requires a module, so fabricate one for this.
    auto fake_module = ImportedClangModuleDecl::Create(
        *tu,
        *macro_tu,
        "__srcc_macro_expansion__",
        {},
        SLoc()
    );

    // If we can’t evaluate this as a constant, instead emit it into an LLVM
    // module and reference it as a global declaration.
    if (
        not var->getInit()->EvaluateAsInitializer(
            init_val,
            macro_tu->getASTContext(),
            var,
            diags,
            false
        ) or not diags.empty()
    )  {
        // Import the variable’s type.
        Importer i{*this, fake_module};
        auto ty = i.ImportType(var->getType());
        if (not ty.has_value()) return LookupResult::FailedToImport(names.back());

        // Emit it.
        defer { generated_cxx_macro_decls++; };
        std::unique_ptr<clang::CodeGenerator> cg{clang::CreateLLVMCodeGen(
            macro_tu->getDiagnostics(),
            fake_module->name.str(),
            macro_tu->getVirtualFileSystemPtr(),
            macro_tu->getPreprocessor().getHeaderSearchInfo().getHeaderSearchOpts(),
            macro_tu->getPreprocessor().getPreprocessorOpts(),
            macro_tu->getCodeGenOpts(),
            tu->llvm_context
        )};

        // I need to fix this API at some point, because the fact that you have to
        // *remember* to call Initialize() is... not great.
        cg->Initialize(macro_tu->getASTContext());
        cg->HandleTopLevelDecl(clang::DeclGroupRef(var));
        cg->HandleTranslationUnit(macro_tu->getASTContext());
        if (not cg->GetModule()) return LookupResult::FailedToImport(names.back());
        tu->link_llvm_modules.emplace_back(cg->ReleaseModule());

        // Create a declaration for the variable.
        auto g = new (*tu) GlobalDecl(
            tu.get(),
            nullptr,
            *ty,
            tu->save(name),
            Linkage::Imported,
            Mangling::None,
            macro_loc
        );

        imported_macros[mi] = g;
        return LookupResult::Success(g);
    }

    // Ok, we managed to constant-evaluate the initialiser; convert
    // it to an RValue.
    eval::RValue val;
    {
        Importer i{*this, fake_module};
        auto v = i.ImportValue(*var->getEvaluatedValue(), var->getType());
        if (not v.has_value()) return LookupResult::FailedToImport(names.back());
        val = std::move(*v);
    }

    auto ce = MakeConstExpr(nullptr, std::move(val), macro_loc);
    auto vd = new (*tu) ValueDecl("", ce, ce->location());
    imported_macros[mi] = vd;
    return LookupResult::Success(vd);
}
