#include <srcc/AST/Enums.hh>
#include <srcc/AST/Eval.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/ClangForward.hh>
#include <srcc/Frontend/Sema.hh>
#include <srcc/Macros.hh>

#include <clang/AST/ASTImporter.h>
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
using clang::QualType;

#define TRY(expr) ({                                                       \
    auto _v = expr;                                                        \
    if (not _v.has_value()) return std::unexpected(std::move(_v.error())); \
    *_v;                                                                   \
})

template <typename Value>
struct Res : std::expected<Value, std::unique_ptr<Diagnostic>> {
    using Base = std::expected<Value, std::unique_ptr<Diagnostic>>;
    using Base::Base;

    Res() = delete("Return a diagnostic instead");
    Res(std::nullptr_t) = delete("Return a diagnostic instead");
    Res(std::nullopt_t) = delete("Return a diagnostic instead");
};

class srcc::CXXImporter {
    Sema& S;
    ImportedClangModuleDecl* clang_module;

public:
    explicit CXXImporter(Sema& S, ImportedClangModuleDecl* clang_module) : S(S), clang_module(clang_module) {}
    auto AST() -> clang::ASTContext& { return clang_module->clang_ast.getASTContext(); }


    auto ImportDecl(clang::Decl* d) -> Res<Decl*>;
    auto ImportName(clang::TagDecl* td) -> StringRef;
    auto ImportReturnType(SLoc sloc, QualType ty) -> Res<TypeAndValueCategory>;
    auto ImportType(clang::SourceLocation sloc, QualType ty) -> Res<Type>;
    auto ImportType(SLoc sloc, QualType ty) -> Res<Type>;
    auto ImportSourceLocation(clang::SourceLocation sloc) -> SLoc;
    auto ImportValue(SLoc loc, const clang::APValue& val, Type ty) -> Res<eval::RValue>;

private:
    auto ImportDeclImpl(clang::Decl* d) -> Res<Decl*>;
    auto ImportRecordImpl(clang::RecordDecl* rd) -> Res<TypeDecl*>;
    auto ImportFunctionImpl(clang::FunctionDecl* d) -> Res<ProcDecl*>;
public:

    // Convert 'QualType' and other clang types that can't be formatted w/o an ASTContext
    // to a string so it can be used in a diagnostic.
    template <typename T>
    static auto PreprocessDiagArg(CXXImporter& i, T val) {
        using NonRef = std::remove_cvref_t<T>;
        if constexpr (utils::is<NonRef, QualType, clang::CanQualType>) {
            SmallString<32> str;
            llvm::raw_svector_ostream os{str};
            val.print(os, clang::PrintingPolicy(i.clang_module->clang_ast.getLangOpts()));
            return str;
        } else if constexpr (utils::is<NonRef, clang::Type*, const clang::Type*>) {
            return PreprocessDiagArg(i, QualType(val, 0));
        } else if constexpr (utils::is<NonRef, clang::APValue>) {
            SmallString<128> str;
            llvm::raw_svector_ostream os{str};
            val.dump(os, i.clang_module->clang_ast.getASTContext());
            return str;
        } else if constexpr (std::convertible_to<NonRef, clang::Decl*>) {
            if (auto n = dyn_cast<clang::NamedDecl>(val)) {
                return std::format("{}", n->getQualifiedNameAsString());
            } else {
                return std::string(static_cast<const clang::Decl*>(val)->getDeclKindName());
            }
        } else {
            return std::move(val);
        }
    }

    template <typename Entity>
    auto GetEntityLoc(Entity e) -> SLoc {
        using Pointee = std::remove_pointer_t<Entity>;
        if constexpr (std::derived_from<Pointee, clang::Decl>) {
            return ImportSourceLocation(e->getLocation());
        } else if constexpr (std::derived_from<Pointee, clang::Stmt>) {
            return ImportSourceLocation(e->getBeginLoc());
        } else if constexpr (std::is_same_v<Entity, clang::SourceLocation>) {
            return ImportSourceLocation(e);
        } else if constexpr (std::is_same_v<Entity, SLoc>) {
            return e;
        } else {
            static_assert(false, "Unknown entity");
        }
    }

    template <typename Entity, typename... Args>
    auto MakeErr(
        Entity e,
        std::format_string<decltype(PreprocessDiagArg(std::declval<CXXImporter&>(), std::declval<Args>()))...> fmt,
        Args&&... args
    ) {
        auto diag = S.CreateNote(GetEntityLoc(e), fmt, PreprocessDiagArg(*this, std::forward<Args>(args))...);
        return std::unexpected(std::make_unique<Diagnostic>(diag));
    }

    template <typename Entity, typename... Args>
    auto NYI(
        Entity e,
        std::format_string<decltype(PreprocessDiagArg(std::declval<CXXImporter&>(), std::declval<Args>()))...> fmt,
        Args&&... args
    ){
        auto msg = Format(fmt, PreprocessDiagArg(*this, std::forward<Args>(args))...);
        if (S.tu->lang_opts().wcxx_import) Warn(GetEntityLoc(e), "Importing {} is not yet supported", msg);
        return MakeErr(e, "Importing {} is not yet supported", msg);
    }

    template <typename... Args>
    void Warn(
        SLoc loc,
        std::format_string<Args...> fmt,
        Args&&... args
    ) {
        if (not S.tu->lang_opts().wcxx_import) return;
        S.ReportDiag(DiagsProducer::CreateWarning(loc, fmt, std::forward<Args>(args)...));
    }

    auto AlreadyDiagnosed() {
        return std::unexpected(std::unique_ptr<Diagnostic>());
    }
};

// The 'canonical decl' of a decl can change, because of course it can,
// so this function instead tries to provide a *stable* 'canonical' decl.
static auto GetStableCanonicalDecl(clang::Decl* d) -> clang::Decl* {
    return llvm::TypeSwitch<clang::Decl*, clang::Decl*>(d)
        .Case<clang::RecordDecl>([](auto* rd) { return rd->getDefinition() ?: rd->getFirstDecl(); })
        .Case<clang::FunctionDecl>([](auto* fd) { return fd->getDefinition() ?: fd->getFirstDecl(); })
        .Case<clang::EnumConstantDecl>([](auto* fd) { return fd->getFirstDecl(); })
        .Default([&](auto* d){ return d->getCanonicalDecl(); });
}

auto srcc::CXXImporter::ImportDecl(clang::Decl* d) -> Res<Decl*> {
    d = GetStableCanonicalDecl(d);

    // If we have attempted to find this before, do not do so again.
    if (auto it = S.imported_decls.find(d); it != S.imported_decls.end()){
        if (auto d = it->second.get_or_null()) return d;
        return AlreadyDiagnosed();
    }

    // Otherwise, try to create it now.
    //
    // Don’t hold on to a reference to 'S.imported_decls[D]' as importing a decl
    // might import more decls, which might invalidate it.
    S.imported_decls[d] = nullptr; // Mark that we’ve already tried doing this.
    auto imported = TRY(ImportDeclImpl(d));
    S.imported_decls[d] = imported;
    return imported;
}

auto srcc::CXXImporter::ImportDeclImpl(clang::Decl* d) -> Res<Decl*> {
    DebugAssert(
        S.imported_decls.at(d).invalid() and d == GetStableCanonicalDecl(d),
        "ImportDeclImpl() should only be called by ImportDecl()"
    );

    // Ignore invalid ones.
    Assert(d);
    if (d->isInvalidDecl()) return MakeErr(d, "declaration of '{}' is invalid", d);
    switch (d->getKind()) {
        using K = clang::Decl::Kind;
        default: break;
        case K::Enum: {
            auto e = cast<clang::EnumDecl>(d);

            for (auto enumerator : e->enumerators())
                S.imported_decls[GetStableCanonicalDecl(enumerator)] = nullptr;

            auto underlying = TRY(ImportType(e->getLocation(), e->getIntegerType()));
            auto loc = ImportSourceLocation(e->getLocation());
            auto name = ImportName(e);
            auto scope = S.tu->create_scope<BlockScope>(S.global_scope());
            auto enum_ty = new (*S.tu) EnumType(*S.tu, scope, S.tu->save(name), underlying, loc);
            for (auto enumerator : e->enumerators()) {
                auto decl = new (*S.tu) EnumeratorDecl(
                    enum_ty,
                    S.tu->save(enumerator->getName()),
                    ImportSourceLocation(enumerator->getLocation())
                );

                decl->value = S.tu->store_int(enumerator->getValue());
                S.AddDeclToScope(scope, decl);
                S.imported_decls[GetStableCanonicalDecl(enumerator)] = decl;
            }

            enum_ty->finalise();
            return enum_ty->decl();
        } break;

        case K::EnumConstant: {
            TRY(ImportDecl(cast<clang::EnumConstantDecl>(d)->getType()->getAsEnumDecl()));
            if (auto decl = S.imported_decls[d].get_or_null()) return decl;
            return AlreadyDiagnosed();
        }

        case K::Function:
            return ImportFunctionImpl(cast<clang::FunctionDecl>(d));

        case K::Namespace:
            // TODO
            break;

        case K::Record:
        case K::CXXRecord:
            return ImportRecordImpl(cast<clang::RecordDecl>(d));

        // There are 3 patterns that we want to handle here.
        //
        //  1. This is a typedef of some tag type whose name is the same as
        //     that of the typedef (e.g. 'typedef struct foo { ... } foo').
        //     Ignore the typedef and just import the underlying tag.
        //
        //  2. As 1, but the tag is anonymous (e.g. 'typedef struct { ... } foo');
        //     Import the tag and set its name to that of the typedef.
        //
        //  3. If the typedef is 'size_t' (and it's _actually_ size_t and not
        //     just called that), map it to 'int'.
        //
        //  4. None of the above; import the underlying type and create a type alias.
        case K::Typedef:
        case K::TypeAlias: {
            auto td = cast<clang::TypedefNameDecl>(d);

            // Case 2.
            //
            // Using the typedef name as the tag name is handled when we
            // import the tag, so no need to deal w/ that here.
            if (auto anon = td->getAnonDeclWithTypedefName(true))
                return TRY(ImportDecl(anon));

            // Case 1.
            //
            // Use 'dyn_cast' rather than 'getAs' because we don't want to
            // discard type sugar here.
            auto underlying = td->getUnderlyingType();
            if (
                auto tt = dyn_cast<clang::TagType>(underlying);
                tt and tt->getDecl()->getDeclName() == td->getDeclName()
            ) return ImportDecl(tt->getDecl());

            // Case 3.
            auto loc = ImportSourceLocation(td->getLocation());
            if (
                td->getName() == "size_t" and
                underlying.getCanonicalType() == AST().getSizeType().getCanonicalType()
            ) return AliasType::Create(*S.tu, Type::IntTy, String("size_t"), loc)->decl();

            // Case 4.
            //
            // Type aliases may be recursive (in that they may refer to a
            // struct that contains the a type alias); create the alias first
            // and fill in the type later.
            auto alias = AliasType::Create(*S.tu, Type::VoidTy, S.tu->save(td->getName()), loc);
            S.imported_decls[td] = alias->decl();
            alias->set_aliased_type(TRY(ImportType(loc, underlying)));
            return alias->decl();
        }

        case K::Var: {
            auto var = cast<clang::VarDecl>(d);
            if (var->isLocalVarDecl() or var->getKind() != clang::Decl::Kind::Var)
                return MakeErr(var, "Cannot import this variable");

            if (var->getFormalLinkage() != clang::Linkage::External)
                return MakeErr(var, "Cannot import variable with non-external linkage");

            if (var->getTSCSpec() != clang::TSCS_unspecified)
                return MakeErr(var, "Cannot import thread-local variable");

            auto name = S.tu->save(var->getName());
            auto loc = ImportSourceLocation(var->getLocation());
            auto ty = TRY(ImportType(loc, var->getType()));
            return new (*S.tu) GlobalDecl(
                S.tu.get(),
                clang_module,
                ty,
                var->getType().isConstQualified(),
                name,
                Linkage::Imported,
                Mangling::CXX,
                loc
            );
        }
    }

    return NYI(d, "declaration of kind {}", d->getDeclKindName());
}

auto srcc::CXXImporter::ImportFunctionImpl(clang::FunctionDecl* d) -> Res<ProcDecl*> {
    DebugAssert(
        S.imported_decls.at(d).invalid() and d == GetStableCanonicalDecl(d),
        "Call ImportDecl() instead of ImportFunctionImpl()"
    );

    if (isa<clang::CXXMethodDecl>(d)) return NYI(d, "member functions");

    // If the return type hasn’t been deduced yet, we can’t import it.
    if (d->getReturnType()->getAs<clang::AutoType>())
        return NYI(d, "function with deduced return type");

    // Don’t import immediate or inline functions for now.
    // TODO: Constant-evaluation of C++ functions.
    if (d->isImmediateFunction())
        return NYI(d, "'%1(consteval%)' function");

    // Don’t import functions with internal linkage, or anything
    // attached to a module.
    if (d->getLinkageInternal() != clang::Linkage::External)
        return NYI(d, "'%1(static%)' function");
    if (d->getOwningModule())
        return NYI(d, "function that is part of a C++20 module");

    // Do not import language builtins.
    if (auto builtin = d->getBuiltinID()) {
        // Note: Clang treats C standard library functions (e.g. 'puts') as
        // builtins as well, but those count as ‘library builtins’.
        if (not AST().BuiltinInfo.isPredefinedLibFunction(builtin))
            return NYI(d, "builtin function");
    }

    // Import the type.
    auto ty = cast<ProcType>(TRY(ImportType(d->getLocation(), QualType(d->getFunctionType(), 0))));

    // Create the procedure.
    auto proc = ProcDecl::Create(
        *S.tu,
        clang_module,
        ty,
        S.tu->save(d->getNameAsString()),
        Linkage::Imported,
        d->isExternC() ? Mangling::None : Mangling::CXX,
        nullptr,
        InheritedProcedureProperties{.is_cxx_inline_function = d->isInlineSpecified() and d->hasBody()},
        ImportSourceLocation(d->getNameInfo().getBeginLoc())
    );

    // We may need to mangle this, so remember where it came from.
    proc->set_clang_decl(d);

    // In C++, parameter names may differ across declarations, so only specify
    // a parameter name if it is the same across all declarations (in which that
    // parameter has a name).
    SmallVector<DeclNameLoc> param_names;
    for (auto i : llvm::seq(ty->param_count())) {
        std::optional<clang::DeclarationName> name;
        clang::SourceLocation name_loc;
        for (auto redecl : d->redecls()) {
            auto clang_param = redecl->getParamDecl(u32(i));
            auto n = clang_param->getDeclName();

            // This parameter declaration has no name.
            if (n.isEmpty()) continue;

            // We don't have a name yet, use this name.
            if (not name.has_value()) {
                name = n;
                name_loc = clang_param->getLocation();
            }

            // We already have a name; if it is different from the current one,
            // clear it and give up.
            else if (name.value() != n) {
                name = std::nullopt;
                break;
            }
        }

        // If we have a name, add it to the type.
        if (name.has_value()) {
            param_names.emplace_back(
                S.tu->save(name.value().getAsIdentifierInfo()->getName()),
                ImportSourceLocation(name_loc)
            );
        } else {
            param_names.emplace_back();
        }
    }

    // Create param decls.
    //
    // Don't use BuildParamDecl() here as that requires creating a
    // scope for the procedure, which we don't do here since we don't
    // need it.
    SmallVector<LocalDecl*> params;
    for (auto [i, name] : enumerate(param_names)) {
        params.push_back(new (*S.tu) ParamDecl(
            &ty->params()[i],
            Expr::MLValue, // We pass by value so this is irrelevant.
            name,
            proc,
            u32(i),
            false
        ));
    }

    proc->finalise(nullptr, params);
    return proc;
}

auto srcc::CXXImporter::ImportName(clang::TagDecl* td) -> StringRef {
    if (auto tdef = td->getTypedefNameForAnonDecl()) return tdef->getName();
    if (td->getDeclName().isIdentifier()) return td->getName();
    return "";
}

auto srcc::CXXImporter::ImportRecordImpl(clang::RecordDecl* rd) -> Res<TypeDecl*> {
    DebugAssert(
        S.imported_decls.at(rd).invalid() and rd == GetStableCanonicalDecl(rd),
        "Call ImportDecl() instead of ImportRecordImpl()"
    );

    // Incomplete types are treated as opaque.
    if (not rd->isCompleteDefinition()) return OpaqueType::Create(
        *S.tu,
        S.tu->save(ImportName(rd)),
        ImportSourceLocation(rd->getLocation())
    )->decl();

    // Neither are types that have a destructor, constructor, etc.
    if (auto CXX = dyn_cast<clang::CXXRecordDecl>(rd)) {
        if (not CXX->isCLike()) return NYI(rd, "non-trivial class type");
    }

    // Declare the type now because it may be recursive.
    auto scope = S.tu->create_scope<StructScope>(S.global_scope());
    auto name = S.tu->save(ImportName(rd));
    auto type = StructType::Create(*S.tu, scope, name, ImportSourceLocation(rd->getLocation()));
    S.imported_decls[rd] = type->decl();

    // Import the fields.
    auto& layout = AST().getASTRecordLayout(rd);
    SmallVector<FieldDecl*> fields;
    for (auto [i, f] : enumerate(rd->fields())) {
        if (f->getMaxAlignment() != 0) return NYI(f, "unaligned or overaligned field");
        if (f->hasInClassInitializer()) return NYI(f, "field with in-class initialiser");
        auto decl = new (*S.tu) FieldDecl(
            TRY(ImportType(f->getLocation(), f->getType())),
            Size::Bits(layout.getFieldOffset(unsigned(i))),
            S.tu->save(f->getName()),
            ImportSourceLocation(f->getLocation())
        );

        fields.push_back(decl);
        S.AddDeclToScope(scope, decl);
    }

    // Determine if this contains a pointer.
    bool contains_pointer = any_of(fields, [&](auto *fd) {
        return fd->type->is_or_contains_pointer();
    });

    // Build the layout.
    auto bits = RecordLayout::Bits::Trivial(contains_pointer);
    bits.is_union = rd->isUnion();
    auto rl = RecordLayout::Create(
        *S.tu,
        fields,
        Size::Bytes(layout.getSize().getQuantity()),
        Size::Bytes(layout.getSize().getQuantity()),
        Align(layout.getAlignment().getQuantity()),
        bits
    );

    type->finalise(rl);
    return type->decl();
}

auto srcc::CXXImporter::ImportReturnType(
    SLoc loc,
    QualType ty
) -> Res<TypeAndValueCategory> {
    auto vc = Expr::RValue;

    if (ty->isReferenceType()) {
        ty = ty->getPointeeType();
        vc = Expr::LValue(ty.isConstQualified());
    }

    return TypeAndValueCategory{TRY(ImportType(loc, ty)), vc};
}

auto srcc::CXXImporter::ImportType(clang::SourceLocation sloc, QualType ty) -> Res<Type> {
    return ImportType(ImportSourceLocation(sloc), ty);
}

auto srcc::CXXImporter::ImportType(SLoc sloc, QualType ty) -> Res<Type> {
    switch (ty->getTypeClass()) {
        using K = clang::Type::TypeClass;
        default: break;
        case K::Builtin: {
            switch (cast<clang::BuiltinType>(ty)->getKind()) {
                using K = clang::BuiltinType::Kind;
                default: break;
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

                case K::Int128:
                case K::UInt128:
                    return S.tu->I128Ty;
            }
        } break;

        case K::LValueReference:
        case K::RValueReference: {
            auto elem = TRY(ImportType(sloc, ty->getPointeeType()));
            return PtrType::Get(*S.tu, elem, ty->getPointeeType().isConstQualified());
        }

        // C++ pointers are nullable, so wrap them in an optional.
        case K::Pointer: {
            auto elem = TRY(ImportType(sloc, ty->getPointeeType()));
            auto ptr = PtrType::Get(*S.tu, elem, ty->getPointeeType().isConstQualified());
            return OptionalType::Get(*S.tu, ptr);
        }

        case K::BitInt: {
            auto b = cast<clang::BitIntType>(ty);
            return IntType::Get(*S.tu, Size::Bits(b->getNumBits()));
        }

        case K::ConstantArray: {
            auto c = cast<clang::ConstantArrayType>(ty);
            auto elem = TRY(ImportType(sloc, c->getElementType()));
            return ArrayType::Get(*S.tu, elem, i64(c->getSize().getZExtValue()));
        }

        case K::Enum:
            return cast<TypeDecl>(TRY(ImportDecl(ty->getAsEnumDecl())))->type;

        case K::FunctionProto: {
            auto fpt = cast<clang::FunctionProtoType>(ty);
            if (fpt->getCallConv() != clang::CallingConv::CC_C) {
                return NYI(
                    sloc,
                    "function with '{}' calling convention",
                    clang::FunctionType::getNameForCallConv(fpt->getCallConv())
                );
            }

            auto ret = fpt->getExtInfo().getNoReturn()
                ? TypeAndValueCategory{Type::NoReturnTy, Expr::RValue}
                : TRY(ImportReturnType(sloc, fpt->getReturnType()));

            SmallVector<ParamTypeData, 6> params;
            for (auto param : fpt->param_types()) {
                auto param_ty = TRY(ImportType(sloc, param));
                params.emplace_back(Intent::ByValue, param_ty);
            }

            return ProcType::Get(
                *S.tu,
                ret,
                params,
                CallingConvention::Native,
                fpt->isVariadic()
            );
        }

        case K::Record: {
            auto rd = ty->castAsRecordDecl();
            return cast<TypeDecl>(TRY(ImportDecl(rd)))->type;
        }

        case K::Typedef: {
            auto td = cast<clang::TypedefType>(ty);
            return cast<TypeDecl>(TRY(ImportDecl(td->getDecl())))->type;
        }

        case K::Using: {
            auto td = cast<clang::UsingType>(ty);
            return cast<TypeDecl>(TRY(ImportDecl(td->getDecl()->getTargetDecl())))->type;
        }
    }

    if (ty != ty.getCanonicalType())
        return ImportType(sloc, ty.getCanonicalType());

    return NYI(sloc, "type '{}'", ty);
}

auto srcc::CXXImporter::ImportSourceLocation(clang::SourceLocation sloc) -> SLoc {
    if (not sloc.isValid()) return {};
    auto& sm = AST().getSourceManager();
    auto name = sm.getFilename(sloc);
    auto f = S.ctx.try_get_file(name.str());
    if (not f.has_value()) {
        Warn(SLoc(), "Could not import file for source location: {}", name);
        return {};
    }
    return SLoc(f.value()->data() + sm.getFileOffset(sloc));
}

auto srcc::CXXImporter::ImportValue(
    SLoc loc,
    const clang::APValue& val,
    Type ty
) -> Res<eval::RValue> {
    switch (val.getKind()) {
        case clang::APValue::Int: {
            Assert(ty->is_integer_or_bool());
            return eval::RValue(val.getInt(), ty);
        }

        case clang::APValue::Struct: {
            // Type must be a struct type.
            auto struct_ty = cast<RecordType>(ty);

            // Field count must match what we expect.
            u32 num_fields = val.getStructNumFields();
            Assert(struct_ty->layout().fields().size() == num_fields);

            // Import each field.
            SmallVector<eval::RValue*> fields;
            for (u32 i = 0; i < num_fields; i++) {
                auto field = TRY(ImportValue(
                    loc,
                    val.getStructField(i),
                    struct_ty->layout().fields()[i]->type
                ));

                fields.push_back(S.tu->save(field));
            }

            return eval::RValue(
                eval::Record(ArrayRef(fields).copy(S.tu->allocator())),
                ty
            );
        }

        default: break;
    }

    return MakeErr(
        loc,
        "Unsupported {} APValue: {}",
        enchantum::to_string(val.getKind()),
        val
    );
}

auto Sema::ParseCXX(
    StringRef code,
    std::optional<std::string> PCH
) -> std::unique_ptr<clang::ASTUnit> {
    llvm::TimeTraceScope _{"[Clang] C++ Parsing"};

    // For PCHs, we need to remember this file and its contents.
    auto Name = std::format("__srcc.imports.{}.cc", cxx_import_file_counter++);
    PCHVFS->addFile(Name, 0, llvm::MemoryBuffer::getMemBufferCopy(code, Name));

    // The arguments need to include the tool name as well as the file name.
    std::vector<std::string> args{
        SOURCE_CLANG_EXE,
        "-x",
        "c++",
        Name,

        // Pass through whatever target we’re compiling for.
        "-target",
        tu->target().triple().getTriple(),

        // Use the latest standard.
        "-std=c++2c",

        // Diagnostic options.
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",

        // Forward our use_colours setting.
        std::format("-fdiagnostics-color={}", tu->context().use_colours ? "always" : "never"),

        // Make sure we always emit declarations if requested, even 'inline' ones.
        "-femit-all-decls",
    };

    // Add opt level.
    int clang_opt_level = 0;
    if (ctx.opt_level == 4) {
        args.push_back("-march=native");
        clang_opt_level = 3;
    } else {
        clang_opt_level = ctx.opt_level;
    }

    args.push_back(std::format("-O{}", clang_opt_level));

    // Add the PCH if we have one.
    if (PCH.has_value()) {
        args.push_back("-include-pch");
        args.push_back(std::move(PCH.value()));
    }

    // Add include paths.
    for (const auto& p : clang_include_paths) {
        args.push_back("-I");
        args.push_back(p);
    }

    // Add user-specified options last.
    append_range(args, clang_options);

    // We don't use clang::tooling::buildASTFromCodeWithArgs() because we handle
    // setting up the file system ourselves.
    struct Action : clang::tooling::ToolAction {
        std::unique_ptr<clang::ASTUnit> ast;
        bool runInvocation(
            std::shared_ptr<clang::CompilerInvocation> invocation,
            clang::FileManager* file_mgr,
            std::shared_ptr<clang::PCHContainerOperations> pch_ops,
            clang::DiagnosticConsumer* consumer
        ) override {
            Assert(not ast, "Expected at most one ASTUnit here!");
            auto clang_diags = clang::CompilerInstance::createDiagnostics(
                file_mgr->getVirtualFileSystem(),
                invocation->getDiagnosticOpts(),
                consumer,
                /*ShouldOwnClient=*/false
            );

            ast = clang::ASTUnit::LoadFromCompilerInvocation(
                std::move(invocation),
                std::move(pch_ops),
                nullptr,
                clang_diags,
                file_mgr,
                false,
                clang::CaptureDiagsKind::None
            );

            return ast != nullptr && not ast->getSema().hasUncompilableErrorOccurred();
        }
    };

    Action the_action;
    auto file_mgr = llvm::makeIntrusiveRefCnt<clang::FileManager>(clang::FileSystemOptions(), ImportVFS);
    clang::tooling::ToolInvocation invocation{
        args,
        &the_action,
        file_mgr.get(),
        std::make_shared<clang::PCHContainerOperations>()
    };

    if (not invocation.run()) return nullptr;
    Assert(the_action.ast);
    return std::move(the_action.ast);
}

auto Sema::ImportCXXHeaders(
    String logical_name,
    ArrayRef<String> header_names,
    SLoc import_loc
) -> Ptr<ImportedClangModuleDecl> {
    auto lc = import_loc.seek_line_column(context());
    auto file = import_loc.file_name(context());

    // Use quoted includes so source-file-relative includes work.
    SmallString<128> code;
    for (auto h : header_names) {
        Assert(h.starts_with("<") and h.ends_with(">"));
        if (lc and file) Format(code, "#line {} \"{}\"\n", lc->line, *file);
        Format(code, "#include \"{}\"\n", h.drop().drop_back());
    }

    auto AST = ParseCXX(code);
    if (not AST) {
        Error(import_loc, "Header import failed");
        return {};
    }

    return ImportedClangModuleDecl::Create(
        *tu,
        *tu->add_clang_ast_unit(std::move(AST)),
        logical_name,
        header_names,
        import_loc
    );
}

namespace {
enum class Kind : u8 {
    Nothing = 0,
    Type = 1,
    Function = 2,
    Enumerator = 4,
    Var = 8,
};

// FIXME: This should be a macro in base.
enum class NegatedKind : u8 {};
constexpr NegatedKind operator~(Kind a) { return NegatedKind(~+a); }
constexpr Kind operator&(Kind a, NegatedKind b) { return Kind(+a & +b); }
constexpr Kind& operator&=(Kind& a, NegatedKind b) {
    a = a & b;
    return a;
}

constexpr bool operator&(Kind a, Kind b) { return (+a & +b) == +b; }
constexpr Kind operator|(Kind a, Kind b) { return Kind(+a | +b); }
constexpr  Kind& operator|=(Kind& a, Kind b) {
    a = a | b;
    return a;
}
}

auto Sema::LookUpCXXNameImpl(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclNameLoc> names,
    LookupHint hint
) -> LookupResult {
    auto ast = &clang_module->clang_ast;
    auto& clang_sema = ast->getSema();
    auto& ast_ctx = ast->getASTContext();
    auto& pp = clang_sema.getPreprocessor();

    // Look up all scopes in the path.
    clang::DeclContext* ctx = ast_ctx.getTranslationUnitDecl();
    for (auto n : names.drop_back()) {
        auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(n.name.str())));
        if (not res.isSingleResult()) return LookupResult::NonScopeInPath(n);
        auto new_ctx = dyn_cast<clang::DeclContext>(res.front());
        if (not new_ctx) return LookupResult::NonScopeInPath(n);
        ctx = new_ctx;
    }

    // Look up the last segment in the scope.
    // TODO: Support operators.
    auto res = ctx->lookup(clang::DeclarationName(pp.getIdentifierInfo(names.back().name.str())));

    // Filters.
    #define FILTER(F, ...)
    static const auto EnumeratorFilter = [](clang::Decl* d) { return isa<clang::EnumConstantDecl>(d); };
    static const auto FunctionFilter = [](clang::Decl* d) { return isa<clang::FunctionDecl>(d); };
    static const auto VarFilter = [](clang::Decl* d) { return isa<clang::VarDecl>(d); };
    static const auto TypeFilter = [](clang::Decl* d) {
        return isa<clang::TypedefNameDecl, clang::EnumDecl, clang::RecordDecl>(d);
    };

    // Figure out what we found.
    auto found = Kind::Nothing;
    CXXImporter i{*this, clang_module};
    for (auto d : res) {
        if (d->isInvalidDecl()) continue;
        if (VarFilter(d)) found |= Kind::Var;
        else if (FunctionFilter(d)) found |= Kind::Function;
        else if (TypeFilter(d)) found |= Kind::Type;
        else if (EnumeratorFilter(d)) found |= Kind::Enumerator;
        else {
            auto diag = CreateNote(
                i.ImportSourceLocation(d->getLocation()),
                "Importing a '{}Decl' is not supported (yet)",
                d->getDeclKindName()
            );

            return LookupResult::FailedToImport(
                names.back(),
                std::make_unique<Diagnostic>(std::move(diag))
            );
        }
    }

    // And import the declarations we care about.
    SmallVector<Decl*> converted;
    auto ImportFiltered = [&](auto filter) {
        for (auto d : res) {
            if (d->isInvalidDecl() or not std::invoke(filter, d)) continue;
            auto decl = i.ImportDecl(d);
            if (not decl.has_value()) return LookupResult::FailedToImport(names.back(), std::move(decl.error()));
            if (not is_contained(converted, *decl)) converted.push_back(*decl);
        }

        if (converted.empty()) return LookupResult::NotFound(names.back());
        if (converted.size() == 1) return LookupResult::Success(converted.front());
        return LookupResult::Ambiguous(names.back(), converted);
    };

    // If we want a type, and we found a type, keep only the types.
    if (hint == LookupHint::Type and found & Kind::Type)
        found = Kind::Type;

    // If we found multiple things, remove any types.
    if (hint != LookupHint::Type and found != Kind::Type)
        found &= ~Kind::Type;

    // If we still have multiple things left, then that's an error.
    if (std::popcount(+found) > 1) {
        auto diag = CreateNote(names.back().loc, "There are multiple incompatible declarations");
        return LookupResult::FailedToImport(
            names.back(),
            std::make_unique<Diagnostic>(std::move(diag))
        );
    }

    switch (found) {
        case Kind::Nothing: return LookupResult::NotFound(names.back());
        case Kind::Function: return ImportFiltered(FunctionFilter);
        case Kind::Enumerator: return ImportFiltered(EnumeratorFilter);
        case Kind::Type: return ImportFiltered(TypeFilter);
        case Kind::Var: return ImportFiltered(VarFilter);
    }

    Unreachable();
}

/// Expand a C++ macro and import the result.
///
/// The general strategy for this is as follows: first, we ask Clang's
/// preprocessor to expand the macro; if this results in a single identifier
/// token, we perform name lookup on that identifier.
///
/// Otherwise, we create a source string that contains a function declaration
/// with a 'return' statement that returns the macro; it is constexpr so we
/// can attempt to constant-fold it. We also build a variable declaration whose
/// initialiser is a call to that function, just so we get a CallExpr that we
/// can evaluate.
///
/// This source string is compiled into a new ASTUnit; the orginal ASTUnit is
/// emitted as a precompiled header and included into the new one via '-include-pch';
/// this allows us to reference types and declarations in the imported C++ headers
/// without having to parse them again or modify the original ASTUnit.
///
/// Assuming there are no errors in this process, we then attempt to constant-evaluate
/// the initialiser of the variable we created. If this succeeds, we get a compile-time
/// constant in the form of a ConstExpr; this is the 'value' of the macro expansion.
///
/// If constant evaluation fails, we may have a macro that expands to either a late
/// compile-time constant (e.g. an integer constant cast to a pointer type), or some
/// runtime code; in this case, we need to generate a runtime call to the function we
/// created. This also means we need to tell Clang to codegen the function so we can
/// actually link against it. We emit it to LLVM IR and then our CodeGen will then link
/// that LLVM module into our own one after IR->LLVM lowering. The result of the call is
/// the 'value' of the macro expansion.
///
/// Either way, the 'value' is some expression; we wrap it in a CXXMacroExpansionDecl
/// since name lookup needs to resolve to a declaration. When referenced, this 'declaration'
/// emits the expression that is the 'value', which in the case of it being a runtime call
/// results in the function being called, and thus the macro expansion being executed anew,
/// every time the macro is 'expanded' in source.
///
/// This entire process is cached, i.e. we create a single CXXMacroExpansionDecl for every
/// MacroInfo; subsequent uses reference the same declaration.
auto Sema::LookUpCXXMacro(
    ImportedClangModuleDecl* clang_module,
    clang::MacroInfo* mi,
    DeclNameLoc macro_name,
    LookupHint hint
) -> LookupResult {
    // Import the source location of the macro.
    CXXImporter main_importer{*this, clang_module};
    SLoc macro_loc = main_importer.ImportSourceLocation(mi->getDefinitionLoc());

    // Return an error.
    auto Err = [&]<typename ...Args>(std::format_string<Args...> fmt, Args&& ...args) {
        return LookupResult::FailedToImport(
            macro_name,
            std::make_unique<Diagnostic>(CreateNote(macro_loc, fmt, std::forward<Args>(args)...))
        );
    };

    // Refuse to import ‘builtin’ macros (i.e. __LINE__, __COUNTER__, and friends)
    // as well as function-like macros.
    if (mi->isBuiltinMacro())
        return Err("Cannot import builtin macro '{}'", macro_name);
    if (mi->isFunctionLike())
        return Err("Cannot import function-like macro '{}'", macro_name);

    // If we have attempted to find this before, do not do so again.
    if (auto it = imported_macros.find(mi); it != imported_macros.end()) {
        if (it->second.present()) return LookupResult::Success(it->second.get());
        return LookupResult::FailedToImport(macro_name);
    }

    // Cache that we attempted to import this.
    imported_macros[mi] = nullptr;

    // Enter the macro definition as a new file.
    auto& clang_sema = clang_module->clang_ast.getSema();
    auto& pp = clang_sema.getPreprocessor();
    auto& sm = clang_sema.getSourceManager();
    auto clang_sloc = sm.getLocForStartOfFile(sm.getMainFileID());
    auto fid = sm.createFileID(
        llvm::MemoryBuffer::getMemBufferCopy(
            macro_name.name.str(),
            "<macro expansion>"
        )
    );

    if (pp.EnterSourceFile(fid, nullptr, clang_sloc))
        return Err("Failed to enter source file");

    // Ask the preprocessor to expand the macro.
    std::vector<clang::Token> toks;

    // Manual implementation of LexTokensUntilEOF that also keeps the
    // EOF token, because 'StringifyArgument' expects an EOF-terminated
    // token list, because of course it does.
    for (;;) {
        auto& tok = toks.emplace_back();
        pp.Lex(tok);
        if (tok.is(clang::tok::eof)) break;
        if (tok.is(clang::tok::unknown)) return Err("Unknown token in macro expansion");
    }

    // Macro expanded to nothing.
    if (toks.size() == 1) return Err("Macro has empty expansion");

    // If it is a single identifier, just look up that name.
    if (
        toks.size() == 2 and
        toks.front().is(clang::tok::identifier)
    ) return LookUpCXXNameImpl(
        clang_module,
        DeclNameLoc{tu->save(toks.front().getIdentifierInfo()->getName()), macro_name.loc},
        hint
    );

    // Otherwise, we need to parse this thing.
    //
    // First, emit the Clang module to a PCH if we haven't already done that.
    auto pch_name = std::format("__srcc_pch_{}", static_cast<void*>(clang_module));
    if (not PCHVFS->exists(pch_name)) {
        SmallString<0> pch;
        llvm::raw_svector_ostream os{pch};
        if (clang_module->clang_ast.serialize(os))
            return Err("AST serialisation failed");
        PCHVFS->addFile(pch_name, 0, llvm::MemoryBuffer::getMemBufferCopy(pch));
    }

    // Wrap the macro in a function that returns it and also create a variable initialised
    // with a call to it; the latter is used to try and constant-evaluate the function.
    //
    // Clang gets mad if we combine 'extern "C"' and 'decltype(auto)', so use '__asm__'
    // instead to set the function name; this however requires separating the definition and
    // declaration for some reason...
    //
    // The 'always_inline' is required, else inlining just doesn't happen at all (possibly
    // because the function has 'linkonce_odr' linkage).
    auto name_base = Format("__srcc_expanded_macro_{}", generated_cxx_macro_decls);
    auto proc_name = Format("{}_init", name_base);
    auto var_name = Format("{}_var", name_base);
    auto code = Format(
        "#line 1 \"<macro-expansion>\"\n"
        "[[_Clang::__always_inline__]] constexpr decltype(auto) {0}() __asm__(\"{0}\");\n"
        "constexpr decltype(auto) {0}() {{ return {1}; }}\n"
        "decltype(auto) {2} = {0}();\n",
        proc_name,
        macro_name,
        var_name
    );

    auto macro_tu = ParseCXX(code, std::move(pch_name));
    if (not macro_tu or macro_tu->getSema().hasUncompilableErrorOccurred())
        // No diagnostic here; Clang already told the user what was wrong.
        // TODO: Introduce a custom Clang diags consumer (or at least set a
        // custom ostream).
        return LookupResult::FailedToImport(macro_name);

    // Get the procedure or variable decl. Take care to retrieve the II
    // in the preprocessor of the new TU.
    auto GetDecl = [&](StringRef name) -> clang::Decl* {
        clang::DeclarationName ident{macro_tu->getPreprocessor().getIdentifierInfo(name)};
        auto res = macro_tu->getASTContext().getTranslationUnitDecl()->lookup(ident);
        Assert(res.isSingleResult() and not res.front()->isInvalidDecl());
        return res.front();
    };

    // Retrieve the variable.
    auto var = cast<clang::VarDecl>(GetDecl(var_name));

    // The importer requires a module, so fabricate one for this.
    auto fake_module = ImportedClangModuleDecl::Create(
        *tu,
        *macro_tu,
        "__srcc_macro_expansion__",
        {},
        SLoc()
    );

    CXXImporter fake_importer{*this, fake_module};

    // Import a type from the Clang ASTUnit used for expansion to Source.
    auto ImportTypeFromExpansionAST = [&](QualType clang_ty) -> Res<Type> {
        // Import the type of the value from the new ASTUnit to the main one;
        // this is to make sure that two references to the same type actually
        // resolve to the same Source type.
        clang::ASTImporter clang_importer{
            clang_module->clang_ast.getASTContext(),
            clang_module->clang_ast.getFileManager(),
            fake_module->clang_ast.getASTContext(),
            fake_module->clang_ast.getFileManager(),
            /*MinimalImport=*/false,
            clang_importer_state
        };

        auto imported_type = clang_importer.Import(clang_ty);
        if (auto err = imported_type.takeError()) {
            return fake_importer.MakeErr(
                macro_loc,
                "Clang ASTImporter failed to import '{}': {}",
                clang_ty,
                toString(std::move(err))
            );
        }

        // Then import the type from the main module to Source.
        return main_importer.ImportType(macro_loc, *imported_type);
    };

    // If we can’t evaluate this as a constant, instead emit the function
    // into an LLVM module and create a *local* value whose initialiser is
    // a call to it.
    clang::Expr::EvalResult er;
    SmallVector<clang::PartialDiagnosticAt> diags;
    er.Diag = &diags;
    Expr* value = nullptr;
    if (
        not var->getInit()->EvaluateAsInitializer(
            macro_tu->getASTContext(),
            var,
            er,
            false
        ) or not diags.empty()
    )  {
        generated_cxx_macro_decls++;

        // Retrieve the procedure.
        auto clang_proc = cast<clang::FunctionDecl>(GetDecl(proc_name));

        // Import its return type.
        auto clang_ret_ty = clang_proc->getType()->castAs<clang::FunctionProtoType>()->getReturnType();
        auto ret_ty = fake_importer.ImportReturnType(macro_loc, clang_ret_ty);
        if (not ret_ty.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(ret_ty.error())
        );

        // Emit it.
        auto cg = fake_module->create_clang_codegen(tu->llvm_context);

        // Make sure we actually emit this.
        // FIXME: I don't think this is needed.
        clang_proc->setIsUsed();

        // Emit the decl; no custom diagnostic here; if this fails, Clang should
        // emit a diagnostic.
        cg->Initialize(macro_tu->getASTContext());
        cg->HandleTopLevelDecl(clang::DeclGroupRef(clang_proc));
        cg->HandleTranslationUnit(macro_tu->getASTContext());
        if (not cg->GetModule()) return LookupResult::FailedToImport(macro_name);
        tu->link_llvm_modules.emplace_back(cg->ReleaseModule());

        // Declare the procedure on our end. We don't associate it with a Clang
        // declaration here since we haven't imported it into the actual clang
        // module (it's only in the one we just parsed), and moreover, we don't
        // need it since this isn't mangled.
        auto proc = ProcDecl::Create(
            *tu,
            clang_module,
            ProcType::Get(*tu, *ret_ty),
            tu->save(proc_name),
            Linkage::Imported,
            Mangling::None,
            nullptr,
            InheritedProcedureProperties{.always_inline = true},
            macro_loc
        );

        // Create a call to it. This should never fail.
        value = BuildCallExpr(
            CreateReference(proc, macro_loc).get(),
            {},
            macro_loc
        ).get();
    }

    // Ok, we managed to constant-evaluate the initialiser; convert
    // it to an RValue.
    else {
        auto ty = ImportTypeFromExpansionAST(var->getType());
        if (not ty.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(ty.error())
        );

        // Finally, import the value.
        auto v = fake_importer.ImportValue(macro_loc, *var->getEvaluatedValue(), *ty);
        if (not v.has_value()) return LookupResult::FailedToImport(
            macro_name,
            std::move(v.error())
        );

        value = MakeConstExpr(nullptr, std::move(*v), macro_loc);
    }

    auto md = new (*tu) CXXMacroExpansionDecl("", value, value->location());
    imported_macros[mi] = md;
    return LookupResult::Success(md);
}

auto Sema::LookUpCXXName(
    ImportedClangModuleDecl* clang_module,
    ArrayRef<DeclNameLoc> names,
    LookupHint hint
) -> LookupResult {
    Assert(not names.empty(), "Empty name lookup?");
    auto DoNameLookup = [&] {
        return LookUpCXXNameImpl(clang_module, names, hint);
    };

    // If this is anything other than a single identifier, it can’t be a macro.
    if (names.size() != 1 or not names.front().name.is_str())
        return DoNameLookup();

    // Check if this is a macro.
    auto ast = &clang_module->clang_ast;
    auto& pp = ast->getSema().getPreprocessor();
    auto id = pp.getIdentifierInfo(names.front().name.str());
    auto *mi = pp.getMacroInfo(id);
    if (not mi) return DoNameLookup();
    return LookUpCXXMacro(clang_module, mi, names.front(), hint);
}
