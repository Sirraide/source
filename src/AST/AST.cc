#include <srcc/AST/AST.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Type.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Core.hh>

#include <clang/Frontend/CompilerInstance.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/TargetParser/Host.h>

using namespace srcc;

// ============================================================================
//  Target
// ============================================================================
Target::Target() = default;
Target::~Target() = default;

// ============================================================================
//  TU
// ============================================================================
TranslationUnit::TranslationUnit(Context& ctx, const LangOpts& opts, StringRef name, bool is_module)
    : ctx{ctx},
      language_opts{opts},
      name{save(name)},
      is_module{is_module} {
    // Get information about the compilation target from Clang.
    std::array args {
        "-x",
        "c++",
        "foo.cc"
    };

    tgt.ci = std::make_unique<clang::CompilerInstance>();
    tgt.ci->createDiagnostics(*llvm::vfs::getRealFileSystem());
    Assert(clang::CompilerInvocation::CreateFromArgs(tgt.ci->getInvocation(), args, tgt.ci->getDiagnostics()));
    Assert(tgt.ci->createTarget());
    tgt.TI = tgt.ci->getTargetPtr();
    vm.init(tgt);

    // Initialise integer types.
    I8Ty = IntType::Get(*this, Size::Bits(8));
    I16Ty = IntType::Get(*this, Size::Bits(16));
    I32Ty = IntType::Get(*this, Size::Bits(32));
    I64Ty = IntType::Get(*this, Size::Bits(64));
    I128Ty = IntType::Get(*this, Size::Bits(128));

    // Initialise FFI types.
    FFIBoolTy = IntType::Get(*this, Size::Bits(target().TI->getBoolWidth()));
    FFICharTy = IntType::Get(*this, Size::Bits(target().TI->getCharWidth()));
    FFIShortTy = IntType::Get(*this, Size::Bits(target().TI->getShortWidth()));
    FFIIntTy = IntType::Get(*this, Size::Bits(target().TI->getIntWidth()));
    FFILongTy = IntType::Get(*this, Size::Bits(target().TI->getLongWidth()));
    FFILongLongTy = IntType::Get(*this, Size::Bits(target().TI->getLongLongWidth()));
    FFISizeTy = IntType::Get(*this, Size::Bits(target().TI->getTypeWidth(target().TI->getSizeType())));

    // Initialise other cached types.
    StrLitTy = SliceType::Get(*this, I8Ty);

    // If the name is empty, this is an imported module. Do not create
    // an initialiser for it as we can just synthesise a call to it, and
    // we also don’t have its name yet.
    if (name.empty()) return;

    // The module entry point differs depending on whether this is
    // is an executable or a library.
    //
    // For executables, the runtime is linked into the executable, and
    // thus, the entry point does not have to be exported. For libraries,
    // the entry point is called from other modules and therefore does
    // have to be exported.
    //
    // The name of an entry point is dependent on the module name and
    // cannot be overloaded, so it doesn’t need to be mangled.
    initialiser_proc = ProcDecl::Create(
        *this,
        ProcType::Get(*this, Type::VoidTy),
        is_module ? save(constants::EntryPointName(name)) : constants::ProgramEntryPoint,
        is_module ? Linkage::Internal : Linkage::Exported,
        Mangling::None,
        nullptr,
        {}
    );
}

auto TranslationUnit::Create(Context& ctx, const LangOpts& opts, StringRef name, bool is_module) -> Ptr {
    Assert(not name.empty(), "Use CreateEmpty() to create an empty module");
    auto m = new TranslationUnit{ctx, opts, name, is_module};
    return std::unique_ptr<TranslationUnit>(m);
}

void TranslationUnit::dump() const {
    bool c = context().use_colours;

    // Print preamble.
    utils::Print(c, "%1({}%) %2({}%)\n", is_module ? "Module" : "Program", name);

    // Dump imports.
    for (auto& [_, i] : logical_imports) i->dump(c);

    // Print content.
    for (auto s : file_scope_block->stmts()) s->dump(c);
}

auto TranslationUnit::save(eval::RValue val) -> eval::RValue* {
    evaluated_constants.push_back(std::make_unique<eval::RValue>(std::move(val)));
    return evaluated_constants.back().get();
}

auto TranslationUnit::store_int(APInt value) -> StoredInteger {
    return integers.front().store_int(std::move(value));
}

// ============================================================================
//  Printer
// ============================================================================
struct Stmt::Printer : PrinterBase<Stmt> {
    bool print_procedure_bodies = true;
    bool print_instantiations = true;
    Printer(bool use_colour, Stmt* E) : PrinterBase{use_colour} { Print(E); }
    void Print(Stmt* E);
    void PrintBasicHeader(Stmt* s, StringRef name);
    void PrintBasicNode(
        Stmt* S,
        StringRef name,
        llvm::function_ref<void()> print_extra_data = {},
        bool print_type = true
    );
};

void Stmt::Printer::PrintBasicHeader(Stmt* s, StringRef name) {
    print(
        "%1({}%) %4({}%) %5(<{}>%)",
        name,
        static_cast<void*>(s),
        s->loc.pos
    );
}

void Stmt::Printer::PrintBasicNode(
    Stmt* s,
    StringRef name,
    llvm::function_ref<void()> print_extra_data,
    bool print_type
) {
    PrintBasicHeader(s, name);

    if (auto e = dyn_cast<Expr>(s); e and print_type) print(" {}", e->type->print());
    if (print_extra_data) {
        print(" ");
        print_extra_data();
    }

    if (auto e = dyn_cast<Expr>(s); e and print_type) {
        if (e->value_category != Expr::SRValue) {
            switch (e->value_category) {
                case ValueCategory::SRValue: Unreachable();
                case ValueCategory::MRValue: print(" mrvalue"); break;
                case ValueCategory::LValue: print(" lvalue"); break;
            }
        }
    }

    print("\n");
}

void Stmt::Printer::Print(Stmt* e) {
    auto VCLowercase = [&](ValueCategory v) -> String {
        switch (v) {
            case ValueCategory::SRValue: return "srvalue";
            case ValueCategory::MRValue: return "mrvalue";
            case ValueCategory::LValue: return "lvalue";
        }
        return "<invalid value category>";
    };

    // FIXME: Should be a visitor.
    switch (e->kind()) {
        case Kind::ArrayBroadcastExpr: {
            auto a = cast<ArrayBroadcastExpr>(e);
            PrintBasicNode(e, "ArrayBroadcastExpr");
            PrintChildren(a->element);
        } break;

        case Kind::ArrayInitExpr: {
            auto a = cast<ArrayInitExpr>(e);
            PrintBasicNode(e, "ArrayInitExpr");
            PrintChildren<Expr>(a->initialisers());
        } break;

        case Kind::AssertExpr: {
            auto a = cast<AssertExpr>(e);
            PrintBasicNode(e, "AssertExpr");
            SmallVector<Stmt*, 2> children;
            children.push_back(a->cond);
            if (auto msg = a->message.get_or_null()) children.push_back(msg);
            PrintChildren(children);
        } break;

        case Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);
            PrintBasicNode(e, "BinaryExpr", [&] { print("%1({}%)", utils::Escape(Spelling(b->op), false, true)); });
            SmallVector<Stmt*, 2> children{b->lhs, b->rhs};
            PrintChildren(children);
        } break;

        case Kind::BlockExpr:
            PrintBasicNode(e, "BlockExpr");
            PrintChildren(cast<BlockExpr>(e)->stmts());
            break;

        case Kind::BoolLitExpr:
            PrintBasicNode(e, "BoolLitExpr", [&] { print("%1({}%)", cast<BoolLitExpr>(e)->value); });
            break;

        case Kind::BuiltinCallExpr: {
            auto& c = *cast<BuiltinCallExpr>(e);
            PrintBasicNode(e, "BuiltinCallExpr", [&] {
                print("%2({}%)", [&] -> std::string_view {
                    switch (c.builtin) {
                        using B = BuiltinCallExpr::Builtin;
                        case B::Print: return "__srcc_print";
                        case B::Unreachable: return "__srcc_unreachable";
                    }

                    return "<invalid>";
                }());
            });

            PrintChildren<Expr>(c.args());
        } break;

        case Kind::BuiltinMemberAccessExpr: {
            auto& m = *cast<BuiltinMemberAccessExpr>(e);
            PrintBasicNode(e, "BuiltinMemberAccessExpr", [&] {
                using AK = BuiltinMemberAccessExpr::AccessKind;
                auto kind = [&] -> std::string_view {
                    switch (m.access_kind) {
                        case AK::SliceData: return "data";
                        case AK::SliceSize: return "size";
                        case AK::RangeStart: return "start";
                        case AK::RangeEnd: return "end";
                        case AK::TypeAlign: return "align";
                        case AK::TypeArraySize: return "arrsize";
                        case AK::TypeBits: return "bits";
                        case AK::TypeBytes: return "bytes";
                        case AK::TypeName: return "name";
                        case AK::TypeMaxVal: return "max";
                        case AK::TypeMinVal: return "min";
                    }
                    return "<invalid>";
                }();
                print("%3({}%)", kind);
            });
            PrintChildren(m.operand);
        } break;

        case Kind::CallExpr: {
            PrintBasicNode(e, "CallExpr");
            auto& c = *cast<CallExpr>(e);
            SmallVector<Stmt*, 10> fields;
            if (c.callee) fields.push_back(c.callee);
            if (auto a = c.args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::CastExpr: {
            auto c = cast<CastExpr>(e);
            PrintBasicHeader(c, "CastExpr");
            print(" {} ", c->type->print());
            switch (c->kind) {
                case CastExpr::Deref: print("deref"); break;
                case CastExpr::ExplicitDiscard: print("discard"); break;
                case CastExpr::LValueToSRValue: print("lvalue->srvalue"); break;
                case CastExpr::Integral: print("int->int"); break;
                case CastExpr::MaterialisePoisonValue: print("poison {}", VCLowercase(c->value_category)); break;
            }
            print("\n");
            PrintChildren(c->arg);
        } break;

        case Kind::ConstExpr: {
            auto c = cast<ConstExpr>(e);
            PrintBasicNode(e, "ConstExpr", [&] { print("{}", c->value->print()); });
            if (c->stmt) PrintChildren(c->stmt.get());
        } break;

        case Kind::DefaultInitExpr: {
            PrintBasicNode(e, "DefaultInitExpr");
        } break;

        case Kind::EmptyStmt:
            PrintBasicNode(e, "EmptyStmt");
            break;

        case Kind::EvalExpr: {
            PrintBasicNode(e, "EvalExpr");
            PrintChildren(cast<EvalExpr>(e)->stmt);
        } break;

        case Kind::FieldDecl: {
            auto f = cast<FieldDecl>(e);
            PrintBasicHeader(e, "FieldDecl");
            print(
                " {} %5({}%) %1(offs%) %3({:y}%)\n",
                f->type->print(),
                f->name,
                f->offset
            );
        } break;

        case Kind::ForStmt: {
            auto f = cast<ForStmt>(e);
            PrintBasicNode(e, "ForStmt");
            SmallVector<Stmt*> children;
            if (auto v = f->enum_var.get_or_null()) children.push_back(v);
            llvm::append_range(children, f->vars());
            llvm::append_range(children, f->ranges());
            children.push_back(f->body);
            PrintChildren(children);
        } break;

        case Kind::IfExpr: {
            auto i = cast<IfExpr>(e);
            PrintBasicNode(e, "IfExpr");
            SmallVector<Stmt*, 3> children{i->cond, i->then};
            if (auto el = i->else_.get_or_null()) children.push_back(el);
            PrintChildren(children);
        } break;

        case Kind::ImportedClangModuleDecl:
            PrintBasicNode(e, "ImportedClangModuleDecl", [&] {
                auto c = cast<ImportedClangModuleDecl>(e);
                print(
                    "%3({}%) %1(as%) {}",
                    utils::Escape(c->linkage_name, false, true),
                    c->name
                );
            });
            break;

        case Kind::ImportedSourceModuleDecl: {
            auto s = cast<ImportedSourceModuleDecl>(e);
            PrintBasicNode(e, "ImportedSourceModuleDecl", [&] {
                if (s->linkage_name != s->name) {
                    print("{} %1(as%) {}", s->linkage_name, s->name);
                } else {
                    print("{}", s->name);
                }
            });
            PrintChildren<Decl>(s->exports.decls() | rgs::to<std::vector>());
        } break;

        case Kind::IntLitExpr: {
            // These always come straight from the parser and are thus
            // always unsigned (because negative literals don’t exist;
            // unary minus is just an operator).
            auto i = cast<IntLitExpr>(e);
            auto PrintValue = [&] { print("%5({}%)", i->storage.str(false)); };
            PrintBasicNode(e, "IntLitExpr", PrintValue);
        } break;

        case Kind::LocalDecl:
        case Kind::ParamDecl: {
            bool is_param = e->kind() == Kind::ParamDecl;
            auto d = cast<LocalDecl>(e);
            auto PrintNameAndType = [&] {
                print(
                    "%{}({}%)",
                    is_param ? '4' : '8',
                    d->name
                );
                if (is_param) print(" %1({}%)", cast<ParamDecl>(d)->intent());
                print(" {}", d->type->print());
            };

            PrintBasicNode(e, is_param ? "ParamDecl" : "LocalDecl", PrintNameAndType);
            if (auto init = d->init.get_or_null()) PrintChildren(init);
        } break;

        case Kind::LocalRefExpr: {
            auto d = cast<LocalRefExpr>(e);
            bool is_param = d->decl->kind() == Kind::ParamDecl;
            auto PrintName = [&] { print("%{}({}%)", is_param ? '4' : '8', d->decl->name); };
            PrintBasicNode(e, "LocalRefExpr", PrintName);
        } break;

        case Kind::LoopExpr: {
            PrintBasicNode(e, "LoopExpr");
            if (auto b = cast<LoopExpr>(e)->body.get_or_null()) PrintChildren(b);
        } break;

        case Kind::MemberAccessExpr: {
            auto m = cast<MemberAccessExpr>(e);
            PrintBasicNode(e, "MemberAccessExpr");
            SmallVector<Stmt*, 2> children{m->base};
            children.push_back(m->field);
            PrintChildren(children);
        } break;

        case Kind::OverloadSetExpr: {
            auto o = cast<OverloadSetExpr>(e);
            PrintBasicNode(e, "OverloadSetExpr");
            tempset print_procedure_bodies = false;
            PrintChildren<Decl>(o->overloads());
        } break;

        case Kind::ProcDecl: {
            auto p = cast<ProcDecl>(e);
            PrintBasicHeader(p, "ProcDecl");
            print(" %2({}%) {}", p->name, p->type->print());

            if (p->instantiated_from) print(" instantiation");
            if (p->linkage == Linkage::Exported or p->linkage == Linkage::Reexported) print(" exported");
            if (p->linkage == Linkage::Imported or p->linkage == Linkage::Reexported) print(" imported");
            if (p->parent) print(" local");
            print("\n");
            if (not print_procedure_bodies) break;

            // Print template parameters and parameters.
            SmallVector<Stmt*> children;
            if (p->instantiated_from) children.push_back(p->instantiated_from);
            if (not p->is_imported()) append_range(children, p->params());

            // Take care we don’t recursively print ourselves when printing our parent.
            tempset print_instantiations = false;

            // And the body, if there is one.
            if (auto body = p->body().get_or_null()) children.push_back(body);
            PrintChildren(children);
        } break;

        case Kind::ProcTemplateDecl: {
            auto p = cast<ProcTemplateDecl>(e);
            PrintBasicHeader(p, "ProcTemplateDecl");
            print(" %2({}%)\n", p->name);
            if (print_instantiations) PrintChildren<ProcDecl>(p->instantiations());
        } break;

        case Kind::ProcRefExpr: {
            auto p = cast<ProcRefExpr>(e);
            PrintBasicHeader(p, "ProcRefExpr");
            print(" %2({}%)\n", p->decl->name);

            tempset print_procedure_bodies = false;
            PrintChildren(p->decl);
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ReturnExpr>(e);
            PrintBasicNode(e, "ReturnExpr", [&] { if (ret->implicit) print("implicit"); }, false);
            if (auto expr = ret->value.get_or_null()) PrintChildren(expr);
        } break;

        case Kind::StrLitExpr: {
            PrintBasicHeader(e, "StrLitExpr");
            print(" %3(\"{}\"%)\n", utils::Escape(cast<StrLitExpr>(e)->value, true, true));
        } break;

        case Kind::StructInitExpr: {
            auto s = cast<StructInitExpr>(e);
            PrintBasicNode(e, "StructInitExpr");
            PrintChildren<Expr>(s->values());
        } break;

        case Kind::TemplateTypeParamDecl: {
            auto t = cast<TemplateTypeParamDecl>(e);
            PrintBasicHeader(t, "TemplateTypeParamDecl");
            print(" %3(${}%) %1(type%) {}\n", t->name, t->arg_type());
        } break;

        case Kind::TypeDecl: {
            auto td = cast<TypeDecl>(e);
            PrintBasicHeader(td, "TypeDecl");
            if (auto s = dyn_cast<StructType>(td->type.ptr())) {
                print(
                    " %1(struct %3({}%) size %3({:y}%)/%3({:y}%) align %3({}%)%)\n",
                    s->name(),
                    s->size(),
                    s->array_size(),
                    s->align()
                );

                SmallVector<Stmt*, 10> children;
                children.append(s->fields().begin(), s->fields().end());
                children.append(s->scope()->inits.begin(), s->scope()->inits.end());
                PrintChildren(children);
            } else {
                print("%3({}%) = {}\n", td->name, td->type->print());
            }
        } break;

        case Kind::TypeExpr:
            PrintBasicHeader(e, "TypeExpr");
            print(" {}\n", cast<TypeExpr>(e)->value->print());
            break;

        case Kind::UnaryExpr: {
            auto u = cast<UnaryExpr>(e);
            PrintBasicNode(e, "UnaryExpr", [&] { print("%1({}%)", u->op); });
            PrintChildren(u->arg);
        } break;

        case Kind::WhileStmt: {
            auto w = cast<WhileStmt>(e);
            PrintBasicNode(e, "WhileStmt");
            SmallVector<Stmt*, 2> children{w->cond, w->body};
            PrintChildren(children);
        } break;
    }
}

void Stmt::dump(bool use_colour) const {
    Printer(use_colour, const_cast<Stmt*>(this));
}

// ============================================================================
//  Checks
// ============================================================================
#define AST_NODE(node)                                                                     \
    static_assert(alignof(node) < __STDCPP_DEFAULT_NEW_ALIGNMENT__, "Alignment to large"); \
    static_assert(__is_trivially_destructible(node), "AST nodes must be trivially destructible");
#include "srcc/AST.inc"
