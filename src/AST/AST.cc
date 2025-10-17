#include <srcc/AST/AST.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Core.hh>

#include <clang/Frontend/CompilerInstance.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/TargetParser/Host.h>
#include <base/Utils.hh>

using namespace srcc;

namespace {
// Hack to access location information in the printer.
thread_local const Context* printer_context_hack;
}

// ============================================================================
//  TU
// ============================================================================
TranslationUnit::~TranslationUnit() = default;
TranslationUnit::TranslationUnit(Context& ctx, const LangOpts& opts, StringRef name, bool is_module)
    : ctx{ctx},
      language_opts{opts},
      name{save(name)},
      is_module{is_module} {
    // Get information about the compilation target from Clang.
    std::array args {
        "-x",
        "c++",
        "foo.cc",
        "-triple",
        ctx.triple().getTriple().c_str(),
    };

    // FIXME: Remove the dependence on 'CompilerInstance'/'CompilerInvocation' and just
    // create the target directly.
    ci = std::make_unique<clang::CompilerInstance>();
    ci->createDiagnostics();
    Assert(clang::CompilerInvocation::CreateFromArgs(ci->getInvocation(), args, ci->getDiagnostics()));
    Assert(ci->createTarget());
    tgt = Target::Create(ci->getTargetPtr());
    vm.init(*tgt);

    // Create the global scope.
    create_scope(nullptr);

    // Initialise integer types.
    I8Ty = IntType::Get(*this, Size::Bits(8));
    I16Ty = IntType::Get(*this, Size::Bits(16));
    I32Ty = IntType::Get(*this, Size::Bits(32));
    I64Ty = IntType::Get(*this, Size::Bits(64));
    I128Ty = IntType::Get(*this, Size::Bits(128));

    // Initialise FFI types.
    FFICharTy = IntType::Get(*this, Size::Bits(target().clang().getCharWidth()));
    FFIWCharTy = IntType::Get(*this, Size::Bits(target().clang().getWCharWidth()));
    FFIShortTy = IntType::Get(*this, Size::Bits(target().clang().getShortWidth()));
    FFIIntTy = IntType::Get(*this, Size::Bits(target().clang().getIntWidth()));
    FFILongTy = IntType::Get(*this, Size::Bits(target().clang().getLongWidth()));
    FFILongLongTy = IntType::Get(*this, Size::Bits(target().clang().getLongLongWidth()));

    // Initialise other cached types.
    StrLitTy = SliceType::Get(*this, I8Ty);
    I8PtrTy = PtrType::Get(*this, I8Ty);
    SliceEquivalentTupleTy = TupleType::Get(*this, {I8PtrTy, Type::IntTy});
    ClosureEquivalentTupleTy = TupleType::Get(*this, {I8PtrTy, I8PtrTy});

    // struct AbortInfo {
    //     i8[] filename;
    //     int line;
    //     int col;
    //     i8[] msg1;
    //     i8[] msg2;
    // }
    AbortInfoEquivalentTy = TupleType::Get(
        *this,
        {StrLitTy, Type::IntTy, Type::IntTy, StrLitTy, StrLitTy}
    );

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
        nullptr,
        ProcType::Get(*this, Type::VoidTy),
        is_module ? save(constants::EntryPointName(name)) : constants::ProgramEntryPoint,
        Linkage::Exported,
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
    tempset printer_context_hack = &context();
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

auto Scope::sorted_decls() -> SmallVector<Decl*> {
    SmallVector<Decl*> ret;
    append_range(ret, decls());
    sort(ret, [](Decl* a, Decl* b) { return a->location() < b->location(); });
    return ret;
}

// ============================================================================
//  Printer
// ============================================================================
struct Stmt::Printer : PrinterBase<Stmt> {
    bool print_procedure_bodies = true;
    bool print_instantiations = true;
    Printer(bool use_colour, Stmt* E) : PrinterBase{use_colour} { Print(E); }
    using PrinterBase::Print;
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
        "%1({}%) %4({}%)",
        name,
        static_cast<void*>(s),
        s->loc.pos
    );

    if (printer_context_hack) {
        auto lc = s->loc.seek_line_column(*printer_context_hack);
        if (lc) {
            print(" %5(<{}:{}>%)", lc->line, lc->col);
            return;
        }

    }

    // Fall back to printing just the position.
    print(" %5(<{}>%)", s->loc.pos);
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
        if (e->value_category != Expr::RValue) {
            print(" lvalue");
        }
    }

    print("\n");
}

void Stmt::Printer::Print(Stmt* e) {
    auto VCLowercase = [&](ValueCategory v) -> String {
        switch (v) {
            case ValueCategory::RValue: return "rvalue";
            case ValueCategory::LValue: return "lvalue";
        }
        return "<invalid value category>";
    };

    e->visit(utils::Overloaded{
        [&](auto* node) {
            PrintBasicNode(e, enchantum::to_string(node->kind()));
        },
        [&](ArrayBroadcastExpr* a) {
            PrintBasicNode(e, "ArrayBroadcastExpr");
            PrintChildren(a->element);
        },

        [&](ArrayInitExpr* a) {
            PrintBasicNode(e, "ArrayInitExpr");
            PrintChildren<Expr*>(a->initialisers());
        },

        [&](AssertExpr* a) {
            PrintBasicNode(e, "AssertExpr");
            SmallVector<Stmt*, 2> children;
            children.push_back(a->cond);
            if (auto msg = a->message.get_or_null()) children.push_back(msg);
            PrintChildren(children);
        },

        [&](BinaryExpr* b) {
            PrintBasicNode(
                e,
                "BinaryExpr",
                [&] { print("%1({}%)", utils::Escape(Spelling(b->op), false, true)); }
            );

            SmallVector<Stmt*, 2> children{b->lhs, b->rhs};
            PrintChildren(children);
        },

        [&](BlockExpr* b) {
            PrintBasicNode(e, "BlockExpr");
            PrintChildren(b->stmts());
        },

        [&](BoolLitExpr* b) {
            PrintBasicNode(e, "BoolLitExpr", [&] { print("%1({}%)", b->value); });
        },

        [&](BuiltinCallExpr* c) {
            PrintBasicNode(e, "BuiltinCallExpr", [&] {
                print("%2({}%)", [&] -> std::string_view {
                    switch (c->builtin) {
                        using B = BuiltinCallExpr::Builtin;
                        case B::Unreachable: return "__srcc_unreachable";
                    }

                    return "<invalid>";
                }());
            });

            PrintChildren<Expr*>(c->args());
        },

        [&](BuiltinMemberAccessExpr* m) {
            PrintBasicNode(e, "BuiltinMemberAccessExpr", [&] {
                using AK = BuiltinMemberAccessExpr::AccessKind;
                auto kind = [&] -> std::string_view {
                    switch (m->access_kind) {
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
            PrintChildren(m->operand);
        },

        [&](CallExpr* c) {
            PrintBasicNode(e, "CallExpr");
            SmallVector<Stmt*, 10> fields;
            if (c->callee) fields.push_back(c->callee);
            if (auto a = c->args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        },

        [&](CastExpr* c) {
            PrintBasicHeader(c, "CastExpr");
            print(" {} ", c->type->print());
            switch (c->kind) {
                case CastExpr::Deref: print("deref"); break;
                case CastExpr::ExplicitDiscard: print("discard"); break;
                case CastExpr::LValueToRValue: print("lvalue->rvalue"); break;
                case CastExpr::Integral: print("int->int"); break;
                case CastExpr::MaterialisePoisonValue: print("poison {}", VCLowercase(c->value_category)); break;
                case CastExpr::Range: print("range->range"); break;
                case CastExpr::SliceFromArray: print("array->slice"); break;
            }
            print("\n");
            PrintChildren(c->arg);
        },

        [&](ConstExpr* c) {
            PrintBasicNode(e, "ConstExpr", [&] { print("{}", c->value->print()); });
            if (c->stmt) PrintChildren(c->stmt.get());
        },

        [&](DeferStmt* d) {
            PrintBasicNode(e, "DeferStmt");
            PrintChildren(d->body);
        },

        [&](EvalExpr* ev) {
            PrintBasicNode(e, "EvalExpr");
            PrintChildren(ev->stmt);
        },

        [&](FieldDecl* f) {
            PrintBasicHeader(e, "FieldDecl");
            print(
                " {}{}%5({}%) %1(offs%) %3({:y}%)\n",
                f->type->print(),
                f->name.empty() ? ""sv : " "sv,
                f->name,
                f->offset
            );
        },

        [&](ForStmt* f) {
            PrintBasicNode(e, "ForStmt");
            SmallVector<Stmt*> children;
            if (auto v = f->enum_var.get_or_null()) children.push_back(v);
            llvm::append_range(children, f->vars());
            llvm::append_range(children, f->ranges());
            children.push_back(f->body);
            PrintChildren(children);
        },

        [&](IfExpr* i) {
            PrintBasicNode(e, "IfExpr");
            SmallVector<Stmt*, 3> children{i->cond, i->then};
            if (auto el = i->else_.get_or_null()) children.push_back(el);
            PrintChildren(children);
        },

        [&](ImportedClangModuleDecl* c) {
            PrintBasicNode(e, "ImportedClangModuleDecl", [&] {
                print(
                    "%3({}%) %1(as%) {}",
                    utils::Escape(utils::join(c->headers()), false, true),
                    c->name
                );
            });
        },

        [&](ImportedSourceModuleDecl* s) {
            PrintBasicNode(e, "ImportedSourceModuleDecl", [&] {
                if (s->linkage_name != s->name) {
                    print("{} %1(as%) {}", s->linkage_name, s->name);
                } else {
                    print("{}", s->name);
                }
            });

            PrintChildren<Decl*>(s->exports.sorted_decls());
        },

        [&](IntLitExpr* i) {
            // These always come straight from the parser and are thus
            // always unsigned (because negative literals don’t exist;
            // unary minus is just an operator).
            auto PrintValue = [&] { print("%5({}%)", i->storage.str(false)); };
            PrintBasicNode(e, "IntLitExpr", PrintValue);
        },

        [&](LocalDecl* d) {
            bool is_param = e->kind() == Kind::ParamDecl;
            auto PrintNameAndType = [&] {
                print("%{}({}%)", is_param ? '4' : '8', d->name);
                if (is_param) print(" %1({}%)", cast<ParamDecl>(d)->intent());
                if (d->captured) print(" captured");
                print(" {}", d->type->print());
            };

            PrintBasicNode(e, is_param ? "ParamDecl" : "LocalDecl", PrintNameAndType);
            if (auto init = d->init.get_or_null()) PrintChildren(init);
        },

        [&](LocalRefExpr* d) {
            bool is_param = d->decl->kind() == Kind::ParamDecl;
            auto PrintName = [&] { print("%{}({}%)", is_param ? '4' : '8', d->decl->name); };
            PrintBasicNode(e, "LocalRefExpr", PrintName);
        },

        [&](LoopExpr* l) {
            PrintBasicNode(e, "LoopExpr");
            if (auto b = l->body.get_or_null()) PrintChildren(b);
        },

        [&](MatchExpr* m) {
            auto PrintPattern = [&](const MatchCase::Pattern& p) {
                if (p.is_wildcard()) print("%1(Wildcard%)\n");
                else Print(p.expr());
            };

            PrintBasicNode(e, "MatchExpr");
            SmallVector<Child> children;
            if (auto c = m->control_expr().get_or_null())
                children.emplace_back([&, c] { Print(c); });
            for (auto& c : m->cases()) {
                children.emplace_back([&] {
                    print("%1(Case%) {}", c.body->type_or_void());
                    if (c.unreachable) print(" unreachable");
                    print("\n");
                    PrintChildren<Child>({
                        Child([&] { PrintPattern(c.cond); }),
                        Child([&] { Print(c.body); }),
                    });
                 });
            }
            PrintChildren<Child>(children);
        },

        [&](MaterialiseTemporaryExpr* m) {
            PrintBasicNode(e, "MaterialiseTemporaryExpr");
            PrintChildren(m->temporary);
        },

        [&](MemberAccessExpr* m) {
            PrintBasicNode(e, "MemberAccessExpr");
            SmallVector<Stmt*, 2> children{m->base};
            children.push_back(m->field);
            PrintChildren(children);
        },

        [&](OverloadSetExpr* o) {
            PrintBasicNode(e, "OverloadSetExpr");
            tempset print_procedure_bodies = false;
            PrintChildren<Decl*>(o->overloads());
        },

        [&](ProcDecl* p) {
            PrintBasicHeader(p, "ProcDecl");
            print(" %2({}%) {}", utils::Escape(p->name.str(), false, true), p->type->print());

            if (p->parent.present()) print(" nested");
            if (p->instantiated_from) print(" instantiation");
            if (p->linkage == Linkage::Exported or p->linkage == Linkage::Reexported) print(" exported");
            if (p->linkage == Linkage::Imported or p->linkage == Linkage::Reexported) print(" imported");
            if (p->has_captures) print(" has_captures");
            if (p->introduces_captures) print(" introduces_captures");
            print("\n");
            if (not print_procedure_bodies) return;

            // Print template parameters and parameters.
            SmallVector<Stmt*> children;
            if (p->instantiated_from) children.push_back(p->instantiated_from);
            if (not p->is_imported()) append_range(children, p->params());

            // Take care we don’t recursively print ourselves when printing our parent.
            tempset print_instantiations = false;

            // And the body, if there is one.
            if (auto body = p->body().get_or_null()) children.push_back(body);
            PrintChildren(children);
        },

        [&](ProcTemplateDecl* p) {
            PrintBasicHeader(p, "ProcTemplateDecl");
            print(" %2({}%)\n", utils::Escape(p->name.str(), false, true));
            if (print_instantiations) PrintChildren<ProcDecl*>(p->instantiations());
        },

        [&](ParenExpr* p) {
            PrintBasicNode(p, "ParenExpr");
            PrintChildren(p->expr);
        },

        [&](ProcRefExpr* p) {
            PrintBasicHeader(p, "ProcRefExpr");
            print(" %2({}%)\n", utils::Escape(p->decl->name.str(), false, true));

            tempset print_procedure_bodies = false;
            PrintChildren(p->decl);
        },

        [&](ReturnExpr* ret) {
            PrintBasicNode(e, "ReturnExpr", [&] { if (ret->implicit) print("implicit"); }, false);
            if (auto expr = ret->value.get_or_null()) PrintChildren(expr);
        },

        [&](StrLitExpr* s) {
            PrintBasicHeader(e, "StrLitExpr");
            print(" %3(\"{}\"%)\n", utils::Escape(s->value, true, true));
        },

        [&](TupleExpr* s) {
            PrintBasicNode(e, "TupleExpr");
            PrintChildren<Expr*>(s->values());
        },

        [&](TemplateTypeParamDecl* t) {
            PrintBasicHeader(t, "TemplateTypeParamDecl");
            print(" %3(${}%) %1(type%) {}\n", t->name, t->arg_type());
        },

        [&](TypeDecl* td) {
            PrintBasicHeader(td, "TypeDecl");
            if (auto s = dyn_cast<StructType>(td->type.ptr())) {
                print(
                    " %1(struct %3({}%) size %3({:y}%)/%3({:y}%) align %3({}%)%)\n",
                    s->name(),
                    s->layout().size(),
                    s->layout().array_size(),
                    s->layout().align()
                );

                SmallVector<Stmt*, 10> children;
                children.append(s->layout().fields().begin(), s->layout().fields().end());
                children.append(s->scope()->inits.begin(), s->scope()->inits.end());
                PrintChildren(children);
            } else {
                print("%3({}%) = {}\n", td->name, td->type->print());
            }
        },

        [&](TypeExpr* t) {
            PrintBasicHeader(e, "TypeExpr");
            print(" {}\n", t->value->print());
        },

        [&](UnaryExpr* u) {
            PrintBasicNode(e, "UnaryExpr", [&] { print("%1({}%)", u->op); });
            PrintChildren(u->arg);
        },

        [&](WhileStmt* w) {
            PrintBasicNode(e, "WhileStmt");
            SmallVector<Stmt*, 2> children{w->cond, w->body};
            PrintChildren(children);
        },
    });
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
