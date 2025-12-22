#include <srcc/AST/AST.hh>
#include <srcc/AST/Printer.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/AST/Type.hh>
#include <srcc/CG/Target/Target.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Core.hh>

#include <clang/Frontend/CompilerInstance.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/IR/Module.h>
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
    //     proc stringifier (inout __src_assert_msg_buf);
    // }
    AbortInfoEquivalentTy = TupleType::Get(
        *this,
        {StrLitTy, Type::IntTy, Type::IntTy, StrLitTy, StrLitTy, ClosureEquivalentTupleTy}
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
        InheritedProcedureProperties(),
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
    print("%1({}%) %4({}%)", name, static_cast<void*>(s));

    if (printer_context_hack) {
        auto lc = s->loc.seek_line_column(*printer_context_hack);
        if (lc) {
            print(" %5(<{}:{}>%)", lc->line, lc->col);
            return;
        }

    }

    // Fall back to printing just the position.
    print(" %5(<{}>%)", s->loc.encode());
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

    auto PrintProps = [&](const InheritedProcedureProperties props) {
        if (props.is_compile_time_only) print(" compile_time");
        if (props.always_inline) print(" inline");
        if (+props.mangling_number) print(" %3(#{}%)", +props.mangling_number);
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
            if (auto str = a->stringifier.get_or_null()) children.push_back(str);
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

        [&](BreakContinueExpr* b) {
            PrintBasicNode(e, "BreakContinueExpr", [&] {
                print(
                    "%1({}%) %3(#{}%)",
                    b->is_continue ? "continue"sv : "break"sv,
                    +b->target_loop
                );
            }, false);
        },

        [&](BuiltinCallExpr* c) {
            PrintBasicNode(e, "BuiltinCallExpr", [&] {
                print("%2({}%)", BuiltinCallExpr::ToString(c->builtin));
            });

            PrintChildren<Expr*>(c->args());
        },

        [&](BuiltinMemberAccessExpr* m) {
            PrintBasicNode(e, "BuiltinMemberAccessExpr", [&] {
                print("%3({}%)", enchantum::to_string(m->access_kind));
            });
            PrintChildren(m->operand);
        },

        [&](CallExpr* c) {
            PrintBasicNode(e, "CallExpr", [&] {
                if (c->is_inline) print("inline");
            });

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
                case CastExpr::OptionalUnwrap: print("unwrap"); break;
                case CastExpr::OptionalWrap: print("value->optional"); break;
                case CastExpr::MaterialisePoisonValue: print("poison {}", VCLowercase(c->value_category)); break;
                case CastExpr::NilToOptional: print("nil->optional"); break;
                case CastExpr::NilToPointer: print("nil->pointer"); break;
                case CastExpr::Pointer: print("pointer"); break;
                case CastExpr::Range: print("range->range"); break;
                case CastExpr::SliceFromArray: print("array->slice"); break;
            }
            print("\n");
            PrintChildren(c->arg);
        },

        [&](ConstExpr* c) {
            auto PrintExtra = [this, c] {
                print("{} %1(:%) {}", c->value->print(printer_context_hack), c->type);
            };

            PrintBasicNode(e, "ConstExpr", PrintExtra, false);
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
            PrintBasicNode(e, "ForStmt", [&] { print("%3(#{}%)", +f->token); });
            SmallVector<Stmt*> children;
            if (auto v = f->enum_var.get_or_null()) children.push_back(v);
            llvm::append_range(children, f->vars());
            llvm::append_range(children, f->ranges());
            children.push_back(f->body);
            PrintChildren(children);
        },

        [&](GlobalDecl* g) {
            PrintBasicNode(e, "GlobalDecl");
        },

        [&](GlobalRefExpr* d) {
            auto PrintName = [&] {
                print(
                    "%8({}%)",
                    d->decl->name.empty() ? "<anonymous>" : d->decl->name.str()
                );
            };

            PrintBasicNode(e, "GlobalRefExpr", PrintName);
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

        [&](QuoteExpr* q) {
            PrintBasicNode(q, "QuoteExpr");
            SmallVector<Stmt*> children;
            // FIXME: Support printing parse tree nodes in the AST.
            // children.push_back(q->quoted);
            append_range(children, q->unquotes());
            PrintChildren(children);
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
            auto PrintName = [&] {
                print(
                    "%{}({}%)",
                    is_param ? '4' : '8',
                    d->decl->name.empty() ? "<anonymous>" : d->decl->name.str()
                );
            };

            PrintBasicNode(e, "LocalRefExpr", PrintName);
        },

        [&](LoopExpr* l) {
            PrintBasicNode(e, "LoopExpr", [&] { print("%3(#{}%)", +l->token); });
            if (auto b = l->body.get_or_null()) PrintChildren(b);
        },

        [&](MatchExpr* m) {
            auto PrintPattern = [&](const MatchCase::Pattern& p) {
                if (p.is_wildcard()) print("%1(Wildcard%)\n");
                else Print(p.expr());
            };

            PrintBasicNode(e, "MatchExpr");
            SmallVector<Child> children;
            if (auto c = m->control_var().get_or_null())
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

        [&](NilExpr* m) {
            PrintBasicNode(e, "NilExpr");
        },

        [&](OptionalNilTestExpr* o) {
            PrintBasicNode(e, "OptionalNilTestExpr", [this, o]{
                print("%1({}%)", o->is_equal ? "=="sv : "!="sv);
            });

            SmallVector<Stmt*> children;
            children.push_back(o->optional);
            children.push_back(o->nil);
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
            PrintProps(p->props);
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
            print(" %2({}%)", utils::Escape(p->name.str(), false, true));
            PrintProps(p->props);
            print("\n");
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

        [&](SliceConstructExpr* s) {
            PrintBasicNode(e, "SliceConstructExpr");
            SmallVector<Stmt*, 4> children;
            children.push_back(s->ptr);
            children.push_back(s->size);
            PrintChildren(children);
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
            PrintBasicNode(e, "UnaryExpr", [&] {
                print("%1({}%)%3({}%)", u->op, u->postfix ? " postfix" : "");
            });

            PrintChildren(u->arg);
        },

        [&](ValueDecl* d) {
            PrintBasicNode(e, "ValueDecl");
            PrintChildren(d->value);
        },

        [&](WhileStmt* w) {
            PrintBasicNode(e, "WhileStmt", [&] { print("%3(#{}%)", +w->token); });
            SmallVector<Stmt*, 2> children{w->cond, w->body};
            PrintChildren(children);
        },

        [&](WithStmt* w) {
            PrintBasicNode(e, "WithStmt");
            SmallVector<Stmt*, 2> children;
            if (auto var = w->temporary_var.get_or_null()) children.push_back(var);
            children.push_back(w->body);
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
