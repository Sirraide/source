module;

#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <print>
#include <srcc/Macros.hh>

module srcc.ast;
import base.colours;
using namespace srcc;

// ============================================================================
//  TU
// ============================================================================
TranslationUnit::TranslationUnit(Context& ctx, const LangOpts& opts, StringRef name, bool is_module)
    : ctx{ctx},
      language_opts{opts},
      name{save(name)},
      is_module{is_module},
      FFIBoolTy{Type::UnsafeNull()},
      FFICharTy{Type::UnsafeNull()},
      FFIShortTy{Type::UnsafeNull()},
      FFIIntTy{Type::UnsafeNull()},
      FFILongTy{Type::UnsafeNull()},
      FFILongLongTy{Type::UnsafeNull()},
      FFISizeTy{Type::UnsafeNull()},
      I8Ty{Type::UnsafeNull()},
      I16Ty{Type::UnsafeNull()},
      I32Ty{Type::UnsafeNull()},
      I64Ty{Type::UnsafeNull()},
      I128Ty{Type::UnsafeNull()},
      StrLitTy{Type::UnsafeNull()} {
    // Initialise integer types.
    I8Ty = IntType::Get(*this, Size::Bits(8));
    I16Ty = IntType::Get(*this, Size::Bits(16));
    I32Ty = IntType::Get(*this, Size::Bits(32));
    I64Ty = IntType::Get(*this, Size::Bits(64));
    I128Ty = IntType::Get(*this, Size::Bits(128));

    // Initialise FFI types.
    // TODO: Get type size from Clang.
    FFIBoolTy = I8Ty;
    FFICharTy = I8Ty;
    FFIShortTy = I16Ty;
    FFIIntTy = I32Ty;
    FFILongTy = I64Ty;
    FFILongLongTy = I64Ty;
    FFISizeTy = FFILongTy;

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
        ProcType::Get(*this, Types::VoidTy),
        is_module ? save(constants::EntryPointName(name)) : constants::ProgramEntryPoint,
        is_module ? Linkage::Internal : Linkage::Exported,
        Mangling::None,
        nullptr,
        {}
    );
}

void TranslationUnit::dump() const {
    bool c = context().use_colours();

    // Print preamble.
    utils::Print(c, "%1({}) %2({})\n", is_module ? "Module" : "Program", name);

    // Print content.
    for (auto s : file_scope_block->stmts()) s->dump(c);
}

// ============================================================================
//  Printer
// ============================================================================
struct Stmt::Printer : PrinterBase<Stmt> {
    bool print_procedure_bodies = true;
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
        "%1({}) %4({}) %5(<{}>)",
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
                case ValueCategory::DValue: print(" dvalue"); break;
            }
        }
    }

    if (s->errored()) print(" has-error");
    if (s->dependent()) print(" dependent");
    print("\n");
}

void Stmt::Printer::Print(Stmt* e) {
    // FIXME: Should be a visitor.
    switch (e->kind()) {
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
            PrintBasicNode(e, "BinaryExpr", [&] { print("%1({})", b->op); });
            SmallVector<Stmt*, 2> children{b->lhs, b->rhs};
            PrintChildren(children);
        } break;

        case Kind::BlockExpr:
            PrintBasicNode(e, "BlockExpr");
            PrintChildren(cast<BlockExpr>(e)->stmts());
            break;

        case Kind::BoolLitExpr:
            PrintBasicNode(e, "BoolLitExpr", [&] { print("%1({})", cast<BoolLitExpr>(e)->value); });
            break;

        case Kind::BuiltinCallExpr: {
            auto& c = *cast<BuiltinCallExpr>(e);
            PrintBasicNode(e, "BuiltinCallExpr", [&] {
                print("%2({})", [&] -> std::string_view {
                    switch (c.builtin) {
                        using B = BuiltinCallExpr::Builtin;
                        case B::Print: return "__srcc_print";
                    }

                    return "<invalid>";
                }());
            });

            PrintChildren<Expr>(c.args());
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
                case CastExpr::LValueToSRValue: print("lvalue->srvalue"); break;
                case CastExpr::Integral: print("int->int"); break;
            }
            print("\n");
            PrintChildren(c->arg);
        } break;

        case Kind::ConstExpr: {
            auto c = cast<ConstExpr>(e);
            PrintBasicNode(e, "ConstExpr", [&] { c->value->dump(use_colour()); });
            if (c->stmt) PrintChildren(c->stmt.get());
        } break;

        case Kind::EvalExpr: {
            PrintBasicNode(e, "EvalExpr");
            PrintChildren(cast<EvalExpr>(e)->stmt);
        } break;

        case Kind::IfExpr: {
            auto i = cast<IfExpr>(e);
            PrintBasicNode(e, "IfExpr");
            SmallVector<Stmt*, 3> children{i->cond, i->then};
            if (auto el = i->else_.get_or_null()) children.push_back(el);
            PrintChildren(children);
        } break;

        case Kind::IntLitExpr: {
            // These always come straight from the parser and are thus
            // always unsigned (because negative literals don’t exist;
            // unary minus is just an operator).
            auto i = cast<IntLitExpr>(e);
            auto PrintValue = [&] { print("%5({})", i->storage.str(false)); };
            PrintBasicNode(e, "IntLitExpr", PrintValue);
        } break;

        case Kind::LocalDecl:
        case Kind::ParamDecl: {
            bool is_param = e->kind() == Kind::ParamDecl;
            auto d = cast<LocalDecl>(e);
            auto PrintNameAndType = [&] {
                print(
                    "%{}({})",
                    is_param ? '4' : '8',
                    d->name
                );
                if (is_param) print(" %1({})", cast<ParamDecl>(d)->intent);
                print(" {}", d->type->print());
            };

            PrintBasicNode(e, is_param ? "ParamDecl" : "LocalDecl", PrintNameAndType);
            if (auto init = d->init.get_or_null()) PrintChildren(init);
        } break;

        case Kind::LocalRefExpr: {
            auto d = cast<LocalRefExpr>(e);
            bool is_param = d->decl->kind() == Kind::ParamDecl;
            auto PrintName = [&] { print("%{}({})", is_param ? '4' : '8', d->decl->name); };
            PrintBasicNode(e, "LocalRefExpr", PrintName);
        } break;

        case Kind::OverloadSetExpr: {
            auto o = cast<OverloadSetExpr>(e);
            PrintBasicNode(e, "OverloadSetExpr");
            tempset print_procedure_bodies = false;
            PrintChildren<ProcDecl>(o->overloads());
        } break;

        case Kind::ParenExpr: {
            PrintBasicNode(e, "ParenExpr");
            PrintChildren(cast<ParenExpr>(e)->expr);
        } break;

        case Kind::ProcDecl: {
            auto p = cast<ProcDecl>(e);
            PrintBasicHeader(p, "ProcDecl");
            print(" %2({}) {}", p->name, p->type->print());

            if (p->errored()) print(" errored");
            if (p->instantiated_from) print(" instantiation");
            print("\n");
            if (not print_procedure_bodies) break;

            // Print template parameters and parameters.
            SmallVector<Stmt*> children{p->template_params()};
            children.append(p->params().begin(), p->params().end());

            // And the body, if there is one.
            if (auto body = p->body().get_or_null()) children.push_back(body);

            // Also, print instantiations.
            for (auto inst : p->owner->template_instantiations[p])
                children.push_back(inst.second);

            PrintChildren(children);
        } break;

        case Kind::ProcRefExpr: {
            auto p = cast<ProcRefExpr>(e);
            PrintBasicHeader(p, "ProcRefExpr");
            print(" %2({})\n", p->decl->name);

            tempset print_procedure_bodies = false;
            PrintChildren(p->decl);
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ReturnExpr>(e);
            PrintBasicNode(e, "ReturnExpr", [&] { if (ret->implicit) print("implicit"); }, false);
            if (auto expr = ret->value.get_or_null()) PrintChildren(expr);
        } break;

        case Kind::SliceDataExpr:
            PrintBasicNode(e, "SliceDataExpr");
            PrintChildren(cast<SliceDataExpr>(e)->slice);
            break;

        case Kind::StrLitExpr: {
            PrintBasicHeader(e, "StrLitExpr");
            print(" %3(\"\002{}\003\")\n", utils::Escape(cast<StrLitExpr>(e)->value));
        } break;

        case Kind::TemplateTypeDecl: {
            auto t = cast<TemplateTypeDecl>(e);
            PrintBasicHeader(t, "TemplateTypeDecl");
            print(" %3(${})\n", t->name);
        } break;

        case Kind::TypeExpr:
            PrintBasicHeader(e, "TypeExpr");
            print(" {}\n", cast<TypeExpr>(e)->value->print());
            break;

        case Kind::UnaryExpr: {
            auto u = cast<UnaryExpr>(e);
            PrintBasicNode(e, "UnaryExpr", [&] { print("%1({})", u->op); });
            PrintChildren(u->arg);
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
