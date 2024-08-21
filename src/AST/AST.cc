module;

#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <print>
#include <srcc/Macros.hh>

module srcc.ast;
using namespace srcc;

// ============================================================================
//  TU
// ============================================================================
TranslationUnit::TranslationUnit(Context& ctx, String name, bool is_module)
    : ctx{ctx},
      name{name},
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
    using enum utils::Colour;
    bool c = context().use_colours();
    utils::Colours C{c};

    // Print preamble.
    std::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);

    // Print content.
    for (auto s : file_scope_block->stmts()) s->dump(c);
}

// ============================================================================
//  Printer
// ============================================================================
struct Stmt::Printer : PrinterBase<Stmt> {
    using enum utils::Colour;
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
    std::print(
        "{}{} {}{} {}<{}>",
        C(Red),
        name,
        C(Blue),
        static_cast<void*>(s),
        C(Magenta),
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

    if (auto e = dyn_cast<Expr>(s); e and print_type) std::print(" {}", e->type.print(C.use_colours));
    if (print_extra_data) {
        std::print("{} ", C(Reset));
        print_extra_data();
    }

    if (auto e = dyn_cast<Expr>(s); e and print_type) {
        std::print("{}", C(White));
        if (e->value_category != Expr::SRValue) {
            switch (e->value_category) {
                case ValueCategory::SRValue: Unreachable();
                case ValueCategory::MRValue: std::print(" mrvalue"); break;
                case ValueCategory::LValue: std::print(" lvalue"); break;
                case ValueCategory::DValue: std::print(" dvalue"); break;
            }
        }
    }

    if (s->dependent()) std::print(" dependent");
    std::print("\n");
}

void Stmt::Printer::Print(Stmt* e) {
    switch (e->kind()) {
        case Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);
            PrintBasicNode(e, "BinaryExpr", [&] { std::print("{}{}", C(Red), b->op); });
            SmallVector<Stmt*, 2> children{b->lhs, b->rhs};
            PrintChildren(children);
        } break;

        case Kind::BlockExpr:
            PrintBasicNode(e, "BlockExpr");
            PrintChildren(cast<BlockExpr>(e)->stmts());
            break;

        case Kind::BuiltinCallExpr: {
            auto& c = *cast<BuiltinCallExpr>(e);
            PrintBasicNode(e, "BuiltinCallExpr", [&] {
                std::print("{}{}", C(Green), [&] -> std::string_view {
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
            std::print(" {}{} ", c->type.print(C.use_colours), C(White));
            switch (c->kind) {
                case CastExpr::LValueToSRValue: std::print("lvalue->srvalue"); break;
            }
            std::print("\n");
            PrintChildren(c->arg);
        } break;

        case Kind::ConstExpr: {
            auto c = cast<ConstExpr>(e);
            PrintBasicNode(e, "ConstExpr", [&] { c->value->dump(C.use_colours); });
            if (c->stmt) PrintChildren(c->stmt.get());
        } break;

        case Kind::EvalExpr: {
            PrintBasicNode(e, "EvalExpr");
            PrintChildren(cast<EvalExpr>(e)->stmt);
        } break;

        case Kind::IntLitExpr: {
            // These always come straight from the parser and are thus
            // always unsigned (because negative literals don’t exist;
            // unary minus is just an operator).
            auto i = cast<IntLitExpr>(e);
            auto PrintValue = [&] { std::print("{}{}", C(Magenta), i->storage.str(false)); };
            PrintBasicNode(e, "IntLitExpr", PrintValue);
        } break;

        case Kind::LocalDecl:
        case Kind::ParamDecl: {
            bool is_param = e->kind() == Kind::ParamDecl;
            auto d = cast<LocalDecl>(e);
            auto PrintNameAndType = [&] {
                std::print(
                    "{}{} {}",
                    C(is_param ? Blue : White),
                    d->name,
                    d->type.print(C.use_colours)
                );
            };

            PrintBasicNode(e, is_param ? "ParamDecl" : "LocalDecl", PrintNameAndType);
            if (auto init = d->init.get_or_null()) PrintChildren(init);
        } break;

        case Kind::LocalRefExpr: {
            auto d = cast<LocalRefExpr>(e);
            bool is_param = d->decl->kind() == Kind::ParamDecl;
            auto PrintName = [&] { std::print("{}{}", C(is_param ? Blue : White), d->decl->name); };
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
            std::print(
                " {}{} {}",
                C(Green),
                p->name,
                p->type.print(C.use_colours)
            );

            if (p->instantiated_from) std::print(" instantiation");
            std::print("\n");
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
            std::print(
                " {}{}\n",
                C(Green),
                p->decl->name
            );

            tempset print_procedure_bodies = false;
            PrintChildren(p->decl);
        } break;

        case Kind::ReturnExpr: {
            auto ret = cast<ReturnExpr>(e);
            PrintBasicNode(e, "ReturnExpr", [ret] { if (ret->implicit) std::print("implicit"); }, false);
            if (auto expr = ret->value.get_or_null()) PrintChildren(expr);
        } break;

        case Kind::SliceDataExpr:
            PrintBasicNode(e, "SliceDataExpr");
            PrintChildren(cast<SliceDataExpr>(e)->slice);
            break;

        case Kind::StrLitExpr: {
            PrintBasicHeader(e, "StrLitExpr");
            std::print(
                " {}\"{}\"\n",
                C(Yellow),
                utils::Escape(cast<StrLitExpr>(e)->value)
            );
        } break;

        case Kind::TemplateTypeDecl: {
            auto t = cast<TemplateTypeDecl>(e);
            PrintBasicHeader(t, "TemplateTypeDecl");
            std::print(" {}${}\n", C(Yellow), t->name);
        } break;

        case Kind::TypeExpr:
            PrintBasicHeader(e, "TypeExpr");
            std::print(" {}\n", cast<TypeExpr>(e)->value->print(C.use_colours));
            break;

        case Kind::UnaryExpr: {
            auto u = cast<UnaryExpr>(e);
            PrintBasicNode(e, "UnaryExpr", [&] { std::print("{}{}", C(Red), u->op); });
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
