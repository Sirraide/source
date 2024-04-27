module;

#include <fmt/format.h>

module srcc.ast;
using namespace srcc;

// ============================================================================
//  Module
// ============================================================================
Module::Module(Context& ctx, String name, bool is_module)
    : ctx{ctx},
      name{name},
      is_module{is_module},
      VoidTy{new(*this) BuiltinType(BuiltinKind::Void)},
      NoReturnTy{new(*this) BuiltinType(BuiltinKind::NoReturn)},
      BoolTy{new(*this) BuiltinType(BuiltinKind::Bool)} {
    // Initialise FFI and cached types.
    // TODO: Get type size from Clang.
    I8Ty = IntType::Get(*this, 8);
    I16Ty = IntType::Get(*this, 16);
    I32Ty = IntType::Get(*this, 32);
    I64Ty = IntType::Get(*this, 64);
    I128Ty = IntType::Get(*this, 128);
    StrLitTy = SliceType::Get(*this, I8Ty);
    FFIBoolTy = IntType::Get(*this, 8);
    FFICharTy = IntType::Get(*this, 8);
    FFIShortTy = IntType::Get(*this, 16);
    FFIIntTy = IntType::Get(*this, 32);
    FFILongTy = IntType::Get(*this, 64);
    FFILongLongTy = IntType::Get(*this, 64);
    FFISizeTy = FFILongTy;

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
    initialiser_proc = new (*this) ProcDecl(
        ProcType::Get(*this, VoidTy),
        is_module ? save(constants::EntryPointName(name)) : constants::ProgramEntryPoint,
        is_module ? Linkage::Internal : Linkage::Exported,
        Mangling::None,
        nullptr,
        nullptr,
        {}
    );
}

void Module::dump() const {
    using enum utils::Colour;
    bool c = context().use_colours();
    utils::Colours C{c};

    // Print preamble.
    fmt::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);

    // Print content.
    for (auto& xs : exports)
        for (auto& d : xs.second)
            d->dump(c);
}

// ============================================================================
//  Printer
// ============================================================================
struct Stmt::Printer : PrinterBase<Stmt> {
    Printer(bool use_colour, Stmt* E) : PrinterBase{use_colour} { Print(E); }
    void Print(Stmt* E);
};

void Stmt::Printer::Print(Stmt* e) {
    switch (e->kind()) {
        using enum utils::Colour;

        case Kind::BlockExpr: {
            [[maybe_unused]] auto& x = *cast<BlockExpr>(e);

            fmt::print(
                "{}BlockExpr {}{} {}<{}>\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            PrintChildren(x.stmts());
        } break;

        case Kind::CallExpr: {
            [[maybe_unused]] auto& x = *cast<CallExpr>(e);

            fmt::print(
                "{}CallExpr {}{} {}<{}>\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos
            );

            SmallVector<Stmt*, 10> fields;
            if (x.callee) fields.push_back(x.callee);
            if (auto a = x.args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::ProcRefExpr: {
            [[maybe_unused]] auto& x = *cast<ProcRefExpr>(e);

            fmt::print(
                "{}ProcRefExpr {}{} {}<{}> {}{}\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Green),
                x.decl->name
            );

            PrintChildren(x.decl);
        } break;

        case Kind::StrLitExpr: {
            fmt::print(
                "{}StrLitExpr {}{} {}<{}> {}\"{}\"\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Yellow),
                utils::Escape(cast<StrLitExpr>(e)->value)
            );
        } break;

        case Kind::ProcDecl: {
            [[maybe_unused]] auto& x = *cast<ProcDecl>(e);

            fmt::print(
                "{}ProcDecl {}{} {}<{}> {}{} {}\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Green),
                x.name,
                x.type->print(C.use_colours)
            );

            SmallVector<Stmt*, 10> fields;
            if (x.parent) fields.push_back(x.parent);
            if (x.body) fields.push_back(x.body);
            PrintChildren(fields);
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
