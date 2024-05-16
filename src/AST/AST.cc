module;

#include <fmt/format.h>
#include <srcc/Macros.hh>

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
      DependentTy{new(*this) BuiltinType(BuiltinKind::Dependent)},
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
    initialiser_proc = new (*this) ProcDecl(
        ProcType::Get(*this, VoidTy),
        is_module ? save(constants::EntryPointName(name)) : constants::ProgramEntryPoint,
        is_module ? Linkage::Internal : Linkage::Exported,
        Mangling::None,
        nullptr,
        nullptr,
        {}
    );

    procs.push_back(initialiser_proc);
}

void Module::dump() const {
    using enum utils::Colour;
    bool c = context().use_colours();
    utils::Colours C{c};

    // Print preamble.
    fmt::print("{}{} {}{}\n", C(Red), is_module ? "Module" : "Program", C(Green), name);

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
    void PrintBasicHeader(Stmt* S, StringRef name);
    void Print(Stmt* E);
};

void Stmt::Printer::PrintBasicHeader(Stmt* s, StringRef name) {
    fmt::print(
        "{}{} {}{} {}<{}>",
        C(Red),
        name,
        C(Blue),
        fmt::ptr(s),
        C(Magenta),
        s->loc.pos
    );

    if (auto e = dyn_cast<Expr>(s)) fmt::print(" {}", e->type.print(C.use_colours));
    fmt::print("\n");
}


void Stmt::Printer::Print(Stmt* e) {
    switch (e->kind()) {
        case Kind::BlockExpr:
            PrintBasicHeader(e, "BlockExpr");
            PrintChildren(cast<BlockExpr>(e)->stmts());
            break;

        case Kind::CallExpr: {
            PrintBasicHeader(e, "CallExpr");
            auto& c = *cast<CallExpr>(e);
            SmallVector<Stmt*, 10> fields;
            if (c.callee) fields.push_back(c.callee);
            if (auto a = c.args(); not a.empty()) fields.append(a.begin(), a.end());
            PrintChildren(fields);
        } break;

        case Kind::ProcRefExpr: {
            auto& p = *cast<ProcRefExpr>(e);
            fmt::print(
                "{}ProcRefExpr {}{} {}<{}> {}{}\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Green),
                p.decl->name
            );

            tempset print_procedure_bodies = false;
            PrintChildren(p.decl);
        } break;

        case Kind::SliceDataExpr:
            PrintBasicHeader(e, "SliceDataExpr");
            PrintChildren(cast<SliceDataExpr>(e)->slice);
            break;

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
            auto& p = *cast<ProcDecl>(e);
            fmt::print(
                "{}ProcDecl {}{} {}<{}> {}{} {}\n",
                C(Red),
                C(Blue),
                fmt::ptr(e),
                C(Magenta),
                e->loc.pos,
                C(Green),
                p.name,
                p.type.print(C.use_colours)
            );

            if (print_procedure_bodies) PrintChildren(p.body);
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
