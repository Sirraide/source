#include <srcc/AST/Enums.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Serialisation.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Sema.hh>
#include <llvm/Support/Compression.h>
#include <llvm/Support/DynamicLibrary.h>
#include <base/Serialisation.hh>
#include <base/Utils.hh>

using namespace srcc;

#define Read(...) Try(read<__VA_ARGS__>())

namespace {
enum struct TypeIndex : u64;
constexpr u32 CurrentVersion = 1;

struct Header {
    LIBBASE_SERIALISE(Header, version, decls_size, types_size);
    u32 version{};
    u64 decls_size{};
    u64 types_size{};
};

struct EncodedSLoc {
    LIBBASE_SERIALISE(EncodedSLoc, file, offs);
    srcc::File::Id file = -1;
    std::ptrdiff_t offs = 0;
};
}

class Sema::ASTWriter : public base::ser::Writer<std::endian::native> {
public:
    using Base = base::ser::Writer<std::endian::native>;
    using Base::Base;
    using Base::write;

    const Context& ctx;
    SmallVector<Type> types;
    DenseMap<Type, TypeIndex> type_indices;
    llvm::MapVector<File::Id, std::string> files;
    TypeIndex next_index = TypeIndex(1);
    bool done_emitting_decls = false;

    ASTWriter(const Context& ctx, ByteBuffer& buf) : Base{buf}, ctx{ctx} {
        types.push_back(Type());
    }

    void emit_type_def(Type ty) {
        using K = TypeBase::Kind;
        ty->visit(utils::Overloaded{
            [&](const ArrayType* ty) {
                *this << K::ArrayType << ty->elem() << i64(ty->dimension());
            },
            [&](const BuiltinType* ty) {
                *this << K::BuiltinType << ty->builtin_kind();
            },
            [&](const IntType* ty) {
                *this << K::IntType << ty->bit_width();
            },
            [&](const ProcType* ty) {
                *this << K::ProcType << ty->has_c_varargs() << ty->cconv()
                      << ty->ret() << ty->params();
            },
            [&](const SingleElementTypeBase* ty) {
                *this << ty->kind() << ty->elem();
            },
            [&](const TupleType* ty) {
                *this << K::TupleType << ty->layout();
            },
            [&](const StructType* ty) {
                *this << K::StructType << ty->decl()->name.str() << ty->decl()->location()
                      << ty->layout();
            },
        });
    }

    template <typename T>
    void write(ArrayRef<T> vals) {
        *this << vals.size();
        for (auto v : vals) *this << v;
    }

    void write(String s) {
        // This needs a custom deserialiser for interning, so we also have to
        // define a custom serialiser.
        *this << s.size();
        append_bytes(s.data(), s.size());
    }

    void write(DeclName name) {
        *this << name.is_str();
        if (name.is_str()) *this << name.str();
        else *this << name.operator_name();
    }

    void write(SLoc loc) {
        EncodedSLoc e;
        if (auto f = loc.file(ctx)) {
            RegisterFile(*f);
            e.offs = loc.pointer() - f->data();
            e.file = f->file_id();
        }
        *this << e;
    }

    void write(Type ty) {
        *this << RegisterType(ty);
    }

    void write(const ParamTypeData& p) {
        *this << p.intent << p.type << p.variadic;
    }

    void write(FieldDecl* f) {
        *this << f->type << f->offset << f->name.str() << f->location();
    }

    void write(const RecordLayout& l) {
        *this << l.size() << l.array_size() << l.align() << l.bits() << l.fields();
    }

private:
    void RegisterFile(const File& f) {
        files[f.file_id()] = absolute(f.path()).string();
    }

    auto RegisterType(Type ty) -> TypeIndex {
        if (auto it = type_indices.find(ty); it != type_indices.end()) return it->second;

        Assert(not done_emitting_decls, "Forgot to register an element type!");
        type_indices[ty] = TypeIndex(); // Support recursion.

        // Register element types.
        ty->visit(utils::Overloaded{
            [&](const BuiltinType* ty) {},
            [&](const IntType*) {},
            [&](const SingleElementTypeBase* ty) { RegisterType(ty->elem()); },
            [&](const RecordType* ty) {
                for (auto f : ty->layout().field_types())
                RegisterType(f);
            },
            [&](const ProcType* ty) {
                for (auto p : ty->param_types()) RegisterType(p);
                RegisterType(ty->ret());
            },
        });

        // Compute the index for this type.
        auto idx = type_indices[ty] = next_index;
        next_index = TypeIndex(+next_index + 1);
        types.push_back(ty);
        return idx;
    }
};

class Sema::ASTReader : public base::ser::Reader<std::endian::native> {
public:
    using Base = base::ser::Reader<std::endian::native>;
    using Base::Base;

    Sema& S;
    ImportedSourceModuleDecl* module_decl;
    llvm::SmallVector<Type> types{};
    TypeIndex next_index = TypeIndex(1);

    /// Map from serialised file IDs to the file IDs they correspond
    /// to in the current context (provided the file in question still
    /// exists on disk).
    ///
    /// TODO: Do we want to hash the file too to ensure there were no
    /// changes? The source locations would be utter bogus otherwise.
    ///
    /// TODO: Do we want to embed the file data in the module description?
    DenseMap<File::Id, std::optional<File::Id>> files;

    ASTReader(Sema& S, ImportedSourceModuleDecl* module_decl, ByteSpan buf)
        : Base{buf}, S{S}, module_decl{module_decl} {
        types.push_back(Type()); // TypeIndex(0).
    }

    auto read_file_data() -> Result<> {
        auto [index, absolute_path] = Read(std::pair<File::Id, std::string>);
        if (auto f = S.context().try_get_file(absolute_path))
            files[index] = f.value()->file_id();
        return {};
    }

    auto read_decl() -> Result<Decl*> {
        using K = Stmt::Kind;
        switch (auto k = Read(K)) {
            default: Unreachable("Invalid exported decl: {}", +k);
            case K::ProcDecl: {
                auto mangling = Read(Mangling);
                auto type = Read(Type);
                auto name = Read(DeclName);
                auto mnum = Read(ManglingNumber);
                auto loc = Read(SLoc);
                return ProcDecl::Create(
                    *S.tu,
                    module_decl,
                    cast<ProcType>(type),
                    name,
                    Linkage::Imported,
                    mangling,
                    nullptr,
                    mnum,
                    loc
                );
            }

            case K::TypeDecl: {
                auto ty = Read(Type);
                return cast<StructType>(ty)->decl();
            }
        }
    }

    auto read_type() -> Result<Type> {
        using K = TypeBase::Kind;
        switch (Read(K)) {
            case K::ArrayType: {
                auto elem = Read(Type);
                auto dimension = Read(i64);
                return ArrayType::Get(*S.tu, elem, dimension);
            }

            case K::BuiltinType: {
                switch (Read(BuiltinKind)) {
                    case BuiltinKind::Bool: return Type::BoolTy;
                    case BuiltinKind::Deduced: return Type::DeducedTy;
                    case BuiltinKind::UnresolvedOverloadSet: return Type::UnresolvedOverloadSetTy;
                    case BuiltinKind::Int: return Type::IntTy;
                    case BuiltinKind::NoReturn: return Type::NoReturnTy;
                    case BuiltinKind::Tree: return Type::TreeTy;
                    case BuiltinKind::Type: return Type::TypeTy;
                    case BuiltinKind::Void: return Type::VoidTy;
                    case BuiltinKind::Nil: return Type::NilTy;
                }

                Unreachable("Invalid builtin type");
            }

            case K::IntType: {
                return IntType::Get(*S.tu, Read(Size));
            }

            case K::ProcType: {
                bool c_varargs = Read(bool);
                auto cconv = Read(CallingConvention);
                auto ret = Read(Type);
                auto params = Read(SmallVector<ParamTypeData>);
                return ProcType::Get(*S.tu, ret, params, cconv, c_varargs);
            }

            case K::OptionalType: {
                return OptionalType::Get(*S.tu, Read(Type));
            }

            case K::PtrType: {
                return PtrType::Get(*S.tu, Read(Type));
            }

            case K::RangeType: {
                return RangeType::Get(*S.tu, Read(Type));
            }

            case K::SliceType: {
                return SliceType::Get(*S.tu, Read(Type));
            }

            case K::StructType: {
                auto decl_name = Read(String);
                auto decl_loc = Read(SLoc);
                auto layout = Read(RecordLayout*);
                return S.BuildCompleteStructType(decl_name, layout, decl_loc);
            }

            case K::TupleType: {
                auto layout = Read(RecordLayout*);
                return TupleType::Get(*S.tu, layout);
            }
        }

        Unreachable("Invalid type kind");
    }

    template <typename T>
    auto read() -> Result<T> { return Base::read<T>(); }

    template <>
    auto read<DeclName>() -> Result<DeclName> {
        bool is_str = Read(bool);
        if (is_str) return Read(String);
        else return Read(Tk);
    }

    template <>
    auto read<FieldDecl*>() -> Result<FieldDecl*> {
        auto type = Read(Type);
        auto offset = Read(Size);
        auto name = Read(String);
        auto loc = Read(SLoc);
        return new (*S.tu) FieldDecl(type, offset, name, loc);
    }

    template <>
    auto read<SLoc>() -> Result<SLoc> {
        auto e = Read(EncodedSLoc);
        auto it = files.find(e.file);
        if (it == files.end() or not it->second.has_value()) return SLoc();
        auto f = S.ctx.file(it->second.value());
        return SLoc(f->data() + e.offs);
    }

    template <>
    auto read<ParamTypeData>() -> Result<ParamTypeData> {
        auto intent = Read(Intent);
        auto type = Read(Type);
        auto variadic = Read(bool);
        return ParamTypeData{intent, type, variadic};
    }

    template <>
    auto read<RecordLayout*>() -> Result<RecordLayout*> {
        auto size = Read(Size);
        auto array_size = Read(Size);
        auto align = Read(Align);
        auto bits = Read(RecordLayout::Bits);
        auto fields = Read(SmallVector<FieldDecl*>);
        return RecordLayout::Create(*S.tu, fields, size, array_size, align, bits);
    }

    template <>
    auto read<String>() -> Result<String> {
        return S.tu->save(Read(std::string));
    }

    template <>
    auto read<Type>() -> Result<Type> {
        auto idx = Read(TypeIndex);
        Assert(+idx < types.size(), "Invalid type index");
        return types[+idx];
    }
};

auto Sema::ReadAST(ImportedSourceModuleDecl* module_decl, const File& f) -> Result<> {
    ASTReader r{*this, module_decl, ByteSpan{f.data(), usz(f.size())}};
    auto hdr = Try(r.read<Header>());
    if (hdr.version != CurrentVersion) return base::Error(
        "Unsupported version number {}; recompile the module",
        hdr.version
    );

    // Grab the declarations to be read later.
    auto decls = Try(r.read_bytes(hdr.decls_size));

    // As well as the types.
    auto types = Try(r.read_bytes(hdr.types_size));

    // Read the files first so we can construct source locations properly.
    while (r.size() != 0) Try(r.read_file_data());

    // Next, read the types so we can construct declarations properly.
    r.set_data(types);
    while (r.size() != 0) r.types.push_back(Try(r.read_type()));

    // Finally, we have all the data required to read the declarations.
    r.set_data(decls);
    while (r.size() != 0) AddDeclToScope(curr_scope(), Try(r.read_decl()));
    return {};
}

auto Sema::LoadModuleFromArchive(
    String logical_name,
    String linkage_name,
    SLoc import_loc
) -> Ptr<ImportedSourceModuleDecl> {
    if (search_paths.empty()) return ICE(import_loc, "No module search path");

    // Append extension.
    std::string desc_name = std::format("{}.{}", linkage_name, constants::ModuleDescriptionFileExtension);

    // Try to find the module in the search path.
    fs::Path mod_path, desc_path, shared_path;
    for (auto& base : search_paths) {
        auto combined = fs::Path{base} / desc_name;
        if (fs::File::Exists(combined)) {
            mod_path = fs::Path{base} / std::format("{}.{}", linkage_name, constants::ModuleFileExtension);
            desc_path = std::move(combined);
            shared_path = fs::Path{base} / std::format("{}.{}", linkage_name, constants::SharedModuleFileExtension);
            break;
        }
    }

    // Couldn’t find it :(.
    if (desc_path.empty()) {
        Error(import_loc, "Could not find module '{}'", linkage_name);
        Remark("Search paths:\n  {}", utils::join(search_paths, "\n  "));
        return nullptr;
    }

    // Load the archive.
    auto file = context().try_get_file(desc_path);
    if (not file) return Error(
        import_loc,
        "Could not load module '{}' ({}): {}",
        linkage_name,
        desc_path.string(),
        file.error()
    );

    Assert(curr_scope() == global_scope());
    EnterScope _{*this};
    auto module_decl = new (*tu) ImportedSourceModuleDecl(
        *curr_scope(),
        logical_name,
        linkage_name,
        tu->save(mod_path.string()),
        import_loc
    );

    if (auto res = ReadAST(module_decl, file.value()); not res) return Error(
        import_loc,
        "Module '{}' is malformed: {}",
        linkage_name,
        res.error()
    );

    // Also attempt to load the shared object so we can perform compile-time
    // evaluation of any functions defined in the module.
    if (fs::File::Exists(shared_path)) {
        std::string err_msg;
        if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(shared_path.string().c_str(), &err_msg)){
            Warn(import_loc, "Could not load shared object for compile-time evaluation");
            Remark("Compile-time evaluation will not be able to access functions in '{}'", linkage_name);
        }

        auto init = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(constants::EntryPointName(linkage_name));
        if (not init) Warn(import_loc, "Could not find module initialiser for module '{}'", linkage_name);
        else std::invoke(reinterpret_cast<void(*)()>(init));
    }

    return module_decl;
}

void Sema::LoadModule(
    String logical_name,
    ArrayRef<String> linkage_names,
    SLoc import_loc,
    bool is_open,
    bool is_cxx_header
) {
    if (is_open) {
        Assert(is_cxx_header);
        if (auto m = ImportCXXHeaders(logical_name, linkage_names, import_loc).get_or_null())
            tu->open_modules.push_back(m);
        return;
    }

    // Clang imports cannot be cached or redefined.
    auto& logical = tu->logical_imports[logical_name];
    if (isa_and_present<ImportedClangModuleDecl>(logical)) {
        Error(import_loc, "Cannot redefine header import '{}'", logical_name);
        Note(logical->location(), "Previous definition was here");
        return;
    }

    if (is_cxx_header) {
        logical = ImportCXXHeaders(logical_name, linkage_names, import_loc).get_or_null();
        return;
    }

    Assert(linkage_names.size() == 1, "Source module imports should consist of a single physical module");
    auto& link = tu->linkage_imports[linkage_names.front()];
    if (not link) {
        logical = link = LoadModuleFromArchive(logical_name, linkage_names.front(), import_loc).get_or_null();
        return;
    }

    // We’ve imported this before; if the logical name isn’t mapped yet, map it
    // to the same module.
    if (not logical) logical = link;

    // Conversely, complain if this name is already mapped to a different module.
    else if (logical != link) {
        Error(import_loc, "Import with name '{}' conflicts with an existing import", logical_name);
        Note(import_loc, "Import here refers to module '{}'", linkage_names.front());
        Note(link->location(), "But import here refers to module '{}'", link->linkage_name);
    }
}

auto TranslationUnit::serialise() -> ByteBuffer {
    Assert(is_module, "Should never be called for programs");

    ByteBuffer buf;
    Sema::ASTWriter w{context(), buf};
    w << Header(); // Allocate space for the header.

    // Serialise each declaration, and collect the types we need.
    u64 decls_begin = buf.size();
    for (auto d : exports.sorted_decls()) {
        using K = Stmt::Kind;
        if (auto proc = dyn_cast<ProcDecl>(d)) {
            Assert(not proc->parent, "Exporting local procedure?");
            w << K::ProcDecl << proc->mangling
              << proc->type << proc->name << proc->mangling_number
              << proc->location();
            continue;
        }

        if (auto td = dyn_cast<TypeDecl>(d)) {
            // FIXME: Make 'StructDecl' its own thing. 'TypeDecl' should
            // be an abstract class that we shouldn’t be checking for here.
            Assert(not isa<TemplateTypeParamDecl>(td));
            Assert(isa<StructType>(td->type));

            // The name+location is serialised with the struct type.
            w << K::TypeDecl << td->type;
            continue;
        }

        // TODO: For templates, have the parser keep track of the first and
        // last token of the function body and simply write out all the tokens
        // in between.

        d->dump_color();
        Todo("Export this decl");
    }

    u64 decls_end = buf.size();
    w.done_emitting_decls = true;

    // Write the types.
    for (auto ty : ArrayRef(w.types).drop_front()) w.emit_type_def(ty);
    u64 types_end = buf.size();

    // Write the files.
    for (const auto& f : w.files) w << f;

    // Write the actual header.
    {
        ByteBuffer tmp_buf;
        ByteWriter tmp{tmp_buf};
        Header hdr{};
        hdr.version = CurrentVersion;
        hdr.decls_size = decls_end - decls_begin;
        hdr.types_size = types_end - decls_end;
        tmp << hdr;
        std::memcpy(buf.data(), tmp_buf.data(), tmp_buf.size());
    }

    return buf;
}
