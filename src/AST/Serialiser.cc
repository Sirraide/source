#include <srcc/AST/AST.hh>
#include <srcc/AST/Stmt.hh>
#include <srcc/Core/Constants.hh>
#include <srcc/Core/Diagnostics.hh>
#include <srcc/Macros.hh>

#include <llvm/Object/Archive.h>
#include <llvm/Object/ObjectFile.h>

using namespace srcc;

using Serialiser = TranslationUnit::Serialiser;
using Deserialiser = TranslationUnit::Deserialiser;

// ============================================================================
//  Serialiser
// ============================================================================
struct TranslationUnit::Serialiser {
    const TranslationUnit& M;
    SmallVectorImpl<char>& buffer;
    SmallVector<char, 0> types_buffer{};
    SmallVector<char, 0> decls_buffer{};
    DenseMap<void*, u64> type_indices{};
    u64 serialised_decls = 0;

    struct Writer {
        SmallVectorImpl<char>& buffer;
        Writer(SmallVectorImpl<char>& buffer) : buffer{buffer} {}

        template <typename T>
        requires (std::is_enum_v<T> or std::integral<T>)
        auto operator<<(T val) -> Writer& {
            u64 v = u64(val);
            buffer.resize(buffer.size() + sizeof(v));
            std::memcpy(buffer.end() - sizeof(v), &v, sizeof(v));
            return *this;
        }

        auto operator<<(StringRef s) -> Writer& {
            *this << s.size();
            buffer.resize(buffer.size() + s.size());
            std::memcpy(buffer.end() - s.size(), s.data(), s.size());
            return *this;
        }

        auto operator<<(Size s) -> Writer& {
            *this << s.bits();
            return *this;
        }
    };

    Serialiser(const TranslationUnit& M, SmallVectorImpl<char>& buffer);
    void SerialiseDecl(const Decl*);
    void SerialiseFieldDecl(const FieldDecl*);
    void SerialiseLocalDecl(const LocalDecl*);
    void SerialiseParamDecl(const ParamDecl*);
    void SerialiseProcDecl(const ProcDecl* proc);
    void SerialiseProcTemplateDecl(const ProcTemplateDecl* proc);
    void SerialiseTemplateTypeParamDecl(const TemplateTypeParamDecl*);
    auto SerialiseType(Type ty) -> u64;
    void SerialiseTypeDecl(const TypeDecl*);
};

Serialiser::Serialiser(const TranslationUnit& M, SmallVectorImpl<char>& buffer)
    : M{M},
      buffer{buffer} {
    for (const auto& exports : M.exports.decls_by_name)
        for (auto d : exports.second)
            SerialiseDecl(d);

    // TODO: Add compiler git hash + compiler build time.
    // TODO: Add mtimes of all source files (and C++ headers).
    // TODO: Compress with ZSTD.
    SmallVector<char, 0> combined;
    Writer W{buffer};
    W << M.name;
    W << type_indices.size();
    W << serialised_decls;
    buffer.append(types_buffer.begin(), types_buffer.end());
    buffer.append(decls_buffer.begin(), decls_buffer.end());
}

void Serialiser::SerialiseDecl(const Decl* d) {
    Writer{decls_buffer} << d->kind();
    serialised_decls++;
    switch (d->kind()) { // clang-format off
        using K = Stmt::Kind;
#       define AST_STMT_LEAF(node) case K::node: Unreachable("Not a declaration!");
#       define AST_DECL_LEAF(node) case K::node: SRCC_CAT(Serialise, node)(cast<node>(d)); return;
#       include "srcc/AST.inc"
    } // clang-format on

    Unreachable("Invalid statement kind");
}

void Serialiser::SerialiseFieldDecl(const FieldDecl*) {
    Todo();
}

void Serialiser::SerialiseLocalDecl(const LocalDecl*) {
    Todo();
}

void Serialiser::SerialiseParamDecl(const ParamDecl*) {
    Todo();
}

void Serialiser::SerialiseProcDecl(const ProcDecl* proc) {
    Assert(not proc->parent, "Exporting local procedure?");
    Writer W{decls_buffer};
    W << SerialiseType(proc->type);
    W << proc->name;
    W << proc->mangling;
    for (auto& param : proc->params()) SerialiseDecl(param);
}

void Serialiser::SerialiseProcTemplateDecl(const ProcTemplateDecl*) {
    Todo();
}

void Serialiser::SerialiseTemplateTypeParamDecl(const TemplateTypeParamDecl*) {
    Unreachable("Should not exist in the AST");
}

auto Serialiser::SerialiseType(Type ty) -> u64 {
    // Avoid serialising the same type twice.
    auto it = type_indices.find(ty.ptr());
    if (it != type_indices.end()) return it->second;

    // First, ensure types that this depends on are serialised.
    // TODO: For structs, we need to figure out some way to do
    // recursion (maybe just add a dummy entry first?).
    switch (ty->kind()) {
        case TypeBase::Kind::BuiltinType:
        case TypeBase::Kind::IntType:
            break;

        case TypeBase::Kind::SliceType:
        case TypeBase::Kind::ReferenceType:
        case TypeBase::Kind::ArrayType:
            SerialiseType(cast<SingleElementTypeBase>(ty)->elem());
            break;

        case TypeBase::Kind::ProcType: {
            auto proc = cast<ProcType>(ty);
            SerialiseType(proc->ret());
            for (const auto& p : proc->params()) SerialiseType(p.type);
        } break;

        case TypeBase::Kind::StructType: {
            Todo();
        }
    }

    // The current index will be the index of this type.
    auto idx = u64(type_indices.size());
    type_indices[ty.ptr()] = idx;

    // Next, serialise the type.
    Writer W{types_buffer};
    W << ty->kind();
    switch (ty->kind()) {
        case TypeBase::Kind::SliceType:
        case TypeBase::Kind::ReferenceType:
            W << SerialiseType(cast<SingleElementTypeBase>(ty)->elem());
            return idx;

        case TypeBase::Kind::BuiltinType:
            W << cast<BuiltinType>(ty)->builtin_kind();
            return idx;

        case TypeBase::Kind::IntType:
            W << cast<IntType>(ty)->bit_width().bits();
            return idx;

        case TypeBase::Kind::ArrayType: {
            auto arr = cast<ArrayType>(ty);
            W << SerialiseType(arr->elem());
            W << arr->dimension();
            return idx;
        }

        case TypeBase::Kind::ProcType: {
            auto proc = cast<ProcType>(ty);
            auto params = proc->params();
            W << proc->cconv();
            W << proc->variadic();
            W << params.size();
            W << SerialiseType(proc->ret());
            for (const auto& p : params) {
                W << p.intent;
                W << SerialiseType(p.type);
            }
            return idx;
        }

        case TypeBase::Kind::StructType: {
            Todo();
        }
    }

    Unreachable("Invalid type kind: {}", ty);
}

void Serialiser::SerialiseTypeDecl(const TypeDecl*) {
    Todo();
}

// ============================================================================
//  Deserialiser
// ============================================================================
struct TranslationUnit::Deserialiser : DefaultDiagsProducer<> {
    Context& ctx;
    Ptr M;
    ArrayRef<char> data;
    SmallVector<Type, 0> deserialised_types;

    template <typename T>
    requires (std::is_enum_v<T> or std::integral<T>)
    auto Read() -> T {
        u64 v;
        std::memcpy(&v, data.data(), sizeof(v));
        data = data.drop_front(sizeof(v));
        return T(v);
    }

    auto ReadInt() -> u64 { return Read<u64>(); }

    auto ReadSize() -> Size {
        u64 bits = Read<u64>();
        return Size::Bits(bits);
    }

    auto ReadString() -> String {
        SmallString<256> buf;
        u64 size = ReadInt();
        buf.resize(size);
        std::memcpy(buf.data(), data.data(), size);
        data = data.drop_front(size);
        return M->save(buf);
    }

    template <typename Ty = TypeBase>
    auto ReadType() -> Ty* {
        u64 idx = ReadInt();
        Assert(idx < deserialised_types.size(), "Type index out of bounds");
        return cast<Ty>(deserialised_types[idx].ptr());
    }

    Deserialiser(Context& ctx, ArrayRef<char> data = {})
        : ctx{ctx},
          M{TranslationUnit::CreateEmpty(ctx, LangOpts{})}, // FIXME: Serialise and deserialise lang opts.
          data{data} {}

    auto DeserialiseFromArchive(StringRef name, StringRef path, Location import_loc) -> Opt<TranslationUnit::Ptr>;
    auto Deserialise() -> TranslationUnit::Ptr;
    auto DeserialiseDecl() -> Decl*;
    auto DeserialiseFieldDecl() -> Decl*;
    auto DeserialiseLocalDecl() -> Decl*;
    auto DeserialiseParamDecl() -> Decl*;
    auto DeserialiseProcDecl() -> Decl*;
    auto DeserialiseProcTemplateDecl() -> Decl*;
    auto DeserialiseTemplateTypeParamDecl() -> Decl*;
    void DeserialiseType();
    auto DeserialiseTypeDecl() -> Decl*;
};

auto Deserialiser::Deserialise() -> TranslationUnit::Ptr {
    M->name = ReadString();
    u64 serialised_types = ReadInt();
    u64 serialised_decls = ReadInt();
    for (u64 i = 0; i < serialised_types; i++) DeserialiseType();
    for (u64 i = 0; i < serialised_decls; i++) {
        auto decl = DeserialiseDecl();
        M->exports.decls_by_name[decl->name].push_back(decl);
    }
    return std::move(M);
}

auto Deserialiser::DeserialiseFromArchive(
    StringRef name,
    StringRef path,
    Location import_loc
) -> Opt<TranslationUnit::Ptr> {
    auto Err = [&]<typename... Args>(std::string message) -> std::nullopt_t {
        Error(
            import_loc,
            "Error reading module '{}' ({}): {}",
            name,
            path,
            message
        );

        return std::nullopt;
    };

    // Load the archive.
    M->import_path = M->save(path);
    auto file = llvm::MemoryBuffer::getFile(path);
    if (not file) return Err(file.getError().message());

    // Parse it.
    auto archive = llvm::object::Archive::create(*file.get());
    if (auto e = archive.takeError()) return Err(toString(std::move(e)));

    // Iterate through all the children (which should be
    // object files) to find the one with the same name
    // as the module we’re looking for.
    auto e = llvm::Error::success();
    std::unique_ptr<llvm::object::Binary> bin;
    for (auto& ch : archive.get()->children(e)) {
        auto child_name = ch.getName();
        if (not child_name) Err(utils::FormatError(e));

        // Found our module!
        if (child_name.get() == name) {
            auto expected_bin = ch.getAsBinary();
            if (auto err = expected_bin.takeError()) return Err(toString(std::move(err)));
            bin = std::move(expected_bin.get());
            break;
        }
    }

    // Check that we actually found something.
    if (e) return Err(toString(std::move(e)));
    if (not bin) {
        Error(import_loc, "Module file '{}' is malformed: '{}' not found.", path, name);
        return std::nullopt;
    }

    // Check that this isn’t random garbage.
    auto obj = dyn_cast<llvm::object::ObjectFile>(bin.get());
    if (not obj) {
        Error(
            import_loc,
            "Module file '{}' is malformed. Entry for module "
            "'{}' is not a valid object file",
            path,
            name
        );

        return std::nullopt;
    }

    // Find the section that contains the module data.
    auto sname = constants::ModuleDescriptionSectionName(name);
    for (auto& s : obj->sections()) {
        auto n = s.getName();
        if (auto err = n.takeError()) return Err(toString(std::move(err)));
        if (n.get() == sname) {
            auto contents = s.getContents();
            if (auto err = contents.takeError()) return Err(toString(std::move(err)));
            data = ArrayRef{contents.get().data(), contents.get().size()};
            return Deserialise();
        }
    }

    // We couldn’t find the module data.
    Error(
        import_loc,
        "Module file '{}' is malformed. Module description for '{}' is missing",
        path,
        name
    );

    return std::nullopt;
}

auto Deserialiser::DeserialiseDecl() -> Decl* {
    switch (Read<Stmt::Kind>()) { // clang-format off
        using K = Stmt::Kind;
#       define AST_STMT_LEAF(node) case K::node: Unreachable("Not a declaration!");
#       define AST_DECL_LEAF(node) case K::node: return SRCC_CAT(Deserialise, node)();
#       include "srcc/AST.inc"
    } // clang-format on

    Unreachable("Invalid statement kind");
}

auto Deserialiser::DeserialiseFieldDecl() -> Decl* {
    Todo();
}

auto Deserialiser::DeserialiseLocalDecl() -> Decl* {
    Todo();
}

auto Deserialiser::DeserialiseProcDecl() -> Decl* {
    auto ty = ReadType<ProcType>();
    auto name = ReadString();
    auto mangling = Read<Mangling>();
    auto proc = ProcDecl::Create(
        *M,
        ty,
        name,
        Linkage::Imported,
        mangling,
        nullptr,
        {}
    );

    SmallVector<LocalDecl*> params;
    for (usz i = 0; i < proc->param_count(); i++)
        params.push_back(cast<LocalDecl>(DeserialiseDecl()));
    proc->finalise({}, params);
    return proc;
}

auto Deserialiser::DeserialiseProcTemplateDecl() -> Decl* {
    Todo();
}

auto Deserialiser::DeserialiseParamDecl() -> Decl* {
    Todo();
}

auto Deserialiser::DeserialiseTemplateTypeParamDecl() -> Decl* {
    Unreachable("Should not exist in AST");
}

void Deserialiser::DeserialiseType() {
    auto k = Read<TypeBase::Kind>();
    switch (k) {
        case TypeBase::Kind::SliceType:
            deserialised_types.push_back(SliceType::Get(*M, ReadType()));
            return;

        case TypeBase::Kind::ReferenceType:
            deserialised_types.push_back(ReferenceType::Get(*M, ReadType()));
            return;

        case TypeBase::Kind::IntType:
            deserialised_types.push_back(IntType::Get(*M, ReadSize()));
            return;

        case TypeBase::Kind::ArrayType: {
            auto elem = ReadType();
            auto dim = ReadInt();
            deserialised_types.push_back(ArrayType::Get(*M, elem, i64(dim)));
            return;
        }

        case TypeBase::Kind::BuiltinType: {
            switch (Read<BuiltinKind>()) {
                case BuiltinKind::Void: deserialised_types.push_back(Type::VoidTy); return;
                case BuiltinKind::NoReturn: deserialised_types.push_back(Type::NoReturnTy); return;
                case BuiltinKind::Bool: deserialised_types.push_back(Type::BoolTy); return;
                case BuiltinKind::Int: deserialised_types.push_back(Type::IntTy); return;
                case BuiltinKind::Deduced: deserialised_types.push_back(Type::DeducedTy); return;
                case BuiltinKind::Type: deserialised_types.push_back(Type::TypeTy); return;
                case BuiltinKind::UnresolvedOverloadSet: deserialised_types.push_back(Type::UnresolvedOverloadSetTy); return;
            }

            Unreachable("Invalid builtin type");
        }

        case TypeBase::Kind::ProcType: {
            SmallVector<ParamTypeData, 6> param_types;
            auto cc = Read<CallingConvention>();
            auto variadic = Read<bool>();
            auto num_params = ReadInt();
            auto ret = ReadType();
            for (u64 i = 0; i < num_params; i++) param_types.emplace_back(Read<Intent>(), ReadType());
            deserialised_types.push_back(ProcType::Get(*M, ret, param_types, cc, variadic));
            return;
        }

        case TypeBase::Kind::StructType: {
            Todo();
        }
    }

    Unreachable("Invalid type kind: {}", +k);
}

auto Deserialiser::DeserialiseTypeDecl() -> Decl* {
    Todo();
}

// ============================================================================
//  API
// ============================================================================
void TranslationUnit::serialise(SmallVectorImpl<char>& buffer) const {
    Serialiser(*this, buffer);
}

auto TranslationUnit::Deserialise(Context& ctx, ArrayRef<char> data) -> Ptr {
    return Deserialiser(ctx, data).Deserialise();
}

auto TranslationUnit::Deserialise(
    Context& ctx,
    StringRef name,
    StringRef path,
    Location import_loc
) -> Opt<Ptr> {
    return Deserialiser(ctx).DeserialiseFromArchive(name, path, import_loc);
}
