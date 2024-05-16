module;

#include <llvm/Support/Compression.h>
#include <srcc/Macros.hh>

module srcc.ast;
using namespace srcc;

// ============================================================================
//  Serialiser
// ============================================================================
struct Serialiser {
    const Module& M;
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

    Serialiser(const Module& M, SmallVectorImpl<char>& buffer);
    void SerialiseDecl(const Decl*);
    void SerialiseProcDecl(const ProcDecl* proc);
    auto SerialiseType(Type ty) -> u64;
};

Serialiser::Serialiser(const Module& M, SmallVectorImpl<char>& buffer)
    : M{M},
      buffer{buffer} {
    for (const auto& exports : M.exports.decls)
        for (auto d : exports.second)
            SerialiseDecl(d);

    // TODO: Add compiler git hash + compiler build time.
    // TODO: Add mtimes of all source files (and C++ headers).
    // TODO: Compress with ZSTD.
    SmallVector<char, 0> combined;
    Writer W{buffer};
    W << M.name;
    W << types_buffer.size();
    W << serialised_decls;
    buffer.append(types_buffer.begin(), types_buffer.end());
    buffer.append(decls_buffer.begin(), decls_buffer.end());
}

void Serialiser::SerialiseDecl(const Decl* d) {
    Writer{decls_buffer} << d->kind();
    serialised_decls++;
    switch (d->kind()) {
        using K = Stmt::Kind;
#       define AST_STMT_LEAF(node) case K::node: Unreachable("Not a declaration!");
#       define AST_DECL_LEAF(node) case K::node: SRCC_CAT(Serialise, node)(cast<node>(d)); return;
#       include "srcc/AST.inc"
    }

    Unreachable("Invalid statement kind");
}

void Serialiser::SerialiseProcDecl(const ProcDecl* proc) {
    Assert(not proc->parent, "Exporting local procedure?");
    Writer W{decls_buffer};
    W << SerialiseType(proc->type);
    W << proc->name;
    W << proc->mangling;
}

auto Serialiser::SerialiseType(Type ty) -> u64 {
    // Avoid serialising the same type twice.
    auto it = type_indices.find(ty.as_opaque_ptr());
    if (it != type_indices.end()) return it->second;

    // The current index will be the index of this type. Add it
    // first to support recursion.
    auto idx = u64(type_indices.size());
    type_indices[ty.as_opaque_ptr()] = idx;

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
            for (auto p : params) W << SerialiseType(p);
            return idx;
        }
    }

    Unreachable("Invalid type kind");
}

// ============================================================================
//  Deserialiser
// ============================================================================
struct Deserialiser {
    Context& ctx;
    Module::Ptr M;
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
        u64 size = Read<u64>();
        buf.resize(size);
        std::memcpy(buf.data(), data.data(), size);
        data = data.drop_front(size);
        return M->save(buf);
    }

    auto ReadType() -> Type {
        u64 idx = ReadInt();
        return deserialised_types[idx];
    }

    Deserialiser(Context& ctx, ArrayRef<char> data)
        : ctx{ctx},
          M{Module::CreateEmpty(ctx)},
          data{data} {}

    auto Deserialise() -> Module::Ptr;
    void DeserialiseDecl();
    void DeserialiseProcDecl();
    void DeserialiseType();
};

auto Deserialiser::Deserialise() -> Module::Ptr {
    M->name = ReadString();
    u64 serialised_decls = ReadInt();
    u64 serialised_types = ReadInt();
    for (u64 i = 0; i < serialised_types; i++) DeserialiseType();
    for (u64 i = 0; i < serialised_decls; i++) DeserialiseDecl();
    return std::move(M);
}

void Deserialiser::DeserialiseDecl() {
    switch (Read<Stmt::Kind>()) {
        using K = Stmt::Kind;
#       define AST_STMT_LEAF(node) case K::node: Unreachable("Not a declaration!");
#       define AST_DECL_LEAF(node) case K::node: SRCC_CAT(Deserialise, node)(); return;
#       include "srcc/AST.inc"
    }

    Unreachable("Invalid statement kind");
}

void Deserialiser::DeserialiseProcDecl() {
    auto ty = ReadType();
    auto name = ReadString();
    auto mangling = Read<Mangling>();
    auto proc = new (*M) ProcDecl{
        ty,
        name,
        Linkage::Imported,
        mangling,
        nullptr,
        nullptr,
        {}
    };

    M->exports.decls[name].push_back(proc);
}

void Deserialiser::DeserialiseType() {
    switch (Read<TypeBase::Kind>()) {
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
                case BuiltinKind::Void: deserialised_types.push_back(M->VoidTy); return;
                case BuiltinKind::Dependent: deserialised_types.push_back(M->DependentTy); return;
                case BuiltinKind::NoReturn: deserialised_types.push_back(M->NoReturnTy); return;
                case BuiltinKind::Bool: deserialised_types.push_back(M->BoolTy); return;
            }

            Unreachable("Invalid builtin type");
        }

        case TypeBase::Kind::ProcType: {
            SmallVector<Type> param_types;
            auto cc = Read<CallingConvention>();
            auto variadic = Read<bool>();
            auto num_params = ReadInt();
            auto ret = ReadType();
            for (u64 i = 0; i < num_params; i++) param_types.push_back(ReadType());
            deserialised_types.push_back(ProcType::Get(*M, ret, param_types, cc, variadic));
            return;
        }
    }

    Unreachable("Invalid type kind");
}

// ============================================================================
//  API
// ============================================================================
void Module::serialise(SmallVectorImpl<char>& buffer) const {
    Serialiser(*this, buffer);
}

auto Module::Deserialise(Context& ctx, ArrayRef<char> data) -> Ptr {
    return Deserialiser(ctx, data).Deserialise();
}
