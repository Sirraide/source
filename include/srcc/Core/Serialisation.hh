#ifndef SRCC_CORE_SERIALISATION_HH
#define SRCC_CORE_SERIALISATION_HH

#include <base/Serialisation.hh>
#include <srcc/Core/Utils.hh>

namespace srcc {
using ByteBuffer = std::vector<std::byte>;
using ByteReader = ser::Reader<std::endian::native>;
using ByteWriter = ser::Writer<std::endian::native>;
}

template <std::endian E>
struct base::ser::Serialiser<llvm::APInt, E> {
    static void Deserialise(Reader<E>& r, llvm::APInt& i) {
        auto size = r.template read<srcc::Size>();
        auto data = r.read_bytes(size.bytes());
        llvm::LoadIntFromMemory(i, reinterpret_cast<const u8*>(data.data()), u32(data.size_bytes()));
    }

    static void Serialise(Writer<E>& w, const llvm::APInt& i) {
        auto size = srcc::Size::Bits(i.getBitWidth());
        w << size;

        // Make sure to only call allocate *after* writing the size.
        auto data = w.allocate(size.bytes());
        llvm::StoreIntToMemory(i, reinterpret_cast<u8*>(data.data()), u32(data.size_bytes()));
    }
};

#endif //SRCC_CORE_SERIALISATION_HH
