#ifndef SRCC_CORE_SERIALISATION_HH
#define SRCC_CORE_SERIALISATION_HH

#include <base/Serialisation.hh>
#include <srcc/Core/Utils.hh>

namespace srcc {
using ByteBuffer = std::vector<std::byte>;
using ByteReader = ser::Reader<std::endian::native>;
using ByteWriter = ser::Writer<std::endian::native>;
}

#endif //SRCC_CORE_SERIALISATION_HH
