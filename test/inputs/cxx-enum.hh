#ifndef HEADER
#define HEADER

#include <cstdint>

enum class ScopedEnumI32 : std::int32_t {
    A,
    B = 5,
    C = 42,
    D,
};

enum UnscopedEnumI32 : std::int32_t {
    U_A,
    U_B = 6,
    U_C = U_A + U_B,
    U_D,
};

enum struct ScopedEnumI16 : std::int16_t {
    A = 1,
    B,
    C = 100,
};

enum class ScopedEnumI128 : __int128 {
    A = __int128(1) << __int128(100),
    B,
    C,
};

#endif