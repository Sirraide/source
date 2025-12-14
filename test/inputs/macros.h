#ifndef HEADER
#define HEADER

#define NESTED 42
#define FOOBAR NESTED + NESTED
#define SIX (1 + 2 + 3 * ((1 << 4) / 16))
#define NOT_CONSTEXPR ((void*)-1)

#endif