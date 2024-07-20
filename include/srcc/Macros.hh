#ifndef SRCC_MACROS_HH
#define SRCC_MACROS_HH

#include <libassert/assert.hpp>
#include <base/Macros.hh>
#include <base/Assert.hh>

#define SRCC_STR(X)  LIBBASE_STR_(X)
#define SRCC_CAT(X, Y)  LIBBASE_CAT_(X, Y)
#define SRCC_IMMOVABLE(X) LIBBASE_IMMOVABLE(X)

#define SRCC_DECLARE_HIDDEN_IMPL(X) \
public:                             \
    SRCC_IMMOVABLE(X);              \
    ~X();                           \
private:                            \
    struct Impl;                    \
    Impl* const impl;

#define SRCC_DEFINE_HIDDEN_IMPL(X) \
    X::~X() { delete impl; }

#define SRCC_BREAK_IF(cond) if (std::getenv("DEBUGGING") and (cond)) asm volatile("int3")

// Here until compilers support delete("message").
//
// clang-scan-deps currently errors if a pp-number token contains a ' inside
// of an `#if`, so prevent clang-format from adding one for now.
// clang-format off
#if __cpp_deleted_function >= 202403L
#    define SRCC_DELETED(Msg) delete (Msg)
#else
#    define SRCC_DELETED(Msg) delete
#endif
// clang-format on

#endif // SRCC_MACROS_HH
