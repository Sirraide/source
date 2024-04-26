#ifndef SRCC_MACROS_HH
#define SRCC_MACROS_HH

#include <fmt/format.h>
#include <libassert/assert.hpp>

#define SRCC_STR_(X) #X
#define SRCC_STR(X)  SRCC_STR_(X)

#define SRCC_CAT_(X, Y) X##Y
#define SRCC_CAT(X, Y)  SRCC_CAT_(X, Y)

#define SRCC_IMMOVABLE(X)            \
    X(const X&) = delete;            \
    X& operator=(const X&) = delete; \
    X(X&&) = delete;                 \
    X& operator=(X&&) = delete

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

#define Assert(X, ...) ASSERT(X __VA_OPT__(, ::fmt::format(__VA_ARGS__)))

#define Unreachable(...) UNREACHABLE(__VA_OPT__(::fmt::format(__VA_ARGS__)))

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

// CLion thinks __builtin_expect_with_probability does not exist.
#if !__has_builtin(__builtin_expect_with_probability)
#    define __builtin_expect_with_probability(X, Y, Z) __builtin_expect((X), (Y))
#endif

#define defer ::srcc::detail::DeferImpl _ = [&]
#define tempset auto _ = ::srcc::detail::Tempset{}->*

namespace srcc::detail {
template <typename Proc>
class DeferImpl {
    Proc proc;

public:
    DeferImpl(Proc proc) : proc(proc) {}
    ~DeferImpl() { proc(); }
};

template <typename T, typename U>
class TempsetStage2 {
    SRCC_IMMOVABLE(TempsetStage2);

    T& lvalue;
    T old_value;

public:
    TempsetStage2(T& lvalue, U&& value)
        : lvalue(lvalue),
          old_value(std::exchange(lvalue, std::forward<U>(value))) {}
    ~TempsetStage2() { lvalue = std::move(old_value); }
};

template <typename T>
struct TempsetStage1 {
    T& lvalue;
    auto operator=(auto&& value) {
        return TempsetStage2{lvalue, std::forward<decltype(value)>(value)};
    }
};

struct Tempset {
    auto operator->*(auto& lvalue) {
        return TempsetStage1{lvalue};
    }
};
} // namespace srcc::detail

#endif // SRCC_MACROS_HH
