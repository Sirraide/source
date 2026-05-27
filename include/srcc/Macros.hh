#ifndef SRCC_MACROS_HH
#define SRCC_MACROS_HH

#include <libassert/assert.hpp>
#include <base/Macros.hh>
#include <base/Assert.hh>

#define SRCC_BREAK_IF(cond) if (std::getenv("DEBUGGING") and (cond)) asm volatile("int3")

#define SRCC_ALLOCATE_IN_CONTEXT(type, context)                                                 \
    void* operator new(usz) = delete ("Use `new (" LIBBASE_STR(context) "&) { ... }` instead"); \
    void* operator new(usz size, context& ctx)

#endif // SRCC_MACROS_HH
