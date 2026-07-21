#ifndef SRCC_MACROS_HH
#define SRCC_MACROS_HH

#include <libassert/assert.hpp>
#include <base/Macros.hh>
#include <base/Assert.hh>

#ifndef __clang__
#   error This project must be compiled with Clang
#endif

#if !__has_cpp_attribute(clang::srcc_diagnose_pointer_comparison)
#   error This project must be build with a specific Clang fork (use the 'srcc' branch from 'github.com:Sirraide/llvm-project.git')
#endif

#define SRCC_BREAK_IF(cond) if (std::getenv("DEBUGGING") and (cond)) asm volatile("int3")

#define SRCC_ALLOCATE_IN_CONTEXT(type, context)                                                 \
    void* operator new(usz) = delete ("Use `new (" LIBBASE_STR(context) "&) { ... }` instead"); \
    void* operator new(usz size, context& ctx)

#define SRCC_DIAGNOSE_POINTER_COMPARISON [[clang::srcc_diagnose_pointer_comparison]]

#endif // SRCC_MACROS_HH
