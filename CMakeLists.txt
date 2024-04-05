cmake_minimum_required(VERSION 3.28)
project(Source VERSION 0.1.0 LANGUAGES C CXX)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "This project must be built with Clang (or, on Windows, clang-cl)")
endif()

## ============================================================================
##  Global CMake Variables.
## ============================================================================
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## ============================================================================
##  Global compiler options.
## ============================================================================
## Turn on diagnostics colours.
add_compile_options(-fcolor-diagnostics)

## Use mold as the default linker, if it exists.
find_program(MOLD_LINKER "mold")
if (MOLD_LINKER)
    add_link_options(-fuse-ld=mold)
endif()

## ============================================================================
##  Compiler options.
## ============================================================================
add_library(options INTERFACE)

## Compile flags.
target_compile_options(options INTERFACE
    -fno-exceptions
    -fno-rtti
    -fwrapv

    ## Warnings.
    -Wall -Wextra     # Enable ‘all’ warnings.
    -Wundef           # Invalid #undef or undefined macro in #if.
    -Wcast-align      # Casting that changes alignment.
    -Wconversion      # Implicit conversions.
    -Wsign-conversion # Implicit sign conversions.
    -Wformat=2        # Stricter format checking.

    ## Disabled warnings.
    -Wno-unused-function
    -Wno-unused-local-typedefs

    ## NULL Errors.
    -Werror=nonnull # Passing NULL to nonnull parameter.

    ## Memory Errors.
    -Werror=address              # Suspicious use of addresses.
    -Werror=init-self            # Initialization of a variable with itself.
    -Werror=uninitialized

    ## Return type.
    -Werror=return-type
    -Wmissing-noreturn

    ## C/C++.
    -Werror=implicit-fallthrough
    -Werror=missing-include-dirs # User-specified include dir does not exist.
    -Werror=pointer-arith        # Disallow void* and function pointer arithmetic.
    -Werror=string-compare       # Nonsensical string comparisons.
    -Werror=switch               # Missing switch cases.
    # -Werror=switch-enum          # Switch on enum (even if there is a default case).
    -Werror=write-strings        # Strings in C should be const char*.

    ## C++.
    -Werror=missing-field-initializers
    -Werror=non-virtual-dtor
    -Werror=pessimizing-move
    -Werror=dangling
    -Werror=return-stack-address

    ## Debug/Release flags.
    $<$<CONFIG:DEBUG>:-O0 -g3 -ggdb3>
    $<$<CONFIG:RELEASE>:-O3 -march=native>
)

## Link flags.
target_link_options(options INTERFACE
    $<$<CONFIG:DEBUG>:-O0 -g3 -ggdb3 -rdynamic>
    $<$<CONFIG:RELEASE>:-O3 -march=native>
)

## Address Sanitiser.
if (ENABLE_ASAN)
    target_compile_options(options INTERFACE -fsanitize=address)
    target_link_options(options INTERFACE -fsanitize=address)
endif()

## ============================================================================
##  Submodules and include dirs.
## ============================================================================
## Find libfmt.
find_package(fmt CONFIG)
if (NOT fmt_FOUND)
    include(FetchContent)
	message(STATUS "Fmtlib not found, downloading ...")
	FetchContent_Declare(fmtlib
		GIT_REPOSITORY https://github.com/fmtlib/fmt.git
		GIT_TAG 10.2.1
		SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libs/fmt"
    )
	FetchContent_MakeAvailable(fmtlib)
endif()

## Link against libfmt.
target_link_libraries(options INTERFACE fmt)

## Add ‘include’ to the include path.
target_include_directories(options INTERFACE "${PROJECT_SOURCE_DIR}/include")

## As well as everything in ‘libs’.
if (EXISTS ${PROJECT_SOURCE_DIR}/libs)
    file(GLOB libs ${PROJECT_SOURCE_DIR}/libs/*)

    ## Add the include directory to the include path, or the root
    ## directory if there is no include directory.
    foreach(lib ${libs})
        if (IS_DIRECTORY ${lib}/include)
            target_include_directories(options INTERFACE ${lib}/include)
        else()
            target_include_directories(options INTERFACE ${lib})
        endif()
    endforeach()

    ## Also add all of them as subdirectories if they have a CMakeLists.txt.
    foreach (lib ${libs})
        if (EXISTS ${lib}/CMakeLists.txt)
            add_subdirectory(${lib})
        endif()
    endforeach()
endif()

## ============================================================================
##  Add LLVM.
## ============================================================================
if (NOT SOURCE_LLVM_ROOT)
    message(FATAL_ERROR "SOURCE_LLVM_ROOT is not set. See README.md for instructions.")
endif()
if (NOT EXISTS "${SOURCE_LLVM_ROOT}/out/bin/clang")
    message(FATAL_ERROR "SOURCE_LLVM_ROOT does not contain a valid Clang build. See README.md for instructions.")
endif()

## LLVM paths.
set(SOURCE_LLVM_BUILD_DIR "${SOURCE_LLVM_ROOT}/out")
set(SOURCE_LLVM_CMAKE_DIR "${SOURCE_LLVM_BUILD_DIR}/lib/cmake")
set(SOURCE_LLVM_BIN_DIR "${SOURCE_LLVM_BUILD_DIR}/bin")
set(LLVM_DIR "${SOURCE_LLVM_CMAKE_DIR}/llvm")

find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

## ============================================================================
##  Executables and libraries.
## ============================================================================
## Modules interfaces go in ‘lib’, implementation files in ‘src’.
file(GLOB_RECURSE headers include/*.hh)
file(GLOB_RECURSE modules lib/*.ccm)
file(GLOB_RECURSE sources src/*.cc)

## Add the executable.
add_llvm_executable(srcc ${sources})
target_sources(srcc PUBLIC
    FILE_SET HEADERS     FILES ${headers}
    FILE_SET CXX_MODULES FILES ${modules}
)

## Add the LLVM include directories.
##
## This is required as the compiler will otherwise find the headers
## that belong to the system LLVM, if present.
target_include_directories(srcc SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_directories(srcc PRIVATE ${LLVM_LIBRARY_DIRS})
target_compile_definitions(srcc PRIVATE ${LLVM_DEFINITIONS})

## Apply our options.
target_link_libraries(srcc PRIVATE options)
