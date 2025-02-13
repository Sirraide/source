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
    add_link_options(-fuse-ld=mold -Wl,--color-diagnostics)
endif()

## ============================================================================
##  Compiler options.
## ============================================================================
add_library(options INTERFACE)

## Compile flags.
target_compile_options(options INTERFACE
    -fnew-infallible
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
    -Wno-unused-parameter ## FIXME: Should periodically reenable this.
    -Wno-unused-function
    -Wno-unused-local-typedefs
    -Wno-ctad-maybe-unsupported

    ## NULL Errors.
    -Werror=nonnull # Passing NULL to nonnull parameter.
    -Wnullable-to-nonnull-conversion

    ## Memory Errors.
    -Werror=address              # Suspicious use of addresses.
    -Werror=init-self            # Initialization of a variable with itself.
    -Werror=uninitialized

    ## Return type.
    -Werror=return-type

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
include(FetchContent)

## Get libbase.
message(STATUS "Downloading libbase ...")
set(LIBBASE_CXXFLAGS -fno-exceptions -fno-rtti -fwrapv -fnew-infallible)
FetchContent_Declare(libbase
    GIT_REPOSITORY https://github.com/Sirraide/libbase
    GIT_TAG master
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libs/libbase"
)
FetchContent_MakeAvailable(libbase)

## Link against libfmt.
target_link_libraries(options INTERFACE libbase)

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
if (NOT EXISTS "${SOURCE_LLVM_ROOT}/bin/clang")
    message(FATAL_ERROR "SOURCE_LLVM_ROOT does not contain a vid Clang build. See README.md for instructions.")
endif()

## LLVM paths.
set(SOURCE_LLVM_CMAKE_DIR "${SOURCE_LLVM_ROOT}/lib/cmake")
set(SOURCE_LLVM_BIN_DIR "${SOURCE_LLVM_ROOT}/bin")
set(LLVM_DIR "${SOURCE_LLVM_CMAKE_DIR}/llvm")
set(Clang_DIR "${SOURCE_LLVM_CMAKE_DIR}/clang")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddClang)
include(HandleLLVMOptions)

include_directories(SYSTEM
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

separate_arguments(llvm_libs)
target_link_libraries(options INTERFACE
    tinfo

    LLVMCore
    LLVMPasses
    LLVMSupport
    LLVMTarget
    LLVMTargetParser
    LLVMX86AsmParser
    LLVMX86CodeGen

    clang
    clangBasic
    clangFrontend
    clangSema
    clangTooling
)

## Global definitions.
add_definitions(
    "-DSOURCE_PROJECT_DIR_NAME=\"${PROJECT_SOURCE_DIR}\""
    "-DSOURCE_CLANG_EXE=\"${SOURCE_LLVM_ROOT}/bin/clang++\""
)

#[[
## Clang’s resource directory is a relative path by default; as we’re
## building this out of tree, we need to set the path explicitly.
execute_process(
    COMMAND "${LLVM_BUILD_DIR}/bin/clang" -print-resource-dir
    OUTPUT_VARIABLE SRCC_CLANG_RESOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
]]

## ============================================================================
##  Helper Functions
## ============================================================================
## Add a library.
function (add_srcc_library name path_name)
    file(GLOB_RECURSE _headers_${name} include/srcc/${path_name}/*.hh)
    file(GLOB_RECURSE _sources_${name} src/${path_name}/*.cc)
    add_library(srcc-${name} STATIC ${_sources_${name}})
    target_sources(srcc-${name} PUBLIC FILE_SET HEADERS FILES ${_headers_${name}})

    ## Link to options and core library.
    target_link_libraries(srcc-${name} PRIVATE options srcc-core)

    ## Add other dependencies.
    set(multiValueArgs LINK)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
    target_link_libraries(srcc-${name} PRIVATE ${ARG_LINK})
endfunction()


## FIXME: Library layering is non-existent at the moment due to the fact that
## constant evaluation depends on codegen, so we’ve thrown it out the window
## for the time being.
file(GLOB_RECURSE _headers_core include/*.hh)
file(GLOB_RECURSE _sources_core
    src/AST/*.cc
    src/CG/*.cc
    src/Core/*.cc
    src/Driver/*.cc
    src/Frontend/*.cc
)

add_library(srcc-core STATIC ${_sources_core})
target_sources(srcc-core PUBLIC FILE_SET HEADERS FILES ${_headers_core})

target_link_libraries(srcc-core PRIVATE options)
target_precompile_headers(srcc-core PUBLIC include/srcc/Core/Utils.hh)
#[[
## ============================================================================
##  Core library
## ============================================================================
## This is what everything else depends on so we handle it separately
## since it must be built first.
file(GLOB_RECURSE _headers_core include/*.hh)
file(GLOB_RECURSE _sources_core src/Core/*.cc)
add_library(srcc-core STATIC ${_sources_core})
target_sources(srcc-core PUBLIC FILE_SET HEADERS FILES ${_headers_core})

target_link_libraries(srcc-core PRIVATE options)
target_precompile_headers(srcc-core PUBLIC include/srcc/Core/Utils.hh)


## ============================================================================
##  Other libraries
## ============================================================================
add_srcc_library(ast AST)
add_srcc_library(codegen CG LINK srcc-ast)
add_srcc_library(frontend Frontend LINK srcc-ast srcc-codegen)
add_srcc_library(driver Driver LINK srcc-ast srcc-frontend srcc-codegen)]]

## ============================================================================
##  Compiler executable
## ============================================================================
## Do NOT recurse here; only top-level files are part of the main driver
## executable only.
file(GLOB _driver_sources src/*.cc)

## Add the executable.
add_clang_executable(srcc ${_driver_sources})

## Link everything together.
target_link_libraries(srcc PRIVATE options srcc-core)

## Tests
include(CTest)
include(fchk/cmake/FCHKDiscoverTests.cmake)
set(FCHK_EXE_PATH "${PROJECT_SOURCE_DIR}/fchk/out/fchk")
FCHKAddAllTestsInDir(
    IN test
    PATTERN "*.src"
    PREFIX  "//"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ARGS -P nocap -P captype -l . -D "srcc=$<TARGET_FILE:srcc>"
    RECURSIVE
)

## ============================================================================
##  Stdlib
## ============================================================================
## Build the runtime first.
add_custom_target(src_runtime ALL
    COMMAND "${SOURCE_LLVM_ROOT}/bin/clang++"
        "${PROJECT_SOURCE_DIR}/std/runtime.cc"
        -std=c++20
        -c
        -Wall -Wextra
        -O3 -march=native
        -fno-exceptions -fno-rtti -fwrapv -fnew-infallible
        -o "${CMAKE_CURRENT_BINARY_DIR}/runtime.cc.o"

    COMMAND $<TARGET_FILE:srcc>
        "${PROJECT_SOURCE_DIR}/std/runtime.src"
        --mo "${PROJECT_SOURCE_DIR}/modules"
        "--link-object=${CMAKE_CURRENT_BINARY_DIR}/runtime.cc.o"
        --noruntime

    DEPENDS srcc
    BYPRODUCTS
        "${CMAKE_CURRENT_BINARY_DIR}/runtime.cc.o"
        "${PROJECT_SOURCE_DIR}/modules/__src_runtime.mod"

    COMMENT "Building Source runtime"

    VERBATIM
    USES_TERMINAL
)
