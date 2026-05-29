#include <srcc/Frontend/Sema.hh>

using namespace base;
using namespace srcc;

// This file embeds the preamble; ths reason this is a separate file is so
// we don't have to recompile the rest of the compiler if the preamble changes.

namespace {
constexpr const char Preamble[]{
#embed "preamble.src" suffix(, 0)
};
}

constexpr const String srcc::sema::preamble::PreambleSource{Preamble};

