#ifndef SRCC_CORE_CONSTANTS_HH
#define SRCC_CORE_CONSTANTS_HH

#include <srcc/Core/Utils.hh>

namespace srcc::constants {
constexpr String ProgramEntryPoint = "__src_main";
constexpr String ModuleEntryPointPrefix = "__src_static_init.";
constexpr String ModuleSectionNamePrefix = ".__src_module_description.";

auto EntryPointName(StringRef ModuleName) -> std::string;
auto ModuleDescriptionSectionName(StringRef ModuleName) -> std::string;
}

#endif // SRCC_CORE_CONSTANTS_HH
