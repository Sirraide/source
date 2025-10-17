#ifndef SRCC_CORE_CONSTANTS_HH
#define SRCC_CORE_CONSTANTS_HH

#include <srcc/Core/Utils.hh>
#include <array>

namespace srcc::constants {
constexpr String ProgramEntryPoint = "__src_main";
constexpr String ModuleEntryPointPrefix = "__src_static_init.";
constexpr String ModuleSectionNamePrefix = ".__src_module_description.";
constexpr String AssertFailureHandlerName = "__src_assert_fail";
constexpr String ArithmeticFailureHandlerName = "__src_int_arith_error";
constexpr String VMEntryPointName = "__srcc_vm_main";
constexpr String ModuleFileExtension = "mod";
constexpr String ModuleDescriptionFileExtension = "mod.d";
constexpr std::array AbortHandlers{
    AssertFailureHandlerName,
    ArithmeticFailureHandlerName,
};

auto EntryPointName(StringRef ModuleName) -> std::string;
auto ModuleDescriptionSectionName(StringRef ModuleName) -> std::string;
}

#endif // SRCC_CORE_CONSTANTS_HH
