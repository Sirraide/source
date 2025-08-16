#include <srcc/Core/Constants.hh>

using namespace srcc;

auto constants::EntryPointName(StringRef ModuleName) -> std::string {
    return std::format("{}{}", ModuleEntryPointPrefix.sv(), ModuleName);
}
