module;

#include <srcc/Macros.hh>

module srcc.frontend.sema;
import srcc.utils;
using namespace srcc;

auto Sema::Analyse(ArrayRef<ParsedModule::Ptr> modules) -> Module::Ptr {
    Assert(not modules.empty(), "No modules to analyse!");
    auto& first = modules.front();
    Sema S{first->context()};
    S.mod = Module::Create(first->context(), first->name, first->is_module);
    S.parsed_modules = modules;
    S.Analyse();
    return S.has_error ? nullptr : std::move(S.mod);
}

void Sema::Analyse() {
    // Don’t import the same file twice.
    StringMap<std::pair<Module::Ptr, Location>> imports;
    for (auto& p : parsed_modules)
        for (auto& i : p->imports)
            imports[i.linkage_name] = {nullptr, i.loc};

    for (auto& i : imports) {
        auto res = ImportCXXHeader(mod->save(i.first()));
        if (not res) continue;
        i.second.first = std::move(*res);
    }

    // Don’t attempt anything else if there was a problem.
    if (has_error) return;

    for (auto& i : imports) i.second.first->dump();
    std::exit(43);
}

// ============================================================================
//  Analysis
// ============================================================================
