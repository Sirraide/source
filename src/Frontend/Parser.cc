module;

#include <memory>
#include <utility>

module srcc.frontend.parser;
using namespace srcc;

auto Parser::Parse(const File& file) -> Result<std::unique_ptr<ParsedModule>> {
    Parser P{file};
    P.ReadTokens(file);
    if (auto res = P.ParseFile(); not res) return res.error();
    return std::move(P.mod);
}

auto Parser::ParseFile() -> Result<> {
    for (const auto& t : stream)
        Diag::Note(ctx, t.location, "Token: {}", t.spelling());
    return {};
}

void ParsedModule::dump() const {
    debug("TODO: Dump module");
}
