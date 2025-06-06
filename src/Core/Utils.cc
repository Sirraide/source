#include <srcc/Core/Utils.hh>

#include <llvm/ADT/StringExtras.h>

using namespace srcc;

auto utils::FormatError(llvm::Error& e) -> std::string {
    std::string text;
    llvm::handleAllErrors(std::move(e), [&](const llvm::ErrorInfoBase& e) {
        if (not text.empty()) text += "; ";
        text += e.message();
    });
    return text;
}

auto utils::NumberWidth(usz number, usz base) -> usz {
    return number == 0 ? 1 : usz(std::log(number) / std::log(base) + 1);
}

// Strip colours from an unrendered string.
auto srcc::StripColours(const SmallUnrenderedString& s) -> std::string {
    return text::RenderColours(false, s.str().str());
}

auto srcc::operator+=(std::string& s, String str) -> std::string& {
    s += str.value();
    return s;
}
