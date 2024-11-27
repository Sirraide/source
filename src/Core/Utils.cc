#include <srcc/Core/Utils.hh>

#include <llvm/ADT/StringExtras.h>

using namespace srcc;

auto utils::Escape(StringRef str, bool escape_quotes) -> std::string {
    std::string s;
    for (auto c : str) {
        switch (c) {
            case '\n': s += "\\n"; break;
            case '\r': s += "\\r"; break;
            case '\t': s += "\\t"; break;
            case '\v': s += "\\v"; break;
            case '\f': s += "\\f"; break;
            case '\a': s += "\\a"; break;
            case '\b': s += "\\b"; break;
            case '\\': s += "\\\\"; break;
            case '\0': s += "\\0"; break;
            case '"':
                if (escape_quotes) s += "\\\"";
                else s += c;
            break;
            default:
                if (llvm::isPrint(c)) s += c;
                else s += std::format("\\x{:02x}", static_cast<u8>(c));
        }
    }
    return s;
}

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
