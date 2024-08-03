module;

#include <llvm/ADT/StringExtras.h>
#include <print>

module srcc.utils;

auto IntegerStorage::store_int(APInt integer) -> StoredInteger {
    StoredInteger si;
    si.bits = integer.getBitWidth();

    if (integer.getBitWidth() <= 64) {
        si.data = integer.getZExtValue();
        return si;
    }

    auto* i = new APInt(integer);
    si.data = reinterpret_cast<uintptr_t>(i);
    saved.emplace_back(i);
    return si;
}

auto StoredInteger::inline_value() const -> std::optional<i64>{
    if (is_inline()) return i64(data);
    return std::nullopt;
}

auto StoredInteger::str(bool is_signed) const -> std::string {
    if (is_inline()) return std::to_string(data);
    return llvm::toString(*reinterpret_cast<APInt*>(data), 10, is_signed);
}

auto StoredInteger::value() const -> APInt {
    if (is_inline()) return APInt(u32(bits), data);
    return *reinterpret_cast<APInt*>(data);
}
