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

    si.data = saved.size();
    saved.push_back(std::move(integer));
    return si;
}

auto StoredInteger::inline_value() const -> std::optional<i64>{
    if (is_inline()) return i64(data);
    return std::nullopt;
}

auto StoredInteger::str(const IntegerStorage* storage, bool is_signed) const -> std::string {
    if (is_inline()) return std::to_string(data);
    if (storage) return llvm::toString(storage->saved[usz(data)], 10, is_signed);
    return "<huge value>";
}

auto StoredInteger::value(const IntegerStorage& storage) const -> APInt {
    if (is_inline()) return APInt(u32(bits), data);
    return storage.saved[usz(data)];
}
