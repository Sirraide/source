#ifndef SRCC_CG_IR_MLIR_FORMATTERS_HH
#define SRCC_CG_IR_MLIR_FORMATTERS_HH

#include <mlir/IR/Types.h>
#include <format>

template <>
struct std::formatter<mlir::Type> : std::formatter<std::string_view> {
    template <typename FormatContext>
    auto format(mlir::Type t, FormatContext &ctx) const {
        llvm::SmallString<128> s;
        llvm::raw_svector_ostream os{s};
        os << t;
        return format_to(ctx.out(), "{}", os.str());
    }
};

#endif // SRCC_CG_IR_MLIR_FORMATTERS_HH
