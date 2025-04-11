#ifndef SRCC_AST_PRINTER_HH
#define SRCC_AST_PRINTER_HH

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/ArrayRef.h>
#include <base/Colours.hh>

namespace srcc {
template <typename NodeType>
class PrinterBase;
}

template <typename NodeType>
class srcc::PrinterBase : protected base::text::ColourFormatter {
    friend ColourFormatter;
    bool use_colour_;

protected:
    SmallString<128> leading;

    explicit PrinterBase(bool use_colour) : use_colour_{use_colour} {}

    bool use_colour() const { return use_colour_; }

    template <typename Node = NodeType>
    void PrintChildren(this auto&& self, std::type_identity_t<ArrayRef<Node*>> children) {
        if (children.empty()) return;
        auto& leading = self.leading;

        // Print all but the last.
        const auto size = leading.size();
        leading += "│ ";
        const auto current = StringRef{leading}.take_front(size);
        for (auto c : children.drop_back(1)) {
            self.print("%1({}├─%)", current);
            self.Print(c);
        }

        // Print the preheader of the last.
        leading.resize(size);
        self.print("%1({}└─%)", StringRef{leading});

        // Print the last one.
        leading += "  ";
        self.Print(children.back());

        // And reset the leading text.
        leading.resize(size);
    }
};

#endif // SRCC_AST_PRINTER_HH
