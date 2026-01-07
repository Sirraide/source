#include <srcc/Core/Token.hh>

using namespace srcc;

/// Stringify a token type.
auto srcc::Spelling(Tk t) -> String {
    switch (t) {
#       define TOKEN(name, spelling) case Tk::name: return spelling;
#       include "srcc/Tokens.inc"
    }

    Unreachable();
}

/// Remove the assignment part of an operator.
auto srcc::StripAssignment(Tk t) -> Tk {
    switch (t) {
        case Tk::PlusEq: return Tk::Plus;
        case Tk::PlusTildeEq: return Tk::PlusTilde;
        case Tk::MinusEq: return Tk::Minus;
        case Tk::MinusTildeEq: return Tk::MinusTilde;
        case Tk::StarEq: return Tk::Star;
        case Tk::StarTildeEq: return Tk::StarTilde;
        case Tk::StarStarEq: return Tk::StarStar;
        case Tk::SlashEq: return Tk::Slash;
        case Tk::PercentEq: return Tk::Percent;
        case Tk::ShiftLeftEq: return Tk::ShiftLeft;
        case Tk::ShiftRightEq: return Tk::ShiftRight;
        case Tk::ShiftLeftLogicalEq: return Tk::ShiftLeftLogical;
        case Tk::ShiftRightLogicalEq: return Tk::ShiftRightLogical;
        default: return t;
    }
}

bool Token::operator==(const Token& b) {
    if (type != b.type) return false;
    switch (type) {
        case Tk::Identifier:
        case Tk::StringLiteral:
        case Tk::CXXHeaderName:
        case Tk::TemplateType:
            return text == b.text;

        case Tk::Integer:
        case Tk::IntegerType:
            return integer == b.integer;

        /// All these are trivially equal.
        default:
            return true;
    }

    Unreachable();
}
