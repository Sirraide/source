#include <srcc/Core/Token.hh>

using namespace srcc;

/// Stringify a token type.
auto srcc::Spelling(Tk t) -> String {
    switch (t) {
        case Tk::Invalid: return "<invalid>";
        case Tk::Eof: return "<eof>";
        case Tk::Identifier: return "<identifier>";
        case Tk::TemplateType: return "<template type>";
        case Tk::CXXHeaderName: return "<cxx header name>";
        case Tk::StringLiteral: return "<string literal>";
        case Tk::Integer: return "<integer>";
        case Tk::IntegerType: return "<integer type>";
        case Tk::Comment: return "<comment>";

        case Tk::Alias: return "alias";
        case Tk::And: return "and";
        case Tk::As: return "as";
        case Tk::AsBang: return "as!";
        case Tk::Asm: return "asm";
        case Tk::Assert: return "assert";
        case Tk::Bool: return "bool";
        case Tk::Break: return "break";
        case Tk::Defer: return "defer";
        case Tk::Delete: return "delete";
        case Tk::Do: return "do";
        case Tk::Dynamic: return "dynamic";
        case Tk::Elif: return "elif";
        case Tk::Else: return "else";
        case Tk::Enum: return "enum";
        case Tk::Eval: return "eval";
        case Tk::Export: return "export";
        case Tk::F32: return "f32";
        case Tk::F64: return "f64";
        case Tk::Fallthrough: return "fallthrough";
        case Tk::False: return "false";
        case Tk::For: return "for";
        case Tk::ForReverse: return "for~";
        case Tk::Goto: return "goto";
        case Tk::Hash: return "#";
        case Tk::If: return "if";
        case Tk::In: return "in";
        case Tk::Init: return "init";
        case Tk::Int: return "int";
        case Tk::Import: return "import";
        case Tk::Is: return "is";
        case Tk::Loop: return "loop";
        case Tk::Match: return "match";
        case Tk::NoReturn: return "noreturn";
        case Tk::Not: return "not";
        case Tk::Or: return "or";
        case Tk::Pragma: return "pragma";
        case Tk::Proc: return "proc";
        case Tk::Range: return "range";
        case Tk::Return: return "return";
        case Tk::Static: return "static";
        case Tk::Struct: return "struct";
        case Tk::Then: return "then";
        case Tk::True: return "true";
        case Tk::Try: return "try";
        case Tk::Type: return "type";
        case Tk::Typeof: return "typeof";
        case Tk::Unreachable: return "unreachable";
        case Tk::Val: return "val";
        case Tk::Var: return "var";
        case Tk::Variant: return "variant";
        case Tk::Void: return "void";
        case Tk::Where: return "where";
        case Tk::While: return "while";
        case Tk::With: return "with";
        case Tk::Xor: return "xor";
        case Tk::Continue: return "continue";

        case Tk::Semicolon: return ";";
        case Tk::Colon: return ":";
        case Tk::ColonColon: return "::";
        case Tk::ColonSlash: return ":/";
        case Tk::ColonPercent: return ":%";
        case Tk::Comma: return ",";
        case Tk::LParen: return "(";
        case Tk::RParen: return ")";
        case Tk::LBrack: return "[";
        case Tk::RBrack: return "]";
        case Tk::LBrace: return "{";
        case Tk::RBrace: return "}";
        case Tk::Ellipsis: return "...";
        case Tk::Dot: return ".";
        case Tk::LArrow: return "<-";
        case Tk::RArrow: return "->";
        case Tk::RDblArrow: return "=>";
        case Tk::Question: return "?";
        case Tk::Plus: return "+";
        case Tk::PlusTilde: return "+~";
        case Tk::Minus: return "-";
        case Tk::MinusTilde: return "-~";
        case Tk::Star: return "*";
        case Tk::StarTilde: return "*~";
        case Tk::Slash: return "/";
        case Tk::Swap: return "><";
        case Tk::Percent: return "%";
        case Tk::Caret: return "^";
        case Tk::Ampersand: return "&";
        case Tk::VBar: return "|";
        case Tk::Tilde: return "~";
        case Tk::Bang: return "!";
        case Tk::Assign: return "=";
        case Tk::DotDot: return "..";
        case Tk::DotDotLess: return "..<";
        case Tk::DotDotEq: return "..=";
        case Tk::MinusMinus: return "--";
        case Tk::PlusPlus: return "++";
        case Tk::StarStar: return "**";
        case Tk::SLt: return "<";
        case Tk::SLe: return "<=";
        case Tk::SGt: return ">";
        case Tk::SGe: return ">=";
        case Tk::ULt: return "<:";
        case Tk::ULe: return "<=:";
        case Tk::UGt: return ":>";
        case Tk::UGe: return ":>=";
        case Tk::EqEq: return "==";
        case Tk::Neq: return "!=";
        case Tk::PlusEq: return "+=";
        case Tk::PlusTildeEq: return "+~=";
        case Tk::MinusEq: return "-=";
        case Tk::MinusTildeEq: return "-~=";
        case Tk::StarEq: return "*=";
        case Tk::StarTildeEq: return "*~=";
        case Tk::SlashEq: return "/=";
        case Tk::PercentEq: return "%=";
        case Tk::ShiftLeft: return "<<";
        case Tk::ShiftLeftLogical: return "<<<";
        case Tk::ShiftRight: return ">>";
        case Tk::ShiftRightLogical: return ">>>";
        case Tk::ShiftLeftEq: return "<<=";
        case Tk::ShiftLeftLogicalEq: return "<<<=";
        case Tk::ShiftRightEq: return ">>=";
        case Tk::ShiftRightLogicalEq: return ">>>=";
        case Tk::StarStarEq: return "**=";
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
