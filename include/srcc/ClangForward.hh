#ifndef SRCC_CLANG_FORWARD_HH
#define SRCC_CLANG_FORWARD_HH

namespace clang {
class ASTUnit;
class Decl;
class RecordDecl;
}

namespace srcc {
using CXXDecl = clang::Decl;
}

#endif //SRCC_CLANG_FORWARD_HH
