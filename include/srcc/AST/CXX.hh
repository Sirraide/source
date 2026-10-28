#ifndef SRCC_AST_CXX_HH
#define SRCC_AST_CXX_HH

#include <clang/AST/Mangle.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Parse/Parser.h>
#include <clang/CodeGen/ModuleBuilder.h>

namespace srcc {

/// Context for C++ parsing.
class CXXCompilation final : public clang::CompilerInstance {
    std::unique_ptr<clang::Parser> parser;
    std::unique_ptr<clang::MangleContext> mangle_context;
    clang::CodeGenerator* code_generator = nullptr;
    bool finished = false;
    unsigned buffers = 0;

public:
    using CompilerInstance::CompilerInstance;
    ~CXXCompilation() override;

    /// Get the Clang code generator.
    auto codegen() -> clang::CodeGenerator& { return *code_generator; }

    /// Run final semantic actions and finish IR generation exactly once.
    bool finish();

    /// Attach CodeGen and initialise a persistent TU scope and parser.
    bool initialise(llvm::LLVMContext& context);

    /// Get the C++ name mangler.
    auto mangler() -> clang::MangleContext* { return mangle_context.get(); }

    /// Parse another buffer and complete pending instantiations, retaining EOF.
    bool parse(llvm::StringRef code);

private:
    bool ParseDeclarations();
};

}
#endif
