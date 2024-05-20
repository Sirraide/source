#ifndef AST_STMT_LEAF
#    define AST_STMT_LEAF(node) AST_STMT(node)
#endif

#ifndef AST_STMT
#    define AST_STMT(node) AST_NODE(node)
#endif

#ifndef AST_DECL_LEAF
#    define AST_DECL_LEAF(node) AST_STMT_LEAF(node)
#endif

#ifndef AST_TYPE_LEAF
#    define AST_TYPE_LEAF(node) AST_TYPE(node)
#endif

#ifndef AST_TYPE
#    define AST_TYPE(node) AST_NODE(node)
#endif

#ifndef AST_NODE
#    define AST_NODE(node)
#endif

// These must be arranged in a preorder traversal of the class hierarchy
// for classof() to work properly.

AST_STMT(Stmt)
AST_STMT(Expr)
AST_STMT_LEAF(BlockExpr)
AST_STMT_LEAF(BuiltinCallExpr)
AST_STMT_LEAF(CallExpr)
AST_STMT_LEAF(ConstExpr)
AST_STMT_LEAF(EvalExpr)
AST_STMT_LEAF(ProcRefExpr)
AST_STMT_LEAF(SliceDataExpr)
AST_STMT_LEAF(StrLitExpr)
AST_STMT(Decl)
AST_STMT(ObjectDecl)
AST_DECL_LEAF(ProcDecl)

AST_TYPE(TypeBase)
AST_TYPE(SingleElementTypeBase)
AST_TYPE_LEAF(ArrayType)
AST_TYPE_LEAF(SliceType)
AST_TYPE_LEAF(ReferenceType)
AST_TYPE_LEAF(BuiltinType)
AST_TYPE_LEAF(IntType)
AST_TYPE_LEAF(ProcType)

#undef AST_STMT_LEAF
#undef AST_STMT
#undef AST_STMT_LEAF
#undef AST_STMT
#undef AST_TYPE_LEAF
#undef AST_TYPE
#undef AST_NODE
#undef AST_DECL_LEAF