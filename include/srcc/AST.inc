#ifndef AST_STMT_LEAF
#    define AST_STMT_LEAF(node) AST_STMT(node)
#endif

#ifndef AST_STMT
#    define AST_STMT(node) AST_NODE(node)
#endif

#ifndef AST_INIT_LEAF
#    define AST_INIT_LEAF(node) AST_STMT_LEAF(node)
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
AST_STMT_LEAF(WhileStmt)
AST_STMT(Expr)
AST_STMT_LEAF(AssertExpr)
AST_STMT_LEAF(BinaryExpr)
AST_STMT_LEAF(BlockExpr)
AST_STMT_LEAF(BoolLitExpr)
AST_STMT_LEAF(BuiltinCallExpr)
AST_STMT_LEAF(BuiltinMemberAccessExpr)
AST_STMT_LEAF(CallExpr)
AST_STMT_LEAF(CastExpr)
AST_STMT_LEAF(ConstExpr)
AST_STMT_LEAF(DefaultInitExpr)
AST_STMT_LEAF(EvalExpr)
AST_STMT_LEAF(IfExpr)
AST_STMT_LEAF(IntLitExpr)
AST_STMT_LEAF(LocalRefExpr)
AST_STMT_LEAF(MemberAccessExpr)
AST_STMT_LEAF(OverloadSetExpr)
AST_STMT_LEAF(ProcRefExpr)
AST_STMT_LEAF(ReturnExpr)
AST_STMT_LEAF(StrLitExpr)
AST_STMT_LEAF(StructInitExpr)
AST_STMT_LEAF(TypeExpr)
AST_STMT_LEAF(UnaryExpr)

AST_STMT(Decl)
AST_DECL_LEAF(FieldDecl)
AST_DECL_LEAF(ProcTemplateDecl)
AST_DECL_LEAF(TemplateTypeParamDecl)
AST_DECL_LEAF(TypeDecl)
AST_STMT(ObjectDecl)
AST_DECL_LEAF(LocalDecl)
AST_DECL_LEAF(ParamDecl)
AST_DECL_LEAF(ProcDecl)

AST_TYPE(TypeBase)
AST_TYPE(SingleElementTypeBase)
AST_TYPE_LEAF(ArrayType)
AST_TYPE_LEAF(SliceType)
AST_TYPE_LEAF(PtrType)
AST_TYPE_LEAF(BuiltinType)
AST_TYPE_LEAF(IntType)
AST_TYPE_LEAF(ProcType)
AST_TYPE_LEAF(StructType)

#undef AST_STMT_LEAF
#undef AST_STMT
#undef AST_INIT_LEAF
#undef AST_DECL_LEAF
#undef AST_TYPE_LEAF
#undef AST_TYPE
#undef AST_NODE
