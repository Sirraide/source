#ifndef PARSE_TREE_LEAF_NODE
#    define PARSE_TREE_LEAF_NODE(node) PARSE_TREE_NODE(node)
#endif

#ifndef PARSE_TREE_NODE
#    define PARSE_TREE_NODE(node)
#endif

// These must be arranged in a preorder traversal of the class hierarchy
// for classof() to work properly.
PARSE_TREE_LEAF_NODE(BlockExpr)
PARSE_TREE_LEAF_NODE(CallExpr)
PARSE_TREE_LEAF_NODE(DeclRefExpr)
PARSE_TREE_LEAF_NODE(StrLitExpr)
PARSE_TREE_LEAF_NODE(MemberExpr)

PARSE_TREE_NODE(Decl)
PARSE_TREE_LEAF_NODE(ProcDecl)

#undef PARSE_TREE_NODE
#undef PARSE_TREE_LEAF_NODE
