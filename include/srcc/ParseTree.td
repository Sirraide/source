// DAGs.
def args;
def fields;

// An expression in the parse tree.
class Expr {
    // Parent class.
    Expr parent;

    // Fields.
    dag fields = (args);

    // Trailing arrays.
    dag trailing_arrays = (fields);

    // Additional printout.
    code extra_printout;
}

// ============================================================================
//  Expressions
// ============================================================================
def BlockExpr : Expr {
    let trailing_arrays = (fields "ParsedExpr*":$stmts);
}

def CallExpr : Expr {
    let fields = (args "ParsedExpr*":$callee);
    let trailing_arrays = (fields "ParsedExpr*":$args);
}

def DeclRefExpr : Expr {
    let fields = (args "String":$name);
    let extra_printout = [{ " {}{}", C(Reset), x.name }];
}

def StrLitExpr : Expr {
    let fields = (args "String":$value);
    let extra_printout = [{ " {}\"{}\"", C(Yellow), utils::Escape(x.value) }];
}

// ============================================================================
//  Declarations
// ============================================================================
def Decl : Expr;

def ProcDecl : Expr {
    let parent = Decl;
    let fields = (args "String":$name, "ParsedExpr*":$body);
}
