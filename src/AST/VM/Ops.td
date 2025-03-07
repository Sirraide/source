defvar AbortReason = "AbortReason";
defvar Block = "Block";
defvar Bool = "Bool";
defvar Location = "Location";
defvar Slice = "Slice";
defvar I8 = "I8";
defvar I16 = "I16";
defvar I32 = "I32";
defvar I64 = "I64";
defvar APInt = "APInt";
defvar Type = "Type";
defvar Closure = "Closure";
defvar Pointer = "Pointer";
defvar AllIntegerTypes = [I8, I16, I32, I64, APInt];
defvar AllTypes = AllIntegerTypes # [Type, Closure, Pointer, Slice, Bool];

class OpBase<string op_name> {
    // Name of the operation.
    string name = !tolower(op_name);

    // List of immediate arguments.
    list<string> immargs = [];

    // List of frame arguments.
    list<string> args = [];

    // Whether every part of this operation needs to be implemented manually.
    bit manual = false;
}

class Op : OpBase<NAME>;

multiclass IntOp {
    def NAME#I8 : OpBase<NAME#".8"> { let args = [I8, I8]; }
    def NAME#I16 : OpBase<NAME#".16"> { let args = [I16, I16]; }
    def NAME#I32 : OpBase<NAME#".32"> { let args = [I32, I32]; }
    def NAME#I64 : OpBase<NAME#".64"> { let args = [I64, I64]; }
    def NAME#APInt : OpBase<NAME#".ap"> { let args = [APInt, APInt]; }
    def _EmitIntOp#NAME;
}

multiclass IConv<string from, list<string> to> {
    foreach t = to in {
        def NAME#from#"To"#t : OpBase<NAME#"."#from#"."#t> {
            let args = [t];
        }
    }
}

// Special operations.
def Abort : Op {
    let immargs = [AbortReason, Location];
    let args = [Slice, Slice];
}

def Branch : Op {
    let immargs = [Block];
}

def CondBranch : Op {
    let args = [Bool];
    let immargs = [Block, Block];
}

def DirectCall : Op {
    let manual = true;
}

foreach type = AllTypes in {
    def Load#type : OpBase<"Load."#type> { let args = [Pointer]; }
    def Ret#type : OpBase<"Ret."#type> { let args = [type]; }
    def Select#type : OpBase<"Select."#type> { let args = [Bool, type, type]; }
    def Store#type : OpBase<"Store."#type> { let args = [Pointer, type]; }
}

def MemZero : Op {
    let args = [Pointer];
    let immargs = [I64];
}

def PtrAdd : Op {
    let args = [Pointer, I64];
}

def RetVoid : Op;
def Unreachable : Op;

def UnresolvedCall : Op {
    let manual = true;
}

// Integer operations.
defm AShr : IntOp;
defm Add : IntOp;
defm And : IntOp;
defm Eq : IntOp;
defm LShr : IntOp;
defm Mul : IntOp;
defm Ne : IntOp;
defm Or : IntOp;
defm SAddOv : IntOp;
defm SDiv : IntOp;
defm SGe : IntOp;
defm SGt : IntOp;
defm SLe : IntOp;
defm SLt : IntOp;
defm SMulOv : IntOp;
defm SRem : IntOp;
defm SSubOv : IntOp;
defm Shl : IntOp;
defm Sub : IntOp;
defm UDiv : IntOp;
defm UGe : IntOp;
defm UGt : IntOp;
defm ULe : IntOp;
defm ULt : IntOp;
defm URem : IntOp;
defm Xor : IntOp;

// ZExt/SExt operations.
defm ZExt : IConv<APInt, [I8, I16, I32, I64, APInt]>;
defm ZExt : IConv<I8, [I16, I32, I64, APInt]>;
defm ZExt : IConv<I16, [I32, I64, APInt]>;
defm ZExt : IConv<I32, [I64, APInt]>;
defm ZExt : IConv<I64, [APInt]>;

defm SExt : IConv<APInt, [I8, I16, I32, I64, APInt]>;
defm SExt : IConv<I8, [I16, I32, I64, APInt]>;
defm SExt : IConv<I16, [I32, I64, APInt]>;
defm SExt : IConv<I32, [I64, APInt]>;
defm SExt : IConv<I64, [APInt]>;

// Truncation.
defm Trunc : IConv<I8, [APInt]>;
defm Trunc : IConv<I16, [I8, APInt]>;
defm Trunc : IConv<I32, [I8, I16, APInt]>;
defm Trunc : IConv<I64, [I8, I16, I32, APInt]>;
defm Trunc : IConv<APInt, [I8, I16, I32, I64, APInt]>;


