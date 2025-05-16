- Store children to be printed in printer
- Throw out name mangling entirely and instead maintain a mapping to automatically generated names in the module (e.g.
 `proc a` might become `_S__src.a.0` or sth like that)

- Serialisation refactor:
  - For templates, just store the tokens that make up the template (similarly to what Flang does)

- MRValues 
  - Currently, we just allow lvalues where mrvalues are expected and then expect codegen to handle
    it; if this ever causes problems, we need to instead create a `StructCopyExpr` or sth like that
    which has an lvalue subexpression and which is an mrvalue that codegen can create a memcpy for.

- TODO: Support shebangs (i.e. if the two bytes of a file are '#!', just skip to the next '\n')

- ARValues
  Introduce a new value category to deal with values that are too large to fit in a single register,
  but small enough to where we don’t want to store them in memory if possible; instead, such values
  are split across several registers; this includes builtin types (slices, ranges, closures), as well
  as small struct types that are trivially constructible and destructible, i.e. which do not require
  a memory location to be passed to the ctor or dtor.

  ARValues only exist in Sema and are eliminated as part of codegen; each ARValue is represented by an
  IR `Aggregate` value; the only operations supported on an aggregate are (there are many ways to *create*
  an aggregate, but once you have one, this is all you can do with it)
  
    1. Loading it from an lvalue; this is done by loading each field individually and then combining all
       fields into an aggregate value.
  
    2. Storing it to an lvalue; this is done by storing each field individually.
    
    3. A `copy`/`move`; this is just a memcpy.

    4. As the argument to a procedure or block; these are split into several arguments, one for each field,
         and an aggregate value is reconstituted from the fields at the start of the block or procedure.
    
    5. As the return value of a function; such aggregates are split across registers when returned.

  In all these cases, the actual aggregate value is only used to bundle other values together; it is never
  actually referenced by anything in the IR. This means that both the evaluator and LLVM backend only have
  to be concerned with SRValues and MRValues (there are some additional nuances here wrt `byval` arguments
  in the C ABI, but that’s a special case we should be able to handle).
