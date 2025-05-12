- Store children to be printed in printer
- Remove 'Alloca' as an instruction and store stack allocations in a 'FrameData' struct instead.
- Throw out name mangling entirely and instead maintain a mapping to automatically generated names in the module (e.g.
 `proc a` might become `_S__src.a.0` or sth like that)

- Serialisation refactor:
  - For templates, just store the tokens that make up the template (similarly to what Flang does)

- MRValues 
  - Currently, we just allow lvalues where mrvalues are expected and then expect codegen to handle
    it; if this ever causes problems, we need to instead create a `StructCopyExpr` or sth like that
    which has an lvalue subexpression and which is an mrvalue that codegen can create a memcpy for.
