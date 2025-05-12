- Store children to be printed in printer
- Remove 'Alloca' as an instruction and store stack allocations in a 'FrameData' struct instead.
- Throw out name mangling entirely and instead maintain a mapping to automatically generated names in the module (e.g.
 `proc a` might become `_S__src.a.0` or sth like that)

- Serialisation refactor:
  - For templates, just store the tokens that make up the template (similarly to what Flang does)

- MRValues 
  Add 'EmitMRValue(Value* into, Expr* mrvalue)' to codegen to handle
    `x = if a then s(1) else s(2)`, `x = { s(1); }`, etc. Move the mrvalue code
    in PerformVariableInitialisation() into this function and rename the former
    to EmitInitialiser()
