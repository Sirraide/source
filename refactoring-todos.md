- Store children to be printed in printer
- Remove 'Alloca' as an instruction and store stack allocations in a 'FrameData' struct instead.
- Throw out name mangling entirely and instead maintain a mapping to automatically generated names in the module (e.g.
 `proc a` might become `_S__src.a.0` or sth like that)
