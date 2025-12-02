- Require `({})` for block expressions; they’re rare, so it shouldn’t be
  an issue (and it resolves the `match {` and `{} a` weirdness.
- Heterogeneous Ranges
- Named Parameters
- Make use of `PromoteMemOpInterface` and friends.
- Refactor: Try using the 'ptr' dialect and replace our store/load/ptradd ops.

- Nested functions
Add a mangling number to each nested functions, based on when TranslateDeclInitial() is first called; this is required for e.g.
```
proc a {
   {
      proc b {} // #1
   }
   {
      proc b {} // #2
   }
}
```
here, #1 and #2 would have the same mangled name without this, but they don’t conflict at the AST level because they’re in different scopes. We should add e.g. the number 0 to the end of the name of #1 and the number 1 to the end of #2 or sth like that. The counter should be reset on a per-function basis (and together with adding the names of parent functions to the mangled name, this should be enough).

- Yeet assert info and instead pass all of its fields as individual parameters; *hopefully*, that’ll make the IR less horrible to look at in the tests when we have a lot of (implicit) asserts in a single file...

- Initial design for optional access checking as an MLIR pass: we need 5 IR operations. These operate on pointers:

0. We first need to figure out what pointers any one pointer is based on, e.g.
if we have `int x = (if a then b else c)`, that load requires both `b` and `c` to be engaged

1. 'engage offs' : Mark that the optional flag at ptr is engaged. This
   stores '1' to it (or is a no-op for transparent optionals) and is emitted every
   time we store a non-nil value to the optional.

2. 'disengage offs' : The opposite of 'engage'; optionals are disengaged by default
   so this is *not* emitted for an optional that is default-initialised in its declaration.

3. 'unwrap offs' : No-op that indicates that the optional must be engaged at this point; this
   is a *compile-time* check that does not emit any code and is used during lifetime analysis;
   every unwrap must be dominated by at least one 'engage' and by no 'disengage's.

   Codegen should keep track whether any 'unwrap's are present in a function at all and skip the
   analysis if there are none (even if there are 'engage' or other operations).

4. 'assert-unwrap offs' : Runtime check that aborts if the optional is not engaged at this
   point; during lifetime analysis, this is equivalent to 'engage'.

5. 'escape offs' : Mark that the address of this optional has escaped; an escaped optional
   is automatically disengaged every time a function is called. If 'ptr' is not an alloca, then
   this is automatically the case.
