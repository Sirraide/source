- Require `({})` for block expressions; they’re rare, so it shouldn’t be
  an issue (and it resolves the `match {` and `{} a` weirdness.
- Heterogeneous Ranges; alternative: make ranges always inclusive and represent the empty range by setting end to a value *less than* begin.
- Named Parameters
- Make use of `PromoteMemOpInterface` and friends.
- Refactor: Try using the 'ptr' dialect and replace our store/load/ptradd ops.

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
