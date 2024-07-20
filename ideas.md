## Intent
A parameter can only have at most one of these intents:
  - (none)  = Move: if trivially movable and cheap to copy, pass by value and copy, otherwise
              pass by reference. If the parameter type is a reference type, we’re ‘passing a reference
              by value’, i.e. just passing by reference.
  - `in`    = Pass by value if cheap to copy, otherwise, pass by reference. The object is of
              `val` type in either case.
  - `out`   = Pass a reference (to potentially uninitialised storage); the callee *must* assign
              to an outparameter on all code paths, but not read from it until it has been written
              to at least once. Note: outparameters may not alias.
  - `inout` = Like `out`, except that its value may be read before it is written. Note: inout
              parameters may not alias with any other parameters.
  - `copy`  = Pass by value; the object behaves like a local variable in the callee.
  - `ref`   = Pass by reference; this is only valid in procedure *types* and indicates whether
              a parameter ends up being passed by reference. This is a very low-level (and sometimes
              platform-dependent) feature that should generally not be used. Maybe only allow this in
              printouts or internally? This is basically what `out`, `inout` and large `in` parameters
              are lowered to internally.

Additional parameter modifiers. These are optional and *precede* the intent. A parameter
can have any number of these, but each one only once:
  - `with`   = Equivalent to writing an open `with` declaration at the top of the function.
  - `static` = This parameter is a compile-time constant, and all of its occurrences are
               replaced with its value at every call site. This has the effect of turning
               a function into a template.
  - `exact`  = Forbid implicit conversions, except for implicit dereferencing and reference
               binding. This is useful for e.g. a function that takes
               an `i8` and an `i64` if there is any risk of confusion as to the order of the two.
  - `retain` = Allow capturing the reference that is passed in. This can only be used if the
               parameter has the `in`, `out`, or `inout` intent. This is a low-level feature;
               regular users should prefer passing reference-counted types or scoped pointers
               instead. `retain` cannot be combined with `static`.

Local variables, except parameters marked with `retain`, may not be captured, i.e. any reference
to a local variable, as well as any value that may—depending on the control flow—hold a reference
to a local variable may not be stored—except in other local variables—or bound to the a `retain`
parameter of a function. Lastly, such values may not be returned from the function.

## Operators
`+%` for wrapping addition and `+|` for saturating addition.

Use `:>`, `<:`, `:>=`, `<=:` for unsigned comparison. `//` and `%%` for unsigned division/remainder. 
(note: multiplication isn’t signed, so `**` can still be exponentiation).

`try` establishes a context that has the effect of
‘raising’ *any* errors up the call stack. E.g., if `.foo` and `.bar` can error, then
write `try x.foo.bar` instead of `try (try x.foo).bar`. If you don’t want this
behaviour, you can still write `x.foo?.bar`, i.e. we also allow postfix `?` as
a synonym, though it only applies to one expression.

Allow `in` as an overloadable operator and use it to check if e.g. a vector contains a value or 
a string a character or substring (e.g. `if "foo" in s { ... }`).

## Bits
Add a `bit` data type. It differs from `i1` in that it is packed, i.e. a `bit[32]` is 4 bytes,
whereas an `i1[32]` is 32 bytes.

## Optionals
If possible, option types (`?`) should use the padding bits of the value type to store whether the 
object is there or not.

## Pragma export
`pragma export` to automatically export all following declarations in the current scope, and 
`pragma internal` to turn it off again. Also consider an `internal` keyword to turn off exporting 
for a specific declaration.

## For loops
Iteration over multiple sequences at once:
```c#
for var i in A, var j in B do ...
```

Iteration + index:
```c#
for enum i, var x in A do ...
```

## Deproceduring
For functions w/ no arguments only.

## `()`
This is our equivalent of `{}`, `null`, or `nil`. No, I don’t actually like LISP. 

## Trivially relocatable
A type is trivially relocatable, iff none of its fields store the `this` pointer, and none 
of its fields’ initialisers take the `this` pointer, except by `nocapture`.

## Scoped pointers
Scoped pointers (‘smart pointers’, ‘unique pointers’):

```c++
int^ x = new int; /// Allocates a new int on the heap.
int^ y = x;       /// Moves x into y.
x = 4;            /// ERROR: x has been moved.
y = 5;            /// Assign 5 to y.
x => y;           /// Rebind x to y.
y = 6;            /// ERROR: y has been moved.
y => new int;     /// Allocate a new int.
delete x;         /// Delete the int pointed to by x.
// y is deleted at end of scope.
```

## Unified sum-product type.
`dynamic` makes this and its variant clauses equivalent to
a ‘class hierarchy’, except that it cannot be extended at
runtime. By adding variant clauses and no void variant clause,
a dynamic variant is implicitly ‘abstract’.

```c#
dynamic struct Expr {
    Location loc;
    property Type type; /// Implemented as accessors.
    proc print; /// Function.

    /// `variant IDENTIFIER {` is *conceptually* short for
    /// `variant struct IDENTIFIER {`. There is no such thing
    /// as a ‘variant type’. There are only struct types that
    /// have variant clauses.
    variant Binary {
        Tk op;
        property Expr^ left();
        property Expr^ right();
    }
}
```

## TrailingObjects, but as a language feature:
E.g.
```c++
class Foo : llvm::TrailingObjects<Foo, int> { ... }
```

would become
```c#
struct Foo {
    dynamic int[] ints;
}
```

This adds a bit to the struct that indicates whether the trailing data is present; the size and
the data are then stored after the struct. Objects of type `Foo` cannot be stack-allocated as a
result.

Also allow specifying a range, e.g.
```c#
struct Foo {
    dynamic(1..=12) int[] ints;
}
```
would mean that this can store 1 to 12 ints. This also means we can store the size inline in padding
bits since 5 bits would be enough in this case; also, since we always have at least one int in this
particular case, we don’t need the flag that indicates whether it’s present unless the size doesn’t
fit in any of the padding bits.