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
              printouts or internally? ~~This is basically what `out`, `inout` and large `in` parameters
              are lowered to internally.~~ Maybe rewrite `inout` to a move instead? (https://www.youtube.com/watch?v=5lecIqUhEl4)

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
`+~` for wrapping addition and `+|` for saturating addition.

Use `:>`, `<:`, `:>=`, `<=:` for unsigned comparison. `:/` and `:%` for unsigned division/remainder. 
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
Iteration over multiple sequences at once (this picks the shorter of the two):
```c#
for a, b in A, B do ...
```

Iteration + index:
```c#
for enum i, a in A do ...
for enum i, a, b in A, B do ...
```

## Deproceduring
For functions w/ no arguments only. ALWAYS do this if the function is not
expressly preceded by `&`.

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
a dynamic variant is implicitly 

The variant storage can be in a different location for individual
variants, for instance, if there is enough space for the variant
data in the padding bytes of the parent struct, use that (only go
up one parent at most tho!)

```c#
dynamic struct Expr {
    Location loc;
    property Type type; /// Implemented as accessors.
    proc print; /// Function.

    /// `variant IDENTIFIER {` is *conceptually* short for
    /// `variant struct { ... } IDENTIFIER`. There is no such thing
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

## Common Type
Add a notion of a ‘common type’ (that is user-definable for user-defined types) that is used to determine
the type of e.g. `if` expressions, deduced return types, etc. (Idea stolen from Carbon).

## Keywords
Make all keywords ‘contextual’, i.e. allow this
```c#
struct foo {
   int else;
}
```

This is fine since it is always accessed as `a.else` or `.else`, and `else` in that position (or any other
keyword, I think) would be syntactically invalid as a keyword anyway.

## Immutability
`ILValue`?

## Initialisers and return values.
Allow initialisers to return a value of any type, i.e. this
```c++
struct S {

}

init S -> int = 4;
```
is valid, if dumb. The intent is that initialisers are just regular
functions that just happen to have the same name as the type they belong
to (the type name is ‘overloaded’, in a sense). This has two consequences.

1. These syntaxes require some thought:
   ```c#
   S s = 4;
   S s = (4);
   ```
   This is only allowed if the initialiser that takes an `int` returns an `S`. Otherwise,
   you have to call the initialiser explicitly.
   ```c#
   S? s = S(4);
   var s = S(4); // For longer type names. 
   ```

2. We need some way to *actually* initialise the fields (since `S(4)` would just call)
   the initialiser again. For this `S::(4)` can be used. This might be a bit ugly, but
   you’re not supposed to use it outside of initialisers anyway.
 
   To prevent accidentally running into an infinite loop in the initialiser, we disallow
   the `S()` syntax there and require writing either `S::()` or `init()` to call another
   initialiser (that is, only for the type that the initialiser belongs to; some other type,
   e.g. `T()` can still be used with that syntax).

   We should probably allow this if the type is (or was at some point) dependent, though, as
   this might cause problems for generic code otherwise. Maybe we should just warn on it if
   the type isn’t dependent and suggest writing either `S::()` or `init()`?

## Closures and C(++) interop.
`__srcc_split_closure` builtin (maybe in the runtime?) to split a closure into a raw function
pointer and a context pointer. The function pointer has an extra argument for the context
pointer (maybe make it configurable where in the signature that argument goes). We also need
a separate function pointer type for this (probably just as a stdlib template).

## Operator `throw`
For returning errors, we run into a problem if the actual return type and the error type are
the same (e.g. `Res<int, int>`). One idea would be to do something like this:
```c#
proc foo -> Res<int, int> {
    return 4; // Returns with value state engaged.
    throw 5;  // Returns with error state engaged.
}
```
To support this for user-defined types, a type can implement a ‘`throw` constructor’:
```c#
init throw Res<$T, $U> (int i) { ... }
```
 which
is called when `throw` is used in a function that returns a value of that type. E.g. desugared,
the function above would be:
```c#
proc foo -> Res<int, int> {
    return Res<int, int>(4);
    return Res<int, int>::Error(4); // Pseudocode, but you get the point.
}
```

## Programs
Allow multiple programs/modules in a single file: `program`/`module` is always
a keyword at the top-level (and at the top-level only). Handle all of this in
the parser.

## Pragmas
Allow setting compile options, what files to compile, what libraries to link
etc. etc. using pragmas within the source code (MSVC-style, a bit). The (if
somewhat lofty) goal would be to just... not require a build system at all.
