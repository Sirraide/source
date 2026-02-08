## Define members of builtin types in the preamble, e.g.
```
proc bool::flip (inout this) __srcc_builtin_member__ {
    this = not this;
}
```

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
  - `var`  = Pass by value; the object behaves like a local variable in the callee.
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

Bit struct members are also packed by default, e.g. the size of this struct:
```c#
struct S {
    bit a;
    bit b;
    bit c;
}
```
is 1 byte.

Taking the address of a `bit`, much like taking the address of a packed struct member, is not
supported. The `align` attribute (see below) can be used to change that.

Note that taking the address of a `bit` *variable* (whether local or static) is allowed since we
can always allocate those in such a way that they are aligned properly. It is very unlikely that
someone will ever allocate so many `bit`s in a single *stack frame* that they’d benefit from
packing, and if they do, they can just use a `bit` array instead.

An integer can be indexed using the subscript operator to extract a bit or range of bits, e.g.
```c#
i64     x = 4;
bit     b = x[2];      // Get the 3rd bit.
bit[10] bs = x[..<10]; // Get the first 10 bits.
```

## `bool`
`bool`s are stored in memory as though they were values of type `bit`. The only difference between
the two is semantics.

## `align`
This attribute specifying a custom alignment for a struct field; this is always in bytes, e.g.
```c#
struct S {
    i8 a align(8); // Aligned to 8 bytes.
}
```

If `align` is specified without a parameter, the type’s natural alignment is used. This may seem
useless at first, but it can be used to force alignment in packed structs or to align a field
of type `bit` or `bool` to a byte boundary:
```c#
struct S {
    bit a align; // Aligned to 1 byte. The address of this can now be taken.
}
```

Taking the address of a packed struct member is only supported if `align` is used to set its
alignment to at least the natural alignment of the type. Conversely, if `align` is used in a
non-packed struct to set a field’s alignment to less than the natural alignment of the type,
the field’s address can no longer be taken.

Whether the field *happens* to be aligned properly even if its alignment is set to less than
the natural alignment is irrelevant since the compiler may reorder fields.

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

## Heterogeneous ranges
I.e. allow `i32..i8` as a *type* (`..` is a type operator only) and store begin+size instead of begin+end; also, allow
iterating downwards if the size is negative.

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

## Array initialisers
Allow this maybe?
```
int[200](1, 2, 3, 4...) // Fill remaining elements with '4'.
```

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

## Bit patterns for optional types.
A bit pattern of length `n` is *possible* for a type `T` if `T.bits` is exactly `n`. Any
other pattern is called *impossible*.

A possible bit pattern is *valid* for a type `T` under the following conditions, and an
invalid pattern is one that is not *valid*:
- If `T` is an integer or floating-point type, any possible pattern is valid.
- If `T` is a pointer or reference type, the platform null pointer is invalid, any other
  possible pattern is valid.
- If `T` is a range type, any possible pattern is valid.
- If `T` is a closure type, any pattern where the function pointer is the platform null
  pointer is invalid.
- If `T` is a struct type, any bit pattern resulting from an invocation of a constructor
  of `T` followed by any number of member function calls that are passed valid arguments
  and the setting of any public members, in any order or repeated combination of the two,
  to any valid bit patterns is valid.

(idea: choose an invalid pattern as the nil value if possible)

## Idea for formatting and syntax highlighting
‘Decorated parse tree’, i.e. optionally (or always, it oughtn’t be too expensive), Sema
builds a mapping from parse tree nodes to AST nodes so that people can both
use the parse tree for syntactic information and also known e.g. what an identifier actually
is.

This way, we also don’t have to worry about preserving too much syntactic sugar in Sema
(we do still have to preserve type sugar for diagnostics, but that’s a different matter);
e.g. we can collapse constant expressions and `static if`s to their evaluated value.

The mapping can either be a pointer in every parse tree node or an external map.

## Properties
Syntax ideas:
```c#
struct S {
    int x: get = 4;
    int x: get = 4; set = value * 2;

    int x { get = 4 }
    int x { get = 4; set = value * 2; }

    property int x: get = 4;
    property int x: get = 4; set = value * 2;

    // Note: this
    int x: get = 4; set = value;
    // is short for
    int x: get { return 4 }; set (int value) { .x = value * 2 };
}
```

Maybe also add something like ‘PutForwards’ from the web idl spec:
```
struct A { int x; }
struct B {
    // E.g.
    A a: set forward x
    // would be equivalent to
    A a: set (int value) { a.x = value }
}
```

Also allow setting a type for the getter, e.g.
```c#
// Note: ‘String’ and ‘StringView’ are strawman syntax.
struct S {
    String x: get -> StringView = x; set = value;
}
```

Shorthand default syntax borrowed from C#:
```c#
struct S {
    int x: get; set = 4; // getter just returns the value.
    int x: get = 4; set; // setter just sets the value.
}
```
I.e. if a property setter or getter is trivial, you can just write `set;`
or `get;` to get the default implementation. This is different from writing
no getter or setter at all, because that makes the property writeonly or readonly,
respectively. Making both trivial is possible for ABI stability reasons, but it’s
discouraged.

A property has a backing field iff the properties name is referenced by
the getter or setter (in which case it refers to the backing field instead).

To perform a recursive call to a setter or getter, call unqualified `set()`
or `get()` from within a setter.

## FFI
For C compatibility, add an implicit conversion from string literals to 'i8^'.

## Effect system?
Maybe something like:
```
proc foo() [io] { ... }
```
`unsafe` could just be another effect. Might be useful to constrain what functions can do (but
have a way of opting out of it maybe; I feel like this is more useful for libraries than executables).

## `inline` keyword
`inline` can be used as an attribute on a procedure declaration to make it `alwaysinline`, or before
a call to force-inline that call (implement this in MLIR); if a directly recursive call is declared `inline`,
the call is converted to a tail call instead (I’m not sure we can always use `musttail` for this since that’s
only supported for `fastcc`, not `ccc`, so we might have to implement this ourselves in MLIR).

## Pass closures as function pointers to C
Closures whose environment is known to be `nil` at compile time are passed to `native` functions as plain
function pointers; in general, closure parameters of `native` functions are always plain function pointers;
if a closure with a not-null environment needs to be passed to a `native` function, the environment must be
passed as some other parameter; have some syntax for that. E.g.
```
proc some_c_function(proc a (int) -> int, __srcc_env_ptr_for(a)) native extern;
```
When such a function is defined in Source, any access to the closure parameter automatically pulls in the
environment too (the closure is reconstituted in the function prologue); the environment parameter must be
unnamed and can thus not be accessed directly:
```
proc callable_from_c(proc a (int) -> int, __srcc_env_ptr_for(a)) native {
    a(42); // Closure call
}
```
Of course, we should have an alias for this in the standard library so we can write this:
```
proc callable_from_c(proc a (int) -> int, std::ffi::env_ptr_for(a)) native;
```

## Syntax ideas for list comprehensions
```
(for x in xs do x + 4)
(for x in xs : x + 4)
```

## Bigint support
Use libbtommath and add a builtin `bigint` type.

## Late variable intialisation
Essentially, allow this
```
var x;
x = 4;
```
This means that we keep the type of `x` as `var` until we encounter the expression that
initialises it; at that point, we can set the type of the variable; this also means that
if we encounter a use of a variable while its type is still `var`, then we can emit an
error about an access to an uninitialised variable. Maybe this means we can also get rid
of the `type ident` syntax for declaring variables and actually just use `var x : type`?.

Out parameters can also initialise a variable; these require special handling, but I think
it suffices to just handle them in BuildInitialiser(): if we have a LocalRefExpr to a
variable whose type is `var`, and we’re passing it to an out pararameter, then we have a
perfect ‘conversion’. We also need a new ‘InitOutParam’ conversion that when applied sets
the type of the variable to the procedure parameter type.

## Recursion in anonymous functions
Add a `__srcc_this_proc` builtin (the standard library can have a nicer name for it using `alias`).

## Operators
- `and` and `or` should not associate.
-

## Variadic templates.
A homogeneous pack can be declared with the syntax `<type> "..." [ IDENT ]`, where
`<type>` can also be a (possibly deduced) template parameter. A heterogeneous pack
is declared using `var` as the `<type>`. When instantiated, a homogeneous pack is
passed as an array, and a heterogeneous pack as a tuple, e.g.
```
struct s { int x; }
proc f1 (int... as) {}
proc f2 ($T...  ts) {}
proc f3 (var... vs) {}

f1(1, 2, 3);       // 'as' is 'int[3]'.
f2("a", "b", "c"); // 'ts' is 'i8[][3]'.
f3(1, "a", s(3));  // 'vs' is '(int, i8[], s)'.
```
These can then be used like any other array/tuple; additionally, the '...' operator
can be used to spread an array/tuple into a function call or initialiser list:
```
// Iterative example.
//
// The 'for' here is actually a compile-time loop, just like how '[]' on a tuple is
// also a compile-time operation; we could require writing '#for' for consistency, but
// it feels a bit to require '#for' if 'for' would then be a syntax error.
proc sum(var ...vs) -> int {
    int x;
    for v in vs do x += v;
    return x;
}

// Recursive example.
proc sum (var ...vs) -> int {
    #if vs.size == 0 return 0;
    #else return vs[0] + sum(vs[1..]...); // '1..' is eqv. to '1..<vs.size' here.
}
```

# Immutable types
Use `<type> val` as the syntax, e.g. `i8 val`, `int val`; this makes more sense than e.g. `val int`
(‘int value’ vs ‘value int’), and also doesn’t run into the problem that it’s not obvious what
`val int^` would be.

# Failed Ideas
## Renaming copy to var
(This doesn’t work because `proc (var x)` would now be parsed with `x` as the type, even though this
is probably an error. Allowing this would just be too weird...)

- Rename the 'copy' intent to 'var', e.g. 'proc foo (var int s) {}'; this feels more natural
  since 'in' parameters are immutable by default.~~
