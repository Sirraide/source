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