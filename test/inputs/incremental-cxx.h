#ifndef SOURCE_INCREMENTAL_CXX_H
#define SOURCE_INCREMENTAL_CXX_H

template<typename T> struct IncrementalBox {
    T value;
    constexpr int get() const { return twice(value); }
    static constexpr int twice(T value) { return value * 2; }
};

#define LATE_TEMPLATE IncrementalBox<int>{21}.get()
#define SECOND_TEMPLATE IncrementalBox<unsigned>{22}.get()

struct IncrementalValue { int value; };
#define BOX_VALUE IncrementalValue{23}

inline int incremental_counter = 0;
#define SIDE_EFFECT (++incremental_counter)
#define COUNTER_VALUE (incremental_counter + 0)

// These require CodeGen callbacks during parsing, even without Source imports.
struct IncrementalInline {
    [[gnu::used]] int method() { return 45; }
};

struct IncrementalVirtual { virtual int method(); };
int IncrementalVirtual::method() { return 46; }

#define LATE_RUNTIME IncrementalBox<int>{incremental_counter}.get()
#define INVALID_LATE (missing_late_name + 1)
#endif
