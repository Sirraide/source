#pragma once

typedef __INT32_TYPE__ int_typedef;
using int_using = __INT32_TYPE__;

typedef struct { int x; } anon_class_typedef;
using anon_class_using = struct { int x; };

struct s { int x; };
typedef s named_class_typedef;
using named_class_using = s;

typedef struct same_name_typedef {
    int x;
} same_name_typedef;

using same_name_using = struct same_name_using {
    int x;
};

typedef struct foo {
    int x;
} different_name_typedef;

using different_name_using = struct bar {
    int x;
};

union Union {
    __INT32_TYPE__ x;
    __INT64_TYPE__ y;
    __INT64_TYPE__ z;
};

typedef Union named_union_typedef;
using named_union_using = Union;

typedef union { int x; } anon_union_typedef;
using anon_union_using = struct { int x; };

typedef union same_name_union_typedef {
    int x;
} same_name_union_typedef;

using same_name_union_using = union same_name_union_using {
    int x;
};

typedef union ufoo {
    int x;
} different_name_union_typedef;

using different_name_union_using = union ubar {
    int x;
};