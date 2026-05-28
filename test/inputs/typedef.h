#pragma once

typedef __INT32_TYPE__ int_typedef;
using int_using = __INT32_TYPE__;

typedef struct { int x; } anon_class_typedef;
using anon_class_using = struct { int x; };

struct s { int x; };
typedef s named_class_typedef;
using named_class_using = s;