#pragma once

struct ConstantBytes { int first; int second; };
struct ConstantPadding { char first; int second; };
struct ConstantBits { unsigned value : 3; };
union ConstantUnion { int value; };
