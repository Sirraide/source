#ifndef SOURCE_INCREMENTAL_CXX_END_H
#define SOURCE_INCREMENTAL_CXX_END_H
// This is diagnosed at the end of the TU, rather than while parsing a buffer.
static int incremental_missing();
inline int incremental_caller() { return incremental_missing(); }
#endif
