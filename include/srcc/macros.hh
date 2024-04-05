#ifndef SOURCE_MACROS_HH
#define SOURCE_MACROS_HH

#define STR_(X) #X
#define STR(X) STR_(X)

#define CAT_(X, Y) X##Y
#define CAT(X, Y) CAT_(X, Y)

#endif // SOURCE_MACROS_HH
