#pragma once

struct CxxObject {
    int value = 17;
    CxxObject() = default;
    explicit CxxObject(int x) : value(x) {}
    CxxObject(const CxxObject&) = delete;
    CxxObject(CxxObject&&) = delete;
    ~CxxObject();
};

inline CxxObject::~CxxObject() { value = -1; }

struct CxxBase {
    int base = 3;
    virtual ~CxxBase() = default;
};

struct CxxDerived : virtual CxxBase {
    int value;
    explicit CxxDerived(int x) : value(x) {}
    ~CxxDerived() override = default;
};

class CxxHidden {
    int hidden = 5;
public:
    CxxHidden() = default;
    ~CxxHidden() = default;
};

struct CxxPlain { int value; };
struct CxxNoDefault { explicit CxxNoDefault(int = 8) {} };
struct CxxPrivate {
    CxxPrivate() = default;
private:
    ~CxxPrivate() = default;
};

struct CxxOverloaded {
    int value;
    explicit CxxOverloaded(int x) : value(x) {}
    explicit CxxOverloaded(bool x) : value(x ? 1 : 0) {}
    ~CxxOverloaded() = default;
};

struct CxxReference {
    int* value;
    explicit CxxReference(int& x) : value(&x) {}
    ~CxxReference() { ++*value; }
};

struct CxxDeleted {
    CxxDeleted() = delete;
};

struct CxxTemplateOnly {
    template<class T> explicit CxxTemplateOnly(T) {}
};

class CxxTrivial {
    int hidden;
public:
    int visible;
};

// This is what we emit for our constructor wrappers; test that this works,
// though a user should never write this manually.
#include <new>
#include <type_traits>
[[clang::always_inline, clang::annotate("__srcc__", "cxx_constructor_wrapper", std::type_identity<CxxObject>{})]]
inline void annotated_construct(void* destination, int value);
inline void annotated_construct(void* destination, int value) {
    ::new (destination) CxxObject(value);
}
