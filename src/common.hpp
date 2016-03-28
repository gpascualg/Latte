#pragma once

template <typename DType>
class Activation;

template <typename DType>
class Filler;

template <typename DType>
struct Bias
{
    DType value;
    Filler<DType>* filler;
};


struct Shape
{
	int m;
	int n;

	Shape() : Shape(0, 0) {}
	Shape(int m, int n) : m(m), n(n) {}
	inline int prod() { return m*n; }
	inline Shape T() { return{ n, m }; }
};

#include <iostream>
#define LATTE_ASSERT(message, ...) do { \
    if(!(__VA_ARGS__)) { \
        std::cout << message << std::endl; \
        std::terminate(); \
    } \
} while(0)

#define SPECIALIZE(What) \
	template class What<float>; \
	template class What<double>;

