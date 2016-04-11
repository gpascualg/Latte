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

struct Iterations
{
	Iterations() : Iterations(0, 0) {}
	Iterations(int max, int every):
		maxIterations(max), printEvery(every)
	{}

	int maxIterations;
	int printEvery;
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

#define SPECIALIZE_S(What) \
	template struct What<float>; \
	template struct What<double>;
