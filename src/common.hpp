#pragma once

template <typename DType>
class Activation;

template <typename DType>
class Filler;

template <typename DType>
struct BiasConfig
{
    bool use_bias;
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
