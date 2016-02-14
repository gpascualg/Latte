#pragma once

#include "magic/factory.hpp"

template <typename DType>
class Matrix;

template <typename DType>
class Filler
{
public:
	virtual void fill(Matrix<DType>* weights) {};

protected:
	Filler() {};
};

REGISTER_FACTORY(Filler);
