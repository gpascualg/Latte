#pragma once

#include "magic/factory.hpp"

template <typename DType>
class Matrix;


template <typename DType>
class Activation
{
public:
	virtual ~Activation() {}

	inline void apply(Matrix<DType>* matrix)
	{
		apply(matrix, matrix);
	}

	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* alpha)
	{
		derivative(matrix, matrix, alpha);
	}

	virtual void apply(Matrix<DType>* matrix, Matrix<DType>* dest) = 0;
	virtual void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) = 0;
    
protected:
	Activation() {};
};

REGISTER_FACTORY(Activation);
