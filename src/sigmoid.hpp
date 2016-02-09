#pragma once

#include "layer.hpp"

template <typename DType>
class Sigmoid : public Layer<DType>
{
public:
	Sigmoid(Shape shape, int num_output);

	Matrix<DType>* forward(Matrix<DType>* in);
	Matrix<DType>* backward(Matrix<DType>* error);
};
