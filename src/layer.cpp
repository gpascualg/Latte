#pragma once

#include "matrix.hpp"

template <typename DType>
Layer<Dtype>::Layer(Shape shape, int num_output) :
	_in_shape(shape),
	_out_shape({ shape.m, num_output })
{
	_weights = new Matrix<DType>(shape.n, num_output);
	_output = new Matrix<DType>(shape.m, num_output);
	_diff = new Matrix<DType>(shape.n, num_output);

	for (int i = 0; i < _weights->shape().prod(); ++i)
	{
		(**_weights)[i] = DType(rand()) / RAND_MAX - DType(0.5);
	}

	std::cout << "Setting up:" << std::endl;
	std::cout << "Indata shape: (" << shape.m << ", " << shape.n << ")" << std::endl;
	std::cout << "Weight shape: (" << _weights->shape().m << ", " << _weights->shape().n << ")" << std::endl;
	std::cout << "Output shape: (" << _output->shape().m << ", " << _output->shape().n << ")" << std::endl;
	std::cout << "--------" << std::endl << std::endl;
}

template <typename DType>
void Layer<DType>::update()
{
	*_weights += *_diff;
}


// Specializations
template Layer<float>::Layer(Shape shape, int num_output);
template Layer<double>::Layer(Shape shape, int num_output);

template void Layer<float>::update();
template void Layer<double>::update();
