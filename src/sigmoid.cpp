#include "sigmoid.hpp"
#include "matrix.hpp"

template <typename DType>
Sigmoid<DType>::Sigmoid(Shape shape, int num_output) :
	Layer(shape, num_output)
{}

template <typename DType>
Matrix<DType>* Sigmoid<DType>::forward(Matrix<DType>* in)
{
	// _in * _weights
	_in = in;
	_in->mul(_weights, _output);

	// 1 / (1 + exp(-_output))
	for (int i = 0; i < _output->shape().prod(); ++i)
	{
		(*_output)[i] = DType(1.0) / (DType(1.0) + exp(-(*_output)[i]));
	}

	return _output;
}

template <typename DType>
Matrix<DType>* Sigmoid<DType>::backward(Matrix<DType>* error)
{
	Matrix<DType>* delta = MatrixFactory<DType>::get()->pop(error->shape());
	for (int i = 0; i < _output->shape().prod(); ++i)
	{
		(*delta)[i] = (DType(1.0) - (*_output)[i]) * (*_output)[i] * (*error)[i];
	}

	_in->T().mul(delta, _diff);
	return delta;
}


// Specializations
template Sigmoid<float>::Sigmoid(Shape shape, int num_output);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output);

template Matrix<float>* Sigmoid<float>::forward(Matrix<float>* in);
template Matrix<double>* Sigmoid<double>::forward(Matrix<double>* in);

template Matrix<float>* Sigmoid<float>::backward(Matrix<float>* error);
template Matrix<double>* Sigmoid<double>::backward(Matrix<double>* error);
