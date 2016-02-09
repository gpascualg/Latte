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

	// 1 / (1 + exp(_output))
	_output->exp(DType(-1.0));
	*_output += 1;
	_output->pdiv(DType(1.0));
	return _output;
}

template <typename DType>
Matrix<DType>* Sigmoid<DType>::backward(Matrix<DType>* error)
{
	Matrix<DType>* derivative = DType(1.0) - *_output;
	*derivative *= *_output;
	*error *= *derivative;
	_in->T().mul(error, _diff);
	return error;
}


// Specializations
template Sigmoid<float>::Sigmoid(Shape shape, int num_output);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output);

template Matrix<float>* Sigmoid<float>::forward(Matrix<float>* in);
template Matrix<double>* Sigmoid<double>::forward(Matrix<double>* in);

template Matrix<float>* Sigmoid<float>::backward(Matrix<float>* error);
template Matrix<double>* Sigmoid<double>::backward(Matrix<double>* error);
