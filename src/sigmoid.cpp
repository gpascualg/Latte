#include "sigmoid.hpp"
#include "matrix.hpp"
#include "activations/sigmoid_activation.hpp"

template <typename DType>
Sigmoid<DType>::Sigmoid(Shape shape, int num_output) :
	Layer(shape, num_output, Activation<DType>::get<SigmoidActivation<DType>>())
{}

template <typename DType>
Matrix<DType>* Sigmoid<DType>::forward(Matrix<DType>* in)
{
	// _in * _weights
	_in = in;
	_in->mul(_weights, _output);

	_activaton->apply(_output);

	return _output;
}

template <typename DType>
Matrix<DType>* Sigmoid<DType>::backward(Matrix<DType>* error)
{
	Matrix<DType>* delta = MatrixFactory<DType>::get()->pop(error->shape());
	_activaton->derivative(_output, delta, error);

	_in->T()->mul(delta, _diff);
	return delta;
}


// Specializations
template Sigmoid<float>::Sigmoid(Shape shape, int num_output);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output);

template Matrix<float>* Sigmoid<float>::forward(Matrix<float>* in);
template Matrix<double>* Sigmoid<double>::forward(Matrix<double>* in);

template Matrix<float>* Sigmoid<float>::backward(Matrix<float>* error);
template Matrix<double>* Sigmoid<double>::backward(Matrix<double>* error);
