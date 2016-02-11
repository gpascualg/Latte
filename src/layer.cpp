#include "layer.hpp"
#include "activations/activation.hpp"
#include "fillers/filler.hpp"
#include "matrix.hpp"
#include "matrix_factory.hpp"


template <typename DType>
Layer<DType>::Layer(Shape shape, int num_output, Activation<DType>* activation, Filler<DType>* filler) :
	_activaton(activation),
	_in_shape(shape),
	_out_shape({ shape.m, num_output })
{
	// Never use MatrixFactory here, they mustn't be recycled
	_weights = new Matrix<DType>(shape.n, num_output);
	_output = new Matrix<DType>(shape.m, num_output);
	_diff = new Matrix<DType>(shape.n, num_output);

	// Fill initial weights
	filler->fill(_weights);

	std::cout << "Setting up:" << std::endl;
	std::cout << "Indata shape: (" << shape.m << ", " << shape.n << ")" << std::endl;
	std::cout << "Weight shape: (" << _weights->shape().m << ", " << _weights->shape().n << ")" << std::endl;
	std::cout << "Output shape: (" << _output->shape().m << ", " << _output->shape().n << ")" << std::endl;
	std::cout << "--------" << std::endl << std::endl;
}

template <typename DType>
Layer<DType>::~Layer()
{
	delete _weights;
	delete _output;
	delete _diff;
}

template <typename DType>
Matrix<DType>* Layer<DType>::forward()
{
	_in->mul(_weights, _output);
	_activaton->apply(_output);
	return _output;
}

template <typename DType>
Matrix<DType>* Layer<DType>::backward(Matrix<DType>* error)
{
	_delta = MatrixFactory<DType>::get()->pop(error->shape());
	_activaton->derivative(_output, _delta, error);
	return _delta;
}

template <typename DType>
void Layer<DType>::update()
{
	// _weights = 1.0 * _in.T * _delta + 1.0 * _weights
	_in->T()->mul(_delta, _weights, 1.0, 1.0);
}

template <typename DType> Matrix<DType>* Layer<DType>::W() { return _weights; }
template <typename DType> Shape Layer<DType>::inShape() { return _in_shape; }
template <typename DType> Shape Layer<DType>::outShape() { return _out_shape; }


// Specializations
template Layer<float>::Layer(Shape shape, int num_output, Activation<float>* activation, Filler<float>* filler);
template Layer<double>::Layer(Shape shape, int num_output, Activation<double>* activation, Filler<double>* filler);

template Matrix<float>* Layer<float>::forward();
template Matrix<double>* Layer<double>::forward();

template Matrix<float>* Layer<float>::backward(Matrix<float>* error);
template Matrix<double>* Layer<double>::backward(Matrix<double>* error);

template void Layer<float>::update();
template void Layer<double>::update();

template Matrix<float>* Layer<float>::W();
template Matrix<double>* Layer<double>::W();

template Shape Layer<float>::inShape();
template Shape Layer<double>::inShape();

template Shape Layer<float>::outShape();
template Shape Layer<double>::outShape();
