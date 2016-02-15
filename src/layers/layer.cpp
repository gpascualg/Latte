#include "layer.hpp"
#include "activations/activation.hpp"
#include "fillers/filler.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "utils/rng.hpp"


template <typename DType>
Layer<DType>::Layer(Shape shape, int num_output, Activation<DType>* activation, 
    Filler<DType>* filler, DType dropout_ratio) :
	_activaton(activation),
	_in_shape(shape),
	_out_shape({ shape.m, num_output }),
    _dropout_ratio(dropout_ratio),
    _has_dropout(dropout_ratio > DType(0.0))
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
    std::cout << "Has Dropout: (" << std::boolalpha << _has_dropout << " (" << _dropout_ratio << ")" << std::endl;
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
    
    if (_has_dropout)
    {
        for (int i = 0; i < _output->shape().prod(); ++i)
        {
            (*_output)[i] *= (rng()->nextFloat() >= _dropout_ratio ? DType(1.0) : DType(0.0)) *
                (DType(1.0) / (1 - _dropout_ratio));
        }
    }
    
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
void Layer<DType>::update(DType learning_rate)
{
	// _weights = learning_rate * _in.T * _delta + 1.0 * _weights
	_in->T()->mul(_delta, _weights, -learning_rate, 1.0);
}

template <typename DType>
void Layer<DType>::connect(Layer<DType>* layer)
{ 
    layer->_next.push_back(this);
    _previous.push_back(layer);
    _in = layer->_output;
}

template <typename DType>
void Layer<DType>::connect(Matrix<DType>* data)
{ 
    _in = data; 
}

template <typename DType>
typename Layer<DType>::LayerIterator Layer<DType>::iterate()
{ 
    return LayerIterator(this); 
}


// Specializations
template Layer<float>::Layer(Shape shape, int num_output, Activation<float>* activation, 
    Filler<float>* filler, float dropout_ratio);
template Layer<double>::Layer(Shape shape, int num_output, Activation<double>* activation, 
    Filler<double>* filler, double dropout_ratio);

template Matrix<float>* Layer<float>::forward();
template Matrix<double>* Layer<double>::forward();

template Matrix<float>* Layer<float>::backward(Matrix<float>* error);
template Matrix<double>* Layer<double>::backward(Matrix<double>* error);

template void Layer<float>::update(float learning_rate);
template void Layer<double>::update(double learning_rate);

template void Layer<float>::connect(Layer<float>* layer);
template void Layer<double>::connect(Layer<double>* layer);

template void Layer<float>::connect(Matrix<float>* data);
template void Layer<double>::connect(Matrix<double>* data);

template Layer<float>::LayerIterator Layer<float>::iterate();
template Layer<double>::LayerIterator Layer<double>::iterate();
