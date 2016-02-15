#include "sgd.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"

#include "layers/layer.hpp"
#include "layers/sigmoid_layer.hpp"


template <typename DType>
SGD<DType>::SGD(GenericParameter* data, GenericParameter* target, GenericParameter* learning_rate, GenericParameter* momentum) :
	_data(data->as<Matrix<DType>*>()),
	_target(target->as<Matrix<DType>*>()),
	_learning_rate(learning_rate->as<DType>()),
	_momentum(momentum->as<DType>()),
	_firstLayer(nullptr),
	_lastLayer(nullptr),
	_k(0)
{}

template <typename DType>
template <typename LType>
void SGD<DType>::stack(int num_output)
{
	LType* layer = nullptr;

	if (!_firstLayer)
	{
		layer = new LType{ NamedArguments, LayerConfig<DType>::shape = _data->shape(), LayerConfig<DType>::num_output = num_output };
	}
	else
	{
		layer = new LType{ NamedArguments, LayerConfig<DType>::shape = _lastLayer->outShape(), LayerConfig<DType>::num_output = num_output };
	}

	stack(layer);
}

template <typename DType>
void SGD<DType>::stack(Layer<DType>* layer)
{
	if (!_firstLayer)
	{
		_firstLayer = layer;
		layer->connect(_data);
	}
	else
	{
		layer->connect(_lastLayer);
	}

	_lastLayer = layer;
}

template <typename DType>
void SGD<DType>::forward()
{
	auto it = _firstLayer->iterate();
	for (; it.next(); ++it)
	{
		(*it)->forward();
	}
}

template <typename DType>
void SGD<DType>::backward()
{
	// Initial error
	Matrix<DType>* predicted = _lastLayer->output();
	Matrix<DType>* error = (*predicted  - *_target);

	if ((_k % 10000) == 0)
	{
		DType sum = error->sum();
		std::cout << "ERROR: " << (sum / error->shape().prod()) << std::endl;
		std::cout << (*_target)(0, 0) << "\t" << (*_target)(1, 0) << "\t" << (*_target)(2, 0) << "\t" << (*_target)(3, 0) << std::endl;
		std::cout << (*predicted)(0, 0) << "\t" << (*predicted)(1, 0) << "\t" << (*predicted)(2, 0) << "\t" << (*predicted)(3, 0) << std::endl << std::endl;
	}
	++_k;

	auto it = _lastLayer->iterate();
	error = _lastLayer->backward(error);

	// Skip last, it will be manually computed
	for (--it; it.next(); --it)
	{
		Matrix<DType>* delta = MatrixFactory<DType>::get()->pop({ it.last()->outShape().m, it.last()->W()->shape().m });
		error->mul(it.last()->W()->T(), delta);
		error = (*it)->backward(delta);
	}

	// Update layers
	it = _lastLayer->iterate();
	for (; it.next(); --it)
	{
		(*it)->update(_learning_rate);
	}
}



// Specialization
template SGD<float>::SGD(GenericParameter* data, GenericParameter* target, GenericParameter* learning_rate, GenericParameter* momentum);
template SGD<double>::SGD(GenericParameter* data, GenericParameter* target, GenericParameter* learning_rate, GenericParameter* momentum);

template void SGD<float>::stack<SigmoidLayer<float>>(int num_output);
template void SGD<double>::stack<SigmoidLayer<double>>(int num_output);

template void SGD<float>::stack(Layer<float>* layer);
template void SGD<double>::stack(Layer<double>* layer);

template void SGD<float>::forward();
template void SGD<double>::forward();

template void SGD<float>::backward();
template void SGD<double>::backward();
