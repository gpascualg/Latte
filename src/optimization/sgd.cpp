#include "sgd.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"

#include "layers/layer.hpp"
#include "layers/sigmoid_layer.hpp"


template <typename DType>
SGD<DType>::SGD(Matrix<DType>* data, Matrix<DType>* target) :
	_data(data),
	_target(target),
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
		layer = new LType(_data->shape(), num_output);
	}
	else
	{
		layer = new LType(_lastLayer->outShape(), num_output);
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
	Matrix<DType>* error = (*_target - *predicted);

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
		(*it)->update();
	}
}



// Specialization
template SGD<float>::SGD(Matrix<float>* target, Matrix<float>* data);
template SGD<double>::SGD(Matrix<double>* target, Matrix<double>* data);

template void SGD<float>::stack<SigmoidLayer<float>>(int num_output);
template void SGD<double>::stack<SigmoidLayer<double>>(int num_output);

template void SGD<float>::stack(Layer<float>* layer);
template void SGD<double>::stack(Layer<double>* layer);

template void SGD<float>::forward();
template void SGD<double>::forward();

template void SGD<float>::backward();
template void SGD<double>::backward();
