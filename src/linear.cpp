#include "linear.hpp"
#include "matrix.hpp"
#include "matrix_factory.hpp"

#include "layer.hpp"
#include "sigmoid.hpp"


template <typename DType>
Linear<DType>::Linear(Matrix<DType>* data, Matrix<DType>* target) :
	_data(data),
	_target(target),
	_firstLayer(nullptr),
	_lastLayer(nullptr),
	_k(0)
{}

template <typename DType>
template <typename LType>
void Linear<DType>::stack(int num_output)
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
void Linear<DType>::stack(Layer<DType>* layer)
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
void Linear<DType>::forward()
{
	auto it = _firstLayer->iterate();
	for (; it.next(); ++it)
	{
		(*it)->forward();
	}
}

template <typename DType>
void Linear<DType>::backward()
{
	// Initial error
	Matrix<DType>* predicted = _lastLayer->output();
	Matrix<DType>* error = (*_target - *predicted);

	if ((_k % 10000) == 0)
	{
		DType sum = DType(0.0);
		for (int i = 0; i < error->shape().prod(); ++i)
			sum += abs((*error)[i]);

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
template Linear<float>::Linear(Matrix<float>* target, Matrix<float>* data);
template Linear<double>::Linear(Matrix<double>* target, Matrix<double>* data);

template void Linear<float>::stack<Sigmoid<float>>(int num_output);
template void Linear<double>::stack<Sigmoid<double>>(int num_output);

template void Linear<float>::stack(Layer<float>* layer);
template void Linear<double>::stack(Layer<double>* layer);

template void Linear<float>::forward();
template void Linear<double>::forward();

template void Linear<float>::backward();
template void Linear<double>::backward();
