#include "graph.hpp"
#include "matrix.hpp"
#include "matrix_factory.hpp"

#include "layer.hpp"
#include "sigmoid.hpp"


template <typename DType>
Graph<DType>::Graph(Matrix<DType>* target, Matrix<DType>* data) :
	_target(target),
	_first_in(data),
	_k(0)
{}

template <typename DType>
template <typename LType>
void Graph<DType>::stack(int num_output)
{
	LType* layer = nullptr;

	if (_layers.empty())
	{
		layer = new LType(_first_in->shape(), num_output);
	}
	else
	{
		layer = new LType(_layers.back()->outShape(), num_output);
	}

	stack(layer);
}

template <typename DType>
void Graph<DType>::stack(Layer<DType>* layer)
{
	_layers.push_back(layer);
}

template <typename DType>
void Graph<DType>::forward()
{
	Matrix<DType>* in = _first_in;

	auto it = _layers.begin();
	for (; it != _layers.end(); ++it)
	{
		in = (*it)->forward(in);
	}

	_last_out = in;
}

template <typename DType>
void Graph<DType>::backward()
{
	// Initial error
	Matrix<DType>* error = (*_target - *_last_out);

	if ((_k % 10000) == 0)
	{
		DType sum = DType(0.0);
		for (int i = 0; i < error->shape().prod(); ++i)
			sum += abs((**error)[i]);

		std::cout << "ERROR: " << (sum / error->shape().prod()) << std::endl;
		std::cout << (*_target)(0, 0) << "\t" << (*_target)(1, 0) << "\t" << (*_target)(2, 0) << "\t" << (*_target)(3, 0) << std::endl;
		std::cout << (*_last_out)(0, 0) << "\t" << (*_last_out)(1, 0) << "\t" << (*_last_out)(2, 0) << "\t" << (*_last_out)(3, 0) << std::endl << std::endl;
	}
	++_k;

	// Skip last, it will be manually computed
	auto last_layer = _layers.back();
	auto it = ++_layers.rbegin();

	error = last_layer->backward(error);

	for (; it != _layers.rend(); ++it)
	{
		Matrix<DType>* delta = MatrixFactory<DType>::get()->pop({ last_layer->outShape().m, last_layer->W()->shape().m });

		error->mul(&last_layer->W()->T(), delta);
		last_layer = *it;
		error = last_layer->backward(delta);
	}

	// Update layers
	it = _layers.rbegin();
	for (; it != _layers.rend(); ++it)
	{
		(*it)->update();
	}

	// Update Matrix pool
	MatrixFactory<DType>::get()->update();
}



// Specialization
template Graph<float>::Graph(Matrix<float>* target, Matrix<float>* data);
template Graph<double>::Graph(Matrix<double>* target, Matrix<double>* data);

template void Graph<float>::stack<Sigmoid<float>>(int num_output);
template void Graph<double>::stack<Sigmoid<double>>(int num_output);

template void Graph<float>::stack(Layer<float>* layer);
template void Graph<double>::stack(Layer<double>* layer);

template void Graph<float>::forward();
template void Graph<double>::forward();

template void Graph<float>::backward();
template void Graph<double>::backward();
