#include <gflags/gflags.h>
#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "matrix_factory.hpp"

DEFINE_bool(testing, false, "Set to true to test");

#define DT float




template <typename DType>
class Graph
{
public:
	Graph(Matrix<DType>* target, Matrix<DType>* data):
		_target(target),
		_first_in(data),
		_k(0)
	{}

	void stack(Layer<DType>* layer)
	{
		_layers.push_back(layer);
	}

	template <typename LType>
	void stack(int num_output)
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

	void forward()
	{
		Matrix<DType>* in = _first_in;

		auto it = _layers.begin();
		for (; it != _layers.end(); ++it)
		{
			in = (*it)->forward(in);
		}

		_last_out = in;
	}

	void backward()
	{
		// Initial error
		Matrix<DT>* error = (*_target - *_last_out);

		if ((_k % 10000) == 0)
		{
			DT sum = DT(0.0);
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
			Matrix<DT>* delta = MatrixFactory<DT>::get()->pop({ last_layer->outShape().m, last_layer->W()->shape().m });

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
		MatrixFactory<DT>::get()->update();
	}


private:
	int _k;
	std::vector<Layer<DType>* > _layers;
	Matrix<DType>* _target;
	Matrix<DType>* _first_in;
	Matrix<DType>* _last_out;
};

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	srand(time(NULL));

	Matrix<DT> x(4, 3);
	x(0, 0) = 0; x(0, 1) = 0; x(0, 2) = 1;
	x(1, 0) = 1; x(1, 1) = 1; x(1, 2) = 1;
	x(2, 0) = 1; x(2, 1) = 0; x(2, 2) = 1;
	x(3, 0) = 0; x(3, 1) = 1; x(3, 2) = 1;

	Matrix<DT> y(4, 1);
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 0;
	
	Graph<DT> graph(&y, &x);
	graph.stack<Sigmoid<DT>>(50);
	graph.stack<Sigmoid<DT>>(50);
	graph.stack<Sigmoid<DT>>(50);
	graph.stack<Sigmoid<DT>>(1);
	
	for (int k = 0; k < 600000; ++k)
	{
		graph.forward();
		graph.backward();
	}

	getchar();
	return 0;
}
