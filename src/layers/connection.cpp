#include "layers/connection.hpp"
#include "layers/layer.hpp"
#include "matrix/matrix.hpp"


namespace Layer
{
	template <typename DType>
	LayerConnection<DType>::LayerConnection(Matrix<DType>* input):
		layer(nullptr),
		input(input),
		bias_weights(nullptr)
	{
		output = new Matrix<DType>*;
	}

	template <typename DType>
	LayerConnection<DType>::LayerConnection(Layer<DType>* input):
		LayerConnection(input->output()[0])
	{
		layer = input;
	}

	template <typename DType>
	LayerConnection<DType>::~LayerConnection()
	{
		delete input;
		delete weights;
		delete error;
		delete delta;
		delete output;

		if (bias_weights != nullptr)
		{
			delete bias_weights;
		}
	}

	template <typename DType>
	BackwardConnection<DType>::BackwardConnection(Matrix<DType>*& error, 
		Matrix<DType>* delta, Matrix<DType>* weights):
		error(error),
		delta(delta),
		weights(weights)
	{}

	template <typename DType>
	BackwardConnection<DType>::~BackwardConnection()
	{}

	SPECIALIZE(LayerConnection);
	SPECIALIZE(BackwardConnection);
}
