#include "layers/layer.hpp"
#include "layers/loss.hpp"
#include "layers/euclidean_loss.hpp"
#include "layers/connection.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"


namespace Layer
{
	template <typename DType>
	Matrix<DType>* EuclideanLoss<DType>::forward()
	{
		Matrix<DType>* predicted = _connections[0]->input;
		_loss = (*predicted - *_target());

		Matrix<DType>* output = *_connections[0]->output;
		(*output)[0] = _loss->sum() / (2 * _loss->shape().prod());
		
		// Flag as done
		_forwardDone = true;

		// FIXME: Return?
		return nullptr;
	}

	template <typename DType>
	std::vector<BackwardConnection<DType>*> EuclideanLoss<DType>::backward()
	{
		for (auto* foreigner : _connections[0]->layer->connections())
		{
			if (*foreigner->output == _connections[0]->input)
			{
				foreigner->error = _loss;
			}
		}

		return {};
	}

	SPECIALIZE(EuclideanLoss);
}
