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
		Matrix<DType>* predicted = this->_connections[0]->input;
		this->_loss = (*predicted - *this->_target());

		Matrix<DType>* output = *this->_connections[0]->output;
		(*output)[0] = this->_loss->sum() / (2 * this->_loss->shape().prod());
		
		// Flag as done
		this->_forwardDone = true;

		// FIXME: Return?
		return nullptr;
	}

	template <typename DType>
	std::vector<BackwardConnection<DType>*> EuclideanLoss<DType>::backward()
	{
		for (auto* foreigner : this->_connections[0]->layer->connections())
		{
			if (*foreigner->output == this->_connections[0]->input)
			{
				foreigner->error = this->_loss;
			}
		}

		return {};
	}

	SPECIALIZE(EuclideanLoss);
}
