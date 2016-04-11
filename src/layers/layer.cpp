#include "layers/layer.hpp"
#include "activations/activation.hpp"
#include "fillers/filler.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "utils/rng.hpp"

namespace Layer
{
	template <typename DType>
	Layer<DType>::Layer() :
		_isFirst(false),
		_forwardDone(false),
		_maxInputs(1),
		_forwardsTo(0)
	{}

	template <typename DType>
	FinalizedLayer<DType>::FinalizedLayer(Layer<DType>* layer) :
		_layer(layer)
	{
	    // Set bias value
	    if (layer->_bias.isSet())
	    {
	        layer->_bias_values = new Matrix<DType>(layer->_inShape->m, 1, layer->_bias->value);
	    }
	
		/*
		std::cout << "Setting up:" << std::endl;
		std::cout << "Indata shape: (" << layer->_inShape->m << ", " << layer->_inShape->n << ")" << std::endl;
		std::cout << "Weight shape: (" << layer->_weights[0]->shape().m << ", " << layer->_weights[0]->shape().n << ")" << std::endl;
		std::cout << "Output shape: (" << layer->_output[0]->shape().m << ", " << layer->_output[0]->shape().n << ")" << std::endl;
	    std::cout << "Has Dropout: (" << std::boolalpha << layer->_dropout.isSet() << ": " << (layer->_dropout.isSet() ? layer->_dropout() : 0) << ")" << std::endl;

	    if (layer->_bias.isSet())
	    {
	    	std::cout << "Bias Weight shape: (" << layer->_bias_weights[0]->shape().m << ", " << layer->_bias_weights[0]->shape().n << ")" << std::endl;
			std::cout << "Bias Values shape: (" << layer->_bias_values->shape().m << ", " << layer->_bias_values->shape().n << ")" << std::endl;
	    }

		std::cout << "--------" << std::endl << std::endl;
		*/
	}

	template <typename DType>
	FinalizedLayer<DType>::FinalizedLayer(const FinalizedLayer& layer)
	{
		_layer = layer._layer;
	}

	template <typename DType>
	Layer<DType>::~Layer()
	{
		for (auto* connection : _connections)
		{
			delete connection;
			//delete _diff;
		}
	}

	template <typename DType>
	Layer<DType>& Layer<DType>::operator<<(Matrix<DType>& other)
	{
		LATTE_ASSERT("Layer can not have that much inputs: " <<
			_connections.size() << " >= " << _maxInputs, 
			_connections.size() < (std::size_t)_maxInputs);

		if (_connections.size() > 0)
		{
			// Size restrictions
			for (auto* conn : _connections)
			{
				LATTE_ASSERT("All inputs must have the same size",
					conn->input->shape().m == outShape().m && 
					conn->input->shape().n == outShape().n);
			}
		}
		else
		{
			// Set size
			*this << ExtConfig::Shape(other.shape());
		}

		LayerConnection<DType>* connection = new LayerConnection<DType>(&other);
		_connections.emplace_back(connection);

		// Set other layer output
		if (_output.size() == 0)
		{
			auto output = new Matrix<DType>(_inShape->m, _numOutput());
			_output.emplace_back(output);
		}

		// Set input link
		connection->input = &other;
	
		// Never use MatrixFactory here, they mustn't be recycled
		connection->weights = new Matrix<DType>(_inShape->n, _numOutput());
		*connection->output = _output[0];

		//layer->_diff = new Matrix<DType>(layer->_inShape->n, layer->_numOutput());

		// Fill initial weights
		if (_filler.isSet())
		{
			_filler->fill(connection->weights);
		}

		// Set bias
	    if (_bias.isSet())
	    {
			connection->bias_weights = new Matrix<DType>(1, _numOutput());
			_bias->filler->fill(connection->bias_weights);
		}

		return *this;
	}

	template <typename DType>
	Layer<DType>& Layer<DType>::operator<<(Layer<DType>& other)
	{
		// Connect matrix
		*this << *other._output[0];

		auto* connection = _connections.back();
		connection->layer = &other;

		++other._forwardsTo;

		return *this;
	}

	template <typename DType>
	bool Layer<DType>::canBeForwarded()
	{
		if (_isFirst)
		{
			return true;
		}

		bool result = true;
		for (std::size_t i = 0; i < _connections.size() && result; ++i)
		{
			result &= _connections[i]->layer->_forwardDone;
		}

		return result;
	}

	template <typename DType>
	bool Layer<DType>::isLast()
	{
		return _forwardsTo == 0 && !_isFirst;
	}

	template <typename DType>
	Matrix<DType>* Layer<DType>::forward()
	{
		for (auto* connection : _connections)
		{
			// Compute weights product
			connection->input->mul(connection->weights, *connection->output);
		    
		    // If using biases, sum them to our output
		    if (_bias.isSet())
		    {
				_bias_values->mul(connection->bias_weights, *connection->output, DType(1.0), DType(1.0));
		    }

			// Apply activation function
			_activation->apply(*connection->output);

			// Drop neurons
			if (_dropout.isSet())
			{
				for (int i = 0; i < (*connection->output)->shape().prod(); ++i)
				{
					(**connection->output)[i] *= (rng()->nextFloat() >= _dropout() ? (DType(1.0) / (1 - _dropout())) : DType(0.0));
				}
			}
		}

		// Flag as done
		_forwardDone = true;

		// FIXME: Return?
		return nullptr;
	}

	template <typename DType>
	std::vector<BackwardConnection<DType>*> Layer<DType>::backward()
	{
		std::vector<BackwardConnection<DType>*> backwardConnections;

		for (auto* connection : _connections)
		{
			Matrix<DType>* delta = MatrixFactory<DType>::get()->pop(connection->error->shape());
			_activation->derivative(*connection->output, delta, connection->error);

			// Save delta
			connection->delta = delta;

			// TODO: Precompute this
			// Find connections from connection->layer whose output is this input
			if (connection->layer)
			{
				for (auto* foreigner : connection->layer->connections())
				{
					if (*foreigner->output == connection->input)
					{
						backwardConnections.emplace_back(new BackwardConnection<DType>(
							foreigner->error, delta, connection->weights));
					}
				}
			}
		}

		return backwardConnections;
	}

	template <typename DType>
	void Layer<DType>::update(float learningRate)
	{
		for (auto* connection : _connections)
		{
		    // Update biases
		    // _bias_weights = 1.0 * _delta * _bias_values.T + 1.0 * _bias_weights
		    if (_bias.isSet())
		    {
				_bias_values->T()->mul(connection->delta, connection->bias_weights, DType(1.0), DType(1.0));
		    }
		    
			// _weights = -learning_rate * _in.T * _delta + 1.0 * _weights
			connection->input->T()->mul(connection->delta, connection->weights, -DType(learningRate), 1.0);
		}

		// Flag as not done (unless its first)
		_forwardDone = false;
	}

	template class Layer<float>;
	template class Layer<double>;

	template class FinalizedLayer<float>;
	template class FinalizedLayer<double>;
}
