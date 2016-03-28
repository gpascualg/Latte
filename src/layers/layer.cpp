#include "layer.hpp"
#include "activations/activation.hpp"
#include "fillers/filler.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "utils/rng.hpp"

namespace Layer
{
	template <typename DType>
	Layer<DType>::Layer() :
		_forwardDone(false),
		_maxInputs(1)
	{}

	template <typename DType>
	FinalizedLayer<DType>::FinalizedLayer(Layer<DType>* layer) :
		_layer(layer)
	{
		// Never use MatrixFactory here, they mustn't be recycled
		for (Layer<DType>* l : _inputs)
		{
			layer->_weights.emplace_back(new Matrix<DType>(layer->_inShape->n, layer->_numOutput()));
			layer->_output.emplace_back(new Matrix<DType>(layer->_inShape->m, layer->_numOutput()));
			//layer->_diff = new Matrix<DType>(layer->_inShape->n, layer->_numOutput());

			// Fill initial weights
			layer->_filler->fill(layer->_weights);

			// Set bias
		    if (layer->_bias.isSet())
		    {
				layer->_bias_weights.emplace(new Matrix<DType>(1, layer->_numOutput()));
				layer->_bias->filler->fill(layer->_bias_weights);
			}
		}
	    
	    // Set bias value
	    if (layer->_bias.isSet())
	    {
	        layer->_bias_values = new Matrix<DType>(layer->_inShape->m, 1, layer->_bias->value);
	    }
	
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
	}

	template <typename DType>
	Layer<DType>::~Layer()
	{
		for (Layer<DType>* l : _inputs)
		{
			delete _weights;
			delete _output;
			//delete _diff;
		}
	}

	template <typename DType>
	Matrix<DType>* Layer<DType>::forward()
	{
		for (int i = 0; i < _inputs.size(); ++i)
		{
			// Compute weights product
			_inputs[i]->_output->mul(_weights[i], _output[i]);
		    
		    // If using biases, sum them to our output
		    if (_bias.isSet())
		    {
				_bias_values->mul(_bias_weights[i], _output[i], DType(1.0), DType(1.0));
		    }

			// Apply activation function
			_activation->apply(_output[i]);

			// Drop neurons
			if (_dropout.isSet())
			{
				for (int i = 0; i < _output[i]->shape().prod(); ++i)
				{
					(*_output[i])[i] *= (rng()->nextFloat() >= _dropout() ? (DType(1.0) / (1 - _dropout())) : DType(0.0));
				}
			}

			// Flag as done
			_forwardDone = true;
		}

		// FIXME: Return?
		return nullptr;
	}

	template <typename DType>
	Matrix<DType>* Layer<DType>::backward(Matrix<DType>* error)
	{
		_delta.clear();
		
		for (int i = 0; i < _inputees.size(); ++i)
		{
			_delta.emplace(MatrixFactory<DType>::get()->pop(error->shape()));
			_activation->derivative(_output[i], _delta[i], error);
		}

		return _delta[0];
	}

	template <typename DType>
	void Layer<DType>::update(DType learning_rate)
	{
		for (int i = 0; i < _inputs.size(); ++i)
		{
		    // Update biases
		    // _bias_weights = 1.0 * _delta * _bias_values.T + 1.0 * _bias_weights
		    if (_bias.isSet())
		    {
				_bias_values->T()->mul(_delta[i], _bias_weights[i], DType(1.0), DType(1.0));
		    }
		    
			// _weights = -learning_rate * _in.T * _delta + 1.0 * _weights
			_inputs[i]->_output->T()->mul(_delta[i], _weights[i], -learning_rate, 1.0);
		}

		// Flag as not done
		_forwardDone = false;
	}

	template <typename DType>
	void Layer<DType>::connect(FinalizedLayer<DType>& layer)
	{ 
	    /*layer->_next.push_back(this);
	    _previous.push_back(layer);*/
	    _in = layer->_output;
	}

	template <typename DType>
	void Layer<DType>::connect(Matrix<DType>* data)
	{ 
	    _in = data; 
	}

	/*
	template <typename DType>
	typename Layer<DType>::LayerIterator Layer<DType>::iterate()
	{ 
	    return LayerIterator(this); 
	}
	*/
	template class Layer<float>;
	template class Layer<double>;

	template class FinalizedLayer<float>;
	template class FinalizedLayer<double>;
}
