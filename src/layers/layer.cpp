#include "layer.hpp"
#include "activations/activation.hpp"
#include "fillers/filler.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "utils/rng.hpp"

namespace Layer
{
	template <typename DType>
	Layer<DType>::Layer()
	{}

	template <typename DType>
	FinalizedLayer<DType>::FinalizedLayer(Layer<DType>* layer) :
		_layer(layer)
	{
		// Never use MatrixFactory here, they mustn't be recycled
		layer->_weights = new Matrix<DType>(layer->_inShape->n, layer->_numOutput());
		layer->_output = new Matrix<DType>(layer->_inShape->m, layer->_numOutput());
		layer->_diff = new Matrix<DType>(layer->_inShape->n, layer->_numOutput());

		// Fill initial weights
		layer->_filler->fill(layer->_weights);
	    
	    // Enable bias
	    if (layer->_bias.isSet())
	    {
			layer->_bias_weights = new Matrix<DType>(1, layer->_numOutput());
	        layer->_bias_values = new Matrix<DType>(layer->_inShape->m, 1, layer->_bias->value);

			layer->_bias->filler->fill(layer->_bias_weights);
	    }
	
		std::cout << "Setting up:" << std::endl;
		std::cout << "Indata shape: (" << layer->_inShape->m << ", " << layer->_inShape->n << ")" << std::endl;
		std::cout << "Weight shape: (" << layer->_weights->shape().m << ", " << layer->_weights->shape().n << ")" << std::endl;
		std::cout << "Output shape: (" << layer->_output->shape().m << ", " << layer->_output->shape().n << ")" << std::endl;
	    std::cout << "Has Dropout: (" << std::boolalpha << layer->_dropout.isSet() << " (" << layer->_dropout() << ")" << std::endl;

	    if (layer->_bias.isSet())
	    {
	    	std::cout << "Bias Weight shape: (" << layer->_bias_weights->shape().m << ", " << layer->_bias_weights->shape().n << ")" << std::endl;
			std::cout << "Bias Values shape: (" << layer->_bias_values->shape().m << ", " << layer->_bias_values->shape().n << ")" << std::endl;
	    }

		std::cout << "--------" << std::endl << std::endl;
	}

	template <typename DType>
	Layer<DType>::~Layer()
	{
		delete _weights;
		delete _output;
		delete _diff;
	}

	template <typename DType>
	Matrix<DType>* Layer<DType>::forward()
	{
		// Compute weights product
		_in->mul(_weights, _output);
	    
	    // If using biases, sum them to our output
	    if (_bias.isSet())
	    {
			_bias_values->mul(_bias_weights, _output, DType(1.0), DType(1.0));
	    }

		// Apply activation function
		_activation->apply(_output);

		// Drop neurons
		if (_dropout.isSet())
		{
			for (int i = 0; i < _output->shape().prod(); ++i)
			{
				(*_output)[i] *= (rng()->nextFloat() >= _dropout() ? (DType(1.0) / (1 - _dropout())) : DType(0.0));
			}
		}

		return _output;
	}

	template <typename DType>
	Matrix<DType>* Layer<DType>::backward(Matrix<DType>* error)
	{
		_delta = MatrixFactory<DType>::get()->pop(error->shape());
		_activation->derivative(_output, _delta, error);
		return _delta;
	}

	template <typename DType>
	void Layer<DType>::update(DType learning_rate)
	{
	    // Update biases
	    // _bias_weights = 1.0 * _delta * _bias_values.T + 1.0 * _bias_weights
	    if (_bias.isSet())
	    {
			_bias_values->T()->mul(_delta, _bias_weights, DType(1.0), DType(1.0));
	    }
	    
		// _weights = -learning_rate * _in.T * _delta + 1.0 * _weights
		_in->T()->mul(_delta, _weights, -learning_rate, 1.0);
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
