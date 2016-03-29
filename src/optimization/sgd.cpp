#include "sgd.hpp"

#include <type_traits>

#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "layers/layer.hpp"


namespace Optimizer
{
	template <typename DType>
	SGD<DType>::SGD(std::vector<Layer::FinalizedLayer<DType>> layers) :
		_layers(layers),
		_k(0)
	{}

	template <typename DType>
	void SGD<DType>::optimize()
	{
		LATTE_ASSERT("The following parameters must be set: " << std::endl <<
			"\tTarget: " << std::boolalpha << _target.isSet() << std::endl <<
			"\tLearningRate: " << std::boolalpha << _learning_rate.isSet() << std::endl <<
			"\tIterations: " << std::boolalpha << _iterations.isSet(),
			_target.isSet() && _learning_rate.isSet() && _iterations.isSet());

		int maxIters = _iterations->maxIterations;
		for (_k = 0; _k < maxIters; ++_k)
		{
			forward();
			backward();

			MatrixFactory<float>::get()->update();
		}
	}

	template <typename DType>
	void SGD<DType>::forward()
	{
		std::vector<Layer::FinalizedLayer<DType>> pending = _layers;

		if (_orderedLayers.empty())
		{
			// TODO: Precompute this
			while (!pending.empty())
			{
				for (std::size_t i = 0; i < pending.size(); ++i)
				{
					auto layer = pending[i];

					if (layer->canBeForwarded())
					{
						// Remove from pending
						pending.erase(pending.begin() + i);

						// Add to ordered list
						_orderedLayers.push_back(layer);

						// Forward layer
						layer->forward();

						// If last, add it
						if (layer->isLast())
						{
							_lastLayers.push_back(layer);
						}

						break;	
					}
				}
			}
		}
		else
		{
			for (auto layer : _orderedLayers)
			{
				layer->forward();
			}
		}
	}

	template <typename DType>
	void SGD<DType>::backward()
	{
		// FIXME: Multiple outputs
		// Initial error
		Matrix<DType>* predicted = _lastLayers[0]->output()[0];
		Matrix<DType>* error = (*predicted  - *_target());

		if ((_k % _iterations->printEvery) == 0)
		{
			DType sum = error->sum();
			std::cout << "ERROR: " << (sum / error->shape().prod()) << std::endl;
			std::cout << (*_target())(0, 0) << "\t" << (*_target())(1, 0) << "\t" << (*_target())(2, 0) << "\t" << (*_target())(3, 0) << std::endl;
			std::cout << (*predicted)(0, 0) << "\t" << (*predicted)(1, 0) << "\t" << (*predicted)(2, 0) << "\t" << (*predicted)(3, 0) << std::endl << std::endl;
		}

		// Setup last layers errors
		std::vector<Layer::BackwardConnection<DType>*> previous;
		auto lastLayer = _orderedLayers.rbegin();
		for (; lastLayer != _orderedLayers.rend() && (*lastLayer)->isLast(); ++lastLayer)
		{
			// Setup error
			for (auto connection : (*lastLayer)->connections())
			{
				connection->error = error;
			}

			// Compute backward connection and append error
			auto backConns = (*lastLayer)->backward();
			previous.insert(previous.end(), backConns.begin(), backConns.end());
		}

		// Backward the rest of layers
		for (; lastLayer != _orderedLayers.rend(); ++lastLayer)
		{
			// Foreach backward connection, update error
			for (auto connection : previous)
			{
				Matrix<DType>* delta = MatrixFactory<DType>::get()->pop({ connection->delta->shape().m, connection->weights->shape().m });
				connection->delta->mul(connection->weights->T(), delta);
				connection->error = delta;
			}

			// Compute backward connection
			previous = (*lastLayer)->backward();
		}

		// Update layers
		for (auto layer : _orderedLayers)
		{
			layer->update(_learning_rate());
		}
	}


	SPECIALIZE(SGD);
}
