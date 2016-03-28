#include "sgd.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"

#include "layers/layer.hpp"


namespace Optimizer
{
	template <typename DType>
	SGD<DType>::SGD(std::vector<Layer::FinalizedLayer<DType>> layers) :
		_layers(layers),
		_k(0)
	{

	}

	template <typename DType>
	void SGD<DType>::forward()
	{
		std::vector<Layer::FinalizedLayer<DType>> pending = _layers;

		if (_orderedLayers.empty())
		{
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

		if ((_k % 10000) == 0)
		{
			DType sum = error->sum();
			std::cout << "ERROR: " << (sum / error->shape().prod()) << std::endl;
			std::cout << (*_target())(0, 0) << "\t" << (*_target())(1, 0) << "\t" << (*_target())(2, 0) << "\t" << (*_target())(3, 0) << std::endl;
			std::cout << (*predicted)(0, 0) << "\t" << (*predicted)(1, 0) << "\t" << (*predicted)(2, 0) << "\t" << (*predicted)(3, 0) << std::endl << std::endl;
		}
		++_k;

		auto lastLayer = _orderedLayers.rbegin();
		(*lastLayer)->backward({error});

		auto previousLayer = *lastLayer;

		for (++lastLayer; lastLayer != _orderedLayers.rend(); ++lastLayer)
		{
			Matrix<DType>* delta = MatrixFactory<DType>::get()->pop({ previousLayer->outShape().m, previousLayer->W()[0]->shape().m });
			error->mul(previousLayer->W()[0]->T(), delta);
			error = (*lastLayer)->backward({delta})[0];

			previousLayer = *lastLayer;
		}

		// Update layers
		for (auto layer : _orderedLayers)
		{
			layer->update(_learning_rate());
		}
	}


	SPECIALIZE(SGD);
}
