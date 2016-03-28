#pragma once

#include <vector>

#include "layers/config/formatter_specialize.hpp"


template <typename DType>
class Matrix;

namespace Layer
{
	template <typename DType>
	class Layer;

	template <typename DType>
	class FinalizedLayer;
}

namespace Optimizer
{
	template <typename DType>
	class SGD
	{
	public:
		SGD(std::vector<Layer::FinalizedLayer<DType>> layers);

		void forward();
		void backward();

		SGD& operator<<(Config::Target<DType>&& target)
		{
			_target = target;
			return *this;
		}

		SGD& operator<<(Config::LearningRate&& lr)
		{
			_learning_rate = lr;
			return *this;
		}

	private:
		Config::Target<DType> _target;
		Config::LearningRate _learning_rate;
		DType _momentum;

	    std::vector<Layer::FinalizedLayer<DType>> _layers;
	    std::vector<Layer::FinalizedLayer<DType>> _orderedLayers;
	    std::vector<Layer::FinalizedLayer<DType>> _lastLayers;

	    int _k;
	};
}

namespace Float
{
	using SGD = Optimizer::SGD<float>;
}

namespace Double
{
	using SGD = Optimizer::SGD<double>;
}
