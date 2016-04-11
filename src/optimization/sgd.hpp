#pragma once

#include <vector>
#include <iostream>

#include "layers/config/formatter_specialize.hpp"
#include "matrix/matrix.hpp"


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

		void optimize();
		
		SGD& operator<<(ExtConfig::Iterations&& iters)
		{
			_iterations = iters;
			return *this;
		}

		SGD& operator<<(ExtConfig::LearningRate&& lr)
		{
			_learning_rate = lr;
			return *this;
		}

	private:
		void forward();
		void backward();

	private:
		ExtConfig::Iterations _iterations;
		ExtConfig::LearningRate _learning_rate;
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
