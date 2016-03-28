#pragma once

#include "layer.hpp"

// TODO: Can this be somehow forward declared?
#include "fillers/filler.hpp"
#include "fillers/random_filler.hpp"
#include "activations/relu_activation.hpp"


template <typename DType>
class Activation;

template <typename DType>
class Filler;

namespace Layer
{
	template <typename DType>
	class DenseLayer : public Layer<DType>
	{
	public:
		DenseLayer() :
			Layer<DType>()
		{
			*this << Config::Filler<DType> { FromFactory(Filler, RandomFiller, DType)() };
			*this << Config::Activation<DType> { FromFactory(Activation, ReluActivation, DType)() };
		}
	};
}


namespace Float
{
	inline ::Layer::LayerWrapper<float> DenseLayer() { return ::Layer::LayerWrapper<float>(new ::Layer::DenseLayer<float>()); }
}

namespace Double
{
	inline ::Layer::LayerWrapper<double> DenseLayer() { return ::Layer::LayerWrapper<double>(new ::Layer::DenseLayer<double>()); }
}
