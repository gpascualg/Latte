#pragma once

#include "layer.hpp"

// TODO: Can this be somehow forward declared?
#include "fillers/filler.hpp"
#include "fillers/random_filler.hpp"
#include "activations/relu_activation.hpp"


namespace Layer
{
	template <typename DType>
	class DenseLayer : public Layer<DType>
	{
	public:
		DenseLayer() :
			Layer<DType>()
		{
			*this << ExtConfig::Filler<DType> { FromFillerFactory<DType, RandomFiller>() };
			*this << ExtConfig::Activation<DType> { FromActivationFactory<DType, ReluActivation>() };
		}
	};
}


REGISTER_LAYER(DenseLayer)
