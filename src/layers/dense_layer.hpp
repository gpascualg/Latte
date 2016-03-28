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
			*this << Config::Filler<DType> { FromFactory(Filler, RandomFiller, DType)() };
			*this << Config::Activation<DType> { FromFactory(Activation, ReluActivation, DType)() };
		}
	};
}


REGISTER_LAYER(DenseLayer)
