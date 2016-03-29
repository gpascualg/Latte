#pragma once

#include "layer.hpp"

// TODO: Can this be somehow forward declared?
#include "activations/sigmoid_activation.hpp"
#include "fillers/filler.hpp"
#include "fillers/random_filler.hpp"


namespace Layer
{
	template <typename DType>
	class SigmoidLayer : public Layer<DType>
	{
	public:
	    SigmoidLayer() :
	        Layer<DType>()
	    {
	        *this << ExtConfig::Filler<DType> { FromFillerFactory<DType, RandomFiller>() };
			*this << ExtConfig::Activation<DType> { FromActivationFactory<DType, SigmoidActivation>() };
	    }
	};
}


REGISTER_LAYER(SigmoidLayer)