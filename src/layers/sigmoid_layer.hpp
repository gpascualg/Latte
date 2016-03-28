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
	        *this << Config::Filler<DType> { FromFactory(Filler, RandomFiller, DType)() };
	        *this << Config::Activation<DType> { FromFactory(Activation, SigmoidActivation, DType)() };
	    }
	};
}


REGISTER_LAYER(SigmoidLayer)