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
	        *this << Config::Filler<DType>(FromFactory(Filler, RandomFiller, DType)());
	        *this << Config::Activation<DType> { FromFactory(Activation, SigmoidActivation, DType)() };
	    }
	};
}


namespace Float
{
	inline ::Layer::SigmoidLayer<float>& SigmoidLayer() { return *(new ::Layer::SigmoidLayer<float>()); }
}

namespace Double
{
	inline ::Layer::SigmoidLayer<double>& SigmoidLayer() { return *(new ::Layer::SigmoidLayer<double>()); }
}

