#pragma once

#include "layer.hpp"
#include "utils/layer_config.hpp"

// TODO: Can this be somehow forward declared?
#include "fillers/filler.hpp"
#include "fillers/random_filler.hpp"


template <typename DType>
class Filler;

template <typename DType>
class SigmoidLayer : public Layer<DType>
{
public:
	SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* filler, 
        GenericParameter* dropout_ratio, GenericParameter* bias);
	template <typename... Args> SigmoidLayer(NamedArguments_t, Args... args) :
		SigmoidLayer{ 
            ARG_REQUIRED(shape), 
            ARG_REQUIRED(num_output), 
            ARG_OPTIONAL(filler, FromFactory(Filler, RandomFiller, DType)()),
            ARG_OPTIONAL(dropout_ratio, DType(0.0)),
            ARG_OPTIONAL(bias, DefaultBias<DType>())
        }
	{}
};
