#pragma once

#include "layer.hpp"
#include "utils/layer_config.hpp"

// TODO: Can this be somehow forward declared?
#include "fillers/filler.hpp"
#include "fillers/random_filler.hpp"
#include "activations/relu_activation.hpp"


template <typename DType>
class Activation;

template <typename DType>
class Filler;

template <typename DType>
class DenseLayer : public Layer<DType>
{
public:
	DenseLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* activation,
		GenericParameter* filler, GenericParameter* dropout_ratio, GenericParameter* bias);
	template <typename... Args> DenseLayer(NamedArguments_t, Args... args) :
		DenseLayer{
            ARG_REQUIRED(shape), 
			ARG_REQUIRED(num_output),
			ARG_OPTIONAL(activation, FromFactory(Activation, ReluActivation, DType)()), 
            ARG_OPTIONAL(filler, FromFactory(Filler, RandomFiller, DType)()),
            ARG_OPTIONAL(dropout_ratio, DType(0.0)),
			ARG_OPTIONAL(bias, DefaultBias<DType>())
        }
	{}
};
