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
	SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* filler);
	template <typename... Args> SigmoidLayer(Args... args) :
		SigmoidLayer{ ARG_REQUIRED(shape), ARG_REQUIRED(num_output), ARG_OPTIONAL(filler, Filler<DType>::get<RandomFiller<DType>>()) }
	{}
};
