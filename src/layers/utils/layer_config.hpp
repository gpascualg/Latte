#pragma once

#include "magic/named_parameters.hpp"

// This should be forward declared and methods declared on a cpp
#include "common.hpp"

template <typename DType>
class Activation;

template <typename DType>
class Filler;

template <typename DType>
struct LayerConfig
{
	static Parameter<Shape> shape;
	static Parameter<int> num_output;
	static Parameter<Activation<DType>*> activation;
	static Parameter<Filler<DType>*> filler;
};

template <> Parameter<Shape> LayerConfig<float>::shape;
template <> Parameter<Shape> LayerConfig<double>::shape;

template <> Parameter<int> LayerConfig<float>::num_output;
template <> Parameter<int> LayerConfig<double>::num_output;

template <> Parameter<Activation<float>*> LayerConfig<float>::activation;
template <> Parameter<Activation<double>*> LayerConfig<double>::activation;

template <> Parameter<Filler<float>*> LayerConfig<float>::filler;
template <> Parameter<Filler<double>*> LayerConfig<double>::filler;
