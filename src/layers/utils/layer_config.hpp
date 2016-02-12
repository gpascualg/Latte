#pragma once

#include "magic/named_parameters.hpp"

// This should be forward declared and methods declared on a cpp
#include "common.hpp"

template <typename DType>
struct LayerConfig
{
	static Parameter<Shape> shape;
	static Parameter<int> num_output;
	static Parameter<Activation<DType>*> activation;
	static Parameter<Filler<DType>*> filler;
};

Parameter<Shape> LayerConfig<float>::shape("shape");
Parameter<Shape> LayerConfig<double>::shape("shape");

Parameter<int> LayerConfig<float>::num_output("num_output");
Parameter<int> LayerConfig<double>::num_output("num_output");

Parameter<Activation<float>*> LayerConfig<float>::activation("activation");
Parameter<Activation<double>*> LayerConfig<double>::activation("activation");

Parameter<Filler<float>*> LayerConfig<float>::filler("filler");
Parameter<Filler<double>*> LayerConfig<double>::filler("filler");
