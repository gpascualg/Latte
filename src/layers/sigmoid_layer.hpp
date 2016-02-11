#pragma once

#include "layer.hpp"


template <typename DType>
class Filler;

template <typename DType>
class SigmoidLayer : public Layer<DType>
{
public:
	SigmoidLayer(Shape shape, int num_output);
	SigmoidLayer(Shape shape, int num_output, Filler<DType>* filler);
};
