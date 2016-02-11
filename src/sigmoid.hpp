#pragma once

#include "layer.hpp"


template <typename DType>
class Filler;

template <typename DType>
class Sigmoid : public Layer<DType>
{
public:
	Sigmoid(Shape shape, int num_output);
	Sigmoid(Shape shape, int num_output, Filler<DType>* filler);
};
