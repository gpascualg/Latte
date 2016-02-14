#pragma once

#include <vector>

#include "utils/sgd_config.hpp"

template <typename DType>
class Matrix;

template <typename DType>
class Layer;

template <typename DType>
class SGD
{
public:
	SGD(GenericParameter* data, GenericParameter* target, GenericParameter* learning_rate, GenericParameter* momentum);
	template <typename... Args> SGD(NamedArguments_t, Args... args):
		SGD{ ARG_REQUIRED(data), ARG_REQUIRED(target), ARG_OPTIONAL(learning_rate, DType(0.01)), ARG_OPTIONAL(momentum, DType(0.0)) }
	{}
	
	template <typename LType>
	void stack(int num_output);
	void stack(Layer<DType>* layer);

	void forward();
	void backward();

private:
	Matrix<DType>* _data;
	Matrix<DType>* _target;
	DType _learning_rate;
	DType _momentum;
    
	Layer<DType>* _firstLayer;
	Layer<DType>* _lastLayer;
    int _k;
};
