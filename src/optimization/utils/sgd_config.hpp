#pragma once 

#include "magic/named_parameters.hpp"


template <typename DType>
class Matrix;

template <typename DType>
struct SGDConfig
{
	static Parameter<Matrix<DType>*> data;
	static Parameter<Matrix<DType>*> target;
	static Parameter<DType> learning_rate;
	static Parameter<DType> momentum;
};

template <> Parameter<Matrix<float>*> SGDConfig<float>::data;
template <> Parameter<Matrix<double>*> SGDConfig<double>::data;

template <> Parameter<Matrix<float>*> SGDConfig<float>::target;
template <> Parameter<Matrix<double>*> SGDConfig<double>::target;

template <> Parameter<float> SGDConfig<float>::learning_rate;
template <> Parameter<double> SGDConfig<double>::learning_rate;

template <> Parameter<float> SGDConfig<float>::momentum;
template <> Parameter<double> SGDConfig<double>::momentum;
