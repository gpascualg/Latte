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

Parameter<Matrix<float>*> SGDConfig<float>::data("data");
Parameter<Matrix<double>*> SGDConfig<double>::data("data");

Parameter<Matrix<float>*> SGDConfig<float>::target("target");
Parameter<Matrix<double>*> SGDConfig<double>::target("target");

Parameter<float> SGDConfig<float>::learning_rate("learning_rate");
Parameter<double> SGDConfig<double>::learning_rate("learning_rate");

Parameter<float> SGDConfig<float>::momentum("momentum");
Parameter<double> SGDConfig<double>::momentum("momentum");
