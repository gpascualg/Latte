#include "sgd_config.hpp"

template <> Parameter<Matrix<float>*> SGDConfig<float>::data("data");
template <> Parameter<Matrix<double>*> SGDConfig<double>::data("data");

template <> Parameter<Matrix<float>*> SGDConfig<float>::target("target");
template <> Parameter<Matrix<double>*> SGDConfig<double>::target("target");

template <> Parameter<float> SGDConfig<float>::learning_rate("learning_rate");
template <> Parameter<double> SGDConfig<double>::learning_rate("learning_rate");

template <> Parameter<float> SGDConfig<float>::momentum("momentum");
template <> Parameter<double> SGDConfig<double>::momentum("momentum");
