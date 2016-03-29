#pragma once

#include "layers/config/formatter.hpp"
#include "fillers/random_filler.hpp"
#include "common.hpp"


struct Shape;

template <typename DType>
class Activation;

template <typename DType>
class Filler;
 

namespace ExtConfig
{
	class Finalizer { public: Finalizer(){} };
}
namespace Float { namespace Config { inline ::ExtConfig::Finalizer Finalizer() { return ::ExtConfig::Finalizer(); } } }
namespace Double { namespace Config { inline ::ExtConfig::Finalizer Finalizer() { return ::ExtConfig::Finalizer(); } } }

// Layer specific config
TEMPLATED_FORMATTER(NumOutput, int);
TEMPLATED_FORMATTER(Dropout, float);
TEMPLATED_FORMATTER(Shape, ::Shape);
EXTENDED_FORMATTER(Bias, ::Bias);
EXTENDED_FORMATTER_PTR(Filler, ::Filler);
EXTENDED_FORMATTER_PTR(Activation, ::Activation);
EXTENDED_FORMATTER_PTR(Data, ::Matrix);

// Optimization config
TEMPLATED_FORMATTER(LearningRate, float);
TEMPLATED_FORMATTER(Momentum, float);
TEMPLATED_FORMATTER(Iterations, ::Iterations);
EXTENDED_FORMATTER_PTR(Target, ::Matrix);


namespace Float
{
	inline Bias<float> NoBias()
	{
		return Bias<float> { 0, nullptr };
	}

	inline Bias<float> DefaultBias()
	{
		return Bias<float> { 1.0f, FromFactory(Filler, RandomFiller, float)() };
	}
}

namespace Double
{
	inline Bias<double> NoBias()
	{
		return Bias<double> { 0, nullptr };
	}

	inline Bias<double> DefaultBias()
	{
		return Bias<double> { 1.0, FromFactory(Filler, RandomFiller, double)() };
	}
}
