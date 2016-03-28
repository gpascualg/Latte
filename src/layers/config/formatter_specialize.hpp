#include "layers/config/formatter.hpp"
#include "fillers/random_filler.hpp"
#include "common.hpp"


struct Shape;

template <typename DType>
class Activation;

template <typename DType>
class Filler;
 

namespace Config
{
	class Finalizer {};

	TEMPLATED_FORMATTER(NumOutput, int);
	TEMPLATED_FORMATTER(Dropout, float);
	TEMPLATED_FORMATTER(Shape, ::Shape);
	EXTENDED_FORMATTER(Bias, ::Bias);
	EXTENDED_FORMATTER_PTR(Filler, ::Filler);
	EXTENDED_FORMATTER_PTR(Activation, ::Activation);
}

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