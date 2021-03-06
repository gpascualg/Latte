#include "layers/loss.hpp"

namespace Layer
{
	template <typename DType>
	Loss<DType>::Loss():
		Layer<DType>()
	{
		*this << ExtConfig::NumOutput(1);
	}

	SPECIALIZE(Loss);
}
