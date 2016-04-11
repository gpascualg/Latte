#pragma once

#include "layers/loss.hpp"


namespace Layer
{
	template <typename DType>
	class EuclideanLoss : public Loss<DType>
	{
	public:
		EuclideanLoss() :
			Loss<DType>()
		{}

		virtual Matrix<DType>* forward() override;
		virtual std::vector<BackwardConnection<DType>*> backward() override;
	};
}

REGISTER_LOSS(EuclideanLoss)
