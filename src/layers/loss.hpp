#pragma once

#include "layers/layer.hpp"

namespace Layer
{
	template <typename DType>
	class Loss : public Layer<DType>
	{
	public:
		Loss();

		// Special for loss
		virtual bool isLast() override { return true; }

		// Must be reimplemented
		virtual Matrix<DType>* forward() override = 0;
		virtual std::vector<BackwardConnection<DType>*> backward() override = 0;
		virtual void update(float learningRate) override {};

		Loss<DType>& operator<<(ExtConfig::Target<DType>&& target)
		{
			_target = target;
			return *this;
		}

		template <typename T>
		Loss<DType>& operator<<(T&& target)
		{
			Layer<DType>::operator<<(std::move(target));
			return *this;
		}

		FinalizedLayer<DType> operator<<(ExtConfig::Finalizer&& f)
		{
			LATTE_ASSERT("Layer not ready, all should be 1:" <<
				std::endl << "\tTarget: " << _target.isSet(),
				_target.isSet());

			return FinalizedLayer<DType>(this);
		}

		//virtual Layer<DType>& operator<<(Layer<DType>& other) override = 0;

		// Disable some configs
		FinalizedLayer<DType> operator<<(ExtConfig::NumOutput) { NOT_IMPLEMENTED("Loss layers can't have NumOutput"); }
		FinalizedLayer<DType> operator<<(ExtConfig::Filler<DType>) { NOT_IMPLEMENTED("Loss layers can't have Filler"); }
		FinalizedLayer<DType> operator<<(ExtConfig::Activation<DType>) { NOT_IMPLEMENTED("Loss layers can't have Activation"); }

	protected:
		ExtConfig::Target<DType> _target;
		Matrix<DType>* _loss;
	};
}
