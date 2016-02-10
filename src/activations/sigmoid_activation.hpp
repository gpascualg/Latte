#pragma once

#include "activation.hpp"

template <typename DType>
class SigmoidActivation : public Activation<DType>
{
private:
	inline void apply(Matrix<DType>* matrix, Matrix<DType>* dest) override;
	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) override;
};

template <typename DType>
void SigmoidActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest);

template <typename DType>
void SigmoidActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha);
