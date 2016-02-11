#pragma once

#include "activation.hpp"

template <typename DType>
class LeakedReluActivation : public Activation<DType>
{
public:
	LeakedReluActivation();
	LeakedReluActivation(DType leak);

private:
	inline void apply(Matrix<DType>* matrix, Matrix<DType>* dest) override;
	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) override;

private:
	DType _leak;
};

template <typename DType>
void LeakedReluActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest);

template <typename DType>
void LeakedReluActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha);
