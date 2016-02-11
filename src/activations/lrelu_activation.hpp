#pragma once

#include "activation.hpp"

template <typename DType>
class LeakyReluActivation : public Activation<DType>
{
public:
	LeakyReluActivation();
	LeakyReluActivation(DType leak);

private:
	inline void apply(Matrix<DType>* matrix, Matrix<DType>* dest) override;
	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) override;

private:
	DType _leak;
};

template <typename DType>
void LeakyReluActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest);

template <typename DType>
void LeakyReluActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha);
