#pragma once

#include "activation.hpp"

template <typename DType>
class LeakyReluActivation : public Activation<DType>
{
public:
	LeakyReluActivation();
	LeakyReluActivation(DType leak);

public:
    static std::string FactoryName() { return "LeakyReLU"; }

private:
	void apply(Matrix<DType>* matrix, Matrix<DType>* dest) override;
	void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) override;

private:
	DType _leak;
};
