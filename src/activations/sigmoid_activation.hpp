#pragma once

#include "activation.hpp"

template <typename DType>
class SigmoidActivation : public Activation<DType>
{
public:
    static std::string FactoryName() { return "Sigmoid"; }

private:
	void apply(Matrix<DType>* matrix, Matrix<DType>* dest) override;
	void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) override;
};
