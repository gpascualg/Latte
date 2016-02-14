#pragma once

#include "filler.hpp"

template <typename DType>
class Matrix;

template <typename DType>
class RandomFiller : public Filler<DType>
{
public:
    static std::string FactoryName() { return "Random"; }
	void fill(Matrix<DType>* weights) override;
};
