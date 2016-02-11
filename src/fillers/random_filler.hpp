#pragma once

#include "filler.hpp"

template <typename DType>
class Matrix;

template <typename DType>
class RandomFiller : public Filler<DType>
{
public:
	void fill(Matrix<DType>* weights) override;
};
