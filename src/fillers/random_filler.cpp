#include "random_filler.hpp"
#include "matrix/matrix.hpp"
#include "utils/rng.hpp"

template <typename DType>
void RandomFiller<DType>::fill(Matrix<DType>* weights)
{
	for (int i = 0; i < weights->shape().prod(); ++i)
	{
		(*weights)[i] = rng()->nextFloat() - DType(0.5);
	}
}


// Specialization
template void RandomFiller<float>::fill(Matrix<float>* weights);
template void RandomFiller<double>::fill(Matrix<double>* weights);