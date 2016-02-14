#include "relu_activation.hpp"
#include "matrix/matrix.hpp"


template <typename DType>
void ReluActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest)
{
	// max(0, x)
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = std::max(DType(0.0), (*matrix)[i]);
	}
}

template <typename DType>
void ReluActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha)
{
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = (*alpha)[i] * ((*matrix)[i] > 0);
	}
}


// Specialization
template void ReluActivation<float>::apply(Matrix<float>* matrix, Matrix<float>* dest);
template void ReluActivation<double>::apply(Matrix<double>* matrix, Matrix<double>* dest);

template void ReluActivation<float>::derivative(Matrix<float>* matrix, Matrix<float>* dest, Matrix<float>* alpha);
template void ReluActivation<double>::derivative(Matrix<double>* matrix, Matrix<double>* dest, Matrix<double>* alpha);
