#include "sigmoid_activation.hpp"
#include "matrix/matrix.hpp"

template <typename DType>
void SigmoidActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest)
{
	// 1 / (1 + exp(-_output))
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = DType(1.0) / (DType(1.0) + exp(-(*matrix)[i]));
	}
}

template <typename DType>
void SigmoidActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha)
{
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = (DType(1.0) - (*matrix)[i]) * (*matrix)[i] * (*alpha)[i];
	}
}


// Specialization
template void SigmoidActivation<float>::apply(Matrix<float>* matrix, Matrix<float>* dest);
template void SigmoidActivation<double>::apply(Matrix<double>* matrix, Matrix<double>* dest);

template void SigmoidActivation<float>::derivative(Matrix<float>* matrix, Matrix<float>* dest, Matrix<float>* alpha);
template void SigmoidActivation<double>::derivative(Matrix<double>* matrix, Matrix<double>* dest, Matrix<double>* alpha);
