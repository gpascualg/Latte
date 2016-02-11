#include "lrelu_activation.hpp"
#include "matrix.hpp"


template <typename DType>
LeakedReluActivation<DType>::LeakedReluActivation() :
	LeakedReluActivation(DType(0.01))
{}

template <typename DType>
LeakedReluActivation<DType>::LeakedReluActivation(DType leak):
	Activation(),
	_leak(leak)
{}

template <typename DType>
void LeakedReluActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest)
{
	// max(0, x)
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = max(0, (*matrix)[i]) + min((*matrix)[i], 0) * _leak;
	}
}

template <typename DType>
void LeakedReluActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha)
{
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = (*alpha)[i] * (((*matrix)[i] > 0) + _leak * ((*matrix)[i] <= 0));
	}
}


// Specialization
template LeakedReluActivation<float>::LeakedReluActivation();
template LeakedReluActivation<double>::LeakedReluActivation();

template LeakedReluActivation<float>::LeakedReluActivation(float leak);
template LeakedReluActivation<double>::LeakedReluActivation(double leak);

template void LeakedReluActivation<float>::apply(Matrix<float>* matrix, Matrix<float>* dest);
template void LeakedReluActivation<double>::apply(Matrix<double>* matrix, Matrix<double>* dest);

template void LeakedReluActivation<float>::derivative(Matrix<float>* matrix, Matrix<float>* dest, Matrix<float>* alpha);
template void LeakedReluActivation<double>::derivative(Matrix<double>* matrix, Matrix<double>* dest, Matrix<double>* alpha);
