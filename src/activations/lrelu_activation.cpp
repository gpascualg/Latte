#include "lrelu_activation.hpp"
#include "matrix/matrix.hpp"


template <typename DType>
LeakyReluActivation<DType>::LeakyReluActivation() :
	LeakyReluActivation(DType(0.01))
{}

template <typename DType>
LeakyReluActivation<DType>::LeakyReluActivation(DType leak) :
	Activation(),
	_leak(leak)
{}

template <typename DType>
void LeakyReluActivation<DType>::apply(Matrix<DType>* matrix, Matrix<DType>* dest)
{
	// max(0, x)
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = max(0, (*matrix)[i]) + min((*matrix)[i], 0) * _leak;
	}
}

template <typename DType>
void LeakyReluActivation<DType>::derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha)
{
	for (int i = 0; i < dest->shape().prod(); ++i)
	{
		(*dest)[i] = (*alpha)[i] * (((*matrix)[i] > 0) + _leak * ((*matrix)[i] <= 0));
	}
}


// Specialization
template LeakyReluActivation<float>::LeakyReluActivation();
template LeakyReluActivation<double>::LeakyReluActivation();

template LeakyReluActivation<float>::LeakyReluActivation(float leak);
template LeakyReluActivation<double>::LeakyReluActivation(double leak);

template void LeakyReluActivation<float>::apply(Matrix<float>* matrix, Matrix<float>* dest);
template void LeakyReluActivation<double>::apply(Matrix<double>* matrix, Matrix<double>* dest);

template void LeakyReluActivation<float>::derivative(Matrix<float>* matrix, Matrix<float>* dest, Matrix<float>* alpha);
template void LeakyReluActivation<double>::derivative(Matrix<double>* matrix, Matrix<double>* dest, Matrix<double>* alpha);
