#include "matrix.hpp"

// Methods
template <> void Matrix<float>::sum(Matrix<float>* other, float alpha, float beta)
{
	cblas_saxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
}

template <> void Matrix<double>::sum(Matrix<double>* other, double alpha, double beta)
{
	cblas_daxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
}

template <> void Matrix<float>::mul(Matrix<float>* other, Matrix<float>* result)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _m, other->_n, _n, 1.0f, _data, _n, other->_data, other->_n, 0.0f, result->_data, other->_n);
}

template <> void Matrix<double>::mul(Matrix<double>* other, Matrix<double>* result)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _m, other->_n, _n, 1.0, _data, _n, other->_data, other->_n, 0.0, result->_data, other->_n);
}

template <> float Matrix<float>::dot(Matrix<float>* other, Matrix<float>* result)
{
	return cblas_sdot(_m, _data, 1, other->_data, 1);
}

template <> double Matrix<double>::dot(Matrix<double>* other, Matrix<double>* result)
{
	return cblas_ddot(_m, _data, 1, other->_data, 1);
}

template <> void Matrix<float>::exp(Matrix<float>* other, float alpha)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		other->_data[i] = expf(alpha*_data[i]);
	}
}

template <> void Matrix<double>::exp(Matrix<double>* other, double alpha)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		other->_data[i] = std::exp(alpha*_data[i]);
	}
}