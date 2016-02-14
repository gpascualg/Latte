#include "matrix.hpp"

#include <cmath>



template <> void Matrix<float>::operator+=(Matrix<float>& other)
{
	cblas_saxpy(shape().prod(), 1.0, other._data, 1, _data, 1);
}

template <> void Matrix<double>::operator+=(Matrix<double>& other)
{
	cblas_daxpy(shape().prod(), 1.0, other._data, 1, _data, 1);
}

template <> void Matrix<float>::operator-=(Matrix<float>& other)
{
	cblas_saxpy(shape().prod(), -1.0, other._data, 1, _data, 1);
}

template <> void Matrix<double>::operator-=(Matrix<double>& other)
{
	cblas_daxpy(shape().prod(), -1.0, other._data, 1, _data, 1);
}

template <> void Matrix<float>::operator*=(float value)
{
	cblas_sscal(shape().prod(), value, _data, 1);
}

template <> void Matrix<double>::operator*=(double value)
{
	cblas_dscal(shape().prod(), value, _data, 1);
}

template <> void Matrix<float>::mul(Matrix<float>* other, Matrix<float>* result, float alpha, float beta)
{
	int lda = _transpose == CblasNoTrans ? _n : _m;
	int ldb = other->_transpose == CblasNoTrans ? other->_n : _n;

	cblas_sgemm(CblasRowMajor, _transpose, other->_transpose, _m, other->_n, _n, alpha, _data, lda, other->_data, ldb, beta, result->_data, other->_n);
}

template <> void Matrix<double>::mul(Matrix<double>* other, Matrix<double>* result, double alpha, double beta)
{
	int lda = _transpose == CblasNoTrans ? _n : _m;
	int ldb = other->_transpose == CblasNoTrans ? other->_n : _n;

	cblas_dgemm(CblasRowMajor, _transpose, other->_transpose, _m, other->_n, _n, alpha, _data, lda, other->_data, ldb, beta, result->_data, other->_n);
}

template <> float Matrix<float>::sum()
{ 
	return cblas_sasum(shape().prod(), _data, 1); 
}

template <> double Matrix<double>::sum()
{
	return cblas_dasum(shape().prod(), _data, 1);
}

// Methods
template <> void Matrix<float>::sum(Matrix<float>* other, float alpha, float beta)
{
	cblas_saxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
}

template <> void Matrix<double>::sum(Matrix<double>* other, double alpha, double beta)
{
	cblas_daxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
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
