#include "matrix.hpp"

// Methods
void Matrix<float>::sum(Matrix<float>* other, float alpha, float beta)
{
	cblas_saxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
}

void Matrix<double>::sum(Matrix<double>* other, double alpha, double beta)
{
	cblas_daxpby(shape().prod(), alpha, _data, 1, beta, other->_data, 1);
}

float Matrix<float>::dot(Matrix<float>* other, Matrix<float>* result)
{
	return cblas_sdot(_m, _data, 1, other->_data, 1);
}

double Matrix<double>::dot(Matrix<double>* other, Matrix<double>* result)
{
	return cblas_ddot(_m, _data, 1, other->_data, 1);
}

void Matrix<float>::exp(Matrix<float>* other, float alpha)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		other->_data[i] = expf(alpha*_data[i]);
	}
}

void Matrix<double>::exp(Matrix<double>* other, double alpha)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		other->_data[i] = std::exp(alpha*_data[i]);
	}
}
