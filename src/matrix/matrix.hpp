#pragma once

#include <iostream>
#include <queue>
#include <functional>
#include <cmath>

#include <cblas.h>

// This should be forward declared and methods declared on a cpp
#include "matrix_factory.hpp"
#include "common.hpp"



#if !defined(max) && defined(_WIN32)
	#define max(a,b)            (((a) > (b)) ? (a) : (b))
	#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif



template <typename DType>
inline Matrix<DType>* transpose(Matrix<DType>* A, Matrix<DType>* B) {
	if (!B)
	{
		B = MatrixFactory<DType>::get()->pop(A->shape().T());
	}

	Matrix<DType>& O = *A;
	Matrix<DType>& D = *B;
	int m = O.shape().m;
	int n = O.shape().n;

	int i, j;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			D[j*m + i] = O[i*n + j];
		}
	}

	return B;
}

template <typename DType = float>
class Matrix
{
public:
	explicit Matrix(int m, int n);
	explicit Matrix(int m, int n, DType value);
	explicit Matrix(int m, int n, DType* data);
	explicit Matrix(Shape shape);
	explicit Matrix(Shape shape, DType value);
	explicit Matrix(Shape shape, Matrix<DType>* data, bool copy = false);

	virtual ~Matrix();

	// Shape
	inline int size() const;
	inline void reshape(Shape shape);
	inline Shape shape();
	inline Matrix<DType>* T();

	// Indexing
	inline DType& operator()(int x, int y) { return _data[(x * _n) + y];	};
	inline DType& operator[](int idx) { return _data[idx]; }

	// Sums
	inline void operator+=(DType value);
	inline void operator+=(Matrix<DType>& value);
	inline void operator-=(DType value);
	inline void operator-=(Matrix<DType>& value);
	DType sum();

	// Mult
	inline void operator*=(DType value);
	inline void operator*=(Matrix<DType>& value);
	inline void operator/=(DType value);
	inline void operator/=(Matrix<DType>& value);

	void sum(Matrix<DType>* other, DType alpha = DType(1.0), DType beta = DType(1.0));
	void pdiv(DType val) { pdiv(val, this); }
	void pdiv(DType val, Matrix<DType>* result);
	void mul(Matrix<DType>* other, Matrix<DType>* result, DType alpha = DType(1.0), DType beta = DType(0.0));
	DType dot(Matrix<DType>* other, Matrix<DType>* result);

	void exp(DType alpha = DType(1.0)) { exp(this, alpha); }
	void exp(Matrix<DType>* other, DType alpha = DType(1.0));

private:
	DType* _data;
	int _data_size;
	int _m;
	int _n;

	Matrix<DType>* _transposed_matrix;
	CBLAS_TRANSPOSE _transpose;
	bool _tranpose_pending;
};


// Constructors
template <typename DType>
Matrix<DType>::Matrix(int m, int n) :
	_data((DType*)malloc(m * n * sizeof(DType))),
	_data_size(m * n),
	_m(m),
	_n(n),
	_transposed_matrix(nullptr),
	_transpose(CblasNoTrans),
	_tranpose_pending(false)
{
	memset(_data, 0, _m * _n * sizeof(DType));
}

template <typename DType>
Matrix<DType>::Matrix(int m, int n, DType value) :
	Matrix(m, n)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] = value;
	}
}

template <typename DType>
Matrix<DType>::Matrix(int m, int n, DType* data) :
	_data(data),
	_data_size(m * n),
	_m(m),
	_n(n),
	_transposed_matrix(nullptr),
	_transpose(CblasNoTrans),
	_tranpose_pending(false)
{}

template <typename DType>
Matrix<DType>::Matrix(Shape shape) :
	Matrix(shape.m, shape.n)
{
	memset(_data, 0, _m * _n * sizeof(DType));
}

template <typename DType>
Matrix<DType>::Matrix(Shape shape, DType value) :
	Matrix(shape.m, shape.n, value)
{}

template <typename DType>
Matrix<DType>::Matrix(Shape shape, Matrix<DType>* data, bool copy) :
	Matrix(shape.m, shape.n, nullptr)
{
	if (copy)
	{
		_data_size = _m * _n;
		int size = _data_size * sizeof(DType);
		_data = (DType*)malloc(size);
		memcpy(_data, data->_data, size);
	}
	else
	{
		_data = data->_data;
	}
}

template <typename DType>
Matrix<DType>::~Matrix()
{ 
	if (_data)
	{
		free(_data);
	}

	if (_transposed_matrix)
	{
		free(_transposed_matrix);
	}
}


// Shape
template <typename DType>
int Matrix<DType>::size() const
{ 
	return _data_size;
}

template <typename DType>
void Matrix<DType>::reshape(Shape shape) 
{ 
	_m = shape.m; 
	_n = shape.n;
}

template <typename DType>
Shape Matrix<DType>::shape()
{ 
	return{ _m, _n }; 
}

template <typename DType>
Matrix<DType>* Matrix<DType>::T()
{
	if (!_transposed_matrix)
	{
		_transposed_matrix = new Matrix<DType>(_n, _m, _data);
	}

	// Reset data, in case the matrix is reused from the pool
	_transposed_matrix->_data = _data;
	_transposed_matrix->_m = _n;
	_transposed_matrix->_n = _m;
	_transposed_matrix->_tranpose_pending = true;
	_transposed_matrix->_transpose = CblasTrans;
	return _transposed_matrix;
}


// Sum operators
template <typename DType>
void Matrix<DType>::operator+=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] += value;
	}
}

template <> void Matrix<float>::operator+=(Matrix<float>& other);
template <> void Matrix<double>::operator+=(Matrix<double>& other);

template <typename DType>
void Matrix<DType>::operator-=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] -= value;
	}
}

template <> void Matrix<float>::operator-=(Matrix<float>& other);
template <> void Matrix<double>::operator-=(Matrix<double>& other);


// Mult operators
template <>
inline void Matrix<float>::operator*=(float value)
{
	cblas_sscal(shape().prod(), value, _data, 1);
}

template <>
inline void Matrix<double>::operator*=(double value)
{
	cblas_dscal(shape().prod(), value, _data, 1);
}

template <typename DType>
void Matrix<DType>::operator*=(Matrix<DType>& other)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] *= other._data[i];
	}
}

template <typename DType>
void Matrix<DType>::operator/=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] /= value;
	}
}

template <typename DType>
void Matrix<DType>::operator/=(Matrix<DType>& other)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] /= other._data[i];
	}
}


// Global operators (scalar - Matrix)
template <typename DType>
inline Matrix<DType>* operator*(DType value, Matrix<DType>& self)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = value * self[i];
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator/(DType value, Matrix<DType>& self)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = value / self[i];
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator+(DType value, Matrix<DType>& self)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = value + self[i];
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator-(DType value, Matrix<DType>& self)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = value - self[i];
	}

	return result;
}


// Global operators (scalar - Matrix)
template <typename DType>
inline Matrix<DType>* operator*(Matrix<DType>& self, DType value)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = self[i] * value;
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator/(Matrix<DType>& self, DType value)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = self[i] / value;
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator+(Matrix<DType>& self, DType value)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = self[i] + value;
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator-(Matrix<DType>& self, DType value)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = self[i] - value;
	}

	return result;
}


// Global operators (Matrix - Matrix)
/*
template <typename DType>
inline Matrix<DType>* operator*(Matrix<DType> self, DType value)
{
	Matrix<DType>* result = new Matrix<DType>(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = (*self)[i] * value;
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator/(Matrix<DType> self, DType value)
{
	Matrix<DType>* result = new Matrix<DType>(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = (*self)[i] / value;
	}

	return result;
}

template <typename DType>
inline Matrix<DType>* operator+(Matrix<DType> self, DType value)
{
	Matrix<DType>* result = new Matrix<DType>(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = (*self)[i] + value;
	}

	return result;
}
*/
template <typename DType>
inline Matrix<DType>* operator-(Matrix<DType>& self, Matrix<DType>& other)
{
	Matrix<DType>* result = MatrixFactory<DType>::get()->pop(self.shape());

	for (int i = 0; i < self.shape().prod(); ++i)
	{
		(*result)[i] = self[i] - other[i];
	}

	return result;
}

template <typename DType>
inline bool operator>(const Matrix<DType>& lhs, const Matrix<DType>& rhs) { return lhs.shape().prod() > rhs.shape().prod(); }

template <typename DType>
inline bool operator<(const Matrix<DType>& lhs, const Matrix<DType>& rhs) { return lhs.shape().prod() < rhs.shape().prod(); }

// Methods
template <typename DType>
void Matrix<DType>::pdiv(DType value, Matrix<DType>* result)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] = value / _data[i];
	}
}

