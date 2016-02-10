#pragma once

#include <iostream>
#include <queue>
#include <functional>

#include <cblas.h>

// This should be forward declared and methods declared on a cpp
#include "matrix_factory.hpp"
#include "common.h"


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

	virtual ~Matrix() { if (_data) free(_data); }

	inline int size() const { return _data_size; }
	inline void reshape(Shape shape) { _m = shape.m; _n = shape.n; }
	inline Shape shape() { return{ _m, _n }; }
	inline Matrix<DType>* T() { return transpose<DType>(this, NULL); }

	// Indexing
	inline DType& operator()(int x, int y) { return _data[(x * _n) + y];	};
	inline DType& operator[](int idx) { return _data[idx]; }

	// Sums
	inline void operator+=(DType value);
	inline void operator+=(Matrix<DType>& value);
	inline void operator-=(DType value);
	inline void operator-=(Matrix<DType>& value);

	// Mult
	inline void operator*=(DType value);
	inline void operator*=(Matrix<DType>& value);
	inline void operator/=(DType value);
	inline void operator/=(Matrix<DType>& value);

	void sum(Matrix<DType>* other, DType alpha = DType(1.0), DType beta = DType(1.0));
	inline void pdiv(DType val) { pdiv(val, this); }
	void pdiv(DType val, Matrix<DType>* result);
	void mul(Matrix<DType>* other, Matrix<DType>* result);
	DType dot(Matrix<DType>* other, Matrix<DType>* result);

	void exp(DType alpha = DType(1.0)) { exp(this, alpha); }
	void exp(Matrix<DType>* other, DType alpha = DType(1.0));

private:
	DType* _data;
	int _data_size;
	int _m;
	int _n;
};


// Constructors
template <typename DType>
Matrix<DType>::Matrix(int m, int n) :
	_data((DType*)malloc(m * n * sizeof(DType))),
	_data_size(m * n),
	_m(m),
	_n(n)
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
	_m(m),
	_n(n)
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

// Sum operators
template <typename DType>
void Matrix<DType>::operator+=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] += value;
	}
}

template <typename DType>
void Matrix<DType>::operator+=(Matrix<DType>& other)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] += other._data[i];
	}
}

template <typename DType>
void Matrix<DType>::operator-=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] -= value;
	}
}

template <typename DType>
void Matrix<DType>::operator-=(Matrix<DType>& other)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] -= other._data[i];
	}
}


// Mult operators
template <typename DType>
void Matrix<DType>::operator*=(DType value)
{
	for (int i = 0; i < shape().prod(); ++i)
	{
		_data[i] *= value;
	}
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
