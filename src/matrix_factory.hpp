#pragma once

#include <iostream>
#include <queue>
#include <functional>

template <typename DType>
class Matrix;

struct Shape;

template <typename DType>
struct MatrixCompare
{
	bool operator() (const Matrix<DType>* lhs, const Matrix<DType>* rhs);
};

template <typename DType>
class MatrixFactory
{
public:
	static MatrixFactory<DType>* get();

	Matrix<DType>* pop(Shape shape);
	Matrix<DType>* pop(Shape shape, DType value);
	Matrix<DType>* pop(Shape shape, DType* other);
	void update();

private:
	MatrixFactory();

private:
	static MatrixFactory<DType>* _instance;
	std::priority_queue<Matrix<DType>*, std::vector<Matrix<DType>* >, MatrixCompare<DType> > _pool;
	std::vector<Matrix<DType>* > _pending;
};

template <typename DType>
MatrixFactory<DType>* MatrixFactory<DType>::_instance = NULL;

template <typename DType>
MatrixFactory<DType>* MatrixFactory<DType>::get()
{
	if (_instance == NULL)
	{
		_instance = new MatrixFactory<DType>();
	}

	return _instance;
}
