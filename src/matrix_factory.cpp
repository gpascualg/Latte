#include "matrix_factory.hpp"
#include "matrix.hpp"


template <typename DType>
MatrixFactory<DType>::MatrixFactory()
{}

template <typename DType>
Matrix<DType>* MatrixFactory<DType>::pop(Shape shape)
{
	Matrix<DType>* matrix = nullptr;

	if (!_pool.empty())
	{
		matrix = _pool.top();
		if (matrix->shape().prod() >= shape.prod())
		{
			_pool.pop();
			matrix->reshape(shape);
		}
	}
	else
	{
		matrix = new Matrix<DType>(shape);
	}

	_pending.push(matrix);
	return matrix;
}

template <typename DType>
Matrix<DType>* MatrixFactory<DType>::pop(Shape shape, DType value)
{
	Matrix<DType>* matrix = pop(shape);
	for (int i = 0; i < shape.prod(); ++i)
	{
		(*matrix)[i] = value;
	}

	return matrix;
}

template <typename DType>
Matrix<DType>* MatrixFactory<DType>::pop(Shape shape, DType* other)
{
	Matrix<DType>* matrix = pop(shape);
	for (int i = 0; i < shape.prod(); ++i)
	{
		(*matrix)[i] = other[i];
	}

	return matrix;
}

template <typename DType>
void MatrixFactory<DType>::update()
{
	while (!_pending.empty())
	{
		_pool.push(_pending.top());
		_pending.pop();
	}
}

// Specializations
template MatrixFactory<float>::MatrixFactory();
template MatrixFactory<double>::MatrixFactory();

template Matrix<float>* MatrixFactory<float>::pop(Shape shape);
template Matrix<double>* MatrixFactory<double>::pop(Shape shape);

template Matrix<float>* MatrixFactory<float>::pop(Shape shape, float value);
template Matrix<double>* MatrixFactory<double>::pop(Shape shape, double value);

template Matrix<float>* MatrixFactory<float>::pop(Shape shape, float* other);
template Matrix<double>* MatrixFactory<double>::pop(Shape shape, double* other);

template void MatrixFactory<float>::update();
template void MatrixFactory<double>::update();