#include "matrix_factory.hpp"
#include "matrix.hpp"


template <typename DType>
bool MatrixCompare<DType>::operator() (const Matrix<DType>* lhs, const Matrix<DType>* rhs)
{
	return lhs->size() < rhs->size();
}

template <typename DType>
MatrixFactory<DType>::MatrixFactory()
{}

template <typename DType>
Matrix<DType>* MatrixFactory<DType>::pop(Shape shape)
{
	Matrix<DType>* matrix = nullptr;

	if (!_pool.empty())
	{
		Matrix<DType>* temp_matrix = _pool.top();
		if (temp_matrix->size() >= shape.prod())
		{
			_pool.pop();
			temp_matrix->reshape(shape);
			matrix = temp_matrix;
		}
	}
	
	if (matrix == nullptr)
	{
		matrix = new Matrix<DType>(shape);
	}

	_pending.push_back(matrix);
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
		_pool.push(_pending.back());
		_pending.pop_back();
	}
}

template <typename DType>
void MatrixFactory<DType>::destroy()
{
	while (!_pending.empty())
	{
		delete _pending.back();
		_pending.pop_back();
	}

	while (!_pool.empty())
	{
		delete _pool.top();
		_pool.pop();
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

template void MatrixFactory<float>::destroy();
template void MatrixFactory<double>::destroy();
