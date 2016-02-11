#pragma once

#include <unordered_map>

template <typename DType>
class Matrix;

template <typename DType>
class Activation
{
public:
	inline void apply(Matrix<DType>* matrix)
	{
		apply(matrix, matrix);
	}

	inline void derivative(Matrix<DType>* matrix, Matrix<DType>* alpha)
	{
		derivative(matrix, matrix, alpha);
	}

	virtual inline void apply(Matrix<DType>* matrix, Matrix<DType>* dest) = 0;
	virtual inline void derivative(Matrix<DType>* matrix, Matrix<DType>* dest, Matrix<DType>* alpha) = 0;

	template <class AType, typename... Params>
	static Activation<DType>* get(Params... params)
	{
		if (!_instance)
		{
			_instance = new AType(params...);
		}

		return _instance;
	}

protected:
	Activation() {};

private:
	static Activation<DType>* _instance;
};

template <typename DType>
Activation<DType>* Activation<DType>::_instance = nullptr;
