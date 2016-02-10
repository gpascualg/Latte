#pragma once

#include <vector>

template <typename DType>
class Matrix;

template <typename DType>
class Layer;

template <typename DType>
class Linear
{
public:
	Linear(Matrix<DType>* data, Matrix<DType>* target);
	
	template <typename LType>
	void stack(int num_output);
	void stack(Layer<DType>* layer);

	void forward();
	void backward();


private:
	int _k;
	Layer<DType>* _firstLayer;
	Layer<DType>* _lastLayer;

	Matrix<DType>* _data;
	Matrix<DType>* _target;
};
