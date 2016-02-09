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
	Linear(Matrix<DType>* target, Matrix<DType>* data);
	
	template <typename LType>
	void stack(int num_output);
	void stack(Layer<DType>* layer);

	void forward();
	void backward();


private:
	int _k;
	std::vector<Layer<DType>* > _layers;
	Matrix<DType>* _target;
	Matrix<DType>* _first_in;
	Matrix<DType>* _last_out;
};
