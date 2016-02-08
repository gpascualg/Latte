#include <gflags/gflags.h>
#include <iostream>

#include "matrix.hpp"

DEFINE_bool(testing, false, "Set to true to test");

#define DT double

template <typename DType>
class Layer
{
public:
	Layer(Matrix<DType>* in, int num_output) :
		_in(in)
	{
		_weights = new Matrix<DType>(in->shape().n, num_output, DType(rand()) / RAND_MAX - DType(0.5));
		_output = new Matrix<DType>(in->shape().m, num_output);
		_diff = new Matrix<DType>(in->shape().n, num_output);
	}

	virtual Matrix<DType>* forward() = 0;
	virtual void backward(Matrix<DType>* error) = 0;

protected:
	Matrix<DType>* _in;
	Matrix<DType>* _weights;
	Matrix<DType>* _output;
	Matrix<DType>* _diff;
};

template <typename DType>
class Sigmoid : public Layer<DType>
{
public:
	Sigmoid(Matrix<DType>* in, int num_output) :
		Layer(in, num_output)
	{
		std::cout << "In shape: (" << in->shape().m << ", " << in->shape().n << ")" << std::endl;
		std::cout << "We shape: (" << _weights->shape().m << ", " << _weights->shape().n << ")" << std::endl;
		std::cout << "Ou shape: (" << _output->shape().m << ", " << _output->shape().n << ")" << std::endl;
	}

	Matrix<DType>* forward()
	{
		// _in * _weights
		_in->mul(_weights, _output);

		// 1 / (1 + exp(_output))
		_output->exp(DType(-1.0));
		*_output += 1;
		_output->pdiv(DType(1.0));
		return _output;
	}

	void backward(Matrix<DType>* error)
	{
		Matrix<DType>* derivative = DType(1.0) - *_output;
		*derivative *= *_output;
		*error *= *derivative;

		_in->T().mul(error, _diff);
		*_weights += *_diff;
	}
};

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	Matrix<DT> x(4, 3);
	x(0, 0) = 0; x(0, 1) = 0; x(0, 2) = 1;
	x(1, 0) = 1; x(1, 1) = 1; x(1, 2) = 1;
	x(2, 0) = 1; x(2, 1) = 0; x(2, 2) = 1;
	x(3, 0) = 0; x(3, 1) = 1; x(3, 2) = 1;

	Matrix<DT> y(4, 1);
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 0;

	
	Sigmoid<DT> sigmoid(&x, 1);
	Matrix<DT>* L1 = nullptr;

	for (int k = 0; k < 100000; ++k)
	{
		// Forward
		L1 = sigmoid.forward();

		// y - L1
		Matrix<DT>* L1_error = y - *L1;

		// Backward
		sigmoid.backward(L1_error);
	}

	std::cout << (*L1)(0, 0) << " " << (*L1)(1, 0) << " " << (*L1)(2, 0) << " " << (*L1)(3, 0) << std::endl;

	getchar();
}
