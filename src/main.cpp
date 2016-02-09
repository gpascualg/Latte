#include <gflags/gflags.h>
#include <iostream>

#include "matrix.hpp"

DEFINE_bool(testing, false, "Set to true to test");

#define DT float

template <typename DType>
class Layer
{
public:
	Layer(Shape shape, int num_output) :
		_in_shape(shape),
		_out_shape({shape.m, num_output})
	{
		_weights = new Matrix<DType>(shape.n, num_output);
		_output = new Matrix<DType>(shape.m, num_output);
		_diff = new Matrix<DType>(shape.n, num_output);

		for (int i = 0; i < _weights->shape().prod(); ++i)
		{
			(**_weights)[i] = DType(rand()) / RAND_MAX - DType(0.5);
		}
	}

	virtual Matrix<DType>* forward(Matrix<DType>* in) = 0;
	virtual Matrix<DType>* backward(Matrix<DType>* error) = 0;
	virtual void update() = 0;

	inline Matrix<DType>* W() { return _weights; }
	inline Shape inShape() { return _in_shape; }
	inline Shape outShape() { return _out_shape; }

protected:
	Matrix<DType>* _in;
	Matrix<DType>* _weights;
	Matrix<DType>* _output;
	Matrix<DType>* _diff;

	Shape _in_shape;
	Shape _out_shape;
};

template <typename DType>
class Sigmoid : public Layer<DType>
{
public:
	Sigmoid(Shape shape, int num_output) :
		Layer(shape, num_output)
	{
		std::cout << "In shape: (" << shape.m << ", " << shape.n << ")" << std::endl;
		std::cout << "We shape: (" << _weights->shape().m << ", " << _weights->shape().n << ")" << std::endl;
		std::cout << "Ou shape: (" << _output->shape().m << ", " << _output->shape().n << ")" << std::endl;
	}

	Matrix<DType>* forward(Matrix<DType>* in)
	{
		// _in * _weights
		_in = in;
		_in->mul(_weights, _output);

		// 1 / (1 + exp(_output))
		_output->exp(DType(-1.0));
		*_output += 1;
		_output->pdiv(DType(1.0));
		return _output;
	}

	Matrix<DType>* backward(Matrix<DType>* error)
	{
		Matrix<DType>* derivative = DType(1.0) - *_output;
		*derivative *= *_output;
		*error *= *derivative;
		_in->T().mul(error, _diff);
		return error;
	}
	
	void update()
	{ 
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
	
	srand(time(NULL));
	Sigmoid<DT>* sigmoid_1 = new Sigmoid<DT>(x.shape(), 50);
	Sigmoid<DT>* sigmoid_2 = new Sigmoid<DT>(sigmoid_1->outShape(), 1);
	
	for (int k = 0; k < 600000; ++k)
	{
		// Forward
		Matrix<DT>* L1 = sigmoid_1->forward(&x);
		Matrix<DT>* L2 = sigmoid_2->forward(L1);

		// Backward L2
		Matrix<DT>* L2_error = y - *L2;

		if ((k % 10000) == 0)
		{
			DT sum = DT(0.0);
			for (int i = 0; i < L2_error->shape().prod(); ++i)
				sum += abs((**L2_error)[i]);

			std::cout << "ERROR: " << (sum / L2_error->shape().prod()) << std::endl;
			std::cout << (y)(0, 0) << "\t" << (y)(1, 0) << "\t" << (y)(2, 0) << "\t" << (y)(3, 0) << std::endl;
			std::cout << (*L2)(0, 0) << "\t" << (*L2)(1, 0) << "\t" << (*L2)(2, 0) << "\t" << (*L2)(3, 0) << std::endl << std::endl;
		}

		Matrix<DT>* L2_delta = sigmoid_2->backward(L2_error);

		// Backward L1
		Matrix<DT>* L1_error = MatrixFactory<DT>::get()->pop({ sigmoid_2->outShape().m, sigmoid_2->W()->shape().m });

		L2_delta->mul(&sigmoid_2->W()->T(), L1_error);
		Matrix<DT>* L1_delta = sigmoid_1->backward(L1_error);

		// Update layers
		sigmoid_2->update();
		sigmoid_1->update();

		// Update Matrix pool
		MatrixFactory<DT>::get()->update();
	}

	getchar();
	return 0;
}
