#include <gflags/gflags.h>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "optimization/sgd.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "layers/sigmoid_layer.hpp"
#include "layers/dense_layer.hpp"


DEFINE_bool(testing, false, "Set to true to test");


template <typename T, typename... R>
std::vector<T> make_vector(T a, R... b)
{
	return std::vector<T> {a, b...};
}

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	srand((unsigned int)time(NULL));

	Matrix<float> x(4, 3);
	x(0, 0) = 0; x(0, 1) = 0; x(0, 2) = 1;
	x(1, 0) = 1; x(1, 1) = 1; x(1, 2) = 1;
	x(2, 0) = 1; x(2, 1) = 0; x(2, 2) = 1;
	x(3, 0) = 0; x(3, 1) = 1; x(3, 2) = 1;

	Matrix<float> y(4, 1);
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 0;

	using namespace Float;

	auto d0 = DenseLayer() << 
		Config::NumOutput(50) << 
		Config::Data(&x) << 
		Config::Finalizer();

	auto d1 = DenseLayer() << 
		Config::NumOutput(50) << 
		Config::Bias(DefaultBias()) <<
		/*Config::Dropout(0.1f) <<*/
		Config::Finalizer();

	auto d2 = SigmoidLayer() << 
		Config::NumOutput(1) << 
		Config::Finalizer();

	d1 << d0;
	d2 << d1;

	SGD sgd = SGD({d0, d1, d2}) <<
		Config::Target(&y) <<
		Config::LearningRate(1.0f) <<
		Config::Iterations({1000000, 100000});

	for (int i = 0; i < 100000; ++i)
	{
		sgd.optimize();
	}

	// Free memory
	MatrixFactory<float>::get()->destroy();
	
	getchar();
	return 0;
}
