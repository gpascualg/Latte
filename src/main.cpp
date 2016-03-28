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
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 1;

	using namespace Float;

	auto d0 = DenseLayer() << 
		Config::NumOutput(50) << 
		Config::Shape(x.shape()) << 
		Config::Finalizer();

	auto d1 = DenseLayer() << 
		Config::NumOutput(50) << 
		Config::Shape(d0->outShape()) << 
		Config::Bias<float>(DefaultBias()) <<
		/*Config::Dropout(0.1f) <<*/
		Config::Finalizer();

	auto d2 = SigmoidLayer() << 
		Config::NumOutput(1) <<
		Config::Shape(d1->outShape()) << 
		Config::Finalizer();

	d0->connect(&x);
	d1->connect(d0);
	d2->connect(d1);

	for (int i = 0; i < 100000; ++i)
	{
		d0->forward();
		d1->forward();
		d2->forward();

		Matrix<float>* predicted = d2->output();
		Matrix<float>* error = (*predicted  - y);

		if ((i % 1000) == 0)
		{
			float sum = error->sum();
			std::cout << "ERROR: " << (sum / error->shape().prod()) << std::endl;
			std::cout << y(0, 0) << "\t" << y(1, 0) << "\t" << y(2, 0) << "\t" << y(3, 0) << std::endl;
			std::cout << (*predicted)(0, 0) << "\t" << (*predicted)(1, 0) << "\t" << (*predicted)(2, 0) << "\t" << (*predicted)(3, 0) << std::endl << std::endl;
		}

		error = d2->backward(error);

		Matrix<float>* delta = MatrixFactory<float>::get()->pop({ d2->outShape().m, d2->W()->shape().m });
		error->mul(d2->W()->T(), delta);
		error = d1->backward(delta);

		delta = MatrixFactory<float>::get()->pop({ d1->outShape().m, d1->W()->shape().m });
		error->mul(d1->W()->T(), delta);
		error = d0->backward(delta);

		// Update layers
		d2->update(0.1f);
		d1->update(0.1f);
		d0->update(0.1f);
	}

/*
	SGD<DT> sgd{ NamedArguments, SGDConfig<DT>::data = &x, SGDConfig<DT>::target = &y, SGDConfig<DT>::learning_rate = 1 };

#define VERSION 3
#if VERSION == 1
	sgd.stack<SigmoidLayer<DT>>(50);
	sgd.stack<SigmoidLayer<DT>>(50);
	sgd.stack<SigmoidLayer<DT>>(50);
	sgd.stack<SigmoidLayer<DT>>(1);
#elif VERSION == 2
    auto sigmoid_L1 = new SigmoidLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = x.shape(), 
        LayerConfig<DT>::num_output = 50 };
    auto sigmoid_L2 = new SigmoidLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = sigmoid_L1->outShape(), 
        LayerConfig<DT>::num_output = 50 };
    auto sigmoid_L3 = new SigmoidLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = sigmoid_L2->outShape(), 
        LayerConfig<DT>::num_output = 50 };
    auto sigmoid_L4 = new SigmoidLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = sigmoid_L3->outShape(), 
		LayerConfig<DT>::num_output = 1, LayerConfig<DT>::bias = NoBias<DT>() };

    sgd.stack(sigmoid_L1);
    sgd.stack(sigmoid_L2);
    sgd.stack(sigmoid_L3);
    sgd.stack(sigmoid_L4);
#elif VERSION == 3
	auto sigmoid_L1 = new DenseLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = x.shape(),
		LayerConfig<DT>::num_output = 50 };
	auto sigmoid_L12 = new DenseLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = sigmoid_L1->outShape(),
		LayerConfig<DT>::num_output = 50 };
	auto sigmoid_L2 = new SigmoidLayer<DT>{ NamedArguments, LayerConfig<DT>::shape = sigmoid_L12->outShape(),
		LayerConfig<DT>::num_output = 1, LayerConfig<DT>::bias = NoBias<DT>() };

	sgd.stack(sigmoid_L1);
	sgd.stack(sigmoid_L12);
	sgd.stack(sigmoid_L2);
#endif

	for (int k = 0; k < 600000; ++k)
	{
		// Forward net
		sgd.forward();
		sgd.backward();

		// Update Matrix pool
		MatrixFactory<DT>::get()->update();
	}

	// Free memory
	MatrixFactory<DT>::get()->destroy();
*/
	
	getchar();
	return 0;
}
