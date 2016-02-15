#include <gflags/gflags.h>
#include <iostream>
#include <vector>

#include "optimization/sgd.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_factory.hpp"
#include "layers/sigmoid_layer.hpp"


DEFINE_bool(testing, false, "Set to true to test");

#define DT float

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	srand((unsigned int)time(NULL));

	Matrix<DT> x(4, 3);
	x(0, 0) = 0; x(0, 1) = 0; x(0, 2) = 1;
	x(1, 0) = 1; x(1, 1) = 1; x(1, 2) = 1;
	x(2, 0) = 1; x(2, 1) = 0; x(2, 2) = 1;
	x(3, 0) = 0; x(3, 1) = 1; x(3, 2) = 1;

	Matrix<DT> y(4, 1);
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 0;
	
	SGD<DT> sgd{ NamedArguments, SGDConfig<DT>::data = &x, SGDConfig<DT>::target = &y, SGDConfig<DT>::learning_rate = 1 };
	//sgd.stack<SigmoidLayer<DT>>(50);
	//sgd.stack<SigmoidLayer<DT>>(50);
	//sgd.stack<SigmoidLayer<DT>>(50);
	//sgd.stack<SigmoidLayer<DT>>(1);
    
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

	getchar();
	return 0;
}
