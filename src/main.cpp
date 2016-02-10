#include <gflags/gflags.h>
#include <iostream>
#include <vector>

#include "linear.hpp"
#include "matrix.hpp"
#include "matrix_factory.hpp"
#include "sigmoid.hpp"

DEFINE_bool(testing, false, "Set to true to test");

#define DT float

int main(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	srand(time(NULL));

	Matrix<DT> x(4, 3);
	x(0, 0) = 0; x(0, 1) = 0; x(0, 2) = 1;
	x(1, 0) = 1; x(1, 1) = 1; x(1, 2) = 1;
	x(2, 0) = 1; x(2, 1) = 0; x(2, 2) = 1;
	x(3, 0) = 0; x(3, 1) = 1; x(3, 2) = 1;

	Matrix<DT> y(4, 1);
	y(0, 0) = 0; y(1, 0) = 1; y(2, 0) = 1; y(3, 0) = 0;
	
	Linear<DT> linear(&y, &x);
	linear.stack<Sigmoid<DT>>(50);
	linear.stack<Sigmoid<DT>>(50);
	linear.stack<Sigmoid<DT>>(50);
	linear.stack<Sigmoid<DT>>(1);
	
	for (int k = 0; k < 600000; ++k)
	{
		// Forward net
		linear.forward();
		linear.backward();

		// Update Matrix pool
		MatrixFactory<DT>::get()->update();
	}

	getchar();
	return 0;
}
