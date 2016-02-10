#include "sigmoid.hpp"
#include "matrix.hpp"
#include "activations/sigmoid_activation.hpp"

template <typename DType>
Sigmoid<DType>::Sigmoid(Shape shape, int num_output) :
	Layer(shape, num_output, Activation<DType>::get<SigmoidActivation<DType>>())
{}


// Specializations
template Sigmoid<float>::Sigmoid(Shape shape, int num_output);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output);
