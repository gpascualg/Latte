#include "sigmoid.hpp"
#include "matrix.hpp"
#include "activations/sigmoid_activation.hpp"
#include "fillers/random_filler.hpp"

template <typename DType>
Sigmoid<DType>::Sigmoid(Shape shape, int num_output) :
	Sigmoid(shape, num_output, Filler<DType>::get<RandomFiller<DType>>())
{}

template <typename DType>
Sigmoid<DType>::Sigmoid(Shape shape, int num_output, Filler<DType>* filler) :
	Layer(shape, num_output, Activation<DType>::get<SigmoidActivation<DType>>(), filler)
{}


// Specializations
template Sigmoid<float>::Sigmoid(Shape shape, int num_output);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output);

template Sigmoid<float>::Sigmoid(Shape shape, int num_output, Filler<float>* filler);
template Sigmoid<double>::Sigmoid(Shape shape, int num_output, Filler<double>* filler);

