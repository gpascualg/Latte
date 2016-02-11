#include "sigmoid_layer.hpp"
#include "matrix/matrix.hpp"
#include "activations/sigmoid_activation.hpp"
#include "fillers/random_filler.hpp"

template <typename DType>
SigmoidLayer<DType>::SigmoidLayer(Shape shape, int num_output) :
	SigmoidLayer(shape, num_output, Filler<DType>::get<RandomFiller<DType>>())
{}

template <typename DType>
SigmoidLayer<DType>::SigmoidLayer(Shape shape, int num_output, Filler<DType>* filler) :
	Layer(shape, num_output, Activation<DType>::get<SigmoidActivation<DType>>(), filler)
{}


// Specializations
template SigmoidLayer<float>::SigmoidLayer(Shape shape, int num_output);
template SigmoidLayer<double>::SigmoidLayer(Shape shape, int num_output);

template SigmoidLayer<float>::SigmoidLayer(Shape shape, int num_output, Filler<float>* filler);
template SigmoidLayer<double>::SigmoidLayer(Shape shape, int num_output, Filler<double>* filler);
