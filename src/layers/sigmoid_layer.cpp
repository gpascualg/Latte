#include "sigmoid_layer.hpp"
#include "matrix/matrix.hpp"
#include "activations/sigmoid_activation.hpp"
#include "fillers/random_filler.hpp"


template <typename DType>
SigmoidLayer<DType>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, 
    GenericParameter* filler, GenericParameter* dropout_ratio) :
	Layer<DType>(
        shape->as<Shape>(), 
        num_output->as<int>(), 
        FromFactory(Activation, SigmoidActivation, DType)(), filler->as<Filler<DType>*>(),
        dropout_ratio->as<DType>()
    )
{}


// Specializations
template SigmoidLayer<float>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, 
    GenericParameter* filler, GenericParameter* dropout_ratio);
template SigmoidLayer<double>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, 
    GenericParameter* filler, GenericParameter* dropout_ratio);
