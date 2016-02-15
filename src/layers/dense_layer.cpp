#include "dense_layer.hpp"
#include "matrix/matrix.hpp"
#include "fillers/random_filler.hpp"


template <typename DType>
DenseLayer<DType>::DenseLayer(GenericParameter* shape, GenericParameter* num_output,
	GenericParameter* activation, GenericParameter* filler, GenericParameter* dropout_ratio, GenericParameter* bias) :
	Layer<DType>(
        shape->as<Shape>(), 
        num_output->as<int>(), 
		activation->as<Activation<DType>*>(),
		filler->as<Filler<DType>*>(),
        dropout_ratio->as<DType>(),
        bias->as<BiasConfig<DType>>()
    )
{}


// Specializations
template DenseLayer<float>::DenseLayer(GenericParameter* shape, GenericParameter* num_output,
	GenericParameter* filler, GenericParameter* activation, GenericParameter* dropout_ratio, GenericParameter* bias);
template DenseLayer<double>::DenseLayer(GenericParameter* shape, GenericParameter* num_output,
	GenericParameter* filler, GenericParameter* activation, GenericParameter* dropout_ratio, GenericParameter* bias);
