#include "sigmoid_layer.hpp"
#include "matrix/matrix.hpp"
#include "activations/sigmoid_activation.hpp"
#include "fillers/random_filler.hpp"

#if defined(HAS_TEMPLATE_TEMPLATE)
    #define REGISTER_FACTORY_(BaseType, DataType) \
        template <template <typename> class SubType, typename... Params> \
        struct get ## BaseType ## _t<DataType, SubType, Params...> { \
            static inline BaseType<DataType>* _(Params... params) \
            { \
                return Factory<BaseType<DataType>>::Register::get<SubType<DataType>>(); \
            } \
        };

    #define REGISTER_FACTORY(BaseType) \
        template <typename DType, template <typename> class SubType, typename... Params> \
        struct get ## BaseType ## _t { \
            static inline BaseType<DType>* _(Params... params); \
        }; \
        REGISTER_FACTORY_(BaseType, float) \
        REGISTER_FACTORY_(BaseType, double); \
        template <typename DType, template <typename> class SubType, typename... Params> \
        BaseType<DType>* get ## BaseType(Params... params) { \
            return get ## BaseType ## _t<DType, SubType>::_(params...); \
        }
    
    #define FromFactory(BaseType, SubType, DType) get ## BaseType<DType, SubType>()
    
#else
    #define REGISTER_FACTORY_(BaseType, DataType) \
        template <class SubType, typename... Params> \
        struct get ## BaseType ## _t<DataType, SubType, Params...> { \
            static inline BaseType<DataType>* _(Params... params) \
            { \
                return Factory<BaseType<DataType>>::Register::get<SubType>(); \
            } \
        };

    #define REGISTER_FACTORY(BaseType) \
        template <typename DType, class SubType, typename... Params> \
        struct get ## BaseType ## _t { \
            static inline BaseType<DType>* _(Params... params); \
        }; \
        REGISTER_FACTORY_(BaseType, float) \
        REGISTER_FACTORY_(BaseType, double); \
        template <typename DType, class SubType, typename... Params> \
        BaseType<DType>* get ## BaseType(Params... params) { \
            return get ## BaseType ## _t<DType, SubType>::_(params...); \
        }
    
    #define FromFactory(BaseType, SubType, DType) get ## BaseType<DType, SubType<DType>>()
#endif
    
REGISTER_FACTORY(Activation);

template <typename DType>
SigmoidLayer<DType>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* filler) :
	Layer<DType>(shape->as<Shape>(), num_output->as<int>(), FromFactory(Activation, SigmoidActivation, DType), filler->as<Filler<DType>*>())
{}


// Specializations
template SigmoidLayer<float>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* filler);
template SigmoidLayer<double>::SigmoidLayer(GenericParameter* shape, GenericParameter* num_output, GenericParameter* filler);
