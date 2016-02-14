#include "layer_config.hpp"

template <> Parameter<Shape> LayerConfig<float>::shape("shape");
template <> Parameter<Shape> LayerConfig<double>::shape("shape");

template <> Parameter<int> LayerConfig<float>::num_output("num_output");
template <> Parameter<int> LayerConfig<double>::num_output("num_output");

template <> Parameter<Activation<float>*> LayerConfig<float>::activation("activation");
template <> Parameter<Activation<double>*> LayerConfig<double>::activation("activation");

template <> Parameter<Filler<float>*> LayerConfig<float>::filler("filler");
template <> Parameter<Filler<double>*> LayerConfig<double>::filler("filler");