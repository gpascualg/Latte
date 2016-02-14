set(Latte_SOURCES
    main.cpp
    common.hpp

    # Magic
    magic/named_parameters.hpp

    # Matrix
    matrix/matrix.hpp matrix/matrix.cpp
    matrix/matrix_factory.hpp matrix/matrix_factory.cpp

    # Optimizations
    optimization/sgd.hpp optimization/sgd.cpp
    optimization/utils/sgd_config.hpp optimization/utils/sgd_config.cpp

    # Layers
    layers/layer.hpp layers/layer.cpp
    layers/sigmoid_layer.hpp layers/sigmoid_layer.cpp
    layers/utils/layer_iterator.cpp
    layers/utils/layer_config.hpp layers/utils/layer_config.cpp

    # Activations
    activations/activation.hpp
    activations/sigmoid_activation.hpp activations/sigmoid_activation.cpp
    activations/relu_activation.hpp activations/relu_activation.cpp
    activations/lrelu_activation.hpp activations/lrelu_activation.cpp

    # Fillers
    fillers/filler.hpp
    fillers/random_filler.hpp fillers/random_filler.cpp
)

add_executable(Latte ${Latte_SOURCES})
target_link_libraries(Latte gflags-static ${OpenBLAS_LIBNAME})

set_property(TARGET Latte PROPERTY CXX_STANDARD 11)
target_compile_features(Latte PRIVATE cxx_decltype cxx_nullptr cxx_variadic_templates)
