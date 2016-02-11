set(Latte_SOURCES
    main.cpp
    common.hpp

    # Matrix
    matrix/matrix.hpp matrix/matrix.cpp
    matrix/matrix_factory.hpp matrix/matrix_factory.cpp

    # Optimizations
    optimization/sgd.hpp
    optimization/sgd.cpp

    # Layers
    layers/layer.hpp layers/layer.cpp
    layers/sigmoid_layer.hpp layers/sigmoid_layer.cpp
    layers/utils/layer_iterator.cpp

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
target_link_libraries(Latte gflags ${OpenBLAS_LIBNAME})
