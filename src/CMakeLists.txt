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
    optimization/utils/sgd_config.hpp

    # Layers
    layers/layer.hpp layers/layer.cpp
    layers/sigmoid_layer.hpp layers/sigmoid_layer.cpp
    layers/utils/layer_iterator.cpp
    layers/utils/layer_config.hpp

    # Activations
    activations/activation.hpp
    activations/sigmoid_activation.hpp activations/sigmoid_activation.cpp
    activations/relu_activation.hpp activations/relu_activation.cpp
    activations/lrelu_activation.hpp activations/lrelu_activation.cpp

    # Fillers
    fillers/filler.hpp
    fillers/random_filler.hpp fillers/random_filler.cpp
)

if(MSVC)
    set(gflags_LIB gflags)
else()
    set(gflags_LIB )
endif()

add_executable(Latte ${Latte_SOURCES})
target_link_libraries(Latte ${gflags_LIB} ${OpenBLAS_LIBNAME})
