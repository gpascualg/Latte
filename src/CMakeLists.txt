set(Latte_SOURCES
    main.cpp
    layer.hpp layer.cpp layer_iterator.cpp
    linear.hpp linear.cpp
    matrix.hpp matrix.cpp
    matrix_factory.hpp matrix_factory.cpp
    sigmoid.hpp sigmoid.cpp

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
