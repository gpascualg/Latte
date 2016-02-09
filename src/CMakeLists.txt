set(Latte_SOURCES
    main.cpp
    graph.hpp graph.cpp
    layer.hpp layer.cpp
    matrix.hpp matrix.cpp
    matrix_factory.hpp matrix_factory.cpp
    sigmoid.hpp sigmoid.cpp
)

add_executable(Latte ${Latte_SOURCES})
target_link_libraries(Latte gflags ${OpenBLAS_LIBNAME})
