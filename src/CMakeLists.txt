set(Latte_SOURCES
    main.cpp
    layer.hpp layer.cpp
    linear.hpp linear.cpp
    matrix.hpp matrix.cpp
    matrix_factory.hpp matrix_factory.cpp
    sigmoid.hpp sigmoid.cpp
)

add_executable(Latte ${Latte_SOURCES})
target_link_libraries(Latte gflags ${OpenBLAS_LIBNAME})
