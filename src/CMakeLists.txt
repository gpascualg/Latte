set(Latte_SOURCES
    main.cpp matrix.cpp matrix.hpp matrix_factory.hpp matrix_factory.cpp
)

add_executable(Latte ${Latte_SOURCES})
target_link_libraries(Latte gflags ${OpenBLAS_LIBNAME})
