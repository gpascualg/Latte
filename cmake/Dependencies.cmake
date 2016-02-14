
# ---[ GFlags
SET(gflags_LOCAL_DIR ${PROJECT_SOURCE_DIR}/dep/gflags)
if(EXISTS "${gflags_LOCAL_DIR}" AND IS_DIRECTORY "${gflags_LOCAL_DIR}" AND NOT USE_GLOBAL_GFLAGS)
    execute_process(COMMAND
        git apply --whitespace=fix ../../cmake/Patches/gflags-submodule.patch
	WORKIND_DIRECTORY ${PROJECT_SOURCE_DIR}/dep/gflags
        OUTPUT_QUIET
        ERROR_QUIET)

    add_subdirectory(dep/gflags)
    include_directories(${PROJECT_BINARY_DIR}/dep/gflags/include)
else()
    find_package(gflags)
endif()

# ---[ OpenBLAS
SET(OpenBLAS_LOCAL_DIR ${PROJECT_SOURCE_DIR}/dep/OpenBLAS)
if(EXISTS "${OpenBLAS_LOCAL_DIR}" AND IS_DIRECTORY "${OpenBLAS_LOCAL_DIR}" AND NOT USE_GLOBAL_BLAS)
    execute_process(COMMAND
        git apply --whitespace=fix ../../cmake/Patches/openblas-submodule.patch
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/dep/OpenBLAS
	OUTPUT_QUIET
	ERROR_QUIET)

    add_subdirectory(dep/OpenBLAS)
    include_directories(dep/OpenBLAS)
else()
    find_package(OpenBLAS)
endif()

if(MSVC)
    set(OpenBLAS_LIBNAME libopenblas CACHE INTERNAL "OpenBLAS Library name")
else()
    set(OpenBLAS_LIBNAME openblas_static CACHE INTERNAL "OpenBLAS Library name")
endif()
