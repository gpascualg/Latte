include(ExternalProject)

set(DEP_BASE ${CMAKE_BINARY_DIR}/dep)
SET_PROPERTY(DIRECTORY PROPERTY EP_BASE ${DEP_BASE})


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
if(NOT USE_GLOBAL_OPENBLAS)
    set(OpenBLAS_DIR ${DEP_BASE}/Install/EP_OpenBLAS)

    ExternalProject_Add(
        EP_OpenBLAS
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
        LOG_UPDATE ON
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${OPENBLAS_DIR} -DINCLUDE_INSTALL_DIR=.
    )

    find_library(OpenBLAS_LIBRARY openblas PATHS ${DEP_BASE}/Build/EP_OpenBLAS/lib)
    include_directories(${DEP_BASE}/Source/EP_OpenBLAS)
endif()

# ---[ pcg32
include_directories(dep/pcg32)
