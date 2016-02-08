cmake_minimum_required(VERSION 2.8.7)

# ---[ Latte project
project(Latte C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# include(ExternalProject)

# ---[ Options
set(python_version "2" CACHE STRING "Specify which python version to use")

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
if(UNIX OR APPLE)
    #add_define(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)
endif()

# ---[ Config generation
#configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Latte_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Latte_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Subdirectories
add_subdirectory(src)
