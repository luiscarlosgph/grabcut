##
# @brief  CMakeLists.txt for the GrabCut project.
# @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   10 Feb 2021



########################################
# Set CMake project-related parameters #
########################################

cmake_minimum_required(VERSION 3.5)
project(GrabCut)
set(GRABCUT_VERSION_MAJOR 0 CACHE STRING "Major version number" FORCE)
set(GRABCUT_VERSION_MINOR 1 CACHE STRING "Minor version number" FORCE)
set(GRABCUT_VERSION_PATCH 0 CACHE STRING "Patch version number" FORCE)
mark_as_advanced(GRABCUT_VERSION_MAJOR)
mark_as_advanced(GRABCUT_VERSION_MINOR)
mark_as_advanced(GRABCUT_VERSION_PATCH)



#####################
# Find dependencies #
#####################

# CUDA
find_package(CUDA)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=compute_61 -code=sm_61,compute_61")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -fopenmp -O3 -march=native")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# Python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# libboost_pythonXX.a
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS python3 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# libpbcvt.so
#find_library(Pbcvt libpbcvt.so)



################################
# Setup building-related flags #
################################

include_directories(
  inc 
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)

# Activate/deactivate the use of shared libraries
option(BUILD_SHARED_LIBS "Build GrabCut with shared libraries." OFF)
mark_as_advanced(BUILD_SHARED_LIBS)

# Set compilation output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

 

#####################
# Build the project #
#####################

add_subdirectory(src)
