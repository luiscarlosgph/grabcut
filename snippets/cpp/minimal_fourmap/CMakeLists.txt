cmake_minimum_required(VERSION 3.5)
project(minimal_snippet)

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
include_directories(
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)
# libboost_pythonXX.a
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS python3 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# libpbcvt.so
find_library(Pbcvt libpbcvt.so)

# libgrabcut.so
find_library(GrabCut libgrabcut.so)


################
# Make targets #
################

cuda_add_executable(minimal_fourmap minimal_fourmap.cpp) 
target_link_libraries(minimal_fourmap
  grabcut
  ${Python3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pbcvt}
)
