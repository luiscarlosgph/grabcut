# Set the minimum CMake version.
cmake_minimum_required(VERSION 3.5)
project(GrabCut)

# Find CUDA
find_package(CUDA)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=compute_61 -code=sm_61,compute_61")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -fopenmp -O3 -march=native")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# Find Python libraries
find_package(Python3 COMPONENTS Interpreter Development)

# Find libboost_pythonXX.a
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS python3 REQUIRED)

# We manually set version numbers. See CMP0048.
set(GRABCUT_VERSION_MAJOR 0 CACHE STRING "Major version number" FORCE)
set(GRABCUT_VERSION_MINOR 0 CACHE STRING "Minor version number" FORCE)
set(GRABCUT_VERSION_PATCH 1 CACHE STRING "Patch version number" FORCE)
mark_as_advanced(GRABCUT_VERSION_MAJOR)
mark_as_advanced(GRABCUT_VERSION_MINOR)
mark_as_advanced(GRABCUT_VERSION_PATCH)

# Now set up flags for this project.
option(BUILD_SHARED_LIBS "Build GRABCUT with shared libraries." OFF)
mark_as_advanced(BUILD_SHARED_LIBS)
message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# Output directories, for when compiling, not installing.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Setup further include folders
include_directories(inc pyboostcvconverter/include ${Boost_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
 
# Decide what subdirectories we are building, and go and build them.
add_subdirectory(src)
