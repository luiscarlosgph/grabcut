# Set the minimum CMake version.
cmake_minimum_required(VERSION 3.1)

# Set some CMake Policies.
# See http://cmake.org/cmake/help/cmake-2-8-docs.html#section_Policies for details
set(project_policies
  CMP0001 # NEW: CMAKE_BACKWARDS_COMPATIBILITY should no longer be used.
  CMP0002 # NEW: Logical target names must be globally unique.
  CMP0003 # NEW: Libraries linked via full path no longer produce linker search paths.
  CMP0004 # NEW: Libraries linked may NOT have leading or trailing whitespace.
  CMP0005 # NEW: Preprocessor definition values are now escaped automatically.
  CMP0006 # NEW: Installing MACOSX_BUNDLE targets requires a BUNDLE DESTINATION.
  CMP0007 # NEW: List command no longer ignores empty elements.
  CMP0008 # NEW: Libraries linked by full-path must have a valid library file name.
  CMP0009 # NEW: FILE GLOB_RECURSE calls should not follow symlinks by default.
  CMP0010 # NEW: Bad variable reference syntax is an error.
  CMP0011 # NEW: Included scripts do automatic cmake_policy PUSH and POP.
  CMP0012 # NEW: if() recognizes numbers and boolean constants.
  CMP0013 # NEW: Duplicate binary directories are not allowed.
  CMP0014 # NEW: Input directories must have CMakeLists.txt
  )
foreach(policy ${project_policies})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

if (NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
  # CMP0048 # OLD: Leave VERSION variables untouched.
  cmake_policy(SET CMP0048 OLD)
endif ()

if (NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  # CMP0053 # OLD: Leave references and variable expansion as pre version 3.1.
  cmake_policy(SET CMP0053 OLD)
endif ()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup dependency directories and paths
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{HOME}/cuda" CACHE STRING "CUDA root directory path")
set(BOOST_PYTHON_INC_DIR "/usr/local/include/boost/python" CACHE STRING "Path to the directory that contains the the Boost header files.")
set(BOOST_PYTHON_LIB_DIR "/usr/local/lib" CACHE STRING "Path to the directory that contains the Boost libraries.")
set(LIB_PYTHON_INC_DIR "/usr/include/python2.7")
set(SHARED_LIB_PATH "/usr/local/lib/libgrabcut.so")
set(SHARED_LIB_LINK "/usr/local/lib/python2.7/dist-packages/grabcut.so")

# This is how to check if a file exists
#if(EXISTS "")
#...
#else()
#...
#endif()

# CUDA
find_package(CUDA)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=compute_52 -code=sm_52,compute_52 --expt-relaxed-constexpr")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=compute_61 -code=sm_61,compute_61")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3 -march=native")

# CUDA libraries
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

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
include_directories(inc pyboostcvconverter/include ${BOOST_PYTHON_INC_DIR} ${LIB_PYTHON_INC_DIR})
 
# Decide what subdirectories we are building, and go and build them.
add_subdirectory(src)
