GrabCut
-------
Implementation of GrabCut with CUDA-based Gaussian Mixture Models. Works in C++ and Python. Run the following steps to get it working.
This code has been tested under the following configuration:
* Ubuntu 20.10
* GNU g++ 9.3.0
* CMake 3.16.3
* Python 3.8.6

Install dependencies
--------------------
* Python >= 3.8.2
* [CUDA](https://developer.nvidia.com/cuda-downloads) >= 8.0 (last tested to be working 11.0.2)
      
      # Ubuntu/Debian
      $ sudo apt update
      $ sudo apt install nvidia-cuda-toolkit
      
* [OpenCV](https://github.com/opencv/opencv) >= 3.4.3 (last tested to be working 4.5.1)
      
      # Ubuntu/Debian
      $ sudo apt update
      $ sudo apt install libopencv-dev python3-opencv
      
* [Numpy](https://pypi.org/project/numpy/) >= 1.20.0

      # Ubuntu/Debian
      $ sudo apt update
      $ sudo apt install python3-pip
      $ python3 -m pip install numpy --user

* [libpbcvt](https://github.com/luiscarlosgph/pyboostcvconverter): the [README](https://github.com/luiscarlosgph/pyboostcvconverter/blob/main/README.md) of the [libpbcvt](https://github.com/luiscarlosgph/pyboostcvconverter) repository explains how to compile and install this library.

<!--
* libboost_python >= 1.70.0 (last tested to be working 1.75.0)
      # Ubuntu/Debian
      $ sudo apt-get install libboost-all-dev
# Quick guide to install libboost_python
If do not want to install Boost from the official Ubuntu/Debian repositories as shown above, 
you may install a particular version from source as follows:
-->

<!--
```bash
$ wget https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz
$ tar xf boost_1_75_0.tar.gz
$ cd boost_1_75_0/
$ ./bootstrap.sh --with-python=/usr/bin/python3
-->

<!--
# If you want to install the Python libboost library **only**
$ ./b2 --with-python link=static cxxflags="-std=c++11 -fPIC" variant=release stage
$ sudo ./b2 --with-python link=static cxxflags="-std=c++11 -fPIC" variant=release install
```
-->

<!--
# If you want to install **all** the libboost libraries
$ ./b2 link=static cxxflags="-std=c++11 -fPIC" variant=release stage
$ sudo ./b2 link=static cxxflags="-std=c++11 -fPIC" variant=release install
-->

Compile and install GrabCut from source
---------------------------------------
```
$ git clone https://github.com/luiscarlosgph/grabcut.git
$ cd grabcut
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ sudo make install
```

Run GrabCut on an image
-----------------------
This commands are supposed to be executed from the root of the repository.

* Using a **trimap** (0 = sure background, 128 = unknown, 255 = sure foreground) as a scribble:

```bash
# Python
$ python3 src/main_grabcut.py --image data/tool_512x409.png --trimap data/trimap_512x409.png --output data/output_512x409_trimap_iter_5_gamma_10.png --iter 5 --gamma 10.0

# C++
$ build/bin/main_grabcut --image data/tool_512x409.png --trimap data/trimap_512x409.png --output data/output_512x409_trimap_iter_5_gamma_10.png --iter 5 --gamma 10.0
```

<table align="center">
  <tr>
    <td align="center">Image</td> <td align="center">Trimap</td> <td align="center">Output</td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/tool_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/trimap_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/output_512x409_trimap_iter_5_gamma_10.png?raw=true" width=205>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/tool_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/trimap_v2_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/output_512x409_trimap_v2_iter_5_gamma_10.png?raw=true" width=205>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/tool_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/trimap_v3_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/output_512x409_trimap_v3_iter_5_gamma_10.png?raw=true" width=205>
    </td>
  </tr>
</table>

It is not mandatory to provide sure foreground and sure background for a trimap, either of the two is sufficient. However, as shown above, performance may vary significantly when specifying just one of them.

* Using a **fourmap** (0 = sure background, 64 = probably background, 128 = probably foreground, 255 = sure foreground) as a scribble:

```bash
# Python
$ python3 src/main_grabcut.py --image data/tool_512x409.png --fourmap data/fourmap_512x409.png --output data/output_512x409_fourmap_iter_5_gamma_10.png --iter 5 --gamma 10.0

# C++
$ build/bin/main_grabcut --image data/tool_512x409.png --fourmap data/fourmap_512x409.png --output data/output_512x409_fourmap_iter_5_gamma_10.png --iter 5 --gamma 10.0
```

<table align="center">
  <tr>
    <td align="center">Image</td> <td align="center">Fourmap</td> <td align="center">Output</td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/tool_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/fourmap_512x409.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/grabcut/blob/main/data/output_512x409_fourmap_iter_5_gamma_10.png?raw=true" width=205>
    </td>
  </tr>
</table>

Exemplary code snippets
-----------------------
TODO

Common errors when installing from source
-----------------------------------------
1. Could NOT find CUDA (missing: CUDA_CUDART_LIBRARY)

Solution: specify cuda directory when compiling GrabCut, e.g.:
```bash
$ cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=[YOUR CUDA VERSION]..
```

2. fatal error: numpy/ndarrayobject.h: No such file or directory

Solution: the compiler cannot find the NumPy headers. Add the locations to CPATH, for example:
```bash
$ export CPATH=[YOUR HOME DIRECTORY]/.local/lib/python3.8/site-packages/numpy/core/include:$CPATH
```

3. If you find the following error:
```
../../pyboostcvconverter/build/libstatic_pbcvt.a(pyboost_cv4_converter.cpp.o): In function `pbcvt::NumpyAllocator::allocate(int, int const*, int, void*, unsigned long*, cv::AccessFlag, cv::UMatUsageFlags) const':
pyboost_cv4_converter.cpp:(.text._ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE[_ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE]+0x9d): undefined reference to `cv::error(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, char const*, char const*, int)'
pyboost_cv4_converter.cpp:(.text._ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE[_ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE]+0x2c5): undefined reference to `cv::format[abi:cxx11](char const*, ...)'
pyboost_cv4_converter.cpp:(.text._ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE[_ZNK5pbcvt14NumpyAllocator8allocateEiPKiiPvPmN2cv10AccessFlagENS5_14UMatUsageFlagsE]+0x2ed): undefined reference to `cv::error(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, char const*, char const*, int)'
../../pyboostcvconverter/build/libstatic_pbcvt.a(pyboost_cv4_converter.cpp.o):(.data.rel.ro._ZTVN5pbcvt14NumpyAllocatorE[_ZTVN5pbcvt14NumpyAllocatorE]+0x38): undefined reference to `cv::MatAllocator::map(cv::UMatData*, cv::AccessFlag) const'
collect2: error: ld returned 1 exit status
src/CMakeFiles/MainGrabcut.dir/build.make:129: recipe for target 'bin/MainGrabcut' failed
make[2]: *** [bin/MainGrabcut] Error 1
CMakeFiles/Makefile2:163: recipe for target 'src/CMakeFiles/MainGrabcut.dir/all' failed
make[1]: *** [src/CMakeFiles/MainGrabcut.dir/all] Error 2
Makefile:129: recipe for target 'all' failed
make: *** [all] Error 2
```

Solution: try installing OpenCV version 3.4.3 or 4.5.1, then recompile **both** pyboostcvconverter and GrabCut.

4. #error -- unsupported GNU version! gcc versions later than 8 are not supported!

Solution: choose your compiler
```bash
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
```
before you call ```cmake```. As you will see this error after you run ```cmake```, you need to run the two lines above, delete all the contents of the ```build``` folder and call ```cmake``` and ```make``` again.

Coding style
--------------
Please follow these guidelines when editing the code.  
C++: https://google.github.io/styleguide/cppguide.html  
Python: https://www.python.org/dev/peps/pep-0008

Commenting style
------------------
Please comment the C++ code using the Doxygen Javadoc style: http://www.doxygen.nl/manual/docblocks.html

License
---------
This project is distributed under an MIT license. See the [LICENSE](https://github.com/luiscarlosgph/grabcut/blob/main/LICENSE) file.
