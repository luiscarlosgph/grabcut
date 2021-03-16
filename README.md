[![Build Status](https://travis-ci.org/luiscarlosgph/grabcut.svg?branch=main)](https://travis-ci.org/luiscarlosgph/grabcut)

GrabCut library (libgrabcut.so)
-------------------------------
This repository contains a library (and exemplary programs in C++ and Python that use the library). 
It implements GrabCut with CUDA-based Gaussian Mixture Models, and works seamlessly in C++ and Python (see minimal code snippets below).
The code has been tested under the following configuration:

* Ubuntu 20.10
* GNU g++ 9.3.0
* CMake 3.16.3
* Python 3.8.6

Install dependencies
--------------------
These dependencies need to be installed no matter if you compile this repo from source or install it using pip.
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

* libboost_python >= 1.70.0 (last tested to be working 1.75.0)
      
      $ wget https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz
      $ tar xf boost_1_75_0.tar.gz
      $ cd boost_1_75_0/
      $ ./bootstrap.sh --with-python=/usr/bin/python3
      $ ./b2 --with-python link=static cxxflags="-std=c++11 -fPIC" variant=release stage
      $ sudo ./b2 --with-python link=static cxxflags="-std=c++11 -fPIC" variant=release install

<!--
# If you want to install the Python libboost library **only**
# If you want to install **all** the libboost libraries
$ ./b2 link=static cxxflags="-std=c++11 -fPIC" variant=release stage
$ sudo ./b2 link=static cxxflags="-std=c++11 -fPIC" variant=release install
-->

Install from source
-------------------
If you prefer to install this repo with pip, skip this section and continue to the next one.
```
$ git clone https://github.com/luiscarlosgph/grabcut.git
$ cd grabcut
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ sudo make install
```

Install with pip
----------------
If you already compiled and installed this repo from source, do not install it again with pip.

The pip package contains a pre-compiled binary grabcut library, which depends on OpenCV 4.5 and Numpy 1.20.1. It is likely that your Linux distribution comes with different versions, but you can install them both following:

* OpenCV
```bash
# Install OpenCV dependencies
$ sudo apt update
$ sudo apt install build-essential cmake git unzip pkg-config libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev libx264-dev python3-dev libv4l-dev libavcodec-dev libavformat-dev libavresample-dev libswscale-dev libxvidcore-dev gfortran openexr libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# Download OpenCV
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv_contrib
$ git checkout 4.5.1
$ cd ../opencv
$ git checkout 4.5.1

# Compile OpenCV
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON ..
$ make -j8

# Install OpenCV
$ sudo make install
$ sudo ldconfig
```

* Numpy
```bash
$ python3 -m pip install numpy==1.20.1 --user
```

Then, you can use pip to install the grabcut package:
```bash
$ python3 -m pip install grabcut --user
```

Run GrabCut on a single image
-----------------------------
The Python script and the C++ program below allow you to run the GrabCut segmentation on an image without coding. These commands are supposed to be executed from the root of the repository. 

* Using a **trimap** (0 = sure background, 128 = unknown, 255 = sure foreground) as a scribble:

```bash
# Python script that uses the library (libgrabcut.so)
$ python3 src/python/examples/main_grabcut.py --image data/tool_512x409.png --trimap data/trimap_512x409.png --output data/output_512x409_trimap_iter_5_gamma_10.png --iter 5 --gamma 10.0
# Time elapsed in GrabCut segmentation: 169 milliseconds

# C++ sample program that uses the library (libgrabcut.so)
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
  <!--    
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
  -->
</table>

It is not mandatory to provide sure foreground and sure background for a trimap, either of the two is sufficient. However, performance may vary when specifying just one of them.

* Using a **fourmap** (0 = sure background, 64 = probably background, 128 = probably foreground, 255 = sure foreground) as a scribble:

```bash
# Python
$ python3 src/python/examples/main_grabcut.py --image data/tool_512x409.png --fourmap data/fourmap_512x409.png --output data/output_512x409_fourmap_iter_5_gamma_10.png --iter 5 --gamma 10.0
# Time elapsed in GrabCut segmentation: 164ms

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

As for the trimap, not all the labels must be specified in a fourmap. However, at least a foreground pixel (probable or sure) and a background pixel (probable or sure) must be present.

Minimal code snippets
---------------------
After you have compiled and installed the library, you can use it as follows:
* Python: [minimal_trimap.py](https://raw.githubusercontent.com/luiscarlosgph/grabcut/main/snippets/python/minimal_trimap.py) (see below), [minimal_fourmap.py](https://raw.githubusercontent.com/luiscarlosgph/grabcut/main/snippets/python/minimal_fourmap.py)
  
  ```python
  # This is the code of the trimap snippet
  import cv2
  import grabcut

  image_path = 'data/tool_512x409.png' 
  trimap_path = 'data/trimap_512x409.png'
  output_path = 'data/output_512x409_trimap_iter_5_gamma_10.png'
  max_iter = 5
  gamma = 10.

  # Read image and trimap
  im = cv2.imread(image_path)
  im_bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
  trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

  # Perform segmentation
  gc = grabcut.GrabCut(max_iter)
  segmentation = gc.estimateSegmentationFromTrimap(im_bgra, trimap, gamma)

  # Save segmentation
  cv2.imwrite(output_path, segmentation)
  ```
  
  Once you have compiled and installed the library, you can run (from the root of the repository) both snippets doing:
  ```bash
  $ python3 snippets/python/minimal_trimap.py
  $ python3 snippets/python/minimal_fourmap.py
  ```

* C++: [minimal_trimap.cpp](https://raw.githubusercontent.com/luiscarlosgph/grabcut/main/snippets/cpp/minimal_trimap/minimal_trimap.cpp) (see below), [minimal_fourmap.cpp](https://raw.githubusercontent.com/luiscarlosgph/grabcut/main/snippets/cpp/minimal_fourmap/minimal_fourmap.cpp)
  
  ```cpp
  #include <opencv2/core/core.hpp>
  #include <grabcut/grabcut.h>

  int main(int argc, char **argv) {
    const std::string imagePath = "data/tool_512x409.png"; 
    const std::string trimapPath = "data/trimap_512x409.png";
    const std::string outputPath = "data/output_512x409_trimap_iter_5_gamma_10.png";
    int maxIter = 5;
    float gamma = 10.;

    // Read image and trimap
    cv::Mat im = cv::imread(imagePath);
    cv::Mat imBgra;
    cv::cvtColor(im, imBgra, cv::COLOR_BGR2BGRA);
    cv::Mat trimap = cv::imread(trimapPath, cv::IMREAD_GRAYSCALE);

    // Perform segmentation
    GrabCut gc(maxIter);
    cv::Mat segmentation = gc.estimateSegmentationFromTrimap(imBgra, trimap, gamma);

    // Save segmentation
    cv::imwrite(outputPath, segmentation);

  return 0;
  }
  ```
  Once you have compiled and installed the library, you can run (from the root of the repository) the snippets doing:
  
  ```bash
  $ # Trimap example
  $ cd snippets/cpp/minimal_trimap
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release ..
  $ make
  $ # You now need to go back to the root of the repo because imagePath = "data/tool_512x409.png" (see the snippet above)
  $ cd ../../../..
  $ ./snippets/cpp/minimal_trimap/build/minimal_trimap
  ```
  
  You can strip out the snippet folder (e.g. snippets/cpp/minimal_trimap) and make it your own separate project.
  Once the library is compiled and installed snippets are no longer dependent on this repository.

Common errors while compiling and installing from source
--------------------------------------------------------
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

3. #error -- unsupported GNU version! gcc versions later than 8 are not supported!

Solution: choose your compiler
```bash
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
```
before you call ```cmake```. As you will see this error after you run ```cmake```, you need to run the two lines above, delete all the contents of the ```build``` folder and call ```cmake``` and ```make``` again.

5. ImportError: libpbcvt.so: cannot open shared object file: No such file or directory

Solution: make sure that ```/usr/local/lib``` is in your LD_LIBRARY_PATH: ```export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH```.

Coding style
--------------
Please follow these guidelines when doing pull requests.
C++: https://google.github.io/styleguide/cppguide.html  
Python: https://www.python.org/dev/peps/pep-0008

Commenting style
------------------
Please comment the C++ code using the Doxygen Javadoc style: http://www.doxygen.nl/manual/docblocks.html

License
---------
This project is distributed under an MIT license. See the [LICENSE](https://github.com/luiscarlosgph/grabcut/blob/main/LICENSE) file.
