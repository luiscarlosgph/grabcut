/**
 * @brief   Header-only liboost_python converter from ndarray to cv::Mat (and vice versa).
 * @author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).  
 * @date    12 Feb 2021.
 *
 * @details The code in this file is based on different parts of the following repository:
 *          - https://github.com/Algomorph/pyboostcvconverter
 *          The code in the repository above (pyboostcvconverter) is distributed under an 
 *          MIT license that can be found here:
 *          - https://raw.githubusercontent.com/Algomorph/pyboostcvconverter/master/LICENSE
 */

#ifndef NDARRAY_CV_CONVERTER_H
#define NDARRAY_CV_CONVERTER_H 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cstdio>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>
#include <boost/python.hpp>


static PyObject* opencv_error = 0;


class PyAllowThreads {
  public:
    PyAllowThreads() : _state(PyEval_SaveThread()) { }
    ~PyAllowThreads() {
      PyEval_RestoreThread(_state);
    }
  protected:
    PyThreadState* _state;
};


class PyEnsureGIL {
  public:
	  PyEnsureGIL() : _state(PyGILState_Ensure()) { }
    ~PyEnsureGIL() { 
      PyGILState_Release(_state);
	  }
  protected:
	  PyGILState_STATE _state;
};


class NumpyAllocator: public cv::MatAllocator {
  public:
    NumpyAllocator() {
      stdAllocator = cv::Mat::getStdAllocator();
    }
    ~NumpyAllocator() {}

    cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
        size_t* step) const {
      cv::UMatData* u = new cv::UMatData(this);
      u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
      npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
      for (int i = 0; i < dims - 1; i++)
        step[i] = (size_t) _strides[i];
      step[dims - 1] = CV_ELEM_SIZE(type);
      u->size = sizes[0] * step[0];
      u->userdata = o;
      return u;
    }

    cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data,
        size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const {
      if (data != NULL) {
        CV_Error(cv::Error::StsAssert, "[ERROR] Data should be NULL.");
        // Probably this is safe to do in such extreme case
        return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
      }

      PyEnsureGIL gil;
      int depth = CV_MAT_DEPTH(type);
      int cn = CV_MAT_CN(type);
      const int f = (int) (sizeof(size_t) / 8);
      int typenum =
        depth == CV_8U ? NPY_UBYTE :
        depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT :
        depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT :
        depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE :
        f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for (i = 0; i < dims; i++)
          _sizes[i] = sizes[i];
      if (cn > 1)
        _sizes[dims++] = cn;
      PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
      if (!o)
        CV_Error_(cv::Error::StsError, 
          ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
      return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, 
        cv::UMatUsageFlags usageFlags) const {
      return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(cv::UMatData* u) const {
      if (u) {
        PyEnsureGIL gil;
        PyObject* o = (PyObject*) u->userdata;
        Py_XDECREF(o);
        delete u;
      }
    }
  
  protected:
    const cv::MatAllocator* stdAllocator;
};


NumpyAllocator g_numpyAllocator;


class matToNDArrayBoostConverter {
  public:
    static PyObject* convert(cv::Mat const& m) {
      if (!m.data)
        Py_RETURN_NONE;
      cv::Mat temp, *p = (cv::Mat*) &m;
      if (!p->u || p->allocator != &g_numpyAllocator) {
        temp.allocator = &g_numpyAllocator;
        // ERRWRAP2(m.copyTo(temp));
        try { 
          PyAllowThreads allowThreads; \
          m.copyTo(temp);
        } 
        catch (const cv::Exception &e) { 
          PyErr_SetString(opencv_error, e.what());
          return 0; 
        }
        p = &temp;
      }
      PyObject* o = (PyObject*) p->u->userdata;
      Py_INCREF(o);
      return o; 
    }
};


class matFromNDArrayBoostConverter {
  public: 
    matFromNDArrayBoostConverter() {
        boost::python::converter::registry::push_back(convertible, construct,
          boost::python::type_id<cv::Mat>());
    }

		static void* convertible(PyObject* object) {
	    if (!PyArray_Check(object))
		    return NULL;
      #ifndef CV_MAX_DIM
	    const int CV_MAX_DIM = 32;
      #endif
	    PyArrayObject* oarr = (PyArrayObject*) object;

	    int typenum = PyArray_TYPE(oarr);
	    if (typenum != NPY_INT64 && typenum != NPY_UINT64 && typenum != NPY_LONG
			    && typenum != NPY_UBYTE && typenum != NPY_BYTE
			    && typenum != NPY_USHORT && typenum != NPY_SHORT
			    && typenum != NPY_INT && typenum != NPY_INT32
			    && typenum != NPY_FLOAT && typenum != NPY_DOUBLE) 
		    return NULL;
	    int ndims = PyArray_NDIM(oarr); // Data type not supported
	    if (ndims >= CV_MAX_DIM) 
		    return NULL; // Too many dimensions
	    return object;
    }

    static void construct(PyObject* object,
		    boost::python::converter::rvalue_from_python_stage1_data* data) {
	    namespace python = boost::python;
	    // Object is a borrowed reference, so create a handle indicting it is
	    // borrowed for proper reference counting
	    python::handle<> handle(python::borrowed(object));

	    // Obtain a handle to the memory block that the converter has allocated
	    // for the C++ type
	    typedef python::converter::rvalue_from_python_storage<cv::Mat> storage_type;
	    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

      // Allocate the C++ type into the converter's memory block, and assign
      // its handle to the converter's convertible variable.  The C++
      // container is populated by passing the begin and end iterators of
      // the python object to the container's constructor
      PyArrayObject* oarr = (PyArrayObject*) object;

      bool needcopy = false, needcast = false;
      int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
      int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
            typenum == NPY_USHORT ? CV_16U :
            typenum == NPY_SHORT ? CV_16S :
            typenum == NPY_INT ? CV_32S :
            typenum == NPY_INT32 ? CV_32S :
            typenum == NPY_FLOAT ? CV_32F :
            typenum == NPY_DOUBLE ? CV_64F : -1;

      if (type < 0) {
        needcopy = needcast = true;
        new_typenum = NPY_INT;
        type = CV_32S;
      }

      #ifndef CV_MAX_DIM
	    const int CV_MAX_DIM = 32;
      #endif
	    int ndims = PyArray_NDIM(oarr);

      int size[CV_MAX_DIM + 1];
      size_t step[CV_MAX_DIM + 1];
      size_t elemsize = CV_ELEM_SIZE1(type);
      const npy_intp* _sizes = PyArray_DIMS(oarr);
      const npy_intp* _strides = PyArray_STRIDES(oarr);
      bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

      for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        if ((i == ndims - 1 && (size_t) _strides[i] != elemsize)
            || (i < ndims - 1 && _strides[i] < _strides[i + 1]))
          needcopy = true;
      }

	    if (ismultichannel && _strides[1] != (npy_intp) elemsize * _sizes[2])
		    needcopy = true;

      if (needcopy) {
        if (needcast) {
          object = PyArray_Cast(oarr, new_typenum);
          oarr = (PyArrayObject*) object;
        } 
        else {
          oarr = PyArray_GETCONTIGUOUS(oarr);
          object = (PyObject*) oarr;
        }
        _strides = PyArray_STRIDES(oarr);
      }

	    for (int i = 0; i < ndims; i++) {
		    size[i] = (int) _sizes[i];
		    step[i] = (size_t) _strides[i];
	    }

      // Handle degenerate case
      if (ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
      }

      if (ismultichannel) {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
      }
      if (!needcopy) 
        Py_INCREF(object);

      cv::Mat* m = new (storage) cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
      m->u = g_numpyAllocator.allocate(object, ndims, size, type, step);
      m->allocator = &g_numpyAllocator;
      m->addref();
      data->convertible = storage;
    } 
}; // class matFromNDArrayBoostConverter


#endif // NDARRAY_CV_CONVERTER
