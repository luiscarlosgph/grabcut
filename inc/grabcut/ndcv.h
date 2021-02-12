#ifndef NDARRAY_CV_CONVERTER_H
#define NDARRAY_CV_CONVERTER_H 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cstdio>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>
#include <boost/python.hpp>

namespace ndcv {

class matToNDArrayBoostConverter {
	static PyObject* convert(cv::Mat const& m) {
    if (!m.data)
      Py_RETURN_NONE;
    
    cv::Mat temp,
    *p = (cv::Mat*) &m;
    if (!p->u || p->allocator != &g_numpyAllocator) {
      temp.allocator = &g_numpyAllocator;
      ERRWRAP2(m.copyTo(temp));
      p = &temp;
    }
    PyObject* o = (PyObject*) p->u->userdata;
    Py_INCREF(o);

    return o;
  }
};

} // namespace ndcv
#endif // NDARRAY_CV_CONVERTER
