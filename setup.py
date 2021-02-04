#!/usr/bin/env python

import setuptools
import unittest

setuptools.setup(name='grabcut',
    version='0.1.0',
    description='GrabCut implementation for C++ with Python wrappers.',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT',
    url='https://github.com/luiscarlosgph/grabcut',
    packages=['grabcut'],
    package_dir={'grabcut' : 'src'}, 
    test_suite = 'tests',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
         ],
)
