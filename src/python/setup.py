#!/usr/bin/env python

from setuptools import setup, dist
from setuptools.command.install import install

class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

setup(name='grabcut',
    version='0.2.0',
    description='GrabCut implementation for C++ with Python wrappers.',
    url='https://github.com/luiscarlosgph/grabcut',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT',
    packages=['grabcut'],
    package_data={'grabcut': ['_grabcut.so']},
    include_package_data=True,
    distclass=BinaryDistribution,
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
