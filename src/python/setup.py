#!/usr/bin/env python

from setuptools import setup, dist
from setuptools.command.install import install

#class BinaryDistribution(dist.Distribution):
#    def has_ext_modules(foo):
#        return True

setup(name='grabcut',
    version='0.4.0',
    description='GrabCut implementation for C++ with Python wrappers.',
    url='https://github.com/luiscarlosgph/grabcut',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT',
    packages=['grabcut'],
    include_package_data=True,
    #package_data={'grabcut': ['_grabcut.so']},
    #distclass=BinaryDistribution
)
