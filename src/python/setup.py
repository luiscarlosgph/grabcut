from setuptools import setup, dist
from setuptools.command.install import install

class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

setup(
    name='grabcut',
    version='0.0.1',
    description="GrabCut implementation.",
    author='Luis Carlos Garcia Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT',
    packages=['grabcut'],
    package_data={'grabcut': ['_grabcut.so']},
    include_package_data=True,
    distclass=BinaryDistribution,
    url='https://github.com/luiscarlosgph/grabcut',
)
