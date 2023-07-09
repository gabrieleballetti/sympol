from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include as numpy_get_include

setup(
    ext_modules=cythonize(["sympol/*.pyx"]),
    include_dirs=[numpy_get_include()],
)
