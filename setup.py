from setuptools import setup, find_packages
from Cython.Build import cythonize
from numpy import get_include as numpy_get_include

setup(
    ext_modules=cythonize(["./src/sympol/*.pyx"]),
    include_dirs=[numpy_get_include()],
    py_modules=["sympol"],
    install_requires=[
        "igraph>=0.10.4",
        "numpy>=1.24.3",
        "pycddlib>=2.1.6",
        "scipy>=0.12.0",
        "sympy>=1.12",
    ],
    packages=find_packages(),
)
