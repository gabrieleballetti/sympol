from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include as numpy_get_include

setup(
    ext_modules=cythonize(["./src/sympol/*.pyx"]),
    include_dirs=[numpy_get_include()],
    install_requires=[
        "igraph>=0.10.4",
        "numpy>=1.24.3",
        "pycddlib>=2.1.6",
        "sympy>=1.12",
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "pytest-cov",
            "ruff",
        ],
        "docs": [
            "myst-parser",
            "pydocstyle",
            "sphinx-rtd-theme",
        ],
    },
)
