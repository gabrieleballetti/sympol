from setuptools import setup

setup(
    install_requires=[
        "igraph>=0.10.4",
        "numpy>=1.24.3",
        "numba>=0.59.0",
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
