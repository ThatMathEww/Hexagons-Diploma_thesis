from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(r"homography_module.pyx"),
    include_dirs=[np.get_include()]
)

# python setup.py build_ext --inplace
