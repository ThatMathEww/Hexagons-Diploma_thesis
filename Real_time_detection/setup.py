from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(r"Python_projects/HEXAGONS/Hexagons-Diploma_thesis/Real_time_detection/homography_module.pyx"),
    include_dirs=[np.get_include()]
)

# python Python_projects/HEXAGONS/Hexagons-Diploma_thesis/Real_time_detection/setup.py build_ext --inplace
