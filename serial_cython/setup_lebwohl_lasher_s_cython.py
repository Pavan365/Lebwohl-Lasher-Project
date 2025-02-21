# Import required libraries.
from Cython.Build import cythonize
import numpy as np
from setuptools import setup
from setuptools.extension import Extension

# Cythonize the "lebwohl_lasher_s_cython.pyx" file and generate an extension.
lebwohl_lasher_s_cython = Extension(
    name="lebwohl_lasher_s_cython",
    sources=["lebwohl_lasher_s_cython.pyx"],
    extra_compile_args=["-O3", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
    include_dirs=[np.get_include()],  
)

# Set up the extension.
setup(ext_modules=cythonize(lebwohl_lasher_s_cython))