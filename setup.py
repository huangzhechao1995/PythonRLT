# distutils: include_dirs = /opt/food/include
# Cython compile instructions


from distutils.core import setup, Extension
from Cython.Build import cythonize

# Use python setup.py build --inplace
# to compile

ext_module = Extension(
    "rect",
    ["rect.pyx","Rectangle.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    libraries = ['armadillo']
)

setup(
  name = "rectangleapp",
  ext_modules = cythonize([ext_module]),
)