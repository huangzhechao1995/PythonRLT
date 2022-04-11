# distutils: include_dirs = /usr/local/include
# Cython compile instructions
from distutils.core import setup, Extension
from Cython.Build import cythonize

import platform

if platform.system() == 'Darwin':
  lapack_lib = []
  lapack_extra = ['-framework', 'Accelerate']
else:
  lapack_lib = ['blas','lapack']
  lapack_extra = []

ext_modules = cythonize(
                Extension('thepackage.cythonmodule.interface',
                sources=['thepackage/cythonmodule/interface.pyx'],
                include_dirs=['/Users/zhechao/Documents/Research/UIUC/RLT/include/src/', "/usr/local/include/"],
                library_dirs=['../lib', "/usr/local/include/"],
                extra_compile_args=["-std=c++11","-lboost_python37"], #,"-I /usr/local/include", , "-larmadillo"
                extra_link_args=["-std=c++11"]+lapack_extra,
                libraries = ['rltlib']+lapack_lib,
                language='c++',
                language_level=3.9
               ))

packages = ['thepackage.cythonmodule']

setup(name='thepackage',
      packages=packages,
      ext_modules=ext_modules,
      package_data={'thepackage.cythonmodule.interface': ['*.so']})


