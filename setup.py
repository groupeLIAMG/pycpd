
# python3 setup.py build_ext --inplace

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
        Extension('radon',
                  sources=['radon.pyx','cradon.c'],
                  include_dirs=[np.get_include()],
                  library_dirs=['.'],
                  extra_compile_args=['-std=c99'])]
setup(
    name = 'radon extension module',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
