from distutils.core import setup, Extension
import numpy

module1 = Extension('supression',
                    include_dirs=['/usr/local/include'],
                   # libraries=['math','pthread','unistd','string'],
                    #library_dirs=['/usr/local/lib'],
                    sources = ['supressionmodule.c'])


setup (name = 'supression',
       version = '1.0',
       description = 'This package contains the functions to perform non-maximum suppression for different structures',
       ext_modules = [module1],
       include_dirs=[numpy.get_include()])
