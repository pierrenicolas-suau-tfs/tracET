from distutils.core import setup, Extension

module1 = Extension('supression',
                    include_dirs=['/usr/local/include'],
                   # libraries=['math','pthread','unistd','string'],
                    #library_dirs=['/usr/local/lib'],
                    sources = ['supressionmodule.c'])

setup (name = 'supression',
       version = '1.0',
       description = 'This pakckage contains the function nonmaxsup',
       ext_modules = [module1])