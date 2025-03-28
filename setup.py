import numpy
from Cython.Build import cythonize
from setuptools import setup

ext_modules = ["execution_engine/task/process/*.pyx"]

setup(
    name="Interval Extension",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
