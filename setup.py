import numpy
from Cython.Build import cythonize
from setuptools import setup

ext_modules = [
    # Extension(
    #     "intervals",
    #     sources=["intervals.pyx", "intervals.cpp"],
    #     include_dirs=[numpy.get_include()],
    #     language="c++",
    # ),
    "execution_engine/task/process/*.pyx"
]

setup(
    name="Interval Extension",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
