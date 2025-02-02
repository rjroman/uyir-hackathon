from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os  
import importlib.util

# Load version dynamically
spec = importlib.util.spec_from_file_location("version", os.path.join('.', 'darkflow', 'version.py'))
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
VERSION = version_module.__version__

# Allow users to disable Cython compilation for low-end systems
USE_CYTHON = os.environ.get('USE_CYTHON', '1') == '1'

# Define Cython extensions (optimized for low-end laptops)
ext_modules = []
if USE_CYTHON:
    common_args = {
        "include_dirs": [numpy.get_include()],
        "extra_compile_args": ['-O2'],  # Moderate optimization (avoid -O3 for less CPU strain)
    }

    if os.name == 'nt':
        ext_modules = [
            Extension("darkflow.cython_utils.nms",
                      sources=["darkflow/cython_utils/nms.pyx"], **common_args),
            Extension("darkflow.cython_utils.cy_yolo2_findboxes",
                      sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"], **common_args),
            Extension("darkflow.cython_utils.cy_yolo_findboxes",
                      sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"], **common_args),
        ]

    elif os.name == 'posix':
        common_args["libraries"] = ["m"]  # Link math library for Unix
        ext_modules = [
            Extension("darkflow.cython_utils.nms",
                      sources=["darkflow/cython_utils/nms.pyx"], **common_args),
            Extension("darkflow.cython_utils.cy_yolo2_findboxes",
                      sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"], **common_args),
            Extension("darkflow.cython_utils.cy_yolo_findboxes",
                      sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"], **common_args),
        ]

# Fallback for low-end laptops (use pure Python if needed)
try:
    ext_modules = cythonize(ext_modules)
except Exception as e:
    print("Warning: Cython extensions could not be compiled. Falling back to pure Python mode.")
    ext_modules = []

setup(
    version=VERSION,
    name='darkflow',
    description='Darkflow - Optimized for Low-End Laptops',
    license='GPLv3',
    url='https://github.com/thtrieu/darkflow',
    packages=find_packages(),
    scripts=['flow'],
    ext_modules=ext_modules,
)
