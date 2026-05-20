from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

VERSION = "1.0.0"

EXTRA_COMPILE_ARGS = []
EXTRA_LINK_ARGS = []

if sys.platform == 'win32':
    EXTRA_COMPILE_ARGS = ['/O2', '/EHsc', '/std:c++17']
else:
    EXTRA_COMPILE_ARGS = ['-O3', '-std=c++17', '-march=native']


class CMakeBuildExt(build_ext):
    def run(self):
        try:
            import pybind11
        except ImportError:
            print("Warning: pybind11 not found. C++ core will not be built.")
            print("Install with: pip install pybind11")
            return

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = os.path.abspath(os.path.dirname(__file__))
        cpp_dir = os.path.join(cwd, 'cpp')

        build_temp = os.path.join(cwd, 'build', 'temp')
        os.makedirs(build_temp, exist_ok=True)

        cfg = 'Release'

        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.join(cwd, "python")}',
            '-S', cpp_dir,
            '-B', build_temp,
        ]

        try:
            subprocess.check_call(['cmake'] + cmake_args, cwd=cwd)
            subprocess.check_call(['cmake', '--build', build_temp, '--config', cfg], cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f"[setup.py] CMake build failed: {e}")
            print("[setup.py] SNA will run in Python-only mode.")


setup(
    name='sna',
    version=VERSION,
    description='Simulated Neuron Architecture - Bio-inspired spiking neural network for autonomous thinking',
    author='XL1126',
    python_requires='>=3.8',
    packages=find_packages(),
    ext_modules=[Extension('core_cpp', sources=[])],
    cmdclass={'build_ext': CMakeBuildExt},
    install_requires=[
        'pybind11>=2.10.0',
        'pyyaml>=6.0',
        'flask>=2.0.0',
    ],
    extras_require={
        'dashboard': ['flask>=2.0.0'],
        'dev': ['pytest>=7.0.0', 'numpy>=1.24.0'],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
    ],
)
