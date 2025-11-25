from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert (
    sys.version_info.major == 3 and sys.version_info.minor >= 6
), "You should have Python 3.6 and greater."

setup(
    name="thrifty",
    py_modules=["thrifty"],
    version="0.0.1",
    install_requires=[
        "mujoco",
        "cloudpickle",
        "gymnasium",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "seaborn",
        "torch",
        "tqdm",
        "moviepy",
        "opencv-python",
        "torchvision",
        "robosuite",
        "h5py",
        "hidapi",
        "pygame",
        "robosuite_models",
        "black",
    ],
    description="Code for ThriftyDAgger.",
    author="Ryan Hoque",
)
