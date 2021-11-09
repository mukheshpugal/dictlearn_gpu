import pathlib
from setuptools import setup, find_packages
import subprocess

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

try:
    cuda_version = (
        subprocess.run(["nvcc", "-V"], stdout=subprocess.PIPE, encoding='utf8')
        .stdout.split("release ")[-1]
        .split(",")[0]
        .replace(".", "")
    )
except FileNotFoundError:
    raise Exception("nvcc not in PATH.")

setup(
    name="dictlearn-gpu",
    version="1.0.0",
    description="Dictionary learning with cuda.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mukheshpugal/dictlearn_gpu",
    author="Mukhesh Pugalendhi",
    author_email="mukheshpugalendhi@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={
        "dictlearn_gpu": ["src/ompbatch.cu"],
    },
    install_requires=[
        "numpy",
        f"cupy-cuda{cuda_version}",
    ],
)
