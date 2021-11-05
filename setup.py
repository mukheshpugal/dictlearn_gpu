import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

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
    install_requires=[
        "numpy",
        "cupy",
    ],
)
