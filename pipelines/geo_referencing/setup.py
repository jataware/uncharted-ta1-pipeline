from distutils.core import setup
from setuptools import find_packages

setup(
    name="georef",
    version="0.0.1",
    description="Geo referencing pipeline",
    packages=find_packages(),
    keywords=["georef", "pipeline"],
    license="Apache-2.0",
    install_requires=["lara_modelling_tasks", "flask", "opencv-python"],
)
