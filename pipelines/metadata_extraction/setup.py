from distutils.core import setup
from setuptools import find_packages

setup(
    name="metadata_extraction",
    version="0.0.1",
    description="Map metadata extraction pipeline",
    packages=find_packages(),
    keywords=["metadata", "extraction", "pipeline"],
    license="Apache-2.0",
    install_requires=["lara_modelling_tasks"],
)
