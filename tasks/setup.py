from distutils.core import setup
from setuptools import find_packages

setup(
    name="lara_modelling_tasks",
    version="0.0.1",
    description="Tasks for critical maas map extraction pipeline composition",
    packages=find_packages(),
    keywords=["critical maas", "segmentation", "metadata", "georeferencing"],
    license="Apache-2.0",
    install_requires=[
        "numpy",
        "tqdm",
        "pydantic",
        "tiktoken",
        "openai",
        "google-cloud-vision",
        "grpcio",
        "boto3",
        "pillow",
        "jsons",
        "geopy",
        "matplotlib",
        "opencv_python",
        "protobuf",
        "scikit_image",
        "scikit_learn",
        "Shapely",
        "utm",
    ],
)
