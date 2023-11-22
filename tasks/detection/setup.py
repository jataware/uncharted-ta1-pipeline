from setuptools import setup, find_packages

setup(
    name="points_and_lines",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Pillow",
        "scipy",
        "tqdm",
        "torch==2.0.1",
        "matplotlib",
        "pydantic",
        "torchvision==0.15.2",
        "scipy",
        "ultralytics",
        "opencv-python",
        "boto3",
        "parmap",
    ],
    python_requires=">=3.10.0",

    entry_points = {
    'console_scripts': [
        'coco_conversion = etl.coco_conversion:main',
    ],
}
)
