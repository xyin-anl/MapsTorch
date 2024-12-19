from setuptools import setup, find_packages

# Read requirements from requirements.txt, with fallback
try:
    with open("requirements.txt") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    # Fallback requirements if file is not found
    requirements = [
        "h5py",
        "tqdm",
        "numpy<2.0.0",
        "scipy",
        "pandas",
        "matplotlib",
        "anywidget",
        "plotly",
        "marimo",
        "torch",
    ]

setup(
    name="mapstorch",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "maps_torch": ["reference/*"],
    },
    install_requires=requirements,
    author="Xiangyu Yin",
    author_email="xyin@anl.gov",
    description="A differentiable modeling package for automating X-ray fluorescence (XRF) analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/xyin-anl/MapsTorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
