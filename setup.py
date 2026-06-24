from setuptools import setup, find_packages

setup(
    name="nebulamem",
    version="2.0.0",
    description="NebulaMem: A local-first, in-process, spreading-activation AI memory layer with lateral inhibition",
    author="Jetski Agent Pair",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
