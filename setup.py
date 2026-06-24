from setuptools import setup, find_packages

setup(
    name="nebulamem",
    version="3.0.0",
    description="NebulaMem: a local-first, model-free, spreading-activation AI memory layer with lateral inhibition (no embedding model, no LLM)",
    author="NebulaMem",
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
