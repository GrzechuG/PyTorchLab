from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="PyTorchLab",
    version="1.0",
    description="Easy to use PyTorch training framework with a single fit method",
    license="GPL 2.0",
    long_description=long_description,
    author="Grzegorz Gajewski",
    author_email="...",
    packages=find_packages(),
    install_requires=["setuptools", "wheel"],
)
