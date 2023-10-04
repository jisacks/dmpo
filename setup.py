# Runs the installation
from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="dmpo",
    author="Jacob Sacks",
    author_email="jsacks6@cs.washington.edu",
    description="PyTorch code for the paper Deep Model Predictive Optimization",
    url="https://github.com/jisacks/dmpo",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.8",
    version='1.0.0',
    packages=find_packages(),
)