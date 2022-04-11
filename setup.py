from setuptools import find_packages, setup

setup(
    name="tracit",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)
