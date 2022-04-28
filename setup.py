from setuptools import find_packages, setup

setup(
    name="tracit",
    version='0.1.4',
    description='tracing exoplanets',
    url='https://github.com/emilknudstrup/tracit',
    author='Emil Knudstrup',
    author_email='emil@phys.au.dk',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True
    
)
