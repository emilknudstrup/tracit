from setuptools import find_packages, setup

dependencies=''
with open("requirements.txt","r") as f:
	dependencies = f.read().splitlines()



setup(
	name="tracit",
	version='0.1.21',
	description='tracing exoplanets',
	url='https://github.com/emilknudstrup/tracit',
	author='Emil Knudstrup',
	author_email='emil@phys.au.dk',
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	include_package_data=True,
    classifiers = ["Programming Language :: Python :: 3"],
	install_requires = dependencies
	# install_requires=[
	# 	'numpy>=1.20.3',
	# 	'pandas>=1.3.2',
	# 	'matplotlib>=3.1.3',
	# 	'celerite>=0.3.1',
	# 	'batman-package>=2.4.8',
	# 'emcee>=3.0.2',
	# 'h5py>=2.10.0',
	# 'scipy>=1.7.1',
	# 'lmfit>=1.0.1',
	# 'arviz',
	# 'statsmodels>=0.12.0',
	# 'astropy',
	# 'tqdm'
	# ]
	
)

