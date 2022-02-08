.. _Usage:

Usage
================

Example usage of a system with 2 planets with measurements from 2 photometers and 2 spectrographs

Creating input (.csv) files
---------------------------
::

	import business

	pfile = 'par.csv'
	dfile = 'dat.csv'

	n_planets = 2 #number of planets
	n_phot = 2 #number of photometric time series
	n_spec = 2 #number of spectroscopic time series

	create_file = 1 #set to 0 to make sure we don't overwrite the files once they are created
	if create_file:
		business.params_temp(pfile,n_spec=n_spec,n_phot=n_phot,n_planets=n_planets)
		business.data_temp(dfile,n_spec=n_spec,n_phot=n_phot)



