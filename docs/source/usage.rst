.. _Usage:

Usage
================
The following is a demonstration on how you might want to use :del:`trace`.

Here we will be looking at a system with 2 planets with measurements from 2 photometers and 2 spectrographs.

Creating input (.csv) files
---------------------------
First we want to create two .csv files: one for the parameters and one for the data. We do this using the functions :py:func:`business.data_temp` and :py:func:`business.params_temp`.

This would be done as in the example below. Here we have a 2-planet system with RVs from 2 different spectrographs, and light curves from 2 different photometers.

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


Seeing the initial plots
---------------------------
Assuming you have set the relevant parameters and included the data files correctly in the .csv files from above, we might want to take a look at how our data compares to the models with our initial values for the parameters.

::

	import expose

	inspect = 1
	if inspect:
		expose.plot_lightcurve(pfile,dfile)
		expose.plot_orbit(pfile,dfile)


Fitting the data
---------------------------
We can find some good starting values for our parameters before we start doing an MCMC. This is done using `lmfit <https://lmfit.github.io/lmfit-py/>`_. In :del:`trace` this is done calling :py:func:`business.lmfitter`, which will return a `fit object`. We will turn into a `pandas <https://pandas.pydata.org/>`_ dataframe.

::

	lfit = 1
	if lfit:
		fit = business.lmfitter(pfile,dfile)
		rdf = business.fit_to_dict(fit)

After that we probably want to verify that our fit has improved things. Therefore, we plot our light curve and orbit again, but this time we provide our dataframe with the updated (and hopefully better fitting) parameters.

::

	post_inspection = 1
	if post_inspection:
		expose.plot_lightcurve(pfile,dfile,updated_pars=rdf)
		expose.plot_orbit(pfile,dfile,updated_pars=rdf)	


Sampling the posterior
---------------------------
Finally, 

::

	mc = 1
	if mc:
		ndraws = 10000
		nwalkers = 50

		rdf = businies.mcmc(pfile,dfile,ndraws,nwalkers,corner=True,chains=True)
	
	post_inspection = 1
	if inspection:
		expose.plot_lightcurve(pfile,dfile,updated_pars=rdf,savefig=True)
		expose.plot_orbit(pfile,dfile,updated_pars=rdf,savefig=True)	
