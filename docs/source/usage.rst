.. _Usage:

General usage
================
The following is a demonstration on how you might want to use :strike:`tracit`.

Here we will be looking at a system with 2 planets with measurements from 2 photometers and 2 spectrographs.

Creating input structures
---------------------------
First we want to create two structures: one for the parameters and one for the data. We do this using the functions :py:func:`tracit.par_struct` and :py:func:`tracit.dat_struct`.

This would be done as in the example below. Here we have a 2-planet system with RVs from 2 different spectrographs, and light curves from 2 different photometers.

:: 

	import tracit

	n_planets = 2 #number of planets
	n_phot = 2 #number of photometric time series
	n_spec = 2 #number of spectroscopic time series
	
	par = tracit.par_struct(n_planets=n_planets,n_phot=n_phot,n_spec=n_spec)
	dat = tracit.dat_struct(n_phot=0,n_rvs=n_spec)

Setting the input
---------------------------
::
	dat['RV filename_1'] = 'rv.txt'
	dat['LC filename_1'] = 'lc.txt'
	dat['Fit RV_1'] = 1
	dat['Fit LC_1'] = 1

	saved_results = 1 #If you have saved results from a previous run you can set them like this
	if saved_resutls:
	    import pandas as pd
    	rdf = pd.read_csv('results_from_old_fit.csv')
    	tracit.update_pars(rdf,par,best_fit=False)  
   	else: #You will have to set all the values individually
   		par['P_b']['Value'] = 3 #days

	tracit.ini_data(dat)
	tracit.run_bus(par,dat,nproc)



Seeing the initial plots
---------------------------
Assuming you have set the relevant parameters and included the data files correctly in the .csv files from above, we might want to take a look at how our data compares to the models with our initial values for the parameters.

::

	n_proc = 1
	tracit.run_exp(n_proc)#ignore this. it's only to tell the plotting routine whether to use LaTeX formatting or not.

	inspect = 1
	if inspect:
		tracit.plot_lightcurve(par,dat)
		tracit.plot_orbit(pat,dat)


Fitting the data
---------------------------
We can find some good starting values for our parameters before we start doing an MCMC. This is done using `lmfit <https://lmfit.github.io/lmfit-py/>`_. In :strike:`tracit` this is done calling :py:func:`tracit.lmfitter`, which will return a `fit object`. We will turn into a `pandas.DataFrame <https://pandas.pydata.org/>`_ .

::

	par['FPs'] = ['P_b']
	lfit = 1
	if lfit:
		fit = tracit.lmfitter(pfile,dfile)
		rdf = tracit.fit_to_dict(fit)

After that we probably want to verify that our fit has improved things. Therefore, we plot our light curve and orbit again, but this time we update our dataframe with the updated (and hopefully better fitting) parameters.

::

	par['lam_b']['Prior'] = 'uni'
	par['lam_b']['Prior_vals'] = [0,2,-180,180]
	post_inspection = 1
	if post_inspection:
		tracit.update_pars(rdf,par,best_fit=False)
		tracit.plot_lightcurve(par,dat)
		tracit.plot_orbit(par,dat)	


Sampling the posterior
---------------------------
Finally, we can sample the posterior using `emcee <https://github.com/dfm/emcee>`_.

::

	mc = 1
	if mc:
		ndraws = 10000
		nwalkers = 50

		rdf = tracit.mcmc(par,dat,ndraws,nwalkers,corner=True,chains=True)
	
	post_inspection = 1
	if post_inspection:
		tracit.update_pars(rdf,par,best_fit=False)
		tracit.plot_lightcurve(par,dat,savefig=True)
		tracit.plot_orbit(par,dat,savefig=True)	
