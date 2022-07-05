#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

.. todo::
	* Make sure data isn't passed around with each evaluation 
	
	* Include fit to binaries - wrap ellc?

	* Find common orbital solution between batman/orbits
		i. best case: we are just solving Kepler's eq. twice
		ii. worst case: we are not getting consistent results between the two
		iii. use batman.TransitModel().get_true_anomaly()?
		iv. switch entirely to ellc?
	
	* Object oriented - just do it
	
	* Check imported packages, redundancy
	
	* Make it possible to look at shadow without giving a file for the RVs ``data_structure``.

	* Instead of looping over the number of spectroscopic systems (n_rv), it would be better to loop over the handles directly (Spec_1).
	* ^Same goes for the photometry.
	
	* Check `poly_pars` CCFs

'''
# =============================================================================
# tracit modules
# =============================================================================

from .dynamics import *
from .support import plot_autocorr, create_chains, create_corner, hpd, significantFormat
from .shady import absline_star, absline
from .priors import tgauss_prior, gauss_prior, flat_prior, tgauss_prior_dis, flat_prior_dis

# =============================================================================
# external modules
# =============================================================================

import emcee
import arviz as az
from multiprocessing import Pool
import celerite
import batman

import glob
import numpy as np
import pandas as pd
import h5py

from scipy.optimize import curve_fit
import lmfit
from scipy import interpolate
import scipy.signal as scisig
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import os


def run_bus(par,dat):
	'''Set global parameters.

	Initialize the :py:func:`structure.par_struct` and :py:func:`structure.dat_struct` dictionaries as global parameters.
	
	:param par: Name for the parameters dict from :py:func:`structure.par_struct`.
	:type par: dict

	:param dat: Name for the data dict from :py:func:`structure.dat_struct`.
	:type dat: dict

	.. note::
		Using global variable to prevent having to pickle and pass the data to the modules every time the code is called,
		see `emcee: pickling, data transfer and arguments <https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments>`_.
	'''

	global parameters
	parameters = par.copy()
	global data
	data = dat.copy()


def RM_path(path=None):
	'''Path to RM code.

	Create path to code by :cite:t:`Hirano2011`.

	:param path: Path. Only set if ``new_analytic7.exe`` has been moved. Default ``None``.
	:type path: str, optional

	'''	
	if not path:
		path = os.path.abspath(os.path.dirname(__file__))
	
	global mpath
	mpath = path


def Gauss(x,amp,mu,sig):
	'''Gaussian.

	:math:`f (x) = A \exp(-(x - \mu)^2/(2 \sigma^2)) \, .`
	
	:param x: :math:`x`-coordinates.
	:type x: array.

	:param amp: :math:`A`: amplitude.
	:type amp: float

	:param mu: :math:`\mu`: mean/location.
	:type mu: float

	:param sig: :math:`\sigma`: standard deviation/width.
	:type sig: float

	:returns: :math:`f (x)`.
	:rtype: array

	'''
	y = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	return y

#def inv2Gauss(x,amp1,amp2,mu1,mu2,sig1,sig2):
def inv2Gauss(x,amp1,amp2,sig1,sig2,mu2):
	'''Inverted double Gaussian.

	:py:func:`Gauss`.

	'''
	mu1 = 0.0
	#mu2 = 0.0
	g1 = Gauss(x,amp1,mu1,sig1)
	g2 = -1*Gauss(x,amp2,mu2,sig2)

	return g1+g2

#def start_vals(parameters,nwalkers,ndim):
def start_vals(nwalkers,ndim):
	pars = parameters['FPs']
	pos = []
	for ii in range(nwalkers):
		start = np.ndarray(ndim)
		for idx, par in enumerate(pars):
			pri = parameters[par]['Prior_vals']
			dist = parameters[par]['Distribution']
			assert dist in ['tgauss','uni'], print('{} is not a valid option for the starting distribution.'.format(dist))
			if dist == 'tgauss':
				start[idx] = tgauss_prior_dis(pri[0],pri[1],pri[2],pri[3])
			elif dist == 'uni':
				start[idx] = flat_prior_dis(np.random.uniform(),pri[2],pri[3])
		pos.append(start)
	return pos

def get_binned(x,y,binfactor=4,yerr=np.array([])):
	bins = np.linspace(np.amin(x),np.amax(x),int(len(x)/binfactor))
	dig = np.digitize(x,bins)
	binx = []
	biny = []
	binye = []
	for ii in range(1,len(bins)):
		grab = ii == dig
		xx = x[grab]
		binx.append(xx.mean())
		yy = y[grab]
		biny.append(yy.mean())
		if len(yerr):
			ye = yerr[grab]
			binye.append(ye.mean())
	if len(yerr):
		return np.asarray(binx), np.asarray(biny), np.asarray(binye)

	return np.asarray(binx), np.asarray(biny)



def lc_model(time,n_planet='b',n_phot=1,
	supersample_factor=30,exp_time=0.0208):
	'''Light curve model.


	Wrapper for batman \cite:p:`Kreidberg2015`.

	:pararm time: Timestamps.
	:type time: array

	:param n_planet: The planet for which to calculate the light curve. Default 'b'.
	:type planet: str, optional

	:param n_phot: The photometric system for which to calculate the light curve. Default 1.
	:type n_phot: int, optional	

	'''
	lclabel = 'LC{}'.format(n_phot)
	pllabel = '_{}'.format(n_planet)

	batpars = batman.TransitParams()
	per = parameters['P'+pllabel]['Value']
	batpars.per = per
	t0 = parameters['T0'+pllabel]['Value']
	batpars.t0 = 0.0#t0#parameters['T0'+pllabel]['Value']
	inc = parameters['inc'+pllabel]['Value']
	batpars.inc = inc
	ecc = parameters['e'+pllabel]['Value']
	batpars.ecc = ecc
	ww = parameters['w'+pllabel]['Value']
	batpars.w = ww
	rp = parameters['Rp_Rs'+pllabel]['Value']
	batpars.rp = rp
	ar = parameters['a_Rs'+pllabel]['Value']
	batpars.a = ar
	LD_law = parameters[lclabel+'_q1']['Unit']
	batpars.limb_dark = LD_law
	if LD_law == 'quadratic':
		q1, q2 = parameters[lclabel+'_q1']['Value'], parameters[lclabel+'_q2']['Value']
		qs = [q1,q2]
	elif LD_law == 'nonlinear':
		q1, q2  = parameters[lclabel+'_q1']['Value'], parameters[lclabel+'_q2']['Value']
		q3, q4  = parameters[lclabel+'_q3']['Value'], parameters[lclabel+'_q4']['Value']
		qs = [q1,q2,q3,q4]
	elif LD_law == 'uniform':
		qs = []
	batpars.u = qs

	tt = time - t0
	model = batman.TransitModel(batpars,tt,supersample_factor=supersample_factor,exp_time=exp_time)

	model_flux = model.light_curve(batpars)


	return model_flux

def rv_model(time,n_planet='b',n_rv=1,RM=False):
	'''Radial velocity model.

	:pararm time: Timestamps.
	:type time: array

	:param n_planet: The planet for which to calculate the light curve. Default 'b'.
	:type planet: str, optional	

	:param n_rv: The spectroscopic system for which to calculate the radial velocity curve. Default 1.
	:type n_rv: int, optional	

	:param RM: Whether to calculate the RM effect using the approach in :cite:t:`Hirano2011`. Default ``False``.
	:type RM: bool

	'''
	pllabel = '_{}'.format(n_planet)
	
	orbpars = OrbitalParams()
	per = parameters['P'+pllabel]['Value']
	orbpars.per = per
	orbpars.K = parameters['K'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']
	omega = omega%360
	orbpars.w = omega
	ecc = parameters['e'+pllabel]['Value']
	orbpars.ecc = ecc
	orbpars.RVsys = 0.0

	T0 = parameters['T0'+pllabel]['Value']

    ## With this you supply the mid-transit time 
    ## and then the time of periastron is calculated
    ## from S. R. Kane et al. (2009), PASP, 121, 886. DOI: 10.1086/648564
	if (ecc > 1e-5) & (omega != 90.):
		f = np.pi/2 - omega*np.pi/180.
		ew = 2*np.arctan(np.tan(f/2)*np.sqrt((1 - ecc)/(1 + ecc)))
		Tw = T0 - per/(2*np.pi)*(ew - ecc*np.sin(ew))
	else:
		Tw = T0

	orbpars.Tw = Tw

	if RM:
		label = 'RV{}'.format(n_rv)	
		stelpars = StellarParams()
		stelpars.vsini = parameters['vsini']['Value']
		stelpars.zeta = parameters['zeta']['Value']
		### HARD-CODED
		xi = parameters['xi']['Value']
		conv_par = np.sqrt(xi**2 + 1.1**2)#parameters['xi']['Value']
		stelpars.xi = conv_par
		
		orbpars.a = parameters['a_Rs'+pllabel]['Value']
		orbpars.Rp = parameters['Rp_Rs'+pllabel]['Value']
		orbpars.inc = parameters['inc'+pllabel]['Value']
		lam = parameters['lam'+pllabel]['Value']
		retrograde = False
		if retrograde:
			lam = lam%360
		orbpars.lam = lam
		LD_law =  parameters[label+'_q1']['Unit']
		fix = parameters[label+'_q1']['Fix']
		if fix != False:
			LDlabel = fix.split('_')[0]
		else:
			LDlabel = label
		if LD_law == 'quadratic':
			q1, q2 = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
			qs = [q1,q2]
		elif LD_law == 'nonlinear':
			q1, q2  = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
			q3, q4  = parameters[LDlabel+'_q3']['Value'], parameters[LDlabel+'_q4']['Value']
			qs = [q1,q2,q3,q4]
		elif LD_law == 'uniform':
			qs = []
		orbpars.cs = qs

		calcRV = get_RV(time,orbpars,RM=RM,stelpars=stelpars,mpath=mpath)
	else:
		calcRV = get_RV(time,orbpars)
	return calcRV

def ls_model2(time,start_grid,ring_grid,vel_grid,
	mu,mu_grid,mu_mean,rad_disk,vels,
	n_planet='b',n_rv=1,oot=False):
	'''The line distortion model.


	.. note::
		The instrumental broadening, :math:`\sigma_\mathrm{PSF}`, is added to :math:`\\xi` in quadrature, i.e., :math:`\sqrt{\\xi^2 + \sigma_\mathrm{PSF}^2}`.	
	'''

	pllabel = '_{}'.format(n_planet)
	
	## Planet parameters
	per = parameters['P'+pllabel]['Value']
	T0 = parameters['T0'+pllabel]['Value']

	inc = parameters['inc'+pllabel]['Value']*np.pi/180.
	a_Rs = parameters['a_Rs'+pllabel]['Value']
	b = a_Rs*np.cos(inc)
	rp = parameters['Rp_Rs'+pllabel]['Value']
	ecc = parameters['e'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']

    ## With this you supply the mid-transit time 
    ## and then the time of periastron is calculated
    ## from S. R. Kane et al. (2009), PASP, 121, 886. DOI: 10.1086/648564
	if (ecc > 1e-5) & (omega != 90.):
		f = np.pi/2 - omega*np.pi/180.
		ew = 2*np.arctan(np.tan(f/2)*np.sqrt((1 - ecc)/(1 + ecc)))
		Tw = T0 - per/(2*np.pi)*(ew - ecc*np.sin(ew))
	else:
		Tw = T0

	T0 = Tw
	
	lam = parameters['lam'+pllabel]['Value']

	omega = omega%360
	omega *= np.pi/180.
	retrograde = False
	if retrograde:
		lam = lam%360
	lam *= np.pi/180.

	## Stellar parameters
	label = 'RV{}'.format(n_rv)	
	
	LD_law =  parameters[label+'_q1']['Unit']
	# fix = parameters[label+'_q1']['Fix']
	# if fix != False:
	# 	LDlabel = fix.split('_')[0]
	# else:
	# 	LDlabel = label
	LDlabel = label


	if LD_law == 'quadratic':
		q1, q2 = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
		qs = [q1,q2]
	elif LD_law == 'nonlinear':
		q1, q2  = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
		q3, q4  = parameters[LDlabel+'_q3']['Value'], parameters[LDlabel+'_q4']['Value']
		qs = [q1,q2,q3,q4]
	elif LD_law == 'uniform':
		qs = []


	vsini = parameters['vsini']['Value']
	zeta = parameters['zeta']['Value']
	xi = parameters['xi']['Value']
	psf = data['PSF_{}'.format(n_rv)]

	conv_par = np.sqrt(xi**2 + psf**2)#parameters['xi']['Value']

	## If we are only fitting OOT CCFS
	if oot:
		vel_1d, line_conv, lum = absline_star(start_grid,vel_grid,ring_grid,
								mu,mu_mean,vsini,conv_par,zeta,vels,
								cs=qs
								)
		line_oot = np.sum(line_conv,axis=0)
		area = np.trapz(line_oot,vel_1d)
		line_oot_norm = line_oot/area
		return vel_1d, line_oot_norm, lum

	## Make shadow                   
	vel_1d, line_conv, line_transit, planet_rings, lum, index_error = absline(
																	start_grid,vel_grid,ring_grid,
																	mu,mu_mean,mu_grid,
																	vsini,conv_par,zeta,vels,
																	cs=qs,radius=rad_disk,
																	times=time,
																	Tw=T0,per=per,
																	Rp_Rs=rp,a_Rs=a_Rs,inc=inc,
																	ecc=ecc,w=omega,lam=lam)
	
	line_oot = np.sum(line_conv,axis=0)
	area = np.trapz(line_oot,vel_1d)
	## Normalize such that out-of-transit lines have heights of unity
	line_oot_norm = line_oot/area
	line_transit = line_transit/area#np.max(line_oot)
	## Shadow
	#shadow = line_oot_norm - line_transit
	

	#return vel_1d, shadow, line_oot_norm, planet_rings, lum, index_error
	return vel_1d, line_transit, line_oot_norm, planet_rings, lum, index_error

def ls_model(time,start_grid,ring_grid,vel_grid,
	mu,mu_grid,mu_mean,rad_disk,vels,
	n_planet='b',n_rv=1,oot=False):
	'''The line distortion model.


	.. note::
		The instrumental broadening, :math:`\sigma_\mathrm{PSF}`, is added to :math:`\\xi` in quadrature, i.e., :math:`\sqrt{\\xi^2 + \sigma_\mathrm{PSF}^2}`.	
	'''

	pllabel = '_{}'.format(n_planet)
	
	## Planet parameters
	per = parameters['P'+pllabel]['Value']
	T0 = parameters['T0'+pllabel]['Value']

	inc = parameters['inc'+pllabel]['Value']*np.pi/180.
	a_Rs = parameters['a_Rs'+pllabel]['Value']
	b = a_Rs*np.cos(inc)
	rp = parameters['Rp_Rs'+pllabel]['Value']
	ecc = parameters['e'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']

    ## With this you supply the mid-transit time 
    ## and then the time of periastron is calculated
    ## from S. R. Kane et al. (2009), PASP, 121, 886. DOI: 10.1086/648564
	if (ecc > 1e-5) & (omega != 90.):
		f = np.pi/2 - omega*np.pi/180.
		ew = 2*np.arctan(np.tan(f/2)*np.sqrt((1 - ecc)/(1 + ecc)))
		Tw = T0 - per/(2*np.pi)*(ew - ecc*np.sin(ew))
	else:
		Tw = T0

	T0 = Tw
	
	lam = parameters['lam'+pllabel]['Value']

	omega = omega%360
	omega *= np.pi/180.
	retrograde = False
	if retrograde:
		lam = lam%360
	lam *= np.pi/180.

	## Stellar parameters
	label = 'RV{}'.format(n_rv)	
	
	LD_law =  parameters[label+'_q1']['Unit']
	# fix = parameters[label+'_q1']['Fix']
	# if fix != False:
	# 	LDlabel = fix.split('_')[0]
	# else:
	# 	LDlabel = label
	LDlabel = label


	if LD_law == 'quadratic':
		q1, q2 = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
		qs = [q1,q2]
	elif LD_law == 'nonlinear':
		q1, q2  = parameters[LDlabel+'_q1']['Value'], parameters[LDlabel+'_q2']['Value']
		q3, q4  = parameters[LDlabel+'_q3']['Value'], parameters[LDlabel+'_q4']['Value']
		qs = [q1,q2,q3,q4]
	elif LD_law == 'uniform':
		qs = []


	vsini = parameters['vsini']['Value']
	zeta = parameters['zeta']['Value']
	xi = parameters['xi']['Value']
	psf = data['PSF_{}'.format(n_rv)]

	conv_par = np.sqrt(xi**2 + psf**2)#parameters['xi']['Value']

	## If we are only fitting OOT CCFS
	if oot:
		vel_1d, line_conv, lum = absline_star(start_grid,vel_grid,ring_grid,
								mu,mu_mean,vsini,conv_par,zeta,vels,
								cs=qs
			)
		line_oot = np.sum(line_conv,axis=0)
		area = np.trapz(line_oot,vel_1d)
		line_oot_norm = line_oot/area

		inverted = False
		if inverted:
			inv_gauss = Gauss(vel_1d,0.1,0.0,6.0)
			line_oot_norm -= inv_gauss	
		#print(inv_gauss,line_oot_norm)
		return vel_1d, line_oot_norm, lum

	## Make shadow                   
	vel_1d, line_conv, line_transit, planet_rings, lum, index_error = absline(
																	start_grid,vel_grid,ring_grid,
																	mu,mu_mean,mu_grid,
																	vsini,conv_par,zeta,vels,
																	cs=qs,radius=rad_disk,
																	times=time,
																	Tw=T0,per=per,
																	Rp_Rs=rp,a_Rs=a_Rs,inc=inc,
																	ecc=ecc,w=omega,lam=lam)
	
	line_oot = np.sum(line_conv,axis=0)
	area = np.trapz(line_oot,vel_1d)
	## Normalize such that out-of-transit lines have heights of unity
	line_oot_norm = line_oot/area
	line_transit = line_transit/area#np.max(line_oot)
	## Shadow
	shadow = line_oot_norm - line_transit
	
	return vel_1d, shadow, line_oot_norm, planet_rings, lum, index_error

def localRV_model(time,n_planet='b'):
	'''Subplanetary velocities.

	Model for the slope of the velocities covered by the transiting planet across the stellar disk.

	Using :py:func:`tracit.dynamics.get_rel_vsini`. 

	:pararm time: Timestamps.
	:type time: array

	:param n_planet: The planet for which to calculate the slope. Default 'b'.
	:type n_planet: str, optional

	'''
	pllabel = '_{}'.format(n_planet)
	 
	## Planet parameters
	per = parameters['P'+pllabel]['Value']
	T0 = parameters['T0'+pllabel]['Value']
	
	inc = parameters['inc'+pllabel]['Value']*np.pi/180.
	a_Rs = parameters['a_Rs'+pllabel]['Value']
	rp = parameters['Rp_Rs'+pllabel]['Value']
	ecc = parameters['e'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']
	lam = parameters['lam'+pllabel]['Value']

	omega = omega%360
	omega *= np.pi/180.

	lam *= np.pi/180.
	pp = time2phase(time,per,T0)*per*24

	tau = total_duration(per,rp,a_Rs,inc,ecc,omega)
	x1 = -24*tau/2
	x2 = 24*tau/2
	
	y1, y2 = get_rel_vsini(a_Rs*np.cos(inc),lam)
	y1 *= -1
	a = (y2 - y1)/(x2 - x1)
	b = y1 - x1*a


	return pp*a + b


def chi2(ycal,yobs,sigy):
	'''Chi squared.

	.. math:: \chi^2 = \sum_i (O_i - C_i)^2/\sigma_i^2 \, ,
	
	where :math:`N` indicates the total number of data points from photometry and RVs. :math:`C_i` represents the model corresponding to the observed data point :math:`O_i`. :math:`\sigma_i` represents the uncertainty for the :math:`i` th data point. 

	:return: :math:`\chi^2`

	'''
	return np.sum((ycal-yobs)**2/sigy**2)

def lnlike(ycal,yobs,sigy):
	'''Log likelihood.
	
	.. math:: \log \mathcal{L} = - \chi^2/2 + \log 2 \pi \sigma_i^2 

	where :math:`\chi^2` is from :py:func:`chi2`, and :math:`\sigma_i` represents the uncertainty for the math:`i` th data point. 
	'''
	nom = chi2(ycal,yobs,sigy)
	den = np.sum(np.log(2*np.pi*sigy**2))
	return -0.5*(den + nom)

def lnprob(positions):
	'''Log probability.

	The log probability is defined as

	.. math:: \log \mathcal{L} + \sum_{j} \log \mathcal{P}_{j}\, ,

	where :math:`\log \mathcal{L}` is the likelihood from :py:func:`lnlike`, and :math:`\mathcal{P}_j` is the prior on the :math:`j` th parameter.	
	
	'''
	log_prob = 0.0
	chisq = 0.0

	pars = parameters['FPs']
	for idx, par in enumerate(pars):
		val = positions[idx]
		parameters[par]['Value'] = val
		pri = parameters[par]['Prior_vals']
		ptype = parameters[par]['Prior']
		
		pval, sigma, lower, upper = pri[0], pri[1], pri[2], pri[3]

		if ptype == 'uni':
			prob = flat_prior(val,lower,upper)
		elif ptype == 'gauss':
			prob = gauss_prior(val,pval,sigma)
		elif ptype == 'tgauss':
			prob = tgauss_prior(val,pval,sigma,lower,upper)
		elif ptype == 'jeff':
			prob = jeff_prior(val,lower,upper)
		elif ptype == 'beta':
			prob = beta_prior(val)
		if prob != 0:
			log_prob += np.log(prob)
		else:
			return -np.inf, -np.inf

	m_fps = len(pars)
	n_dps = 0

	constraints = parameters['ECs']
	pls = parameters['Planets']
	for pl in pls:
		if ('ecosw_{}'.format(pl) in pars) and ('esinw_{}'.format(pl) in pars):
			ecosw = parameters['ecosw_{}'.format(pl)]['Value']
			esinw = parameters['esinw_{}'.format(pl)]['Value']
			esinw = abs(esinw)
			ecc = ecosw**2 + esinw**2
			#if ecc > 1.0:
			parameters['e_{}'.format(pl)]['Value'] = ecc
			omega = np.arctan2(esinw,ecosw)*180./np.pi
			parameters['w_{}'.format(pl)]['Value'] = omega%360
		aR = parameters['a_Rs_'+pl]['Value']
		rp = parameters['Rp_Rs_'+pl]['Value']
		if parameters['e_{}'.format(pl)]['Value'] > (1.0 - (1+rp)/aR):
			return -np.inf, -np.inf
		if ('cosi_{}'.format(pl) in pars):
			cosi = parameters['cosi_{}'.format(pl)]['Value']
			parameters['inc_{}'.format(pl)]['Value'] = np.arccos(cosi)*180./np.pi
		
		if ('T41_{}'.format(pl) in constraints):
			pri_41 = parameters['T41_{}'.format(pl)]['Prior_vals']
			pval_41, sigma_41 = pri_41[0], pri_41[1]

			per = parameters['P_'+pl]['Value']
			rp = parameters['Rp_Rs_'+pl]['Value']
			aR = parameters['a_Rs_'+pl]['Value']
			inc = parameters['inc_'+pl]['Value']*np.pi/180.
			ecc = parameters['e_'+pl]['Value']
			ww = parameters['w_'+pl]['Value']*np.pi/180.
			b = aR*np.cos(inc)*(1 - ecc**2)/(1 + ecc*np.sin(ww))
			
			ecc = parameters['e_'+pl]['Value']
			omega = (parameters['w_'+pl]['Value']%360)*np.pi/180.

			t41 = per/np.pi * np.arcsin( np.sqrt( ((1 + rp)**2 - b**2))/(np.sin(inc)*aR) )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(omega))

			prob = gauss_prior(t41*24.,pval_41,sigma_41)
			if prob != 0:
				log_prob += np.log(prob)
			else:
				return -np.inf, -np.inf

			if ('T21_{}'.format(pl) in constraints):
				pri_21 = parameters['T21_{}'.format(pl)]['Prior_vals']
				pval_21, sigma_21 = pri_21[0], pri_21[1]

				t32 = per/np.pi * np.arcsin( np.sqrt( ((1 - rp)**2 - b**2))/(np.sin(inc)*aR) )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(omega))
				t21 = (t41 - t32)*0.5
				prob = gauss_prior(t21*24,pval_21,sigma_21)
				if prob != 0:
					log_prob += np.log(prob)
				else:
					return -np.inf, -np.inf

	if 'rho_s' in constraints:
		rho = parameters['rho_s']['Prior_vals']
		pval, sigma = rho[0], rho[1]

		for pl in pls:
			aR = parameters['a_Rs_'+pl]['Value']
			per = parameters['P_'+pl]['Value']
			circ = 3*np.pi*aR**3/((per*24*3600)**2*constants.grav)

			ecc = parameters['e_'+pl]['Value']
			omega = parameters['w_'+pl]['Value']*np.pi/180.
			ecc_nom = (1 - ecc**2)**(3/2)
			ecc_den = (1 + ecc*np.sin(omega))**3
			
			val = circ*ecc_nom/ecc_den
			prob = gauss_prior(val,pval,sigma)
			if prob != 0:
				log_prob += np.log(prob)
			else:
				return -np.inf, -np.inf


	LD_lincombs = parameters['LinCombs']
	for LDlc in LD_lincombs:
		LD1 = LDlc + '1'
		LD2 = LDlc + '2'
		LDsum = parameters[LDlc+'_sum']['Value']
		LDdiff = parameters[LDlc+'_diff']['Value']
		q1 = 0.5*(LDsum + LDdiff)
		q2 = 0.5*(LDsum - LDdiff)
		parameters[LD1]['Value'] = q1
		parameters[LD2]['Value'] = q2
		if q1 < 0.0 or q2 < 0.0:
			return -np.inf, -np.inf

	n_phot, n_rv, n_ls, n_sl = data['LCs'], data['RVs'], data['LSs'], data['SLs']
	for nn in range(1,n_phot+1):
		if data['Fit LC_{}'.format(nn)]:
			arr = data['LC_{}'.format(nn)]
			time, flux, flux_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()


			ofactor = data['OF LC_{}'.format(nn)]
			exp = data['Exp. LC_{}'.format(nn)]
			log_jitter = parameters['LCsigma_{}'.format(nn)]['Value']
			deltamag = parameters['LCblend_{}'.format(nn)]['Value']
			sigma = np.sqrt(flux_err**2 + np.exp(log_jitter)**2)
			
			flux_m = np.ones(len(flux))



			trend = data['Detrend LC_{}'.format(nn)]


			if (trend == 'poly') or (trend == True): in_transit = np.array([],dtype=np.int)

			for pl in pls:
				try:
					t0n = parameters['Phot_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass

				per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']

				flux_pl = lc_model(time,n_planet=pl,n_phot=nn,
							supersample_factor=ofactor,exp_time=exp)
				if deltamag > 0.0:
					dilution = 10**(-deltamag/2.5)
					flux_pl = flux_pl/(1 + dilution) + dilution/(1+dilution)

				flux_m -= (1 - flux_pl)

				if (trend == 'poly') or (trend == True):
					per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']
					ph = time2phase(time,per,t0)*per*24
					aR = parameters['a_Rs_{}'.format(pl)]['Value']
					rp = parameters['Rp_Rs_{}'.format(pl)]['Value']
					inc = parameters['inc_{}'.format(pl)]['Value']
					ecc = parameters['e_{}'.format(pl)]['Value']
					ww = parameters['w_{}'.format(pl)]['Value']
					dur = total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
					
					indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
					in_transit = np.append(in_transit,indxs)				


			
			#chi2scale = data['Chi2 LC_{}'.format(nn)]
			
			n_dps += len(flux)


			if (trend == 'poly') or (trend == True):
				deg_w = data['Poly LC_{}'.format(nn)]
				
				in_transit.sort()
				in_transit = np.unique(in_transit)
				dgaps = np.where(np.diff(time[in_transit]) > 1)[0]
				start = 0
				temp_fl = flux - flux_m + 1
				for dgap in dgaps:
					idxs = in_transit[start:int(dgap+1)]
					t = time[idxs]
					tfl = temp_fl[idxs]
					poly_pars = np.polyfit(t,tfl,deg_w)
					slope = np.zeros(len(t))
					for dd, pp in enumerate(poly_pars):
						slope += pp*t**(deg_w-dd)
					flux[idxs] /= slope

					start = int(dgap + 1)

				idxs = in_transit[start:]
				t = time[idxs]
				tfl = temp_fl[idxs]
				poly_pars = np.polyfit(t,tfl,deg_w)
				slope = np.zeros(len(t))
				for dd, pp in enumerate(poly_pars):
					slope += pp*t**(deg_w-dd)
				flux[idxs] /= slope

				log_prob += lnlike(flux_m,flux,sigma)
				chisq += chi2(flux_m,flux,sigma)
			elif (trend == 'savitsky'):

				temp_fl = flux - flux_m + 1
				window = data['FW LC_{}'.format(nn)]

				gap = 0.5
				dls = np.where(np.diff(time) > gap)[0]
				sav_arr = np.array([])
				start = 0
				for dl in dls:
					
					sav_fil = scisig.savgol_filter(temp_fl[start:int(dl+1)],window,2)
					sav_arr = np.append(sav_arr,sav_fil)
					
					start = int(dl + 1)
				sav_fil = scisig.savgol_filter(temp_fl[start:],window,2)
				sav_arr = np.append(sav_arr,sav_fil)
				flux /= sav_arr

				log_prob += lnlike(flux_m,flux,sigma)
				chisq += chi2(flux_m,flux,sigma)

			elif data['GP LC_{}'.format(nn)]:
				gp = data['LC_{} GP'.format(nn)]
				res_flux = flux - flux_m

				gp_type = data['GP type LC_{}'.format(nn)]
				if gp_type == 'SHO':
					log_S0 = parameters['LC_{}_log_S0'.format(nn)]['Value']
					log_Q = parameters['LC_{}_log_Q'.format(nn)]['Value']
					log_w0 = parameters['LC_{}_log_w0'.format(nn)]['Value']
				
					gp_list = [log_S0,log_Q,log_w0]
				else:
					loga = parameters['LC_{}_GP_log_a'.format(nn)]['Value']
					logc = parameters['LC_{}_GP_log_c'.format(nn)]['Value']
					gp_list = [loga,logc]

				gp.set_parameter_vector(np.array(gp_list))
				gp.compute(time,sigma)
				lprob = gp.log_likelihood(res_flux)
				log_prob += lprob#lnlike(flux_m,flux,sigma)

				chisq += -2*lprob - np.sum(np.log(2*np.pi*sigma**2))#chi2(flux_m,flux,sigma)
			else:
				log_prob += lnlike(flux_m,flux,sigma)
				chisq += chi2(flux_m,flux,sigma)

	rv_times, rvs, ervs = np.array([]), np.array([]), np.array([])  
	model_rvs = np.array([])
	add_drift = False
	rv_gps = []
	rv_idxs = np.array([])
	for nn in range(1,n_rv+1):
		if data['Fit RV_{}'.format(nn)]:
			add_drift = True
			arr = data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			
			fix = parameters['RVsys_{}'.format(nn)]['Fix']
			if fix != False:
				v0 = parameters[fix]['Value']
			else:
				v0 = parameters['RVsys_{}'.format(nn)]['Value']
			rv -= v0

			log_jitter = parameters['RVsigma_{}'.format(nn)]['Value']
			jitter = log_jitter
			sigma = np.sqrt(rv_err**2 + jitter**2)
			#chi2scale = data['Chi2 RV_{}'.format(nn)]

			rv_times, rvs, ervs = np.append(rv_times,time), np.append(rvs,rv), np.append(ervs,sigma)  
			
			calc_RM = data['RM RV_{}'.format(nn)]
			
			rv_m = np.zeros(len(rv))
			for pl in pls:
				try:
					t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass
				per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']
				rv_pl = rv_model(time,n_planet=pl,n_rv=nn,RM=calc_RM)
				rv_m += rv_pl
			model_rvs = np.append(model_rvs,rv_m)
			n_dps += len(rvs)

			rv_gps.append(data['GP RV_{}'.format(nn)])
			rv_idxs = np.append(rv_idxs,np.ones(len(time))*nn)

	if add_drift:# n_rv > 0:
		aa = np.array([parameters['a{}'.format(ii)]['Value'] for ii in range(1,3)])
		idx = np.argmin(rv_times)
		zp, off = rv_times[idx], rvs[idx]

		drift = aa[1]*(rv_times-zp)**2 + aa[0]*(rv_times-zp)# + off
		
		model_rvs += drift

		if any(rv_gps):
			for nn in range(1,n_rv+1): 
				idxs = rv_idxs == nn
				if data['GP RV_{}'.format(nn)]:

					gp = data['RV_{} GP'.format(nn)]

					res_rv = rvs[idxs] - model_rvs[idxs]

					gp_type = data['GP type RV_{}'.format(nn)]
					if gp_type == 'SHO':
						log_S0 = parameters['RV_{}_log_S0'.format(nn)]['Value']
						log_Q = parameters['RV_{}_log_Q'.format(nn)]['Value']
						log_w0 = parameters['RV_{}_log_w0'.format(nn)]['Value']
					
						gp_list = [log_S0,log_Q,log_w0]
					else:
						loga = parameters['RV_{}_GP_log_a'.format(nn)]['Value']
						logc = parameters['RV_{}_GP_log_c'.format(nn)]['Value']
						gp_list = [loga,logc]

					gp.set_parameter_vector(np.array(gp_list))
					gp.compute(rv_times[idxs],ervs[idxs])
					lprob = gp.log_likelihood(res_rv)
					log_prob += lprob#lnlike(flux_m,flux,sigma)

					chisq += -2*lprob - np.sum(np.log(2*np.pi*ervs[idxs]**2))#chi2(flux_m,flux,sigma)
				else:
					log_prob += lnlike(model_rvs[idxs],rvs[idxs],ervs[idxs])
					chisq += chi2(model_rvs[idxs],rvs[idxs],ervs[idxs])
		else:
			log_prob += lnlike(model_rvs,rvs,ervs)
			chisq += chi2(model_rvs,rvs,ervs)

	avg_ccf = np.array([])
	for nn in range(1,n_ls+1):
		if data['Fit LS_{}'.format(nn)]:
			shadow_data = data['LS_{}'.format(nn)]

			times = []
			for key in shadow_data.keys():
				try:
					times.append(float(key))
				except ValueError:
					pass
			times = np.asarray(times)
			ss = np.argsort(times)
			times = times[ss]


			#nRVs = dynamics.get_RV(time,op)
			v0 = parameters['RVsys_{}'.format(nn)]['Value']
			rv_m = np.zeros(len(times))
			for pl in pls:
				try:
					t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass
				p2, t02 = parameters['P_{}'.format(pl)]['Value'], parameters['T0_{}'.format(pl)]['Value'] 
				rv_pl = rv_model(times,n_planet=pl,n_rv=nn,RM=False)
				rv_m += rv_pl
			rv_m += v0


			idxs = [ii for ii in range(len(times))]
			oots = data['idxs_{}'.format(nn)]

			its = [ii for ii in idxs if ii not in oots]	


			## Hard coded -- modify
			## currently assumes planet b and spectroscopic system 1
			pl = 'b'
			try:
				t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
				parameters['T0_{}'.format(pl)]['Value'] = t0n			
			except KeyError:
				pass

			#P, T0 = parameters['P_{}'.format(pl)]['Value'], parameters['T0_{}'.format(pl)]['Value'] 
			## Resolution of velocity grid
			vel_res = data['Velocity_resolution_{}'.format(nn)]
			vels = np.array([])			
			## Range without a bump in the CCFs
			no_bump = data['No_bump_{}'.format(nn)]
			span = data['Velocity_range_{}'.format(nn)]
			assert span > no_bump, print('\n ### \n The range of the velocity grid must be larger than the specified range with no bump in the CCF.\n Range of velocity grid is from +/-{} km/s, and the no bump region isin the interval m +/-{} km/s \n ### \n '.format(span,no_bump))
			vels = np.arange(-span,span,vel_res)

			resol = data['Resolution_{}'.format(nn)]
			start_grid = data['Start_grid_{}'.format(nn)]
			ring_grid = data['Ring_grid_{}'.format(nn)]
			vel_grid = data['Velocity_{}'.format(nn)]
			mu = data['mu_{}'.format(nn)]
			mu_grid = data['mu_grid_{}'.format(nn)]
			mu_mean = data['mu_mean_{}'.format(nn)]			
			only_oot = data['Only_OOT_{}'.format(nn)]			
			fit_oot = data['OOT_{}'.format(nn)]			
			if only_oot:
				vel_model, model_ccf, _ = ls_model2(
					times[oots],start_grid,ring_grid,
					vel_grid,mu,mu_grid,mu_mean,resol,vels,
					oot=only_oot,n_rv=nn
					)
			else:
				vel_model, model_ccf_transit, model_ccf, darks, oot_lum, index_error = ls_model2(
					times,start_grid,ring_grid,
					vel_grid,mu,mu_grid,mu_mean,resol,vels,
					n_rv=nn,
					)
				if index_error:
					return -np.inf, -np.inf
				bright = np.sum(oot_lum)

			
			avg_ccf = np.zeros(len(vels))
			oot_ccfs = np.zeros(shape=(len(vels),len(oots)))
			## GP or not?
			use_gp = data['GP LS_{}'.format(nn)]
			if use_gp:
				loga = parameters['LS_{}_GP_log_a'.format(nn)]['Value']
				logc = parameters['LS_{}_GP_log_c'.format(nn)]['Value']
				gp = data['LS_{} GP'.format(nn)]
				gp.set_parameter_vector(np.array([loga,logc]))


				## Create average out-of-transit CCF
				## Used to create shadow for in-transit CCFs
				## Shift CCFs to star rest frame
				## and detrend CCFs
				oot_sd = []
				for ii, idx in enumerate(oots):
					time = times[idx]
					vel = shadow_data[time]['vel'].copy() - rv_m[idx]*1e-3
					# if not ii:
					# 	## Interpolate to grid in stellar restframe
					# 	# vel_min, vel_max = min(vel), max(vel)
					# 	# span  = (vel_max - vel_min)
					# 	# vels = np.arange(vel_min+span/10,vel_max-span/10,vel_res)
					# 	vels = np.arange(-span,span,vel_res)
					# 	avg_ccf = np.zeros(len(vels))
					# 	oot_ccfs = np.zeros(shape=(len(vels),len(oots)))

					no_peak = (vel > no_bump) | (vel < -no_bump)

					ccf = shadow_data[time]['ccf'].copy()
					
					ccf -= np.median(ccf[no_peak])


					area = np.trapz(ccf,vel)
					ccf /= abs(area)

					ccf_int = interpolate.interp1d(vel,ccf,kind='cubic',fill_value='extrapolate')
					nccf = ccf_int(vels)
					
					no_peak = (vels > no_bump) | (vels < -no_bump)
					oot_sd.append(np.std(nccf[no_peak]))
						

					oot_ccfs[:,ii] = nccf
					avg_ccf += nccf

				avg_ccf /= len(oots)
				jitter = parameters['LSsigma_{}'.format(nn)]['Value']
				jitter = np.exp(jitter)

				unc = np.ones(len(vels))*np.sqrt((np.mean(oot_sd)**2 + jitter**2))
				gp.compute(vels,unc)
				model_int = interpolate.interp1d(vel_model,model_ccf,kind='cubic',fill_value='extrapolate')
				newline = model_int(vels)		
				mean, var = gp.predict(avg_ccf  - newline, vels, return_var=True)
				avg_ccf -= mean

				if fit_oot:
						## Here we simply fit our average out-of-transit CCF
						## to an out-of-transit model CCF
						## IF we opted to do so
						res_cff = avg_ccf - newline

						lprob = gp.log_likelihood(res_cff)
						log_prob += lprob#lnlike(flux_m,flux,sigma)

						chisq += -2*lprob - np.sum(np.log(2*np.pi*unc**2))#chi2(flux_m,flux,sigma)

						#chisq += chi2(newline,avg_ccf,unc)
						#log_prob += lnlike(newline,avg_ccf,unc)

						n_dps += len(vel)


			else:
				## Create average out-of-transit CCF
				## Used to create shadow for in-transit CCFs
				## Shift CCFs to star rest frame
				## and detrend CCFs
				oot_sd = []
				for ii, idx in enumerate(oots):
					time = times[idx]
					vel = shadow_data[time]['vel'].copy() - rv_m[idx]*1e-3
					# if not ii:
					# 	## Interpolate to grid in stellar restframe
					# 	# vel_min, vel_max = min(vel), max(vel)
					# 	# span  = (vel_max - vel_min)
					# 	# vels = np.arange(vel_min+span/10,vel_max-span/10,vel_res)
					# 	vels = np.arange(-span,span,vel_res)
					# 	avg_ccf = np.zeros(len(vels))
					# 	oot_ccfs = np.zeros(shape=(len(vels),len(oots)))

					#vels[:,idx] = vel
					no_peak = (vel > no_bump) | (vel < -no_bump)
					

					ccf = shadow_data[time]['ccf'].copy()
					poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
					ccf -= vel*poly_pars[0] + poly_pars[1]

					area = np.trapz(ccf,vel)

					ccf /= abs(area)
					oot_sd.append(np.std(ccf[no_peak]))

					ccf_int = interpolate.interp1d(vel,ccf,kind='cubic',fill_value='extrapolate')
					nccf = ccf_int(vels)

					oot_ccfs[:,ii] = nccf
					avg_ccf += nccf

				avg_ccf /= len(oots)
				#avg_vel /= len(oots)

				model_int = interpolate.interp1d(vel_model,model_ccf,kind='cubic',fill_value='extrapolate')
				newline = model_int(vels)
				sd = np.mean(oot_sd)
				unc = np.ones(len(vels))*sd
				chi2scale_shadow = data['Chi2 LS_{}'.format(nn)]
				#chi2scale_oot = data['Chi2 OOT_{}'.format(nn)]
				unc *= chi2scale_shadow
			
				if fit_oot:
					## Here we simply fit our average out-of-transit CCF
					## to an out-of-transit model CCF
					## IF we opted to do so

					chisq += chi2(newline,avg_ccf,unc)
					log_prob += lnlike(newline,avg_ccf,unc)

					n_dps += len(vel)

		
			if not only_oot:
				#chi2scale_shadow = 5.0

				## Again shift CCFs to star rest frame
				## and detrend CCFs
				## Compare to shadow model
				jitter = 0
				for ii, idx in enumerate(its):
					#arr = data[time]
					time = times[idx]
					vel = shadow_data[time]['vel'].copy() - rv_m[idx]*1e-3
					#vels[:,idx] = vel
					no_peak = (vel > no_bump) | (vel < -no_bump)
						
					ccf = shadow_data[time]['ccf'].copy()
					if use_gp:
						ccf -= np.median(ccf[no_peak])
					else:
						poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
						ccf -= vel*poly_pars[0] + poly_pars[1]
					
					area = np.trapz(ccf,vel)
					ccf /= area
					
					

					ccf *= darks[idx]/bright#blc[ii]		

					ccf_int = interpolate.interp1d(vel,ccf,kind='cubic',fill_value='extrapolate')
					nccf = ccf_int(vels)
					

					no_peak = (vels > no_bump) | (vels < -no_bump)
					sd = np.std(nccf[no_peak])
					if use_gp:			
						unc = np.ones(len(vels))*np.sqrt(sd**2 + jitter**2)
						nccf -= mean
					else:
						unc = np.ones(len(vels))*sd
						unc *= chi2scale_shadow

					
					#shadow = nccf
					shadow = avg_ccf - nccf

					#ff = interpolate.interp1d(vel_model,shadow_model[idx],kind='cubic',fill_value='extrapolate')
					#ishadow = ff(vels)

					model_to_obs = interpolate.interp1d(vel_model,model_ccf_transit[idx],kind='cubic',fill_value='extrapolate')
					ishadow = newline - model_to_obs(vels)


					# vv,ss = get_binned(vel,shadow)

					# no_peak = (vv > 15) | (vv < -15)
					# sd = np.std(ss[no_peak])
					# poly_pars = np.polyfit(vv[no_peak],ss[no_peak],1)
					# nvv,nss = get_binned(vel,ishadow)

					# unc = np.ones(len(vv))*np.sqrt(sd**2 + jitter**2)

					chisq += chi2(shadow,ishadow,unc)
					log_prob += lnlike(shadow,ishadow,unc)
					
					n_dps += len(shadow)
					
					#chisq += chi2(ccf,ishadow,unc)
					#log_prob += lnlike(ccf,ishadow,unc)
					#n_dps += len(ccf)
					
					# chisq += chi2(ss,nss + vv*poly_pars[0] + poly_pars[1],unc)
					# log_prob += lnlike(ss,nss + vv*poly_pars[0] + poly_pars[1],unc)

					# n_dps += len(ss)

	for nn in range(1,n_sl+1):
		if data['Fit SL_{}'.format(nn)]:

			slope_data = data['SL_{}'.format(nn)]
			#errs = data['SL_{}_errs'.format(nn)]

			times = []
			for key in slope_data.keys():
				try:
					times.append(float(key))
				except ValueError:
					pass
			times = np.asarray(times)
			ss = np.argsort(times)
			times = times[ss]

			v0 = parameters['RVsys_{}'.format(nn)]['Value']
			rv_m = np.zeros(len(times))
			for pl in pls:
				#rv_pl = rv_model(parameters,time,n_planet=pl,n_rv=nn,RM=calc_RM)
				rv_pl = rv_model(times,n_planet=pl,n_rv=nn,RM=False)
				rv_m += rv_pl
			rv_m += v0


			for pl in pls:
				model_slope = localRV_model(times,n_planet=pl)				

				darks = lc_model(times,n_planet=pl,n_phot=1)

				idxs = [ii for ii in range(len(times))]
				oots = data['idxs_{}'.format(nn)]

				its = [ii for ii in idxs if ii not in oots]	


				nvel = len(slope_data[times[0]]['vel'])
				vels = np.zeros(shape=(nvel,len(times)))
				oot_ccfs = np.zeros(shape=(nvel,len(oots)))
				oot_sd = []
				if not len(avg_ccf):
					vel_res = data['Velocity_resolution_{}'.format(nn)]
					vels = np.array([])
					
					no_bump = data['No_bump_{}'.format(nn)]
					for ii, idx in enumerate(oots):
						time = times[idx]
						vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
						if not ii:
							vel_min, vel_max = min(vel), max(vel)
							span  = (vel_max - vel_min)
							vels = np.arange(vel_min+span/10,vel_max-span/10,vel_res)
							avg_ccf = np.zeros(len(vels))
							oot_ccfs = np.zeros(shape=(len(vels),len(oots)))

						no_peak = (vel > no_bump) | (vel < -no_bump)
						

						ccf = slope_data[time]['ccf']
						poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
						ccf -= vel*poly_pars[0] + poly_pars[1]
						
						area = np.trapz(ccf,vel)

						ccf /= abs(area)
						oot_sd.append(np.std(ccf[no_peak]))

						ccf_int = interpolate.interp1d(vel,ccf,kind='cubic',fill_value='extrapolate')
						nccf = ccf_int(vels)

						# vv,cc = get_binned(vels,nccf)
						# no_peak_b = (vv > no_bump) | (vv < -no_bump)
						# oot_sd_b.append(np.std(cc[no_peak_b]))
							

						oot_ccfs[:,ii] = nccf
						avg_ccf += nccf

					# avg_ccf = np.zeros(nvel)

					# for ii, idx in enumerate(oots):
					# 	time = times[idx]
					# 	vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
					# 	vels[:,idx] = vel
					# 	no_peak = (vel > 15) | (vel < -15)
						

					# 	ccf = slope_data[time]['ccf']
					# 	poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
					# 	ccf -= vel*poly_pars[0] + poly_pars[1]
					# 	area = np.trapz(ccf,vel)
					# 	ccf /= area	



					# 	oot_ccfs[:,ii] = ccf
					# 	avg_ccf += ccf
					avg_ccf /= len(oots)

				try:
					t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass

				T0 = parameters['T0_{}'.format(pl)]['Value']
				P = parameters['P_{}'.format(pl)]['Value']
				ecc = parameters['e_{}'.format(pl)]['Value']
				omega = parameters['w_{}'.format(pl)]['Value']
				ww = parameters['w_{}'.format(pl)]['Value']*np.pi/180
				ar = parameters['a_Rs_{}'.format(pl)]['Value']
				inc = parameters['inc_{}'.format(pl)]['Value']*np.pi/180
				lam = parameters['lam_{}'.format(pl)]['Value']*np.pi/180
				vsini = parameters['vsini']['Value'] 
				#P, T0 = parameters['P_{}'.format(pl)]['Value'], parameters['T0_{}'.format(pl)]['Value'] 
				#ar, inc = parameters['a_Rs_{}'.format(pl)]['Value'], parameters['inc_{}'.format(pl)]['Value']*np.pi/180.
				rp = parameters['Rp_Rs_{}'.format(pl)]['Value']
				
				## With this you supply the mid-transit time 
				## and then the time of periastron is calculated
				## from S. R. Kane et al. (2009), PASP, 121, 886. DOI: 10.1086/648564
				if (ecc > 1e-5) & (omega != 90.):
					f = np.pi/2 - ww
					ew = 2*np.arctan(np.tan(f/2)*np.sqrt((1 - ecc)/(1 + ecc)))
					Tw = T0 - P/(2*np.pi)*(ew - ecc*np.sin(ew))
				else:
					Tw = T0

				#fit_params = lmfit.Parameters()
				cos_f, sin_f = true_anomaly(times[its], Tw, ecc, P, ww)
				xxs, yys = xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
				rvs = np.array([])
				errs = np.array([])
				for ii, idx in enumerate(its):
					time = times[idx]
					vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
					#vels[:,idx] = vel
					no_peak = (vel > no_bump) | (vel < -no_bump)

					xx = xxs[idx]
					#cos_f, sin_f = true_anomaly(time, T0, ecc, P, ww)
					#xx, yy = xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
					
					#ccf = np.zeros(len(vel))
					#shadow_arr = shadow_data[time]['ccf']
					#ccf = 1 - shadow_arr/np.median(shadow_arr[no_peak])

					ccf = slope_data[time]['ccf']
					poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
					
					ccf -= vel*poly_pars[0] + poly_pars[1]
					area = np.trapz(ccf,vel)
					ccf /= area
					

					ccf *= darks[idx]#/bright#blc[ii]		

					sd = np.std(ccf[no_peak])
					ccf_int = interpolate.interp1d(vel,ccf,kind='cubic',fill_value='extrapolate')
					nccf = ccf_int(vels)
					shadow = avg_ccf - nccf

					#peak = np.where((vel > -15) & (vel < 15))
					peak = np.where((vels > (xx*vsini - vsini/2)) & (vels < (xx*vsini + vsini/2)))
					#peak = np.where((vels > -no_bump) & (vels < no_bump))
					try:
						midx = np.argmax(shadow[peak])
					except ValueError:
						return -np.inf, -np.inf
					amp, mu1 = shadow[peak][midx], vels[peak][midx]# get max value of CCF and location
					shadow = shadow/amp

					try:
						gau_par, pcov = curve_fit(Gauss,vels,shadow,p0=[1.0,xx*vsini,1.0])
						perr = np.sqrt(np.diag(pcov))
						if any(np.isnan(perr)):
							return -np.inf, -np.inf
						rv = gau_par[1]
						std = perr[1]
					except RuntimeError:
						return -np.inf, -np.inf

					rvs = np.append(rvs,rv)
					errs = np.append(errs,std)



				per = parameters['P_'+pl]['Value']
				T0 = parameters['T0_'+pl]['Value']
				slope = localRV_model(times[its])

				
				vsini = parameters['vsini']['Value']
				rv_scale = rvs/vsini
				erv_scale = errs/vsini


				pp = time2phase(times[its],per,T0)*24*per

				chi2scale = data['Chi2 SL_{}'.format(nn)]
				erv_scale *= chi2scale


				chisq += chi2(rv_scale,slope,erv_scale)
				log_prob += lnlike(rv_scale,slope,erv_scale)
				n_dps += len(rv_scale)


	if np.isnan(log_prob) or np.isnan(chisq):
		return -np.inf, -np.inf
	else:
		return log_prob, chisq/(n_dps - m_fps)


#def mcmc(param_fname,data_fname,maxdraws,nwalkers,
def mcmc(par,dat,maxdraws,nwalkers,
		save_results = True, results_filename='results.csv',
		sample_filename='samples.h5',reset=True,burn=0.5,
		plot_convergence=True,save_samples=False,
		corner=True,chains=False,nproc=1,
		stop_converged=True,post_name='posteriors.npy',
		triangles=[],moves=None,**kwargs):
	'''Markov Chain Monte Carlo.

	Wrapper for `emcee <https://github.com/dfm/emcee>`_ :cite:p:`emcee`.

	:param par: Parameters.
	:type par: :py:class:`tracit.structure.par_struct`

	:param dat: Data.
	:type dat: :py:class:`tracit.structure.dat_struct`

	:param save_results: Whether to save the results in a .csv file. Default ``True``.
	:type save_results: bool

	:param results_filename: Name of the .csv file of the results. Default `results.csv`.
	:type results_filename: str

	:param sample_filename: Name of the .hdf5 file for the samples. Default `samples.h5`.
	:type sample_filename: str

	:param save_samples: Whether to save the samples. Default ``True``.
	:type save_samples: bool

	:param corner: Whether to create a `corner <https://github.com/dfm/corner.py>`_ plot :cite:p:`corner`. Default ``True``.
	:type corner: bool

	:param chains: Whether to plot the chains/samples. Default ``False``.
	:type chains: bool

	:param nproc: Number of processors for multiprocessing. Default 1.
	:type nproc: int

	:param stop_converged: Whether to stop when the MCMC has converged following `emcee/Saving & monitoring progress <https://emcee.readthedocs.io/en/stable/tutorials/monitor/>`_. Default ``True``.
	:type stop_converged: bool

	:param post_name: Name of the "flat" samples, which is a ``numpy.recarray``. Default `posteriors.npy`.
	:type post_name: str

	:param triangles: List of tuples containing parameters for which to create correlation plots for. For example, ``[('vsini','lam_b'),('inc_b','a_Rs_b')]`` will create two corner plots. Default ``[]``.
	:type triangles: list

	:param moves: Algorithm for updating the coordinates of the walkers. See `emcee/Moves <https://emcee.readthedocs.io/en/stable/user/moves/#moves-user>`_ and `emcee/The Ensemble Sampler <https://emcee.readthedocs.io/en/stable/user/sampler/>`_. Default ``None``, equivalent to :py:class:`emcee.moves.StretchMove`.
	:type moves: list

	.. note::
		A very poor fit might result in ``ValueError``: math domain error from :py:func:`support`. In that case check the boundaries of the priors.

	'''

	run_bus(par,dat)
	RM_path()

	fit_params = list(set(parameters['FPs']))
	ndim = len(fit_params)
	assert ndim > 0, print('Error: No parameters to fit...')
	for fp in fit_params:
		print('Fitting {}.'.format(fp))
	print('Fitting {} parameters in total.'.format(ndim))

	backend = emcee.backends.HDFBackend(sample_filename)
	if reset: backend.reset(nwalkers,ndim)
	if not save_samples: backend = None

	if np.log10(maxdraws) > 5.3: kk = 20000
	elif np.log10(maxdraws) > 5.0: kk = 10000
	elif np.log10(maxdraws) > 4.0: kk = 1000
	else: kk = 100

	with Pool(nproc) as pool:
		sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
				pool=pool,backend=backend,
				moves=moves)  

		cdraw = sampler.iteration # current draw
		ndraws = maxdraws - cdraw
		print('\nMaximum number of draws is {}.'.format(maxdraws))
		print('Starting from {} draws.'.format(cdraw))
		print('{} draws remaining.\n\n'.format(ndraws))
		
		## Starting coordinates for walkers
		#if reset: coords = start_vals(parameters,nwalkers,ndim)
		if reset: coords = start_vals(nwalkers,ndim)
		else: coords = sampler.get_last_sample().coords
		## We'll track how the average autocorrelation time estimate changes
		index = 0
		autocorr = np.empty(ndraws)
		## This will be useful to testing convergence
		old_tau = np.inf

		for sample in sampler.sample(coords,iterations=ndraws,progress=True):
			## Only check convergence every kk steps
			if sampler.iteration % kk: continue

			## Compute the autocorrelation time so far
			## Using tol=0 means that we'll always get an estimate even
			## if it isn't trustworthy
			tau = sampler.get_autocorr_time(tol=0)
			autocorr[index] = np.mean(tau)
			index += 1
			## Check convergence
			converged = np.all(tau * 100 < sampler.iteration)
			converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
			if converged: 
				print('MCMC converged after {} iterations.'.format(sampler.iteration))
				break
			
			old_tau = tau
	
	res_df = check_convergence(parameters,data,
		sampler=sampler,post_name=post_name,
		plot_corner=corner,plot_chains=chains,per_burn=burn,
		save_df=save_results,results_filename=results_filename,
		triangles=triangles,**kwargs)

	## Plot autocorrelation time
	## Converged if: 
	## - chain is longer than 100 times the estimated autocorrelation time 
	## - & this estimate changed by less than 1%.
	if plot_convergence:
		plot_autocorr(autocorr,index,kk,**kwargs)
	
	return res_df


def check_convergence(parameters_local,data_local,filename=None, 
					sampler=None, post_name='posteriors.npy',
					plot_chains=True, plot_corner=True,chain_ival=5,
					save_df=True,results_filename='results.csv',
					n_auto=4,per_burn=None,plot_priors=True,
					triangles=[],**kwargs):
	'''Check convergence and diagnostics.

	Function that checks the convergence of the MCMC and provides some diagnostics. 
	It also outputs a .csv file with the results from the MCMC, and can call plotting routines to generate corner and chain plots.

	:param parameters_local,: Parameters, but only parsed locally.
	:type parameters_local,: :py:class:`tracit.structure.par_struct`

	:param data_local: Data, but only parsed locally.
	:type data_local: :py:class:`tracit.structure.dat_struct`

	:param filename: Name of the .hdf5 file for the samples. 
	:type filename: str

	:param sampler: The `emcee <https://github.com/dfm/emcee>`_ sampler.
	:type sampler: :py:class:`emcee.EnsembleSampler`

	:param post_name: Name of the "flat" samples, which is a ``numpy.recarray``. Default `posteriors.npy`.
	:type post_name: str

	:param save_df: Whether to save the .csv file for the results. Default ``True``.
	:type save_df: bool

	:param results_filename: Name of the .csv file for the results. Default `results.csv`.
	:type results_filename: str

	:param plot_chains: Whether toplot the chains. Default ``True``.
	:type plot_chains: bool

	:param plot_corner: Whether to create a `corner <https://github.com/dfm/corner.py>`_ plot :cite:p:`corner`. Default ``True``.
	:type plot_corner: bool

	:param chain_ival: Number of chains to put in same plot.
	:type chain_ival: int

	:param n_auto: Number of autocorrelation times used to estimate burn-in. See `emcee/Autocorrelation analysis & convergence <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_. Default 4.
	:type n_auto: int

	:param per_burn: Percentage of burn-in, 0.5 means half the samples are tossed. Default ``None``.
	:type per_burn: float

	:param plot_priors: Whether to plot the priors in the corner plot. Default ``True``.
	:type plot_priors: bool

	:param triangles: List of tuples containing parameters for which to create correlation plots for. For example, ``[('vsini','lam_b'),('inc_b','a_Rs_b')]`` will create two corner plots. Default ``[]``.
	:type triangles: list


	'''
	if filename:
		reader = emcee.backends.HDFBackend(filename)	
	elif sampler:
		reader = sampler
		del sampler
	elif filename == None and sampler == None:
		raise FileNotFoundError('You need to either provide a filename for the stored sampler \
			or the sampler directly.')

	# if pars == None:
	# 	try:
	# 		params_structure(param_fname)
	# 	except FileNotFoundError as err:
	# 		print(err.args)
	# 		print('Either provide the name of the .csv-file with the fitting parameters_local \
	# 			or give the parameter structure directly.')

	# if dat == None:
	# 	try:
	# 		data_structure(data_fname)
	# 	except FileNotFoundError as err:
	# 		print(err.args)
	# 		print('Either provide the name of the .csv-file for the data \
	# 			or give the data structure directly.')

	fit_params = parameters_local['FPs']


	steps, nwalkers, ndim = reader.get_chain().shape
	tau = reader.get_autocorr_time(tol=0)

	if not per_burn:
		burnin = int(n_auto * np.max(tau))
		
		last_taun_samples = int(steps-burnin)
		last_flat = reader.get_chain(discard=burnin,flat=True)[last_taun_samples:,:]
		
		comp = {}
		for ii, fp in enumerate(fit_params):
			val = np.median(last_flat[:,ii])
			hpds = hpd(last_flat[:,ii],0.68)
			lower = val - hpds[0]
			upper = hpds[1] - val
			sigma = np.mean([lower,upper])
			comp[fp] = [val,sigma]
		
		while burnin < int(0.9*steps):
			ini_flat = reader.get_chain(discard=burnin,flat=True)[:burnin,:]
			outpars = []
			for ii, fp in enumerate(fit_params):
				val, sigma = comp[fp][0], comp[fp][1]
				first_vals = ini_flat[:,ii]
				dist = abs(first_vals-val)/sigma
				lab = parameters_local[fp]['Label']
				if any(dist > 10):
					outpars.append(True)
				else:
					outpars.append(False)
			if not any(outpars):
				break
			n_auto += 1
			burnin = int(n_auto * np.max(tau))
		if burnin > int(0.9*steps):
			print('Warning: 90% of the samples will be discarded')
			print('if this is alright set per_burn to 0.9 (or above).')
		del ini_flat, last_flat
	else:
		burnin = int(per_burn*steps)



	labels = []
	hands = []
	for ii, fp in enumerate(fit_params):
		lab = parameters_local[fp]['Label']
		labels.append(lab)	
		hands.append(fp)	

	raw_samples = reader.get_chain()
	## plot the trace/chains
	if plot_chains:
		create_chains(raw_samples,labels=labels,savefig=True,ival=chain_ival,**kwargs)
	del raw_samples

	thin = int(0.5 * np.min(tau))
	samples = reader.get_chain(discard=burnin)
	reshaped_samples = samples.reshape(samples.shape[1],samples.shape[0],samples.shape[2])
	conv_samples = az.convert_to_dataset(reshaped_samples)
	del reshaped_samples
	rhats = az.rhat(conv_samples).x.values
	del conv_samples
	flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)	
	log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
	chi2_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
	del reader

	pls = parameters_local['Planets']
	for pl in pls:
		if ('ecosw_{}'.format(pl) in fit_params) and ('esinw_{}'.format(pl) in fit_params):
			idx_ecosw = fit_params.index('ecosw_{}'.format(pl))
			ecosw = flat_samples[:,idx_ecosw]
			idx_esinw = fit_params.index('esinw_{}'.format(pl))
			esinw = flat_samples[:,idx_esinw]
			esinw = abs(esinw)
			flat_samples[:,idx_esinw] = esinw
			ecc = ecosw**2 + esinw**2
			omega = np.arctan2(esinw,ecosw)*180./np.pi
			flat_samples = np.concatenate(
				(flat_samples, ecc[:, None], omega[:, None]), axis=1)
			labels += [r'$e \rm _{}$'.format(pl), r'$\omega \rm _{} \ (^\circ)$'.format(pl)]
			hands += ['e_{}'.format(pl), 'w_{}'.format(pl)]

			# val_ecc = np.median(ecc)
			# hpds_ecc = hpd(ecc,0.68)
			# lower_ecc = val_ecc - hpds_ecc[0]
			# upper_ecc = hpds_ecc[1] - val_ecc
			# results['e_{}'.format(pl)] = [val_ecc,lower_ecc,upper_ecc]

			# omega = omega%360
			# val_omega = np.median(omega)
			# hpds_omega = hpd(omega,0.68)
			# lower_omega = val_omega - hpds_omega[0]
			# upper_omega = hpds_omega[1] - val_omega
			# results['w_{}'.format(pl)] = [val_omega,lower_omega,upper_omega]
		elif ('e_{}'.format(pl) in fit_params):
			idx_ecc = fit_params.index('e_{}'.format(pl))
			ecc = flat_samples[:,idx_ecc]
			idx_ww = fit_params.index('w_{}'.format(pl))
			omega = flat_samples[:,idx_ww]
		else:
			ecc = parameters_local['e_'+pl]['Value']
			omega = parameters_local['w_'+pl]['Value']

		if ('cosi_{}'.format(pl) in fit_params):
			ecc_fac = (1 - ecc**2)/(1 + ecc*np.sin(omega))
			idx_cosi = fit_params.index('cosi_{}'.format(pl))
			cosi = flat_samples[:,idx_cosi]
			inc = np.arccos(cosi)*180./np.pi
			flat_samples = np.concatenate(
				(flat_samples, inc[:, None]), axis=1)
			labels += [r'$i \rm _{} \ (^\circ)$'.format(pl)]
			hands += ['inc_{}'.format(pl)]

			if ('a_Rs_{}'.format(pl) in fit_params):
				idx_ar = fit_params.index('a_Rs_{}'.format(pl))
				aR = flat_samples[:,idx_ar]
				flat_samples = np.concatenate(
					(flat_samples, cosi[:, None]*aR[:,None]), axis=1)
			else:
				aR = parameters_local['a_Rs_'+pl]['Value']
				flat_samples = np.concatenate(
					(flat_samples, cosi[:, None]*aR), axis=1)

			labels += [r'$b \rm _{}$'.format(pl)]
			hands += ['b_{}'.format(pl)]

		
		if ('lam_{}'.format(pl) in fit_params):
			idx_lam = fit_params.index('lam_{}'.format(pl))
			retrograde = False
			if retrograde:
				flat_samples[:,idx_lam] = flat_samples[:,idx_lam]%360


		if ('a_Rs_{}'.format(pl) in fit_params) and ('inc_{}'.format(pl) in fit_params or 'cosi_{}'.format(pl) in fit_params):
			req_pars = ['P_','Rp_Rs_','a_Rs_','inc_','e_','w_']

			#for pl in pls:
			pars_pl = {
				'P_{}'.format(pl) : parameters_local['P_{}'.format(pl)]['Value'],
				'Rp_Rs_{}'.format(pl) : parameters_local['Rp_Rs_{}'.format(pl)]['Value'],
				'a_Rs_{}'.format(pl) : parameters_local['a_Rs_{}'.format(pl)]['Value'],
				'inc_{}'.format(pl) : parameters_local['inc_{}'.format(pl)]['Value'],
				'e_{}'.format(pl) : parameters_local['e_{}'.format(pl)]['Value'],
				'w_{}'.format(pl) : parameters_local['w_{}'.format(pl)]['Value']
			}

			create_samples = False
			for rpar in req_pars:
				try:
					par_idx = hands.index('{}{}'.format(rpar,pl))
					pars_pl[rpar+pl] = flat_samples[:,par_idx]
					create_samples = True
				except ValueError:
					pass

			if not create_samples: continue

			per = pars_pl['P_'+pl]
			rp = pars_pl['Rp_Rs_'+pl]
			aR = pars_pl['a_Rs_'+pl]
			inc = pars_pl['inc_'+pl]*np.pi/180.

			ecc = pars_pl['e_'+pl]
			omega = pars_pl['w_'+pl]*np.pi/180.
			b = aR*np.cos(inc)*(1 - ecc**2)/(1 + ecc*np.sin(omega))

			t41 = per/np.pi * np.arcsin( np.sqrt( ((1 + rp)**2 - b**2))/(np.sin(inc)*aR) )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(omega))*24
			t32 = per/np.pi * np.arcsin( np.sqrt( ((1 - rp)**2 - b**2))/(np.sin(inc)*aR) )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(omega))*24
			t21 = (t41 - t32)*0.5
	
			if any(np.isnan(t41)): continue
			if any(np.isnan(t32)): continue

			#t41 *= 24
			flat_samples = np.concatenate(
				(flat_samples, t41[:, None]), axis=1)
			labels += [r'$T \rm _{41,'+pl+'} (hours)$']
			hands += ['T41_{}'.format(pl)]
			

			#t21 *= 24
			flat_samples = np.concatenate(
				(flat_samples, t21[:, None]), axis=1)

			labels += [r'$T \rm _{21,'+pl+'} (hours)$']
			hands += ['T21_{}'.format(pl)]


	
	#n_phot, n_rv, n_ls, n_sl = data['LCs'], data['RVs'], data['LSs'], data['SLs']
	n_phot, n_rv = data_local['LCs'], data_local['RVs']
	
	for nn in range(1,n_rv+1):
		if ('RV{}_q_sum'.format(nn) in fit_params):
			idx_qs = fit_params.index('RV{}_q_sum'.format(nn))
			qs = flat_samples[:,idx_qs]
			qd = parameters_local['RV{}_q1'.format(nn)]['Prior_vals'][0] - parameters_local['RV{}_q2'.format(nn)]['Prior_vals'][0]
			q1 = 0.5*(qs + qd)
			q2 = 0.5*(qs - qd)
			flat_samples = np.concatenate(
				(flat_samples, q1[:, None]), axis=1)
			flat_samples = np.concatenate(
				(flat_samples, q2[:, None]), axis=1)
			labels += [r'$q_1: \ \rm RV{}$'.format(nn)]
			labels += [r'$q_2: \ \rm RV{}$'.format(nn)]
			hands += ['RV{}_q1'.format(nn)]
			hands += ['RV{}_q2'.format(nn)]

	for nn in range(1,n_phot+1):
		if ('LC{}_q_sum'.format(nn) in fit_params):
			idx_qs = fit_params.index('LC{}_q_sum'.format(nn))
			qs = flat_samples[:,idx_qs]
			#qd = parameters_local['LC{}_q1'.format(nn)]['Value'] - parameters_local['LC{}_q2'.format(nn)]['Value']
			qd = parameters_local['LC{}_q1'.format(nn)]['Prior_vals'][0] - parameters_local['LC{}_q2'.format(nn)]['Prior_vals'][0]
			q1 = 0.5*(qs + qd)
			q2 = 0.5*(qs - qd)
			flat_samples = np.concatenate(
				(flat_samples, q1[:, None]), axis=1)
			flat_samples = np.concatenate(
				(flat_samples, q2[:, None]), axis=1)
			labels += [r'$q_1: \ \rm LC{}$'.format(nn)]
			labels += [r'$q_2: \ \rm LC{}$'.format(nn)]
			hands += ['LC{}_q1'.format(nn)]
			hands += ['LC{}_q2'.format(nn)]


	del samples

	labels += [r'$\ln \mathcal{L}$', r'$\chi^2$']
	hands += ['lnL','chi2']

	## print information
	print('Burn-in applied: {0}'.format(burnin))
	print('Chains are thinned by: {0}'.format(thin))

	all_samples = np.concatenate(
		(flat_samples, log_prob_samples[:, None], chi2_samples[:, None]), axis=1)
	del flat_samples
	dtypes = [(hand,float) for hand in hands]
	recarr = np.recarray((all_samples.shape[0],),dtype=dtypes)
	for ii, hand in enumerate(hands):
		recarr[hand] = all_samples[:,ii]

	np.save(post_name,recarr)


	priors = {}
	fps = parameters_local['FPs']
	for i in range(ndim): priors[i] = ['none',0.0,0.0,0.0,0.0]
	for i, fp in enumerate(fps): priors[i] = [parameters_local[fp]['Prior'],parameters_local[fp]['Prior_vals'][0],parameters_local[fp]['Prior_vals'][1],parameters_local[fp]['Prior_vals'][2],parameters_local[fp]['Prior_vals'][3]]
	results = {}
	results ['Parameter'] = ['Label','Median','Lower','Upper','Best-fit','Mode','Prior','Location','Width','Lower','Upper','Rhat']
	qs = []
	value_formats = []
	ndim = all_samples.shape[-1]
	for i in range(ndim):
		#mode = ss.gaussian_kde(all_samples[:,i]).mode[0]
		kdee = KDE(all_samples[:,i])
		kdee.fit(kernel='gau', bw='scott', fft=True, gridsize=1000)
		ksupport, kdensity = kdee.support, kdee.density
		mode_idx = np.argmax(kdensity)
		mode = ksupport[mode_idx]

		val = np.median(all_samples[:,i])
		bounds = hpd(all_samples[:,i], 0.68)

		qs.append([bounds[0],val,bounds[1]])
		val, low, up = significantFormat(qs[i][1],qs[i][1]-qs[i][0],qs[i][2]-qs[i][1])
		try:
			mode, _, _ = significantFormat(mode,mode-qs[i][0],qs[i][2]-mode)
		except ValueError:
			print('ERROR:')
			print('Failed to estimate mode.')
			print('Probably a poor fit, check boundaries of priors.')
		label = labels[i][:-1] + '=' + str(val) + '_{-' + str(low) + '}^{+' + str(up) + '}'
		value_formats.append(label)

		best_idx = np.argmax(log_prob_samples)
		best_fit = all_samples[best_idx,i]
		best_val, _, _ = significantFormat(best_fit,float(low),float(up))

		hand = hands[i]
		try:
			pri, mu, std, aa, bb = priors[i][0],priors[i][1],priors[i][2],priors[i][3],priors[i][4]
		except KeyError:
			pri, mu, std, aa, bb = 'Derived','Derived','Derived','Derived','Derived'
	
		try:
			rhat = rhats[i]
		except IndexError:
			rhat = 'Derived'
		results[hand] = [labels[i],val,low,up,best_val,mode,pri,mu,std,aa,bb,rhat]
	
	## pandas DataFrame of results
	res_df = pd.DataFrame(results)
	if save_df: res_df.to_csv(results_filename,index=False)
	
	
	## corner plot of samples
	if plot_corner:
		if plot_priors:
			#pass
			priors = {}
			ndim = all_samples.shape[1]
			fps = parameters_local['FPs']
			for i in range(ndim): priors[i] = ['none',0.0,0.0,0.0,0.0]
			for i, fp in enumerate(fps): priors[i] = [parameters_local[fp]['Prior'],parameters_local[fp]['Prior_vals'][0],parameters_local[fp]['Prior_vals'][1],parameters_local[fp]['Prior_vals'][2],parameters[fp]['Prior_vals'][3]]
		else:
			priors = None
		create_corner(all_samples,labels=labels,truths=None,savefig=True,priors=priors,quantiles=qs,diag_titles=value_formats,**kwargs)

		del all_samples
		for triangle in triangles:	
			if priors: 
				subpriors = {}
			else:
				subpriors = None
			sublabels = []
			subvalue_formats = []
			subarr = np.zeros(shape=(recarr.shape[0],len(triangle)))
			for i, hand in enumerate(hands):
				for j, tri in enumerate(triangle):
					if tri == hand:
						sublabels.append(labels[i])
						subarr[:,j] = recarr[tri]
						subvalue_formats.append(value_formats[i])
						if priors: subpriors[j] = priors[i]
			fname = 'corner'
			for tri in triangle: fname += '_' + tri
			fname += '.pdf'
			create_corner(subarr,labels=sublabels,fname=fname,truths=None,savefig=True,priors=subpriors,quantiles=qs,diag_titles=subvalue_formats,**kwargs)


	return res_df





def residuals(params,parameters,data):
	n_phot, n_rv = data['LCs'], data['RVs']
	n_ls = data['LSs']
	pls = parameters['Planets']

	res = np.array([])
	chi2 = np.array([])
	for param in params:
		parameters[param]['Value'] = params[param].value
	
	for param in parameters['FPs']:
		prior = parameters[param]['Prior']
		if prior in ['gauss','tgauss']:
			draw = parameters[param]['Value']
			mu = parameters[param]['Prior_vals'][0]
			sig = parameters[param]['Prior_vals'][1]
			res = np.append(res,(draw-mu)/sig)

	for nn in range(1,n_phot+1):
		if data['Fit LC_{}'.format(nn)]:
			arr = data['LC_{}'.format(nn)]
			time, flux, flux_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()

			ofactor = data['OF LC_{}'.format(nn)]
			exp = data['Exp. LC_{}'.format(nn)]
			log_jitter = parameters['LCsigma_{}'.format(nn)]['Value']
			#jitter = np.exp(log_jitter)
			jitter = log_jitter
			sigma = np.sqrt(flux_err**2 + jitter**2)
			flux_m = np.ones(len(flux))
			for pl in pls:
				#flux_pl = lc_model(parameters,time,n_planet=pl,n_phot=n_phot,
				flux_pl = lc_model(time,n_planet=pl,n_phot=n_phot,
								supersample_factor=ofactor,exp_time=exp)
				flux_m -= (1 - flux_pl)
			
			lc_res = (flux_m - flux)/flux_err
			res = np.append(res,lc_res)

	for nn in range(1,n_rv+1):
		if data['Fit RV_{}'.format(nn)]:
			arr = data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			
			fix = parameters['RVsys_{}'.format(nn)]['Fix']
			if fix != False:
				v0 = parameters[fix]['Value']
			else:
				v0 = parameters['RVsys_{}'.format(nn)]['Value']
			rv -= v0

			log_jitter = parameters['RVsigma_{}'.format(nn)]['Value']
			#jitter = np.exp(log_jitter)
			jitter = log_jitter
			sigma = np.sqrt(rv_err**2 + jitter**2)

			calc_RM = data['RM RV_{}'.format(nn)]
			
			rv_m = np.zeros(len(rv))
			for pl in pls:
				#rv_pl = rv_model(parameters,time,n_planet=pl,n_rv=nn,RM=calc_RM)
				rv_pl = rv_model(time,n_planet=pl,n_rv=nn,RM=calc_RM)
				rv_m += rv_pl

			rv_res = (rv_m - rv)/sigma
			res = np.append(res,rv_res)
	for nn in range(1,n_ls+1):
		if data['Fit LS_{}'.format(nn)]:
			time = data['LS_time_{}'.format(nn)]
			vel_o = data['LS_vel_{}'.format(nn)]
			shadow = data['LS_shadow_{}'.format(nn)]
			unc = data['LS_sigma_{}'.format(nn)]

			resol = data['Resolution_{}'.format(nn)]
			start_grid = data['Start_grid_{}'.format(nn)]
			ring_grid = data['Ring_grid_{}'.format(nn)]
			vel_grid = data['Velocity_{}'.format(nn)]
			mu = data['mu_{}'.format(nn)]
			mu_grid = data['mu_grid_{}'.format(nn)]
			mu_mean = data['mu_mean_{}'.format(nn)]			
			vel_model, shadow_model, model_ccf = ls_model(
				#parameters,time,start_grid,ring_grid,
				time,start_grid,ring_grid,
				vel_grid,mu,mu_grid,mu_mean,resol
				)

			ll = len(time)
			vel_m_arr = np.asarray([vel_model]*ll)
			keep = (vel_m_arr[0,:] > min(vel_o[0])) & (vel_m_arr[0,:] < max(vel_o[0]))
			vel_m_arr = vel_m_arr[:,keep]
			shadow_model = shadow_model[:,keep]
			
			data_vec = np.array([])
			model_vec = np.array([])
			errs_vec = np.array([])
			#cc = np.array([])
			for ii in range(ll):
				vo = vel_o[ii,:]
				obs = shadow[ii,:]
				vm = vel_m_arr[ii,:]
				ff = interpolate.interp1d(vm,shadow_model[ii],kind='linear',fill_value='extrapolate')
				ishadow = ff(vo)

				sig = np.mean(unc[ii,:])*np.ones(len(obs))
				res = np.append(res,(obs - ishadow)/sig)
				#cc = np.append(cc,(obs - ishadow)**2/sig**2)
			#print(np.sum(cc/len(cc)))
	return res


#def lmfitter(param_fname,data_fname,method='leastsq',eps=0.01,
def lmfitter(parameters,data,method='nelder',eps=0.01,
		print_fit=True,convert_to_df=True):#,path_to_tracit='./'):
	'''Fit data using ``lmfit``.

	'''


	RM_path()

	fit_pars = parameters['FPs']
	pars = list(parameters.keys())
	pls = parameters['Planets']
	pars.remove('Planets')
	pars.remove('FPs')
	pars.remove('ECs')
	pars.remove('LinCombs')
	params = lmfit.Parameters()
	for par in pars:
		val = parameters[par]['Value']
		pri = parameters[par]['Prior_vals']
		#print(pri[0])
		if par in fit_pars:
			lower, upper = pri[2],pri[3]
			if np.isnan(lower): lower = -np.inf
			if np.isnan(upper): upper = np.inf
			params.add(par,value=pri[0],vary=True,min=lower,max=upper)
		else:
			params.add(par,value=val,vary=False)
	#return params
	#print(params)
	n_ls = data['LSs']
	fit_line = False
	for nn in range(1,n_ls+1):
		if data['Fit LS_{}'.format(nn)]:
			fit_line = True
			resol = data['Resolution_{}'.format(nn)]
			thick = data['Thickness_{}'.format(nn)]
			start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = ini_grid(resol,thick)

			data['Start_grid_{}'.format(nn)] = start_grid
			data['Ring_grid_{}'.format(nn)] = ring_grid
			data['Velocity_{}'.format(nn)] = vel_grid
			data['mu_{}'.format(nn)] = mu
			data['mu_grid_{}'.format(nn)] = mu_grid
			data['mu_mean_{}'.format(nn)] = mu_mean

	parameters['Planets'] = pls
	if method == 'nelder':
		fit = lmfit.minimize(residuals,params,
			args=(parameters,data),method=method)
	else:
		fit = lmfit.minimize(residuals,params,
			args=(parameters,data),method=method,
			max_nfev=500,xtol=1e-8,ftol=1e-8,epsfcn=eps)
	if print_fit: print(lmfit.fit_report(fit, show_correl=False))

	if convert_to_df: fit = fit_to_df(fit)

	return fit

def fit_to_df(fit):
	updated_pars = {}
	updated_pars['Parameter'] = [' ','Value','Fit?']
	pars = fit.params.keys()
	for par in pars:
		var = fit.params[par].vary
		updated_pars[par] = ['']
		updated_pars[par].append(fit.params[par].value)
		if var:
			updated_pars[par].append('True')
		else:
			updated_pars[par].append('False')

	df = pd.DataFrame(updated_pars)
	return df

