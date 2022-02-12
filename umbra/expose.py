#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:30:38 2021

@author: emil

.. todo::
    * Look at autocorrelation plot.

"""
# =============================================================================
# umbra modules
# =============================================================================

from umbra import stat_tools
from umbra import business
from umbra import dynamics
from umbra import shady
from umbra.priors import tgauss_prior, gauss_prior, flat_prior, tgauss_prior_dis, flat_prior_dis


# =============================================================================
# external modules
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
#from astropy.modeling import models, fitting
from scipy import interpolate
#import scipy.optimize as sop
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle
from matplotlib.gridspec import GridSpec


def Gauss(x, amp, mu,sig ):
	y = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	return y

def gaussian(x,amp,mu,sig):
	return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def residuals(pars,x,y):
	vals = pars.valuesdict()
	amp = vals['amp']
	mu = vals['mu']
	sig = vals['sig']

	gau = gaussian(x,amp,mu,sig)

	return gau-y

def run_sys(nproc):
	global plot_tex

	if nproc > 4:
		plot_tex = False
	else:
		plot_tex = True

	global colors
	colors = {
		'b' : 'C3',
		'c' : 'C4',
		'd' : 'C5',
		'e' : 'C0',
		'f' : 'C1',
		'g' : 'C2',
		'h' : 'C6',
		'i' : 'C8'
	}


# =============================================================================
# Statistics
# =============================================================================

def plot_autocorr(autocorr,index,kk,savefig=True):
	'''Autocorrelation plot.

	Plot the autocorrelation of the MCMC sampling.

	Following the example in `emcee <https://emcee.readthedocs.io/en/stable/tutorials/monitor/>`_ from [1].

	References
	----------
		[1] `Foreman-Mackey et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract>`_
	
	'''
	figc = plt.figure()
	axc = figc.add_subplot(111)
	nn, yy = kk*np.arange(1,index+1), autocorr[:index]
	axc.plot(nn,nn/100.,'k--')#,color='C7')
	axc.plot(nn,yy,'k-',lw=3.0)
	axc.plot(nn,yy,'-',color='C0',lw=2.0)
	axc.set_xlabel(r'$\rm Step \ number$')
	axc.set_ylabel(r'$\rm \mu(\hat{\tau})$')
	if savefig: figc.savefig('autocorr.pdf')

def create_chains(samples,labels=None,savefig=False,fname='chains',ival=5):
	'''Chains from the sampling.

	Plot the chains from the sampling to monitor the behaviour of the walkers during the run.

	:param samples: The samples from the MCMC.
	:type samples: array

	:param labels: Parameter names for the samples. Default ``None``.
	:type labels: list, optional

	'''
	plt.rc('text',usetex=plot_tex)
	pre = fname.split('.')[0]
	if not fname.lower().endswith(('.png', '.pdf')): ext = '.pdf'
	else: 
		ext = '.' + fname.split('.')[-1]

	_, _, ndim = samples.shape
	if labels == None: 
		labels = list(map(r'$\theta_{}$'.format(ii),range(ndim)))

	if ndim%ival:
		nivals = ival*(int(ndim/ival)+1)
	else:
		nivals = ival*int(ndim/ival)
	ivals = np.arange(0,nivals,ival)

	for ii, iv in enumerate(ivals):
		if ii == (len(ivals)-1): ival = ndim - ival*ii#(len(ivals)-1)
		if ival > 1:
			fig, axes = plt.subplots(ival,sharex=True)
			for jj in range(ival):
				ax = axes[jj]
				ax.plot(samples[:,:,jj+iv],'k',alpha=0.2)
				ax.set_ylabel(labels[jj+iv])
			axes[-1].set_xlabel(r'$\rm Step \ number$')
		else:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(samples[:,:,iv],'k',alpha=0.2)
			ax.set_ylabel(labels[iv])
			ax.set_xlabel(r'$\rm Step \ number$')

		if savefig:
			cname = pre + '_{}'.format(ii) + ext
			plt.savefig(cname)
			plt.close()

def create_corner(samples,labels=None,truths=None,savefig=True,fname='corner',
		#quantiles=[16,50,84], show_titles=True, priors=None):
		quantiles=[], diag_titles=None, priors=None):
	'''Corner plot.

	Create corner plot to investigate the covariance between the samples using `corner` [2].

	:param samples: The samples from the MCMC.
	:type samples: array

	:param labels: Parameter names for the samples. Default ``None``.
	:type labels: list, optional
	
	References
	----------
		[2] `Foreman-Mackey (2016) <https://corner.readthedocs.io/en/latest/index.html>`_



	'''
	plt.rc('text',usetex=plot_tex)
	ndim = samples.shape[-1]
	if labels == None: 
		labels = list(map(r'$\theta_{{{0}}}$'.format, range(ndim)))
	


	percentile = False
	import corner
	#fig = corner.corner(samples, labels=labels, truths=truths, show_titles=show_titles)#, quantiles=quantiles)
	fig = corner.corner(samples, labels=labels, truths=truths)
	quantiles = []
	if len(quantiles) != ndim:
		for i in range(ndim):
			
			if percentile:
				quantiles.append(np.percentile(samples[:,i],quantiles))
			else:
				val = np.median(samples[:,i])
				bounds = stat_tools.hpd(samples[:,i], 0.68)
				#up = bounds[1] - val
				#low = val - bounds[0]
				quantiles.append([bounds[0],val,bounds[1]])



	axes = np.array(fig.axes).reshape((ndim, ndim))

 
	# if priors == None:
	# 	priors = {}
	# 	for i in range(ndim): priors[1] = ['none']

	# Loop over the diagonal
	for i in range(ndim):
		ax = axes[i, i]
		#ax.axvline(value1[i], color="g")
		ax.axvline(quantiles[i][1], color='C0')
		ax.axvline(quantiles[i][0], color='C0', linestyle='--')
		ax.axvline(quantiles[i][2], color='C0', linestyle='--')
		if diag_titles is None:
			val, low, up = stat_tools.significantFormat(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1])
			label = labels[i][:-1] + '=' + str(val) + '_{-' + str(low) + '}^{+' + str(up) + '}'
		else:
			label = diag_titles[i]
		ax.text(0.5,1.05,r'{}$'.format(label),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
		if priors is not None:
			prior = priors[i][0]
			xmin, xmax = ax.get_xlim()
			ymin, ymax = ax.get_ylim()
			mu, sigma, a, b = priors[i][1], priors[i][2], priors[i][3], priors[i][4]
			if prior == 'uni':
				ax.hlines(y=0.5*ymax,color='C3',xmin=a,xmax=b)			
			elif prior != 'none':
				xs = np.arange(a,b,sigma/100.)
				if prior == 'tgauss':
					ys = np.array([tgauss_prior(x, mu, sigma, a, b) for x in xs])
				elif prior == 'gauss':
					ys = np.array([gauss_prior(x, mu, sigma) for x in xs])
				ax.plot(xs,0.9*ymax*ys/np.max(ys),color='C3')


	# Loop over the histograms
	for yi in range(ndim):
		for xi in range(yi):
			ax = axes[yi, xi]
			#ax.axvline(value1[xi], color="g")
			ax.axvline(quantiles[xi][1], color='C0')
			ax.axvline(quantiles[xi][0], color='C0', linestyle='--')
			ax.axvline(quantiles[xi][2], color='C0', linestyle='--')
			#ax.axhline(value1[yi], color="g")
			ax.axhline(quantiles[yi][1], color='C0')
			ax.axhline(quantiles[yi][0], color='C0', linestyle='--')
			ax.axhline(quantiles[yi][2], color='C0', linestyle='--')
			#ax.plot(value1[xi], value1[yi], "sg")
			ax.plot(quantiles[xi][1], quantiles[yi][1], marker='s', color='C0')


	if savefig:
		if not fname.lower().endswith(('.png', '.pdf')): fname += '.pdf'
		fig.savefig(fname)
		plt.close()


# =============================================================================
# Radial velocity curve
# =============================================================================

def plot_orbit(param_fname,data_fname,updated_pars=None,
	savefig=False,path='',OC_rv=True,n_pars=0,
	best_fit=True):
	'''Plot the radial velocity curve.

	:param param_fname: Name for the parameter .csv file. See :py:class:`business.params_temp`.
	:type param_fname: str

	:param data_fname: Name for the data .csv file. See :py:class:`business.data_temp`.
	:type data_fname: str

	:param updated_pars: Updated parameters. Default ``None``.
	:type updated_pars: :py:class:`pandas.DataFrame`, optional

	:param savefig: Whether to save the figure. Default ``True``.
	:type savefig: bool, optional

	:param path: Where to save the figure. Default ''.
	:type path: str, optional

	:param best_fit: Whether to use best-fit as opposed to median from MCMC. Default ``True``.
	:type best_fit: bool, optional

	:param n_pars: Number of fitting parameters to use for reduced chi-squared calculation. Default 0. If 0 they will be grabbed from **updated_pars**.
	:type n_pars: int, optional

	'''
	plt.rc('text',usetex=plot_tex)

	font = 15
	plt.rc('xtick',labelsize=3*font/4)
	plt.rc('ytick',labelsize=3*font/4)


	bms = 6.0 # background markersize
	fms = 4.0 # foreground markersize

	business.data_structure(data_fname)
	business.params_structure(param_fname)

	if updated_pars is not None:
		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		idx = 1
		if (updated_pars.shape[0] > 3) & best_fit: idx = 4
		for par in pars:
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass
	
	n_rv = business.data['RVs']
	pls = business.parameters['Planets']


	if n_rv >= 1:
		aa = [business.parameters['a{}'.format(ii)]['Value'] for ii in range(1,3)]
		fig = plt.figure(figsize=(12,6))
		ax = fig.add_subplot(211)
		axoc = fig.add_subplot(212)


		times, rvs, rv_errs = np.array([]), np.array([]), np.array([])
		for nn in range(1,n_rv+1):
			arr = business.data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			times, rvs, rv_errs = np.append(times,time), np.append(rvs,rv), np.append(rv_errs,rv_err)

		zp = np.amin(times)

		RMs = []
		m_rvs = np.array([])
		for nn in range(1,n_rv+1):
			label = business.data['RV_label_{}'.format(nn)]
			arr = business.data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
			jitter = business.parameters['RVsigma_{}'.format(nn)]['Value']
			jitter_err = np.sqrt(rv_err**2 + jitter**2)
			
			#chi2scale = business.data['Chi2 RV_{}'.format(nn)]
			#jitter_err *= chi2scale

			drift = aa[1]*(time-zp)**2 + aa[0]*(time-zp)
			RM = business.data['RM RV_{}'.format(nn)]
			RMs.append(RM)

			ax.errorbar(time,rv-v0,yerr=jitter_err,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
			ax.errorbar(time,rv-v0,yerr=rv_err,marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5,label=r'$\rm {}$'.format(label))
			mnrv = np.zeros(len(time))
			for pl in pls: 
				mnrv += business.rv_model(time,n_planet=pl,n_rv=nn,RM=RM)
			axoc.errorbar(time,rv-v0-drift-mnrv,yerr=jitter_err,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
			axoc.errorbar(time,rv-v0-drift-mnrv,yerr=rv_err,marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)


			print('## Spectroscopic system {}/{} ##:'.format(nn,label))
			red_chi2 = np.sum((rv-v0-drift-mnrv)**2/jitter_err**2)/(len(rv)-n_pars)
			print('\nReduced chi-squared for the radial velocity curve is:\n\t {:.03f}'.format(red_chi2))
			print('Factor to apply to get a reduced chi-squared around 1.0 is:\n\t {:.03f}\n'.format(np.sqrt(red_chi2)))
			print('Number of data points: {}'.format(len(rv)))
			print('Number of fitting parameters: {}'.format(n_pars))
			print('#########################'.format(nn))

		calc_RM = any(RMs)
		
		npoints = 50000
		unp_m = np.linspace(min(times)-10.,max(times)+10.,npoints)

		step = 100
		ivals = [(n,n+step) for n in np.arange(0,npoints,step)]
		rv_m_unp = np.zeros(len(unp_m))
		for ival in ivals:
			tt = unp_m[ival[0]:ival[1]]
			for pl in pls: 
				rv_m_unp[ival[0]:ival[1]] += business.rv_model(tt,n_planet=pl,n_rv=nn,RM=calc_RM)
				rv_m_unp[ival[0]:ival[1]] += aa[1]*(tt-zp)**2 + aa[0]*(tt-zp)

		ax.errorbar(unp_m,rv_m_unp,color='k',lw=1.)
		ax.errorbar(unp_m,rv_m_unp,color='C7',lw=0.5)
		ax.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
		axoc.axhline(0.0,linestyle='--',color='C7',zorder=-2)
		axoc.set_ylabel(r'$\rm O-C \ (m/s)$',fontsize=font)
		axoc.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)
		ax.set_xticks([])
		ax.legend(bbox_to_anchor=(0, 1.2, 1, 0),ncol=n_rv)

		fig.subplots_adjust(hspace=0.0)
		if savefig: fig.savefig(path+'rv_unphased.pdf')


		for pl in pls:
			per = business.parameters['P_{}'.format(pl)]['Value']
			t0 = business.parameters['T0_{}'.format(pl)]['Value']
			aR = business.parameters['a_Rs_{}'.format(pl)]['Value']
			rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
			inc = business.parameters['inc_{}'.format(pl)]['Value']
			ecc = business.parameters['e_{}'.format(pl)]['Value']
			ww = business.parameters['w_{}'.format(pl)]['Value']
			K = business.parameters['K_{}'.format(pl)]['Value']
			dur = dynamics.total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
			if np.isfinite(dur): x1, x2 = -1*dur/2-1.0,dur/2+1.0

			aa_pl = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]


			RMs = []
			for nn in range(1,n_rv+1):
				RM = business.data['RM RV_{}'.format(nn)]
				RMs.append(RM)
			calc_RM = any(RMs)
			if calc_RM and np.isfinite(dur):
				figrm = plt.figure()
				axrm = figrm.add_subplot(211)
				axrm_oc = figrm.add_subplot(212)
			elif K == 0.0:
				continue


			figpl = plt.figure()
			axpl = figpl.add_subplot(211)
			axpl_oc = figpl.add_subplot(212)

			for nn in range(1,n_rv+1):
				try:
					t0n = business.parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					business.parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					#business.parameters['T0_{}'.format(pl)]['Value'] = t0
					pass
				label = business.data['RV_label_{}'.format(nn)]
				arr = business.data['RV_{}'.format(nn)]
				time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
				v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
				log_jitter = business.parameters['RVsigma_{}'.format(nn)]['Value']
				#jitter = np.exp(log_jitter)
				jitter = log_jitter
				jitter_err = np.sqrt(rv_err**2 + jitter**2)
				chi2scale = business.data['Chi2 RV_{}'.format(nn)]
				jitter_err *= chi2scale



				drift = aa[1]*(time-zp)**2 + aa[0]*(time-zp)			
				RM = business.data['RM RV_{}'.format(nn)]
				for pl2 in pls: 
					if pl != pl2:			
						aa2_pl = [business.parameters['a{}_{}'.format(ii,pl2)]['Value'] for ii in range(1,3)]
						p2, t2 = business.parameters['P_{}'.format(pl2)]['Value'],business.parameters['T0_{}'.format(pl2)]['Value']
						try:
							t0n = business.parameters['Spec_{}:T0_{}'.format(nn,pl2)]
							business.parameters['T0_{}'.format(pl)]['Value'] = t0n				
						except KeyError:
							pass
						off_arr = np.round((time-t2)/p2)
						n_pers = np.unique(off_arr)
						pp = np.zeros(len(time))
						for n_per in n_pers:
							t_idxs = n_per == off_arr
							t0_off2 = n_per*p2*aa2_pl[0]#0.0
							t0_off2 += (n_per*p2)**2*aa2_pl[1]#0.0
							#t0_off2 *= -1#0.0
							t0_off2 *= 0.0
							rv[t_idxs] -= business.rv_model(time[t_idxs],n_planet=pl2,n_rv=nn,RM=RM,t0_off=t0_off2)
				

				off_arr = np.round((time-t0)/per)
				n_pers = np.unique(off_arr)
				pp = np.zeros(len(time))
				plo = np.zeros(len(time))#business.rv_model(time,n_planet=pl,n_rv=nn,RM=RM)
				for n_per in n_pers:
					t_idxs = n_per == off_arr
					t0_off = n_per*per*aa_pl[0]#0.0
					t0_off += (n_per*per)**2*aa_pl[1]#0.0
					t0_off = 0.0
					#(n_per*per)**2*aa[1]#0.0
					#t0_off *= 0#-1#(n_per*per)**2*aa[1]#0.0

					pp[t_idxs] = dynamics.time2phase(time[t_idxs],per,t0+t0_off)
					#pp[t_idxs] = dynamics.time2phase(time[t_idxs],per,t0)
					plo[t_idxs] = business.rv_model(time[t_idxs],n_planet=pl,n_rv=nn,RM=RM,t0_off=t0_off)
				#plo = business.rv_model(time,n_planet=pl,n_rv=nn,RM=RM)

				axpl.errorbar(pp,rv-v0-drift,yerr=jitter_err,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
				#axpl.errorbar(pp,rv-v0-drift,yerr=rv_err,marker='o',markersize=6.0,color='k',linestyle='none',zorder=4)
				axpl.errorbar(pp,rv-v0-drift,yerr=rv_err,marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5,label=r'$\rm {}$'.format(label))
				
				axpl_oc.errorbar(pp,rv-v0-drift-plo,yerr=jitter_err,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
				#axpl_oc.errorbar(pp,rv-v0-drift-plo,yerr=rv_err,marker='o',markersize=6.0,color='k',linestyle='none',zorder=4)
				axpl_oc.errorbar(pp,rv-v0-drift-plo,yerr=rv_err,marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)

				if calc_RM and np.isfinite(dur):
					plot = (pp*per*24 > x1) & (pp*per*24 < x2)
					axrm.errorbar(pp[plot]*per*24,rv[plot]-v0-drift[plot],yerr=jitter_err[plot],marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
					axrm.errorbar(pp[plot]*per*24,rv[plot]-v0-drift[plot],yerr=rv_err[plot],marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)
					axrm_oc.errorbar(pp[plot]*per*24,rv[plot]-v0-drift[plot]-plo[plot],yerr=jitter_err[plot],marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
					axrm_oc.errorbar(pp[plot]*per*24,rv[plot]-v0-drift[plot]-plo[plot],yerr=rv_err[plot],marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)






				
			model_time = np.linspace(t0-0.1,t0+per+0.1,2000)
			model_pp = dynamics.time2phase(model_time,per,t0)
			rv_m = business.rv_model(model_time,n_planet=pl,n_rv=nn,RM=calc_RM)#,t0_off=t0_off)

			ss = np.argsort(model_pp)
			axpl.plot(model_pp[ss],rv_m[ss],linestyle='-',color='k',lw=1.5,zorder=6)
			axpl.plot(model_pp[ss],rv_m[ss],linestyle='-',color='C7',lw=1.0,zorder=7)


			axpl_oc.axhline(0.0,linestyle='--',color='C7',zorder=-2)
			axpl_oc.set_xlabel(r'$\rm Orbital \ Phase$',fontsize=font)
			axpl_oc.set_ylabel(r'$\rm O-C \ (m/s)$',fontsize=font)
			axpl.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
			axpl.legend(bbox_to_anchor=(0, 1.2, 1, 0),ncol=n_rv)
			figpl.subplots_adjust(hspace=0.0)
			if savefig: figpl.savefig(path+'rv_{}.pdf'.format(pl))


			if calc_RM and np.isfinite(dur):

				axrm.plot(model_pp[ss]*per*24,rv_m[ss],linestyle='-',color='k',lw=1.5,zorder=6)
				axrm.plot(model_pp[ss]*per*24,rv_m[ss],linestyle='-',color='C7',lw=1.0,zorder=7)

				axrm_oc.axhline(0.0,linestyle='--',color='C7',zorder=-2)
				axrm_oc.set_xlabel(r'$\rm Hours \ From \ Midtransit$',fontsize=font)
				axrm_oc.set_ylabel(r'$\rm O-C \ (m/s)$',fontsize=font)
				axrm.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
				axrm.set_xlim(x1,x2)
				axrm_oc.set_xlim(x1,x2)
				figrm.subplots_adjust(hspace=0.0)

				if savefig: figrm.savefig(path+'rm_{}.pdf'.format(pl))


# =============================================================================
# Light curve
# =============================================================================


def plot_lightcurve(param_fname,data_fname,updated_pars=None,savefig=False,
	path='',n_pars=0,errorbar=True,best_fit=True):
	'''Plot the light curve.

	Function to plot a light curve
	
	More thorough description

	:param param_fname: Name for the parameter .csv file. See :py:class:`business.params_temp`.
	:type param_fname: str

	:param data_fname: Name for the data .csv file. See :py:class:`business.data_temp`.
	:type data_fname: str

	:param updated_pars: Updated parameters. Default ``None``.
	:type updated_pars: :py:class:`pandas.DataFrame`, optional

	:param savefig: Whether to save the figure. Default ``True``.
	:type savefig: bool, optional

	:param path: Where to save the figure. Default ''.
	:type path: str, optional

	:param best_fit: Whether to use best-fit as opposed to median from MCMC. Default ``True``.
	:type best_fit: bool, optional

	:param n_pars: Number of fitting parameters to use for reduced chi-squared calculation. Default 0. If 0 they will be grabbed from **updated_pars**.
	:type n_pars: int, optional

	'''

	plt.rc('text',usetex=plot_tex)

	font = 15
	plt.rc('xtick',labelsize=3*font/4)
	plt.rc('ytick',labelsize=3*font/4)



	business.data_structure(data_fname)
	business.params_structure(param_fname)

	if updated_pars is not None:
		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		idx = 1
		if (updated_pars.shape[0] > 3) & best_fit: idx = 4
		for par in pars:
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass
	n_phot= business.data['LCs']
	pls = business.parameters['Planets']



	if n_phot >= 1:

		npoints = 100000
		fig = plt.figure(figsize=(12,6))
		ax = fig.add_subplot(111)
		#axoc = fig.add_subplot(212)#,sharex=ax)

		time_range = []
		flux_range = []
		for nn in range(1,n_phot+1):
			arr = business.data['LC_{}'.format(nn)]
			time_range.append(min(arr[:,0]))
			time_range.append(max(arr[:,0]))
			flux_range.append(min(arr[:,1]))
			flux_range.append(max(arr[:,1]))

		times = np.linspace(min(time_range)-3./24.,max(time_range)+3./24.,npoints)
		max_fl = max(flux_range)
		min_fl = min(flux_range)
		off = 0.
		for nn in range(1,n_phot+1):
			arr = business.data['LC_{}'.format(nn)]
			label = business.data['LC_label_{}'.format(nn)]
			time, fl, fl_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()


			log_jitter = business.parameters['LCsigma_{}'.format(nn)]['Value']
			jitter_err = np.sqrt(fl_err**2 + np.exp(log_jitter)**2)

			deltamag = business.parameters['LCblend_{}'.format(nn)]['Value']
			dilution = 10**(-deltamag/2.5)	

			flux_m = np.ones(len(times))
			flux_oc = np.ones(len(time))

			in_transit = np.array([],dtype=np.int)
			in_transit_model = np.array([],dtype=np.int)

			for pl in pls:
				aa = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				try:
					t0n = business.parameters['Phot_{}:T0_{}'.format(nn,pl)]['Value']
				except KeyError:
					pass


				t0_off = 0.0
				
				flux_oc_pl = business.lc_model(time,n_planet=pl,n_phot=nn,t0_off=t0_off)

				if deltamag > 0.0:
					flux_oc_pl = flux_oc_pl/(1 + dilution) + dilution/(1+dilution)



				flux_oc -= (1 - flux_oc_pl)

				flux_model = business.lc_model(times,n_planet=pl,n_phot=nn)
				if deltamag > 0.0:
					flux_model = flux_model/(1 + dilution) + dilution/(1+dilution)
				flux_m -= 1 - flux_model  

				per, t0 = business.parameters['P_{}'.format(pl)]['Value'],business.parameters['T0_{}'.format(pl)]['Value']
				ph = dynamics.time2phase(time,per,t0)*per*24
				ph_model = dynamics.time2phase(times,per,t0)*per*24

				aR = business.parameters['a_Rs_{}'.format(pl)]['Value']
				rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
				inc = business.parameters['inc_{}'.format(pl)]['Value']
				ecc = business.parameters['e_{}'.format(pl)]['Value']
				ww = business.parameters['w_{}'.format(pl)]['Value']
				dur = dynamics.total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
				
				indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
				in_transit = np.append(in_transit,indxs)


				indxs_model = np.where((ph_model < (dur/2 + 6)) & (ph_model > (-dur/2 - 6)))[0]
				in_transit_model = np.append(in_transit_model,indxs_model)
				flux_m_trend = flux_m.copy()





			trend = business.data['Detrend LC_{}'.format(nn)]
			plot_gp = business.data['GP LC_{}'.format(nn)]
			if (trend == 'poly') or (trend == True) or (trend == 'savitsky') or plot_gp:
				ax.plot(time,fl+off,marker='.',markersize=6.0,color='C7',linestyle='none',alpha=0.5,label=r'$\rm {} \ w/o \ detrending$'.format(label))
			if (trend == 'poly') or (trend == True):
				tr_fig = plt.figure(figsize=(12,6))
				ax_tr = tr_fig.add_subplot(111)

				deg_w = business.data['Poly LC_{}'.format(nn)]
				in_transit.sort()
				in_transit = np.unique(in_transit)
				
				dgaps = np.where(np.diff(time[in_transit]) > 1)[0]
				start = 0
				temp_fl = fl - flux_oc + 1
				for dgap in dgaps:
						
					idxs = in_transit[start:int(dgap+1)]
					t = time[idxs]
					tfl = temp_fl[idxs]

					ax_tr.plot(time[start:int(dgap+1)],fl[start:int(dgap+1)],marker='.',markersize=6.0,color='k',linestyle='none')
					ax_tr.plot(time[start:int(dgap+1)],fl[start:int(dgap+1)],marker='.',markersize=4.0,color='C{}'.format(nn-1),linestyle='none')

					poly_pars = np.polyfit(t,tfl,deg_w)
					slope = np.zeros(len(t))
					for dd, pp in enumerate(poly_pars):
						slope += pp*t**(deg_w-dd)
					ax_tr.plot(t,slope,'-',color='k',lw=2.0,zorder=7)
					ax_tr.plot(t,slope,'-',color='w',lw=1.0,zorder=7)
					fl[idxs] /= slope
					

					start = int(dgap + 1)

				idxs = in_transit[start:]
				ax_tr.plot(time[start:],fl[start:],marker='.',markersize=6.0,color='k',linestyle='none')
				ax_tr.plot(time[start:],fl[start:],marker='.',markersize=4.0,color='C{}'.format(nn-1),linestyle='none')

				t = time[idxs]
				tfl = temp_fl[idxs]
				ax.plot(t,tfl,'.')
				poly_pars = np.polyfit(t,tfl,deg_w)
				slope = np.zeros(len(t))
				for dd, pp in enumerate(poly_pars):
					slope += pp*t**(deg_w-dd)
				fl[idxs] /= slope
				ax_tr.plot(t,slope,'-',color='k',lw=2.0,zorder=7)
				ax_tr.plot(t,slope,'-',color='w',lw=1.0,zorder=7)
				ax_tr.set_ylabel(r'$\rm Relative \ Brightness$',fontsize=font)
				ax_tr.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)
				if savefig: tr_fig.savefig(path+'lc_{}_Polynomial-deg{}.pdf'.format(nn,deg_w))

			elif (trend == 'savitsky'):
				sg_fig = plt.figure(figsize=(12,6))
				ax_sg = sg_fig.add_subplot(111)

				temp_fl = fl - flux_oc + 1
				window = business.data['FW LC_{}'.format(nn)]

				gap = 0.5

				dls = np.where(np.diff(time) > gap)[0]
				sav_arr = np.array([])
				start = 0
				for dl in dls:						
					sav_fil = savgol_filter(temp_fl[start:int(dl+1)],window,2)
		
					ax_sg.plot(time[start:int(dl+1)],temp_fl[start:int(dl+1)],'.',markersize=6.0,color='k')
					ax_sg.plot(time[start:int(dl+1)],temp_fl[start:int(dl+1)],'.',markersize=4.0,color='C{}'.format(nn-1))
					
					ax_sg.plot(time[start:int(dl+1)],sav_fil,color='k',lw=2.0,zorder=7)
					ax_sg.plot(time[start:int(dl+1)],sav_fil,color='w',lw=1.0,zorder=7)
					sav_arr = np.append(sav_arr,sav_fil)
					start = int(dl + 1)

				sav_fil = savgol_filter(temp_fl[start:],window,2)
				sav_arr = np.append(sav_arr,sav_fil)
				
				ax_sg.plot(time[start:],temp_fl[start:],'.',markersize=6.0,color='k')
				ax_sg.plot(time[start:],temp_fl[start:],'.',markersize=4.0,color='C{}'.format(nn-1))
				ax_sg.plot(time[start:],sav_fil,color='k',lw=2.0,zorder=7)
				ax_sg.plot(time[start:],sav_fil,color='w',lw=1.0,zorder=7)

				ax_sg.set_ylabel(r'$\rm Relative \ Brightness$',fontsize=font)
				ax_sg.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)

				if savefig: sg_fig.savefig(path+'lc_{}_Savitsky-Golay.pdf'.format(nn))

				fl /= sav_arr

			elif plot_gp:
				gp_fig = plt.figure(figsize=(12,6))
				ax_gp = gp_fig.add_subplot(111)

				loga = business.parameters['LC_{}_GP_log_a'.format(nn)]['Value']
				logc = business.parameters['LC_{}_GP_log_c'.format(nn)]['Value']
				gp = business.data['LC_{} GP'.format(nn)]

				gp.set_parameter_vector(np.array([loga,logc]))
				gp.compute(time,jitter_err)


				res_flux = fl - flux_oc


				gap = 0.5

				dls = np.where(np.diff(time) > gap)[0]
				start = 0
				for dl in dls:
					t_lin = np.linspace(min(time[start:int(dl+1)]),max(time[start:int(dl+1)]),500)
					mu, var = gp.predict(res_flux, t_lin, return_var=True)
					std = np.sqrt(var)
					ax_gp.plot(time[start:int(dl+1)],fl[start:int(dl+1)],marker='.',markersize=6.0,color='k',linestyle='none')
					ax_gp.plot(time[start:int(dl+1)],fl[start:int(dl+1)],marker='.',markersize=4.0,color='C{}'.format(nn-1),linestyle='none')

					ax_gp.fill_between(t_lin, mu+std+1, mu-std+1, color='C7', alpha=0.9, edgecolor="none",zorder=6)
					ax_gp.plot(t_lin,mu+1,color='k',lw=2.0,zorder=7)
					ax_gp.plot(t_lin,mu+1,color='w',lw=1.0,zorder=7)
					
					start = int(dl + 1)
				
				t_lin = np.linspace(min(time[start:]),max(time[start:]),500)
				mu, var = gp.predict(res_flux, t_lin, return_var=True)
				std = np.sqrt(var)
				ax_gp.plot(time[start:],fl[start:],marker='.',markersize=6.0,color='k',linestyle='none')
				ax_gp.plot(time[start:],fl[start:],marker='.',markersize=4.0,color='C{}'.format(nn-1),linestyle='none')
				ax_gp.fill_between(t_lin, mu+std+1, mu-std+1, color='C7', alpha=0.9, edgecolor="none",zorder=6)
				ax_gp.plot(t_lin,mu+1,color='k',lw=2.0,zorder=7)
				ax_gp.plot(t_lin,mu+1,color='w',lw=1.0,zorder=7)
				ax_gp.set_ylabel(r'$\rm Relative \ Brightness$',fontsize=font)
				ax_gp.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)

				if savefig: gp_fig.savefig(path+'lc_{}_GP.pdf'.format(nn))


				in_transit.sort()
				in_transit = np.unique(in_transit)
				in_transit_model.sort()
				in_transit_model = np.unique(in_transit_model)

				dgaps = np.where(np.diff(in_transit) > 1)[0]
				start = 0
				for dgap in dgaps:						
					idxs = in_transit[start:int(dgap+1)]
					t = time[idxs]
					mu, var = gp.predict(res_flux, t, return_var=True)
					std = np.sqrt(var)
					fl[idxs] -= mu
	

					start = int(dgap + 1)

				idxs = in_transit[start:]
				t = time[idxs]
				mu, var = gp.predict(res_flux, t, return_var=True)
				std = np.sqrt(var)
				fl[idxs] -= mu


			ax.plot(time,fl+off,marker='.',markersize=6.0,color='k',linestyle='none')
			ax.plot(time,fl+off,marker='.',markersize=4.0,color='C{}'.format(nn-1),linestyle='none',label=r'$\rm {}$'.format(label))




			for ii, pl in enumerate(pls):
				aa = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				per, t0 = business.parameters['P_{}'.format(pl)]['Value'],business.parameters['T0_{}'.format(pl)]['Value']
				try:
					t0n = business.parameters['Phot_{}:T0_{}'.format(nn,pl)]['Value']
					business.parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass				
				aR = business.parameters['a_Rs_{}'.format(pl)]['Value']
				rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
				inc = business.parameters['inc_{}'.format(pl)]['Value']
				ecc = business.parameters['e_{}'.format(pl)]['Value']
				ww = business.parameters['w_{}'.format(pl)]['Value']
				#dur = duration(per,rp,aR,inc)
				dur = dynamics.total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)
				full = dynamics.full_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)
				if np.isfinite(dur):
					dur *= 24
					figpl = plt.figure()
					#if OC_lc:
					axpl = figpl.add_subplot(211)
					axocpl = figpl.add_subplot(212,sharex=axpl)

					tt = dynamics.time2phase(times,per,t0)*24*per#(time%per - t0%per)/per
					ss = np.argsort(tt)


					axpl.plot(tt[ss],flux_m[ss],color='k',lw=2.0,zorder=7)
					axpl.plot(tt[ss],flux_m[ss],color='C7',lw=1.0,zorder=8)

					off_arr = np.round((time-t0)/per)
					n_pers = np.unique(off_arr)


					for n_per in n_pers:
						t0_off = 0.0#(n_per*per)**2*aa[1]#0.0

						idxs = n_per == off_arr
						phase = dynamics.time2phase(time[idxs],per,t0+t0_off)*24*per
						ff = fl[idxs]
						jerr = jitter_err[idxs]
						ferr = fl_err[idxs]
						mfl = flux_oc[idxs]

						if errorbar:
							axpl.errorbar(phase,ff,yerr=jerr,linestyle='none',marker='.',markersize=0.1,color='k')
							axpl.errorbar(phase,ff,yerr=ferr,linestyle='none',color='C{}'.format(nn-1))

						axpl.plot(phase,ff,'.',markersize=6.0,color='k')
						axpl.plot(phase,ff,'.',markersize=4.0,color='C{}'.format(nn-1))
						

						axpl.set_xlim(-1*dur/2-2.5,dur/2+2.5)
						axpl.set_ylabel(r'$\rm Relative \ Brightness$',fontsize=font)

						axocpl.axhline(0.0,linestyle='--',color='C7')
						if errorbar:
							axocpl.errorbar(phase,ff - mfl,yerr=jerr,linestyle='none',marker='.',markersize=0.1,color='k')
							axocpl.errorbar(phase,ff - mfl,yerr=ferr,linestyle='none',marker='.',markersize=0.1,color='C{}'.format(nn-1))
						axocpl.plot(phase,ff - mfl,'.',markersize=6.0,color='k')
						axocpl.plot(phase,ff - mfl,'.',markersize=4.0,color='C{}'.format(nn-1))




					axocpl.set_ylabel(r'$\rm Residuals$',fontsize=font)
					axocpl.set_xlabel(r'$\rm Hours \ From \ Midtransit$',fontsize=font)

					figpl.subplots_adjust(hspace=0.0)
					if savefig: plt.savefig(path+'lc_{}_pl_{}.pdf'.format(nn,pl))



			print('## Photometric system {}/{} ##:'.format(nn,label))
			red_chi2 = np.sum((fl - flux_oc)**2/jitter_err**2)/(len(fl)-n_pars)
			print('\nReduced chi-squared for the light curve is:\n\t {:.03f}'.format(red_chi2))
			print('Factor to apply to get a reduced chi-squared around 1.0 is:\n\t {:.03f}\n'.format(np.sqrt(red_chi2)))
			print('Number of data points: {}'.format(len(fl)))
			print('Number of fitting parameters: {}'.format(n_pars))
			print('#########################'.format(nn))



			ax.plot(times,flux_m_trend+off,color='k',lw=2.0)
			ax.plot(times,flux_m_trend+off,color='C7',lw=1.0)	


			off += 1 - min_fl + 0.001
		
		ax.legend(bbox_to_anchor=(0, 1, 1, 0),ncol=int(n_phot))
		ax.set_xlim(min(time_range)-12./24.,max(time_range)+12./24.)
		ax.set_ylabel(r'$\rm Relative \ Brightness$',fontsize=font)
		ax.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)

		if savefig: fig.savefig(path+'lc_unphased.pdf')

# =============================================================================
# Shadow plot
# =============================================================================

def create_shadow(phase,vel,shadow,exp_phase,per,
	savefig=False,fname='shadow',zmin=None,zmax=None,
	xlims=[],contour=False,vsini=None,cmap='bone_r',
	ax=None,colorbar=True,cbar_pos='right',latex=True,
	font = 12,tickfontsize=10):
	'''Shadow plot.

	Creates the shadow plot.
	
	:param vel: Velocity vector.
	:type vel: array - 

	:param phase: Phase.
	:type phase: array

	:param shadow: Shahow vector (out-of-transit absline minus in-transit absline).
	:type shadow: array
	
	:param exp_phase: Exposure time in phase units.
	:type exp_phase: array
	
	:param per: Orbital period (days).
	:type per: array 

	'''
	if not fname.lower().endswith(('.png','.pdf')): 
		ext = '.pdf'
		fname = fname.split('.')[0] + ext
	
	plt.rc('text',usetex=True)
	plt.rc('xtick',labelsize=3*font/4)
	plt.rc('ytick',labelsize=3*font/4)	
	## sort in phase
	sp = np.argsort(phase)
	shadow = shadow[sp]
	if zmin == None: zmin = np.min(shadow)
	if zmax == None: zmax = np.max(shadow)

	nn = len(phase)
	low_phase, high_phase = np.zeros(nn), np.zeros(nn)
	low_phase[:] = phase[sp] - exp_phase[sp]/2. - exp_phase[sp]/5.
	high_phase[:] = phase[sp] + exp_phase[sp]/2. + exp_phase[sp]/5.
	if not ax:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	else:
		fig = ax.get_figure()
	for ii in range(nn):
		xi = vel[ii] - np.append(np.diff(vel[ii])/2.,np.diff(vel[ii])[-1]/2.)
		x_low, y_low = np.meshgrid(xi,low_phase)
		x_high, y_high = np.meshgrid(xi,high_phase)
		xx = np.array([x_low[ii],x_high[ii]])
		yy = np.array([y_low[ii],y_high[ii]])
		mm = ax.pcolormesh(xx,yy*per*24,shadow[ii:ii+1],cmap=cmap,vmin=zmin,vmax=zmax)
	
	if contour:
		XX, YY = np.meshgrid(vel[0,:],phase*per*24)	
		ax.contour(XX,YY,shadow,1,colors='k')
	ax.set_xlabel(r'$\rm Velocity \ (km/s)$',fontsize=font)
	ax.set_ylabel(r'$\rm Hours \ from \ midtransit$',fontsize=font)
	if colorbar:
		if cbar_pos == 'right':
			cb = fig.colorbar(mm,ax=ax)
		else:
			# Now adding the colorbar
			cbaxes = fig.add_axes([0.128, 0.75, 0.43, 0.04]) 
			cb = fig.colorbar(mm,ax=ax, cax = cbaxes, orientation='horizontal') 
			cb.ax.xaxis.set_ticks_position('top')

		cb.ax.tick_params(labelsize=tickfontsize)

	if len(xlims) == 0:	
		xl, xh = ax.get_xlim()
		xlims = [xl,xh]
	ax.set_xlim(xlims[0],xlims[1])
	if vsini: 
		ax.axvline(vsini,linestyle='--',color='C0',lw=2.0)
		ax.axvline(-vsini,linestyle='--',color='C0',lw=2.0)

	if savefig: plt.savefig(fname)


def plot_shadow(param_fname,data_fname,updated_pars=None,oots=None,n_pars=0,
	cmap='gray',contact_color='C3',font = 12,savefig=True,path='',
	no_bump=15,best_fit=True,xmin=None,xmax=None,tickfontsize=10):

	

	plt.rc('text',usetex=plot_tex)

	business.data_structure(data_fname)
	business.params_structure(param_fname)
	
	pls = business.parameters['Planets']
	n_ls = business.data['LSs']
	def time2phase(time,per,T0):
		phase = ((time-T0)%per)/per
		for ii in range(len(phase)):
			if phase[ii] > 0.5: phase[ii] = phase[ii] - 1
		return phase
	if updated_pars is not None:

		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		idx = 1
		if (updated_pars.shape[0] > 3) & best_fit: idx = 4
		for par in pars:
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass	

	vsini, zeta = business.parameters['vsini']['Value'], business.parameters['zeta']['Value'] 
	for pl in pls:
		P, T0 = business.parameters['P_{}'.format(pl)]['Value'], business.parameters['T0_{}'.format(pl)]['Value'] 

		aa_pl = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]

		ar, inc = business.parameters['a_Rs_{}'.format(pl)]['Value'], business.parameters['inc_{}'.format(pl)]['Value']*np.pi/180.
		rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
		ecc, ww = business.parameters['e_{}'.format(pl)]['Value'], business.parameters['w_{}'.format(pl)]['Value']*np.pi/180.
		b = ar*np.cos(inc)*(1 - ecc**2)/(1 + ecc*np.sin(ww))
		t14 = P/np.pi * np.arcsin( np.sqrt( ((1 + rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))
		t23 = P/np.pi * np.arcsin( np.sqrt( ((1 - rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))
		if np.isnan(t14): continue

		for nn in range(1,n_ls+1):
			shadow_data = business.data['LS_{}'.format(nn)]
			label = business.data['RV_label_{}'.format(nn)]
			jitter = 0.0
			chi2scale = business.data['Chi2 LS_{}'.format(nn)]
			times = []

			for key in shadow_data.keys():
				times.append(float(key))


			times = np.asarray(times)
			ss = np.argsort(times)
			times = times[ss]
			
			if any(np.array(abs(time2phase(times,P,T0)*P) < t14)):pass
			else:continue



			v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
			rv_m = np.zeros(len(times))
			for pl in pls:
				p2, t02 = business.parameters['P_{}'.format(pl)]['Value'], business.parameters['T0_{}'.format(pl)]['Value'] 
				aa2_pl = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				off_arr2 = np.round((times-t02)/p2)
				n_pers = np.unique(off_arr2)
				for n_per in n_pers:
					t_idxs = n_per == off_arr2
					t0_off = n_per*p2*aa2_pl[0]#0.0
					t0_off += (n_per*p2)*aa2_pl[1]#0.0
					rv_pl = business.rv_model(times[t_idxs],n_planet=pl,n_rv=nn,RM=False,t0_off=t0_off)
				rv_m[t_idxs] += rv_pl
			rv_m += v0


			resol = business.data['Resolution_{}'.format(nn)]
			thick = business.data['Thickness_{}'.format(nn)]
			start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = business.ini_grid(resol,thick)



			off_arr = np.round((times-T0)/P)
			n_per = np.unique(off_arr)
			t0_off = n_per*P*aa_pl[0]
			t0_off += (n_per*P)**2*aa_pl[1]

			vel_model, shadow_model, model_ccf, darks, oot_lum, index_error = business.ls_model(
				times,start_grid,ring_grid,
				vel_grid,mu,mu_grid,mu_mean,resol,
				t0_off=t0_off
				)

			vel_m_arr = np.asarray([vel_model]*len(times))

			bright = np.sum(oot_lum)

			## Select out-of/in-transit CCFs
			## Hard coded -- modify
			idxs = [ii for ii in range(len(times))]
			if oots is None:
				#oots = [ii for ii in range(len(times)-3,len(times))]
				oots = business.data['idxs_{}'.format(nn)]

			print('Using indices {} as out-of-transit spectra'.format(oots))
			its = [ii for ii in idxs if ii not in oots]
			nvel = len(shadow_data[times[0]]['vel'])


			avg_ccf = np.zeros(nvel)



			oot_sd_b = []
			for ii, idx in enumerate(oots):
				time = times[idx]
				vel = shadow_data[time]['vel'] - rv_m[idx]*1e-3
				ccf = shadow_data[time]['ccf']

				zp_idx = np.argmin(ccf)
				zp_x = abs(vel[zp_idx])
				
				under_curve = (vel < zp_x) & (vel > -zp_x)

				ccf_u = ccf[under_curve]
				vel_u = vel[under_curve]
				pos = ccf_u > 0.0
				ccf_p = ccf_u[pos]
				vel_p = vel_u[pos]
				area = np.trapz(ccf_p,vel_p)
						
				ccf /= area	


				no_peak = (vel > no_bump) | (vel < -no_bump)
					
				poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
	

				avg_ccf += ccf
			avg_ccf /= len(oots)


			scale = np.amax(avg_ccf)
			obs_shadows = np.zeros(shape=(len(times),len(vel)))
			int_shadows = np.zeros(shape=(len(times),len(vel_model)))
			
			red_chi2_avg = []
			#oot_sd_b = []
			nps = []

			areas = np.array([])
			p0s = np.array([])
			p1s = np.array([])
			for idx in range(len(times)):
				time = times[idx]
				vel = shadow_data[time]['vel'] - rv_m[idx]*1e-3

				ccf = shadow_data[time]['ccf']


				no_peak = (vel > no_bump) | (vel < -no_bump)

				zp_idx = np.argmin(ccf)
				zp_x = abs(vel[zp_idx])
				
				under_curve = (vel < zp_x) & (vel > -zp_x)

				ccf_u = ccf[under_curve]
				vel_u = vel[under_curve]
				pos = ccf_u > 0.0
				ccf_p = ccf_u[pos]
				vel_p = vel_u[pos]
				area = np.trapz(ccf_p,vel_p)

				ccf /= area
				areas = np.append(areas,area)
				

				ccf *= darks[idx]/bright#blc[idx]		
				shadow = avg_ccf - ccf
				poly_pars = np.polyfit(vel[no_peak],shadow[no_peak],1)
				p0s = np.append(p0s,poly_pars[0])
				p1s = np.append(p1s,poly_pars[1])
								

				int_to_model = interpolate.interp1d(vel,shadow,kind='cubic',fill_value='extrapolate')
				ishadow = int_to_model(vel_model)
				obs_shadows[idx,:] = shadow# - (poly_pars[0]*vel + poly_pars[1])
				
				int_shadows[idx,:] = ishadow

				no_peak = (vel > no_bump) | (vel < -no_bump)
				
				int_to_obs = interpolate.interp1d(vel_model,shadow_model[idx],kind='cubic',fill_value='extrapolate')
				model_to_obs = int_to_obs(vel)


				vv,cc = business.get_binned(vel,shadow)
				vn,ncc = business.get_binned(vel,model_to_obs)	
				no_peak_b = (vv > no_bump) | (vv < -no_bump)
				sd = np.std(cc[no_peak_b])
				unc_b = np.ones(len(vv))*np.sqrt((sd**2 + jitter**2))*chi2scale
				red_chi2 = np.sum((cc-ncc)**2/unc_b**2)/(len(cc)-n_pars)
				
				unc = np.ones(len(vel))*np.sqrt((np.std(shadow[no_peak])**2 + jitter**2))*chi2scale
				red_chi2 = np.sum((shadow-model_to_obs)**2/unc**2)/(len(shadow)/4 - n_pars)
				nps.append(len(shadow)/4)
				red_chi2_avg.append(red_chi2)
			n_points = np.mean(nps)



			print('## Spectroscopic system {}/{} ##:'.format(nn,label))
			print('\nReduced chi-squared for the shadow is:\n\t {:.03f}'.format(np.mean(red_chi2_avg)))
			print('Factor to apply to get a reduced chi-squared around 1.0 is:\n\t {:.03f}\n'.format(np.sqrt(np.mean(red_chi2_avg))))
			print('Number of data points: {}'.format(n_points))
			print('Number of fitting parameters: {}'.format(n_pars))
			print('#########################')

			phase = time2phase(times,P,T0) #phase of observing times
			exptime = np.mean(np.diff(times))*np.ones(len(times))*24*60*60
			exptime_phase = exptime/(P*24.*60.*60.) #exptimes converted to phase-units for making shadow figure
			zmin, zmax = np.min(int_shadows), np.max(int_shadows)
			zmin, zmax = np.min(obs_shadows), np.max(obs_shadows)

			plt.rcParams['ytick.labelsize']	= tickfontsize
			plt.rcParams['xtick.labelsize']	= tickfontsize
			plt.figure(figsize=(16,6))
			gs = GridSpec(5, 7)
			ax1 = plt.subplot(gs[1:, 0:2])
			ax2 = plt.subplot(gs[1:, 2:4])
			ax3 = plt.subplot(gs[1:, 4:6])

			axes = [ax1,ax2,ax3]

			axres2 = plt.subplot(gs[1:, 6])
			axres1 = plt.subplot(gs[0, 4:6])


			create_shadow(phase, vel_m_arr, -1*int_shadows/scale, exptime_phase,P,cmap=cmap,
									vsini=vsini,zmin=zmin,zmax=zmax,contour=False,ax=ax1,colorbar=False,latex=plot_tex,font=font)

			create_shadow(phase, vel_m_arr, -1*shadow_model/scale, exptime_phase,P, vsini=vsini,cmap=cmap,font=font,
									zmin=zmin,zmax=zmax,contour=False,ax=ax2,cbar_pos='top',latex=plot_tex,tickfontsize=tickfontsize)

			diff = -1*(int_shadows - shadow_model)
			create_shadow(phase, vel_m_arr, diff/scale, exptime_phase,P, cmap=cmap,font=font,
									vsini=vsini,zmin=zmin,zmax=zmax,contour=False,ax=ax3,colorbar=False,latex=plot_tex);# plt.show()


			for ax in axes:
				x1, x2 = ax.get_xlim()
				y1, y2 = ax.get_ylim()
				ax.axhline(-1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)
				ax.axhline(1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)

				ax.axhline(1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)
				ax.axhline(-1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)
				if xmax != None: x2 = xmax
				if xmin != None: x1 = xmin
				ax.set_xlim(x1,x2)
				ax.set_ylim(y1,y2)

			ax1.tick_params(axis='both',labelsize=tickfontsize)
			ax2.tick_params(axis='both',labelsize=tickfontsize)
			ax3.tick_params(axis='both',labelsize=tickfontsize)
	
			ax2.set_ylabel('')
			ax3.set_ylabel('')
			plt.setp(ax2.get_yticklabels(),visible=False)
			plt.setp(ax3.get_yticklabels(),visible=False)


			low = np.min(diff)
			diff *= -1
			res2 = np.zeros(len(phase))
			for ii in range(len(phase)):
				res2[ii] = np.sum(abs(diff[ii,:]))
			res1 = np.zeros(diff.shape[1])
			for ii in range(len(res1)):
				res1[ii] = np.sum(abs(diff[:,ii]))
				
			axres2.plot(res2,phase,'k-')
			axres2.set_yticks([])
			axres2.set_xlabel(r'$\Sigma \rm |O-C|$',fontsize=font)# \abs{\delta}$')

			axres2.set_ylim(min(phase),max(phase))

			if (xmin != None) & (xmax != None):
				keep = (vel_m_arr[0,:] > xmin) & (vel_m_arr[0,:] < xmax)
				vel_m = vel_m_arr[0,keep]
				res1 = res1[keep]
			elif xmax != None:
				keep = vel_m_arr[0,:] < xmax
				vel_m = vel_m_arr[0,keep]
				res1 = res1[keep]
			elif xmin != None:
				keep = vel_m_arr[0,:] > xmin
				vel_m = vel_m_arr[0,keep]
				res1 = res1[keep]
			else:
				vel_m = vel_m_arr[0,:]

			axres1.plot(vel_m,res1,'k-')
			axres1.set_xticks([])
			axres1.set_ylabel(r'$\Sigma \rm |O-C|$',fontsize=font)# \abs{\delta}$')
			axres1.set_xlim(min(vel_m),max(vel_m))
			axres1.set_ylim(ymin=0.0)
			#axres1.set_xlim(x1,x2)
			axres2.set_xlim(xmin=0.0)
			axres1.yaxis.tick_right()
			axres1.yaxis.set_label_position("right")

			plt.subplots_adjust(wspace=0.0,hspace=0.0)
			if savefig: plt.savefig(path+'shadow.png')

# =============================================================================
# Out-of-transit plot
# =============================================================================

def plot_oot_ccf(param_fname,data_fname,updated_pars=None,oots=None,n_pars=0,chi2_scale=1.0,
	font = 12,savefig=True,path='',no_bump=15,best_fit=True,xmajor=None,xminor=None,
	ymajor1=None,yminor1=None,ymajor2=None,yminor2=None,plot_intransit=True,xmax=None,xmin=None):

	plt.rc('text',usetex=plot_tex)

	business.data_structure(data_fname)
	business.params_structure(param_fname)

	
	pls = business.parameters['Planets']
	n_ls = business.data['LSs']

	import celerite

	if updated_pars is not None:
		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		idx = 1
		if (updated_pars.shape[0] > 3) & best_fit: idx = 4
		for par in pars:
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass	
	for nn in range(1,n_ls+1):
		label = business.data['RV_label_{}'.format(nn)]

		shadow_data = business.data['LS_{}'.format(nn)]
		chi2scale = business.data['Chi2 OOT_{}'.format(nn)]

		times = []
		for key in shadow_data.keys():
			try:
				times.append(float(key))
			except ValueError:
				pass
		times = np.asarray(times)
		ss = np.argsort(times)
		times = times[ss]

		v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
		rv_m = np.zeros(len(times))
		for pl in pls:
			#rv_pl = business.rv_model(business.parameters,time,n_planet=pl,n_rv=nn,RM=calc_RM)
			rv_pl = business.rv_model(times,n_planet=pl,n_rv=nn,RM=False)
			rv_m += rv_pl
		rv_m += v0
		#print(business.parameters['xi']['Value'])
		# resol = business.data['Resolution_{}'.format(nn)]
		# start_grid = business.data['Start_grid_{}'.format(nn)]
		# ring_grid = business.data['Ring_grid_{}'.format(nn)]
		# vel_grid = data['Velocity_{}'.format(nn)]
		# mu = business.data['mu_{}'.format(nn)]
		# mu_grid = business.data['mu_grid_{}'.format(nn)]
		# mu_mean = business.data['mu_mean_{}'.format(nn)]			
		#only_oot = data['Only_OOT_{}'.format(nn)]			
		#fit_oot = data['OOT_{}'.format(nn)]	

		resol = business.data['Resolution_{}'.format(nn)]
		thick = business.data['Thickness_{}'.format(nn)]
		start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = business.ini_grid(resol,thick)

		#for pl in pls:
		# vel_1d, line_oot_norm, lum = business.ls_model(
		# 	#parameters,time,start_grid,ring_grid,
		# 	times[-3:],start_grid,ring_grid,
		# 	vel_grid,mu,mu_grid,mu_mean,resol,
		# 	n_planet=pl,n_rv=nn,oot=True
		# 	)

		#vel_model, shadow_model, model_ccf, darks, oot_lum, index_error = business.ls_model(
		vel_model, model_ccf, oot_lum = business.ls_model(
			#business.parameters,time,start_grid,ring_grid,
			times,start_grid,ring_grid,
			vel_grid,mu,mu_grid,mu_mean,resol,
			n_planet='b',n_rv=nn,oot=True
			)

		bright = np.sum(oot_lum)


		## Select out-of/in-transit CCFs
		## Hard coded -- modify
		#oots = [ii for ii in range(len(times)-3,len(times))]
		#its = [ii for ii in range(len(times)-3)]
		idxs = [ii for ii in range(len(times))]
		#oots = [-3,-2,-1]
		if oots is None:
			#oots = [ii for ii in range(len(times)-3,len(times))]
			oots = business.data['idxs_{}'.format(nn)]

		print('Using indices {} as out-of-transit spectra'.format(oots))

		its = [ii for ii in idxs if ii not in oots]	
		
		nvel = len(shadow_data[times[0]]['vel'])
		vels = np.zeros(shape=(nvel,len(times)))
		oot_ccfs = np.zeros(shape=(nvel,len(oots)))
		#in_ccfs = np.zeros(shape=(nvel,len(its)))
		avg_ccf = np.zeros(nvel)
		avg_vel = np.zeros(nvel)

		## Create average out-of-transit CCF
		## Used to create shadow for in-transit CCFs
		## Shift CCFs to star rest frame
		## and detrend CCFs
		oot_sd_b = []
		for ii, idx in enumerate(oots):
			time = times[idx]
			vel = shadow_data[time]['vel'] - rv_m[idx]*1e-3
			vels[:,idx] = vel
			no_peak = (vel > no_bump) | (vel < -no_bump)
			


			#ccf = np.zeros(len(vel))
			#shadow_arr = shadow_data[time]['ccf']
			#ccf = 1 - shadow_arr/np.median(shadow_arr[no_peak])

			ccf = shadow_data[time]['ccf']
			#area = np.trapz(ccf,vel)

			zp_idx = np.argmin(ccf)
			zp_x = abs(vel[zp_idx])
			
			under_curve = (vel < zp_x) & (vel > -zp_x)
			#area = np.trapz(ccf[under_curve],vel[under_curve])

			ccf_u = ccf[under_curve]
			vel_u = vel[under_curve]
			pos = ccf_u > 0.0
			ccf_p = ccf_u[pos]
			vel_p = vel_u[pos]
			area = np.trapz(ccf_p,vel_p)

			ccf /= abs(area)
			#oot_sd.append(np.std(ccf[no_peak]))

			vv,cc = business.get_binned(vels[:,idx],ccf)
			no_peak_b = (vv > no_bump) | (vv < -no_bump)
			oot_sd_b.append(np.std(cc[no_peak_b]))
				
			#poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)

			#cc += vv*poly_pars[0] + poly_pars[1]
			#ccf -= vel*poly_pars[0] + poly_pars[1]

			oot_ccfs[:,ii] = ccf
			avg_ccf += ccf
			avg_vel += vel

		avg_ccf /= len(oots)
		avg_vel /= len(oots)







			#avg_ccf += ccf
			#avg_vel += vel

		## Here we simply fit our average out-of-transit CCF
		## to an out-of-transit model CCF
		## Hard-coded
		log_jitter = business.parameters['RVsigma_{}'.format(nn)]['Value']
		#jitter = np.exp(log_jitter)
		jitter = log_jitter
		jitter = 0.0

		model_int = interpolate.interp1d(vel_model,model_ccf,kind='cubic',fill_value='extrapolate')
		newline = model_int(vels[:,idx])

		loga = np.log(np.var(oot_ccfs[:,ii]  - newline))
		logc = -np.log(10)
		print(loga,logc)
		logc = -4.3
		loga = -12
		kernel = celerite.terms.RealTerm(log_a=loga, log_c=logc)
		#kernel = celerite.terms.Matern32Term(log_sigma=loga, log_rho=logc)
		unc = np.ones(len(vel))*np.sqrt((np.mean(oot_sd_b)**2 + jitter**2))
		unc *= chi2scale#*2
		print(np.mean(unc))
		gp = celerite.GP(kernel)
		gp.compute(avg_vel,unc)

		mu, var = gp.predict(oot_ccfs[:,ii]  - newline, avg_vel, return_var=True)
		std = np.sqrt(var)
		fig = plt.figure()
		ax = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		#ax.errorbar(avg_vel,oot_ccfs[:,ii]-newline,yerr=unc)
		ax.errorbar(avg_vel,oot_ccfs[:,ii],yerr=unc)
		ax.errorbar(avg_vel,newline+mu)
		ax.fill_between(avg_vel, newline+mu+std, newline+mu-std, color='C1', alpha=0.3, edgecolor="none")
		#ax.plot(avg_vel, mu, color='C1')
		#ax.fill_between(avg_vel, mu+std, mu-std, color='C1', alpha=0.3, edgecolor="none")
		ax2.errorbar(avg_vel,oot_ccfs[:,ii]-newline,yerr=unc,linestyle='none')
		ax2.fill_between(avg_vel, mu+std, mu-std, color='C1', alpha=0.3, edgecolor="none")

		import sys
		sys.exit()
		#unc = np.ones(len(vel))*np.mean(oot_sd_b)*jitter
		vv,cc = business.get_binned(vels[:,idx],avg_ccf)
		vn,ncc = business.get_binned(vels[:,idx],newline)
		unc_b = np.ones(len(vv))*np.sqrt((np.mean(oot_sd_b)**2 + jitter**2))
		unc = np.ones(len(vel))*np.sqrt((np.mean(oot_sd_b)**2 + jitter**2))
		unc_b *= chi2scale
		red_chi2 = np.sum((cc-ncc)**2/unc_b**2)/(len(cc)-n_pars)
		print('## Spectroscopic system {}/{} ##:'.format(nn,label))
		print('\nReduced chi-squared for the oot CCF is:\n\t {:.03f}'.format(red_chi2))
		print('Factor to apply to get a reduced chi-squared around 1.0 is:\n\t {:.03f}\n'.format(np.sqrt(red_chi2)))
		print('Number of data points: {}'.format(len(cc)))
		print('Number of fitting parameters: {}'.format(n_pars))
		print('#########################')


		figccf = plt.figure()
		ax1_ccf = figccf.add_subplot(211)
		ax2_ccf = figccf.add_subplot(212)

		ax1_ccf.plot(vels[:,idx],avg_ccf,'-',color='k',label=r'$\rm Observed\ avg. \ CCF$',lw=5.0,zorder=0)
		ax1_ccf.plot(vels[:,idx],newline,'--',color='C7',label=r'$\rm Model \ CCF$',lw=2.0)
		ax2_ccf.plot(vels[:,idx],avg_ccf  - newline,color='k',linestyle='-',lw=5.0,zorder=0)#,mfc='C7')
		out = (vv < -no_bump) | (no_bump < vv)
		out2 = (vels[:,idx] < -no_bump) | (no_bump < vels[:,idx])
		ax2_ccf.errorbar(vels[:,idx][out2],avg_ccf[out2]  - newline[out2],yerr=unc[out2],color='k',marker='.',mfc='C7',linestyle='none')
		for ii, idx in enumerate(oots):
			ax1_ccf.plot(vels[:,idx],oot_ccfs[:,ii],zorder=0,label=r'$\rm OOT\ idx.\ {}$'.format(idx),lw=1.0)
			ax2_ccf.plot(vels[:,idx],oot_ccfs[:,ii] - newline,zorder=0,lw=1.0)


		ax2_ccf.errorbar(vv[out],cc[out]-ncc[out],yerr=unc_b[out],color='k',marker='o',mfc='C3',ecolor='C3',linestyle='none')
		ax2_ccf.axhline(0.0,linestyle='--',color='C7',zorder=-4)

		ax2_ccf.set_xlabel(r'$\rm Velocity \ (km/s)$',fontsize=font)
		ax2_ccf.set_ylabel(r'$\rm Residuals$',fontsize=font)
		ax1_ccf.set_ylabel(r'$\rm CCF$',fontsize=font)
		ax1_ccf.legend(fancybox=True,shadow=True,fontsize=0.9*font,
			ncol=round(len(oots)/2+1),loc='upper center',bbox_to_anchor=(0.5, 1.35))
			#ncol=1,loc='right',bbox_to_anchor=(1.0, 0.5))

		if (xmajor != None) & (xminor != None):
			from matplotlib.ticker import MultipleLocator

			ax1_ccf.xaxis.set_major_locator(MultipleLocator(xmajor))
			ax1_ccf.xaxis.set_minor_locator(MultipleLocator(xminor))
			ax2_ccf.xaxis.set_major_locator(MultipleLocator(xmajor))
			ax2_ccf.xaxis.set_minor_locator(MultipleLocator(xminor))
		if (ymajor1 != None) & (yminor1 != None):
			from matplotlib.ticker import MultipleLocator

			ax1_ccf.yaxis.set_major_locator(MultipleLocator(ymajor1))
			ax1_ccf.yaxis.set_minor_locator(MultipleLocator(yminor1))
		if (ymajor2 != None) & (yminor2 != None):
			from matplotlib.ticker import MultipleLocator
			ax2_ccf.yaxis.set_major_locator(MultipleLocator(ymajor2))
			ax2_ccf.yaxis.set_minor_locator(MultipleLocator(yminor2))

		ax1_ccf.set_xlim(xmin,xmax)
		ax2_ccf.set_xlim(xmin,xmax)
		plt.setp(ax1_ccf.get_xticklabels(),visible=False)
		figccf.subplots_adjust(hspace=0.05)
		figccf.tight_layout()
		if savefig: figccf.savefig('oot_ccf.pdf')

		if plot_intransit:

			_, _, _, darks, oot_lum, _ = business.ls_model(
				#business.parameters,time,start_grid,ring_grid,
				times,start_grid,ring_grid,
				vel_grid,mu,mu_grid,mu_mean,resol
				)

			#vel_m_arr = np.asarray([vel_model]*len(times))

			bright = np.sum(oot_lum)

			fig_in = plt.figure()

			cmap = plt.get_cmap('Spectral',len(its))
			#cmap = plt.get_cmap('tab20b',len(its))
			sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=plt.normalize(min=0, max=1))
			cbaxes = fig_in.add_axes([0.91, 0.11, 0.02, 0.78])
			cticks = [ii/len(its)+0.05 for ii in range(len(its))]
			#print(cticks)
			cbar = fig_in.colorbar(sm,cax=cbaxes,ticks=cticks)
			cbar.set_label(r'$\rm Exposure \ index \ (Time \Rightarrow)$')
			cticklabs = ['${}$'.format(ii) for ii in range(len(its))]
			cbar.ax.set_yticklabels(cticklabs)
			#ax2_ccf.yaxis.set_minor_locator(MultipleLocator(yminor2))

			ax1 = fig_in.add_subplot(211)
			ax2 = fig_in.add_subplot(212)


			ax1.axhline(0.0,color='C7',linestyle='--')
			ax1.plot(vel,avg_ccf,'k-',lw=4.0,label=r'$\rm Observed\ avg.$')
			ax2.axhline(0.0,color='k')

			for ii, idx in enumerate(its):
				time = times[idx]
				vel = shadow_data[time]['vel'] - rv_m[idx]*1e-3
				vels[:,idx] = vel
				no_peak = (vel > no_bump) | (vel < -no_bump)
				


				#ccf = np.zeros(len(vel))
				#shadow_arr = shadow_data[time]['ccf']
				#ccf = 1 - shadow_arr/np.median(shadow_arr[no_peak])

				ccf = shadow_data[time]['ccf']

				zp_idx = np.argmin(ccf)
				zp_x = abs(vel[zp_idx])

				under_curve = (vel < zp_x) & (vel > -zp_x)
				#area = np.trapz(ccf[under_curve],vel[under_curve])

				ccf_u = ccf[under_curve]
				vel_u = vel[under_curve]
				pos = ccf_u > 0.0
				ccf_p = ccf_u[pos]
				vel_p = vel_u[pos]
				area = np.trapz(ccf_p,vel_p)
				
				#area = np.trapz(ccf,vel)
				ccf /= abs(area)
				
				ccf *= darks[idx]/bright#blc[idx]
				#oot_sd.append(np.std(ccf[no_peak]))

				vv,cc = business.get_binned(vels[:,idx],ccf)
				no_peak_b = (vv > no_bump) | (vv < -no_bump)
				oot_sd_b.append(np.std(cc[no_peak_b]))
					
				poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)

				#cc += vv*poly_pars[0] + poly_pars[1]
				ccf -= vel*poly_pars[0] + poly_pars[1]

				ax1.plot(vel,ccf,'-',color=cmap(ii),lw=1.0)
				ax2.plot(vel,ccf-avg_ccf,'-',color=cmap(ii),lw=1.0)

				#in_ccfs[:,ii] = ccf - avg_ccf

			ax1.legend(fancybox=True,shadow=True,fontsize=0.9*font)
			plt.setp(ax1.get_xticklabels(),visible=False)
			fig_in.subplots_adjust(hspace=0.05)
			ax2.set_xlabel(r'$\rm Velocity \ (km/s)$',fontsize=font)
			ax2.set_ylabel(r'$\rm Exp.\ idx.-Avg.$',fontsize=font)
			ax1.set_ylabel(r'$\rm CCF$',fontsize=font)
			ax1.set_xlim(xmin,xmax)
			ax2.set_xlim(xmin,xmax)
			if savefig: fig_in.savefig('in_minus_out_ccf.pdf')




def plot_distortion(param_fname,data_fname,updated_pars=None,observation=False,
	oots=None,n_pars=0,display=[],background='white',model=False,ax=None,stack = {},
	font = 14,savefig=True,path='',contact_color='C3',movie_time=False,return_slopes=False,
	no_bump=15,best_fit=True,get_vp=False,tickfontsize=10):

	#from matplotlib.gridspec import GridSpec

	plt.rc('text',usetex=plot_tex)

	if not get_vp:
		business.params_structure(param_fname)
		business.data_structure(data_fname)
	if updated_pars is not None:

		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		for par in pars:
			if best_fit: idx = 4
			else: idx = 1
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass				
	pls = business.parameters['Planets']
	n_sl = business.data['SLs']

	from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter

	for nn in range(1,n_sl+1):
		slope_data = business.data['SL_{}'.format(nn)]
		label = business.data['RV_label_{}'.format(nn)]
		times = []
		for key in slope_data.keys():
			try:
				times.append(float(key))
			except ValueError:
				pass
		times = np.asarray(times)
		ss = np.argsort(times)
		times = times[ss]
		v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
		rv_m = np.zeros(len(times))
		for pl in pls:
			aa2_pl = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
			p2, t2 = business.parameters['P_{}'.format(pl)]['Value'],business.parameters['T0_{}'.format(pl)]['Value']
			off_arr = np.round((times-t2)/p2)
			n_pers = np.unique(off_arr)
			for n_per in n_pers:
				t_idxs = n_per == off_arr
				t0_off2 = n_per*p2*aa2_pl[0]#0.0
				t0_off2 += (n_per*p2)**2*aa2_pl[1]#0.0
				rv_pl = business.rv_model(times[t_idxs],n_planet=pl,n_rv=nn,RM=False,t0_off=t0_off2)
	
			rv_m += rv_pl
		rv_m += v0


		for pl in pls:
			try:
				t0n = business.parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
				business.parameters['T0_{}'.format(pl)]['Value'] = t0n				
			except KeyError:
				pass




			per, T0 = business.parameters['P_{}'.format(pl)]['Value'], business.parameters['T0_{}'.format(pl)]['Value'] 
			ar, inc = business.parameters['a_Rs_{}'.format(pl)]['Value'], business.parameters['inc_{}'.format(pl)]['Value']*np.pi/180.
			rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
			ecc, ww = business.parameters['e_{}'.format(pl)]['Value'], business.parameters['w_{}'.format(pl)]['Value']*np.pi/180.
			b = ar*np.cos(inc)*(1 - ecc**2)/(1 + ecc*np.sin(ww))

			t14 = per/np.pi * np.arcsin( np.sqrt( ((1 + rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))

			t23 = per/np.pi * np.arcsin( np.sqrt( ((1 - rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))
			if np.isnan(t14): continue


			off_arr = np.round((times-T0)/per)
			n_pers = np.unique(off_arr)[0]
	
			t0_off = n_per*per*aa2_pl[0]#0.0
			t0_off += (n_per*per)**2*aa2_pl[1]#0.0

			model_slope = business.localRV_model(times,n_planet=pl,t0_off=t0_off)				

			### HARD-CODED
			darks = business.lc_model(times,n_planet=pl,n_phot=1,t0_off=t0_off)


			idxs = [ii for ii in range(len(times))]
			if oots is None:
				oots = business.data['idxs_{}'.format(nn)]

			if model:
				resol = business.data['Resolution_{}'.format(nn)]
				thick = business.data['Thickness_{}'.format(nn)]
				start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = business.ini_grid(resol,thick)

	

				vel_model, shadow_model, _, _, _, _ = business.ls_model(
					times,start_grid,ring_grid,
					vel_grid,mu,mu_grid,mu_mean,resol,
					t0_off=t0_off,oot=False,n_planet=pl,n_rv=nn
					)

				vel_m_arr = np.asarray([vel_model]*len(times))



			print('Using indices {} as out-of-transit spectra'.format(oots))

			its = [ii for ii in idxs if ii not in oots]	

			pp = dynamics.time2phase(times[its],per,T0)*24*per

			nvel = len(slope_data[times[0]]['vel'])
			vels = np.zeros(shape=(nvel,len(its)))
			ccfs = np.zeros(shape=(nvel,len(its)))
			avg_ccf = np.zeros(nvel)
			for ii, idx in enumerate(oots):
				time = times[idx]
				vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
				no_peak = (vel > no_bump) | (vel < -no_bump)
				

				ccf = slope_data[time]['ccf']
				area = np.trapz(ccf,vel)
				ccf /= area	

				poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
				ccf -= vel*poly_pars[0] + poly_pars[1]

				# oot_ccfs[:,ii] = ccf
				avg_ccf += ccf

			avg_ccf /= len(oots)

			lam = business.parameters['lam_{}'.format(pl)]['Value']*np.pi/180
			vsini = business.parameters['vsini']['Value'] 


			fnames = []
			xs = np.array([])
			ys = np.array([])
			cs = np.array([])
			mus = np.array([])
			for ii, idx in enumerate(its):
				time = times[idx]
				vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
				vels[:,ii] = vel
				no_peak = (vel > no_bump) | (vel < -no_bump)

				cos_f, sin_f = dynamics.true_anomaly(time, T0+t0_off, ecc, per, ww)
				xx, yy = dynamics.xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
				# xs = np.append(xs,xx)
				# ys = np.append(ys,yy)
				rr = np.sqrt(xx**2 + yy**2)
				mm = np.sqrt(1 - (rr)**2)
				if rr > 1.0: mm = 0.0
				mus = np.append(mus,mm)
				#print(xx,yy)
				cs = np.append(cs,xx*vsini)
				
				ccf = slope_data[time]['ccf']
				area = np.trapz(ccf,vel)
				ccf /= area
				
				sd = np.std(ccf[no_peak])

				ccf *= darks[idx]#/bright#blc[ii]		
				shadow = avg_ccf - ccf
				poly_pars = np.polyfit(vel[no_peak],shadow[no_peak],1)
				
				shadow -=  vel*poly_pars[0] + poly_pars[1]
				ccfs[:,ii] = shadow


			cmap = plt.get_cmap('RdBu_r',len(cs))
			if ax == None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			ax.set_facecolor(background)
			ilikedis = False

			if ilikedis:
				for ii, idx in enumerate(its):
					vel = vels[:,ii]
					shadow = ccfs[:,ii]

					if len(display):
						if ii in display:
							if observation:
								ax.plot(vel,shadow,color='k',lw=2.0,zorder=8)
								ax.plot(vel,shadow,color=cmap(ii),lw=1.5,zorder=9,linestyle='--')
							if model:
								#print(shadow_model[idx])
								ax.plot(vel_model,shadow_model[idx],color='k',lw=2.,zorder=5,linestyle='-')
								ax.plot(vel_model,shadow_model[idx],color=cmap(ii),lw=1.5,zorder=6,linestyle='-')


					else:
						ax.plot(vel,shadow,color=cmap(ii))
				ax.set_xlim(-16,16)
				ax.yaxis.set_major_locator(MultipleLocator(0.0004))
				ax.yaxis.set_minor_locator(MultipleLocator(0.0002))
			else:
				from matplotlib.legend_handler import HandlerBase

				class AnyObjectHandler(HandlerBase):
					def create_artists(self, legend, orig_handle,
									x0, y0, width, height, fontsize, trans):
						l1 = plt.Line2D([x0,y0+15.0], [0.4*height,0.4*height],linestyle='-', color='k',lw=2.5,zorder=-1)	        
						l2 = plt.Line2D([x0,y0+15.0], [0.4*height,0.4*height],linestyle='-', color=orig_handle[1],lw=1.5)

						return [l1, l2]

				labs, hands = [], []
				labs2 = []
				off = 0
				disp = -0.0007
				for key in stack.keys():
					idxs = stack[key]
					avg_shadow = np.zeros(len(ccfs[:,0]))
					if model:
						avg_vel_model = np.zeros(len(shadow_model[0,:]))
						avg_model = np.zeros(len(shadow_model[0,:]))
					
					avg_vel = np.zeros(len(ccfs[:,0]))
					avg_mu = 0
					for idx in idxs:
						vel = vels[:,idx]
						shadow = ccfs[:,idx]
						avg_vel += vel - cs[idx]
						if model:
							avg_vel_model += vel_model - cs[idx]
							avg_model += shadow_model[idx]
						avg_shadow += shadow
						avg_mu += mus[idx]

						#print(mus[idx])
					avg_vel /= len(idxs)
					avg_shadow /= len(idxs)
					if model:
						avg_model /= len(idxs)
						avg_vel_model /= len(idxs)
					
					avg_color = int(np.mean(idxs))
					avg_mu /= len(idxs)
					#print(avg_mu)
					label = r'$\rm Index \ ' + str(idxs)[1:-1] + ':\ \langle \mu \\rangle={:.2f}'.format(avg_mu) + '$'
					#label = r'$\rm Index \ ' + str(idxs)[1:-1] + '$'
					labs.append(label)
					label2 = r'$\rm  \mu={:.1f}'.format(avg_mu) + '$'
					labs2.append(label2)
					hands.append((0.2,cmap(avg_color)))

					ax.plot(avg_vel,avg_shadow+off,color='k',lw=2.5,zorder=5,linestyle='-')
					ax.plot(avg_vel,avg_shadow+off,color=cmap(avg_color),lw=1.5,zorder=6,linestyle='-')
					if model:
						ax.plot(avg_vel_model,avg_model+off,color='k',lw=2.5,zorder=5,linestyle='-')
						ax.plot(avg_vel_model,avg_model+off,color=cmap(avg_color),lw=1.5,zorder=6,linestyle='-')

					ax.axhline(off,color='C7',zorder=0,linestyle='--')
					off += disp

				ax.legend(hands, labs,ncol=1,#bbox_to_anchor=(0.7,1.0),
							handler_map={tuple: AnyObjectHandler()},
							fancybox=True,shadow=True,fontsize=0.7*font)#,


				ax.set_xlim(-14,14)
				
				ax.yaxis.set_major_locator(MultipleLocator(0.0005))
				ax.yaxis.set_minor_locator(MultipleLocator(0.00025))

			ax.set_xlabel(r'$\rm Velocity \ (km/s)$',fontsize=font)
			ax.set_ylabel(r'$\rm Distortion$',fontsize=font)
			
			ax.xaxis.set_major_locator(MultipleLocator(5))
			ax.xaxis.set_minor_locator(MultipleLocator(2.5))
			ax.tick_params(axis='both',labelsize=tickfontsize)

			plt.tight_layout()

			if savefig: plt.savefig(path+'distortion2.pdf')


# =============================================================================
# Slope of planet across disk
# =============================================================================

def plot_slope(param_fname,data_fname,
	updated_pars=None,
	oots=None,n_pars=0,
	font = 12,savefig=True,path='',
	contact_color='C3',movie_time=False,return_slopes=False,
	no_bump=15,best_fit=True,get_vp=False):


	plt.rc('text',usetex=plot_tex)

	if not get_vp:
		business.params_structure(param_fname)
		business.data_structure(data_fname)
	if updated_pars is not None:

		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		for par in pars:
			if best_fit: idx = 4
			else: idx = 1
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass				
	pls = business.parameters['Planets']
	n_sl = business.data['SLs']

	slopes = {}

	for nn in range(1,n_sl+1):
		slope_data = business.data['SL_{}'.format(nn)]
		label = business.data['RV_label_{}'.format(nn)]
		slopes['RV_'+str(nn)] = {}
		times = []
		for key in slope_data.keys():
			try:
				times.append(float(key))
			except ValueError:
				pass
		times = np.asarray(times)
		ss = np.argsort(times)
		times = times[ss]
		v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
		rv_m = np.zeros(len(times))
		for pl in pls:
			aa2_pl = [business.parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
			p2, t2 = business.parameters['P_{}'.format(pl)]['Value'],business.parameters['T0_{}'.format(pl)]['Value']
			off_arr = np.round((times-t2)/p2)
			n_pers = np.unique(off_arr)
			for n_per in n_pers:
				t_idxs = n_per == off_arr
				t0_off2 = n_per*p2*aa2_pl[0]#0.0
				t0_off2 += (n_per*p2)**2*aa2_pl[1]#0.0
				rv_pl = business.rv_model(times[t_idxs],n_planet=pl,n_rv=nn,RM=False,t0_off=t0_off2)
	
			rv_m += rv_pl
		rv_m += v0



		for pl in pls:
			try:
				t0n = business.parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
				business.parameters['T0_{}'.format(pl)]['Value'] = t0n				
			except KeyError:
				pass

			per, T0 = business.parameters['P_{}'.format(pl)]['Value'], business.parameters['T0_{}'.format(pl)]['Value'] 
			ar, inc = business.parameters['a_Rs_{}'.format(pl)]['Value'], business.parameters['inc_{}'.format(pl)]['Value']*np.pi/180.
			rp = business.parameters['Rp_Rs_{}'.format(pl)]['Value']
			ecc, ww = business.parameters['e_{}'.format(pl)]['Value'], business.parameters['w_{}'.format(pl)]['Value']*np.pi/180.
			b = ar*np.cos(inc)*(1 - ecc**2)/(1 + ecc*np.sin(ww))

			t14 = per/np.pi * np.arcsin( np.sqrt( ((1 + rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))

			t23 = per/np.pi * np.arcsin( np.sqrt( ((1 - rp)**2 - b**2))/(np.sin(inc)*ar)  )*np.sqrt(1 - ecc**2)/(1 + ecc*np.sin(ww))
			if np.isnan(t14): continue


			off_arr = np.round((times-T0)/per)
			n_pers = np.unique(off_arr)[0]

			t0_off = n_per*per*aa2_pl[0]#0.0
			t0_off += (n_per*per)**2*aa2_pl[1]#0.0


			model_slope = business.localRV_model(times,n_planet=pl,t0_off=t0_off)
		

			### HARD-CODED
			darks = business.lc_model(times,n_planet=pl,n_phot=1,t0_off=t0_off)


			idxs = [ii for ii in range(len(times))]
			if oots is None:
				oots = business.data['idxs_{}'.format(nn)]

			print('Using indices {} as out-of-transit spectra'.format(oots))

			its = [ii for ii in idxs if ii not in oots]	

			pp = dynamics.time2phase(times[its],per,T0)*24*per

			nvel = len(slope_data[times[0]]['vel'])
			vels = np.zeros(shape=(nvel,len(times)))
			oot_ccfs = np.zeros(shape=(nvel,len(oots)))
			avg_ccf = np.zeros(nvel)
			for ii, idx in enumerate(oots):
				time = times[idx]
				vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
				vels[:,idx] = vel
				no_peak = (vel > no_bump) | (vel < -no_bump)
				

				ccf = slope_data[time]['ccf']
				area = np.trapz(ccf,vel)
				ccf /= area	

				poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
				ccf -= vel*poly_pars[0] + poly_pars[1]

				oot_ccfs[:,ii] = ccf
				avg_ccf += ccf

			avg_ccf /= len(oots)
			rvs = np.array([])
			errs = np.array([])

			lam = business.parameters['lam_{}'.format(pl)]['Value']*np.pi/180
			vsini = business.parameters['vsini']['Value'] 


			fit_params = business.lmfit.Parameters()
			fnames = []
			xs = np.array([])
			ys = np.array([])
			for ii, idx in enumerate(its):
				time = times[idx]
				vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
				vels[:,idx] = vel
				no_peak = (vel > no_bump) | (vel < -no_bump)

				cos_f, sin_f = dynamics.true_anomaly(time, T0+t0_off, ecc, per, ww)
				xx, yy = dynamics.xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
				xs = np.append(xs,xx)
				ys = np.append(ys,yy)

				ccf = slope_data[time]['ccf']
				area = np.trapz(ccf,vel)
				ccf /= area
				
				sd = np.std(ccf[no_peak])

				ccf *= darks[idx]#/bright#blc[ii]		
				shadow = avg_ccf - ccf
				poly_pars = np.polyfit(vel[no_peak],shadow[no_peak],1)
				
				shadow -=  vel*poly_pars[0] + poly_pars[1]

				peak = np.where((vel > -no_bump) & (vel < no_bump))
				#print(vel)
				#peak = np.where((vel > (xx*vsini - vsini/2)) & (vel < (xx*vsini + vsini/2)))
				midx = np.argmax(shadow[peak])
				amp, mu1 = shadow[peak][midx], vel[peak][midx]# get max value of CCF and location

				gau_par, pcov = curve_fit(Gauss,vel,shadow,p0=[amp,xx*vsini,0.2])
				perr = np.sqrt(np.diag(pcov))
				rv = gau_par[1]
				std = perr[1]



				rvs = np.append(rvs,rv)
				errs = np.append(errs,std)
				if movie_time:
					print('Making movie shadow.mp4 - this may take a while')
					movie_fig = plt.figure()
					movie_ax = movie_fig.add_subplot(111)
					movie_ax.axhline(0.0,color='C7',linestyle='--')
					movie_ax.plot(vel,shadow,'k',lw=2.0)
					movie_ax.plot(vel,shadow,'C0',lw=1.5)
					movie_ax.axvline(xx*vsini,color='C1',linestyle='-')
					#movie_ax.plot(vel,Gauss(vel,gau_par[0],gau_par[1],gau_par[2]),'k',lw=2.0)
					#movie_ax.plot(vel,Gauss(vel,gau_par[0],gau_par[1],gau_par[2]),'C7',lw=1.5)

					movie_ax.set_xlabel(r'$\rm Velocity \ (kms/s)$',fontsize=font)
					movie_ax.set_ylabel(r'$\rm Shadow$',fontsize=font)
					#movie_ax.text(min(vel)+1,0.95,r'$\rm Hours \ From \ Midtransit \ {:.3f}$'.format(pp[ii]),fontsize=font)
					fname = 'shadow_no_{:03d}.png'.format(ii)
					fnames.append(fname)
					movie_ax.set_ylim(-0.0004,0.001)
					movie_fig.savefig(fname)
					plt.close()

			if movie_time:
				import subprocess
				import os
				subprocess.call("ffmpeg -framerate 4 -i ./shadow_no_%3d.png -c:v libx264 -r 30 -pix_fmt yuv420p ./shadow.mp4", shell=True)
				for fname in fnames: os.remove(fname)

			slope = business.localRV_model(times[its])
			#print(xs)
			vsini = business.parameters['vsini']['Value']
			rv_scale = rvs/vsini
			erv_scale = errs/vsini
			chi2scale = business.data['Chi2 SL_{}'.format(nn)]
			#chi2scale = 1.0#business.data['Chi2 SL_{}'.format(nn)]
			#erv_scale *= chi2scale
			full = (pp > -1*t23*24/2) & (pp < 1*t23*24/2)
			part = (pp < -1*t23*24/2) | (pp > 1*t23*24/2)
			erv_scale[full] *= chi2scale
			erv_scale[part] *= chi2scale*1.5

			
			print('## Spectroscopic system {}/{} ##:'.format(nn,label))
			red_chi2 = np.sum((rv_scale - slope)**2/erv_scale**2)/(len(rv_scale)-n_pars)
			print('\nReduced chi-squared for the slope is:\n\t {:.03f}'.format(red_chi2))
			print('Factor to apply to get a reduced chi-squared around 1.0 is:\n\t {:.03f}\n'.format(np.sqrt(red_chi2)))
			print('Number of data points: {}'.format(len(rv_scale)))
			print('Number of fitting parameters: {}'.format(n_pars))
			print('#########################'.format(nn))


			if get_vp:
				arr = np.zeros(shape=(len(rv_scale),3))
				ss = np.argsort(pp)
				arr[:,0] = times[its][ss]
				arr[:,1] = rv_scale[ss]
				arr[:,2] = erv_scale[ss]
				return arr

			fig = plt.figure()
			ax = fig.add_subplot(211)
			ax2 = fig.add_subplot(212)
			ax.errorbar(pp,rv_scale,yerr=erv_scale,marker='o',markersize=6.0,color='k',linestyle='none',zorder=4)
			ax.errorbar(pp,rv_scale,yerr=erv_scale,marker='o',markersize=4.0,color='C{}'.format(nn-1),linestyle='none',zorder=5)
		
			ax.axhline(0.0,color='C7',zorder=-1,linestyle='--')
			ax.axhline(1.0,color='C0',zorder=-1,linestyle='--')
			ax.axhline(-1.0,color='C0',zorder=-1,linestyle='--')
			ax.axvline(-1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)
			ax.axvline(1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)

			ax.axvline(1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)
			ax.axvline(-1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)

			ax2.axvline(-1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)
			ax2.axvline(1*t23*24/2,linestyle='--',color=contact_color,lw=2.0)

			ax2.axvline(1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)
			ax2.axvline(-1*t14*24/2,linestyle='-',color=contact_color,lw=2.0)

			slope = business.localRV_model(times[its],n_planet=pl)
			ax.plot(pp,slope,'-',color='k',lw=2.0)
			ax.plot(pp,slope,'-',color='C7',lw=1.0)
			ax.set_ylabel(r'$\mathrm{Local} \ \mathrm{RV} \ (v\sin i)$',fontsize=font)
			ax2.set_xlabel(r'$\rm Hours \ From \ Midtransit$',fontsize=font)
			ax2.set_ylabel(r'$\rm Residuals$',fontsize=font)

			ax2.errorbar(pp,rv_scale-slope,yerr=erv_scale,marker='o',markersize=6.0,color='k',linestyle='none',zorder=4)
			ax2.errorbar(pp,rv_scale-slope,yerr=erv_scale,marker='o',markersize=4.0,color='C{}'.format(nn-1),linestyle='none',zorder=5)
			ax2.axhline(0.0,color='C7',zorder=-1,linestyle='--')

			slopes['RV_'+str(nn)]['pl_'+pl] = [pp,rv_scale,erv_scale,slope,xs,ys]

			plt.subplots_adjust(wspace=0.0,hspace=0.0)

			if savefig: plt.savefig(path+'slope.png')
	if return_slopes:
		return slopes

# =============================================================================
# Radial velocity periodogram
# =============================================================================


def plot_rv_pgram(param_fname,data_fname,updated_pars=None,savefig=False,path='',pls=None,
	freq_grid=None,samples_per_peak=5,savefile=False,best_fit=True):#,
#	xminLS=0.0,xmaxLS=None):

	plt.rc('text',usetex=plot_tex)

	font = 15
	plt.rc('xtick',labelsize=3*font/4)
	plt.rc('ytick',labelsize=3*font/4)


	bms = 6.0 # background markersize
	fms = 4.0 # foreground markersize
	tms = 40.0 # triangle markersize
	flw = 1.3 # freq linewidth

	business.data_structure(data_fname)
	business.params_structure(param_fname)

	if updated_pars is not None:
		pars = business.parameters['FPs']
		pars = updated_pars.keys()[1:-2]
		if n_pars == 0: n_pars = len(pars)
		idx = 1
		if (updated_pars.shape[0] > 3) & best_fit: idx = 4
		for par in pars:
			try:
				business.parameters[par]['Value'] = float(updated_pars[par][idx])	
			except KeyError:
				pass

	n_rv = business.data['RVs']
	if not pls:
		pls = business.parameters['Planets']

	if n_rv >= 1:
		aa = [business.parameters['a{}'.format(ii)]['Value'] for ii in range(1,3)]
		fig = plt.figure(figsize=(8,8))
		figls = plt.figure(figsize=(8,8))
		#figls_phase = plt.figure(figsize=(8,8))
		n_subs = len(pls) + 1
		if any(np.asarray(aa) != 0): n_subs += 1
		axes_rvs = []
		axes_ls = []
		#axes_phase = []
		for ii in range(1,n_subs+1):
			axes_rvs.append(fig.add_subplot(n_subs,1,ii))
			axes_ls.append(figls.add_subplot(n_subs,1,ii))


		times, rvs, rv_errs = np.array([]), np.array([]), np.array([])
		for nn in range(1,n_rv+1):
			arr = business.data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			times, rvs, rv_errs = np.append(times,time), np.append(rvs,rv), np.append(rv_errs,rv_err)

		zp = np.amin(times)

		RMs = []
		all_times, all_rvs, all_errs = np.array([]), np.array([]), np.array([])
		idxs = np.array([],dtype=np.int)
		all_rvs_signal_removed = np.array([])
		ins_idxs = np.array([])


		for nn in range(1,n_rv+1):
			#label = business.data['RV_label_{}'.format(nn)]
			arr = business.data['RV_{}'.format(nn)]
			time, rv, rv_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()
			v0 = business.parameters['RVsys_{}'.format(nn)]['Value']
			jitter = business.parameters['RVsigma_{}'.format(nn)]['Value']
			#jitter = np.exp(log_jitter)
			#jitter = log_jitter
			jitter_err = np.sqrt(rv_err**2 + jitter**2)

			drift = aa[1]*(time-zp)**2 + aa[0]*(time-zp)
			#RM = business.data['RM RV_{}'.format(nn)]


			all_times = np.append(all_times,time)
			idxs = np.append(idxs,np.ones(len(time))*nn)
			all_rvs = np.append(all_rvs,rv-v0)
			all_errs = np.append(all_errs,jitter_err)
			all_rvs_signal_removed = np.append(all_rvs_signal_removed,rv-v0-drift)


		#### REMEMBER TO INSTALL CHECK FOR RM FOR NON-TRANSITING PLANETS ####
		#### FOR NOW RM SIGNAL IS NOT INCLUDED IN THE PLOTTED MODEL ###
		#### IT IS HOWEVER PROPERLY REMOVED FROM THE RVS ###


		npoints = 50000
		unp_m = np.linspace(min(all_times)-10.,max(all_times)+10.,npoints)
		model_rvs = np.zeros(npoints)
		temp_rvs = aa[1]*(unp_m-zp)**2 + aa[0]*(unp_m-zp)
		if any(temp_rvs != 0.0):
			ax = axes_rvs[0]
			ax.plot(unp_m,temp_rvs,'-',color='k',lw=1.0,zorder=-1)
		model_rvs += temp_rvs


		ax0 = axes_rvs[0]
		ax0.errorbar(all_times,all_rvs,yerr=all_errs,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
		for nn in range(1,n_rv+1):
			RM = business.data['RM RV_{}'.format(nn)]
			label = business.data['RV_label_{}'.format(nn)]
			idx = nn == idxs
			ax0.errorbar(all_times[idx],all_rvs[idx],yerr=all_errs[idx],marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5,label=r'$\rm {}$'.format(label))
		ax0.legend(bbox_to_anchor=(0, 1.05, 1, 0),ncol=n_rv)
		freqs = []
		for ii, pl in enumerate(pls): 
			per = business.parameters['P_{}'.format(pl)]['Value']
			freqs.append(1/per)
			temp_rvs = business.rv_model(unp_m,n_planet=pl,n_rv=1,RM=False)
			ax0.plot(unp_m,temp_rvs,'-',color=colors[pl],lw=1.0,zorder=-1)
			model_rvs += temp_rvs

		ax0.plot(unp_m,model_rvs,'-',color='k',lw=2.0,zorder=-1)
		ax0.plot(unp_m,model_rvs,'-',color='C7',lw=1.0,zorder=0)		

		max_freq = max(freqs)#*2.0
		ax0ls = axes_ls[0]
		LS = LombScargle(all_times, all_rvs, dy=all_errs)
		if freq_grid is None:
			frequency, power = LS.autopower(maximum_frequency=max_freq*1.5,samples_per_peak=samples_per_peak)
		else:			
			power = LS.power(freq_grid)
			frequency = freq_grid
		FAP = LS.false_alarm_probability(power.max())


		midx = np.argmax(power)
		mper = frequency[midx]
		ax0ls.plot(frequency,power,'-',color='k',lw=flw)
		ax0.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
		y1,y2 = ax0ls.get_ylim()
		x1,x2 = ax0ls.get_xlim()
		ax0ls.set_ylabel(r'$\rm LS \ power$',fontsize=font)
		ax0ls.text(0.7*x2,0.8*y2,r'$P_{} = {:0.1f} \ \rm d$'.format('\mathrm{max}',1/mper),color='k',bbox=dict(edgecolor='k',facecolor='w'))
		ax0ls.scatter(mper,y2,marker='v',facecolor='C7',edgecolor='k',s=tms,zorder=5)

		ii = 1
		if any(np.asarray(aa) != 0):
			ax = axes_rvs[ii]
			axls = axes_ls[ii]
			ii += 1
			ax.errorbar(all_times,all_rvs_signal_removed,yerr=all_errs,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
			for nn in range(1,n_rv+1):
				idx = nn == idxs
				ax.errorbar(all_times[idx],all_rvs_signal_removed[idx],yerr=all_errs[idx],marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)
			LS2 = LombScargle(all_times, all_rvs_signal_removed, dy=all_errs)
			if freq_grid is None:
				frequency, power = LS2.autopower(maximum_frequency=max_freq*1.5,samples_per_peak=samples_per_peak)
			else:			
				power = LS.power(freq_grid)
				frequency = freq_grid

			axls.plot(frequency,power,'-',color='k',lw=flw)
			ax.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
			axls.set_ylabel(r'$\rm LS \ power$',fontsize=font)
			model_rvs -= aa[1]*(unp_m-zp)**2 + aa[0]*(unp_m-zp)
			ax.plot(unp_m,model_rvs,'-',color='k',lw=2.0,zorder=-1)
			ax.plot(unp_m,model_rvs,'-',color='C7',lw=1.0,zorder=0)
			


			for kk, pl in enumerate(pls): 
				temp_rvs = business.rv_model(unp_m,n_planet=pl,n_rv=1,RM=False)
				ax.plot(unp_m,temp_rvs,'-',color=colors[pl],lw=1.0,zorder=-1)

		pers = []
		removed_pls = []
		for jj, pl in enumerate(pls):
			ax = axes_rvs[ii]
			axls = axes_ls[ii]
			axls_vert = axes_ls[ii-1]
			ii += 1
			per = business.parameters['P_{}'.format(pl)]['Value']
			pers.append(per)

			
			for nn in range(1,n_rv+1):
				label = business.data['RV_label_{}'.format(nn)]
				arr = business.data['RV_{}'.format(nn)]
				time = arr[:,0].copy()
				RM = business.data['RM RV_{}'.format(nn)]
				idx = nn == idxs
				all_rvs_signal_removed[idx] -= business.rv_model(all_times[idx],n_planet=pl,n_rv=nn,RM=RM)
				ax.errorbar(all_times[idx],all_rvs_signal_removed[idx],yerr=all_errs[idx],marker='o',markersize=fms,color='C{}'.format(nn-1),linestyle='none',zorder=5)
			ax.errorbar(all_times,all_rvs_signal_removed,yerr=all_errs,marker='o',markersize=bms,color='k',linestyle='none',zorder=4)
			LS3 = LombScargle(all_times, all_rvs_signal_removed, dy=all_errs)
			if freq_grid is None:
				frequency, power = LS3.autopower(maximum_frequency=max_freq*1.5,samples_per_peak=samples_per_peak)
			else:			
				power = LS.power(freq_grid)
				frequency = freq_grid

			axls.plot(frequency,power,'-',color='k',lw=flw)
			midx = np.argmax(power)
			mper = frequency[midx]
			y1,y2 = axls.get_ylim()
			x1,x2 = axls.get_xlim()
			axls.text(0.7*x2,0.8*y2,r'$P_{} = {:0.1f} \ \rm d \ removed$'.format(pl,per),color=colors[pl],bbox=dict(edgecolor='k',facecolor='w'))
			axls.text(0.7*x2,0.4*y2,r'$P_{} = {:0.1f} \ \rm d$'.format('\mathrm{max}',1/mper),color='C7',bbox=dict(edgecolor='k',facecolor='w'))
			#axls.axvline(mper,color='C7',zorder=-1)
			#axls_vert.axvline(1/per,color='C{}'.format(jj),zorder=-1)
			y1_vert,y2_vert = axls_vert.get_ylim()
			axls.scatter(mper,y2,marker='v',facecolor='C7',edgecolor='k',s=tms,zorder=5)
			axls_vert.scatter(1/per,y2_vert,marker='v',facecolor=colors[pl],edgecolor='k',s=tms,zorder=6)

			#ax0ls.axvline(1/per,color='C{}'.format(jj))
			ax.set_ylabel(r'$\rm RV \ (m/s)$',fontsize=font)
			axls.set_ylabel(r'$\rm LS \ power$',fontsize=font)
			removed_pls.append(pl)
			model_rvs = np.zeros(npoints)
			for kk, pl2 in enumerate(pls):
				if pl2 not in removed_pls:
					temp_rvs = business.rv_model(unp_m,n_planet=pl2)
					model_rvs += temp_rvs
					ax.plot(unp_m,temp_rvs,'-',color=colors[pl2],lw=1.0,zorder=-1)
				else:
					per = business.parameters['P_{}'.format(pl2)]['Value']
					axls.scatter(1/per,0.0,marker='^',facecolor=colors[pl2],edgecolor='k',s=tms,zorder=5)


			ax.plot(unp_m,model_rvs,'-',color='k',lw=2.0,zorder=-1)
			ax.plot(unp_m,model_rvs,'-',color='C7',lw=1.0,zorder=0)

		if savefile:
			labels = []
			ii = 1
			for nn in range(1,n_rv+1):
				idx = nn == idxs
				label = business.data['RV_label_{}'.format(nn)]
				if label in labels: 
					label += str(ii)
					ii += 1
				labels.append(label)
				tt, rr, ee = all_times[idx], all_rvs_signal_removed[idx], all_errs[idx]
				arr = np.zeros(shape=(len(tt),3))
				arr[:,0] = tt
				arr[:,1] = rr
				arr[:,2] = ee
				ll = label.replace(' ','')
				np.savetxt(ll+'_rvs_signal_removed.txt',arr)

		for jj, pl in enumerate(pls):
			per = business.parameters['P_{}'.format(pl)]['Value']
			axls.scatter(1/per,0.0,marker='^',facecolor=colors[pl],edgecolor='k',s=tms,zorder=5)
		# 	#axls.axvline(1/per,linestyle='--',color='C{}'.format(kk),zorder=-1)
	
		ax.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)
		axls.set_xlabel(r'$\rm Frequency \ (c/d)$',fontsize=font)
		fig.tight_layout()
		fig.subplots_adjust(hspace=0.0)
		figls.tight_layout()
		figls.subplots_adjust(hspace=0.0)
		if savefig: fig.savefig(path+'rvs_subtracted.pdf')
		if savefig: figls.savefig(path+'rv_periodogram.pdf')

# =============================================================================
# Light curve periodogram
# =============================================================================

def plot_lc_pgram(param_fname,data_fname,updated_pars=None,savefig=False,
	path='',pls=None,tls = False,best_fit=True):#,
#	xminLS=0.0,xmaxLS=None):
	'''Periodogram from light curves.

	'''


	plt.rc('text',usetex=plot_tex)

	font = 15
	plt.rc('xtick',labelsize=3*font/4)
	plt.rc('ytick',labelsize=3*font/4)


	bms = 6.0 # background markersize
	fms = 4.0 # foreground markersize
	tms = 40.0 # triangle markersize
	flw = 1.3 # freq linewidth
	blw = 1.5 # back linewidth
	plw = 1.0 # planet linewidth

	business.data_structure(data_fname)
	business.params_structure(param_fname)

	if updated_pars is not None:
		#pars = updated_pars.keys()[1:]
		pars = business.parameters['FPs']
		for par in pars:
			if best_fit:
				business.parameters[par]['Value'] = float(updated_pars[par][4])
			else:
				business.parameters[par]['Value'] = float(updated_pars[par][1])		
	n_phot = business.data['LCs']
	if not pls:
		pls = business.parameters['Planets']

	
	if tls: from transitleastsquares import transitleastsquares as transitls

	if n_phot >= 1:

		npoints = 10000
		fig = plt.figure(figsize=(8,8))
		figls = plt.figure(figsize=(8,8))
		figp = plt.figure()

		n_pls = len(pls) + 1
		axes = []
		axesls = []
		axesp = []
		for ii in range(2): axesp.append(figp.add_subplot(2,1,ii+1))
		if tls:
			figtls = plt.figure()
			axestls = figtls.add_subplot(111)
		for nn in range(n_pls):
			axesls.append(figls.add_subplot(n_pls,1,nn+1))
			axes.append(fig.add_subplot(n_pls,1,nn+1))

		times, fluxs, flux_errs = np.array([]), np.array([]), np.array([])
		idxs = np.array([])
		#ii = 0
		ax = axes[0]
		for nn in range(1,n_phot+1):
			arr = business.data['LC_{}'.format(nn)]
			time, flux, flux_err = arr[:,0].copy(), arr[:,1].copy(), arr[:,2].copy()

			ax.plot(time,flux,'.',color='k',markersize=bms)
			ax.plot(time,flux,'.',color='C{}'.format(nn-1),markersize=fms)
			times, fluxs, flux_errs = np.append(times,time), np.append(fluxs,flux), np.append(flux_errs,flux_err)
			idxs = np.append(idxs,np.ones(len(time))*nn)		
			#ii += 1

		LS = LombScargle(times, fluxs, dy=flux_errs)
		frequency, power = LS.autopower()
		FAP = LS.false_alarm_probability(power.max())
		mper = 1/frequency[np.argmax(power)]

		#axls.text(0.7*max(frequency),0.8*max(power),r'$P_{} = {:0.1f} \ \rm d \ removed$'.format(pl,per),color=colors[pl],bbox=dict(edgecolor='k',facecolor='w'))
		axesls[0].axvline(mper,lw=flw,color='C7')
		axesls[0].semilogx(1/frequency,power,'-',lw=flw,color='k')
		axesp[0].loglog(frequency,power,'-',lw=flw,color='k')
		axesp[0].set_ylabel(r'$\rm LS \ Power$',fontsize=font)
		axesp[1].set_ylabel(r'$\rm LS \ Power$',fontsize=font)
		axesls[0].set_ylabel(r'$\rm LS \ Power$',fontsize=font)
		axes[0].set_ylabel(r'$\rm Rel. \ Int.$',fontsize=font)


		mts = np.linspace(min(times)-1.0,max(times)+1.0,npoints)
		lcs = np.ones(npoints)
		pers = []
		for kk,pl in enumerate(pls):
			#for nn in range(1,n_phot+1):
			lc_pl = business.lc_model(mts,n_planet=pl,n_phot=1)
			#lcs += 1.0 - lc_pl
			ax.plot(mts,lc_pl,'-',color='k',lw=blw)
			ax.plot(mts,lc_pl,'-',color=colors[pl],lw=plw)
			per = business.parameters['P_{}'.format(pl)]['Value']
			axesls[0].axvline(per,color=colors[pl])
			pers.append(per)

		#axesls[0].set_xlim(0.0,max(pers)+1.05*max(pers))
		axesls[0].text(0.7*(max(pers)+max(pers)),0.8*max(power),r'$P_{} = {:0.1f} \ \rm d $'.format('\mathrm{max}',mper),color='C7',bbox=dict(edgecolor='k',facecolor='w'))



		removed_pls = []
		ii = 1
		for pl in pls:
			ax = axes[ii]
			tt_sub, fl_sub, er_sub = np.array([]), np.array([]), np.array([])
			for nn in range(1,n_phot+1):
				
				idx = nn == idxs
				lc_pl = business.lc_model(times[idx],n_planet=pl,n_phot=nn)
				fluxs[idx] = fluxs[idx] - lc_pl + 1.0	

				ax.plot(times[idx],fluxs[idx],'.',color='k',markersize=bms)
				ax.plot(times[idx],fluxs[idx],'.',color='C{}'.format(nn-1),markersize=fms)
			removed_pls.append(pl)
			#for aa, pl2 in enumerate(removed_pls):


			for aa, pl2 in enumerate(pls):
				axesls[ii].axvline(business.parameters['P_{}'.format(pl2)]['Value'],linestyle='--',color=colors[pl2])
				if pl2 not in removed_pls:
					lc_pl = business.lc_model(mts,n_planet=pl2,n_phot=1)
					axesls[ii].axvline(business.parameters['P_{}'.format(pl2)]['Value'],linestyle='-',color=colors[pl2])
					ax.plot(mts,lc_pl,'-',color='k',lw=blw)
					ax.plot(mts,lc_pl,'-',color=colors[pl2],lw=plw)#8-len(removed_pls)))

			LS = LombScargle(times, fluxs, dy=flux_errs)
			frequency, power = LS.autopower()
			FAP = LS.false_alarm_probability(power.max())
			axesls[ii].semilogx(1/frequency,power,'-',lw=flw,color='k')			
			#axesls[ii].set_xlim(0.0,max(pers)+1.05*max(pers))
			axesls[ii].set_ylabel(r'$\rm LS \ Power$',fontsize=font)
			ax.set_ylabel(r'$\rm Rel. \ Int.$',fontsize=font)

			mper = 1/frequency[np.argmax(power)]
			axesls[ii].axvline(mper,lw=flw,color='C7')
			axesls[ii].text(0.7*(max(pers)+max(pers)),0.8*max(power),r'$P_{} = {:0.1f} \ \rm d$'.format('\mathrm{max}',mper),color='C7',bbox=dict(edgecolor='k',facecolor='w'))
			axesls[ii].text(0.7*(max(pers)+max(pers)),0.4*max(power),r'$P_{} = {:0.1f} \ \rm d \ removed$'.format(pl,per),color=colors[pl],bbox=dict(edgecolor='k',facecolor='w'))

			ii += 1

			# 	#os = np.argsort(tt)
		for pl in pls:
			per = business.parameters['P_{}'.format(pl)]['Value']
			axesp[0].axvline(1/per,linestyle='-',lw=flw,color=colors[pl])
			axesp[1].axvline(1/per,linestyle='--',lw=flw,color=colors[pl])
		axesp[1].loglog(frequency,power,'-',lw=flw,color='k')
		axesp[1].set_xlim(min(frequency),max(frequency))
		axesp[0].set_xlim(min(frequency),max(frequency))

		axesls[-1].set_xlabel(r'$\rm Period \ (d)$',fontsize=font)
		axesp[-1].set_xlabel(r'$\rm Frequency \ (c/d)$',fontsize=font)
		for ii in range(len(axesls)-1):
			axesls[ii].set_xticks([])


		ax.set_xlabel(r'$\rm Time \ (BJD)$',fontsize=font)
		fig.tight_layout()
		fig.subplots_adjust(hspace=0.0)
		figls.tight_layout()
		figls.subplots_adjust(hspace=0.0)
		figp.subplots_adjust(hspace=0.0)
		if savefig:
			fig.savefig('full_lc.pdf')
			figls.savefig('LS_period.pdf')
			figp.savefig('LS_freq.pdf')

		if tls:
			import seaborn as sns
			blues = sns.color_palette("Blues")

			c1 = business.parameters['LC1_q1']['Value']
			c2 = business.parameters['LC1_q2']['Value']

			model = transitls(times,fluxs,flux_errs)
			results = model.power(oversampling_factor=2,
					limb_dark='quadratic', u=[c1,c2])
			per = results.period
			axestls.plot(results.periods,results.power,'k',lw=flw)
			axestls.set_xlabel(r'$\rm Period \ (days)$')
			axestls.set_ylabel(r'$\rm SDE$')
			axestls.set_xlim(np.amin(results.periods),np.amax(results.periods))
			axestls.axvline(per,color=blues[2],lw=3,zorder=-1)
			for nn in range(2,30):
				axestls.axvline(nn*per,color=blues[0],ls='--',zorder=-2)
				axestls.axvline(per/nn,color=blues[0],ls='--',zorder=-2)
			if savefig: figtls.savefig('TLS_result.pdf')


