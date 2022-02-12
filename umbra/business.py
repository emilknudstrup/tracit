#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:02:15 2021

@author: emil
"""

'''
.. todo::
	* Make sure data isn't passed around with each evaluation 
		- global variables? CHECK, seems to work
		- data class/object? REDUNDANT
	
	* Include fit to binaries - wrap ellc?

	* Find common orbital solution between batman/orbits
		i. best case: we are just solving Kepler's eq. twice
		ii. worst case: we are not getting consistent results between the two
		iii. use batman.TransitModel().get_true_anomaly()?
		iv. switch entirely to ellc?
	
	* Object oriented - just do it
	
	* Restructure - move .csv generator and reader to different module

	* Check imported packages, redundancy
'''
# =============================================================================
# umbra modules
# =============================================================================

# import umbra.dynamics as dynamics
# import umbra.shady as shady
# import umbra.expose as expose
# import umbra.stat_tools as stat_tools
# import umbra.constants as constants

from umbra import expose
from umbra import stat_tools
from umbra import dynamics
from umbra import shady
from umbra.priors import tgauss_prior, gauss_prior, flat_prior, tgauss_prior_dis, flat_prior_dis


# =============================================================================
# external modules
# =============================================================================

import lmfit
import emcee
import arviz as az
from multiprocessing import Pool
import celerite

import glob
import numpy as np
import pandas as pd
import h5py
#import matplotlib.pyplot as plt
import string
import batman

from scipy import interpolate
#from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
import scipy.signal as scisig
from statsmodels.nonparametric.kde import KDEUnivariate as KDE

#def run_params(nproc):
def run_sys(nproc):
	#global plot_tex
	global mpath
	expose.run_sys(nproc)
	if nproc > 4:
		#plot_tex = False
		mpath = './'
	else:
		#plot_tex = True
		mpath = '/home/emil/Desktop/PhD/exoplanets'

def str2bool(arg):
	if isinstance(arg, bool):
		return arg
	if arg.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise TypeError('Boolean value expected.')

def gaussian(pars,y,x):
	vals = pars.valuesdict()
	amp = vals['amp']
	mu = vals['mu']
	sig = vals['sig']

	return y - amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def Gauss(x, amp, mu,sig ):
	y = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	return y


def params_temp(filename='parameter_template.csv',
				n_phot=1,n_spec=1,n_planets=1,
				LD_law='quad'):
	'''Generate a .csv file template for the parameters.
	
	Generate a .csv file to put in the parameters for the planet/instrument.
	
	:param filename: Name for the parameter .csv file. Default `'parameter_template.csv'`.
	:type filename: str, optional

	:param n_phot: Number of photometric systems. Default 1.
	:type n_phot: int
	
	:param n_spec: Number of spectroscopic systems. Default 1.
	:type n_spec: int

	:param n_phot: Number of planets in system. Default 1.
	:type n_spec: int

	:param LD_law: limb darkening law used. Default `'quad'`. See :py:class:`dynamics.StellarParams`.
	:type LD_law: str    

	'''

	LD_laws = ['uni','quad','nonlinear']
	assert LD_law in LD_laws, print('Limb darkening law must be in {}'.format(LD_laws))

	
	abc = list(string.ascii_lowercase)[1:]
	ABC = list(string.ascii_uppercase)

	orbpars = ['P','T0','e','w','Rp_Rs','a_Rs','inc','K','Tw','lam','b','ecosw','esinw','T14','T23','T0_a2','T0_a1']
	stelpars = ['vsini','zeta','xi']
	rows = {'Handle' : ['Parameter','Unit','Label',
						'Value','Uncertainty','Lower','Upper',
						'Prior','Start','Fit parameter','Fix to']}	
	
	ncols = 62	
	cols = ['Planet parameters'] + [ii for ii in range(ncols)]
	for ii in range(n_planets): cols[ii+1] = abc[ii]
	n_rows = len(rows['Handle'])+1
	df = pd.DataFrame(columns=cols)
	nn = 1
	dfn = pd.DataFrame(rows)
	default = ['uni','tgauss','false','none']
	for ii in range(n_planets):
		pl = abc[ii]
		df.loc[nn] = ['Planet {}'.format(pl)] + ['##### PLANET {} #####'.format(pl.lower())]*ncols#len(orbpars)
		pars = {'P_{}'.format(pl) : 
			['Period','days',r'$P \rm _{} \  (days)$'.format(pl),9.0,0.5,5.0,20.0],	
			'T0_{}'.format(pl) : 
			['Midtransit','BJD',r'$T \rm _{} \ (BJD)$'.format('{0,'+pl+'}'),2457000.,0.5,2456999.,2457001],
			'e_{}'.format(pl) : 
			['Eccentricity',' ',r'$e \rm _{}$'.format(pl),0.0,0.1,0.0,1.0],
			'w_{}'.format(pl) : 
			['Omega','deg',r'$\omega \rm _{} \ (^\circ)$'.format(pl),90.,10.,0.0,360.],
			'Rp_Rs_{}'.format(pl) : 
			['Radius ratio','Rs',r'$(R_{}/R_\star)\rm _{}$'.format('\mathrm{p}',pl),0.1,0.05,0.0,1.0],
			'a_Rs_{}'.format(pl) : 
			['Semi-major axis','Rs',r'$(a/R_\star) \rm _{}$'.format(pl),30.,1.0,1.0,50.],
			'inc_{}'.format(pl) : 
			['Inclination','deg',r'$i \rm _{} \ (^\circ)$'.format(pl),90.,1.0,0.0,90.],
			'K_{}'.format(pl) : 
			['K-amplitude','m/s',r'$K  \rm _{}\ (m/s)$'.format(pl),30.,1.0,5.0,50.],
			'Tw_{}'.format(pl) : 
			['Time of periastron passage','BJD',r'$T_\omega  \rm _{} \ (BJD)$'.format(pl),2457000.,0.5,2456999.,2457001],
			'lam_{}'.format(pl) : 
			['Projected obliquity','deg',r'$\lambda \rm _{} \ (^\circ)$'.format(pl),0.0,10.,0.0,360.],
			'cosi_{}'.format(pl) : 
			['Cosine inclination',' ',r'$\cos i \rm _{}$'.format(pl),0.0,0.1,0.0,1.0],
			'ecosw_{}'.format(pl) : 
			['sqrt(e) cos(w)',' ',r'$\sqrt{e} \ \cos \omega_'+pl+'$',0.0,0.1,-1.,1.],
			'esinw_{}'.format(pl) : 
			['sqrt(e) sin(w)',' ',r'$\sqrt{e} \ \sin \omega_'+pl+'$',0.0,0.1,-1.,1.],
			'T41_{}'.format(pl) : 
			['T_{41}','hours',r'$T \rm _{41,'+pl+'} (hours)$',4.0,0.1,0.,10.],
			'T21_{}'.format(pl) : 
			['T_{21}','hours',r'$T \rm _{21,'+pl+'} (hours)$',3.0,0.1,0.,10.],
			'a2_{}'.format(pl) :
			['T0 curve',' ',r'$a \rm _{2,'+pl+'}$',0.0,0.01,0.,10.],
			'a1_{}'.format(pl) :
			['T0 slope',' ',r'$a \rm _{1,'+pl+'}$',0.0,0.01,0.,10.]
			}

		for par in pars.keys(): 
			for de in default:
				pars[par].append(de)

		for jj in range(len(orbpars),ncols): pars[jj] = [' ']*len(pars['P_{}'.format(pl)])


		dfp = pd.DataFrame(pars)
		nn += n_rows + 1
		tdf = pd.concat([dfn,dfp],axis=1)
		hh = tdf.columns
		tdf.columns = cols
		tdf.loc[0] = hh
		df = df.append(tdf)

	star_df = pd.DataFrame([['System parameters'] + ['******** STAR ********']*ncols])
	star_df.columns = cols
	df = df.append(star_df)

	star = {'vsini' : 
			['Projected stellar rotation velocity','km/s',r'$v \sin i_\star \ \rm (km/s)$',2.0,1.0,0.0,10.],
			'zeta' : 
			['Macroturbulence','km/s',r'$\zeta \ \rm (km/s)$',1.0,1.0,0.0,10.],
			'xi' : 
			['Microturbulence','km/s',r'$\xi \ \rm (km/s)$',1.0,1.0,0.0,10.]}

	def set_LD(LD_pars,LD_law,label):
		if LD_law == 'quad':
			LD_pars[label+'_q1'] = ['Linear LD coeff for {}'.format(label),'quadratic',r'$q_1: \ \rm {}$'.format(label),0.4,0.1,0.0,1.0]
			LD_pars[label+'_q2'] = ['Quadratic LD coeff for {}'.format(label),'quadratic',r'$q_2: \ \rm {}$'.format(label),0.3,0.1,0.0,1.0]
		elif LD_law == 'uni':
			LD_pars[label+'_q1'] = ['No coefficients','uniform']
		elif LD_law == 'nonlinear':
			for ii in range(1,5):
				LD_pars[label+'_q{}'.format(ii)] = ['LD coeff {}'.format(ii),'nonlinear',r'$q_2 \ \rm {}$'.format(label,ii)]

	add_lc_pars = {}
	for ii in range(1,n_phot+1):
		star['LCblend_{}'.format(ii)] = ['Dilution (deltaMag=-2.5log(F2/F1)) photometer {}'.format(ii),' ',r'$\rm \delta M{}$'.format('_{LC,'+str(ii)+'}'),0.0,0.5,0.0,7.0]
		star['LCsigma_{}'.format(ii)] = ['Log jitter photometer {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LC,'+str(ii)+'}'),-30,0.05,-50,1.0]
		star['LC_{}_GP_log_a'.format(ii)] = ['GP log amplitude, photometer {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{LC,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		star['LC_{}_GP_log_c'.format(ii)] = ['GP log exponent, photometer {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{LC,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		label = 'LC'+str(ii)
		set_LD(star,LD_law,label)

	add_rv_pars = {}
	for ii in range(1,n_spec+1):
		star['RVsys_{}'.format(ii)] = ['Systemic velocity instrument {}'.format(ii),'m/s',r'$\gamma_{} \ \rm (m/s)$'.format(ii),0.0,1.,-20.,20.]
		star['RVsigma_{}'.format(ii)] = ['Jitter RV instrument {}'.format(ii),'m/s',r'$\rm \sigma{} \ (m/s)$'.format('_{RV,'+str(ii)+'}'),0.0,1.0,0.0,100.]
		label = 'RV'+str(ii)
		set_LD(star,LD_law,label)

	star['a2'] = ['RV curve','m/(s*d^2)',r'$\ddot{\gamma} \ \rm (m \ s^{-1} \ d^{-2})$',0.0,1.0,-10.,10.]
	star['a1'] = ['RV slope','m/(s*d)',r'$\dot{\gamma} \ \rm (m \ s^{-1} \ d^{-1})$',0.0,1.0,-10.,10.]


	for par in star.keys(): 
		for de in default:
			star[par].append(de)
	
	for ii in range(len(star.keys()),ncols): star[ii] = [' ']*len(star['vsini'])

	sdf = pd.DataFrame(star)
	sdf = pd.concat([dfn,sdf],axis=1)
	hh = sdf.columns
	sdf.columns = cols
	sdf.loc[0] = hh
	df = df.append(sdf)

	dfn['Handle'][9] = 'Constrain parameter'
	xdf = pd.DataFrame([['External constraints'] + ['$$$$$$$$ X $$$$$$$$']*ncols])
	xdf.columns = cols
	df = df.append(xdf)

	ext = {'rho_s' : 
			['Stellar density','g/cm^3',r'$\rho_\star \ \rm (g/cm^{3})$',0.5,0.1,0.0,2.]}
	for par in ext.keys(): 
		for de in default:
			ext[par].append(de)
	for ii in range(len(ext.keys()),ncols): ext[ii] = [' ']*len(ext['rho_s'])
	xdf = pd.DataFrame(ext)
	xdf = pd.concat([dfn,xdf],axis=1)
	hh = xdf.columns
	xdf.columns = cols
	xdf.loc[0] = hh
	df = df.append(xdf)

	df.to_csv(filename,index=False)
	print('\n\n')
	print('Generated template file {} for input/fitting parameters.'.format(filename))
	print('Insert values in the rows: Value, Error, Upper, and Lower.')
	print('Insert choices for the type of prior and starting distribtuion in: Prior and Start.')
	print('Put in in true to fit a given parameter.')
	print('The label is the one displayed in plots - this can also be altered.')
	print('It is possible to step in cos(i) and sqrt(e)*cos(w) & sqrt(e)*sin(w), if:')
	print('\t-fit cos(i) is True, walkers will step in cos(i).')
	print('\t-fit (ecos(w) & esin(w)) is True, walkers will step in sqrt(e)*cos(w) & sqrt(e)*sin(w).')
	print('Warning: Do not touch the rows: Handle, Parameter, or Unit.')
	print('Warning: Make sure to format labels correctly, i.e., $label$.')
	print('\n\n')

def data_temp(filename='data_template.csv',n_phot=1,n_spec=1):
	'''Generate a .csv file template for the data.
	
	Generate a .csv file to put in the filenames for the data.
	
	:param filename: Name for the data .csv file. Default `'data_template.csv'`.
	:type filename: str, optional

	:param n_phot: Number of photometric systems. Default 1.
	:type n_phot: int
	
	:param n_spec: Number of spectroscopic systems. Default 1.
	:type n_spec: int

	'''
	ph = ['Handle','Light curve:','Instrument','LC Filename','Units',
		'Fit LC','Oversample factor','Exposure time (min.)','Detrend','Poly. deg./filt. w.','LC chi2 scaling',
		'Gaussian Process','GP type',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
	
	sp = ['Handle','Radial velocities:','Instrument','RV Filename','Units',
		'Fit RV','Fit RM','RV chi2 scaling','Lineshape:','Fit LS','LS Filename',
		'Units','Disk Resolution','Ring Thickness','LS chi2 scaling','Fit OOT',
		'Only fit OOT','OOT chi2 scaling','OOT indices','Slope','Fit SL','SL Filename','Units','SL chi2 scaling','Gaussian Process','GP type']

	df = pd.DataFrame(columns = ['Photometry']+['##--_--_--_--##']*(len(sp)-1))
	df.loc[0] = ph
	nn = 1
	for ii in range(1,n_phot+1): 
		df.loc[nn] = ['Phot_{}'.format(ii),'Single .txt file',' ','lc*{}.txt'.format(ii),
					'BJD & Relative brightness','true',1,2,'false',1,1,'false','Matern32',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
		nn += 1
	
	df.loc[nn] = ['Spectroscopy']+['##^vv^^\/^^^v##']*(len(ph)-1)
	nn += 1
	df.loc[nn] = sp
	nn += 1
	for ii in range(1,n_spec+1): 
		df.loc[nn] = ['Spec_{}'.format(ii),'Single .txt file',' ','rv*{}.txt'.format(ii),
					'BJD & m/s','true','false',1.0,'One .hdf5 file /bjd/[vel,ccf]',
					'false','*.hdf5','BJD & km/s',100,20,1.0,'false','false',1.0,'0,1,2',
					'One .hdf5 file /bjd/[vel,ccf]','false','*.hdf5','BJD & km/s',1.0]
		nn += 1

	df.to_csv(filename,index=False)
	print('\n\n')
	print('Generated template file {} to load in data from .txt files.'.format(filename))
	print('Data should be formatted in a .txt with columns: times flux/RV uncertainty.')
	print('The subscript number corresponds to the number given for parameters')
	print('that might differ from instrument to instrument.')
	print('It is possible not to fit the data in the file by putting in false.')
	print('It is possible to either detrend the photometric data just around transit and/or')
	print('account for the Rossiter-McLaughlin effect in the spectroscopic data if relevant.')
	print('Detrending is done by fitting a polynomial of the specified degree to out-of-transit data.')
	print('-1 corresponds to no detrending. The RM effect will be accounted for by putting in true.')
	print('\n\n')


def data_structure(file):
	'''Structure the data for :strike:`umbra`.

	Function that reads in the data .csv file, and structures the content in a dictionary.

	:param filename: Path to the parameter file, e.g., `./dat.csv`.
	:type filename: str

	.. note::
		Using global variable to prevent having to pickle and pass the data to the modules every time the code is called,
		see `emcee <https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments>`_'s documention.

	'''
	
	df = pd.read_csv(file,skip_blank_lines=True)
	split = np.where(df['Photometry'].str.find('Spectroscopy') == 0)[0][0]
	df_phot_sub, df_spec_sub = df[0:split], df[split+1:]
	phot_header, spec_header = df_phot_sub.iloc[0], df_spec_sub.iloc[0]

	n_phot = len(df_phot_sub['Photometry']) - 1
	n_spec = len(df_spec_sub['Photometry']) - 1

	df_phot, df_spec = df_phot_sub[1:n_phot+1], df_spec_sub[1:n_spec+1]
	spec_header.name = 0
	df_phot.columns, df_spec.columns = phot_header, spec_header

	df_spec.reset_index(drop=True,inplace=True)
	df_spec.index += 1

	global data 
	data = {}
	data['LCs'] = n_phot
	for ii in range(1,n_phot+1):
		data['LC_label_{}'.format(ii)] = df_phot['Instrument'][ii]
		data['Fit LC_{}'.format(ii)] = str2bool(df_phot['Fit LC'][ii])
		lfile = df_phot['LC Filename'][ii]
		if lfile.find('*') != -1: lfile = glob.glob(lfile)[0] 
		phot_arr = np.loadtxt(lfile)
		data['LC_{}'.format(ii)] = phot_arr
		data['OF LC_{}'.format(ii)] = int(df_phot['Oversample factor'][ii])
		data['Exp. LC_{}'.format(ii)] = float(df_phot['Exposure time (min.)'][ii])/(24*60)
		data['Chi2 LC_{}'.format(ii)] = float(df_phot['LC chi2 scaling'][ii])
		data['GP LC_{}'.format(ii)] = str2bool(df_phot['Gaussian Process'][ii])
		
		trend = df_phot['Detrend'][ii]
		detrends = ['poly','savitsky']
		try:
			trend = str2bool(trend)
		except TypeError:
			assert trend in detrends, print('Detrending must be {}'.format(detrends))
		data['Detrend LC_{}'.format(ii)] = trend

		deg_w = int(df_phot['Poly. deg./filt. w.'][ii])
		if (trend == detrends[0]) or (trend == True):
			if (deg_w == 1) or (deg_w == 2):
				data['Poly LC_{}'.format(ii)] = deg_w
			else:
				print('Put in 1 or 2 to choose between either')
				print('1st or 2nd order polynomial for detrending.')
		elif trend == detrends[1]:
			if deg_w%2:
				data['FW LC_{}'.format(ii)] = deg_w
			else:
				deg_w += 1
				print('Filter width for Savitsky-Filter must be odd.')
				print('Adding 1 to get a width of {} data points.'.format(deg_w))
				data['FW LC_{}'.format(ii)] = deg_w
		#else:
		elif data['GP LC_{}'.format(ii)]:
			gp_type = df_phot['GP type'][ii]
			assert gp_type in ['Real','Matern32'], print('Error: GP type needs to be [...,...]')
			data['GP type LC_{}'.format(ii)] = gp_type
			gp_bin = float(df_phot['GP bin (hours)'][ii])
			t_diff = np.median(np.diff(phot_arr[:,0]))
			data['GP Nbin LC_{}'.format(ii)] = int(gp_bin/(t_diff*24))

			# Set up the GP model
			try:
				loga = parameters['LC_{}_GP_log_a'.format(ii)]['Value']
				logc = parameters['LC_{}_GP_log_c'.format(ii)]['Value']
				#jitt = parameters['LCsigma_{}'.format(ii)]['Value']
			except NameError:
				loga = -0.3
				logc = -7.3
				#jitt = -10
			#log_jitter = celerite.terms.JitterTerm(log_sigma=jitt)
			if gp_type == 'Real':
				noise = celerite.terms.RealTerm(log_a=loga, log_c=logc)
			else:
				noise = celerite.terms.Matern32Term(log_sigma=loga, log_rho=logc)
			kernel = noise# + log_jitter

			gp = celerite.GP(kernel)

			gp.compute(phot_arr[:,0],phot_arr[:,2]) #probably redundant
			data['LC_{} GP'.format(ii)] = gp

		#if trend_poly > 0:
		#	time = phot_arr[;,0]

  
	data['RVs'] = n_spec
	n_ls = 0
	n_sl = 0
	for ii in range(1,n_spec+1):
		data['RV_label_{}'.format(ii)] = df_spec['Instrument'][ii]
		data['Fit RV_{}'.format(ii)] = str2bool(df_spec['Fit RV'][ii])
		data['RM RV_{}'.format(ii)] = str2bool(df_spec['Fit RM'][ii])
		data['Chi2 RV_{}'.format(ii)] = float(df_spec['RV chi2 scaling'][ii])
		rfile = df_spec['RV Filename'][ii]
		if rfile.find('*') != -1: rfile = glob.glob(rfile)
		else: rfile = [rfile]
		if len(rfile): 
			rv_arr = np.loadtxt(rfile[0])
			data['RV_{}'.format(ii)] = rv_arr
		else:
			data['Fit RV_{}'.format(ii)] = False
			#data['RVs'] = n_spec - 1

		### To-do ###
		### FIX THE LOOPING PROBLEM ###
		### IF LS SCREWS UP THE ORDERING, e.g., Fit RV_4  ###

		ls = str2bool(df_spec['Fit LS'][ii])
		if ls: 
			data['Fit LS_{}'.format(ii)] = ls
			
			sfile = df_spec['LS Filename'][ii]
			if sfile.find('*.hdf5') != -1: sfile = glob.glob(sfile)
			else: sfile = [sfile]
			
			if len(sfile):
				with h5py.File(sfile[0],'r') as ff:
					shadow_data = {}
					times = []
					# for key in ff.keys():
					# 	try:
					# 		times.append(float(key))
					# 	except ValueError:
					# 		shadow_data['oot'] = ff[key][:]
					for key in ff.keys():
						times.append(float(key))
					times = np.asarray(times)
					ss = np.argsort(times)
					times = times[ss]					
					for time in times:
						arr = ff[str(time)][:]
						raw_vel = arr[:,0]
						shadow_data[time] = {
							'vel' : raw_vel,
							'ccf' : arr[:,1]
							}

				data['LS_{}'.format(ii)] = shadow_data

				data['Resolution_{}'.format(ii)] = int(df_spec['Disk Resolution'][ii])
				data['Thickness_{}'.format(ii)] = int(df_spec['Ring Thickness'][ii])
				data['Only_OOT_{}'.format(ii)] = str2bool(df_spec['Only fit OOT'][ii])
				data['OOT_{}'.format(ii)] = str2bool(df_spec['Fit OOT'][ii])
				idxs = df_spec['OOT indices'][ii]
				data['idxs_{}'.format(ii)] = [int(ii) for ii in idxs.split(',')]
				data['Chi2 LS_{}'.format(ii)] = float(df_spec['LS chi2 scaling'][ii])
				data['Chi2 OOT_{}'.format(ii)] = float(df_spec['OOT chi2 scaling'][ii])

			n_ls += 1	

		sl = str2bool(df_spec['Fit SL'][ii])
		if sl: 
			data['Fit SL_{}'.format(ii)] = sl
			sfile = df_spec['LS Filename'][ii]
			if sfile.find('*') != -1: sfile = glob.glob(sfile)
			else: sfile = [sfile]
			
			if len(sfile):
				with h5py.File(sfile[0],'r') as ff:
					shadow_data = {}
					times = []
					for key in ff.keys():
						try:
							times.append(float(key))
						except ValueError:
							shadow_data['oot'] = ff[key][:]
					times = np.asarray(times)
					ss = np.argsort(times)
					times = times[ss]					
					for time in times:
						arr = ff[str(time)][:]
						raw_vel = arr[:,0]
						shadow_data[time] = {
							'vel' : raw_vel,
							'ccf' : arr[:,1]
							}

				data['SL_{}'.format(ii)] = shadow_data
				idxs = df_spec['OOT indices'][ii]
				data['idxs_{}'.format(ii)] = [int(ii) for ii in idxs.split(',')]
				data['Chi2 SL_{}'.format(ii)] = float(df_spec['SL chi2 scaling'][ii])

			n_sl += 1	


	data['LSs'] = n_ls
	data['SLs'] = n_sl



def params_structure(filename):
	'''Structure the parameters for :strike:`umbra`.

	Function that reads in the parameter .csv file, and structures the content in a dictionary.

	:param filename: Path to the parameter file, e.g., `./par.csv`.
	:type filename: str

	.. note::
		Using global variable to prevent having to pickle and pass the data to the modules every time the code is called,
		see `emcee <https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments>`_'s documention.

	'''
	df = pd.read_csv(filename)
	def new_df(df,start,end=None):
		sub_df = df[start:end]
		header = sub_df.iloc[1]
		handles = [hh for hh in header if hh.isdigit() == False][1:]
		ndf = sub_df[2:end]
		ndf.columns = header
		ndf = ndf.reset_index(drop=True)
		return ndf, handles

	abc = list(string.ascii_lowercase)


	head = 'Planet parameters'
	pls = [pl for pl in df.keys()[1:] if pl in abc]

	splits = []
	for pl in pls:
		split = np.where(df[head].str.find('Planet {}'.format(pl)) == 0)[0][0]
		splits.append(split)

	split = np.where(df[head].str.find('System parameters') == 0)[0][0]
	splits.append(split)
	split = np.where(df[head].str.find('External constraints') == 0)[0][0]
	splits.append(split)

	global parameters
	parameters = {}
	fps = []
	cons = [] # constraints, but not fitting parameter
	for ii, pl in enumerate(pls): 
		ndf, handles = new_df(df,splits[ii],splits[ii+1])
		for handle in handles:
			fix = ndf[handle][9]
			if fix != 'none':
				fit = False
				if 'T0' in handle:
					#fit = True
					for transit in fix.split(','):
						if ('Phot' in transit ) or ('Spec' in transit):
							new_handle = transit + ':' + handle
							fps.append(new_handle)
							parameters[new_handle] = {
								'Unit'         : ndf[handle][0],
								'Label'        : ndf[handle][1][:-1] + ' ' + transit + '$',
								'Value'        : float(ndf[handle][2]),
								'Prior_vals'   : [float(ndf[handle][ii]) for ii in range(2,6)],
								'Prior'        : 'uni',#ndf[handle][6],
								'Distribution' : ndf[handle][7],
								'Comment' : ndf[handle][9]
							}						
						elif 'T0' in transit: fit = True
			else:
				fit = str2bool(ndf[handle][8])
				fix = False

			parameters[handle] = {
				'Unit'         : ndf[handle][0],
				'Label'        : ndf[handle][1],
				'Value'        : float(ndf[handle][2]),
				'Prior_vals'   : [float(ndf[handle][ii]) for ii in range(2,6)],
				'Prior'        : ndf[handle][6],
				'Distribution' : ndf[handle][7],
				'Comment' : ndf[handle][9]
			}
			if fit: fps.append(handle)
		if ('ecosw_{}'.format(pl) in fps) and ('esinw_{}'.format(pl) in fps):
			try:
				fps.remove('e_{}'.format(pl))
			except ValueError:
				pass
			try:
				fps.remove('w_{}'.format(pl))
			except ValueError:
				pass
		elif ('ecosw_{}'.format(pl) in fps):
			fps.remove('ecosw_{}'.format(pl))
		elif ('esinw_{}'.format(pl) in fps):
			fps.remove('esinw_{}'.format(pl))
		if ('cosi_{}'.format(pl) in fps):
			try:
				fps.remove('inc_{}'.format(pl))
			except ValueError:
				pass
		if ('T41_{}'.format(pl) in fps):
			fps.remove('T41_{}'.format(pl))
			cons.append('T41_{}'.format(pl))
		if ('T21_{}'.format(pl) in fps):
			fps.remove('T21_{}'.format(pl))
			cons.append('T21_{}'.format(pl))



	ndf, handles = new_df(df,splits[ii+1],splits[ii+2])
	LD_lincombs = []
	for handle in handles:
		fix = ndf[handle][9]
		#if (fix != 'none') & (fix != 'scale'):
		#if (fix != 'none'):
		if fix == 'lincom':
			fit = False
			if '_q' in handle:
				fix = False
				LDhandle = handle[:-1]
				if LDhandle not in LD_lincombs:
					LD_lincombs.append(LDhandle)
		elif fix != 'none':
			fit = False
		else:
			fit = str2bool(ndf[handle][8])
			fix = False
  
		parameters[handle] = {
			'Unit'         : ndf[handle][0],
			'Label'        : ndf[handle][1],
			'Value'        : float(ndf[handle][2]),
			'Prior_vals'   : [float(ndf[handle][ii]) for ii in range(2,6)],
			'Prior'        : ndf[handle][6],
			'Distribution' : ndf[handle][7],
			'Fix' : fix,
			'Comment' : ndf[handle][9]
		}
		if fit: fps.append(handle)

	for LDhandle in LD_lincombs:
		LDhandle1 = LDhandle + '1'
		LDhandle2 = LDhandle + '2'
		sumhandle = LDhandle + '_sum'
		fps.append(sumhandle)
		diffhandle = LDhandle + '_diff'
		parameters[sumhandle] = {
			'Unit'         : ndf[LDhandle1][0],
			'Label'        : r'$q_1 + q_2: \rm {}$'.format(LDhandle1.split('_')[0]),
			'Value'        : float(ndf[LDhandle1][2]) + float(ndf[LDhandle2][2]),
			'Prior_vals'   : [float(ndf[LDhandle1][2]) + float(ndf[LDhandle2][2]),0.1,0.0,2.0],
			'Prior'        : 'tgauss',
			'Distribution' : 'tgauss',
			'Fix' : False
		}			
		parameters[diffhandle] = {
			'Unit'         : ndf[LDhandle1][0],
			'Label'        : r'$q_1 - q_2: \rm {}$'.format(LDhandle1.split('_')[0]),
			'Value'        : float(ndf[LDhandle1][2]) - float(ndf[LDhandle2][2]),
			'Prior_vals'   : [float(ndf[LDhandle1][2]) - float(ndf[LDhandle2][2]),0.1,-1.0,1.0],
			'Prior'        : 'gauss',
			'Distribution' : ndf[handle][7],
			'Fix' : False
		}



	ndf, handles = new_df(df,splits[ii+2])
	for handle in handles:
		con = str2bool(ndf[handle][8])
		parameters[handle] = {
			'Unit'         : ndf[handle][0],
			'Label'        : ndf[handle][1],
			'Value'        : float(ndf[handle][2]),
			'Prior_vals'   : [float(ndf[handle][ii]) for ii in range(2,6)],
			'Prior'        : ndf[handle][6],
			'Distribution' : ndf[handle][7],
    	}
		if con: cons.append(handle)

	## Parameters to fit
	parameters['FPs'] = fps
	## External constraints
	parameters['ECs'] = cons
	## Linear combinations for stepping in LD coeffs.
	parameters['LinCombs'] = LD_lincombs
	## Planets
	parameters['Planets'] = pls

	#return parameters
	#global parameters

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
	supersample_factor=30,exp_time=0.0208,
	t0_off=0.0):
	lclabel = 'LC{}'.format(n_phot)
	pllabel = '_{}'.format(n_planet)

	batpars = batman.TransitParams()
	per = parameters['P'+pllabel]['Value']
	batpars.per = per
	t0 = parameters['T0'+pllabel]['Value']# + t0_off
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

def rv_model(time,n_planet='b',n_rv=1,RM=False,t0_off=0.0):
	pllabel = '_{}'.format(n_planet)
	
	orbpars = dynamics.OrbitalParams()
	per = parameters['P'+pllabel]['Value']
	orbpars.per = per
	orbpars.K = parameters['K'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']
	omega = omega%360
	orbpars.w = omega
	ecc = parameters['e'+pllabel]['Value']
	orbpars.ecc = ecc
	orbpars.RVsys = 0.0

	T0 = parameters['T0'+pllabel]['Value'] + t0_off

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
		stelpars = dynamics.StellarParams()
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

		calcRV = dynamics.get_RV(time,orbpars,RM=RM,stelpars=stelpars,mpath=mpath)
	else:
		calcRV = dynamics.get_RV(time,orbpars)
	return calcRV

def ini_grid(rad_disk=100,thickness=20):
	## Make initial grid
	start_grid, vel, mu = shady.grid(rad_disk) #make grid of stellar disk
	## The grid is made into rings for faster calculation of the macroturbulence (approx. constant in each ring)
	ring_grid, vel, mu_grid, mu_mean = shady.grid_ring(rad_disk,thickness) 
	
	return start_grid, ring_grid, vel, mu, mu_grid, mu_mean


def ls_model(time,
	start_grid,ring_grid,vel_grid,mu,mu_grid,mu_mean,rad_disk,
	n_planet='b',n_rv=1,oot=False,t0_off=0.0):
	pllabel = '_{}'.format(n_planet)
	
	## Planet parameters
	per = parameters['P'+pllabel]['Value']
	T0 = parameters['T0'+pllabel]['Value'] + t0_off
	
	inc = parameters['inc'+pllabel]['Value']*np.pi/180.
	a_Rs = parameters['a_Rs'+pllabel]['Value']
	b = a_Rs*np.cos(inc)
	rp = parameters['Rp_Rs'+pllabel]['Value']
	ecc = parameters['e'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']
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


	vsini = parameters['vsini']['Value']
	zeta = parameters['zeta']['Value']
	xi = parameters['xi']['Value']
	### HARD-CODED
	conv_par = np.sqrt(xi**2 + 1.1**2)#parameters['xi']['Value']


	if oot:
		vel_1d, line_conv, lum = shady.absline_star(start_grid,vel_grid,ring_grid,
			mu,mu_mean,
			vsini,conv_par,zeta,cs=qs
			)
		line_oot = np.sum(line_conv,axis=0)
		area = np.trapz(line_oot,vel_1d)
		line_oot_norm = line_oot/area
		return vel_1d, line_oot_norm, lum

	## Make shadow                   
	vel_1d, line_conv, line_transit, planet_rings, lum, index_error = shady.absline(
																	start_grid,vel_grid,ring_grid,
																	mu,mu_mean,mu_grid,
																	vsini,conv_par,zeta,
																	cs=qs,radius=rad_disk,
																	times=time,
																	Tw=T0,per=per,
																	Rp_Rs=rp,a_Rs=a_Rs,inc=inc,
																	ecc=ecc,w=omega,lam=lam)
	
	line_oot = np.sum(line_conv,axis=0)
	area = np.trapz(line_oot,vel_1d)
	line_oot_norm = line_oot/area
	## Normalize such that out-of-transit lines have heights of unity
	line_transit = line_transit/area#np.max(line_oot)
	## Shadow
	shadow = line_oot_norm - line_transit
	
	return vel_1d, shadow, line_oot_norm, planet_rings, lum, index_error

def localRV_model(time,n_planet='b',t0_off=0.0):
	pllabel = '_{}'.format(n_planet)
	 
	## Planet parameters
	per = parameters['P'+pllabel]['Value']
	T0 = parameters['T0'+pllabel]['Value'] + t0_off
	
	inc = parameters['inc'+pllabel]['Value']*np.pi/180.
	a_Rs = parameters['a_Rs'+pllabel]['Value']
	#b = a_Rs*np.cos(inc)
	rp = parameters['Rp_Rs'+pllabel]['Value']
	ecc = parameters['e'+pllabel]['Value']
	omega = parameters['w'+pllabel]['Value']
	lam = parameters['lam'+pllabel]['Value']

	omega = omega%360
	omega *= np.pi/180.

	lam *= np.pi/180.
	pp = dynamics.time2phase(time,per,T0)*per*24

	tau = dynamics.total_duration(per,rp,a_Rs,inc,ecc,omega)
	x1 = -24*tau/2
	x2 = 24*tau/2
	
	y1, y2 = dynamics.get_rel_vsini(a_Rs*np.cos(inc),lam)
	y1 *= -1
	a = (y2 - y1)/(x2 - x1)
	b = y1 - x1*a


	return pp*a + b


def chi2(ycal,yobs,sigy):
	return np.sum((ycal-yobs)**2/sigy**2)

def lnlike(ycal,yobs,sigy):
	nom = chi2(ycal,yobs,sigy)
	den = np.sum(np.log(2*np.pi*sigy**2))
	return -0.5*(den + nom)

def lnprob(positions):
	
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




				aa_pl = [parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				try:
					t0n = parameters['Phot_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass

				per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']
				off_arr = np.round((time-t0)/per)
				n_pers = np.unique(off_arr)
				for n_per in n_pers:
					t_idxs = n_per == off_arr
					t0_off = 0.0#(n_per*per)**2*aa_pl[1]#0.0

					flux_pl = lc_model(time[t_idxs],n_planet=pl,n_phot=nn,
								supersample_factor=ofactor,exp_time=exp,t0_off=t0_off)
					if deltamag > 0.0:
						dilution = 10**(-deltamag/2.5)
						flux_pl = flux_pl/(1 + dilution) + dilution/(1+dilution)

					flux_m[t_idxs] -= (1 - flux_pl)

				if (trend == 'poly') or (trend == True):
					per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']
					ph = dynamics.time2phase(time,per,t0)*per*24
					aR = parameters['a_Rs_{}'.format(pl)]['Value']
					rp = parameters['Rp_Rs_{}'.format(pl)]['Value']
					inc = parameters['inc_{}'.format(pl)]['Value']
					ecc = parameters['e_{}'.format(pl)]['Value']
					ww = parameters['w_{}'.format(pl)]['Value']
					dur = dynamics.total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
					
					indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
					in_transit = np.append(in_transit,indxs)				


			
			chi2scale = data['Chi2 LC_{}'.format(nn)]
			
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
				loga = parameters['LC_{}_GP_log_a'.format(nn)]['Value']
				logc = parameters['LC_{}_GP_log_c'.format(nn)]['Value']
				gp = data['LC_{} GP'.format(nn)]
				res_flux = flux - flux_m


				gp.set_parameter_vector(np.array([loga,logc]))
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
			chi2scale = data['Chi2 RV_{}'.format(nn)]

			if add_drift: 
				rv_times, rvs, ervs = np.append(rv_times,time), np.append(rvs,rv), np.append(ervs,sigma)  
			
			calc_RM = data['RM RV_{}'.format(nn)]
			
			rv_m = np.zeros(len(rv))
			for pl in pls:
				try:
					t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass
				aa_pl = [parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				per, t0 = parameters['P_{}'.format(pl)]['Value'],parameters['T0_{}'.format(pl)]['Value']
				off_arr = np.round((time-t0)/per)
				n_pers = np.unique(off_arr)
				for n_per in n_pers:
					t_idxs = n_per == off_arr
					t0_off = n_per*per*aa_pl[0]#0.0
					t0_off += (n_per*per)**2*aa_pl[1]#0.0
					t0_off = 0.0#(n_per*per)**2*aa_pl[1]#0.0
					rv_pl = rv_model(time[t_idxs],n_planet=pl,n_rv=nn,RM=calc_RM,t0_off=t0_off)
					rv_m[t_idxs] += rv_pl
			model_rvs = np.append(model_rvs,rv_m)
			n_dps += len(rvs)

	if add_drift:# n_rv > 0:
		aa = np.array([parameters['a{}'.format(ii)]['Value'] for ii in range(1,3)])
		idx = np.argmin(rv_times)
		zp, off = rv_times[idx], rvs[idx]

		drift = aa[1]*(rv_times-zp)**2 + aa[0]*(rv_times-zp)# + off
		
		model_rvs += drift
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
				aa2_pl = [parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]
				off_arr2 = np.round((times-t02)/p2)
				n_pers = np.unique(off_arr2)
				for n_per in n_pers:
					t_idxs = n_per == off_arr2
					t0_off = n_per*p2*aa2_pl[0]#0.0
					t0_off += (n_per*p2)*aa2_pl[1]#0.0
					t0_off = 0.#(n_per*p2)*aa2_pl[1]#0.0
					rv_pl = rv_model(times[t_idxs],n_planet=pl,n_rv=nn,RM=False,t0_off=t0_off)
				rv_m[t_idxs] += rv_pl
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

			P, T0 = parameters['P_{}'.format(pl)]['Value'], parameters['T0_{}'.format(pl)]['Value'] 
			aa_pl = [parameters['a{}_{}'.format(ii,pl)]['Value'] for ii in range(1,3)]

			off_arr = np.round((times-T0)/P)
			n_per = np.unique(off_arr)
			t0_off = n_per*P*aa_pl[0]
			t0_off += (n_per*P)**2*aa_pl[1]
			t0_off = 0.

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
				vel_model, model_ccf, _ = ls_model(
					#parameters,time,start_grid,ring_grid,
					times[oots],start_grid,ring_grid,
					vel_grid,mu,mu_grid,mu_mean,resol,
					oot=only_oot,n_rv=nn
					)
			else:
				vel_model, shadow_model, model_ccf, darks, oot_lum, index_error = ls_model(
					#parameters,time,start_grid,ring_grid,
					times,start_grid,ring_grid,
					vel_grid,mu,mu_grid,mu_mean,resol,
					n_rv=nn,t0_off=t0_off
					)
				if index_error:
					return -np.inf, -np.inf
				bright = np.sum(oot_lum)

			
			nvel = len(shadow_data[times[0]]['vel'])
			vels = np.zeros(shape=(nvel,len(times)))
			oot_ccfs = np.zeros(shape=(nvel,len(oots)))
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
				no_peak = (vel > 15) | (vel < -15)
				

				ccf = shadow_data[time]['ccf']
				
				area = np.trapz(ccf,vel)
				ccf /= area	

				vv,cc = get_binned(vels[:,idx],ccf)
				no_peak_b = (vv > 15) | (vv < -15)
				oot_sd_b.append(np.std(cc[no_peak_b]))

					
				poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
	
				ccf -= vel*poly_pars[0] + poly_pars[1]
	
	
				oot_ccfs[:,ii] = ccf
				avg_ccf += ccf
				avg_vel += vel
			avg_ccf /= len(oots)
			avg_vel /= len(oots)

			## Here we simply fit our average out-of-transit CCF
			## to an out-of-transit model CCF
			## Hard-coded
			log_jitter = parameters['RVsigma_{}'.format(nn)]['Value']
			jitter = log_jitter
			jitter = 0.0


			
			model_int = interpolate.interp1d(vel_model,model_ccf,kind='cubic',fill_value='extrapolate')
			newline = model_int(avg_vel)
			
			vv,cc = get_binned(avg_vel,avg_ccf)
			vv,ncc = get_binned(avg_vel,newline)
			
			sd = np.mean(oot_sd_b)
			unc = np.ones(len(vv))*sd

			chi2scale_oot = data['Chi2 OOT_{}'.format(nn)]
			#chi2scale_oot = 5.245
			unc *= chi2scale_oot

			if fit_oot:
				chisq += chi2(ncc,cc,unc)
				log_prob += lnlike(ncc,cc,unc)

				n_dps += len(cc)
		
			if not only_oot:
				chi2scale_shadow = data['Chi2 LS_{}'.format(nn)]

				## Again shift CCFs to star rest frame
				## and detrend CCFs
				## Compare to shadow model
				for ii, idx in enumerate(its):
					#arr = data[time]
					time = times[idx]
					vel = shadow_data[time]['vel'] - rv_m[idx]*1e-3
					vels[:,idx] = vel
					no_peak = (vel > 15) | (vel < -15)
						
					ccf = shadow_data[time]['ccf']
					area = np.trapz(ccf,vel)
					ccf /= area
					
					

					ccf *= darks[idx]/bright#blc[ii]		
					shadow = avg_ccf - ccf
					

					ff = interpolate.interp1d(vel_model,shadow_model[idx],kind='cubic',fill_value='extrapolate')
					ishadow = ff(vel)


					vv,ss = get_binned(vel,shadow)

					no_peak = (vv > 15) | (vv < -15)
					sd = np.std(ss[no_peak])
					poly_pars = np.polyfit(vv[no_peak],ss[no_peak],1)
					nvv,nss = get_binned(vel,ishadow)

					unc = np.ones(len(vv))*np.sqrt(sd**2 + jitter**2)
					unc *= chi2scale_shadow

					chisq += chi2(ss,nss + vv*poly_pars[0] + poly_pars[1],unc)
					log_prob += lnlike(ss,nss + vv*poly_pars[0] + poly_pars[1],unc)

					n_dps += len(ss)

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

				### HARD-CODED
				darks = lc_model(times,n_planet=pl,n_phot=1)


				idxs = [ii for ii in range(len(times))]
				oots = data['idxs_{}'.format(nn)]

				its = [ii for ii in idxs if ii not in oots]	


				nvel = len(slope_data[times[0]]['vel'])
				vels = np.zeros(shape=(nvel,len(times)))
				oot_ccfs = np.zeros(shape=(nvel,len(oots)))
				if not len(avg_ccf):
					avg_ccf = np.zeros(nvel)

					for ii, idx in enumerate(oots):
						time = times[idx]
						vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
						vels[:,idx] = vel
						no_peak = (vel > 15) | (vel < -15)
						

						ccf = slope_data[time]['ccf']
						area = np.trapz(ccf,vel)
						ccf /= area	

						poly_pars = np.polyfit(vel[no_peak],ccf[no_peak],1)
						ccf -= vel*poly_pars[0] + poly_pars[1]


						oot_ccfs[:,ii] = ccf
						avg_ccf += ccf
					avg_ccf /= len(oots)

				try:
					t0n = parameters['Spec_{}:T0_{}'.format(nn,pl)]['Value']
					parameters['T0_{}'.format(pl)]['Value'] = t0n				
				except KeyError:
					pass

				T0 = parameters['T0_{}'.format(pl)]['Value']
				P = parameters['P_{}'.format(pl)]['Value']
				ecc = parameters['e_{}'.format(pl)]['Value']
				ww = parameters['w_{}'.format(pl)]['Value']*np.pi/180
				ar = parameters['a_Rs_{}'.format(pl)]['Value']
				inc = parameters['inc_{}'.format(pl)]['Value']*np.pi/180
				lam = parameters['lam_{}'.format(pl)]['Value']*np.pi/180
				vsini = parameters['vsini']['Value'] 
				#P, T0 = parameters['P_{}'.format(pl)]['Value'], parameters['T0_{}'.format(pl)]['Value'] 
				#ar, inc = parameters['a_Rs_{}'.format(pl)]['Value'], parameters['inc_{}'.format(pl)]['Value']*np.pi/180.
				rp = parameters['Rp_Rs_{}'.format(pl)]['Value']


				#fit_params = lmfit.Parameters()
				cos_f, sin_f = dynamics.true_anomaly(times[its], T0, ecc, P, ww)
				#print(cos_f)
				xxs, yys = dynamics.xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
				rvs = np.array([])
				errs = np.array([])
				for ii, idx in enumerate(its):
					time = times[idx]
					vel = slope_data[time]['vel'] - rv_m[idx]*1e-3
					vels[:,idx] = vel
					no_peak = (vel > 15) | (vel < -15)

					xx = xxs[idx]
					#cos_f, sin_f = dynamics.true_anomaly(time, T0, ecc, P, ww)
					#xx, yy = dynamics.xy_pos(cos_f,sin_f,ecc,ww,ar,inc,lam)
					
					#ccf = np.zeros(len(vel))
					#shadow_arr = shadow_data[time]['ccf']
					#ccf = 1 - shadow_arr/np.median(shadow_arr[no_peak])

					ccf = slope_data[time]['ccf']
					area = np.trapz(ccf,vel)
					ccf /= area
					
					sd = np.std(ccf[no_peak])

					ccf *= darks[idx]#/bright#blc[ii]		
					shadow = avg_ccf - ccf
					poly_pars = np.polyfit(vel[no_peak],shadow[no_peak],1)
					
					shadow -= vel*poly_pars[0] + poly_pars[1]

					#peak = np.where((vel > -15) & (vel < 15))
					peak = np.where((vel > (xx*vsini - vsini/2)) & (vel < (xx*vsini + vsini/2)))
					try:
						midx = np.argmax(shadow[peak])
					except ValueError:
						return -np.inf, -np.inf
					amp, mu1 = shadow[peak][midx], vel[peak][midx]# get max value of CCF and location
					shadow = shadow/amp

					try:
						gau_par, pcov = curve_fit(Gauss,vel,shadow,p0=[1.0,xx*vsini,1.0])
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


				pp = dynamics.time2phase(times[its],per,T0)*24*per

				chi2scale = data['Chi2 SL_{}'.format(nn)]
				erv_scale *= chi2scale


				chisq += chi2(rv_scale,slope,erv_scale)
				log_prob += lnlike(rv_scale,slope,erv_scale)
				n_dps += len(rv_scale)


	if np.isnan(log_prob) or np.isnan(chisq):
		return -np.inf, -np.inf
	else:
		return log_prob, chisq/(n_dps - m_fps)


def mcmc(param_fname,data_fname,maxdraws,nwalkers,
		save_results = True, results_filename='results.csv',
		sample_filename='samples.h5',reset=True,burn=0.5,
		plot_convergence=True,save_samples=True,
		corner=True,chains=False,nproc=1,
		stop_converged=True):


	run_sys(nproc)
	#data = data_structure(data_fname)
	#parameters = params_structure(param_fname)
	params_structure(param_fname)
	data_structure(data_fname)

	n_phot = data['LCs']
	n_ls = data['LSs']
	n_sl = data['SLs']
	fit_line = False
	for nn in range(1,n_ls+1):
		if data['Fit LS_{}'.format(nn)]:
			resol = data['Resolution_{}'.format(nn)]
			thick = data['Thickness_{}'.format(nn)]
			start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = ini_grid(resol,thick)

			data['Start_grid_{}'.format(nn)] = start_grid
			data['Ring_grid_{}'.format(nn)] = ring_grid
			data['Velocity_{}'.format(nn)] = vel_grid
			data['mu_{}'.format(nn)] = mu
			data['mu_grid_{}'.format(nn)] = mu_grid
			data['mu_mean_{}'.format(nn)] = mu_mean

	fit_params = parameters['FPs']
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
				#kwargs=master_dict,pool=pool,backend=backend)#,
				moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])  

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
	
	res_df = check_convergence(sampler=sampler,pars=parameters,
		plot_corner=corner,plot_chains=chains,dat=data,per_burn=burn,
		save_df=save_results,results_filename=results_filename)

	## Plot autocorrelation time
	## Converged if: 
	## - chain is longer than 100 times the estimated autocorrelation time 
	## - & this estimate changed by less than 1%.
	if plot_convergence:
		expose.plot_autocorr(autocorr,index,kk)
		# figc = plt.figure()
		# axc = figc.add_subplot(111)
		# nn, yy = 1000*np.arange(1,index+1), autocorr[:index]
		# axc.plot(nn,nn/100.,'k--')#,color='C7')
		# axc.plot(nn,yy,'k-',lw=3.0)
		# axc.plot(nn,yy,'-',color=colors[0],lw=2.0)
		# axc.set_xlabel(r'$\rm Step \ number$')
		# axc.set_ylabel(r'$\rm \mu(\hat{\tau})$')
		# plt.savefig('autocorr.pdf')
	
	return res_df


def check_convergence(filename='samples.h5', sampler=None, 
					pars=None,param_fname='parameters.csv',
					dat=None,data_fname='data.csv',
					plot_chains=True, plot_corner=True,chain_ival=5,
					save_df=True,results_filename='results.csv',
					n_auto=4,per_burn=None,plot_priors=True):
	if filename:
		reader = emcee.backends.HDFBackend(filename)	
	elif sampler:
		reader = sampler
		del sampler
	elif filename == None and sampler == None:
		raise FileNotFoundError('You need to either provide a filename for the stored sampler \
			or the sampler directly.')

	if pars == None:
		try:
			params_structure(param_fname)
		except FileNotFoundError as err:
			print(err.args)
			print('Either provide the name of the .csv-file with the fitting parameters \
				or give the parameter structure directly.')

	if dat == None:
		try:
			data_structure(data_fname)
		except FileNotFoundError as err:
			print(err.args)
			print('Either provide the name of the .csv-file for the data \
				or give the data structure directly.')

	fit_params = parameters['FPs']


	steps, nwalkers, ndim = reader.get_chain().shape
	tau = reader.get_autocorr_time(tol=0)

	if not per_burn:
		burnin = int(n_auto * np.max(tau))
		
		last_taun_samples = int(steps-burnin)
		last_flat = reader.get_chain(discard=burnin,flat=True)[last_taun_samples:,:]
		
		comp = {}
		for ii, fp in enumerate(fit_params):
			val = np.median(last_flat[:,ii])
			hpds = stat_tools.hpd(last_flat[:,ii],0.68)
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
				lab = parameters[fp]['Label']
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
		lab = parameters[fp]['Label']
		labels.append(lab)	
		hands.append(fp)	

	raw_samples = reader.get_chain()
	## plot the trace/chains
	if plot_chains:
		expose.create_chains(samples,labels=labels,savefig=True,ival=chain_ival)
	del raw_samples


	thin = int(0.5 * np.min(tau))
	samples = reader.get_chain(discard=burnin)
	reshaped_samples = samples.reshape(samples.shape[1],samples.shape[0],samples.shape[2])
	conv_samples = az.convert_to_dataset(reshaped_samples)
	del reshaped_samples
	rhats = az.rhat(conv_samples).x.values
	del conv_samples
	#print(rhats)
	flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)	
	log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
	chi2_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
	del reader

	pls = parameters['Planets']
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
			# hpds_ecc = stat_tools.hpd(ecc,0.68)
			# lower_ecc = val_ecc - hpds_ecc[0]
			# upper_ecc = hpds_ecc[1] - val_ecc
			# results['e_{}'.format(pl)] = [val_ecc,lower_ecc,upper_ecc]

			# omega = omega%360
			# val_omega = np.median(omega)
			# hpds_omega = stat_tools.hpd(omega,0.68)
			# lower_omega = val_omega - hpds_omega[0]
			# upper_omega = hpds_omega[1] - val_omega
			# results['w_{}'.format(pl)] = [val_omega,lower_omega,upper_omega]
		elif ('e_{}'.format(pl) in fit_params):
			idx_ecc = fit_params.index('e_{}'.format(pl))
			ecc = flat_samples[:,idx_ecc]
			idx_ww = fit_params.index('w_{}'.format(pl))
			omega = flat_samples[:,idx_ww]
		else:
			ecc = parameters['e_'+pl]['Value']
			omega = parameters['w_'+pl]['Value']

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
				aR = parameters['a_Rs_'+pl]['Value']
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
				'P_{}'.format(pl) : parameters['P_{}'.format(pl)]['Value'],
				'Rp_Rs_{}'.format(pl) : parameters['Rp_Rs_{}'.format(pl)]['Value'],
				'a_Rs_{}'.format(pl) : parameters['a_Rs_{}'.format(pl)]['Value'],
				'inc_{}'.format(pl) : parameters['inc_{}'.format(pl)]['Value'],
				'e_{}'.format(pl) : parameters['e_{}'.format(pl)]['Value'],
				'w_{}'.format(pl) : parameters['w_{}'.format(pl)]['Value']
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
	n_phot, n_rv = data['LCs'], data['RVs']
	
	for nn in range(1,n_rv+1):
		if ('RV{}_q_sum'.format(nn) in fit_params):
			idx_qs = fit_params.index('RV{}_q_sum'.format(nn))
			qs = flat_samples[:,idx_qs]
			qd = parameters['RV{}_q1'.format(nn)]['Prior_vals'][0] - parameters['RV{}_q2'.format(nn)]['Prior_vals'][0]
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
			#qd = parameters['LC{}_q1'.format(nn)]['Value'] - parameters['LC{}_q2'.format(nn)]['Value']
			qd = parameters['LC{}_q1'.format(nn)]['Prior_vals'][0] - parameters['LC{}_q2'.format(nn)]['Prior_vals'][0]
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


	priors = {}
	fps = parameters['FPs']
	for i in range(ndim): priors[i] = ['none',0.0,0.0,0.0,0.0]
	for i, fp in enumerate(fps): priors[i] = [parameters[fp]['Prior'],parameters[fp]['Prior_vals'][0],parameters[fp]['Prior_vals'][1],parameters[fp]['Prior_vals'][2],parameters[fp]['Prior_vals'][3]]
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
		bounds = stat_tools.hpd(all_samples[:,i], 0.68)

		qs.append([bounds[0],val,bounds[1]])
		#print(qs[i][1],qs[i][1]-qs[i][0],qs[i][2]-qs[i][1])
		val, low, up = stat_tools.significantFormat(qs[i][1],qs[i][1]-qs[i][0],qs[i][2]-qs[i][1])
		#print(mode,mode-qs[i][0],qs[i][2]-mode)
		mode, _, _ = stat_tools.significantFormat(mode,mode-qs[i][0],qs[i][2]-mode)

		label = labels[i][:-1] + '=' + str(val) + '_{-' + str(low) + '}^{+' + str(up) + '}'
		value_formats.append(label)

		#best_idx = np.argmax(log_prob_samples)
		best_idx = np.argmin(chi2_samples)
		#print(val,mode)
		best_fit = all_samples[best_idx,i]
		#print(labels[i][:-1],best_fit,qs[i])
		#print(best_fit,best_fit-qs[i][0],qs[i][2]-best_fit)
		#print(low,up)
		best_val, _, _ = stat_tools.significantFormat(best_fit,float(low),float(up))
		#print(best_val)
		#best_val, _, _ = stat_tools.significantFormat(best_fit,best_fit-qs[i][0],qs[i][2]-best_fit)

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
			fps = parameters['FPs']
			for i in range(ndim): priors[i] = ['none',0.0,0.0,0.0,0.0]
			for i, fp in enumerate(fps): priors[i] = [parameters[fp]['Prior'],parameters[fp]['Prior_vals'][0],parameters[fp]['Prior_vals'][1],parameters[fp]['Prior_vals'][2],parameters[fp]['Prior_vals'][3]]
		else:
			priors = None
		expose.create_corner(all_samples,labels=labels,truths=None,savefig=True,priors=priors,quantiles=qs,diag_titles=value_formats)

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


def lmfitter(param_fname,data_fname,method='leastsq',eps=0.01,
		print_fit=True):
	'''Fit data using lmfit

	'''
	#data = data_structure(data_fname)
	#parameters = params_structure(param_fname)

	data_structure(data_fname)
	params_structure(param_fname)

	fit_pars = parameters['FPs']
	pars = list(parameters.keys())
	pls = parameters['Planets']
	pars.remove('Planets')
	pars.remove('FPs')
	pars.remove('ECs')
	pars.remove('LinCombs')
	params = lmfit.Parameters()
	for par in pars:
		pri = parameters[par]['Prior_vals']
		if par in fit_pars:
			lower, upper = pri[2],pri[3]
			if np.isnan(lower): lower = -np.inf
			if np.isnan(upper): upper = np.inf
			params.add(par,value=pri[0],vary=True,min=lower,max=upper)
		else:
			params.add(par,value=pri[0],vary=False)

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
