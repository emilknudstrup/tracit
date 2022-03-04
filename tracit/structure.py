#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:30:39 2022

@author: emil

.. todo::
	* parameters, data only need to be made global within `business.mcmc/lmfitter` and `expose`, maybe set that in run_sys/run_bus.
	* move updated_pars out of par_struct

"""
import string
import numpy as np

def par_struct(n_phot=1,n_spec=1,n_planets=1,LD_law='quad',
	fps=[],cons=[],LD_lincombs=[],updated_pars=None):
	'''Structure for the parameters.

	Dictionary to hold the values, priors, etc., for the parameters.

	:param n_phot: Number of photometric systems. Default 1.
	:type n_phot: int
	
	:param n_spec: Number of spectroscopic systems. Default 1.
	:type n_spec: int

	:param n_phot: Number of planets in system. Default 1.
	:type n_spec: int

	:param LD_law: limb darkening law used. Default `'quad'`. See :py:class:`dynamics.StellarParams`.
	:type LD_law: str    

	.. note::
		It is possible to step in :math:`\cos i` and :math:`\sqrt{e}\cos \omega` & :math:`\sqrt{e}\sin \omega`:
			* Add 'cosi_b' to `parameters['FPs']`, walkers will step in cos(i) for planet 'b'.
			* Add 'ecosw_b', 'esinw_b', walkers will step in sqrt(e)*cos(w) & sqrt(e)*sin(w) for planet 'b'.
			* Remember to remove 'i_b', e_b', 'w_b', from `parameters['FPs']` accordingly.


	'''


	LD_laws = ['uni','quad','nonlinear']
	assert LD_law in LD_laws, print('Limb darkening law must be in {}'.format(LD_laws))

	
	abc = list(string.ascii_lowercase)[1:]
	ABC = list(string.ascii_uppercase)

	# orbpars = ['P','T0','e','w','Rp_Rs','a_Rs','inc','K','Tw','lam','b','ecosw','esinw','T14','T23','T0_a2','T0_a1']
	# stelpars = ['vsini','zeta','xi']
	# rows = {'Handle' : ['Parameter','Unit','Label',
	# 					'Value','Uncertainty','Lower','Upper',
	# 					'Prior','Start','Fit parameter','Fix to']}	
	
	#global parameters
	parameters = {}


	abc = list(string.ascii_lowercase)[1:]

	pls = [abc[ii] for ii in range(n_planets)]


	default = ['uni','tgauss','false','none']
	for ii, pl in enumerate(pls):
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
			['T_{21}','hours',r'$T \rm _{21,'+pl+'} (hours)$',3.0,0.1,0.,10.]
			}

		for par in pars.keys():
			parameters[par] = {
				'Name'         : pars[par][0],
				'Unit'         : pars[par][1],
				'Label'        : pars[par][2],#ndf[handle][1][:-1] + ' ' + transit + '$',
				'Value'        : pars[par][3],
				'Prior_vals'   : [pars[par][ii] for ii in range(3,7)],
				'Prior'        : 'uni',
				'Distribution' : 'tgauss',
				'Comment' : 'none',
			}


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
			LD_pars[label+'_q_sum'] = ['Sum of LD coeffs for {}'.format(label),'quadratic',r'$q_1 + q_2: \rm {}$'.format(label),0.7,0.1,0.0,1.0]
			LD_pars[label+'_q_diff'] = ['Difference between LD coeffs for {}'.format(label),'quadratic',r'$q_1 - q_2: \rm {}$'.format(label),0.1,0.1,0.0,1.0]
		elif LD_law == 'uni':
			LD_pars[label+'_q1'] = ['No coefficients','uniform']
		elif LD_law == 'nonlinear':
			for ii in range(1,5):
				LD_pars[label+'_q{}'.format(ii)] = ['LD coeff {}'.format(ii),'nonlinear',r'$q_2 \ \rm {}$'.format(label,ii)]

	add_lc_pars = {}
	for ii in range(1,n_phot+1):
		star['LCblend_{}'.format(ii)] = ['Dilution (deltaMag=-2.5log(F2/F1)) photometer {}'.format(ii),' ',r'$\rm \delta M{}$'.format('_{LC,'+str(ii)+'}'),0.0,0.5,0.0,7.0]
		star['LCsigma_{}'.format(ii)] = ['Log jitter photometer {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LC,'+str(ii)+'}'),-30,0.05,-50,1.0]
		star['LC_{}_GP_log_a'.format(ii)] = ['GP log amplitude, photometer {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{LC,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		star['LC_{}_GP_log_c'.format(ii)] = ['GP log time scale, photometer {}'.format(ii),' ',r'$\rm \log \tau{} \ (days)$'.format('_{LC,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		label = 'LC'+str(ii)
		set_LD(star,LD_law,label)

	add_rv_pars = {}
	for ii in range(1,n_spec+1):
		star['RVsys_{}'.format(ii)] = ['Systemic velocity instrument {}'.format(ii),'m/s',r'$\gamma_{} \ \rm (m/s)$'.format(ii),0.0,1.,-20.,20.]
		star['RVsigma_{}'.format(ii)] = ['Jitter RV instrument {}'.format(ii),'m/s',r'$\rm \sigma{} \ (m/s)$'.format('_{RV,'+str(ii)+'}'),0.0,1.0,0.0,100.]
		star['RV_{}_GP_log_a'.format(ii)] = ['GP log amplitude, CCF {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{RV,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		star['RV_{}_GP_log_c'.format(ii)] = ['GP log exponent, CCF {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{RV,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		label = 'RV'+str(ii)
		set_LD(star,LD_law,label)

	star['a2'] = ['RV curve','m/(s*d^2)',r'$\ddot{\gamma} \ \rm (m \ s^{-1} \ d^{-2})$',0.0,1.0,-10.,10.]
	star['a1'] = ['RV slope','m/(s*d)',r'$\dot{\gamma} \ \rm (m \ s^{-1} \ d^{-1})$',0.0,1.0,-10.,10.]


	for par in star.keys(): 
		#for de in default:
		#	star[par].append(de)

		for par in star.keys():
			parameters[par] = {
				'Name'         : star[par][0],
				'Unit'         : star[par][1],
				'Label'        : star[par][2],#ndf[handle][1][:-1] + ' ' + transit + '$',
				'Value'        : star[par][3],
				'Prior_vals'   : [star[par][ii] for ii in range(3,7)],
				'Prior'        : 'uni',
				'Distribution' : 'tgauss',
				'Comment' : 'none',
			}

	ext = {'rho_s' : 
			['Stellar density','g/cm^3',r'$\rho_\star \ \rm (g/cm^{3})$',0.5,0.1,0.0,2.]}
	for par in ext.keys(): 
		parameters[par] = {
			'Name'         : ext[par][0],
			'Unit'         : ext[par][1],
			'Label'        : ext[par][2],#ndf[handle][1][:-1] + ' ' + transit + '$',
			'Value'        : ext[par][3],
			'Prior_vals'   : [ext[par][ii] for ii in range(3,7)],
			'Prior'        : 'uni',
			'Distribution' : 'tgauss',
			'Comment' : 'none',
		}


	## Parameters to fit
	parameters['FPs'] = fps
	## External constraints
	parameters['ECs'] = cons
	## Linear combinations for stepping in LD coeffs.
	parameters['LinCombs'] = LD_lincombs
	## Planets
	parameters['Planets'] = pls

	return parameters

def update_pars(rdf,parameters,best_fit=True):

	pars = rdf.keys()[1:-2]
	idx = 1
	if (rdf.shape[0] > 3) & best_fit: idx = 4
	for par in pars:
		try:
			parameters[par]['Value'] = float(rdf[par][idx])	
		except KeyError:
			pass

def dat_struct(n_phot=1,n_rvs=1,n_ls=0,n_sl=0):
	'''Structure for the data.
	
	Dictionary to hold the data, specifications, etc. 

	:param n_phot: Number of light curves from given photometric systems. Default 1.
	:type n_phot: int
	
	:param n_rvs: Number of radial velocities from given spectrograph. Default 1.
	:type n_rvs: int

	:param n_ls: Number of CCFs from given spectrograph for the shadow. Default 0.
	:type n_ls: int

	:param n_sl: Number of CCFs from given spectrograph for the slope. Default 0.
	:type n_sl: int
	'''

	data = {}


	data['LCs'] = n_phot
	for ii in range(1,n_phot+1):
		data['LC_label_{}'.format(ii)] = 'Photometer {}'.format(ii)
		data['Fit LC_{}'.format(ii)] = False
		#lfile = df_phot['LC Filename'][ii]
		#if lfile.find('*') != -1: lfile = glob.glob(lfile)[0] 
		#phot_arr = np.loadtxt(lfile)
		#data['LC_{}'.format(ii)] = np.zeros(shape=(10,3))#phot_arr
		data['LC filename_{}'.format(ii)] = 'lc.txt'#np.zeros(shape=(10,3))#phot_arr
		data['OF LC_{}'.format(ii)] = 1 #Oversample factor
		data['Exp. LC_{}'.format(ii)] = 2/(24*60) #Exposure time (min.)
		data['Detrend LC_{}'.format(ii)] = False
		data['Poly LC_{}'.format(ii)] = 2
		data['FW LC_{}'.format(ii)] = 1001
		data['GP LC_{}'.format(ii)] = False#str2bool(df_phot['Gaussian Process'][ii])
		data['GP type LC_{}'.format(ii)] = 'Matern32'


	data['RVs'] = n_rvs
	for ii in range(1,n_rvs+1):
		data['RV_label_{}'.format(ii)] = 'Spectrograph {}'.format(ii)
		data['Fit RV_{}'.format(ii)] = False#str2bool(df_spec['Fit RV'][ii])
		data['RM RV_{}'.format(ii)] = False#str2bool(df_spec['Fit RM'][ii])		
		data['RV filename_{}'.format(ii)] = 'rv.txt'#np.zeros(shape=(10,3))

	data['LSs'] = n_ls
	for ii in range(1,n_ls+1):
		data['Fit LS_{}'.format(ii)] = False
		data['LS filename_{}'.format(ii)] = 'shadow.hdf5'

		data['Resolution_{}'.format(ii)] = 100 # Disc resolution
		data['Thickness_{}'.format(ii)] = 20 # Ring thickness
		data['Only_OOT_{}'.format(ii)] = False#str2bool(df_spec['Only fit OOT'][ii])
		data['OOT_{}'.format(ii)] = False#str2bool(df_spec['Fit OOT'][ii])
		#idxs = df_spec['OOT indices'][ii]
		data['idxs_{}'.format(ii)] = [0,1,2]#[int(ii) for ii in idxs.split(',')]
		data['Chi2 LS_{}'.format(ii)] = 1.0#float(df_spec['LS chi2 scaling'][ii])
		data['Chi2 OOT_{}'.format(ii)] = 1.0#float(df_spec['OOT chi2 scaling'][ii])
	
	data['SLs'] = n_sl
	for ii in range(1,n_sl+1):
		data['Fit SL_{}'.format(ii)] = False
		data['SL filename_{}'.format(ii)] = 'shadow.hdf5'
		data['SL_{}'.format(ii)] = shadow_data
		data['idxs_{}'.format(ii)] = [0,1,2]#[int(ii) for ii in idxs.split(',')]
		data['Chi2 SL_{}'.format(ii)] = 1.0#float(df_spec['SL chi2 scaling'][ii])		


	return data

def ini_data(data):
	'''Initialize the data.

	Load in data and initialize the GP (if `True`).

	'''

	#global data

	#data = dstruct.copy()

	n_phot = data['LCs']
	for ii in range(1,n_phot+1):
		fname = data['LC filename_{}'.format(ii)]
		arr = np.loadtxt(fname)
		data['LC_{}'.format(ii)] = arr
		if data['GP LC_{}'.format(ii)]:
			gp_type = data['GP type LC_{}'.format(ii)]
			if gp_type == 'Real':
				kernel = celerite.terms.RealTerm(log_a=0.5, log_c=0.1)
			else:
				kernel = celerite.terms.Matern32Term(log_sigma=-0.3, log_rho=-0.7)			
			gp = celerite.GP(kernel)
			gp.compute(arr[:,0],arr[:,2]) #probably redundant
			data['LC_{} GP'.format(ii)] = gp
		
	n_rvs = data['RVs']
	for ii in range(1,n_rvs+1):
		fname = data['RV filename_{}'.format(ii)]
		arr = np.loadtxt(fname)
		data['RV_{}'.format(ii)] = arr		

	n_ls = data['LSs']
	for ii in range(1,n_ls+1):
		fname = data['LS filename_{}'.format(ii)]
		with h5py.File(fname,'r') as ff:
			shadow_data = {}
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

	n_sl = data['SLs']
	for ii in range(1,n_sl+1):
		fname = data['SL filename_{}'.format(ii)]
		with h5py.File(fname,'r') as ff:
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
