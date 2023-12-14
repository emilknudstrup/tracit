#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. todo::
	* parameters, data only need to be made global within `business.mcmc/lmfitter` and `expose`, maybe set that in run_sys/run_bus.
	* move updated_pars out of par_struct
	* export kernel building -- probably easier, simpler to do it explicitly following `celerite: Kernel building <https://celerite.readthedocs.io/en/stable/python/kernel/>`_


"""
import string
import numpy as np
import h5py
import celerite
import pickle
from .shady import grid, grid_ring
from .dynamics import total_duration, time2phase

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
			- Add 'cosi_b' to ``parameters['FPs']``, walkers will step in :math:`\cos i` for planet 'b'.
			- Add 'ecosw_b', 'esinw_b', walkers will step in :math:`\sqrt{e}\cos \omega` & :math:`\sqrt{e}\sin \omega` for planet 'b'.
			- Remember to remove 'i_b', e_b', 'w_b', from ``parameters['FPs']`` accordingly.


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
			['Period','days',r'$P \rm _{} \  (days)$'.format(pl),4.8,0.5,0.0,1e4],	
			'T0_{}'.format(pl) : 
			['Midtransit','BJD',r'$T \rm _{} \ (BJD)$'.format('{0,'+pl+'}'),2457000.,0.5,0.0,1e10],
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
			['K-amplitude','m/s',r'$K  \rm _{}\ (m/s)$'.format(pl),30.,1.0,0.0,1e5],
			'Tw_{}'.format(pl) : 
			['Time of periastron passage','BJD',r'$T_\omega  \rm _{} \ (BJD)$'.format(pl),2457000.,0.5,2456999.,2457001],
			'lam_{}'.format(pl) : 
			['Projected obliquity','deg',r'$\lambda \rm _{} \ (^\circ)$'.format(pl),0.0,10.,-180.,180.],
			'cosi_{}'.format(pl) : 
			['Cosine inclination',' ',r'$\cos i \rm _{}$'.format(pl),0.0,0.1,0.0,1.0],
			'ecosw_{}'.format(pl) : 
			['sqrt(e) cos(w)',' ',r'$\sqrt{e} \ \cos \omega_'+pl+'$',0.0,0.1,-1.,1.],
			'esinw_{}'.format(pl) : 
			['sqrt(e) sin(w)',' ',r'$\sqrt{e} \ \sin \omega_'+pl+'$',0.0,0.1,-1.,1.],
			'vcosl_{}'.format(pl) : 
			['sqrt(v sin i) cos(lambda)',' ',r'$\sqrt{v \sin i} \ \cos \lambda_'+pl+' (\sqrt{km/s})$',0.0,5,-10.,10.],
			'vsinl_{}'.format(pl) : 
			['sqrt(v sin i) sin(lambda)',' ',r'$\sqrt{v \sin i} \ \sin \lambda_'+pl+' (\sqrt{km/s})$',0.0,5,-10.,10.],
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
				'Fix'          : False,
				'Comment'      : 'none',
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
				LD_pars[label+'_q{}'.format(ii)] = ['LD coeff {}'.format(ii),'nonlinear',r'$q_2 \ \rm {}$'.format(label,ii),0.4,0.1,0.0,1.0]

	#add_lc_pars = {}
	for ii in range(1,n_phot+1):
		star['LCblend_{}'.format(ii)] = ['Dilution (deltaMag=-2.5log(F2/F1)) photometer {}'.format(ii),' ',r'$\rm \delta M{}$'.format('_{LC,'+str(ii)+'}'),0.0,0.5,0.0,7.0]
		star['LCsigma_{}'.format(ii)] = ['Log jitter photometer {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LC,'+str(ii)+'}'),-30,0.05,-50,1.0]
		star['LClogsigma_{}'.format(ii)] = ['Log jitter photometer {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LC,'+str(ii)+'}'),-30,0.05,-50,1.0]
		star['LC_{}_GP_log_a'.format(ii)] = ['GP log amplitude, photometer {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{LC,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		star['LC_{}_GP_log_c'.format(ii)] = ['GP log time scale, photometer {}'.format(ii),' ',r'$\rm \log \tau{} \ (days)$'.format('_{LC,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		
		star['LC_{}_GP_log_S0'.format(ii)] = ['GP log amplitude, photometer {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{LC,'+str(ii)+'}'),-1.6,0.05,-5.0,5.0]
		star['LC_{}_GP_log_Q'.format(ii)] = ['GP log qaulity factor, photometer {}'.format(ii),' ',r'$\rm \log Q{}$'.format('_{LC,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		star['LC_{}_GP_log_w0'.format(ii)] = ['GP log frequency, photometer {}'.format(ii),' ',r'$\rm \log \omega{} \ (days{})$'.format('_{0,LC,'+str(ii)+'}','^{-1}'),-0.329,0.05,-5.0,5.0]
		
		star['LC_{}_GP_log_P'.format(ii)] = ['GP log period, photometer {}'.format(ii),' ',r'$\rm \log P{} \ (days{})$'.format('_{LC,'+str(ii)+'}','^{-1}'),1.16,0.05,0.01,2.0]
		star['LC_{}_GP_log_sig'.format(ii)] = ['GP log sigma, photometer {}'.format(ii),' ',r'$\rm \log \sigma{} $'.format('_{LC,'+str(ii)+'}'),0.15,0.05,0.0,5.0]
		star['LC_{}_GP_log_dQ'.format(ii)] = ['GP log qaulity factor difference, photometer {}'.format(ii),' ',r'$\rm \delta Q{}$'.format('_{LC,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		star['LC_{}_GP_f'.format(ii)] = ['GP log qaulity fractional amplitude, photometer {}'.format(ii),' ',r'$\rm f{}$'.format('_{LC,'+str(ii)+'}'),0.5,0.05,0.0,1.0]





		label = 'LC'+str(ii)
		set_LD(star,LD_law,label)

	#add_rv_pars = {}
	for ii in range(1,n_spec+1):
		star['RVsys_{}'.format(ii)] = ['Systemic velocity instrument {}'.format(ii),'m/s',r'$\gamma_{} \ \rm (m/s)$'.format(ii),0.0,1.,-1e5,1e5]
		star['RVsigma_{}'.format(ii)] = ['Jitter RV instrument {}'.format(ii),'m/s',r'$\rm \sigma{} \ (m/s)$'.format('_{RV,'+str(ii)+'}'),0.0,1.0,0.0,100.]
		star['RVlogsigma_{}'.format(ii)] = ['Jitter RV instrument {}'.format(ii),'m/s',r'$\rm \sigma{} \ (m/s)$'.format('_{RV,'+str(ii)+'}'),-30,1.0,-40.0,10.0]
		star['LSsigma_{}'.format(ii)] = ['Jitter LS instrument {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LS,'+str(ii)+'}'),-30.0,1.0,-50.0,10]

		star['RV_{}_GP_log_a'.format(ii)] = ['GP log amplitude, RV instrument {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{RV,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		star['RV_{}_GP_log_c'.format(ii)] = ['GP log time scale, RV instrument {}'.format(ii),' ',r'$\rm \log \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]


		star['LS_{}_GP_log_rho'.format(ii)] = ['GP log time scale, lineshape {}'.format(ii),' ',r'$\rm \log \tau{} \ (days)$'.format('_{LS,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		star['LS_{}_GP_log_diag'.format(ii)] = ['GP log diagonal elements, lineshape {}'.format(ii),' ',r'$\rm \log \sigma{}$'.format('_{LS,'+str(ii)+'}'),-6,0.05,-20.0,20.0]
		star['LS_{}_GP_log_a'.format(ii)] = ['GP log amplitude, lineshape {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{LS,'+str(ii)+'}'),-7,0.05,-10.0,5.0]
		star['LS_{}_GP_log_c'.format(ii)] = ['GP log time scale, lineshape {}'.format(ii),' ',r'$\rm \log \tau{} \ (days)$'.format('_{LS,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		star['LS_{}_GP_log_sigma'.format(ii)] = ['GP log amplitude, lineshape {}'.format(ii),' ',r'$\rm \log A{}$'.format('_{LS,'+str(ii)+'}'),-6,0.05,-15.0,-3.0]
		

		#parameters['GP pars RV_{}'.format(ii)] = []
		
		# for nn in range(1,n_kernels+1):
		# 	star['RV_{}_GP_log_a_{}'.format(ii,nn)] = ['GP log amplitude, RV {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{RV,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		# 	star['RV_{}_GP_log_c_{}'.format(ii,nn)] = ['GP log exponent, RV {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{RV,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		# 	star['LS_{}_GP_log_a_{}'.format(ii,nn)] = ['GP log amplitude, CCF {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{LS,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		# 	star['LS_{}_GP_log_c_{}'.format(ii,nn)] = ['GP log exponent, CCF {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{LS,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
			
		# 	star['RV_{}_GP_log_S0_{}'.format(ii,nn)] = ['GP log amplitude, RV {}'.format(ii),' ',r'$\rm \log S0{} \ (m~s{})$'.format('_{RV,'+str(ii)+'}','^{-1}'),-1.6,0.05,-5.0,5.0]
		# 	star['RV_{}_GP_log_Q_{}'.format(ii,nn)] = ['GP log qaulity factór, RV {}'.format(ii),' ',r'$\rm \log Q{}$'.format('_{RV,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		# 	star['RV_{}_GP_log_w0_{}'.format(ii,nn)] = ['GP log frequency, RV {}'.format(ii),' ',r'$\rm \log \omega{} \ (days{})$'.format('_{0,RV,'+str(ii)+'}','^{-1}'),-0.329,0.05,-5.0,5.0]
			
		# 	star['RV_{}_GP_sigma_{}'.format(ii,nn)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),-1.6,0.05,-5.0,5.0]
		# 	star['RV_{}_GP_tau_{}'.format(ii,nn)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		# 	star['RV_{}_GP_rho_{}'.format(ii,nn)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),-0.329,0.05,-5.0,5.0]

		# star['RV_{}_GP_log_a'.format(ii)] = ['GP log amplitude, RV {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{RV,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		# star['RV_{}_GP_log_c'.format(ii)] = ['GP log exponent, RV {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{RV,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		# star['LS_{}_GP_log_a'.format(ii)] = ['GP log amplitude, CCF {}'.format(ii),' ',r'$\rm \log a{}$'.format('_{LS,'+str(ii)+'}'),-7,0.05,-20.0,5.0]
		# star['LS_{}_GP_log_c'.format(ii)] = ['GP log exponent, CCF {}'.format(ii),' ',r'$\rm \log c{}$'.format('_{LS,'+str(ii)+'}'),-0.7,0.05,-5.0,5.0]
		
		# star['RV_{}_GP_log_S0'.format(ii)] = ['GP log amplitude, RV {}'.format(ii),' ',r'$\rm \log S0{} \ (m~s{})$'.format('_{RV,'+str(ii)+'}','^{-1}'),-1.6,0.05,-5.0,5.0]
		# star['RV_{}_GP_log_Q'.format(ii)] = ['GP log qaulity factór, RV {}'.format(ii),' ',r'$\rm \log Q{}$'.format('_{RV,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		# star['RV_{}_GP_log_w0'.format(ii)] = ['GP log frequency, RV {}'.format(ii),' ',r'$\rm \log \omega{} \ (days{})$'.format('_{0,RV,'+str(ii)+'}','^{-1}'),-0.329,0.05,-5.0,5.0]

		# star['RV_{}_GP_S0'.format(ii)] = ['GP amplitude, RV {}'.format(ii),' ',r'$\rm S0{} \ (m~s{})$'.format('_{RV,'+str(ii)+'}','^{-1}'),-1.6,0.05,-5.0,5.0]
		# star['RV_{}_GP_Q'.format(ii)] = ['GP qaulity factór, RV {}'.format(ii),' ',r'$\rm Q{}$'.format('_{RV,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		# star['RV_{}_GP_w0'.format(ii)] = ['GP frequency, RV {}'.format(ii),' ',r'$\rm \\omega{} \ (days{})$'.format('_{0,RV,'+str(ii)+'}','^{-1}'),-0.329,0.05,-5.0,5.0]

		
		# star['RV_{}_GP_sigma'.format(ii)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),-1.6,0.05,-5.0,5.0]
		# star['RV_{}_GP_tau'.format(ii)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),3.37,0.05,0.0,10.0]
		# star['RV_{}_GP_rho'.format(ii)] = ['GP time scale, RV {}'.format(ii),' ',r'$\rm  \tau{} \ (days)$'.format('_{RV,'+str(ii)+'}'),-0.329,0.05,-5.0,5.0]

		label = 'RV'+str(ii)
		set_LD(star,LD_law,label)

	star['a2'] = ['RV curve','m/(s*d^2)',r'$\ddot{\gamma} \ \rm (m \ s^{-1} \ d^{-2})$',0.0,1.0,-10.,10.]
	star['a1'] = ['RV slope','m/(s*d)',r'$\dot{\gamma} \ \rm (m \ s^{-1} \ d^{-1})$',0.0,1.0,-10.,10.]


	#for par in star.keys(): 
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
			'Fix'          : False,
			'Comment'      : 'none',
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
	## TTVs
	parameters['TTVs'] = []
	
	return parameters

def setTTVs(parameters,data,lightcurves=[],rvcurves=[],pls=['b']):
	## TTVs
	#parameters['TTVs'] = []
	#pls = parameters['Planets']
	#n_phot = data['LCs']

	parameters['TTVs'] = pls
	for pl in pls:
		#parameters['FPs'].remove('T0_{}'.format(pl))
		t0 = parameters['T0_{}'.format(pl)]['Value']
		per = parameters['P_{}'.format(pl)]['Value']

		aR = parameters['a_Rs_{}'.format(pl)]['Value']
		rp = parameters['Rp_Rs_{}'.format(pl)]['Value']
		inc = parameters['inc_{}'.format(pl)]['Value']
		ecc = parameters['e_{}'.format(pl)]['Value']
		ww = parameters['w_{}'.format(pl)]['Value']
		dur = total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
		if not np.isfinite(dur): continue

		n_uniques = []
		#for ii in range(1,n_phot+1):
		#print(lightcurves)
		for ii in lightcurves:
			data['LC_{} TTVs'.format(ii)] = 1
			fname = data['LC filename_{}'.format(ii)]
			arr = np.loadtxt(fname)
			#print('akjsdhkj',ii)
			time = arr[:,0]
			ph = time2phase(time,per,t0)*per*24						
			indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
			
			nn = np.array(np.round((time-t0)/per),dtype=int)
			ns = np.unique(nn)
			#print(ns)
			subn = []
			for n in ns:
				nidxs = np.where(nn == n)[0]
				idx = np.intersect1d(indxs,nidxs)
				if len(idx) > 0:
					n_uniques.append(n)
					subn.append(n)
			
			data['LC_{}_{}_n'.format(pl,ii)] = [nn,np.asarray(subn)]
			#data['LC_{}_n'.format(ii)] = [nn,np.asarray(subn)]
			#data['LC_{}'.format(ii)] = arr
		n_uniques = np.unique(n_uniques)

		#t0s = ['T0_b_{}'.format(int(n)) for n in n_uniques]
		for ii in rvcurves:
			data['RV_{} TTVs'.format(ii)] = 1
			fname = data['RV filename_{}'.format(ii)]
			arr = np.loadtxt(fname)
			time = arr[:,0]
			ph = time2phase(time,per,t0)*per*24						
			indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
			#print('akjsdhkj')
			#print('akjsdhkj')
			nn = np.array(np.round((time-t0)/per),dtype=int)
			ns = np.unique(nn)
			subn = []
			for n in ns:
				nidxs = np.where(nn == n)[0]
				idx = np.intersect1d(indxs,nidxs)
				if len(idx) > 0:
					if n not in n_uniques: n_uniques = np.append(n_uniques,n)
					subn.append(n)
			
			data['RV_{}_{}_n'.format(pl,ii)] = [nn,np.asarray(subn)]			

		for n in n_uniques: 
			lab = 'T0_{}_{}'.format(pl,n)
			parameters['FPs'].append(lab)
			parameters[lab] = {
				'Name'         : 'Midtransit',
				'Unit'         : 'BJD',
				'Label'        : r'$T \rm _{} \ (BJD)$'.format('{0,'+pl+','+str(n)+'}'),
				'Value'        : parameters['T0_{}'.format(pl)]['Value'],
				'Prior_vals'   : parameters['T0_{}'.format(pl)]['Prior_vals'],
				'Prior'        : 'uni',
				'Distribution' : 'tgauss',
				'Fix'          : False,
				'Comment'      : 'none',
			}
			#mid = parameters['T0_b']
			#parameters[lab] = ['Midtransit','BJD',r'$T \rm _{}_{} \ (BJD)$'.format('{0,'+pl+','+str(n)+'}'),2457000.,0.5,0.0,1e10],

			#'T0_{}'.format(pl) : 


def setTransits(parameters,data,lightcurves=[],pls=['b'],durations=[7]):
	## TTVs
	#parameters['TTVs'] = []
	#pls = parameters['Planets']
	#n_phot = data['LCs']

	parameters['TTVs'] = pls

		#n_uniques = []
		#for ii in range(1,n_phot+1):
		#print(lightcurves)
	import matplotlib.pyplot as plt
	for ii in lightcurves:
		#data['LC_{} TTVs'.format(ii)] = 1
		
		fname = data['LC filename_{}'.format(ii)]
		arr = np.loadtxt(fname)
		time = arr[:,0]

		idxs = np.array([],dtype=int)
		#fig = plt.figure()
		#ax = fig.add_subplot(111)
		#ax.plot(time,arr[:,1],'ko')
		for jj, pl in enumerate(pls):
			#parameters['FPs'].remove('T0_{}'.format(pl))
			t0 = parameters['T0_{}'.format(pl)]['Value']
			per = parameters['P_{}'.format(pl)]['Value']

			aR = parameters['a_Rs_{}'.format(pl)]['Value']
			rp = parameters['Rp_Rs_{}'.format(pl)]['Value']
			inc = parameters['inc_{}'.format(pl)]['Value']
			ecc = parameters['e_{}'.format(pl)]['Value']
			ww = parameters['w_{}'.format(pl)]['Value']
			dur = total_duration(per,rp,aR,inc*np.pi/180.,ecc,ww*np.pi/180.)*24
			if not np.isfinite(dur): 
				dur = durations[jj]
				print('Duration for {} is not finite (given parameters). Setting to {} hr.'.format(pl,dur))
				#continue
			#print('akjsdhkj',ii)
			ph = time2phase(time,per,t0)*per*24						
			indxs = np.where((ph < (dur/2 + 6)) & (ph > (-dur/2 - 6)))[0]
			
			nn = np.array(np.round((time-t0)/per),dtype=int)
			ns = np.unique(nn)
			#print(ns)
			subn = []
			for n in ns:
				nidxs = np.where(nn == n)[0]
				idx = np.intersect1d(indxs,nidxs)
				if len(idx) > 0:
					#n_uniques.append(n)
					#subn.append(n)
					idxs = np.append(idxs,idx)
					#ax.plot(time[idx],arr[idx,1],'.')	
		#print(idxs)
		data['LC_{}_n'.format(ii)] = np.unique(idxs)
		data['LC_{}_gaps'.format(ii)] = np.where(np.diff(np.unique(idxs)) > 1)[0]

			
			#data['LC_{}_{}_n'.format(pl,ii)] = [nn,np.asarray(subn)]
			#data['LC_{}_n'.format(ii)] = [nn,np.asarray(subn)]
			#data['LC_{}'.format(ii)] = arr
		#n_uniques = np.unique(n_uniques)
		#data['LC_{}_n'.format(ii)] = [nn,np.asarray(subn)]

def check_fps(parameters):
	'''Check fitting parameters.

	Run an initial check of the fitting parameters to avoid stepping in, for instance, both :math:`i` and :math:`\cos i`.

	:param parameters: The parameters.
	:type parameters: dict

	'''

	fps = parameters['FPs']
	snowflake = np.unique(fps)
	if len(fps) != len(snowflake):
		print('Some fitting paramaters entered twice (or more).\nRemoving...')
		parameters['FPs'] = list(snowflake)

	pls = parameters['Planets']

	for pl in pls:
		if 'ecosw_{}'.format(pl) in fps:
			try:
				fps.remove('e_{}'.format(pl))
			except ValueError:
				pass
			try:
				fps.remove('w_{}'.format(pl))
			except ValueError:
				pass
			if not 'esinw_{}'.format(pl) in fps:
				fps.append('esinw_{}'.format(pl))
		if 'esinw_{}'.format(pl) in fps:
			try:
				fps.remove('e_{}'.format(pl))
			except ValueError:
				pass
			try:
				fps.remove('w_{}'.format(pl))
			except ValueError:
				pass
			if not 'ecosw_{}'.format(pl) in fps:
				fps.append('ecosw_{}'.format(pl))
		if 'cosi_{}'.format(pl) in fps:
			try:
				fps.remove('inc_{}'.format(pl))
			except ValueError:
				pass

		if 'vcosl_{}'.format(pl) in fps:
			try:
				fps.remove('vsini')
			except ValueError:
				pass
			try:
				fps.remove('lambda_{}'.format(pl))
			except ValueError:
				pass
			if not 'vsinl_{}'.format(pl) in fps:
				fps.append('vsinl_{}'.format(pl))

		if 'vsinl_{}'.format(pl) in fps:
			try:
				fps.remove('vsini')
			except ValueError:
				pass
			try:
				fps.remove('lambda_{}'.format(pl))
			except ValueError:
				pass
			if not 'vcosl_{}'.format(pl) in fps:
				fps.append('vcosl_{}'.format(pl))

	LD_lincombs = parameters['LinCombs']
	for fp in fps:
		if '_sum' in fp:
			if fp not in LD_lincombs:
				LD_lincombs.append(fp.split('_sum')[0])

	for LDlc in LD_lincombs:
		LD1 = LDlc + '1'
		LD2 = LDlc + '2'
		try:
			fps.remove(LD1)
		except ValueError:
			pass
		try:
			fps.remove(LD1)
		except ValueError:
			pass
		ldsum = LDlc+'_sum'
		if ldsum not in fps:
			fps.append(ldsum)

	for fp in fps:
		if ('Phot' in fp ) or ('Spec' in fp):
			first_label = fp.split(':')[-1]
			instrument = fp.split(':')[0]
			label = '$' + first_label.split('_')[0] + '\\rm _' + '{' + first_label.split('_')[-1] + '}, \ ' + instrument + '$'
			
			parameters[fp] = {
				'Name'         : 'Midtransit, ' + instrument,
				'Unit'         : 'BJD',
				'Label'        : label,#ndf[handle][1][:-1] + ' ' + transit + '$',
				'Value'        : parameters[first_label]['Value'],
				'Prior_vals'   : [parameters[first_label]['Prior_vals'][ii] for ii in range(4)],
				'Prior'        : 'uni',
				'Distribution' : 'tgauss',
				'Fix'          : False,
				'Comment'      : 'none',
			}


# def ini_pars(parameters):
# 	LD_lincombs = parameters['LinCombs']

# 	for LDhandle in LD_lincombs:
# 		LDhandle1 = LDhandle + '1'
# 		LDhandle2 = LDhandle + '2'
# 		sumhandle = LDhandle + '_sum'
# 		parameters['FPs'].append(sumhandle)
# 		diffhandle = LDhandle + '_diff'
# 		parameters[sumhandle] = {
# 			'Unit'         : ndf[LDhandle1][0],
# 			'Label'        : r'$q_1 + q_2: \rm {}$'.format(LDhandle1.split('_')[0]),
# 			'Value'        : True,
# 			'Value'        : float(ndf[LDhandle1][2]) + float(ndf[LDhandle2][2]),
# 			'Prior_vals'   : [float(ndf[LDhandle1][2]) + float(ndf[LDhandle2][2]),0.1,0.0,2.0],
# 			'Prior'        : 'tgauss',
# 			'Distribution' : 'tgauss',
# 			'Fix' : False
# 		}			
# 		parameters[diffhandle] = {
# 			'Unit'         : ndf[LDhandle1][0],
# 			'Label'        : r'$q_1 - q_2: \rm {}$'.format(LDhandle1.split('_')[0]),
# 			'Value'        : float(ndf[LDhandle1][2]) - float(ndf[LDhandle2][2]),
# 			'Prior_vals'   : [float(ndf[LDhandle1][2]) - float(ndf[LDhandle2][2]),0.1,-1.0,1.0],
# 			'Prior'        : 'gauss',
# 			'Distribution' : ndf[handle][7],
# 			'Fix' : False
# 		}


# def update_pars(rdf,parameters,best_fit=True):
# 	'''Update parameters.

# 	Updates the parameters dictionary.

# 	:param rdf: Results from fit or MCMC.
# 	:type rdf: `pandas.DataFrame`

# 	:param parameters: The parameters.
# 	:type parameters: dict

# 	:param best_fit: Use best-fitting (:math:`\max(\log \mathcal{L})`), if ``False`` median is used.
# 	:type best_fit: bool


# 	'''
# 	pars = rdf.keys()[1:-2]
# 	idx = 1
# 	if (rdf.shape[0] > 3) & best_fit: idx = 4
# 	for par in pars:
# 		try:
# 			parameters[par]['Value'] = float(rdf[par][idx])	
# 		except KeyError:
# 			pass

def str2bool(arg):
	if isinstance(arg, bool):
		return arg
	if arg.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise TypeError('Boolean value expected.')

def update_pars(rdf,parameters,best_fit=True,mcmc=True):
	'''Update parameters.

	Updates the parameters dictionary.

	:param rdf: Results from fit or MCMC.
	:type rdf: `pandas.DataFrame`

	:param parameters: The parameters.
	:type parameters: dict

	:param best_fit: Use best-fitting (:math:`\max(\log \mathcal{L})`), if ``False`` median is used.
	:type best_fit: bool


	'''
	pars = rdf.keys()[1:-2]
	idx = 1
	if mcmc:
		if best_fit: idx = 4
		for par in pars:
			try:
				parameters[par]['Value'] = float(rdf[par][idx])	
			except KeyError:
				pass
	else:
		for par in pars:
			if str2bool(rdf[par][2]):
				parameters[par]['Value'] = float(rdf[par][idx])	


def writeInput(parameters={},data={},filename='./input.pkl'):

	w = {'Parameters' : parameters, 'Data' : data}
	with open(filename,'wb') as file:
			pickle.dump(w,file)


def readInput(filename='./input.pkl'):
	with open(filename,'rb') as file:
		w = pickle.load(file)
	return w['Parameters'], w['Data']

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
		data['Poly LC_{}'.format(ii)] = 1
		data['FW LC_{}'.format(ii)] = 1001
		data['GP LC_{}'.format(ii)] = False#str2bool(df_phot['Gaussian Process'][ii])
		data['GP type LC_{}'.format(ii)] = 'Matern32'
		data['LC_{} TTVs'.format(ii)] = 0


	data['RVs'] = n_rvs
	for ii in range(1,n_rvs+1):
		data['RV_label_{}'.format(ii)] = 'Spectrograph\ {}'.format(ii)
		data['Fit RV_{}'.format(ii)] = False#str2bool(df_spec['Fit RV'][ii])
		data['RM RV_{}'.format(ii)] = False#str2bool(df_spec['Fit RM'][ii])		
		data['RV filename_{}'.format(ii)] = 'rv.txt'#np.zeros(shape=(10,3))
		data['GP RV_{}'.format(ii)] = False#str2bool(df_phot['Gaussian Process'][ii])
		data['GP type RV_{}'.format(ii)] = 'Matern32'
		data['RV_{} TTVs'.format(ii)] = 0
		data['PSF_{}'.format(ii)] = 1.1 # Point spread function of PSF | FWHM=c/R, sigma=FWHM/(2sqrt(2ln2))) | default PSF=1.1km/s -- R=115,000 (HARPS/HARPS-N)

	data['LSs'] = n_ls
	for ii in range(1,n_ls+1):
		data['LS_label_{}'.format(ii)] = 'Spectrograph\ {}'.format(ii)
		data['Fit LS_{}'.format(ii)] = False
		data['LS filename_{}'.format(ii)] = 'shadow.hdf5'
		
		data['GP LS_{}'.format(ii)] = False#str2bool(df_phot['Gaussian Process'][ii])
		data['GP type LS_{}'.format(ii)] = 'Matern32'

		data['Resolution_{}'.format(ii)] = 100 # Disc resolution
		data['Thickness_{}'.format(ii)] = 20 # Ring thickness
		data['PSF_{}'.format(ii)] = 1.1 # Point spread function of PSF | FWHM=c/R, sigma=FWHM/(2sqrt(2ln2))) | default PSF=1.1km/s -- R=115,000 (HARPS/HARPS-N)
		data['Velocity_resolution_{}'.format(ii)] = 0.25 # Resolution of grid in velocity space in km/s
		data['Velocity_range_{}'.format(ii)] = 15 # Range of velocity grid in km/s
		data['No_bump_{}'.format(ii)] = 15 # Flat part of CCF (i.e., no bump) in km/s
		data['Only_OOT_{}'.format(ii)] = False#str2bool(df_spec['Only fit OOT'][ii])
		data['OOT_{}'.format(ii)] = False#str2bool(df_spec['Fit OOT'][ii])
		#idxs = df_spec['OOT indices'][ii]
		data['idxs_{}'.format(ii)] = [0,1,2]#[int(ii) for ii in idxs.split(',')]
		data['Chi2 LS_{}'.format(ii)] = 1.0#float(df_spec['LS chi2 scaling'][ii])
		data['Chi2 OOT_{}'.format(ii)] = 1.0#float(df_spec['OOT chi2 scaling'][ii])
	
	data['SLs'] = n_sl
	for ii in range(1,n_sl+1):
		data['SL_label_{}'.format(ii)] = 'Spectrograph\ {}'.format(ii)
		data['Fit SL_{}'.format(ii)] = False
		data['SL filename_{}'.format(ii)] = 'shadow.hdf5'
		#data['SL_{}'.format(ii)] = shadow_data
		data['idxs_{}'.format(ii)] = [0,1,2]#[int(ii) for ii in idxs.split(',')]
		data['Chi2 SL_{}'.format(ii)] = 1.0#float(df_spec['SL chi2 scaling'][ii])		
		data['Velocity_resolution_{}'.format(ii)] = 0.25 # Resolution of grid in velocity space in km/s
		data['No_bump_{}'.format(ii)] = 15 # Flat part of CCF (i.e., no bump) in km/s
		data['PSF_{}'.format(ii)] = 1.1 # Point spread function of PSF | FWHM=c/R, sigma=FWHM/(2sqrt(2ln2))) | default PSF=1.1km/s -- R=115,000 (HARPS/HARPS-N)


	return data

def ini_grid(rad_disk=100,thickness=20):
	'''Initialize stellar grid.

	:param rad_disk: Stellar radius in pixels.
	:type rad_disk: int

	:param thickness: Thickness of the stellar rings in pixels.
	:type thickness: int

	:return: start grid, rings of same :math:`\mu`, velocity grid, radial :math:`\mu` values, approx :math:`\mu` in each ring
	:rtype: array, array, array, array, array

	'''
	## Make initial grid
	start_grid, vel, mu = grid(rad_disk) #make grid of stellar disk
	## The grid is made into rings for faster calculation of the macroturbulence (approx. constant in each ring)
	ring_grid, vel, mu_grid, mu_mean = grid_ring(rad_disk,thickness) 
	
	return start_grid, ring_grid, vel, mu, mu_grid, mu_mean


def ini_data(data):
	'''Initialize the data.

	Load in data and initialize the GP (if `True`).
	
	:param data: The data.
	:type data: dict
	
	'''

	#global data

	#data = dstruct.copy()

	n_phot = data['LCs']
	for ii in range(1,n_phot+1):
		fname = data['LC filename_{}'.format(ii)]
		arr = np.loadtxt(fname)
		data['LC_{}'.format(ii)] = arr
		try:
			if data['GP LC_{}'.format(ii)]:
				gp_type = data['GP type LC_{}'.format(ii)]
				if gp_type == 'Real':
					kernel = celerite.terms.RealTerm(log_a=0.5, log_c=0.1)
				elif gp_type == 'Matern32':
					kernel = celerite.terms.Matern32Term(log_sigma=-0.3, log_rho=-0.7)			
				elif gp_type == 'SHO':
					kernel = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)			
				elif gp_type == 'mix':
					kernel = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)
					kernel += celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)			
				jitter = 1
				if jitter:
					kernel += celerite.terms.JitterTerm(log_sigma=-30)
				gp = celerite.GP(kernel)
				#gp.compute(arr[:,0],arr[:,2]) #probably redundant

				data['LC_{} GP'.format(ii)] = gp
		except KeyError:
			pass
		
	n_rvs = data['RVs']
	for ii in range(1,n_rvs+1):
		fname = data['RV filename_{}'.format(ii)]
		arr = np.loadtxt(fname)
		data['RV_{}'.format(ii)] = arr		

		try:
			data['PSF_{}'.format(ii)]
		except KeyError:
			data['PSF_{}'.format(ii)] = 1.1 # Point spread function of spectrograph (km/s)


		try:
			if data['GP RV_{}'.format(ii)]:
				gp_type = data['GP type RV_{}'.format(ii)]
				if gp_type == 'Real':
					kernel = celerite.terms.RealTerm(log_a=0.5, log_c=0.1)
				elif gp_type == 'Matern32':
					kernel = celerite.terms.Matern32Term(log_sigma=-0.3, log_rho=-0.7)			
				elif gp_type == 'logSHO':
					kernel = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)	
					# import celerite2
					# from celerite2 import terms
					# kernel = terms.SHOTerm(sigma=-0.3,rho=-0.2,tau=0.1)		
					# gp = celerite2.GaussianProcess(kernel,mean=0.0)

				elif gp_type == 'SHO':
					#kernel = celerite.terms.SHOTerm(sigma=1.0, tau=2.0, rho=10)			
					#kernel = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)	
					t1 = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)	
					print(gp_type)
					P = 2*np.pi/(90*24*3600)
					#P = 1000
					a = 8
					t2 = celerite.terms.RealTerm(log_a=np.log(a), log_c=np.log(P))

					P3 = 2*np.pi/(2*24*3600)
					#P = 1000
					a3 = 8
					t3 = celerite.terms.RealTerm(log_a=np.log(a3), log_c=np.log(P3))
					kernel = t1 + t2 + t3
				#gp = celerite.GP(kernel)
				# gp_types = data['GP type RV_{}'.format(ii)]
				# #if type(gp_types) != list: gp_types = list(gp_types)
				# assert type(gp_types) == list, 'GP type RV_{}'.format(ii) + "must be a list of GP types, for instance, ['Real','logSHO'] or ['logSHO']"
				# kernel = None
				# for gp_type in gp_types:
				# 	#kernel = None
				# 	if gp_type == 'Real':
				# 		term = celerite.terms.RealTerm(log_a=0.5, log_c=0.1)
				# 	elif gp_type == 'Matern32':
				# 		term = celerite.terms.Matern32Term(log_sigma=-0.3, log_rho=-0.7)			
				# 	elif gp_type == 'logSHO':
				# 		#term = celerite.terms.SHOTerm(log_S0=-0.3, log_Q=-0.7,log_omega0=-0.139)			
				# 		term = celerite.terms.SHOTerm(log_S0=8.03,log_omega0=-6.697629451765305, log_Q=1.38)
				# 		# S0 = 3000
				# 		# logS0=np.log(S0)
				# 		# w0 = 189*1e-6*2*np.pi
				# 		# Q = 4
				# 		# import celerite2
				# 		# from celerite2 import terms

				# 		# term = terms.SHOTerm(sigma=np.sqrt(w0*np.exp(logS0)*Q),rho=2*np.pi/w0,tau=2*Q/w0)

				# 	elif gp_type == 'SHO':
				# 		term = celerite.terms.SHOTerm(sigma=1.0, tau=2.0, rho=10)			

				# 	if not kernel:
				# 		kernel = term

				# 	else:
				# 		kernel += term
				jitter = 1
				if jitter:
					kernel += celerite.terms.JitterTerm(log_sigma=-10)
				gp = celerite.GP(kernel)
				data['RV_{} GP'.format(ii)] = gp
				#gp = celerite2.GaussianProcess(kernel,mean=0.0)
				#gp.compute(arr[:,0],arr[:,2]) #probably redundant

		except KeyError:
			pass

	n_ls = data['LSs']
	for ii in range(1,n_ls+1):
		fname = data['LS filename_{}'.format(ii)]
		with h5py.File(fname,'r') as ff:
			shadow_data = {}
			times = []
			for key in ff.keys():
				times.append(float(key))
			times = np.asarray(times)
			ss = np.argsort(times)
			times = times[ss]					
			for time in times:
				arr = ff[str(time)][:]
				#raw_vel = arr[:,0]
				shadow_data[time] = {
					'vel' : arr[:,0],#raw_vel,
					'ccf' : arr[:,1],
					'err' : arr[:,2]
					}
		
		try:
			if data['GP LS_{}'.format(ii)]:
				gp_type = data['GP type LS_{}'.format(ii)]
				if gp_type == 'Real':
					kernel = celerite.terms.RealTerm(log_a=0.5, log_c=0.1)
				else:
					kernel = celerite.terms.Matern32Term(log_sigma=-0.3, log_rho=-0.7)			
				jitter = 1
				if jitter:
					kernel += celerite.terms.JitterTerm(log_sigma=-10)
				gp = celerite.GP(kernel)
				#gp.compute(arr[:,0],arr[:,2]) #probably redundant
				data['LS_{} GP'.format(ii)] = gp
				
		except KeyError:
			pass

		try:
			data['PSF_{}'.format(ii)]
		except KeyError:
			data['PSF_{}'.format(ii)] = 1.1 # Point spread function of spectrograph (km/s)
		try:
			data['idxs_{}'.format(ii)] 
		except KeyError:
			data['idxs_{}'.format(ii)] = [0,1,2] # OOT indices
		try:
			data['Only_OOT_{}'.format(ii)] 
		except KeyError:
			data['Only_OOT_{}'.format(ii)] = False
		try:
			data['OOT_{}'.format(ii)] 
		except KeyError:
			data['OOT_{}'.format(ii)] = False

		try:
			data['Fit LS_{}'.format(ii)] 
		except KeyError:
			data['Fit LS_{}'.format(ii)] = False # Fit shadow?
		try:
			data['LS_label_{}'.format(ii)] 
		except KeyError:
			data['LS_label_{}'.format(ii)] = 'Spectrograph\ {}'.format(ii)

		data['LS_{}'.format(ii)] = shadow_data	

		try:
			resol = data['Resolution_{}'.format(ii)]
		except KeyError:
			resol = 100 # Disc resolution
			data['Resolution_{}'.format(ii)] = resol
		try:
			thick = data['Thickness_{}'.format(ii)]
		except KeyError:
			thick = 20 # Ring thickness
			data['Thickness_{}'.format(ii)] = thick
		
		start_grid, ring_grid, vel_grid, mu, mu_grid, mu_mean = ini_grid(resol,thick)
		data['Start_grid_{}'.format(ii)] = start_grid
		data['Ring_grid_{}'.format(ii)] = ring_grid
		data['Velocity_{}'.format(ii)] = vel_grid
		data['mu_{}'.format(ii)] = mu
		data['mu_grid_{}'.format(ii)] = mu_grid
		data['mu_mean_{}'.format(ii)] = mu_mean
		
		## Resolution of velocity grid
		vel_res = data['Velocity_resolution_{}'.format(ii)]

		## Range without a bump in the CCFs
		no_bump = data['No_bump_{}'.format(ii)]
		span = data['Velocity_range_{}'.format(ii)]
		assert span > no_bump, print('\n ### \n The range of the velocity grid must be larger than the specified range with no bump in the CCF.\n Range of velocity grid is from +/-{} km/s, and the no bump region isin the interval m +/-{} km/s \n ### \n '.format(span,no_bump))
		vels = np.arange(-span,span,vel_res)
		data['Velocity_grid_{}'.format(ii)] = vels

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

		try:
			data['Fit SL_{}'.format(ii)] 
		except KeyError:
			data['Fit SL_{}'.format(ii)] = False # Fit slope?

		try:
			data['SL_label_{}'.format(ii)] 
		except KeyError:
			data['SL_label_{}'.format(ii)] = 'Spectrograph\ {}'.format(ii)

		data['SL_{}'.format(ii)] = shadow_data


def createPars(parameters,pars=[]):

	for par in pars:
		parameters[par] = {
			'Name'         : 'Name',
			'Unit'         : 'unit',
			'Label'        : r'$\rm Label$',
			'Value'        : 1.0,
			'Prior_vals'   : [2.0,0.5,0.,10.],
			'Prior'        : 'uni',
			'Distribution' : 'tgauss',
			'Fix'          : False,
			'Comment'      : 'none',
		}



def get_expTime(data,setexp=False):
	'''Estimate exposure times.

	Exposure time estimated as the mean of the difference between timestamps in the time series (in minutes).
	
	:param data: The data.
	:type data: dict

	:param setexp: Whether to set data['Exp. LC_i'] to the estimated time.
	:type setexp: bool 
	
	'''

	n_phot = data['LCs']
	for ii in range(1,n_phot+1):
		arr = data['LC_{}'.format(ii)]
		exp = np.median(np.diff(arr[:,0]))
		print('Exsposure time for LC_{} is {:0.3f} min.'.format(ii,exp*24*60))
		if setexp: data['Exp. LC_{}'.format(ii)] = exp
	
	# n_rvs = data['RVs']
	# for ii in range(1,n_rvs+1):
	# 	arr = data['RV_{}'.format(ii)]
	# 	exp = np.mean(np.diff(arr[:,0]))*24*60
	# 	print('Exsposure time for RV_{} is {:0.3f} min.'.format(ii,exp))