#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:27:04 2020

@author: emil
"""

import numpy as np
import scipy.signal as ss
import dynamics
import sys
# =============================================================================
# Grids
# =============================================================================

def grid_coordinates(Rs,xoff=0.,yoff=0.):
	'''Coordinates of the stellar grid.

	Calculate the coordinates of the stellar grid.
	
	'''
	xx, yy = np.arange(-Rs+xoff,Rs+1+xoff),np.arange(-Rs+yoff,Rs+1+yoff)
	coord = np.array(np.meshgrid(xx,yy)).T.reshape(-1,2)
	#rr = np.sqrt(np.sum(coord**2,axis=1))
	rr = np.sqrt(np.add.reduce(coord**2,axis=1))
	
	dd = np.arange(0,2*Rs+1,1)
	cidxs = [tuple(cc) for cc in np.array(np.meshgrid(dd,dd)).T.reshape(-1,2)] # Indices for coordinates
	return coord, cidxs, rr

def grid(Rs,xoff=0,yoff=0):
	'''Initial grid of the star.

	:param Rs: Stellar radius in number of pixels.
	:type Rs: int 

	'''
	coord, cidxs, rr = grid_coordinates(Rs,xoff=xoff,yoff=yoff)

	## Empty, quadratic array with dimensions diameter*diameter
	start_grid = np.zeros((2*Rs+1,2*Rs+1))
	vel = start_grid.copy()
	mu = start_grid.copy()
	#t0 = tt.time()
	for ii in range(len(cidxs)):
		if rr[ii] <= Rs:
			start_grid[cidxs[ii]] = 1
			mu[cidxs[ii]] = np.sqrt(1-(rr[ii]/float(Rs))**2) # used for limb darkening. Same as cos(inclination)
		vel[cidxs[ii]] = coord[ii][0]/float(Rs) #velocity field. [0]-axis (rows) are cst vel, while [1]-axis (columns) are the equator

	return start_grid, vel, mu


def grid_ring(Rs,thick,xoff=0.,yoff=0.):
	'''Initial grid of star in rings of mu.

	Initial grid of star in rings of aprrox same mu = cos(theta).
	Useful for macroturbulence calculations.
	
	:param Rs: Stellar radius in number of pixels.
	:type Rs: int 
	:param thick: Thickness of rings.
	:type thick: int

	:return: 
	:rtype: (array, array, array, array)

	'''
	assert Rs/thick > 2.0, print('The stellar radius must be at least twice the size of the rings.')
	coord, cidxs, rr = grid_coordinates(Rs,xoff=xoff,yoff=yoff)
	
	## Empty, quadratic array with dimensions diameter*diameter
	start_grid = np.zeros((2*Rs+1,2*Rs+1))
	vel = start_grid.copy()	

	## Divide star into rings
	rings = np.arange(0,Rs+1,thick)
	nr = len(rings) # number of rings
	rings_in = rings - thick
	## Make sure to get all of the star
	if rings[-1] < Rs:
		rings_in = np.append(rings_in,rings[-1])
		rings = np.append(rings,Rs)

	#used for limb darkening and macro. Same as cos(theta)
	mu_grid = np.asarray([start_grid.copy() for i in range(nr)])
	ring_grid = mu_grid.copy()#np.asarray([start_grid.copy() for i in range(nr)]) #startgrid for ring
	for ii in range(len(cidxs)):
		for jj in range(nr):
			if rings_in[jj] < rr[ii] <= rings[jj]:
				ring_grid[jj][cidxs[ii]] = 1
				#print(np.sqrt(1-(rr[ii]/float(Rs))**2))#used for limb darkening and macro. Same as cos(theta)
				mu_grid[jj][cidxs[ii]] = np.sqrt(1-(rr[ii]/float(Rs))**2)#used for limb darkening and macro. Same as cos(theta)
		vel[cidxs[ii]] = coord[ii][0]/float(Rs) #velocity field. [0]-axis (rows) are cst vel, while [1]-axis (columns) are the equator
	mu_mean = np.zeros(shape=(1,nr))[0]

	## Calculate the approx mu in each ring to be used when calculating the macroturbulence
	for kk in range(nr):
		mu_mean[kk] = np.mean(mu_grid[kk][np.where(mu_grid[kk])])

	return ring_grid, vel, mu_grid, mu_mean

def transit_ring(vel,vel_ext,ring_LD,mu_grid,mu_mean,lum,
				vsini,xi,zeta,Rp_Rs,radius,
				time,Tw,ecc,per,w,a_Rs,inc,lam):
	'''Planet position in each ring.

	Function that calculates the planet signal in each ring.
	This includes the effects of limb-darkening, as well as micro- and macroturbulence.
	
	'''
	rot_profile = vel*vsini
	nn = len(time)

	rp = int(round(Rp_Rs*radius))
	pl_grid, pl_vel, mu = grid(rp)
	cos_f, sin_f = dynamics.true_anomaly(time,Tw,ecc,per,w)
	xx, yy = dynamics.xy_pos(cos_f,sin_f,ecc,w,a_Rs,inc,lam)
	xnorm, ynorm = xx*radius, yy*radius
	x_off, y_off = np.rint(xnorm), np.rint(ynorm)
	x_pl, y_pl = abs(x_off)-rp, abs(y_off)-rp
	## Coordinate values at planet position in grid
	sum_mu_grid = np.add.reduce(mu_grid,axis=0)

	xn = np.empty(shape=(nn,len(vel_ext)))
	line_conv = np.empty(shape=(nn,len(ring_LD),len(vel_ext)))
	

	rescape_pl = np.reshape(pl_grid,np.size(pl_grid))
	dark = np.zeros(nn)
	mu_size = np.size(mu_grid,axis=1)

	ring_planet = np.zeros_like(ring_LD)
	index_error = False
	for kk in range(nn):
		## Only do calculation if planet is "touching" stellar disk
		it_lum = lum.copy()
		if (x_pl[kk] < radius) and (y_pl[kk] < radius):
			
			x_pos = int(x_off[kk] + radius)
			y_pos = int(y_off[kk] + radius)

			pl_coord, pl_coord_arr, pl_coord_idx = grid_coordinates(rp,x_pos,y_pos)
			
			pl_zip = [(pl_coord[ii,0],pl_coord[ii,1]) for ii in range(pl_coord.shape[0])]
			## Makes sure that when a planet is about to disappear 
			## it does not appear on the other side.
			coord_pl = [i for (i,v) in zip(pl_zip,rescape_pl) if v==1. and i[0] >= 0 and i[1] >= 0 and i[0] < ring_LD[0].shape[0] and i[1] < ring_LD[0].shape[0]]


			try:
				ring_planet[:,np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]] = ring_LD[:,np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]]
				it_lum[np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]] = 0
			except IndexError:
				## mu values at planet position in mu_grid
				mu_pl = np.asarray([sum_mu_grid[i] for (i,v) in zip(pl_zip,rescape_pl) if v==1. and i[0]<mu_size and i[1]<mu_size]) 
				for ii in range(len(ring_LD)):
					for jj in range(len(mu_pl)):
						try:
							## Makes sure that when a planet is about to disappear 
							## it does not appear on the other side.
							ring_planet[ii][coord_pl[jj]] = ring_LD[ii][coord_pl[jj]]
						except IndexError:
							index_error = True
					it_lum[ring_planet[ii].astype(bool)] = 0

			xn[kk,:], line_conv[kk,:,:] = convolve(rot_profile,ring_planet,mu_mean,xi,zeta)
			ring_planet[:,:,:] = 0

		else:
			xn[kk,:] = vel_ext
			line_conv[kk,:,:] = np.zeros(shape=(len(ring_LD),len(vel_ext)))
		
		dark[kk] = np.sum(it_lum)

	return xn, line_conv, dark, index_error#planet_rings


# =============================================================================
# Convolutions
# =============================================================================

def macro(vel,mu_mean,zeta):
	'''Macroturbulence at given distance from center.

	Function that calculates the macroturbulence for given :math:`\zeta` and `\mu` using the radial-tangential profile.


	'''

	A = 0.5 #area of curve covered by radial and tangential
	vel_1d = vel[0]

	mac = np.zeros(shape=(len(mu_mean),len(vel_1d)))


	for ii, mu in enumerate(mu_mean):
		rad = np.exp(-1*np.power(vel_1d/(zeta*mu),2))/mu
		rad[np.isnan(rad)] = 0.
		y = np.sin(np.arccos(mu))
		tan = np.exp(-1*np.power(vel_1d/(zeta*y),2))/y
		tan[np.isnan(tan)] = 0.
		
		mac[ii] = A*(rad + tan)/(np.sqrt(np.pi)*zeta)

	return vel_1d, mac

def gauss_conv(lum,vel,xi,sigma=3):
	'''

	Convolves the rotation profile with a gaussian to take microturbulence
	and the instrumental profile into account. 
	See 
		- Hirano et al. 2011 | doi:10.1088/0004-637X/742/2/69
		- Gray 2005 | doi:10.1017/CBO9781316036570 p. 430.
	
	:params:
		sigma : float - number of sigmas we go out on our x-axis to get the borders of the gaussian
	'''
	sep = (vel[-1]-vel[0])/(len(vel)-1) #seperation of velocity vector
	## x-axis for the gaussian. The steps are the same as vel, but the borders go further out.
	x = np.arange(-sigma*xi,sigma*xi+sep,sep)

	
	## Gaussian function for microturbuelence
	## make Gaussian with new velocity vector as x-axis
	gau = np.exp(-1*np.power(x/xi,2))/(xi*np.sqrt(np.pi))#gauss(x,xi) 
	gau /= np.add.reduce(gau)

	length = len(lum[0])+len(gau)-1
	gau_arr = np.empty([len(lum),len(gau)])
	gau_arr[:] = gau
	velocity, line_profile = np.empty([len(lum),length]), np.empty([len(lum),length])
	line = np.add.reduce(lum,axis=2)
	velocity[:] = np.linspace(-length*sep/2.,length*sep/2.,num=length,endpoint=False)
	line_profile[:] = ss.fftconvolve(line[:],gau_arr[:],'full',axes=1)

	return velocity, line_profile

def convolve(vel,ring_LD,mu_mean,xi,zeta,sigma=3.):
	'''Convolves limb-darkened rings.

	Function that convolves limb-darkened rings with gaussian to get microturbulence, 
	and then convolve this with the macroturbulence.
	
	'''
	n_LD = len(ring_LD)
	## 1D-velocity (same for each ring)
	vel_1d = vel[:,0]
	sep = (vel_1d[-1]-vel_1d[0])/(len(vel_1d)-1)
		
	## x-axis for the gaussian. The steps are the same as vel, but the borders go further out.
	x = np.arange(-sigma*xi,sigma*xi+sep,sep)
	
	## Gaussian function for microturbuelence
	## make Gaussian with new velocity vector as x-axis
	gau = np.exp(-1*np.power(x/xi,2))/(xi*np.sqrt(np.pi))#gauss(x,xi) 
	gau /= np.add.reduce(gau)

	length = len(ring_LD[0])+len(gau)-1
	gau_arr = np.empty([len(ring_LD),len(gau)])
	gau_arr[:] = gau

	line = np.add.reduce(ring_LD,axis=2)
	micro_profile = np.empty([len(ring_LD),length])
	micro_profile[:] = ss.fftconvolve(line[:],gau_arr[:],'full',axes=1)

	## Calculate macroturbulence of rings
	A = 0.5 #area of curve covered by radial and tangential
	mac = np.zeros(shape=(len(mu_mean),len(vel_1d)))	

	for ii, mu in enumerate(mu_mean):
		rad = np.exp(-1*np.power(vel_1d/(zeta*mu),2))/mu
		rad[np.isnan(rad)] = 0.
		y = np.sin(np.arccos(mu))
		tan = np.exp(-1*np.power(vel_1d/(zeta*y),2))/y
		tan[np.isnan(tan)] = 0.
		mac[ii] = A*(rad + tan)/(np.sqrt(np.pi)*zeta)

	
	## Convolve each ring of LD+microturbulence and macroturbulence
	lc_len = mac[0].shape[0] + micro_profile[0].shape[0] - 1 
	xn = np.linspace(-lc_len*sep/2.,lc_len*sep/2.,num=lc_len,endpoint=False)

	line_conv = np.zeros(shape=(len(ring_LD),lc_len))
	line_conv[:,:] = ss.fftconvolve(micro_profile,mac,'full',axes=1)

	return xn, line_conv

# =============================================================================
# Stellar surface/atmosphere
# =============================================================================

def limb_darkening(gridini,mu,cs=[],LD_law='quad'):
	'''
	Calculate the limb darkening at each position of the stellar grid.

	:params:
		gridini : array - grid of positions on the stellar disk.
		mu      : array - normalized radial coordinate.
		cs      : list of floats - limb darkening coefficients.
		LD_law  : string - limb darkening law.
	:return:
		lum     : array - limb-darkened surface.
	'''

	if LD_law == 'quad':
		c1, c2 = cs
		law = 1 - c1*(1 - mu) - c2*np.power(1 - mu,2)
	elif LD_law == 'nonlinear':
		c1, c2, c3, c4 = cs
		law = 1. - c1*(1. - np.sqrt(mu)) - c2*(1. - mu) - c3*(1. - np.power(mu,3/2)) - c4*(1. - np.power(mu,2))
	elif LD_law == 'uni':
		law = 1

	lum = gridini*law
	return lum 

def absline_star(gridini,vel,ring_grid,
	mu,mu_mean,vsini,xi,zeta,
	cs=[0.3,0.2],LD_law='quad'):
	
	rot_profile = vel*vsini
	if LD_law == 'quad':
		c1, c2 = cs
		law = 1 - c1*(1 - mu) - c2*np.power(1 - mu,2)
	elif LD_law == 'nonlinear':
		c1, c2, c3, c4 = cs
		law = 1. - c1*(1. - np.sqrt(mu)) - c2*(1. - mu) - c3*(1. - np.power(mu,3/2)) - c4*(1. - np.power(mu,2))
	elif LD_law == 'uni':
		law = 1

	lum = gridini*law

	#Makes the limb-darkened (LD) stellar grid into rings
	ring_LD = np.zeros(len(ring_grid))
	ring_LD = ring_grid*lum

	vel_1d_ext, line_conv = convolve(rot_profile,ring_LD,mu_mean,xi,zeta)

	return vel_1d_ext, line_conv, ring_LD


def absline(gridini,vel,ring_grid,
			mu,mu_mean,mu_grid,
			vsini,xi,zeta,
			cs=[0.3,0.2],LD_law='quad',
			times=np.array([]),
			radius=100,
			Tw=0.,per=10.,
			Rp_Rs=0.1,a_Rs=20.,inc=90.,
			ecc=0.,w=90.,lam=0.
			):
	'''
	Calculates the line-shape as a function of velocity (km/sec) 
	for a transitting star-planet system with the given parameters 
	at a given time (including LD, micro- and macroturbulence).

	'''


	rot_profile = vel*vsini
	if LD_law == 'quad':
		c1, c2 = cs
		law = 1 - c1*(1 - mu) - c2*np.power(1 - mu,2)
	elif LD_law == 'nonlinear':
		c1, c2, c3, c4 = cs
		law = 1. - c1*(1. - np.sqrt(mu)) - c2*(1. - mu) - c3*(1. - np.power(mu,3/2)) - c4*(1. - np.power(mu,2))
	elif LD_law == 'uni':
		law = 1

	lum = gridini*law

	#Makes the limb-darkened (LD) stellar grid into rings
	ring_LD = np.zeros(len(ring_grid))
	ring_LD = ring_grid*lum
	vel_1d_ext, line_conv = convolve(rot_profile,ring_LD,mu_mean,xi,zeta)
	
	nn = len(times)

	vel_1d_ext_pl, line_conv_pl, planet_rings, index_error = transit_ring(vel,vel_1d_ext,ring_LD,mu_grid,mu_mean,lum,
											vsini,xi,zeta,Rp_Rs,radius,
											times,Tw,ecc,per,w,a_Rs,inc,lam)


	line_conv_total = np.empty(shape=(nn,len(vel_1d_ext)))
	for ii in range(nn): 
		line_conv_total[ii,:] = np.add.reduce(line_conv-line_conv_pl[ii,:,:],axis=0)

	return vel_1d_ext, line_conv, line_conv_total, planet_rings, lum, index_error

# =============================================================================
# Shadow plot
# =============================================================================

def create_shadow(phase,vel,shadow,exp_phase,per,
	savefig=False,fname='shadow',zmin=None,zmax=None,
	xlims=[],contour=False,vsini=None,cmap='bone_r',
	ax=None,colorbar=True,cbar_pos='right',latex=True,
	font = 12,tickfontsize=10):
	'''
	Creates the shadow plot.
	
	:params:
		vel       : array - velocity vector.
		phase     : array - phase stamps.
		shadow    : array - shahow vector (out-of-transit absline minus all absline).
		exp_phase : array - exposure time in phase units.
		per       : array - orbital period (days).
	'''
	if not fname.lower().endswith(('.png','.pdf')): 
		ext = '.pdf'
		fname = fname.split('.')[0] + ext

	import matplotlib.pyplot as plt
	
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
