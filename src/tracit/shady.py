#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:27:04 2020

@author: emil

.. todo::
	* Finish documentation
	* In convolve, look at resolution
	* Make it faster? numba?
	* Some arrays seem to be passed on and created multiple times -- redundant?
"""

import numpy as np
import scipy.signal as ss
import sys
#import dynamics
from .dynamics import true_anomaly, xy_pos

# =============================================================================
# Grids
# =============================================================================

def grid_coordinates(Rs,xoff=0.,yoff=0.):
	'''Coordinates of the stellar grid.

	Calculate the coordinates of the stellar grid, from -Rs to +Rs.

	:param Rs: Stellar radius in number of pixels.
	:type Rs: int

	:param xoff: Offset in x-direction. Default is 0.
	:type xoff: float, optional

	:param yoff: Offset in y-direction. Default is 0.
	:type yoff: float, optional

	:return: grid coordinates, indices of coordinates, distance from limb
	:rtype: (array, array, array)

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

	:param xoff: Offset in x-direction. Default 0.
	:type xoff: float, optional

	:param yoff: Offset in y-direction. Default 0.
	:type yoff: float, optional

	:return: initial stellar grid, velocity grid, normalized radial coordinate.
	:rtype: (array, array, array)

	'''
	coord, cidxs, rr = grid_coordinates(Rs,xoff=xoff,yoff=yoff)

	## Empty, quadratic array with dimensions diameter*diameter
	start_grid = np.zeros((2*Rs+1,2*Rs+1))
	vel = start_grid.copy()
	mu = start_grid.copy()
	for ii in range(len(cidxs)):
		if rr[ii] <= Rs:
			start_grid[cidxs[ii]] = 1
			mu[cidxs[ii]] = np.sqrt(1-(rr[ii]/float(Rs))**2) # used for limb darkening. Same as cos(inclination)
		vel[cidxs[ii]] = coord[ii][0]/float(Rs) #velocity field. [0]-axis (rows) are cst vel, while [1]-axis (columns) are the equator

	return start_grid, vel, mu

def grid_ring(Rs,thick,xoff=0.,yoff=0.):
	'''Initial grid of star in rings of :math:`\mu`.

	Initial grid of star in rings of aprrox same :math:`\mu = \cos(\\theta)`,
	where :math:`\\theta` is the angle between a line through the center of the star and the limb.

	Useful for macroturbulence calculations.

	:param Rs: Stellar radius in number of pixels.
	:type Rs: int

	:param thick: Thickness of rings.
	:type thick: int

	:param xoff: Potential offset in x-direction. Default 0.
	:type xoff: float, optional

	:param yoff: Potential offset in y-direction. Default 0.
	:type yoff: float, optional

	:return: pixels within stellar disk, velocity grid, radial :math:`\mu` values, approx :math:`\mu` in each ring
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

	mu_grid = np.asarray([start_grid.copy() for i in range(nr)])
	ring_grid = mu_grid.copy()#np.asarray([start_grid.copy() for i in range(nr)]) #startgrid for ring
	for ii in range(len(cidxs)):
		for jj in range(nr):
			if rings_in[jj] < rr[ii] <= rings[jj]:
				ring_grid[jj][cidxs[ii]] = 1
				mu_grid[jj][cidxs[ii]] = np.sqrt(1-(rr[ii]/float(Rs))**2) ## Used for limb darkening and macro. Same as cos(theta)
		vel[cidxs[ii]] = coord[ii][0]/float(Rs) ## Velocity field. [0]-axis (rows) are cst vel, while [1]-axis (columns) are the equator
	mu_mean = np.zeros(shape=(1,nr))[0]

	## Calculate the approx mu in each ring to be used when calculating the macroturbulence
	for kk in range(nr):
		mu_mean[kk] = np.mean(mu_grid[kk][np.where(mu_grid[kk])])

	return ring_grid, vel, mu_grid, mu_mean


def spot(time,radius,ring_LD,lum,
		theta,phi,t_ref,
		Pspot,Rspot,Tspot,Teff):
	'''Spot model.

	'''
	#Teff = 5675

	frac = (Tspot/Teff)**4

	rp = int(round(Rspot*radius))
	spot_grid, spot_vel, mu = grid(rp)

	rot_phase = 2*np.pi*((time - t_ref)/Pspot)

	xx = np.sin(theta+rot_phase)*np.sin(phi)
	yy = np.cos(phi*np.ones(len(time)))
	zz = np.cos(theta+rot_phase)*np.sin(phi)

	xnorm, ynorm = xx*radius, yy*radius
	x_off, y_off = np.rint(xnorm), np.rint(ynorm)

	x_sp, y_sp = abs(x_off)-rp, abs(y_off)-rp

	rescape_spot = np.reshape(spot_grid,np.size(spot_grid))
	nn = len(xx)
	line_conv = np.empty(shape=(nn,lum.shape[0],lum.shape[1]))

	for kk in range(nn):
		## Only do calculation if spot is "touching" stellar disk
		it_lum = lum.copy()
		#if (x_sp[kk] < radius) and (y_sp[kk] < radius):
		if zz[kk] > 0.:#(x_sp[kk] < radius) and (y_sp[kk] < radius):

		#x_pos = int(x_off + radius)
		#y_pos = int(y_off + radius)
			x_pos = int(x_off[kk] + radius)
			y_pos = int(y_off[kk] + radius)

			spot_coord, spot_coord_arr, spot_coord_idx = grid_coordinates(rp,x_pos,y_pos)
			spot_zip = [(spot_coord[ii,0],spot_coord[ii,1]) for ii in range(spot_coord.shape[0])]
			coord_spot = [i for (i,v) in zip(spot_zip,rescape_spot) if v==1. and i[0] >= 0 and i[1] >= 0 and i[0] < ring_LD[0].shape[0] and i[1] < ring_LD[0].shape[0]]

			it_lum[np.asarray(coord_spot)[:,0],np.asarray(coord_spot)[:,1]] = it_lum[np.asarray(coord_spot)[:,0],np.asarray(coord_spot)[:,1]]*frac
		line_conv[kk,:,:] = it_lum

	return it_lum, line_conv


def transit_ring(vel,vel_ext,ring_LD,mu_grid,mu_mean,lum,
				vsini,xi,zeta,Rp_Rs,radius,vels,
				time,Tw,ecc,per,w,a_Rs,inc,lam):
	'''Planet signal.

	Function that calculates the planet signal in each ring.
	This includes the effects of limb-darkening, as well as micro- and macroturbulence.

	:param mu_grid: Grid divided into rings of appproximately same :math:`\mu`.
	:type mu_grid: array

	:param mu_mean: Approximate :math:`\mu` in each ring.
	:type mu_mean: array

	:param lum: Limb-darkened grid.
	:type lum: array

	:param vsini: Projected stellar rotation in km/s.
	:type vsini: float

	:param xi: Micro-turbulence in km/s. Defaults to 3.
	:type xi: float, optional

	:param zeta: Macro-turbulence in km/s. Defaults to 1.0.
	:type zeta: float, optional

	:param Rp_Rs: Planet-to-star radius ratio.
	:type Rp_Rs: float

	:param radius: Number of pixels from center to limb, i.e, 100 yields a 200 by 200 stellar grid.
	:type radius: int

	:param time: Times of observations.
	:type time: array

	:param a_Rs: Semi-major axis in stellar radii.
	:type a_Rs: float

	:param inc: Inclination in degrees.
	:type inc: float

	:param lam: Projected obliquity in degrees.
	:type lam: float

	:param ecc: Eccentricity.
	:type ecc: float

	:param per: Orbital period.
	:type per: float

	:param w: Argument of periastron in degress.
	:type w: float

	:param Tw: Time of periastron passage in days.
	:type Tw: float


	:return: velocity grid, convoluted stellar line, light curve, error (for MCMC)
	:rtype: array, array, array, bool

	'''
	rot_profile = vel*vsini
	nn = len(time)

	rp = int(round(Rp_Rs*radius))
	pl_grid, pl_vel, mu = grid(rp)
	#cos_f, sin_f = true_anomaly(time,Tw,ecc,per,w)
	cos_f, sin_f = true_anomaly(time,Tw,ecc,per)
	xx, yy = xy_pos(cos_f,sin_f,ecc,w,a_Rs,inc,lam)
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
			coord_pl = [i for (i,v) in zip(pl_zip,rescape_pl) if v==1. and i[0] >= 0 and i[1] >= 0 and i[0] < ring_LD[0].shape[0] and i[1] < ring_LD[0].shape[0]]

			try:
				ring_planet[:,np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]] = ring_LD[:,np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]]
				it_lum[np.asarray(coord_pl)[:,0],np.asarray(coord_pl)[:,1]] = 0
			## Makes sure that when a planet is about to disappear
			## it does not appear on the other side.
			except IndexError:
				## mu values at planet position in mu_grid
				mu_pl = np.asarray([sum_mu_grid[i] for (i,v) in zip(pl_zip,rescape_pl) if v==1. and i[0]<mu_size and i[1]<mu_size])
				for ii in range(len(ring_LD)):
					for jj in range(len(mu_pl)):
						try:
							ring_planet[ii][coord_pl[jj]] = ring_LD[ii][coord_pl[jj]]
						except IndexError:
							index_error = True
					it_lum[ring_planet[ii].astype(bool)] = 0

			xn[kk,:], line_conv[kk,:,:] = convolve(rot_profile,ring_planet,mu_mean,xi,zeta,vels)
			ring_planet[:,:,:] = 0


		else:
			xn[kk,:] = vel_ext
			line_conv[kk,:,:] = np.zeros(shape=(len(ring_LD),len(vel_ext)))

		dark[kk] = np.sum(it_lum)
	return xn, line_conv, dark, index_error#planet_rings


# =============================================================================
# Convolution
# =============================================================================

# def macro(vel,mu_mean,zeta):
# 	'''Macroturbulence at given distance from center.

# 	Function that calculates the macroturbulence for given :math:`\zeta` and :math:`\mu` using the radial-tangential profile.


# 	:param vel: Velocity grid.
# 	:type vel: array

# 	:param mu_mean: Approximate :math:`\mu` in each ring.
# 	:type mu_mean: array

# 	:param zeta: Macro-turbulence in km/s.
# 	:type zeta: float

# 	:return: velocity grid as 1D array, macro-turbulence
# 	:rtype: (array, array)

# 	'''

# 	A = 0.5 #area of curve covered by radial and tangential
# 	vel_1d = vel[0]

# 	mac = np.zeros(shape=(len(mu_mean),len(vel_1d)))


# 	for ii, mu in enumerate(mu_mean):
# 		rad = np.exp(-1*np.power(vel_1d/(zeta*mu),2))/mu
# 		rad[np.isnan(rad)] = 0.
# 		y = np.sin(np.arccos(mu))
# 		tan = np.exp(-1*np.power(vel_1d/(zeta*y),2))/y
# 		tan[np.isnan(tan)] = 0.

# 		mac[ii] = A*(rad + tan)/(np.sqrt(np.pi)*zeta)

# 	return vel_1d, mac

# def gauss_conv(lum,vel,xi,sigma=3):
# 	'''Convolve the rotation profile.

# 	Function that convolves the rotation profile with a gaussian to take microturbulence
# 	and the instrumental profile into account. Following the approaches in [1] and [2].

# 	:param lum: Limb-darkened grid.
# 	:type lum: array

# 	:param sigma: Number of sigmas we go out on our x-axis to get the borders of the Gaussian.
# 	:type sigma: float

# 	:return: velocity grid, line profile after convolution
# 	:rtype: (array, array)

# 	References
# 	----------
# 		[1] `Hirano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract>`_

# 		[2] `Gray (2005), p. 430. <https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G/abstract>`_

# 	'''
# 	sep = (vel[-1]-vel[0])/(len(vel)-1) #seperation of velocity vector
# 	## x-axis for the gaussian. The steps are the same as vel, but the borders go further out.
# 	x = np.arange(-sigma*xi,sigma*xi+sep,sep)


# 	## Gaussian function for microturbuelence
# 	## make Gaussian with new velocity vector as x-axis
# 	gau = np.exp(-1*np.power(x/xi,2))/(xi*np.sqrt(np.pi))#gauss(x,xi)
# 	gau /= np.add.reduce(gau)

# 	length = len(lum[0])+len(gau)-1
# 	gau_arr = np.empty([len(lum),len(gau)])
# 	gau_arr[:] = gau
# 	velocity, line_profile = np.empty([len(lum),length]), np.empty([len(lum),length])
# 	line = np.add.reduce(lum,axis=2)
# 	velocity[:] = np.linspace(-length*sep/2.,length*sep/2.,num=length,endpoint=False)
# 	line_profile[:] = ss.fftconvolve(line[:],gau_arr[:],'full',axes=1)

# 	return velocity, line_profile

def convolve(vel,ring_LD,mu_mean,xi,zeta,vels,sigma=3.):
	'''Convolves rotational profile.

	Function that convolves the rotation profile with a gaussian to take microturbulence for a given :math:`\\xi` into account.
	This is then convolved with the macroturbulence for a given :math:`\zeta` at a given :math:`\mu` using the radial-tangential profile.
	Following the approaches in :cite:t:`Hirano2011` and :cite:t:`Gray2005`.

	:param vel: Velocity grid.
	:type vel: array

	:param ring_LD: Limb-darkened stellar grid in rings.
	:type array:

	:param mu_mean: Approximate :math:`\mu` in each ring.
	:type mu_mean: array

	:param xi: Micro-turbulence in km/s.
	:type xi: float

	:param zeta: Macro-turbulence in km/s.
	:type zeta: float

	:param sigma: Number of sigmas we go out on our x-axis to get the borders of the Gaussian. Default 3.
	:type sigma: float, optional

	:param vels: :math:`x`-axis for the Gaussian. Set by the 'Velocity_range' and 'Velocity_resoluation' in :py:func:`tracit.structure.dat_struct`.
	:type vels: array

	.. note::
		:math:`\\xi` here also includes the instrumental broadening, :math:`\sigma_\mathrm{PSF}`. See :py:func:`tracit.business.ls_model`.


	'''
	n_LD = len(ring_LD)
	## 1D-velocity (same for each ring)
	vel_1d = vel[:,0]
	sep = (vel_1d[-1]-vel_1d[0])/(len(vel_1d)-1)

	# ## x-axis for the gaussian. The steps are the same as vel, but the borders go further out.
	# x = np.arange(-sigma*xi,sigma*xi+sep,sep)

	## Gaussian function for microturbuelence
	## make Gaussian with new velocity vector as x-axis
	gau = np.exp(-1*np.power(vels/xi,2))/(xi*np.sqrt(np.pi))#gauss(x,xi)
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

# def limb_darkening(gridini,mu,cs=[],LD_law='quad'):
# 	'''Limb darkening in stellar grid.

# 	Function that calculates the limb darkening at each position of the stellar grid.

# 	:param gridini: Grid of positions on the stellar disk.
# 	:type gridini: array

# 	:param mu: Normalized radial coordinates. See :py:func:`grid_ring` for the definition of :math:`\mu`.
# 	:type mu: array

# 	:param cs: Limb darkening coefficients. Default ``[]``.
# 	:type cs: list, optional

# 	:param	LD_law: limb darkening law. Default ``'quad'``. See :py:class:`dynamics.StellarParams`.
# 	:type LD_law: str, optional

# 	:return: Limb-darkened surface.
# 	:rtype: array
# 	'''

# 	if LD_law == 'quad':
# 		c1, c2 = cs
# 		law = 1 - c1*(1 - mu) - c2*np.power(1 - mu,2)
# 	elif LD_law == 'nonlinear':
# 		c1, c2, c3, c4 = cs
# 		law = 1. - c1*(1. - np.sqrt(mu)) - c2*(1. - mu) - c3*(1. - np.power(mu,3/2)) - c4*(1. - np.power(mu,2))
# 	elif LD_law == 'uni':
# 		law = 1

# 	lum = gridini*law
# 	return lum

def absline_star(gridini,vel,ring_grid,
	mu,mu_mean,vsini,xi,zeta,vels,
	cs=[0.3,0.2],LD_law='quad'):
	'''The shape of the absorption line.

	Function that calculates the line shape as a function of velocity (km/s).

	:param gridini: Grid of positions on the stellar disk.
	:type gridini: array

	:param mu: Normalized radial coordinate.
	:type mu: array

	:param mu_mean: Approximate :math:`\mu` in each ring.
	:type mu_mean: array

	:param cs: Limb darkening coefficients. Default ``[]``.
	:type cs: list, optional

	:param	LD_law: limb darkening law. Default ``'quad'``. See :py:class:`dynamics.StellarParams`.
	:type LD_law: str, optional



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
	# ring_LD = np.zeros(len(ring_grid))
	ring_LD = ring_grid*lum

	vel_1d_ext, line_conv = convolve(rot_profile,ring_LD,mu_mean,xi,zeta,vels)

	return vel_1d_ext, line_conv, ring_LD


def absline(gridini,vel,ring_grid,
			mu,mu_mean,mu_grid,
			vsini,xi,zeta,vels,
			cs=[0.3,0.2],LD_law='quad',
			times=np.array([]),
			radius=100,
			Tw=0.,per=10.,
			Rp_Rs=0.1,a_Rs=20.,inc=90.,
			ecc=0.,w=90.,lam=0.
			):
	'''The shape of the absorption line during transit.

	Function that calculates the line shape as a function of velocity (km/s) for a transitting star-planet system.
	Effects of limb-darkening, as well as micro- and macroturbulence are included.

	:param mu_mean: Approximate :math:`\mu` in each ring.
	:type mu_mean: array

	:param gridini: The initial stellar grid.
	:type gridini: array

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
	vel_1d_ext, line_conv = convolve(rot_profile,ring_LD,mu_mean,xi,zeta,vels)

	nn = len(times)

	vel_1d_ext_pl, line_conv_pl, planet_rings, index_error = transit_ring(vel,vel_1d_ext,ring_LD,mu_grid,mu_mean,lum,
											vsini,xi,zeta,Rp_Rs,radius,vels,
											times,Tw,ecc,per,w,a_Rs,inc,lam)


	line_conv_total = np.empty(shape=(nn,len(vel_1d_ext)))
	for ii in range(nn):
		line_conv_total[ii,:] = np.add.reduce(line_conv-line_conv_pl[ii,:,:],axis=0)

	return vel_1d_ext, line_conv, line_conv_total, planet_rings, lum, index_error


# if __name__=='__main__':
# 	print('Go!')