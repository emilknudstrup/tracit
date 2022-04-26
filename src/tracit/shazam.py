#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import astropy.io.fits as pyfits
import numpy as np
import scipy.stats as sct
from astropy import constants as const
from astropy.time import Time
from scipy.signal import fftconvolve
import lmfit

from scipy.optimize import curve_fit


# =============================================================================
# Templates
# =============================================================================

def read_phoenix(filename, wl_min=3600,wl_max=8000):
  '''Read Phoenix stellar template.


  '''
  fhdu = pyfits.open(filename)
  flux = fhdu[0].data
  fhdu.close()

  whdu = pyfits.open(os.path.dirname(filename)+'/'+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
  wave = whdu[0].data
  whdu.close()

  #from Ciddor (1996) to go from vacuum to air wavelength
  sig2 = (1e4/wave)**2.0
  f = 1.0 + 0.05792105/(238.0185-sig2) + 0.00167917/(57.362-sig2)
  wave /= f
  
  keep = (wave > wl_min) & (wave < wl_max)
  wave, flux = wave[keep], flux[keep]
  flux /= np.amax(flux)

  #flux /= np.median()
  return wave, flux

def read_kurucz(filename):
  '''Read Kurucz/ATLAS9 stellar template.

  Extract template wavelength and flux from :cite:t:`Castelli2003`.
  
  Available `here <http://130.79.128.5/ftp/more/splib120/>`_.
  
  :param filename: Path to template.
  :type filename: str

  :return: template wavelength, template flux
  :rtype: array, array

  '''
  temp = pyfits.open(filename)
  th = temp[0].header
  flux = temp[0].data[0]
  temp.close()

  # create wavelengths
  wls = th['CRVAL1']
  wld = th['CDELT1']
  wave = np.arange(len(flux))*wld + wls
  return wave, flux 


# =============================================================================
# SONG specifics
# =============================================================================

def SONG_request(filename):
  '''Extract SONG data.

  Extract data and header from SONG file.
  
  :param filename: Name of .fits file.
  :type filename: str
  
  :return: observed spectrum, number of orders for spectrum, epoch for spectrum in BJD, barycentric velocity correction for epoch in km/s, name of star, date of observations in UTC, exposure time in seconds
  :rtype: array, int, float, float, str, str, float
  '''
  fits = pyfits.open(filename)
  hdr = fits[0].header
  data = fits[0].data
  fits.close()

  star = hdr['OBJECT']
  date = hdr['DATE-OBS']
  exp =  hdr['EXPTIME']

  bjd = hdr['BJD-MID'] + 2400000
  bvc = hdr['BVC']
  orders = hdr['NAXIS2']

  return data, orders, bjd, bvc, star, date, exp

def SING(data,order=30):
  '''Extract SONG spectrum.

  Extract spectrum from SONG at given order.
  
  :params data: Observed spectrum.
  :type data: array
  :param order: Extract spectrum at order. Default 30.
  :type order: int, optional
  
  :return: observed wavelength, observed raw flux, blaze function
  :rtype: array, array, array
  '''
  
  wl, fl = data[3,order,:], data[1,order,:] # wavelength, flux
  bl = data[2,order,:] # blaze
  return wl, fl, bl


# =============================================================================
# FIES specifics
# =============================================================================

def FIES_caliber(filename,return_hdr=False,check_ThAr=True):
  '''Extract FIES data (FIEStool).

  Reads a wavelength calibrated spectrum in IRAF (FIEStool) format.
  Returns a record array with wavelength and flux order by order as a two column 
  array in data['order_i'].
  
  Adapted and extended from functions written by R. Cardenes and J. Jessen-Hansen.
  
  :param filename: Name of .fits file.
  :type filename: str
  
  :return: observed spectrum, wavelength and flux order by order, number of orders for spectrum, name of object, date of observations in UTC, exposure time in seconds
  :rtype: array, int, str, str, float 
  
  '''
  try:
    hdr = pyfits.getheader(filename)
    star = hdr['OBJECT']
    if star == 'ThAr' and check_ThAr:
      raise Exception('\n###\nThis looks like a {} frame\nPlease provide a wavelength calibrated file\n###\n'.format(star))

    date = hdr['DATE-OBS']
    date_mid = hdr['DATE-AVG']
    bjd = Time(date_mid, format='isot', scale='utc').jd
    exp = hdr['EXPTIME']
    vhelio = hdr['VHELIO']
  except Exception as e:
    print('Problems extracting headers from {}: {}'.format(filename, e))
    print('Is filename the full path?')

  # Figure out which header cards we want
  cards = [x for x in sorted(hdr.keys()) if x.startswith('WAT2_')]
  # Extract the text from the cards
  # We're padding each card to 68 characters, because PyFITS won't
  # keep the blank characters at the right of the text, and they're
  # important in our case
  text_from_cards = ''.join([hdr[x].ljust(68) for x in cards])
  data = text_from_cards.split('"')

  #Extract the wavelength info (zeropoint and wavelength increment for each order) to arrays
  info = [x for x in data if (x.strip() != '') and ("spec" not in x)]
  zpt = np.array([x.split(' ')[3] for x in info]).astype(np.float64)
  wstep = np.array([x.split(' ')[4] for x in info]).astype(np.float64)
  npix = np.array([x.split(' ')[5] for x in info]).astype(np.float64)
  orders = np.array([x.split(' ')[0] for x in info])
  no_orders = len(orders)

  #Extract the flux
  fts  = pyfits.open(filename)
  data = fts[0].data
  fts.close()
  wave = data.copy()

  #Create record array names to store the flux in each order
  col_names = [ 'order_'+order for order in orders]

  #Create wavelength arrays
  for i,col_name in enumerate(col_names):
    wave[0,i] = zpt[i] + np.arange(npix[i]) * wstep[i]

  #Save wavelength and flux order by order as a two column array in data['order_i']
  data = np.rec.fromarrays([np.column_stack([wave[0,i],data[0,i]]) for i in range(len(col_names))],
    names = list(col_names))
  if return_hdr:
	  return data, no_orders, bjd, vhelio, star, date, exp, hdr
  return data, no_orders, bjd, vhelio, star, date, exp

def FIES_gandolfi(filename,return_hdr=False,check_ThAr=True):
  '''Extract FIES data (D. Gandolfi).

  Same functionality as :py:func:`FIES_caliber`, but for FIES data reduced by D. Gandolfi.
  
  :param filename: Name of .fits file.
  :type filename: str
  
  :return: observed spectrum, wavelength and flux order by order, number of orders for spectrum, name of object, date of observations in UTC, exposure time in seconds
  :rtype: array, int, str, str, float 
  
  '''
  try:
    hdr = pyfits.getheader(filename)
    star = hdr['OBJECT']
    if star == 'ThAr' and check_ThAr:
      raise Exception('\n###\nThis looks like a {} frame\nPlease provide a wavelength calibrated file\n###\n'.format(star))

    date = hdr['DATE-OBS']
    date_mid = hdr['DATE-AVG']
    bjd = Time(date_mid, format='isot', scale='utc').jd
    exp = hdr['EXPTIME']
    #vhelio = hdr['VHELIO']

  except Exception as e:
    print('Problems extracting headers from {}: {}'.format(filename, e))
    print('Is filename the full path?')

  # Figure out which header cards we want
  cards = [x for x in sorted(hdr.keys()) if x.startswith('WAT2_')]
  # Extract the text from the cards
  # We're padding each card to 68 characters, because PyFITS won't
  # keep the blank characters at the right of the text, and they're
  # important in our case
  text_from_cards = ''.join([hdr[x].ljust(68) for x in cards])
  data = text_from_cards.split('"')

  #Extract the wavelength info (zeropoint and wavelength increment for each order) to arrays
  info = [x for x in data if (x.strip() != '') and ("spec" not in x)]
  zpt = np.array([x.split(' ')[3] for x in info]).astype(np.float64)
  wstep = np.array([x.split(' ')[4] for x in info]).astype(np.float64)
  npix = np.array([x.split(' ')[5] for x in info]).astype(np.float64)
  orders = np.array([x.split(' ')[0] for x in info])
  no_orders = len(orders)

  #Extract the flux
  fts  = pyfits.open(filename)
  data = fts[0].data
  fts.close()
  wave = data.copy()

  #Create record array names to store the flux in each order
  col_names = [ 'order_'+order for order in orders]

  #Create wavelength arrays
  for i,col_name in enumerate(col_names):
    wave[i] = zpt[i] + np.arange(npix[i]) * wstep[i]

  #Save wavelength and flux order by order as a two column array in data['order_i']
  data = np.rec.fromarrays([np.column_stack([wave[i],data[i]]) for i in range(len(col_names))],
    names = list(col_names))
  if return_hdr:
    #return data, no_orders, bjd, vhelio, star, date, exp, hdr
    return data, no_orders, bjd, star, date, exp, hdr
  return data, no_orders, bjd, star, date, exp
  #return data, no_orders, bjd, vhelio, star, date, exp


def getFIES(data,order=40):
  '''Extract FIES spectrum.

  Extract calibrated spectrum from FIES at given order.
    
  :params data: Observed spectrum.
  :type data: array
  :param order: Extract spectrum at order. Default 30.
  :type order: int, optional
  
  :return: observed wavelength, observed raw flux
  :rtype: array, array

  '''
  arr = data['order_{:d}'.format(order)]
  wl, fl = arr[:,0], arr[:,1]
  return wl, fl

# =============================================================================
# Preparation/normalization
# =============================================================================

def normalize(wl,fl,bl=np.array([]),poly=1,gauss=True,lower=0.5,upper=1.5):
  '''Normalize spectrum.

  Nomalization of observed spectrum.
  
  :param wl: Observed wavelength 
  :type wl: array 
  :param fl: Observed raw flux
  :type fl: array
  :param bl: Blaze function. Default ``numpy.array([])``. If empty, no correction for the blaze function.
  :type bl: array, optional
  :param poly: Degree of polynomial fit to normalize. Default 1. Set to ``None`` for no polynomial fit.
  :type poly: int, optional
  :param gauss: Only fit the polynomial to flux within, (mu + upper*sigma) > fl > (mu - lower*sigma). Default ``True``.
  :type gauss: bool, optional
  :param lower: Lower sigma limit to include in poly fit. Default 0.5.
  :type lower: float, optional
  :param upper: Upper sigma limit to include in poly fit. Default 1.5.
  :type upper: float, optional

  :return: observed wavelength, observed normalized flux
  :rtype: array, array
  '''

  if len(bl) > 0:
    fl = fl/bl # normalize w/ blaze function
  
  keep = np.isfinite(fl)
  wl, fl = wl[keep], fl[keep] # exclude nans
  
  if poly is not None:
    if gauss:
      mu, sig = sct.norm.fit(fl)
      mid  = (fl < (mu + upper*sig)) & (fl > (mu - lower*sig))
      pars = np.polyfit(wl[mid],fl[mid],poly)
    else:
      pars = np.polyfit(wl,fl,poly)
    fit = np.poly1d(pars)
    nfl = fl/fit(wl)
  else:
    nfl = fl/np.median(fl)

  return wl, nfl

def crm(wl,nfl,iters=1,q=[99.0,99.9,99.99]):
  '''Cosmic ray mitigation.
  
  Excludes flux over qth percentile.
  
  :param wl: Observed wavelength.
  :type wl: array 
  :param nfl: Observed normalized flux.
  :type: array 
  :param iters: Iterations of removing upper q[iter] percentile. Default 1.
  :type iters: int, optional
  :param q: Percentiles. Default ``[99.0,99.9,99.99]``.
  :type q: list, optional
  
  :return: observed wavelength, observed normalized flux
  :rtype: array, array
  '''
  assert iters <= len(q), 'Error: More iterations than specified percentiles.'

  for ii in range(iters):
    cut = np.percentile(nfl,q[ii]) # find upper percentile
    cosmic = nfl > cut # cosmic ray mitigation
    wl, nfl = wl[~cosmic], nfl[~cosmic]

  return wl, nfl
        
def resample(wl,nfl,twl,tfl,dv=1.0,edge=0.0):
  '''Resample spectrum.

  Resample wavelength and interpolate flux and template flux.
  Flips flux, i.e., 1-flux.
  
  :param wl: Observed wavelength.
  :type wl: array
  :param nfl: Observed normalized flux.
  :type nfl: array
  :param twl: Template wavelength.
  :type twl: array
  :param tfl: Template flux.
  :type tfl: array
  :param dv: RV steps in km/s. Default 1.0.
  :type dv: float, optional
  :param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0.
  :type edge: float, optional
  
  :return: resampled wavelength, resampled and flipped flux, resampled and flipped template flux
  :rtype: array, array, array
  '''

  wl1, wl2 = min(wl) + edge, max(wl) - edge
  nn = np.log(wl2/wl1)/np.log(np.float64(1.0) + dv/(const.c.value/1e3))
  lam = wl1*(np.float64(1.0)  + dv/(const.c.value/1e3))**np.arange(nn,dtype='float64')
  if len(lam)%2 != 0: lam = lam[:-1] # uneven number of elements

  keep = (twl >= lam[0]) & (twl <= lam[-1]) # only use resampled wl interval
  twl, tfl_order = twl[keep], tfl[keep]
  
  flip_fl, flip_tfl = 1-nfl, 1-tfl_order
  rf_fl = np.interp(lam,wl,flip_fl)
  rf_tl = np.interp(lam,twl,flip_tfl)
  return lam, rf_fl, rf_tl

# =============================================================================
# Cross-correlation
# =============================================================================

def getCCF(fl,tfl,rvr=401,ccf_mode='full'):
  '''Cross-correlation function.

  Perform the cross correlation and trim array to only include points over RV range.
  
  :param fl: Flipped and resampled flux.
  :type  fl: array
  :param tfl: Flipped and resampled template flux.
  :type tfl: array
  :param rvr: Range for velocity grid in km/s. Default 401.
  :type rvr: int, optional
  
  :return: velocity grid, CCF
  :rtype: array, array
  '''
  ccf = np.correlate(fl,tfl,mode=ccf_mode)
  ccf = ccf/(np.std(fl) * np.std(tfl) * len(tfl)) # normalize ccf
  rvs = np.arange(len(ccf)) - len(ccf)//2
  
  mid = (len(ccf) - 1) // 2 # midpoint
  lo = mid - rvr//2
  hi = mid + (rvr//2 + 1)
  rvs, ccf = rvs[lo:hi], ccf[lo:hi] # trim array

  ccf = ccf/np.mean(ccf) - 1 # shift 'continuum' to zero
  #cut = np.percentile(ccf,85)
  #ccf = ccf/np.median(ccf[ccf < cut]) - 1 # shift 'continuum' to zero
  return rvs, ccf

def Gauss(x, amp, mu,sig ):
  y = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
  return y

def getRV(vel,ccf,nbins=0,zucker=True,no_peak=50,poly=True,degree=1):
  '''Extract radial velocities.
  
  Get radial velocity from CCF by fitting a Gaussian and collecting the location, respectively.
  Error estimation follows that of :cite:t:`Zucker2003`:
  
  .. math:: 
    \sigma^2 (v) = - \\left [ N  \\frac{C^{\prime \prime}(\hat{s})}{C(\hat{s})} \\frac{C^2(\hat{s})}{1 - C^2(\hat{s})} \\right ]^{-1} \, ,

  where :math:`C(\hat{s})` is the cross-correlation function, and :math:`C^{\prime \prime}(\hat{s})` the second derivative. :math:`N` is the number of bins.
  

  :param vel: Velocity in km/s.
  :type vel: array
  :param ccf: CCF.
  :type ccf: array
  :param zucker: Error estimation using :cite:t:`Zucker2003`. Default ``True``, else covariance.
  :type zucker: bool, optional
  :param ccf: CCF.
  :type ccf: array
  :param nbins: Number of bins.
  :type nbins: int, optional IF zucker=False
  :param no_peak: Range with no CCF peak, i.e, a range that can constitute a baseline. Default 50.
  :type no_peak: float, optional
  :param poly: Do a polynomial fit to set the CCF baseline to zero. Default ``True``.
  :type poly: bool
  :param degree: Degree of polynomial fit. Default 1.
  :type degree: int
  
  :returns: position of Gaussian--radial velocity in km/s, uncertainty of radial velocity in km/s
  :rtype: float, float

  '''
  # 

  ## starting guesses
  idx = np.argmax(ccf)
  amp, mu1 = ccf[idx], vel[idx]# get max value of CCF and location
  
  ## fit for offset in CCFs
  if poly:
    no_peak = (vel - mu1 > no_peak) | (vel - mu1 < -no_peak)
    pars = np.polyfit(vel[no_peak],ccf[no_peak],degree)
    for ii, par in enumerate(pars):
      ccf -= par*vel**(degree-ii)

  gau_par, pcov = curve_fit(Gauss,vel,ccf,p0=[amp,mu1,1.0])
  rv = gau_par[1]
  
  if zucker:
    assert nbins > 0, print('To estimate uncertainties from Zucker (2003) the number bins must be provided.')
    y = ccf
    dx = np.mean(np.diff(vel))
    ## derivatives
    yp = np.gradient(y,dx)
    ypp = np.gradient(yp,dx)
    peak = np.argmax(y)
    y_peak = y[peak]
    ypp_peak = ypp[peak]
    
    sharp = ypp_peak/y_peak
    
    snr = np.power(y_peak,2)/(1 - np.power(y_peak,2))
    
    erv = np.sqrt(np.abs(1/(nbins*sharp*snr)))
  else:
    perr = np.sqrt(np.diag(pcov))
    erv = perr[1]
    #vsini = gau_par[2]
    #esini = perr[2]

  return rv, erv



# =============================================================================
# Broadening function
# =============================================================================

def getBF(fl,tfl,rvr=401,dv=1):
  '''Broadening function.

  Carry out the singular value decomposition (SVD) of the "design  matrix" following the approach in :cite:t:`Rucinski1999`.

  This method creates the "design matrix" by applying a bin-wise shift to the template and uses ``numpy.linalg.svd`` to carry out the decomposition. 
  The design matrix, :math:`\hat{D}`, is written in the form :math:`\hat{D} = \hat{U} \hat{W} \hat{V}^T`. The matrices :math:`\hat{D}`, :math:`\hat{U}`, and :math:`\hat{W}` are stored in homonymous attributes.
  
  :param fl: Flipped and resampled flux.
  :type fl: array
  :param tfl: Flipped and resampled template flux.
  :type tfl: array
  :param rvr: Width (number of elements) of the broadening function. Needs to be odd.
  :type rvr: int
  :param dv: Velocity stepsize in km/s.
  :type dv: int
 
  :returns: velocity in km/s, the broadening function
  :rtype: array, array 

  '''
  bn = rvr/dv
  if bn % 2 != 1: bn += 1
  bn = int(bn) # Width (number of elements) of the broadening function. Must be odd.
  bn_arr = np.arange(-int(bn/2), int(bn/2+1), dtype=float)
  vel = -bn_arr*dv

  nn = len(tfl) - bn + 1
  des = np.matrix(np.zeros(shape=(bn, nn)))
  for ii in range(bn): des[ii,::] = tfl[ii:ii+nn]

  ## Get SVD deconvolution of the design matrix
  ## Note that the `svd` method of numpy returns
  ## v.H instead of v.
  u, w, v = np.linalg.svd(des.T, full_matrices=False)

  wlimit = 0.0
  w1 = 1.0/w
  idx = np.where(w < wlimit)[0]
  w1[idx] = 0.0
  diag_w1 = np.diag(w1)

  vT_diagw_uT = np.dot(v.T,np.dot(diag_w1,u.T))

  ## Calculate broadening function
  bf = np.dot(vT_diagw_uT,np.matrix(fl[int(bn/2):-int(bn/2)]).T)
  bf = np.ravel(bf)

  return vel, bf

def smoothBF(vel,bf,sigma=5.0):
  '''Smooth broadening function.

  Smooth the broadening function with a Gaussian.

  :param vel: Velocity in km/s.
  :type vel: array
  :param bf: The broadening function.
  :type bf: array
  :param sigma: Smoothing factor. Default 5.0.
  :type sigma: float, optional
  
  :returns: Smoothed BF.
  :rtype: array
  
  '''
  nn = len(vel)
  gauss = np.zeros(nn)
  gauss[:] = np.exp(-0.5*np.power(vel/sigma,2))
  total = np.sum(gauss)

  gauss /= total

  bfgs = fftconvolve(bf,gauss,mode='same')

  return bfgs



def rotbf_func(vel,ampl,vrad,vsini,gwidth,const=0.,limbd=0.68):
  '''Rotational profile. 

  The rotational profile obtained by convolving the broadening function with a Gaussian following :cite:t:`Kaluzny2006`.

  :param vel: Velocity in km/s.
  :type vel: array
  :param ampl: Amplitude of BF.
  :type ampl: float
  :param vrad: Radial velocity in km/s, i.e., position of BF.
  :type vrad: float
  :param vsini: Projected rotational velocity in km/s, i.e., width of BF.
  :type vsini: float
  :param const: Offset for BF. Default 0.
  :type const: float, optional
  :param limbd: Value for linear limb-darkening coefficient. Default 0.68.
  :type limbd: float, optional

  :returns: rotational profile
  :rtype: array

  '''

  nn = len(vel)
  #bf = np.zeros(nn)
  bf = np.ones(nn)*const

  a1 = (vel - vrad)/vsini
  idxs = np.where(abs(a1) < 1.0)[0]
  asq = np.sqrt(1.0 - np.power(a1[idxs],2))
  bf[idxs] += ampl*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)
  
  gs = np.zeros(nn)
  cgwidth = np.sqrt(2*np.pi)*gwidth
  gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))/cgwidth

  rotbf = fftconvolve(bf,gs,mode='same')

  return rotbf

def rotbf_res(params,vel,bf,wf):
  '''Residual rotational profile.
  
  Residual function for :py:func:`rotbf_fit`.

  :param params: Parameters. 
  :type params: ``lmfit.Parameters()``
  :param vel: Velocity in km/s.
  :type vel: array
  :param bf: Broadening function.
  :type bf: array
  :param wf: Weights.
  :type wf: array

  :returns: residuals
  :rtype: array

  '''
  ampl  = params['ampl1'].value
  vrad  = params['vrad1'].value
  vsini = params['vsini1'].value
  gwidth = params['gwidth'].value
  const =  params['const'].value
  limbd = params['limbd1'].value

  res = bf - rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd)
  return res*wf

def rotbf_fit(vel,bf,fitsize,res=67000,smooth=5.0,vsini=5.0,print_report=True):
  '''Fit rotational profile.

  :params:
  :param vel: Velocity in km/s.
  :type vel: array
  :param bf: Broadening function.
  :type bf: array
  :param fitsize: Interval to fit within.
  :type fitsize: int
  :param res: Resolution of spectrograph. Default 67000 (FIES).
  :type res: int, optional
  :param vsini: Projected rotational velocity in km/s. Default 5.0.
  :type vsini: float, optional
  :param smooth: Smoothing factor. Default 5.0.
  :type smooth: float, optional
  :param print_report: Print the ``lmfit`` report. Default ``True``
  :type print_report: bool, optional 
    
  :returns: result, the resulting rotational profile, smoothed BF
  :rtype: ``lmfit`` object, array, array

  '''
  bfgs = smoothBF(vel,bf,sigma=smooth)
  c = np.float64(299792.458)
  gwidth = np.sqrt((c/res)**2 + smooth**2)

  peak = np.argmax(bfgs)
  idx = np.where((vel > vel[peak] - fitsize) & (vel < vel[peak] + fitsize+1))[0]

  wf = np.zeros(len(bfgs))
  wf[idx] = 1.0

  params = lmfit.Parameters()
  params.add('ampl1', value = bfgs[peak])
  params.add('vrad1', value = vel[peak])
  params.add('gwidth', value = gwidth,vary = True)
  params.add('const', value = 0.0)
  params.add('vsini1', value = vsini)
  params.add('limbd1', value = 0.68,vary = False)  

  fit = lmfit.minimize(rotbf_res, params, args=(vel,bfgs,wf),xtol=1.e-8,ftol=1.e-8,max_nfev=500)
  if print_report: print(lmfit.fit_report(fit, show_correl=False))
  
  ampl, gwidth = fit.params['ampl1'].value, fit.params['gwidth'].value
  vrad, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
  limbd, const = fit.params['limbd1'].value, fit.params['const'].value
  model = rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd)

  return fit, model, bfgs

# =============================================================================
# Collection of calls
# =============================================================================
def auto_RVs_BF(wl,fl,bl,twl,tfl,q=[99.99],
                dv=1.0,edge=0.0,rvr=201,vsini=5.0,
                fitsize=30,res=90000,smooth=5.0,
                bvc=0.0):
  '''Automatic RVs from BF.

  Collection of function calls to get radial velocity and vsini from the broadening function following the recipe by J. Jessen-Hansen.
  
  :param wl: Observed wavelength.
  :type wl: array 
  :param nfl: Observed normalized flux.
  :type nfl: array 
  :param twl: Template wavelength.
  :type twl: array 
  :param tfl: Template flux.
  :type tfl: array 
  :param q: Percentiles for ``crm``. Default ``[99.99]``.
  :type q: list, optional
  :param dv: RV steps in km/s. Default 1.0.
  :type dv: float, optional
  :param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0.
  :type edge: float, optional 
  :param rvr: Range for RVs in km/s. Default 201.
  :type rvr: int, optional
  :param fitsize: Interval to fit within. Default 30.
  :type fitsize: int, optional
  :param res: Resolution of spectrograph. Default 90000 (SONG).
  :type res: int, optional
  :param vsini: Projected rotational velocity in km/s. Default 5.0.
  :type vsini: float, optional
  :param smooth: Smoothing factor. Default 5.0.
  :type smooth: float, optional
  :param bvc: Barycentric velocity correction for epoch in km/s. Default 0.0.
  :type bvc: float, optional 
    
  :returns: position of Gaussian, radial velocity in km/s, uncertainty of radial velocity in km/s, width of Gaussian, rotational velocity in km/s, uncertainty of rotational velocity in km/s
  :rtype: float, float, float, float

  '''  
  wl, nfl = normalize(wl,fl,bl=bl)
  wl, nfl = crm(wl,nfl,q=q)
  lam, rf_fl, rf_tl = resample(wl,nfl,twl,tfl)
  rvs, bf = getBF(rf_fl,rf_tl,rvr=201)
  rot = smoothBF(rvs,bf,sigma=smooth)

  fit, _, _ = rotbf_fit(rvs,rot,fitsize,res=res,vsini=vsini,smooth=smooth)
  RV, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
  eRV, evsini = fit.params['vrad1'].stderr, fit.params['vsini1'].stderr
  return RV, eRV, vsini, evsini

def auto_RVs(wl,fl,bl,twl,tfl,q=[99.99],
  dv=1.0,edge=0.0,rvr=401,bvc=0.0):
  '''Automatic RVs from CCF.

  Collection of function calls to get radial velocity from the cross-correlation function.
  
  :param wl: Observed wavelength 
  :type wl: array 
  :param nfl: Observed normalized flux
  :type nfl: array 
  :param twl: Template wavelength.
  :type twl: array 
  :param tfl: Template flux.
  :type tfl: array 
  :param q: Percentiles for ``crm``. Default ``[99.99]``.
  :type q: list, optional
  :param dv: RV steps in km/s. Default 1.0.
  :type dv: float, optional
  :param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0.
  :type edge: float, optional 
  :param rvr: Range for RVs in km/s. Default 401.
  :type rvr: int, optional
  :param bvc: Barycentric velocity correction for epoch in km/s. Default 0.0.
  :type bvc: float, optional 
    
  :return: position of Gaussian, radial velocity in km/s
  :rtype: float, float

  '''
  wl, nfl = normalize(wl,fl,bl)
  wl, nfl = crm(wl,nfl,q=q)
  lam, resamp_flip_fl, resamp_flip_tfl = resample(wl,nfl,twl,tfl,dv=dv,edge=edge)
  nbins = len(lam)
  rvs, ccf = getCCF(resamp_flip_fl,resamp_flip_tfl,rvr=rvr)
  rvs = rvs + bvc
  rv, erv = getRV(rvs,ccf,nbins=nbins)
  
  return rv, erv


def get_val_err(vals,out=True,sigma=5):
  '''Outlier rejection.

  Value and error estimation with simple outlier rejection.
  
  :params:
    vals : array/list of values
    out  : bool, if True an outlier rejection will be performed - Gaussian
    sigma: float, level of the standard deviation
    
  :return:
    val  : float, median of the values
    err  : float, error of the value
  '''
  if out:
    mu, sig = sct.norm.fit(vals)
    sig *= sigma
    keep = (vals < (mu + sig)) & (vals > (mu - sig))
    vals = vals[keep]
  
  val, err = np.median(vals), np.std(vals)/np.sqrt(len(vals))
  return val, err

# =============================================================================
# Command line calls
# =============================================================================
  
if __name__=='__main__':
  import sys
  import argparse 
  import glob
  import os
  
  def str2bool(arg):
    if isinstance(arg, bool):
      return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
  
  def low(arg):
    return arg.lower()
  ### Command line arguments to parse to script    
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
  
  ### Input
  putin = parser.add_argument_group('Input')
  putin.add_argument('path', type=str, help='Path to .fits files.')
  putin.add_argument('-tpath', '--template_path', type=str, help='Path to template.',
                     default='/home/emil/Desktop/PhD/scripts/templates/6250_35_p00p00.ms.fits')
  putin.add_argument('-t','--template',type=low, default='kurucz',
                     choices=['kurucz','phoenix'], help='Type of template.')
  
  ### Output
  output = parser.add_argument_group('Output')
  output.add_argument('-w', '--write', type=str2bool, default=True,
                      nargs='?', help='Should RVs be written to file?')
  output.add_argument('-s','--store', default=os.getcwd() + '/',
                      type=str, help='Path to store the output - RV_star.txt.')

  ### Custom
  custom = parser.add_argument_group('Custom')
  custom.add_argument('-dv','--delta_velocity',type=float,default=1.0,
                      help='Velocity step for radial velocity arrray in km/s.')
  custom.add_argument('-rvr','--rv_range',type=int,default=401,
                      help='Radial velocity range for RV fit in km/s.')
  custom.add_argument('-edge','--remove_edge',type=float,default=0.0,
                      help='Omit spectrum at edge of detector in Angstrom.')
  custom.add_argument('-so','--start_order',type=int,default=0,
                      help='Order to start from - skip orders before.')
  custom.add_argument('-eo','--end_order',type=int,default=0,
                      help='Order to end at - skip orders after.')
  custom.add_argument('-xo','--exclude_orders',type=str,default='none',
                      help='Orders to exclude - e.g., "0,12,15,36".')
  custom.add_argument('-out','--outlier_rejection', type=str2bool, default=True,
                      nargs='?', help='Outlier rejection?')
  custom.add_argument('-sig','--sigma_outlier',type=float,default=5,
                      help='Level of outliger rejection.')
  
  args, unknown = parser.parse_known_args(sys.argv[1:])

  tfile = args.template_path
  template = args.template
  if template == 'kurucz':
    twl, tfl = read_kurucz(tfile)
  elif  template == 'phoenix':
    twl, tfl = read_phoenix(tfile)
  print(args.path + '*.fits')
  
  filepath = args.path
  if filepath[-1] != '/': filepath = filepath + '/'
  fits_files = glob.glob(filepath + '*.fits')

  edge = args.remove_edge
  out, sigma = args.outlier_rejection, args.sigma_outlier
  write = args.write

  if write:
    first = pyfits.open(fits_files[0])
    star = first[0].header['OBJECT']
    first.close()
    fname = args.store + '{}_rvs.txt'.format(star.replace(' ',''))
    rv_file = open(fname,'w')
    lines = ['# bjd rv (km/s) srv (km/s)\n']

  vsinis = np.array([])
  
  for file in fits_files:
    data, no_orders, bjd, bvc, star, date, exp = SONG_request(file)
    print(star,date)
    print('Exposure time: {} s'.format(exp))
    rv_range = args.rv_range
    vel_step = args.delta_velocity
    rv, cc = np.empty([no_orders,rv_range]), np.empty([no_orders,rv_range]) 
    RVs = np.array([])
    
    orders = np.arange(no_orders)
    so, eo = args.start_order, args.end_order
    if (eo > 0) & (no_orders > (eo+1)):
      assert eo > so, "Error: Starting order is larger than or equal to ending order."
      orders = orders[so:eo+1]
    else:
      orders = orders[so:]
    
    ex_orders = args.exclude_orders
    if ex_orders != 'none':
      ex_orders = [int(ex) for ex in ex_orders.split(',')]
      orders = [order for order in orders if order not in ex_orders]

    for order in orders:
      wl, fl, bl = SING(data,order)
      RV, vsini = auto_RVs(wl,fl,bl,twl,tfl,dv=vel_step,edge=edge,rvr=rv_range,bvc=bvc)
      RVs = np.append(RVs,RV)
      vsinis = np.append(vsinis,vsini)

    RV, sigRV = get_val_err(RVs,out=out,sigma=sigma)
    
    print('RV = {:.6f} +/- {:0.6f} km/s\n'.format(RV,sigRV))
    line = '{} {} {}\n'.format(bjd,RV,sigRV)
    if write: lines.append(line)

  vsini, sigvsini = get_val_err(vsinis,out=out,sigma=sigma)
  print('vsini = {:.3f} +/- {:0.3f} km/s'.format(vsini,sigvsini))
  
  if write: 
    rv_file.write('# {} \n# vsini = {} +/- {} km/s\n'.format(star,vsini,sigvsini))
    for line in lines: rv_file.write(line)
    rv_file.close()

