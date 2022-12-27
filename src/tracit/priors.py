#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
# =============================================================================
# Packages
# =============================================================================
from scipy.special import erf
import scipy.stats as ss
import numpy as np

# =============================================================================
# Starting distributions
# =============================================================================

def flat_prior_dis(val,xmin,xmax):
	'''Flat distribution.

	See :py:func:`flat_prior`.

	:param val: :math:`x`.
	:type val: float
	
	:param xmin: :math:`a`.
	:type xmin: float

	:param xmax: :math:`b`.
	:type xmax: float

	:returns: :math:`f(x)`
	:rtype: float
		
	'''
	return xmin + val*(xmax - xmin)

def tgauss_prior_dis(mu,sigma,xmin,xmax):
	'''Truncated Gaussian distribution.

	See :py:func:`tgauss_prior`.

	:param xmid: :math:`\mu`.
	:type xmid: float

	:param xwid: :math:`\sigma`.
	:type xwid: float
	
	:param xmin: :math:`a`.
	:type xmin: float

	:param xmax: :math:`b`.
	:type xmax: float

	:returns: :math:`f(x)`
	:rtype: float

	'''
	a = (xmin - mu)/sigma
	b = (xmax - mu)/sigma
	return ss.truncnorm.rvs(a,b,loc=mu,scale=sigma)

def gauss_prior_dis(mu,sigma):
	'''Gaussian distribution.

	See :py:func:`gauss_prior`.

	:param xmid: :math:`\mu`.
	:type xmid: float

	:param xwid: :math:`\sigma`.
	:type xwid: float
	

	:returns: :math:`f(x)`
	:rtype: float

	'''
	return np.random.normal(mu,sigma)


def jeff_prior_dis(val,xmin,xmax):
	if r <= 0.0:
		return 0.0
	else:
		if xmin == 0:
			xmin == 1e-10
		return np.exp(val*np.log(xmax/xmin) + np.log(xmin))

def beta_prior_dis(alphas=[1.12,0.1],betas=[3.09,0.3]):
	'''Beta prior distribution.

	See :py:func:`beta_prior`.

	:param alphas: :math:`a` is drawn from a normal distribution with :math:`\mu=` `alphas[0]` and :math:`\sigma=` `alphas[1]`. Default ``[1.12,0.1]``.
	:type alphas: list

	:param betas: :math:`b` is drawn from a normal distribution with :math:`\mu=` `betas[0]` and :math:`\sigma=` `betas[1]`. Default ``[3.09,0.3]``.
	:type betas: list

	:returns: :math:`f(x)`
	:rtype: float

	'''
	alpha = np.random.normal(alphas[0],alphas[1])
	beta = np.random.normal(betas[0],betas[1])

	return np.random.beta(alpha,beta)

# =============================================================================
# Priors
# =============================================================================
  
def flat_prior(val,xmin,xmax):
	'''Uniform prior.

	.. math:: 
		f(x) = \\frac{1}{b-a}, \, \ \mathrm{for} \ a \leq x \leq b,\ \mathrm{else} \, f(x) = 0 \, .
	
	:param val: :math:`x`.
	:type val: float
	
	:param xmin: :math:`a`.
	:type xmin: float

	:param xmax: :math:`b`.
	:type xmax: float

	:returns: :math:`f(x)`
	:rtype: float

	'''
	if val < xmin or val > xmax:
		ret = 0.0
	else:
		ret = 1/(xmax - xmin)
	return ret

def gauss_prior(val,xmid,xwid):
	'''Gaussian prior.
	
	.. math:: 
		f (x) = \\frac{1}{\sqrt{2 \pi}\sigma} \exp \\left (-\\frac{(x - \mu)^2}{2 \sigma^2} \\right) \, .

	:param val: :math:`x`.
	:type val: float
	
	:param xmid: :math:`\mu`.
	:type xmid: float

	:param xwid: :math:`\sigma`.
	:type xwid: float

	:returns: :math:`f(x)`
	:rtype: float

	'''
	nom = np.exp(-1.0*np.power(val - xmid,2)/(2*np.power(xwid,2)))
	den = np.sqrt(2*np.pi)*xwid
	return nom/den

def tgauss_prior(val,xmid,xwid,xmin,xmax):
	'''Truncated Gaussian prior.
	
	.. math:: 
		f (x; \mu, \sigma, a, b) = \\frac{1}{\sigma} \\frac{g(x)}{\Phi(\\frac{b-\mu}{\sigma}) - \Phi(\\frac{a-\mu}{\sigma})} \, ,

	where :math:`g(x)` is the Gaussian from :py:func:`gauss_prior` and :math:`\Phi` is the (Gauss) error function (`scipy.special.erf <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html?highlight=erf#scipy.special.erf>`_ ).

	:param val: :math:`x`.
	:type val: float

	:param xmid: :math:`\mu`.
	:type xmid: float

	:param xwid: :math:`\sigma`.
	:type xwid: float
	
	:param xmin: :math:`a`.
	:type xmin: float

	:param xmax: :math:`b`.
	:type xmax: float	

	:returns: :math:`f(x)`
	:rtype: float

	'''
	if val < xmin or val > xmax:
		ret = 0.0
	else:
		nom = np.exp(-1.0*np.power(val-xmid,2) / (2*np.power(xwid,2))) / (np.sqrt(2*np.pi)*xwid)
		den1 = (1. + erf((xmax - xmid)/(np.sqrt(2)*xwid)))/2
		den2 = (1. + erf((xmin - xmid)/(np.sqrt(2)*xwid)))/2
		ret = nom/(den1 - den2)
	return ret

def jeff_prior(val, xmin, xmax):
	'''Jeffrey's prior.
	
	'''
	if val < xmin or val > xmax:
		return 0.0
	else:
		return 1/(val*np.log(xmax/xmin))

def beta_prior(val,alphas=[1.12,0.1],betas=[3.09,0.3]):
	'''Beta prior.

	From `scipy.stats.beta <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>`_.

	.. math::
		f (x;a,b) = \\frac{\Gamma (a + b) x^{a-1} (1-x)^{b-1}}{\Gamma (a) \Gamma (b)} \, , \ \mathrm{for} \ 0 \leq x \leq 1, \  a > 0, b > 0 \, ,
	
	where :math:`\Gamma` is the gamma function (`scipy.stats.gamma <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html#scipy.special.gamma>`_)

	:param val: :math:`x`.
	:type val: float

	:param alphas: :math:`a` is drawn from a normal distribution with :math:`\mu=` `alphas[0]` and :math:`\sigma=` `alphas[1]`. Default ``[1.12,0.1]``.
	:type alphas: list

	:param betas: :math:`b` is drawn from a normal distribution with :math:`\mu=` `betas[0]` and :math:`\sigma=` `betas[1]`. Default ``[3.09,0.3]``.
	:type betas: list

	:returns: :math:`f(x)`
	:rtype: float

	'''
	alpha = np.random.normal(alphas[0],alphas[1])
	beta = np.random.normal(betas[0],betas[1])

	dist = ss.beta(alpha,beta)
	return dist.pdf(val)