#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:37:23 2020

@author: emil
"""
# =============================================================================
# Packages
# =============================================================================
from scipy.special import erf
#from scipy.stats import truncnorm, beta
import scipy.stats as ss
import numpy as np

# =============================================================================
# Starting distribution
# =============================================================================

def flat_prior_dis(r,x1,x2):
	return x1 + r*(x2 - x1)

def tgauss_prior_dis(mu,sigma,xmin,xmax):
	a = (xmin - mu)/sigma
	b = (xmax - mu)/sigma
	return ss.truncnorm.rvs(a,b,loc=mu,scale=sigma)

def jeff_prior_dis(r,xmin,xmax):
	if r <= 0.0:
		return 0.0#-1.0e-32
	else:
		if xmin == 0:
			xmin == 1e-10
		return np.exp(r*np.log(xmax/xmin) + np.log(xmin))

def beta_prior_dis():
	alpha = np.random.normal(1.12,0.1)
	beta = np.random.normal(3.09,0.3)

	return np.random.beta(alpha,beta)

# =============================================================================
# Priors
# =============================================================================
  
def flat_prior(val,xmin,xmax):
	if val < xmin or val > xmax:
		ret = 0.0
	else:
		ret = 1/(xmax - xmin)
	return ret

def gauss_prior(val,xmid,xwid):
	nom = np.exp(-1.0*np.power(val - xmid,2)/(2*np.power(xwid,2)))
	den = np.sqrt(2*np.pi)*xwid
	return nom/den

def tgauss_prior(val,xmid,xwid,xmin,xmax):
	if val < xmin or val > xmax:
		ret = 0.0
	else:
		nom = np.exp(-1.0*np.power(val-xmid,2) / (2*np.power(xwid,2))) / (np.sqrt(2*np.pi)*xwid)
		den1 = (1. + erf((xmax - xmid)/(np.sqrt(2)*xwid)))/2
		den2 = (1. + erf((xmin - xmid)/(np.sqrt(2)*xwid)))/2
		ret = nom/(den1 - den2)
	return ret

def jeff_prior(val, xmin, xmax):
	if val < xmin or val > xmax:
		return 0.0
	else:
		return 1/(val*np.log(xmax/xmin))

def beta_prior(val):
	alpha = np.random.normal(1.12,0.1)
	beta = np.random.normal(3.09,0.3)

	dist = ss.beta(alpha,beta)
	return dist.pdf(val)