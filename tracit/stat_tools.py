#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:47:00 2019

@author: emil
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from .priors import tgauss_prior, gauss_prior, flat_prior, tgauss_prior_dis, flat_prior_dis

def hpd(data, level) :
    """ The Highest Posterior Density.
    
    The Highest Posterior Density (credible) interval of data at level level.
  
    :param data: sequence of real values
    :param level: (0 < level < 1)
    """ 
    
    d = list(data)
    d.sort()
  
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
      raise RuntimeError("not enough data")
    
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
      rk = d[k+nIn-1] - d[k]
      if rk < r :
        r = rk
        i = k
  
    assert 0 <= i <= i+nIn-1 < len(d)
    
    return (d[i], d[i+nIn-1], i, i+nIn-1)


def significantFormat(val,low,up=None):
    """ Format to significant digit.
    
    Format the value and uncertainties according to significant digit.

    :param val: value
    :type val: float
    
    :param low: lower uncertainty
    :type low: float
    
    :param up: upper uncertainty
    :type up: float
    """ 
    def orderOfMagnitude(number):
        return math.floor(math.log(number, 10))
    def findLeadingDigit(number,idx):
        b = str(number).split('.')
        n = str(float(b[idx]))
        return n.startswith('1')
        
    loom = orderOfMagnitude(low)
    if up:
        uoom = orderOfMagnitude(up)
    else:
        uoom = loom
        up = low
    if low < up:         
        if loom < 0:
            one = findLeadingDigit(low,-1)
            if one: loom -= 1
            p = '0.{}f'.format(-1*loom)
        elif loom == 0:
            one = findLeadingDigit(low,0)
            if one: 
                p = '{}.1f'.format(loom)
            else:
                p = '{}.0f'.format(loom)
        else:
            one = findLeadingDigit(low,0)
            if one: loom -= 1
            p = '{}.0f'.format(loom)                
    else:
        if uoom < 0:
            one = findLeadingDigit(up,-1)
            if one: uoom -= 1
            p = '0.{}f'.format(-1*uoom)
        elif uoom == 0:
            one = findLeadingDigit(up,0)
            if one: 
                p = '{}.1f'.format(uoom)
            else:
                p = '{}.0f'.format(uoom)
        else:
            one = findLeadingDigit(up,0)
            if one: uoom -= 1
            p = '{}.0f'.format(uoom)
                                
    prec = '{:' + p + '}'
    val = prec.format(val)
    low = prec.format(low)
    up = prec.format(up)    

    return val, low, up

# =============================================================================
# Plots
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
    plt.rc('text',usetex=True)
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
    plt.rc('text',usetex=True)
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
                #bounds = stat_tools.hpd(samples[:,i], 0.68)
                bounds = hpd(samples[:,i], 0.68)
                #up = bounds[1] - val
                #low = val - bounds[0]
                quantiles.append([bounds[0],val,bounds[1]])



    axes = np.array(fig.axes).reshape((ndim, ndim))

 
    # if priors == None:
    #   priors = {}
    #   for i in range(ndim): priors[1] = ['none']

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        #ax.axvline(value1[i], color="g")
        ax.axvline(quantiles[i][1], color='C0')
        ax.axvline(quantiles[i][0], color='C0', linestyle='--')
        ax.axvline(quantiles[i][2], color='C0', linestyle='--')
        if diag_titles is None:
            #val, low, up = stat_tools.significantFormat(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1])
            val, low, up = significantFormat(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1])
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

