#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:47:00 2019

@author: emil
"""

import math

def hpd(data, level) :
    """ The Highest Posterior Density (credible) interval of data at level level.
  
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
    """ Format the value and uncertainties correctly
  
    :param val: value
    :param low: lower uncertainty
    :param up: uper uncertainty
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