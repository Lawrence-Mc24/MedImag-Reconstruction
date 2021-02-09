# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:34:21 2021

@author: lawrence
"""

import numpy as np
import scipy.constants


x1 = np.array([0, 0, 0])
x2 = np.array([1, 1, 1])
vector = x2-x1
print(vector)

#Scientific constants
h = scipy.constants.h
m_e = scipy.constants.m_e
c = scipy.constants.c
e = scipy.constants.e

def compton_angle(E_initial, E_final):
    '''Function calculating Compton scatter angle from initial and final
    energy (Joules)'''
    E_initial = E_initial*e
    E_final = E_final*e
    initial = h*c/E_initial
    final = h*c/E_final
    angle = np.arccos(1 - (m_e*c/h)*(final-initial))
    return angle

def compton_angle_err(err_initial, err_final):
    '''Error on the Compton angle'''
    denom = np.sqrt(1-(1-m_e*c/h)**2)
    err_angle = np.sqrt((err_initial*denom)+(err_final*denom))
    return err_angle

def theta(x, x0, y, y0, z, z0):
    '''Finding the angle between the line of response and the normal to the
    detector plane'''
    diff_x = x-x0
    diff_y = y-y0
    diff_z = z-z0
    denom = np.sqrt((diff_x)**2+(diff_y)**2+(diff_z)**2)
    theta = np.arccos(diff_z*denom)
    return theta