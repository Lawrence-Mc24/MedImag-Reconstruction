# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:34:21 2021

@author: lawrence
"""

import numpy as np
import scipy.constants
import sympy
from sympy import symbols, Function, acos
import math

x1 = np.array([0, 0, 0])
x2 = np.array([1, 1, 1])
vector = x2-x1
print(vector)

#Scientific constants
h = scipy.constants.h
m_e = scipy.constants.m_e
c = scipy.constants.c
e = scipy.constants.e
comp = m_e*(c**2)/(e*1000)

def compton_angle(E_initial, E_final):
    '''Function calculating Compton scatter angle (radians) from initial and final
    energy (keV)'''
    E_i = E_initial
    E_f = E_final
    angle = np.arccos(1 - (comp)*((1/E_f)-(1/E_i)))
    return angle

def compton_angle_err(E_i, E_f, err_i, err_f):
    '''Error on the Compton angle (radians) taking energies as (keV)'''
    En_f, En_i = sympy.symbols('En_f En_i')
    f = sympy.Function('f')
    f = acos(1 - (comp)*((1/En_f)-(1/En_i)))
    subs={En_i:E_i, En_f:E_f}
    div_f = sympy.diff(f,En_f)
    div_i = sympy.diff(f,En_i)
    dee_i = div_i.evalf(subs=subs)
    dee_f = div_f.evalf(subs=subs)
    f_error = math.sqrt((dee_i*err_i)**2+(dee_f*err_f)**2)
    return f_error

def theta(x, x0, y, y0, z, z0):
    '''Finding the angle between the line of response and the normal to the
    detector plane'''
    diff_x = x-x0
    diff_y = y-y0
    diff_z = z-z0
    denom = np.sqrt((diff_x)**2+(diff_y)**2+(diff_z)**2)
    theta = np.arccos(diff_z*denom)
    return theta

def theta_err(xval, yval, zval, x0val, y0val, z0val, dx, dx0, dy, dy0, dz, dz0):
    '''Finding error on the calculated value of theta'''
    x,y,z,x0,y0,z0 = sympy.symbols('x y z x0 y0 z0')
    f = sympy.Function('f')
    f = acos((z-z0)/sympy.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))
    subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val}
    div_x = sympy.diff(f,x)
    div_y = sympy.diff(f,y)
    div_z = sympy.diff(f,z)
    div_x0 = sympy.diff(f,x0)
    div_y0 = sympy.diff(f,y0)
    div_z0 = sympy.diff(f,z0)
    dee_x = div_x.evalf(subs=subs)
    dee_y = div_y.evalf(subs=subs)
    dee_z = div_z.evalf(subs=subs)
    dee_x0 = div_x0.evalf(subs=subs)
    dee_y0 = div_y0.evalf(subs=subs)
    dee_z0 = div_z0.evalf(subs=subs)
    f_error = math.sqrt((dee_x*dx)**2+(dee_y*dy)**2+(dee_z*dz)**2+
                      (dee_x0*dx0)**2+(dee_y0*dy0)**2+(dee_z0*dz0)**2)
    return f_error