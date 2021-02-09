# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:34:21 2021

@author: lawrence
"""

import numpy as np
import scipy.constants
import sympy as sym
from sym import symbols, function, acos

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

def theta_err(xval, yval, zval, x0val, y0val, z0val, dx, dx0, dy, dy0, dz, dz0):
    '''Finding error on the calculated value of theta'''
    x,y,z,x0,y0,z0 = sym.symbols('x y z x0 y0 z0')
    f = sym.Function('f')
    f = acos((z-z0)/sym.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))
    div_x = sym.diff(f,x)
    div_y = sym.diff(f,y)
    div_z = sym.diff(f,z)
    div_x0 = sym.diff(f,x0)
    div_y0 = sym.diff(f,y0)
    div_z0 = sym.diff(f,z0)
    dee_x = div_x.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    dee_y = div_y.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    dee_z = div_z.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    dee_x0 = div_x0.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    dee_y0 = div_y0.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    dee_z0 = div_z0.evalf(subs={x:xval,x0:x0val,y:yval,y0:y0val,z:zval,z0:z0val})
    f_error = np.sqrt((dee_x*dx)**2+(dee_y*dy)**2+(dee_z*dz)**2+
                      (dee_x0*dx0)**2+(dee_y0*dy0)**2+(dee_z0*dz0)**2)
    return f_error
    