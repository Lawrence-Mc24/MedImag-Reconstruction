# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:09:55 2021

@author: laure
"""

import numpy as np

def theta_angle(x_prime, x_0_prime, y_prime, y_0_prime, z_prime, z_0_prime):
    '''
    Calculate the angle between the cone axial vector and the z_prime axis

    Parameters
    ----------
    x_prime, y_prime, z_prime : TYPE - float
        DESCRIPTION - x,y,z coordinate of the hit on the scattering detector in primed (global) coordinate system. z=0 by definition.
    x_0__prime, y_0__prime, z_0__prime : TYPE - float
        DESCRIPTION - x_0, y_0, z_0 coordinate of the hit on the absorbing detector in primed (global) coordinate system.

    Returns
    -------
    theta : float
        DESCRIPTION.

    '''
    theta = np.arccos((z_prime - z_0_prime)/np.sqrt((x_prime - x_0_prime)**2 + (y_prime - y_0_prime)**2 + (z_prime - z_0_prime)**2))
    return theta



def cone_vector(x_prime, x_0_prime, y_prime, y_0_prime, z_prime, z_0_prime):
    '''
    Returns cone axis vector as a list in primed axes.
    '''
    return [(x_prime-x_0_prime), (y_prime-y_0_prime), (z_prime-z_0_prime)]
    
def N(z):
    '''
    Calculate vector for the line of nodes, N.

    Parameters
    ----------
    z : TYPE - array-like
        DESCRIPTION - of the form [a, b, c] where a,b,c are the components of
    the cone axial vector in the primed coordinate system.
    z_prime : TYPE
        DESCRIPTION.

    Returns
    -------
    N vector of the form [i, j, k] where ijk are the components of the N vector
    in the primed coordinate frame.

    '''
    
    return [-z[1], z[0], 0]

def phi_angle(N):
    '''
    Calculate the angle, phi, between N and the x_prime axis.

    Parameters
    ----------
    N : TYPE - array-like 
        DESCRIPTION - line of nodes of the form [a, b, c] where a,b,c are the components of
    the vector in the primed coordinate system.

    Returns
    -------
    phi

    '''
    
    return np.arccos(N[0]/((N[0]**2 + N[1]**2)**0.5))

def x_prime_y_prime_output(z_prime, theta, phi, alpha, steps):
    a = np.tan(alpha)
    
    x_prime_vals = []
    y_prime_vals = []
    
    for i in range(0,2*np.pi, steps): #i is our psi variable
        
        z = z_prime/(-a*np.cos(i)*np.sin(theta) + np.cos(theta))

        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi))

        x_prime = z*(a*np.cos(i)*np.cos(phi)*np.cos(theta) - a*np.sin(i)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) 
        
        y_prime_vals.append(y_prime)
    
        x_prime_vals.append(x_prime)
    
    return x_prime_vals, y_prime_vals
