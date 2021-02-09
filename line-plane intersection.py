# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:09:55 2021

@author: laure
"""

import numpy as np

def line_plane_angle(x, x_0, y, y_0, z, z_0): #(x,y,z) coords are the plane of the scattering detector, others are absorbing detector. z always equals 0 by defintion
    theta = np.arccos((z- z_0)/np.sqrt((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))
    return theta

print(line_plane_angle(1, 2, 2, 3, 0, -5))





















































































def x_prime_y_prime_output(z_prime, theta, phi, alpha, steps):
    a = np.tan(alpha)
    
    x_prime_vals = []
    y_prime_vals = []
    
    for i in range(0,2*np.pi, steps): #i is our psi variable
        
        z = z_prime/(a*np.cos(i)*np.cos(theta)*np.cos(phi)
            - a*np.sin(i)*np.sin(phi) + np.sin(theta)*np.cos(phi))

        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi))

        x_prime = z*(a*np.cos(i)*np.sin(theta) + np.cos(theta)) 
        
        y_prime_vals.append(y_prime)
    
        x_prime_vals.append(x_prime)
    
    return x_prime_vals, y_prime_vals