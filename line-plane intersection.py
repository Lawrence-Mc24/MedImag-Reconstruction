# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:09:55 2021

@author: laure
"""

import numpy as np
import matplotlib.pyplot as plt

def theta_angle(x_prime, x_0_prime, y_prime, y_0_prime, z_prime, z_0_prime):
    '''
    Calculate the angle between the cone axial vector and the z_prime axis

    Parameters
    ----------
    x_prime, y_prime, z_prime : TYPE - float
        DESCRIPTION - x,y,z coordinate of the hit on the scattering detector in primed (global) coordinate system. z=0 by definition.
    x_0_prime, y_0_prime, z_0_prime : TYPE - float
        DESCRIPTION - x_0, y_0, z_0 coordinate of the hit on the absorbing detector in primed (global) coordinate system.

    Returns
    -------
    theta : float
        The angle between the cone axial vector and the z_prime axis (radians).

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

def x_prime_y_prime_output(z_prime, theta, phi, alpha, steps, r1):
    a = np.tan(alpha)
    
    x_prime_vals = []
    y_prime_vals = []
    
    z_prime = z_prime - r1[2]
    for i in np.linspace(0, 2*np.pi, steps): #i is our psi variable
        
        z = z_prime/(-a*np.cos(i)*np.sin(theta) + np.cos(theta))

        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi)) + r1[1]

        x_prime = z*(a*np.cos(i)*np.cos(phi)*np.cos(theta) - a*np.sin(i)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) + r1[0]
        
        y_prime_vals.append(y_prime)
    
        x_prime_vals.append(x_prime)
    
    return x_prime_vals, y_prime_vals

def plot_it(x, ys, y_labels, x_name='x', y_name='y', plot_title='Plot', individual_points=False):
    '''
    Plot many different ys versus the same x on the same axes, graph, and figure.
    
    ---------------------
    Parameters:
    ---------------------
    
    x : array_like
        The independent variable to be plotted on be x-axis.
    
    ys : array_like
        Array of dependent variables to be plotted on be y-axis, where each row of ys is an array
        of y-values to plot against x.
    
    x_name : string
        The name on the x-axis.
    
    y_name : string
        The name on the y-axis.
        
    y_labels : array_like
        Array of size np.shape(ys)[0] where each element y_labels[i] corresponds to ys[i].
    
    plot_title : string
        The title on the graph.
        
    individual_points : Boolean, optional
        If True, will plot individual points as 'r.'. The default is False.
    
    --------------------
    Returns:
    --------------------
    
    figure : matplotlib.figure.Figure
        The plot.
    '''
    
    # Plot
    figure = plt.figure(figsize=(10,6))
    plt.axis('equal')

    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    for i, k in enumerate(ys):
        plt.plot(x, k, label=y_labels[i])
        # Useful to plot individual points for mean duration against square.
        if individual_points:
            plt.plot(x, k, 'r.')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()
    return figure

plot_it(x=np.linspace(0, 10, 11), ys=np.array([np.linspace(0, 10, 11)]), y_labels=np.array(['y=x']))

r1 = np.array([0.001, 0, -0.5])
r2 = np.array([0, 0, -10])
theta = theta_angle(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])
phi = phi_angle(N(cone_vector(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])))
# print(f'theta = {theta}, phi = {phi}')
x, y = x_prime_y_prime_output(1, theta, phi, alpha=np.pi/4, steps=180, r1=r1)
# print(x, y)
plot_it(x, ys=np.array([y]), y_labels=np.array(['Cone intersection on plane']), individual_points=True)
# test change