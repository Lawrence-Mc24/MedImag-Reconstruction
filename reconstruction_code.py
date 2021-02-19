# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:09:55 2021

@author: laure
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import sys

h = scipy.constants.h
m_e = scipy.constants.m_e
c = scipy.constants.c
e = scipy.constants.e

def compton_angle(E_initial, E_final):
    '''Function calculating Compton scatter angle from initial and final
    energy (Joules)'''
    # May not need e factor depending on detector energy output
    E_initial = E_initial*e
    E_final = E_final*e
    initial = h*c/E_initial
    final = h*c/E_final
    angle = np.arccos(1 - (m_e*c/h)*(final-initial))
    return angle

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
    if N[0] == 0:
        return 0
    else:
        return np.arccos(N[0]/((N[0]**2 + N[1]**2)**0.5))

def dz(theta, phi, psi, z_prime, a):
    '''Calculate dz_prime/dpsi'''
    return - z_prime*np.sin(psi)*np.sin(theta)*a/((np.cos(theta) - a*np.cos(psi)*np.sin(theta)))**2

def dpsi(ds, theta, phi, psi, z_prime, a):
    '''Calculate the dpsi increment for a given ds at a given psi for given angles.'''
    h = dz(theta, phi, psi, z_prime, a)
    z = z_prime/(-a*np.cos(psi)*np.sin(theta) + np.cos(theta))
    c11 = np.cos(phi)*np.cos(theta)
    c12 = -np.sin(phi)
    c13 = np.cos(phi)*np.sin(theta)
    dx_psi = a*h*(c11*np.cos(psi) + c12*np.sin(psi) + c13/a) + a*z*(-c11*np.sin(psi) + c12*np.cos(psi))
    
    c21 = np.sin(phi)*np.cos(theta)
    c22 = np.cos(phi)
    c23 = np.sin(phi)*np.sin(theta)
    dy_psi = a*h*(c21*np.cos(psi) + c22*np.sin(psi) + c23/a) + a*z*(-c21*np.sin(psi) + c22*np.cos(psi))
    return ds/np.sqrt(dx_psi**2 + dy_psi**2)

def dpsi_for_equal_dx(dx, theta, phi, psi, z_prime, a):
    '''Calculate the dpsi increment for a given dx at a given psi for given angles.'''
    h = dz(theta, phi, psi, z_prime, a)
    z = z_prime/(-a*np.cos(psi)*np.sin(theta) + np.cos(theta))
    c11 = np.cos(phi)*np.cos(theta)
    c12 = -np.sin(phi)
    c13 = np.cos(phi)*np.sin(theta)
    print(dx, theta, phi, psi, z_prime, a)
    print(f'a = {a}, h = {h}, z = {z}, c11 = {c11}, c12 = {c12}, c13 = {c13}')
    dx_psi = a*h*(c11*np.cos(psi) + c12*np.sin(psi) + c13/a) + a*z*(-c11*np.sin(psi) + c12*np.cos(psi))
    print(f'dx_psi = {dx_psi}')
    # sys.exit()
    return dx/dx_psi

def dpsi_for_equal_dy(dy, theta, phi, psi, z_prime, a):
    '''Calculate the dpsi increment for a given dy at a given psi for given angles.'''
    h = dz(theta, phi, psi, z_prime, a)
    z = z_prime/(-a*np.cos(psi)*np.sin(theta) + np.cos(theta))
    c21 = np.sin(phi)*np.cos(theta)
    c22 = np.cos(phi)
    c23 = np.sin(phi)*np.sin(theta)
    dy_psi = a*h*(c21*np.cos(psi) + c22*np.sin(psi) + c23/a) + a*z*(-c21*np.sin(psi) + c22*np.cos(psi))
    print(dy, theta, phi, psi, z_prime, a)
    print(f'a = {a}, h = {h}, z = {z}, c21 = {c21}, c22 = {c22}, c23 = {c23}')
    print(f'dy_psi = {dy_psi}')
    # sys.exit()
    return dy/dy_psi

def psi_calculator2(dx, theta, phi, z_prime, a, n, alpha):
    '''Calculate list of psi values required to keep the point spacing at a fixed dx'''
    psi = 0
    psi_list = [0]
    while True:
        d = dpsi_for_equal_dy(dx, theta, phi, psi, z_prime, a)
        print(f'd = {d}')
        psi += d
        print(f'psi = {psi}')
        if np.abs(psi) >= 2*np.pi:
            break
        else:
            psi_list.append(psi)
    
    print(f'psi_list = {psi_list}')
    return psi_list

def psi_calculator(ds, theta, phi, z_prime, a, n, alpha):
    '''calculate list of psi values required to keep the point spacing at a fixed length, ds, 
    along the curve'''
    psi = 0
    psi_list = [0]
    while True:
        d = dpsi(ds, theta, phi, psi, z_prime, a)
        psi += d
        if psi >= 2*np.pi:
            break
        else:
            psi_list.append(psi)
    return psi_list

def x_prime_y_prime_output2(z_prime, theta, phi, alpha, steps, r1, estimate):
    a = np.tan(alpha)
    
    x_prime_vals = []
    y_prime_vals = []
    
    z_prime = z_prime - r1[2]
    dx = 2*np.pi*estimate*np.tan(alpha)/(steps-1)
    for i in psi_calculator2(dx, theta, phi, z_prime, a, steps, alpha): #i is our psi variable
        
        z = z_prime/(-a*np.cos(i)*np.sin(theta) + np.cos(theta))
        
        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi)) + r1[1]

        x_prime = z*(a*np.cos(i)*np.cos(phi)*np.cos(theta) - a*np.sin(i)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) + r1[0]
        
        y_prime_vals.append(y_prime)
    
        x_prime_vals.append(x_prime)
    return x_prime_vals, y_prime_vals

def x_prime_y_prime_output(z_prime, theta, phi, alpha, steps, r1, estimate):
    a = np.tan(alpha)
    
    x_prime_vals = []
    y_prime_vals = []
    
    z_prime = z_prime - r1[2]
    ds = 2*np.pi*estimate*np.tan(alpha)/(steps-1)
    for i in psi_calculator(ds, theta, phi, z_prime, a, steps, alpha): #i is our psi variable
        
        z = z_prime/(-a*np.cos(i)*np.sin(theta) + np.cos(theta))
        
        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi)) + r1[1]

        x_prime = z*(a*np.cos(i)*np.cos(phi)*np.cos(theta) - a*np.sin(i)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) + r1[0]
        
        y_prime_vals.append(y_prime)
    
        x_prime_vals.append(x_prime)
    
    return x_prime_vals, y_prime_vals

def plot_it(x, ys, r1, x_name='x', y_name='y', plot_title='Plot', individual_points=False):
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
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.plot(r1[0], r1[1], 'ro')
    plt.axhline(y=r1[1], color='g')
    plt.axvline(x=r1[0], color='g')
    for i, k in enumerate(ys):
        plt.plot(x, k)
        # Useful to plot individual points for mean duration against square.
        if individual_points:
            plt.plot(x, k, 'r.')
    plt.grid(True)
    plt.show()
    return figure

def give_x_y_for_two_points(r1, r2, z_prime, alpha, steps, estimate):
    '''
    Computes an array of (x, y) points on the imaging plane a distance z_prime from the scatter detector
    for an event at points r1, r2.
    Parameters
    ----------
    r1 : array_like
        Hit point on the (first) Compton scatter detector of the form np.array([x1, y1, z1]), 
        where the coordinates are in the global (primed) frame.
    r2 : array_like
        Hit point on the (second) absorbing detector of the form np.array([x2, y2, z2]), 
        where the coordinates are in the global (primed) frame.
    z_prime : float
        Perpendicuar distance between the scatter detector and the imaging plane.
    Returns
    -------
    ndarray
        Numpy array of the form np.array([x, y]) of the (x, y) values imaged on the imaging plane.
    '''
    theta = theta_angle(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])
    phi = phi_angle(N(cone_vector(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])))
    # print(f'theta = {theta}, phi = {phi}')
    x, y = x_prime_y_prime_output(z_prime, theta, phi, alpha, steps, r1, estimate)
    # print(x, y)
    return np.array([x, y])

def plot_it2(xys, r1s, x_name='x', y_name='y', plot_title='Plot', individual_points=False):
    '''
    Plot many different sets of (x, y) arrays on the same axes, graph, and figure.
    Parameters
    ----------
    xys : array_like
        Array where each item is an array of the form np.array([x, y]) and x, y are the arrays to be
        plotted.
    r1s : array_like
        Array of points on the first detector of the form np.array([x1, y1, z1]). Plot (x1, y1) and
        axes around it.
    x_name : string
        The name on the x-axis.
    y_name : string
        The name on the y-axis.
    plot_title : string
        The title on the graph.
    individual_points : Boolean, optional
        If True, will plot individual points as 'r.'. The default is False.
    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot.
    '''
    
    # Plot
    figure = plt.figure(figsize=(10,6))
    plt.axis('equal')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    for i, k in enumerate(r1s):
        plt.plot(k[0], k[1], 'ro')
        plt.axhline(y=k[1], color='g')
        plt.axvline(x=k[0], color='g')
    for i, k in enumerate(xys):
        plt.plot(k[0], k[1])
        # Useful to plot individual points for mean duration against square.
        if individual_points:
            plt.plot(k[0], k[1], 'r.')
    plt.grid(True)
    plt.show()
    return figure

def calculate_heatmap(x, y, bins=50):
    '''
    Calculate heatmap and its extent using np.histogram2d() from x and y values for a given 
    number of bins.

    Parameters
    ----------
    x : numpy_array
        Must be a numpy array, not a list!
    y : numpy_array
        DESCRIPTION.
    bins : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    heatmap : numpy_array
        A numpy array of the shape (bins, bins) containing the histogram values: x along axis 0 and
        y along axis 1.
    extent : TYPE
        DESCRIPTION.

    '''
    heatmap, xedges, yedges = np.histogram2d(x, y, bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent

def plot_heatmap(heatmap, extent):
    '''Plot a heatmap using plt.imshow().'''
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.colorbar()
    plt.show()


r1 = np.array([0, 0.1, 0])
r2 = np.array([0, 0.1, -1])
r3 = np.array([0.2, 0.3, 0.1])
r4 = np.array([0.5, 0.1, -1])
r5 = np.array([0, 0.4, -0.1])
r6 = np.array([0.5, 0.1, -1])
xy1 = give_x_y_for_two_points(r1, r2, z_prime=1, alpha=np.pi/4, steps=180, estimate=1)
xy2 = give_x_y_for_two_points(r3, r4, z_prime=1, alpha=np.pi/4, steps=180, estimate=1)
xy3 = give_x_y_for_two_points(r5, r6, z_prime=1, alpha=np.pi/4, steps=180, estimate=1)

# NB: We must make xys a list rather than a numpy array because the 2d arrays can be of different
# sizes in which case they cannot be broadcast into the same 3d np array.
xys = [xy1, xy2, xy3]
plot_it2(xys, np.array([r1, r3, r5]), individual_points=True)
#plot_it(x, ys=np.array([y]), r1=r1, individual_points=False)


# Iterate through alpha
# Arbitrarily choose alpha to be 45 degrees and its error to be 5%
alpha = np.pi/4
alpha_err = alpha*0.05
# Plot for alpha and its min & max boundaries
alpha_bounds = np.linspace(alpha-alpha_err, alpha+alpha_err, num=50)
xy1s = np.array([])
x1s = np.array([])
x2s = np.array([])
x3s = np.array([])
y1s = np.array([])
y2s = np.array([])
y3s = np.array([])

for angle in alpha_bounds:
    xy1s = np.append(xy1s, give_x_y_for_two_points(r1, r2, z_prime=1, alpha=angle, steps=180, estimate=1))
    x1s = np.append(x1s, give_x_y_for_two_points(r1, r2, z_prime=1, alpha=angle, steps=180, estimate=1)[0])
    y1s = np.append(y1s, give_x_y_for_two_points(r1, r2, z_prime=1, alpha=angle, steps=180, estimate=1)[1])
    x2s = np.append(x2s, give_x_y_for_two_points(r3, r4, z_prime=1, alpha=angle, steps=180, estimate=1)[0])
    y2s = np.append(y2s, give_x_y_for_two_points(r3, r4, z_prime=1, alpha=angle, steps=180, estimate=1)[1])
    x3s = np.append(x3s, give_x_y_for_two_points(r5, r6, z_prime=1, alpha=angle, steps=180, estimate=1)[0])
    y3s = np.append(y3s, give_x_y_for_two_points(r5, r6, z_prime=1, alpha=angle, steps=180, estimate=1)[1])
# plot_it2(np.array([x1s, y1s]), np.array([r1, r1, r1]), individual_points=True)
# plot_it2(np.array([x2s, y2s]), np.array([r3, r3, r3]), individual_points=True)
# plot_it2(np.array([x3s, y3s]), np.array([r5, r5, r5]), individual_points=True)

xs = np.array([x1s, x2s, x3s])
ys = np.array([y1s, y2s, y3s])
heatmap_combined, extent_combined = calculate_heatmap(np.concatenate((x1s, x2s, x3s)), np.concatenate((y1s, y2s, y3s)), bins=175)
# plot_heatmap(xs, ys)
heatmap1, extent1 = calculate_heatmap(x1s, y1s)
heatmap2, extent2 = calculate_heatmap(x2s, y2s)
heatmap3, extent3 = calculate_heatmap(x3s, y3s)

plot_heatmap(heatmap_combined, extent_combined)

heatmap_combined[heatmap_combined != 0] = 1
plot_heatmap(heatmap_combined, extent_combined)

x = np.concatenate((xy1[0], xy2[0], xy3[0]))
y = np.concatenate((xy1[1], xy2[1], xy3[1]))
plt.hist2d(x, y, bins=50)
plt.colorbar()
plt.show()

plot_heatmap(*calculate_heatmap(x, y))


