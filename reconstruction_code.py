# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:09:55 2021

@author: laurence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants
from scipy import ndimage

h = scipy.constants.h
m_e = scipy.constants.m_e
c = scipy.constants.c
e = scipy.constants.e

#path = r"C:\Users\laure\Documents\Physics\Year 3\Group Study\Point_Source-Truth_Data_3-Lab_Experiment_1-Run_3.csv"
# path = r"C:\Users\laure\Documents\Physics\Year 3\Group Study\Point_Source-Truth_Data_2.csv"
path = 'D:/University/Year 3/Group Studies/Monte Carlo data/compt_photo_chain_data_4_detectors.csv'
#path =  r'C:\Users\laure\Documents\Physics\Year 3\Group Study\Point_Source-Truth_Data_1.csv'
#path = 'D:/University/Year 3/Group Studies/Monte Carlo data/compt_photo_chain_data_4_detectors.csv'
# path = "U:\Physics\Yr 3\MI Group Studies\MC data/compt_photo_chain_data_45_degrees_point_source.csv"
dataframe = pd.read_csv(path)

x_prime = dataframe['X_1 [cm]']
y_prime = dataframe['Y_1 [cm]']
z_prime = dataframe['Z_1 [cm]']
x_0_prime = dataframe['X_2 [cm]']
y_0_prime = dataframe['Y_2 [cm]']
z_0_prime = dataframe['Z_2 [cm]']
E_loss = np.abs(dataframe['Energy Loss [MeV]'])*10**6

r1 = np.array([x_prime, y_prime, z_prime])
r2 = np.array([x_0_prime, y_0_prime, z_0_prime])

points = np.array([r1[0][:], r1[1][:], r1[2][:], r2[0][:], r2[1][:], r2[2][:], E_loss]).T

def compton_angle(E_initial, E_final):
    '''Function calculating Compton scatter angle from initial and final
    energy (eV)'''
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
    '''Calculate dz/dpsi'''
    # print(f'denom is {(np.cos(theta) - a*np.cos(psi)*np.sin(theta))}')
    # print(f'theta is {theta}')
    # print(f'psi is {psi}')
    return - z_prime*np.sin(psi)*np.sin(theta)*a/((np.cos(theta) - a*np.cos(psi)*np.sin(theta)))**2

def dpsi(ds, theta, phi, psi, z_prime, a):
    '''Calculate the dpsi increment for a given ds at a given psi for given angles.'''
    h = dz(theta, phi, psi, z_prime, a)
    # print(f'dz={h}')
    z = z_prime/(-a*np.cos(psi)*np.sin(theta) + np.cos(theta))
    c11 = np.cos(phi)*np.cos(theta)
    c12 = -np.sin(phi)
    c13 = np.cos(phi)*np.sin(theta)
    dx_psi = a*h*(c11*np.cos(psi) + c12*np.sin(psi) + c13/a) + a*z*(-c11*np.sin(psi) + c12*np.cos(psi))
    c21 = np.sin(phi)*np.cos(theta)
    c22 = np.cos(phi)
    c23 = np.sin(phi)*np.sin(theta)
    dy_psi = a*h*(c21*np.cos(psi) + c22*np.sin(psi) + c23/a) + a*z*(-c21*np.sin(psi) + c22*np.cos(psi))
    # print(f'dx_psi={dx_psi}')
    # print(f'dy_psi={dy_psi}')
    return ds/np.sqrt(dx_psi**2 + dy_psi**2)

def psi_calculator(ds, theta, phi, z_prime, a, n, alpha):
    '''calculate list of psi values required to keep the point spacing at a fixed length, ds, 
    along the curve'''
    psi = 0
    psi_list = [0]
    print(f'a value = {a}')
    print(f'value is {(theta+np.arctan(a))*(180/np.pi)}')
    while True:
        if (theta+np.arctan(a)) > np.pi/2:
            break
        d = dpsi(ds, theta, phi, psi, z_prime, a)
        psi += d
        # print(f'psi is {psi}')
        # print(f'd is {d}')
        # print(psi)
        if np.abs(psi) >= 2*np.pi:
            break
        else:
            psi_list.append(psi)
    return psi_list

def x_prime_y_prime_output(z_prime, theta, phi, alpha, steps, r1, estimate, ds=0, ):
    a = np.tan(alpha)
    # print(f'a is {a}')
    if alpha + theta > np.pi/2:
        return x_prime_y_prime_parabola(z_prime, theta, phi, alpha, steps, r1, estimate, ds)
    x_prime_vals = np.array([])
    y_prime_vals = np.array([])
    ds=ds
    z_prime = z_prime - r1[2]
    if ds == 0:
        ds = 2*np.pi*estimate*np.tan(alpha)/(steps-1)
    # print(ds)
    # print(ds)
    for i in psi_calculator(ds, theta, phi, z_prime, a, steps, alpha): #i is our psi variable
        
        z = z_prime/(-a*np.cos(i)*np.sin(theta) + np.cos(theta))
        
        x_prime = z*(a*np.cos(i)*np.cos(phi)*np.cos(theta) - a*np.sin(i)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) + r1[0]
        
        y_prime = z*(a*np.cos(i)*np.cos(theta)*np.sin(phi)
            + a*np.sin(i)*np.cos(phi) + np.sin(theta)*np.sin(phi)) + r1[1]

        
        x_prime_vals = np.append(x_prime_vals, x_prime)
        y_prime_vals = np.append(y_prime_vals, y_prime)

    return x_prime_vals, y_prime_vals, ds

def x_prime_y_prime_parabola(z_prime, theta, phi, alpha, steps, r1, estimate, ds):
    a = np.tan(alpha)
    
    x_prime_vals = np.array([])
    y_prime_vals = np.array([])
   ## 
    z_prime = z_prime - r1[2]
    
    #ds = 2*np.pi*estimate*np.tan(alpha)/(steps-1)
    psi = 0
    anticlockwise = True
    iteration = 'first'
    counter = 0
    while True:
        counter += 1
        z = z_prime/(-a*np.cos(psi)*np.sin(theta) + np.cos(theta))
        
        if counter > 5000:
            if psi == 0 or psi == np.pi or psi == 2*np.pi:
                print(f'psi = {psi}')
                print(f'iteration = {iteration}')
                print(f'anticlockwise = {anticlockwise}')
                print(f'z= {z}')
        if z<0:
            iteration = 'second'
            psi+=np.pi
            continue
        
        x_prime = z*(a*np.cos(psi)*np.cos(phi)*np.cos(theta) - a*np.sin(psi)*np.sin(phi) + 
                     np.cos(phi)*np.sin(theta)) + r1[0]
        
        y_prime = z*(a*np.cos(psi)*np.cos(theta)*np.sin(phi)
            + a*np.sin(psi)*np.cos(phi) + np.sin(theta)*np.sin(phi)) + r1[1]

        x_prime_vals = np.append(x_prime_vals, x_prime)
        y_prime_vals = np.append(y_prime_vals, y_prime)
        
        d = dpsi(ds, theta, phi, psi, z_prime, a)
        if counter > 5000:
            if d < np.pi/steps*10**-3*1.01:
                print(f'd = {d}')
        if anticlockwise:            
            psi += d
        else:
            psi-=d
        if d < np.pi/steps*10**-3 and anticlockwise:
            anticlockwise=False
            if iteration=='first':
                psi=0
                continue
            elif iteration=='second':
                psi=np.pi
                iteration='stop'
                continue
        if d < np.pi/steps*10**-3 and not anticlockwise:
            if iteration=='stop':
                break
            iteration = 'second'
            anticlockwise=True
            psi=np.pi
            continue
        
        if np.abs(psi) > 2*np.pi:
            return x_prime_vals, y_prime_vals, ds
    # print(f'counter = {counter}')
    return x_prime_vals, y_prime_vals, ds
    
def binary_dilate(image, iterations):
    dilated_image = ndimage.binary_dilation(image, iterations=iterations)
    dilated_image = np.array(dilated_image, dtype=float)
    return dilated_image

def binary_erode(image, iterations):
    eroded_image = ndimage.binary_erosion(image, iterations=iterations, border_value=0)
    eroded_image = np.array(eroded_image, dtype=float)
    return eroded_image

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

def give_x_y_for_two_points(r1, r2, z_prime, alpha, steps, estimate, ds):
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
    x, y, ds = x_prime_y_prime_output(z_prime, theta, phi, alpha, steps, r1, estimate, ds)
    # print(x, y)
    return x, y, ds

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

def calculate_heatmap_old(x, y, bins=50):
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

def calculate_heatmap(x, y, bins=50, dilate_erode_iterations=5, ZoomOut=0):
    '''
    Calculate heatmap and its extent using np.histogram2d() from x and y values for a given 
    number of bins.
    Parameters
    ----------
    x : numpy_array containg arrays of x points for each cone
        Must be a numpy array, not a list!
    y : numpy_array containg arrays of x points for each cone
        DESCRIPTION.
    bins : TYPE, optional
        DESCRIPTION. The default is 50.
    dilate_erode_iterations : number of iterations that are carried out for the dilation and erodation
    Returns
    -------
    heatmap : numpy_array
        A numpy array of the shape (bins, bins) containing the histogram values: x along axis 0 and
        y along axis 1.
    extent : TYPE
        DESCRIPTION.d = 
    '''
    xtot = np.hstack(x)
    ytot = np.hstack(y)
    h_, xedges_, yedges_ = np.histogram2d(xtot, ytot, bins)
    pixel_size_x = abs(xedges_[0] - xedges_[1])
    pixel_size_y = abs(yedges_[0] - yedges_[1])
    print(f'pixel_size_x = {pixel_size_x}')
    print(f'pixel_size_y = {pixel_size_y}')
    extend_x = 5*dilate_erode_iterations*pixel_size_x #might need to replace 5* with less, eg 2
    extend_y = 5*dilate_erode_iterations*pixel_size_y
    y_bins = int(round((pixel_size_y/pixel_size_x)*bins))
    print(f'y_bins = {y_bins}')
    print(f'type(y_bins) = {type(y_bins)}')
    h, xedges, yedges = np.histogram2d(xtot, ytot, [bins, y_bins], range=[[xedges_[0]- extend_x, xedges_[-1] + extend_x], [yedges_[0] - extend_y, yedges_[-1] + extend_y]])
    
    pixel_size_x = abs(xedges[0] - xedges[1])
    pixel_size_y = abs(yedges[0] - yedges[1])
    print(f'pixel_size_x = {pixel_size_x}')
    print(f'pixel_size_y = {pixel_size_y}')
    # np.where(xtot<extent[0], 0, xtot)
    # np.where(xtot>extent[1], 0, xtot)
    # np.where(ytot<extent[2], 0, ytot)
    # np.where(ytot>extent[3], 0, ytot)

    heatmaps = []
    for i in range(len(x)):
        hist = np.histogram2d(x[i], y[i], np.array([xedges, yedges]))[0]
        hist[hist != 0] = 1
        if dilate_erode_iterations>0:
            hist = binary_erode(binary_dilate(hist, dilate_erode_iterations), dilate_erode_iterations)
        heatmaps.append(hist)
    heatmap = np.sum(heatmaps, 0)
    plot_heatmap(heatmap, np.array([xedges[0], xedges[-1], yedges[0], yedges[-1]]), bins, y_bins, n_points='no chop')
    
    chop_indices, ind = image_slicer(heatmap, ZoomOut)
    print(f'[xedges[ind[0]], yedges[ind[1]]] = {[xedges[ind[0]], yedges[ind[1]]]}')
    print(f'chop_indices = {chop_indices}')
    print(chop_indices[0])
    # x_chop = xedges[chop_indices[0]+1:chop_indices[1]]
    # y_chop = yedges[chop_indices[2]+1:chop_indices[3]]
    
    x_chop = xedges[chop_indices[0]+1], xedges[chop_indices[1]]
    y_chop = yedges[chop_indices[2]+1], yedges[chop_indices[3]]
    bins2 = 100
    print(f'y_bins = {y_bins}', f', x_bins = {bins}')
    heatmaps2 = []
    for i in range(len(x)):
        hist = np.histogram2d(x[i], y[i], bins2, range=np.array([x_chop, y_chop]))[0]
        hist[hist != 0] = 1
        if dilate_erode_iterations>0:
            hist = binary_erode(binary_dilate(hist, dilate_erode_iterations), dilate_erode_iterations)
        heatmaps2.append(hist)
    heatmap2 = np.sum(heatmaps2, 0)
    
    extent = np.array([x_chop[0], x_chop[-1], y_chop[0], y_chop[-1]])
    plot_heatmap(heatmap[chop_indices[0]+1:chop_indices[1], chop_indices[2]+1:chop_indices[3]], extent, bins, y_bins, n_points='chopped')
    return heatmap2, extent, bins, y_bins

def plot_heatmap(heatmap, extent, bins, y_bins, n_points):
    '''Plot a heatmap using plt.imshow().'''
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.colorbar()
    plt.title(f'bins, y_bins, points = {bins, y_bins, n_points}')
    plt.show()

def image_slicer(h, ZoomOut=0):
    ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    h[h < 0.5*np.amax(h)] = 0
    chop_indices = np.arange(4)
    for i in range(np.shape(h)[0]):
        if np.sum(h[ind[0]-i]) == 0:
            chop_indices[0] = ind[0] - (i+ZoomOut)
            break
    for i in range(np.shape(h)[0]):
        if np.sum(h[ind[0]+i]) == 0:
            chop_indices[1] = ind[0] + (i+ZoomOut)
            break
    for i in range(np.shape(h)[0]):
        if np.sum(h.T[ind[1]-i]) == 0:
            chop_indices[2] = ind[1] - (i+ZoomOut)
            break
    for i in range(np.shape(h)[0]):
        if np.sum(h.T[ind[1]+i]) == 0:
            chop_indices[3] = ind[1] + (i+ZoomOut)
            break
        
    return chop_indices, ind

def get_image(points, n, estimate, image_distance, source_energy, bins, R, steps=180, plot=True, ZoomOut=0):
    '''
    Parameters
    ----------
    points : TYPE - array
        DESCRIPTION - each item in the array is an array-like object consisting of [r1, r2, dE] where
        r1 is an array of the coordinates of the hit in the Compton detector (x, y, z) and r2 the absorbing detector, dE is energy loss
    n : TYPE - integer
        DESCRIPTION - number of angles to iterate through in alpha_bounds 
    estimate : TYPE - float
        DESCRIPTION - estimate of z distance of source from Compton detector
    image_distance : TYPE - float
        DESCRIPTION - distance of imaging plane from the Compton detector
    source_energy : TYPE - float
        DESCRIPTION - energy of the source in eV
    bins : TYPE - integer
        DESCRIPTION - number of bins to construct heatmap
    R : TYPE - float
        DESCRIPTION - resolution of the detector
    steps : TYPE - integer
        DESCRIPTION - approximate number of steps to take in psi to calculate cone projections 
    plot : TYPE - boolean
        DESCRIPTION - plots heatmap if set to True

    Returns
    -------
    heatmap: A numpy array of the shape (bins, bins) containing the histogram values: x along axis 0 and
        y along axis 1.

    '''
    n_points = 400
    if n_points > np.shape(points)[0]:
        n_points = np.shape(points)[0]
            
    x_list = []
    y_list = []
    j = 0
    ds=0
    for point in points[:n_points]:
        xs2 = np.array([])
        ys2 = np.array([])
        # print(source_energy-point[6])
        alpha = compton_angle(source_energy, source_energy-point[6])
        # print(alpha)
        Ef = source_energy - point[6]
        Ef = Ef*e
        r1 = np.array([point[0], point[1], point[2]])
        r2 = np.array([point[3], point[4], point[5]])
        # print(f'alpha={alpha}')
        theta = theta_angle(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])
        if theta+alpha < np.pi/2:
            j+=1
        if theta + alpha >= np.pi/2-0.001:
            if j < 1: #if an ellipse hasn't already been plotted, don't plot a parabola (no accurate ds)
                continue
            else:
                pass
        if R>0:
            alpha_err = (R*m_e*c**2) / (2.35*np.sin(alpha)*Ef)
            # print(f'alpha_err is {alpha_err}')
            alpha_min = alpha-alpha_err
            alpha_max = alpha+alpha_err
            if alpha_min < 0:
                alpha_min = 0
            if alpha_max >= np.pi/2:
                alpha_max = (np.pi/2)-0.01
            alpha_bounds = np.linspace(alpha-alpha_err, alpha+alpha_err, num=n)
            for angle in alpha_bounds:
                # print(f'r1={r1}')
                # print(f'r2={r2}')
                x, y, ds = give_x_y_for_two_points(r1, r2 , z_prime=image_distance, alpha=angle, steps=steps, estimate=estimate, ds=ds)
                xs2 = np.append(xs2, x, axis=0)
                ys2 = np.append(ys2, y, axis=0)
                x_list.append(xs2)
                y_list.append(ys2)
        else:
            # print(f'r1={r1}')
            # print(f'r2={r2}')
            x, y, ds = give_x_y_for_two_points(r1, r2 , z_prime=image_distance, alpha=alpha, steps=steps, estimate=estimate, ds=ds)
            xs2 = np.append(xs2, x, axis=0)
            ys2 = np.append(ys2, y, axis=0)
            x_list.append(xs2)
            y_list.append(ys2)
 
    if R>0:
        heatmap_combined, extent_combined, bins, y_bins  = calculate_heatmap(x_list, y_list, bins=bins, ZoomOut=ZoomOut)
    else:
        # Need to not dilate for zero error (perfect resolution: R=0)
        print('R=0')
        heatmap_combined, extent_combined, bins, y_bins  = calculate_heatmap(x_list, y_list, bins=bins, dilate_erode_iterations=0, ZoomOut=ZoomOut)

    if plot is True:
        plot_heatmap(heatmap_combined, extent_combined, bins, y_bins, n_points)
    
    return heatmap_combined, extent_combined

heatmap, extent = get_image(points, 50, 30, 30, 662E3, 25, R=0, steps=50, ZoomOut=0)
