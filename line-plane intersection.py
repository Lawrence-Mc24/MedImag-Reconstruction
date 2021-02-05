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
