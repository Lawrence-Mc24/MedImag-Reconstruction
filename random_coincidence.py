# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:15:24 2021

@author: lawre
"""
#change
import numpy as np
import scipy
import random

def random_coincidence(dx1, dy1, dz1, dx2, dy2, dz2, E_source, d_12):
    '''Returns list containing detector 1 hit location, detector 2 hit location and absorbed energy
    in the form [a, b, E] where a and b have the form a = (x, y, z).
    Assumes origin of coordinate axes in the centre of detector 1, with the z-axis coming out of the (square) detector face.
    Detector 2 assumed to be parallel to detector 1 centered at (0, 0, -d_12).
    If z goes from left to right across the page, y goes up the page and x comes out of the page'''
    
    E = round(random.uniform(0, E_source), 5)
    #detector 1
    x1 = round(random.uniform(-dx1/2, dx1/2), 5)
    y1 = round(random.uniform(-dy1/2, dy1/2), 5)
    z1 = round(random.uniform(-dz1/2, dz1/2), 5)
    #detector 2
    x2 = round(random.uniform(-dx2/2, dx2/2), 5)
    y2 = round(random.uniform(-dy2/2, dy2/2), 5)
    z2 = round(random.uniform(-dz2/2, dz2/2), 5)
    
    return [(x1, y1, z1), (x2, y2, z2), E]
    