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

#Function calculating scatter angle from initial and final energy (Joules)
def compton_angle(E_initial, E_final):
    E_initial = E_initial*e
    E_final = E_final*e
    initial = h*c/E_initial
    final = h*c/E_final
    angle = np.arccos(1 - (m_e*c/h)*(final-initial))
    return angle