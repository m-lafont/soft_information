# -*- coding: utf-8 -*-
"""A set of functions that make it easy to play with soft information"""

import numpy as np
from numpy import pi, exp, sqrt, log
from math import atan 
from math import atan2



def normal(x, mean, std):
    """
    Normal distribution

    Args:
        x: input value
        mean: the mean of the distribution
        std: standard deviation 
    """
    return 1/(std*sqrt(2*pi))*exp(-(x-mean)**2/(2*std**2))


def dist(xa, ya, xb, yb):
    """Return the distance between a and b"""
    a = np.array((xa, ya))
    b = np.array((xb, yb))
    return np.linalg.norm(a-b)

#def angle(xa, ya, xb, yb, xc, yc): 
#    """returns the angle of the points a, b and c"""
#    vector1=np.array((xa-xb,ya-yb))
#    vector2=np.array((xc-xb,yc-yb))
#    unitvector1=vector1/np.linalg.norm(vector1)
#    unitvector2=vector2/np.linalg.norm(vector2)
#    prodscal=np.dot(unitvector1,unitvector2)
#    if prodscal<-1:
#        prodscal=-1
#    elif prodscal>1:
#        prodscal=1
#    return np.arccos(prodscal)*180/pi

#def velocity(xa,ya,xb,yb):
#    L=get_time()
#    n=len(L)
#    delta_t= L[n-1]-L[n-2]
#    return dist(xa,ya,xb,yb)/delta_t


def log_normal(x, mean, std):
    """
    Natural logarithm of a normal distribution

    Args:
        x: input value
        mean: the mean of the distribution
        std: standard deviation 
    """
    return -log(std) - (log(2) + log(pi))/2 - (x-mean)**2/(2*std**2)

def logprob_angle(xa, ya, xb, yb,ha,measured_angle, std):
    """logprob that a, b and c are in (xa,ya),(xb,yb),(xc,yc) under the measured angle.
    
    Args:
        xa: abscissa of point a
        ya: ordinate of point a
        xb: abscissa of point b
        yb: ordinate of point b
        measured_dist: measured distance between a and b
        std: standard deviation of the measurement"""
    
#    if xb>xa:
#        angle= atan2((yb-ya)/(xb-xa))-ha
#        
#    else : 
#        angle= np.pi-atan((yb-ya)/(xb-xa))-ha
    angle= atan2((yb-ya),(xb-xa))-ha
    return log_normal(angle, measured_angle, std)

def logprob_distance(xa, ya, xb, yb, measured_dist, std):
    """
    Logprob that a and b are in (xa, ya), (xb, yb) under the measured distance.

    Args:
        xa: abscissa of point a
        ya: ordinate of point a
        xb: abscissa of point b
        yb: ordinate of point b
        measured_dist: measured distance between a and b
        std: standard deviation of the measurement
    """
    points_dist = dist(xa, ya, xb, yb)
    return log_normal(points_dist, measured_dist, std)

#def logprob_velocity(xa,ya,xb,yb,mesured_velocity,std):
#    """
#    
#    Args:
#    mesured_velocity: mesured velocity of the agent
#    xa: abscissa of previous point 
#    ya: ordinante of previous point
#    xb: abscissa of previous point 
#    yb: ordinate of previous point
#    
#    """
#    points_velocity= velocity(xa,ya,xb,yb)
#    return log_normal(points_velocity,mesured_velocity,std)
#        


def make_logprob_angle(idx_a, idx_b, measured_angle, std):
    """
    Make the function that return the logprob of positions under the measured distance

    Args:
        idx_a: index of point a
        idx_b: index of point b
        idx_c : index of point c
        measured_angle : measured angle between a b and c
        std: standard deviation of the measurement
    """
    def func(points):
        """
        Return the logprob of positions under the measured distance

        Args:
            points: estimated positions ([[x0, y0, h0], [x1, y1, h1], ...])
        """
        xa, ya, ha = points[idx_a]
        xb, yb, hb = points[idx_b]
        return logprob_angle(xa, ya, xb, yb,ha,measured_angle, std)

    return func
    
    
def make_logprob_distance(idx_a, idx_b, measured_dist, std):
    """
    Make the function that return the logprob of positions under the measured distance

    Args:
        idx_a: index of point a
        idx_b: index of point b
        measured_dist: measured distance between a and b
        std: standard deviation of the measurement
    """
    def func(points):
        """
        Return the logprob of positions under the measured distance

        Args:
            points: estimated positions ([[x0, y0, h0], [x1, y1, h1], ...])
        """
        xa, ya, ha = points[idx_a]
        xb, yb, hb = points[idx_b]
        return logprob_distance(xa, ya, xb, yb, measured_dist, std)

    return func

def make_logprob_vitesse(pos,posprecedent,delta_t):
#    delta_t= liste_temps[-1]-liste_temps[-2]
#    print(delta_t)
    x=pos[0]
    y=pos[1]
    x_precedent=posprecedent[0]
    y_precedent=posprecedent[1]
    distance = dist(x,y,x_precedent,y_precedent)
    vitesse = distance/delta_t
    if vitesse <= 6 :
        return log(1/7)
    if vitesse > 6 and vitesse <= 12: 
        return ((1/7)*exp(-vitesse+6))
    else : 
        return 0
        
    


def make_logprob_position(idx, measured_x, measured_y, measured_h, std):

    """
    Returns the soft position information function that can be applied to the estimated positions

    Args:
        idx: index of point
        measured_x: measured abscissa
        measured_y: measured ordinate
        std: standard deviation of the measurement

    Returns:
        (function): the logSI function that takes the estimated positions of the points as input.
    """
    def func(points):
        """
        Returns log-prob of the estimated positions under the measured position.

        Args:
            points: estimated positions ([[x0, y0, h0], [x1, y1, h1], ...])
        """
        estimated_x, estimated_y, estimated_h = points[idx]
        return logprob_distance(estimated_x, estimated_y, measured_x, measured_y, 0, std)

    return func

def make_logprob_cap(idx, measured_h, std):

    """
    Returns the soft position information function that can be applied to the estimated positions

    Args:
        idx: index of point
        measured_x: measured abscissa
        measured_y: measured ordinate
        std: standard deviation of the measurement

    Returns:
        (function): the logSI function that takes the estimated positions of the points as input.
    """
    def func(points):
        """
        Returns log-prob of the estimated positions under the measured position.

        Args:
            points: estimated positions ([[x0, y0, h0], [x1, y1, h1], ...])
        """
        estimated_x, estimated_y, estimated_h = points[idx]
        return logprob_distance(estimated_h, 0, measured_h, 0, 0, std)

    return func
