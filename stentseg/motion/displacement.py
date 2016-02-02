""" Module to perform calculations on displacement within cardiac cycle

"""
import numpy as np


def _calculateAmplitude(pointpositions, dim = 'x'):
    """ From a cloud of pointpositions for 1 point (or pointDeforms) (e.g. 10 per cardiac 
    cycle), calculate the largest distance between 2 positions, in xyz, z, y, or x
    """
    dmax = 0.0
    for i in range(len(pointpositions)):
        for j in range(len(pointpositions)):
            v = pointpositions[i] - pointpositions[j]
            if dim == 'xyz':
                d = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            elif dim == 'z':
                d = v[2]
            elif dim == 'y':
                d = v[1]
            elif dim == 'x':
                d = v[0]
            dmax = max(dmax, d)
            if dmax == d: # d is so far largest distance
                p1 = i # remember which positions
                p2 = j
    return dmax, p1, p2


def _calculateSumMotion(pointpositions, dim = 'x'):
    """ From a successive sequence of pointpositions for 1 point (or pointDeforms) (e.g. 10
    per cardiac cycle), calculate the sum of motion for this point, in xyz, z, y, or x
    """
    vectors = []
    npositions = len(pointpositions) # 10 deforms
    for j in range(npositions):
        if j == npositions-1:  # -1 as range starts at 0
            # vector from point at 90% RR to 0%% RR
            vectors.append(pointpositions[j]-pointpositions[0])
        else:
            vectors.append(pointpositions[j]-pointpositions[j+1])
    vectors = np.vstack(vectors)
    if dim == 'xyz':
        d = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5  # 3Dvector length in mm
    elif dim == 'xy':
        d = (vectors[:,0]**2 + vectors[:,1]**2) **0.5  # 2Dvector length in mm
    elif dim == 'z':
        d = abs(vectors[:,2])  # 1Dvector length in mm
    elif dim == 'y':
        d = abs(vectors[:,1])  # 1Dvector length in mm
    elif dim == 'x':
        d = abs(vectors[:,0])  # 1Dvector length in mm    
    dsum = d.sum() # total displacement of a point
    return dsum


def calculateMeanAmplitude(points,pointsDeforms, dim = 'x'):
    """ Calculate the mean amplitude of motion for a set of points (e.g. nodes 
    of model) during a cardiac cycle
    """
    meanAmplitude = []
    for i, point in enumerate(points):
        pointpositions = point + pointsDeforms[i]
        dmax_xyz = _calculateAmplitude(pointpositions, dim='xyz')[0]
        dmax_z = _calculateAmplitude(pointpositions, dim='z')[0]
        dmax_y = _calculateAmplitude(pointpositions, dim='y')[0]
        dmax_x = _calculateAmplitude(pointpositions, dim='x')[0]
        if dim=='xyz':
            meanAmplitude.append(dmax_xyz)
        elif dim=='z':
            meanAmplitude.append(dmax_z)
        elif dim=='y':
            meanAmplitude.append(dmax_y)
        elif dim=='x':
            meanAmplitude.append(dmax_x)
    return np.mean(meanAmplitude), np.std(meanAmplitude), min(meanAmplitude), max(meanAmplitude)

    
    