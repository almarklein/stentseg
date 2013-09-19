""" Module stentPoints3d
Detect points on the stent from a 3D dataset containing the stent.

"""

# Compile cython
from pyzolib import pyximport
pyximport.install()
from . import stentPoints3d_ 

# Normal imports
import os, sys
import numpy as np
import scipy as sp, scipy.ndimage
from visvis import Point, Pointset, Aarray
from visvis.utils.pypoints import is_Aarray


from stentseg import gaussfun


def getStentSurePositions(data, th1):
    
    # Init pointset
    pp = Pointset(data.ndim) 
    
    # Get mask
    if data.dtype == np.float32:
        mask = stentPoints3d_.getMaskWithStentSurePositions_float(data, th1)
    elif data.dtype == np.int16:
        mask = stentPoints3d_.getMaskWithStentSurePositions_short(data, th1)
    else:
        raise ValueError('Data must be float or short.')
    
    # Convert mask to points
    indices = np.where(mask==2)
    if pp.ndim==2:
        Iy, Ix = indices
        for i in range(len(Ix)):
            pp.append(Ix[i], Iy[i])
    elif pp.ndim==3:
        Iz, Iy, Ix = indices
        for i in range(len(Ix)):
            pp.append(Ix[i], Iy[i], Iz[i])
    else:
        raise ValueError('Can only handle 2D and 3D data.')
    
    # Correct for anisotropy
    if is_Aarray(data):
        scale = Point( tuple(reversed(data.sampling)) ) 
        if hasattr(data, 'get_start'):            
            offset = data.get_start()
        else:
            offset = data.get_start()
        pp = pp * scale + offset
    
    # Done
    return pp




def detect_stent_points_3d(data, th):
    
    mask = np.zeros_like(data, np.uint8)
    
    # Criterium 1: voxel must be above th
    # Note that we omit the edges
    mask[1:-1,1:-1,1:-1] = (data[1:-1,1:-1,1:-1] > th) * 3
    
    
    for z, y, x in zip(*np.where(mask==3)):
        
        # Only proceed if this voxel is "free"
        if mask[z,y,x] == 3:
            
            # Set to 0 initially
            mask[z,y,x] = 0  
            
            # Get value
            val = data[z,y,x]
            
            # Get maximum of neighbours
            patch = data[z-1:z+2, y-1:y+2, x-1:x+2].copy()
            patch[1,1,1] = 0
            themax = patch.max()
            
            # Criterium 2: must be local max
            if themax > val:
                continue
            # Also ensure at least one neighbour to be *smaller*
            if (val > patch).sum() == 0:
                continue
            
            # Criterium 3: one neighbour must be above th
            if themax <= th:
                continue
            
            # Set, and suppress stent points at direct neightbours
            mask[z-1:z+2, y-1:y+2, x-1:x+2] = 1
            mask[z,y,x] = 2
    
    return mask

