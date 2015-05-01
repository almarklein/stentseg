# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Module stentpoints3d

Detect points on the stent from a 3D dataset containing the stent.

"""

# Normal imports
import os, sys
import numpy as np
import scipy as sp, scipy.ndimage

from stentseg.utils import PointSet, quadraticfit


def get_stent_likely_positions(data, th):
    """ Get a pointset of positions that are likely to be on the stent.
    If the given data has a "sampling" attribute, the positions are
    scaled accordingly. 
    
    Detection goes according to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    """
    
    # Get mask
    mask = get_mask_with_stent_likely_positions(data, th)
    
    # Convert mask to points
    indices = np.where(mask==2)  # Tuple of 1D arrays
    pp = PointSet( np.column_stack(reversed(indices)), dtype=np.float32)
    
    # Correct for anisotropy and offset
    if hasattr(data, 'sampling'):
        pp *= PointSet( list(reversed(data.sampling)) ) 
    if hasattr(data, 'origin'):
        pp += PointSet( list(reversed(data.origin)) ) 
    
    return pp


def get_mask_with_stent_likely_positions(data, th):
    """ Get a mask image where the positions that are likely to be
    on the stent, subject to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    Returns a mask, which can easily be turned into a set of points by 
    detecting the voxels with the value 2.
    
    This is a pure-Python implementation.
    """
    
    # NOTE: this pure-Python implementation is little over twice as slow
    # as the Cython implementation, which is a neglectable cost since
    # the other steps in stent segmentation take much longer. By using
    # pure-Python, installation and modification are much easier!
    # It has been tested that this algorithm produces the same results
    # as the Cython version.
    
    # Init mask
    mask = np.zeros_like(data, np.uint8)
    
    # Criterium 1: voxel must be above th
    # Note that we omit the edges
    mask[2:-2,2:-2,2:-2] = (data[2:-2,2:-2,2:-2] > th) * 3
    
    
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


def get_subpixel_positions(vol, pp):
    """ Given a set of points, return a new set with the same positions,
    but refined to their subpixel positions based on a quadratic fit.
    """
    # Ensure float32
    pp = pp.astype(np.float32, copy=False)
    
    # Get origin and sampling
    sampling = 1.0, 1.0, 1.0 
    origin = 0.0, 0.0, 0.0
    if hasattr(vol, 'sampling'):
        sampling = vol.sampling
    if hasattr(vol, 'origin'):
        origin = vol.origin
    
    # ... in xyz order
    sampling_xyz, origin_xyz =  np.flipud(sampling), np.flipud(origin)
    
    # Transform from world coordinates to volume indices
    pp1 = ((pp - origin_xyz) / sampling_xyz + 0.25).astype('int32')
    
    # Fit in this domain
    pp2 = np.zeros_like(pp1, np.float32)
    for i in range(pp1.shape[0]):
        x, y, z = pp1[i, :]
        dz, _ = quadraticfit.fitLQ1( vol[z-1:z+2, y, x] )
        dy, _ = quadraticfit.fitLQ1( vol[z, y-1:y+2, x] )
        dx, _ = quadraticfit.fitLQ1( vol[z, y, x-1:x+2] )
        dx = 0.0 if abs(dx) > 1.0 else dx
        dy = 0.0 if abs(dy) > 1.0 else dy
        dz = 0.0 if abs(dz) > 1.0 else dz
        assert abs(dx) < 1 and abs(dy) < 1 and abs(dz) < 1
        pp2[i] = x+dx, y+dy, z+dz
    
    # Transform to world coordinates
    return pp2 * sampling_xyz + origin_xyz
