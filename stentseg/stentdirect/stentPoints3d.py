""" Module stentPoints3d
Detect points on the stent from a 3D dataset containing the stent.

"""

# Normal imports
import os, sys
import numpy as np
import scipy as sp, scipy.ndimage

from stentseg.utils import PointSet


# # Compile cython
# from pyzolib import pyximport
# pyximport.install()
# from . import stentPoints3d_ 


def get_stent_likely_positions(data, th1):
    """ Get a pointset of positions that are likely to be on the stent.
    If the given data has a "sampling" attribute, the positions are
    scaled accordingly. 
    
    Detection goes according to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    """
    
    # Get mask
    mask = get_mask_with_stent_likely_positions(data, th1)
    
    # Convert mask to points
    indices = np.where(mask==2)  # Tuple of 1D arrays
    pp = PointSet( np.column_stack(reversed(indices)), dtype=np.float32)
    
    # Correct for anisotropy and offset
    if hasattr(data, 'sampling'):
        pp *= PointSet( list(reversed(data.sampling)) ) 
    if hasattr(data, 'origin'):
        # todo: or call it/support offset?
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
    
    This is the pure-Python implementation.
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

