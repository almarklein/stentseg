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
from visvis.pypoints import Point, Pointset, Aarray
from visvis.pypoints import is_Aarray

#import diffgeo
# Not sure were I'll put the gfilter functionality, pirt?
diffgeo = None


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


def thin(data):
    """ thin(data) 
    Thin the given binary data, converting all blobs of voxels to single
    voxels if they are max 3x3(x3). Larger blobs are discarted.
    
    This can be used to get rid of local maxima close together...
    """
    # create strels
    strel1 = np.array([0,1,1])
    strel2 = np.array([1,1,0])
    
    # preform reduction of size
    for i in range(1):
        for i in range(data.ndim):
            # make shapes right
            shape = [1 for tmp in data.shape]; shape[i]=3
            strel1.shape = tuple(shape)
            strel2.shape = tuple(shape)
            # apply "erosion"
            data2 = sp.ndimage.binary_hit_or_miss(data, strel1)    
            data = data-data2
            data2 = sp.ndimage.binary_hit_or_miss(data, strel2)
            data = data-data2
    
    # keep only singleton pixels
    strel3 = np.zeros([3 for tmp in data.shape])
    index = tuple( [1 for tmp in data.shape] ) 
    strel3[index] = 1
    data = sp.ndimage.binary_hit_or_miss(data,strel3)
    return data


def getStentSurePositions_wrong(data, th1, th2=0):
    """ getStentSurePositions(data, th)
    Based on three measures, establish positions where 
    we can be sure to have a stent's wire:
    - intensity above given threshold
    - local maximum
    - at least one neighbour with intensity above threshold
    
    Another nice feature is that these points will also often be found
    on (in the centre of) crossings.
    """
    
    # ===== THE DISCREPANCY WITH THE CYTHON IMPLEMENTATION ====
    # The Cython implementation finds slightly less points. Due to the extra
    # points found by the Python implementation, there are considerably more
    # edges in the end result, which are mostly to be on the spinal cord.
    # (Use the piece of code in the demo of  stentDirect.py that removes
    # most of these to see the difference.)
    # With the Cython implementation, the method scores a few per cent better
    # for the Aneurx.
    # 
    # Number of seedpoints in different situations
    # Python on Zenith: 1789
    # Cython on Zenith: 1732
    # Python on Aneurx: 3007  /w cleanup: 1187
    # Cython on Aneurx: 2873  /w cleanup: 1156
    # Python without thinning: 3043
    # Cython without thinning: 3043
    # --> It seems the thinning operation of Python does not work very well
    
    # todo: thresholds
    # - a threshold of 1000 works for pat01
    # - but for pat12 around 700 is better
    # - 500 works allright also
    # - but 500 on pat01 will find many point in contrast fluid
    # - pat13 needs th1<=500
    
    sampling = (1,1,1)
    if is_Aarray(data):
        sampling = data.sampling
    
    # calculate Laplacian
    if th2:
        sigma = 0.8 # in mm
        # take anisotropy into account
        sigmas = [sigma/sam for sam in sampling]
        Lxx = diffgeo.gfilter(data, sigmas, [0,0,2])
        Lyy = diffgeo.gfilter(data, sigmas, [0,2,0])
        Lzz = diffgeo.gfilter(data, sigmas, [2,0,0])
        Lap = -(Lxx + Lyy + Lzz)
        #Lap = Aarray( Lap, sampling)
    
    # init pointset
    pp = Pointset(data.ndim) 
    
    # create localmax array
    localmax = data.copy()
    localmax[data<th1] = 0    
    kernel = np.ones(3**data.ndim); kernel[(kernel.shape[0]-1)/2] = 0
    kernel.shape = {2:(3,3), 3:(3,3,3)}[pp.ndim]
    localmax = sp.ndimage.maximum_filter(localmax,footprint=kernel)
    
    # get voxels that are ok
    if th2:
        mask = (    (data>0) & # min th
                    ((data-localmax)>=0) & # is-local-max
                    (localmax>th1) & # data larger than th1
                    (Lap>th2)  ) # Laplacian larger than th2        
    else:
        mask = (    (data>0) & # min th
                    ((data-localmax)>=0) & # is-local-max
                    (localmax>th1)  ) # larger than th
    
    # reduce groups to single pixel
    #mask = thin(mask)
    
    # convert mask to points
    indices = np.where(mask)
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
    
    # correct for anisotropy
    if is_Aarray(data):
        scale = Point( tuple(reversed(data.sampling)) ) 
        if hasattr(data, 'get_start'):            
            offset = data.get_start()
        else:
            offset = data.get_start()
        pp = pp * scale + offset
    
    # done
    return pp