# NOTE: THIS ALG HAS BEEN REPLACED WITH A PURE PYTHON VERSION. 
# THIS CODE IS KEPT FOR REFERENCE ONLY

import numpy as np
cimport numpy as np #cimport imports pxd files
import cython 

DTYPE = np.float32
ctypedef np.float32_t DTYPE_f
ctypedef np.int16_t DTYPE_s

cdef inline float maxf(float a, float b): return a if a >= b else b
cdef inline short maxs(short a, short b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def get_mask_with_stent_likely_positions_float(np.ndarray[DTYPE_f,ndim=3] data, float th1):
    """ get_mask_with_stent_likely_positions(data)
    Detect seed points on the stents subject to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    Returns a mask, which can easily be turned into a set of points by 
    detecting the voxels with the value 2.
    """
    
    # Create mask
    shape = np.shape(data)
    cdef np.ndarray[np.uint8_t, ndim=3] mask = np.zeros(shape,dtype=np.uint8)
    
    # Init help variables
    cdef int x, y, z, dx, dy, dz
    cdef int xmax, ymax, zmax
    zmax = data.shape[0]
    ymax = data.shape[1]
    xmax = data.shape[2]
    
    cdef DTYPE_f theMax
    cdef char maskMax
    
    # Loop!
    for z in range(1,zmax-1):
        for y in range(1,ymax-1):
            for x in range(1,xmax-1):
                
                theMax = -99999.0
                #
               #theMax = maxf( theMax, data[z  ,y  ,x  ]  )
                theMax = maxf( theMax, data[z-1,y  ,x  ]  )
                theMax = maxf( theMax, data[z+1,y  ,x  ]  )
                theMax = maxf( theMax, data[z  ,y-1,x  ]  )               
                theMax = maxf( theMax, data[z-1,y-1,x  ]  )
                theMax = maxf( theMax, data[z+1,y-1,x  ]  )
                theMax = maxf( theMax, data[z  ,y+1,x  ]  )
                theMax = maxf( theMax, data[z-1,y+1,x  ]  )
                theMax = maxf( theMax, data[z+1,y+1,x  ]  )
                #
                theMax = maxf( theMax, data[z  ,y  ,x-1]  )
                theMax = maxf( theMax, data[z-1,y  ,x-1]  )
                theMax = maxf( theMax, data[z+1,y  ,x-1]  )
                theMax = maxf( theMax, data[z  ,y-1,x-1]  )                
                theMax = maxf( theMax, data[z-1,y-1,x-1]  )
                theMax = maxf( theMax, data[z+1,y-1,x-1]  )
                theMax = maxf( theMax, data[z  ,y+1,x-1]  )
                theMax = maxf( theMax, data[z-1,y+1,x-1]  )
                theMax = maxf( theMax, data[z+1,y+1,x-1]  )
                #
                theMax = maxf( theMax, data[z  ,y  ,x+1]  )
                theMax = maxf( theMax, data[z-1,y  ,x+1]  )
                theMax = maxf( theMax, data[z+1,y  ,x+1]  )
                theMax = maxf( theMax, data[z  ,y-1,x+1]  )                
                theMax = maxf( theMax, data[z-1,y-1,x+1]  )
                theMax = maxf( theMax, data[z+1,y-1,x+1]  )
                theMax = maxf( theMax, data[z  ,y+1,x+1]  )
                theMax = maxf( theMax, data[z-1,y+1,x+1]  )
                theMax = maxf( theMax, data[z+1,y+1,x+1]  )
                
                # All criteria, 1 follows from 2 and 3
                if theMax <= data[z,y,x] and theMax > th1:
                    # Before setting the voxel in the mask, make sure
                    # not to set two righ next to each other.                    
                    # Unwrapping the indexes as above would save 0.1 sec
                    # (on 3 secs), but this looks better.
                    if mask[z,y,x] == 0:
                        mask[z-1:z+2, y-1:y+2, x-1:x+2] = 1
                        mask[z,y,x] = 2
    
    return mask


@cython.boundscheck(False)
@cython.wraparound(False)
def get_mask_with_stent_likely_positions_short(np.ndarray[DTYPE_s,ndim=3] data, float th1):
    """ get_mask_with_stent_likely_positions(data)
    Detect seed points on the stents subject to three criteria:
      * intensity above given threshold
      * local maximum
      * at least one neighbour with intensity above threshold
    Returns a mask, which can easily be turned into a set of points by 
    detecting the voxels with the value 2.
    """
    
    # Create mask
    shape = np.shape(data)
    cdef np.ndarray[np.uint8_t, ndim=3] mask = np.zeros(shape,dtype=np.uint8)
    
    # Init help variables
    cdef int x, y, z, dx, dy, dz
    cdef int xmax, ymax, zmax
    zmax = data.shape[0]
    ymax = data.shape[1]
    xmax = data.shape[2]
    
    cdef DTYPE_s theMax
    cdef char maskMax
    
    # Loop!
    for z in range(1,zmax-1):
        for y in range(1,ymax-1):
            for x in range(1,xmax-1):
                
                theMax = -9999
                #
               #theMax = maxs( theMax, data[z  ,y  ,x  ]  )
                theMax = maxs( theMax, data[z-1,y  ,x  ]  )
                theMax = maxs( theMax, data[z+1,y  ,x  ]  )
                theMax = maxs( theMax, data[z  ,y-1,x  ]  )               
                theMax = maxs( theMax, data[z-1,y-1,x  ]  )
                theMax = maxs( theMax, data[z+1,y-1,x  ]  )
                theMax = maxs( theMax, data[z  ,y+1,x  ]  )
                theMax = maxs( theMax, data[z-1,y+1,x  ]  )
                theMax = maxs( theMax, data[z+1,y+1,x  ]  )
                #
                theMax = maxs( theMax, data[z  ,y  ,x-1]  )
                theMax = maxs( theMax, data[z-1,y  ,x-1]  )
                theMax = maxs( theMax, data[z+1,y  ,x-1]  )
                theMax = maxs( theMax, data[z  ,y-1,x-1]  )                
                theMax = maxs( theMax, data[z-1,y-1,x-1]  )
                theMax = maxs( theMax, data[z+1,y-1,x-1]  )
                theMax = maxs( theMax, data[z  ,y+1,x-1]  )
                theMax = maxs( theMax, data[z-1,y+1,x-1]  )
                theMax = maxs( theMax, data[z+1,y+1,x-1]  )
                #
                theMax = maxs( theMax, data[z  ,y  ,x+1]  )
                theMax = maxs( theMax, data[z-1,y  ,x+1]  )
                theMax = maxs( theMax, data[z+1,y  ,x+1]  )
                theMax = maxs( theMax, data[z  ,y-1,x+1]  )                
                theMax = maxs( theMax, data[z-1,y-1,x+1]  )
                theMax = maxs( theMax, data[z+1,y-1,x+1]  )
                theMax = maxs( theMax, data[z  ,y+1,x+1]  )
                theMax = maxs( theMax, data[z-1,y+1,x+1]  )
                theMax = maxs( theMax, data[z+1,y+1,x+1]  )
                
                # All criteria, 1 follows from 2 and 3
                if theMax <= data[z,y,x] and theMax > th1:
                    # Before setting the voxel in the mask, make sure
                    # not to set two righ next to each other.                    
                    # Unwrapping the indexes as above would save 0.1 sec
                    # (on 3 secs), but this looks better.
                    if mask[z,y,x] == 0:
                        mask[z-1:z+2, y-1:y+2, x-1:x+2] = 1
                        mask[z,y,x] = 2
    
    return mask
