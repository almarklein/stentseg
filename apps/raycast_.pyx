""" Cython module raycast
High performance functionality for raycasting.
"""

from __future__ import division

# cython specific imports
import numpy as np
cimport numpy as np
import cython 

# Define types
ctypedef np.float32_t FLOAT_T


cdef linearInterp3(data, float x, float y, float z):
    
    # Get typed reference to the data
    cdef np.ndarray[FLOAT_T, ndim=3] D = data
    
    # Get integer pos
    ix = <int>x
    iy = <int>y
    iz = <int>z
    # Get fractional pos
    fx1 = x - ix
    fy1 = y - iy
    fz1 = z - iz
    fx2 = 1-fx1
    fy2 = 1-fy1
    fz2 = 1-fz1
    
    # Sample value from data
    val = 0
    val += D[iz,iy,ix]
    val += fz2 * fy2 * fx2 * D[iz,iy,ix]
    val += fz2 * fy2 * fx1 * D[iz,iy,ix+1]        
    val += fz2 * fy1 * fx2 * D[iz,iy+1,ix]
    val += fz2 * fy1 * fx1 * D[iz,iy+1,ix+1]
    
    val += fz1 * fy2 * fx2 * D[iz+1,iy,ix]
    val += fz1 * fy2 * fx1 * D[iz+1,iy,ix+1]
    val += fz1 * fy1 * fx2 * D[iz+1,iy+1,ix]
    val += fz1 * fy1 * fx1 * D[iz+1,iy+1,ix+1]


#@cython.boundscheck(False)
def mipRay(data, pos, vec):
    """ mipRay(data, pos, vec) -> (valMax, posMax)
    Casts a ray from pos to pos+vec.
    data should be float32 and 3D. All coordinates should be in voxels.
    """
    
    # Get typed reference to the data
    cdef np.ndarray[FLOAT_T, ndim=3] D = data
    
    # Init position vector
    cdef float x = pos.x
    cdef float y = pos.y
    cdef float z = pos.z
    
    # Get step vectors
    cdef int nsteps = int(vec.norm() * 1.73 + 0.5) # 3**05
    cdef float nsteps_ = nsteps
    cdef float dx = vec.x / nsteps_
    cdef float dy = vec.y / nsteps_
    cdef float dz = vec.z / nsteps_
    
    # Define typed variables here
    cdef int ix1, iy1, iz1
    cdef int ix2, iy2, iz2
    cdef float fx1, fy1, fz1 # weight for left voxel
    cdef float fx2, fy2, fz2 # weight for right voxel
    cdef float val
    
    #print x, y 
    
    # Loop
    cdef int step
    cdef float maxval = -99999
    cdef int refstep = 0
    for step in range(nsteps):
        
        # Calculate new position
        x += dx
        y += dy
        z += dz
        
        # Get integer position of voxels on the left on on the right
        ix1 = <int>x
        iy1 = <int>y
        iz1 = <int>z
        ix2 = ix1+1
        iy2 = iy1+1
        iz2 = iz1+1
        # Get fractional pos between left and right voxel
        fx2 = x - ix1
        fy2 = y - iy1
        fz2 = z - iz1
        fx1 = 1-fx2
        fy1 = 1-fy2
        fz1 = 1-fz2
        
        # Sample value from data
        val = 0        
        val += fz1 * fy1 * fx1 * D[iz1,iy1,ix1]
        val += fz1 * fy1 * fx2 * D[iz1,iy1,ix2]        
        val += fz1 * fy2 * fx1 * D[iz1,iy2,ix1]
        val += fz1 * fy2 * fx2 * D[iz1,iy2,ix2]
        #
        val += fz2 * fy1 * fx1 * D[iz2,iy1,ix1]
        val += fz2 * fy1 * fx2 * D[iz2,iy1,ix2]
        val += fz2 * fy2 * fx1 * D[iz2,iy2,ix1]
        val += fz2 * fy2 * fx2 * D[iz2,iy2,ix2]
        
        # If largest so far, update
        if val > maxval:
            maxval = val
            refstep = step
    
    # Done
    fraction = float(refstep) / float(nsteps)
    return maxval, pos+vec*fraction
    