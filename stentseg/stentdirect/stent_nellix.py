# 2014-2016 Maaike Koenrades
"""
Implementation of StentDirect algorithm for the Endurant stent graft

Imports functions needed from stentgraph 
"""

from __future__ import print_function, division, absolute_import

import sys, os, time
import numpy as np
import networkx as nx

import visvis as vv
from visvis import ssdf

from ..utils.new_pointset import PointSet
from ..utils import gaussfun

from . import stentgraph
from .stentgraph import _sorted_neighbours
from .base import StentDirect


SORTBY = 'cost'


class NellixDirect(StentDirect):
    """ An implementation of the StentDirect algorithm targeted at 
    the Nellix stent graft.
    """
    #Todo: to modify as suitable for nellix
    
    def Step1(self):
        """ Step1()
        Detect seed points.
        """
        import random
        print('get mask for seedpoints NELLIX is used')
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        
        t0 = time.time()
        
        # Detect points
        th = self._params.seed_threshold
        pp = get_stent_likely_positions(self._vol, th) # call below
        
        # Create nodes object from found points
        nodes = stentgraph.StentGraph()
        for p in pp:
            p_as_tuple = tuple(p.flat) # todo: perhaps seed detector should just yield list of tuples.
            nodes.add_node(p_as_tuple)
        
        t1 = time.time()
        if self._verbose:
            print()
            print('Found %i seed points, which took %1.2f s.' % (len(nodes), t1-t0))
        
        # Store the nodes
        self._nodes1 = nodes
        
        # Draw?
        if self._draw:
            self.Draw(1)
        
        return nodes


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
    
    # Criterium 1A: voxel must be above th
    # Note that we omit the edges
    mask[25:-25,25:-25,25:-25] = (data[25:-25,25:-25,25:-25] > th[0]) * 3
    
    cnt = 0
    seed = None
    seeds = []
    values = []
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
            if themax <= th[0]:
                continue
            
            # Criterium 1B: voxel must be below upper seed th, if given
            if len(th) ==2:
                if val > th[1]:
                    print('Seed removed by higher th: ',(z,y,x),'ctvalue=', val)
                    continue
            
            # Criterium 4: seed must be at least 5 voxels away from other seeds
            if not seed == None:
                newseed = np.asarray([z,y,x])
                v = seeds - newseed
                d = (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 # np.linalg.norm(v) # magnitude
                if d.min() < 5:
                    cnt+=1
                    continue
            seed = np.asarray([z,y,x])
            seeds.append(seed)
            
            # Set, and suppress stent points at direct neighbours
            mask[z-1:z+2, y-1:y+2, x-1:x+2] = 1
            mask[z,y,x] = 2
            values.append(data[z,y,x])
    
    print()
    print('Seed ctvalues: {}'.format(sorted(values)))
    print('-------')
    print('Seeds removed by criterium 4: {}'.format(cnt))
    
    return mask
    
    
    
    