""" Package mcp
Functions that implement Minimum Cost Path algorithms.

The Minimum Cost Path Finder is a class that applies the
minimum cost path algorithm to calculate a cost map and calculate
the minimum cost between the seed point and another.     
    
"""

import numpy as np
#import scipy as sp, scipy.ndimage
from visvis import Point, Pointset, Aarray

import time
import visvis as vv

# Compile cython if needed
from pyzolib import pyximport
pyximport.install()
import heap
pyximport.install()
import mcpx

from mcpx import (McpBase, McpSimple, McpDistance, McpWithEndPoints,
                    McpConnectedSourcePoints)


class McpVisual(McpSimple):
    """ McpVisual(costs, sourcePoints)
    An MCP object that visualizes the evolution of the front,
    using the original EvolveFront algorithm.
    """
    
    def Draw(self):
        self._t1.Refresh()
        data = self._t2._texture1._dataRef
        self._t2.SetClim(0,data[data<np.inf].max()*1.1)        
        self._t2.Refresh()
        self._f.DrawNow()        
        if self._stepsize<1:
            time.sleep(self._stepsize)
    
    
    def EvolveFront(self, stepsize=5, im2show=None):
        """ Calculate the map of cumulative costs  to go from the seed point 
        to any point in the grid. 
        If costs is an anisotropic array (Aarray), the anisotropy is taken
        into account in calculating the transition costs.
        
        im2show is the grayscale image to show (costs is used when None)
        stepsize is the amount of steps to do before visualizing. If smaller 
        than 1, it represents the pause (in seconds) to wait between two 
        iterations.        
        """
        
        # init show image
        if im2show is None:
            im2show = self.costs
        shape = im2show.shape
        self.im2showmax = im2show.max()
        if isinstance(im2show, Aarray):
            tmp = im2show.sampling[0], im2show.sampling[1], 1
            im2show_rgb = Aarray((shape[0], shape[1], 3), tmp, dtype=np.float32)
        else:
            im2show_rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        im2show_rgb[:,:,0] = im2show
        im2show_rgb[:,:,1] = im2show
        im2show_rgb[:,:,2] = im2show
        self._im2show_rgb, self._im2show = im2show_rgb, im2show
        # some visualization params
        self._cfactor = 0.2
        self._drawIter = 0
        self._curcord = None
        self._stepsize = stepsize
        
        # init visualization
        f = vv.figure(100);
        #
        a1 = vv.subplot(121)    
        t1 = vv.imshow( im2show_rgb, axes=a1 )
        a2 = vv.subplot(122)    
        t2 = vv.imshow( self.cumCosts, axes=a2 )
        a1.showAxis = 0; a2.showAxis = 0
        #
        time.sleep(0.5)
        self._f, self._t1, self._t2 = f, t1, t2
        
        # make it stoppable
        a1.eventMouseDown.Bind(self._OnClick)
        self._clicked = False
        
        # go
        self.Reset()
        while McpSimple.EvolveFront(self, 1) and not self._clicked:
            pass
        
    
    def _OnClick(self, event):
        self._clicked = True
    
    
    def GoalReached(self, ii, cumCost1):
        """ Not check goal, but draw stuff! """
        
        # get stuff
        cfactor, drawIter, curcord = self._cfactor,self._drawIter,self._curcord
        im2show_rgb, im2show = self._im2show_rgb, self._im2show
        
        # reset
        if curcord:
            tmp1, tmp2 = im2show[curcord], cfactor * self.im2showmax
            im2show_rgb[curcord[0],curcord[1],0] = tmp1 - tmp2
            im2show_rgb[curcord[0],curcord[1],1] = tmp1 + tmp2
            im2show_rgb[curcord[0],curcord[1],2] = tmp1 - tmp2
        # currentpos
        curcord = self.MakeTuplePos(ii)
        if self._stepsize<1:                
            im2show_rgb[curcord[0],curcord[1],0] = self.im2showmax
            im2show_rgb[curcord[0],curcord[1],1] = 0
            im2show_rgb[curcord[0],curcord[1],2] = 0
        # front            
        #for i in self.front.GetReferences():
        for i in self.front.references():
            y,x = self.MakeTuplePos(i)
            tmp = im2show[y,x]
            im2show_rgb[y,x,0] = tmp - cfactor * self.im2showmax
            im2show_rgb[y,x,1] = tmp - cfactor * self.im2showmax
            im2show_rgb[y,x,2] = tmp + cfactor * self.im2showmax            
        # update visualization?
        drawIter += 1
        if self._stepsize<1 or (drawIter >= self._stepsize):
            self._t1.SetClim()
            drawIter = 0
            self.Draw()
            
        # done
        self._cfactor,self._drawIter,self._curcord = cfactor, drawIter, curcord
        return 0
    
  



#     # get a mask for voxels that are a local max
#     localmin = sp.ndimage.minimum_filter(self._costs,3)
#     localmin = (self._costs == localmin).flat




