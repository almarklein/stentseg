""" Package stentDirect

Implements a class to perform a step by step approach to segment 
the stent, showing results in between. This is also where the
parameter struct is used to steer the algorithm.

Also has a few cells to show a demo at the end.

Notes on memory usage. On the full 512x512x360 datasets a memory error is
produced. I found that for each patient a 256x256x256 dataset can be 
obtained (by cropping) in which the stent is fully visible. This cropping
has to be done by hand (some attempts to automate this failed), and I 
wrote a nice tool for that. A size of a power of two is preferred, otherwise
I cannot render the data on my laptop (which has an ATI card).

So assuming a dataset of 256x256x256, the memory requirement is:
- 32 MB for the int16 dataset represented in CT-values
- 64 MB for the float32 speed map
- 64 MB for the float32 time map
- 64 MB for the int32 crossref array (in the binary heap)
- 64 MB for the int32 traceback array
- 16 MB for the int8 edge map array (ndindex)
- 64 MB for the int32 idmap
total: 5*64+32+16 = 368 MB

For the detection of seedpoints the memory requirements can be substantial
too, because I use some morphology.



"""

# Imports
import sys, os, time
import numpy as np
import visvis as vv
from visvis.pypoints import Point, Pointset, Aarray
from visvis import ssdf

# This code goes a long way; from before when I fixed a bug in the subtraction
# of pointsets
vv.pypoints.SHOW_SUBTRACTBUG_WARNING = True 

from stentseg import mcp
from . import stentGraph
from . import stentPoints3d


class StentDirect:
    """ StentDirect(vol=None, params=None, draw=False)
    A class to apply the stent segmentation algorithm.
    The algorithm can be applied in three steps or in one go.
    Drawing can optionally be performed.
    """
    
    def __init__(self, vol=None, params=None, draw=False, verbose=True):
        
        self._vol = vol
        self._params = params
        
        self._nodes1 = None
        self._nodes2 = None
        self._nodes3 = None
        
        self._draw = draw
        self._verbose = verbose
        self._tex1 = None
    
    
    def SetVol(self, vol):
        """ setVol(self, vol)
        Set the volume.
        """
        self._vol = vol
    
    
    def SetParams(self, params):
        """ setParams(self, params)
        Set the parameter structure.
        """
        self._params = params
    
    
    def Draw(self, nodesNr=0, drawTex=True, fignr=101, **kwargs):
        # Init visualization
        f=vv.figure(fignr); vv.clf();
        a = vv.gca()
        a.cameraType = '3d'
        a.daspect = 1,1,-1
        a.daspectAuto = False        
        # f.position = -769.00, 342.00,  571.00, 798.00
        #f.position = 269.00, 342.00,  571.00, 798.00
        
        # Draw texture?
        if drawTex:
            self._tex1 = vv.volshow(self._vol)
        
        # Draw nodes?
        nodes = {0:None, 1:self._nodes1, 2:self._nodes2, 3:self._nodes3}[nodesNr]
        if nodes is not None:
            nodes.Draw(**kwargs)            
            a.SetLimits()
    
    
    def Step1(self):
        """ Step1()
        Detect seed points.
        """
        
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        
        t0 = time.time()
        
        # Detect points
        th = self._params.seed_threshold
        pp = stentPoints3d.getStentSurePositions(self._vol,th)
#         pp = stentPoints3d.getStentSurePositions_wrong(self._vol, th)
        
        # Create nodes object from found points
        nodes = stentGraph.StentGraph()
        for p in pp:
            nodes.AppendNode(p)
        
        t1 = time.time()
        if self._verbose:
            print('Found %i seed points, which took %1.2f s.' % (len(nodes), t1-t0))
        
        # Store the nodes
        self._nodes1 = nodes
        
        # Draw?
        if self._draw:
            self.Draw(1)
        
        return nodes
    
    
    def Step2(self):
        """ Step2()
        Find edges using MCP.
        """
        
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        if self._nodes1 is None:
            raise ValueError('Seed points not yet calculated.')
        
        # Get nodes
        nodes = stentGraph.StentGraph()
        nodes.Unpack( self._nodes1.Pack() )
        
        # Create speed image (the devision makes it a float array)
        factor = float( self._params.mcp_speedFactor )        
        speed = 1/2**((self._vol)/factor).astype(np.float32)
        
        costToCtValue = lambda x: np.log2(1.0/x)*factor
        
        # Create MCP object
        th = self._params.mcp_evolutionThreshold
        m = mcp.McpConnectedSourcePoints(speed, nodes, th)

        # Evolve front and trace paths        
        t0 = time.time()
        m.EvolveFront()
        t1 = time.time()
        nodes.ConvertPotentialEdges(m, costToCtValue)
        t2 = time.time()
        
        if self._verbose:
            tmp = 'Found %i edges. Evolving and tracing took %1.2f and %1.2f s.'
            print(tmp % (nodes.CountEdges(), t1-t0, t2-t1))
        
        # Store result
        self._nodes2 = nodes
        
        # Draw?
        if self._draw:
            self.Draw(2)
        
        return nodes
    
    
    def _CleanGraph(self, nodes):
        t0 = time.time()
        nodes.Prune_redundant()        
        nodes.Prune_unconnectedNodes()
        return time.time() - t0
    
    
    def Step3(self):
        """ Step3()
        Process graph to remove unwanted edges.
        """
        
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        if self._nodes2 is None:
            raise ValueError('Edges not yet calculated.')
        
        # Get nodes and params
        nodes = stentGraph.StentGraph()
        nodes.Unpack( self._nodes2.Pack() )
        params = self._params
        
        # Init times        
        t_start = time.time()
        t_clean = 0
        
        # 1 - Pruning edges
        t0 = time.time()        
        ene = params.graph_expectedNumberOfEdges
        nodes.SortEdges()
        nodes.Prune_veryWeak(params.graph_weakThreshold)
        nodes.Prune_weak(ene, params.graph_strongThreshold)
        t1 = time.time() - t0
        
        t_clean += self._CleanGraph(nodes)            
        
        # Trimming in between so little tails won't interfere with corners
        nodes.Prune_trim(params.graph_trimLength)        
        nodes.Prune_smallGroups(params.graph_minimumClusterSize)
        
        # 2 - Remove and insert nodes
        t0 = time.time()
        nodes.Prune_pop()
        nodes.Prune_addCornerNodes()
        t2 = time.time() - t0
        
        t_clean += self._CleanGraph(nodes)     
        
        # 3 - Reposition crossings (not for aneurx!)
        t0 = time.time()
        if params.graph_expectedNumberOfEdges == 2:
            nodes.Prune_repositionCrossings()
        t3 = time.time() - t0
        
        t_clean += self._CleanGraph(nodes)     
        
        # 4- Remove loose ends and small clusters
        t0 = time.time()
        nodes.Prune_pop() # does not pop inserted nodes
        nodes.Prune_trim(params.graph_trimLength)
        nodes.Prune_smallGroups(params.graph_minimumClusterSize)
        nodes.Prune_unconnectedNodes()
        t4 = time.time() - t0
        
        # 5 - Finishing
        nodes.SmoothPaths()
        t_total = time.time()-t_start
        
        # Notify times  
        if self._verbose:      
            tmp = "Reduced to %i edges, "
            tmp += "which took %1.2f s: cleaning %1.2f, pruning %1.2f, repos. %1.2f"
            print(tmp % (nodes.CountEdges(), t_total, t_clean, t1, t2+t3))
        
        # Store result
        self._nodes3 = nodes
        
        # Draw?
        if self._draw:
            self.Draw(3)
        
        return nodes
    
    
    def Go(self):
        """ Go()
        Perform all.
        """
        self.Step1()
        self.Step2()
        return self.Step3()
        

# Get default params
def getDefaultParams(stentType=''):
    """ getDefaultParams()
    Get the paramater stuct filled with defaults.
    """
    
    # Generic params
    params = ssdf.new()
    # The threshold for detecting seed points
    params.seed_threshold = 650 
    # The scale factor for the data to create speed image 
    params.mcp_speedFactor = 100         
    # The MCP threshold. Small=faster, but will miss connections if too small!   
    params.mcp_evolutionThreshold = 0.06    
    # The Expected Number of Connections
    params.graph_expectedNumberOfEdges = 2  
    # The th to determine a really strong connection
    params.graph_strongThreshold = 1200     
    # The th to determine a really weak connection
    params.graph_weakThreshold = 100        
    # The size of tails to trim and clusters to remove
    params.graph_trimLength = 3             
    params.graph_minimumClusterSize = 8
    
    # Stent type dependencies    
    if stentType == 'zenith':
        params.graph_expectedNumberOfEdges = 2
    elif stentType == 'talent':
        params.graph_expectedNumberOfEdges = 2
    elif stentType == 'aneurx':
        params.graph_expectedNumberOfEdges = 4
        params.graph_minimumClusterSize = 400
    elif stentType:
        raise ValueError('Unknown stent type %s' % stentType)

    # Done
    return params


# todo: MCP stop criterion: after segmentation of a certain region?

