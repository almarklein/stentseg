# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Module base

High level implementation of the MCP algorithm. This module defines
the StentDirect class, which can detect a stent frame in three steps.
It also allows visualization of the intermediate results.

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

from . import stentgraph
from . import stentpoints3d
from .stentmcp import MCP_StentDirect


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
        if fignr is not None:
            f=vv.figure(fignr); vv.clf();
            a = vv.gca()
        else:
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
        pp = stentpoints3d.get_stent_likely_positions(self._vol, th)
        
        # Create nodes object from found points
        nodes = stentgraph.StentGraph()
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
        nodes = stentgraph.StentGraph()
        nodes.Unpack( self._nodes1.Pack() )
        
        # Create speed image (the devision makes it a float array)
        factor = float( self._params.mcp_speedFactor )        
        speed = 1/2**((self._vol)/factor).astype(np.float64)
        
        # Get sampling and origin
        sam = tuple([1.0 for s in self._vol.shape])
        ori = tuple([0.0 for s in self._vol.shape])
        if hasattr(self._vol, 'sampling'):
            sam = speed.sampling = self._vol.sampling
        if hasattr(self._vol, 'origin'):
            ori = speed.origin = self._vol.origin
        
        # Inverse Cost function
        costToCtValue = lambda x: np.log2(1.0/x)*factor
        
        # Create seeds
        #seeds = [(n.x, n.y, n.z) for n in nodes]
        seeds = [(int(0.5 + (n.z-ori[0]) / sam[0]), 
                  int(0.5 + (n.y-ori[1]) / sam[1]), 
                  int(0.5 + (n.x-ori[2]) / sam[2])) for n in nodes]
        
        # Create MCP object
        th = self._params.mcp_evolutionThreshold
        self._mcp = m = MCP_StentDirect(speed, nodes, th, sam, ori)
        
        # Evolve front and trace paths        
        t0 = time.time()
        m.find_costs(seeds, max_coverage=0.2)
        t1 = time.time()
        m.finalize_connections(nodes, costToCtValue)
        #m.ConvertPotentialEdges(nodes, costToCtValue)
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
        nodes = stentgraph.StentGraph()
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



class StentDirect_oldMCP(StentDirect):
    """ Variant that uses the old MCP algorithm. Temporarily kept in place.
    """
    
    def Step2(self):
        """ Step2()
        Find edges using MCP.
        """
        
        from stentseg import _mcp as mcp  # Temporary
        
        # Check if we can go
        if self._vol is None or self._params is None:
            raise ValueError('Data or params not yet given.')
        if self._nodes1 is None:
            raise ValueError('Seed points not yet calculated.')
        
        # Get nodes
        nodes = stentgraph.StentGraph()
        nodes.Unpack( self._nodes1.Pack() )
        
        # Create speed image (the devision makes it a float array)
        factor = float( self._params.mcp_speedFactor )        
        speed = 1/2**((self._vol)/factor).astype(np.float64)
        
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
    