""" 
Example demonstrating the stent segmentation algorithm on the stent CT
volume that comes with visvis.
"""

import time

import numpy as np
import networkx
import visvis as vv
from visvis import ssdf

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph
from stentseg.stentdirect.stentgraph import create_mesh

BASEDIR = '/home/almar/data/cropped/lspeas/'

# Load volume data, use Aarray class for anisotropic volumes
s = ssdf.load(BASEDIR+'lspeas_001.ssdf')
vol = vv.Aarray(s.vol, s.sampling)


##

class StentDirect_test(StentDirect):
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
        #nodes = stentgraph.StentGraph()
        #nodes.unpack( self._nodes2.pack() )
        nodes = self._nodes2.copy()
        params = self._params
        
        # Init times        
        t_start = time.time()
        t_clean = 0
        
        print('hi, this function is indeed used :)')
        
        # Iteratively prune the graph. The order of operations should
        # not matter too much, although in practice there is a
        # difference. In particular the prune_weak and prene_redundant
        # have a similar function and should be executed in this order.
        cur_edges = 0
        count = 0
        while cur_edges != nodes.number_of_edges():
            count += 1
            cur_edges = nodes.number_of_edges()
            ene = params.graph_expectedNumberOfEdges
            
            stentgraph.prune_very_weak(nodes, params.graph_weakThreshold)
            stentgraph.prune_weak(nodes, ene, params.graph_strongThreshold)
            stentgraph.prune_redundant(nodes, params.graph_strongThreshold)            
            stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize)
            stentgraph.prune_tails(nodes, params.graph_trimLength)
        
        
        t0 = time.time()-t_start
        tmp = "Reduced to %i edges, "
        tmp += "which took %1.2f s (%i iters)"
        print(tmp % (nodes.number_of_edges(), t0, count))
        
        # Finish
        self._nodes3 = nodes
        if self._draw:
            self.Draw(3)
        
        return nodes


# Get parameters. Different scanners/protocols/stent material might need
# different parameters. 
p = getDefaultParams()
p.graph_expectedNumberOfEdges = 2 # 2 for zig-zag, 4 for diamond shaped
p.seed_threshold = 800
p.mcp_evolutionThreshold = 0.06
p.graph_weakThreshold = 10

# Instantiate stentdirect segmenter object
#sd = StentDirect_old(vol, p)
#sd = StentDirect(vol, p)
sd = StentDirect_test(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
# sd._nodes2 = stentgraph.StentGraph(),
# sd._nodes2.Unpack(ssdf.load('/home/almar/tmp.ssdf'))
sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
if hasattr(sd._nodes3, 'CreateMesh'):
    bm = sd._nodes3.CreateMesh(0.6)  # old
else:
    bm = create_mesh(sd._nodes3, 0.6) # new


# Create figue
vv.figure(1); vv.clf()

# Show volume and segmented stent as a graph
a1 = vv.subplot(131)
t = vv.volshow(vol)
t.clim = -1000, 4000
sd._nodes2.Draw(mc='g')

# Show cleaned up
a2 = vv.subplot(132)
sd._nodes3.Draw(mc='g', lc='b')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,-1,-1
m = vv.mesh(bm)
m.faceColor = 'g'

# Use same camera
a1.camera = a2.camera = a3.camera

# Take a screenshot 
#vv.screenshot('/home/almar/projects/valve_result_pat001.jpg', vv.gcf(), sf=2)
