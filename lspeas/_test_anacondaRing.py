""" 
Test for stent segmentation algorithm on the Anaconda CT data.
Class StentDirect_test is created to work the stent segmentation algorithm; 
Inherits from Class StentDirect. def Step3(self) is originally copied from 
Class StentDirect in base.py

Modifications in Step3(self): uses stentgraph_anacondaRing
"""

import time

import numpy as np
import networkx
import visvis as vv
from visvis import ssdf
import os

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph, stentgraph_anacondaRing
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.utils.datahandling import select_dir, loadvol

# BASEDIR = r'C:\Users\Maaike\Dropbox\UT MA3\Research Aortic Stent Grafts\Data_nonECG-gated\lspeas\\'
# 
# # Load volume data, use Aarray class for anisotropic volumes
# s = ssdf.load(BASEDIR+'lspeas_001.ssdf')
# #s = ssdf.load(BASEDIR+'lspeas_001_ring.ssdf')
# vol = vv.Aarray(s.vol, s.sampling)

# Automatically select basedir for ssdf data
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Set params to load the data
ptcode = 'LSPEAS_003'
ctcode = '1month'
cropname = 'ring'
what = 'avg3090'

# Load volume data
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol

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
        print()
        print('this function is indeed used :)')
        
        # Iteratively prune the graph. The order of operations should
        # not matter too much, although in practice there is a
        # difference. In particular the prune_weak and prune_redundant
        # have a similar function and should be executed in this order.
        cur_edges = 0
        count = 0
        if stentType == 'anacondaRing':
            while cur_edges != nodes.number_of_edges():
                count += 1
                cur_edges = nodes.number_of_edges()
                ene = params.graph_expectedNumberOfEdges
                
                stentgraph.prune_very_weak(nodes, params.graph_weakThreshold)
                stentgraph.prune_weak(nodes, ene, params.graph_strongThreshold)
#                 stentgraph_anacondaRing.prune_weak(nodes, ene, params.graph_strongThreshold,
#                                                         params.graph_min_strutlength,
#                                                         params.graph_max_strutlength)
                stentgraph_anacondaRing.prune_redundant(nodes, params.graph_strongThreshold,
                                                        params.graph_min_strutlength,
                                                        params.graph_max_strutlength)
                stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize)
                stentgraph.prune_tails(nodes, params.graph_trimLength)
        else:
            while cur_edges != nodes.number_of_edges():
                count += 1
                cur_edges = nodes.number_of_edges()
                ene = params.graph_expectedNumberOfEdges
                
                stentgraph.prune_very_weak(nodes, params.graph_weakThreshold)
                stentgraph.prune_weak(nodes, ene, params.graph_strongThreshold)
                stentgraph.prune_redundant(nodes, params.graph_strongThreshold)           
                stentgraph.prune_clusters(nodes, params.graph_minimumClusterSize)
                stentgraph.prune_tails(nodes, params.graph_trimLength)
        
        stentgraph.pop_nodes(nodes)
        stentgraph.add_corner_nodes(nodes)
        stentgraph.smooth_paths(nodes)
        
        t0 = time.time()-t_start
        tmp = "Reduced to %i edges, "
        tmp += "which took %1.2f s (%i iters)"
        print(tmp % (nodes.number_of_edges(), t0, count))
        print("****************************************")
        
        # Finish
        self._nodes3 = nodes  # x,y,z
        if self._draw:
            self.Draw(3)
        
        return nodes


# Get parameters. Different scanners/protocols/stent material might need
# different parameters. 
stentType = 'anacondaRing'

p = getDefaultParams(stentType)
p.seed_threshold = 2300                 # step 1
p.mcp_speedFactor = 190                 # step 2, speed image (delta), costToCtValue
p.mcp_maxCoverageFronts = 0.003         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 100             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 3       # step 3, stentgraph.prune_weak
p.graph_trimLength =  5                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 10         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 3500          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_min_strutlength = 4            # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_max_strutlength = 12           # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
# todo: write function to estimate maxCoverageFronts

# Instantiate stentdirect segmenter object
#sd = StentDirect_old(vol, p)
#sd = StentDirect(vol, p)
sd = StentDirect_test(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
if hasattr(sd._nodes3, 'CreateMesh'):
    bm = sd._nodes3.CreateMesh(0.6)  # old
else:
    bm = create_mesh(sd._nodes3, 0.6) # new


## Create figure

fig = vv.figure(2); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00

# Show volume
a1 = vv.subplot(141)
t = vv.volshow(vol)
t.clim = 0, 3000

# Show volume and nodes/edges
a2 = vv.subplot(142)
t = vv.volshow(vol)
t.clim = 0, 3000
sd._nodes1.Draw(mc='b')      # draw seeded nodes
#sd._nodes2.Draw(mc='b', lc = 'g')    # draw seeded and MCP connected nodes

# Show volume and nodes/edges
a3 = vv.subplot(143)
t = vv.volshow(vol)
t.clim = 0, 3000
sd._nodes2.Draw(mc='b', lc='g')
#sd._nodes3.Draw(mc='b', lc='g')
#new_nodes1.Draw(mc='b', mw = 6)

vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')

# Show volume and cleaned up
a4 = vv.subplot(144)
a4.daspect = 1,-1,-1
t = vv.volshow(vol)
t.clim = 0, 3000
sd._nodes3.Draw(mc='b', lc='g')
#m = vv.mesh(bm)
#m.faceColor = 'g'

vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')

# # Use same camera
a1.camera = a2.camera = a3.camera = a4.camera

# get view through: a1.GetView()
#viewlegs = {'zoom': 0.007449745477089642, 'loc': (99.67564087275072, 44.54316599137399, 136.99019904163677), 'roll': 0.0, 'azimuth': -72.0689655172414, 'elevation': 10.451895043731781, 'daspect': (1.0, -1.0, -1.0), 'fov': 0.0}
# a1.SetView(viewlegs)

viewringcrop = {'azimuth': 103.35642000428354,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 22.77944593951248,
 'fov': 0.0,
 'loc': (179.3373928028404, 98.45327580567643, 60.80823406568838),
 'roll': 0.0,
 'zoom': 0.025718541865111827}
a1.SetView(viewringcrop)

#viewring = {'fov': 0.0, 'elevation': 17.01166180758017, 'zoom': 0.019322721160865336, 'roll': 0.0, 'daspect': (1.0, -1.0, -1.0), 'loc': (85.07098073292472, 61.048256073622596, 60.822988663458425), 'azimuth': 95.31034482758619}
#a1.SetView(viewring)

# Take a screenshot 
#vv.screenshot(r'C:\Users\Maaike\Dropbox\UT MA3\Research Aortic Stent Grafts\Data_nonECG-gated\figures\001_hooks_form_triangle.png', vv.gcf(), sf=2)