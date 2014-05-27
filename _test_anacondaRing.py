""" 
Test for stent segmentation algorithm on the Anaconda CT data.
Class StentDirect_test is created to work the stent segmentation algorithm; inherits from Class StentDirect. def Step3(self) is originally copied from Class StentDirect in base.py

Modifications in Step3(self): uses stentgraph_anacondaRing
"""

import time

import numpy as np
import networkx
import visvis as vv
from visvis import ssdf

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph, stentgraph_anacondaRing
from stentseg.stentdirect.stentgraph import create_mesh

BASEDIR = r'C:\Users\Maaike\Dropbox\UT MA3\Research Aortic Stent Grafts\Data_nonECG-gated\lspeas\\'
#BASEDIR = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\\'

# Load volume data, use Aarray class for anisotropic volumes
s = ssdf.load(BASEDIR+'lspeas_001.ssdf')
#s = ssdf.load(BASEDIR+'lspeas_001_ring.ssdf')
#s = ssdf.load(BASEDIR+'LSPEAS_003_discharge_20.ssdf') 
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
                stentgraph_anacondaRing.prune_weak(nodes, ene, params.graph_strongThreshold, 
                                                    params.graph_min_strutlength,
                                                    params.graph_max_strutlength)
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
        
        # New 1/5/2014
        #stentgraph.pop_nodes(nodes)
        #stentgraph.add_corner_nodes(nodes)
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
p.graph_weakThreshold = 100             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
#p.graph_trimLength =  3                # step 3, stentgraph.prune_tails
p.graph_strongThreshold = 1200          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.seed_threshold = 700                  # step 1
#p.graph_minimumClusterSize = 8         # step 3, stentgraph.prune_clusters
p.mcp_speedFactor = 140                 # step 2, speed image (delta), costToCtValue
p.mcp_maxCoverageFronts = 0.015         # step 2, base.py; replaces mcp_evolutionThreshold
#p.graph_min_strutlength = 7            # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
#p.graph_max_strutlength = 12           # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
# todo: also remove evolutionThreshold in other files

# Instantiate stentdirect segmenter object
#sd = StentDirect_old(vol, p)
#sd = StentDirect(vol, p)
sd = StentDirect_test(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
    # For fast step 3 testing: save output from step 2 and load afterwards for use
#ssdf.save(r'C:\Users\Maaike\Dropbox\ut ma3\research aortic stent grafts\data_nonecg-gated\tmp_nodes2\tmp_nodes2_700_50_005.ssdf', sd._nodes2.pack())
#sd._nodes2 = stentgraph_prune_exception.StentGraph()
#sd._nodes2.unpack(ssdf.load(r'C:\Users\Maaike\Dropbox\ut ma3\research aortic stent grafts\data_nonecg-gated\tmp_nodes2\tmp_nodes2_700_50_005.ssdf'))

sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
if hasattr(sd._nodes3, 'CreateMesh'):
    bm = sd._nodes3.CreateMesh(0.6)  # old
else:
    bm = create_mesh(sd._nodes3, 0.6) # new


## Create figure
fig = vv.figure(3); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00

# Show volume and segmented stent as a graph
a1 = vv.subplot(131)
a1.axis.showBox = False
t = vv.volshow(vol)
t.clim = 0, 2500
sd._nodes1.Draw(mc='g', mw = 6)    # draw seeded nodes
#sd._nodes2.Draw(mc='g', lc = 'r')    # draw seeded and MCP connected nodes

# Show cleaned up
a2 = vv.subplot(132)
t = vv.volshow(vol)
t.clim = 0, 2500
#sd._nodes2.Draw(mc='g', lc='r')
#sd._nodes3.Draw(mc='g', lc='r')
new_nodes1.Draw(mc='g', mw = 6)
#m = vv.mesh(bm)
#m.faceColor = 'g'
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,-1,-1
t = vv.volshow(vol)
t.clim = 0, 2500
#sd._nodes3.Draw(mc='g', lc='g')
#m = vv.mesh(bm)
#m.faceColor = 'g'

vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')

# # Use same camera
a1.camera = a2.camera = a3.camera

# get view through: a1.GetView()
viewlegs = {'zoom': 0.007449745477089642, 'loc': (99.67564087275072, 44.54316599137399, 136.99019904163677), 'roll': 0.0, 'azimuth': -72.0689655172414, 'elevation': 10.451895043731781, 'daspect': (1.0, -1.0, -1.0), 'fov': 0.0}

#viewringcrop = {'loc': (86.3519211709867, 61.10752367572089, 62.86534422588542), 'daspect': (1.0, -1.0, -1.0), 'elevation': 21.47230320699707, 'roll': 0.0, 'fov': 0.0, 'zoom': 0.025718541865111768, 'azimuth': 19.607237589996213}
#a1.SetView(viewringcrop)

#viewring = {'fov': 0.0, 'elevation': 17.01166180758017, 'zoom': 0.019322721160865336, 'roll': 0.0, 'daspect': (1.0, -1.0, -1.0), 'loc': (85.07098073292472, 61.048256073622596, 60.822988663458425), 'azimuth': 95.31034482758619}
#a1.SetView(viewring)

# Take a screenshot 
#vv.screenshot(r'C:\Users\Maaike\Dropbox\UT MA3\Research Aortic Stent Grafts\Data_nonECG-gated\figures\001_hooks_form_triangle.png', vv.gcf(), sf=2)