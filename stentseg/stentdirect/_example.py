""" 
Example demonstrating the stent segmentation algorithm on the stent CT
volume that comes with visvis.
"""

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph
from stentseg.stentdirect.stentgraph import create_mesh

# Load volume data, use Aarray class for anisotropic volumes
vol = vv.volread('stent')
vol = vv.Aarray(vol,(1,1,1))
#stentvol = vv.ssdf.load(r'C:\Users\Maaike\Dropbox\UT MA3\Research Aortic Stent Grafts\Data_nonECG-gated\lspeas\lspeas_003.ssdf')
#vol = vv.Aarray(stentvol.vol,stentvol.sampling)
#stentvol = vv.ssdf.load('/home/almar/data/lspeas/LSPEAS_002/LSPEAS_002_1month_ring_avg3090.ssdf')
#vol = vv.Aarray(stentvol.vol,stentvol.sampling)

# Get parameters. Different scanners/protocols/stent material might need
# different parameters. 
p = getDefaultParams()
p.graph_weakThreshold = 10              # step 3, stentgraph.prune_very_weak:
p.mcp_maxCoverageFronts = 0.03         # step 2, create MCP object
p.graph_expectedNumberOfEdges = 2 # 2 for zig-zag, 4 for diamond shaped
#                                       # step 3, in stentgraph.prune_weak
#p.graph_trimLength =                   # step 3, stentgraph.prune_tails
#p.graph_strongThreshold                # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.seed_threshold = 800                  # step 1
#p.graph_minimumClusterSize             # step 3, stentgraph.prune_clusters
#p.mcp_speedFactor                      # step 2, speed image


# Instantiate stentdirect segmenter object
#sd = StentDirect_old(vol, p)
sd = StentDirect(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
# sd._nodes2 = stentgraph.StentGraph()
# sd._nodes2.Unpack(ssdf.load('/home/almar/tmp.ssdf'))
sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
if hasattr(sd._nodes3, 'CreateMesh'):
    bm = sd._nodes3.CreateMesh(0.6)  # old
else:
    bm = create_mesh(sd._nodes3, 0.6) # new


# Create figue
vv.figure(2); vv.clf()

# Show volume and segmented stent as a graph
a1 = vv.subplot(131)
t = vv.volshow(vol)
t.clim = 0, 3000
#sd._nodes1.Draw(mc='g', mw = 6)    # draw seeded nodes
#sd._nodes2.Draw(mc='g')            # draw seeded and MCP connected nodes

# Show cleaned up
a2 = vv.subplot(132)
sd._nodes3.Draw(mc='g', lc='b')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,-1,1
m = vv.mesh(bm)
m.faceColor = 'g'

# Use same camera
a1.camera = a2.camera = a3.camera

# 
test = vv.plot([0,0,0], axesAdjust=False, ls='', ms='.', mc='r', mw=15)
if False:
    node = sd._nodes3.nodes()[3]
    pp = vv.Pointset( np.array(list(node)).reshape(1,3) )
    test.SetPoints(pp)
# Take a screenshot 
#vv.screenshot('/home/almar/projects/valve_result_pat001.jpg', vv.gcf(), sf=2)
