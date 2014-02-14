""" 
Example demonstrating the stent segmentation algorithm on the stent CT
volume that comes with visvis.
"""

import visvis as vv
from visvis import ssdf

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph
from stentseg.stentdirect.stentgraph import create_mesh

# Load volume data, use Aarray class for anisotropic volumes
vol = vv.volread('stent')
vol = vv.Aarray(vol, (1,1,1))


# Get parameters. Different scanners/protocols/stent material might need
# different parameters. 
p = getDefaultParams()
p.graph_expectedNumberOfEdges = 2 # 2 for zig-zag, 4 for diamond shaped
p.seed_threshold = 800
p.mcp_evolutionThreshold = 0.06
p.graph_weakThreshold = 10

# Instantiate stentdirect segmenter object
#sd = StentDirect_old(vol, p)
sd = StentDirect(vol, p)

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
