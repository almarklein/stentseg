from stentseg.stentdirect import StentDirect, getDefaultParams, stentgraph
import visvis as vv
vv.pypoints.SHOW_SUBTRACTBUG_WARNING = True # Importand for converted legacy code


# Somehow obtain a volume (replace the three lines below)
# use Aarray class for anisotropic volumes
from visvis import ssdf
s = ssdf.load('/home/almar/data/cropped/stents_valve/cropped_pat101.bsdf')
vol = vv.Aarray(s.vol, s.sampling, s.origin)

##

# Get parameters. Different scanners/protocols/stent material might need
# different parameters. 
p = getDefaultParams()
p.graph_expectedNumberOfEdges = 4 # 2 for zig-zag, 4 for diamond shaped
p.seed_threshold = 1600
p.mcp_evolutionThreshold = 0.001
p.graph_weakThreshold = 10
sd = StentDirect(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
#sd._nodes2 = stentgraph.StentGraph(),
#sd._nodes2.Unpack(ssdf.load('/home/almar/tmp.ssdf'))
sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
bm = sd._nodes3.CreateMesh(0.6)

# Create figue
vv.figure(1); vv.clf()

# Show volume and segmented stent as a graph
a1 = vv.subplot(121)
t = vv.volshow(vol)
t.clim = -1000, 4000
sd._nodes3.Draw(mc='g')

# Show the mesh
a2 = vv.subplot(122)
a2.daspect = 1,-1,-1
m = vv.mesh(bm)
m.faceColor = 'g'

# Use same camera
a1.camera = a2.camera

# Take a screenshot 
#vv.screenshot('/home/almar/projects/valve_result_pat001.jpg', vv.gcf(), sf=2)
