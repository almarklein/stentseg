""" Script to do the segmentation and store the result.
A Graphical User Interface allows to restore and remove edges

Do not run file but execute cells (overwrites!)
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.utils import PointSet
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import StentDirect, getDefaultParams, stentgraph, stentgraph_anacondaRing

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode = 'discharge'
cropname = 'ring'
what = 'avgreg'


## Perform segmentation

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol

# Initialize segmentation parameters
stentType = 'anaconda'  # 'anacondaRing' runs stentgraph_anacondaRing.prune_redundant in Step3
cleanNodes = False  # False when using GUI: clean nodes and smooth after correct/restore

p = getDefaultParams(stentType)
p.seed_threshold = 1500                 # step 1
p.mcp_speedFactor = 170                 # step 2, speed image (delta), costToCtValue
p.mcp_maxCoverageFronts = 0.003         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 1000             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 3       # step 3, stentgraph.prune_weak
p.graph_trimLength =  0                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 10         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 4600          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_min_strutlength = 6            # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_max_strutlength = 12            # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
# todo: write function to estimate maxCoverageFronts

# Instantiate stentdirect segmenter object
sd = StentDirect(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
sd.Step3(stentType, cleanNodes)

# Visualize
fig = vv.figure(4); vv.clf()
fig.position = 0, 22, 1366, 706
# fig.position = 1528.00, 123.00,  1246.00, 730.00
viewringcrop = {'azimuth': 158.76671619613668,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 40.40540540540541,
 'fov': 0.0,
 'loc': (140.99391104474705, 110.92390904403258, 100.60072175508655),
 'roll': 0.0,
 'zoom': 0.0332902282323454}

# Show volume and graph
a2 = vv.subplot(121)
t = vv.volshow(vol)
t.clim = 0, 3000
sd._nodes2.Draw(mc='b', lc='g') # draw seeded and MCP connected nodes
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show cleaned up graph
a3 = vv.subplot(122)
a3.daspect = 1,-1,-1
t = vv.volshow(vol)
t.clim = 0, 3000
sd._nodes3.Draw(mc='b', lc = 'b')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')


# Start visualization and GUI
from visvis import Pointset
from stentseg.stentdirect.stentgraph_anacondaRing import _edge_length

#Add clickable nodes
node_points = []
for i, node in enumerate(sd._nodes3.nodes()):
    node_point = vv.solidSphere(translation = (node), scaling = (0.4,0.4,0.4))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    node_points.append(node_point)


# Initialize labels
t1 = vv.Label(a3, 'Edge ctvalue: ', fontSize=11, color='c')
t1.position = 0.1, 5, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a3, 'Edge cost: ', fontSize=11, color='c')
t2.position = 0.1, 25, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a3, 'Edge length: ', fontSize=11, color='c')
t3.position = 0.1, 45, 0.5, 20
t3.bgcolor = None
t3.visible = False

def on_key(event):
    if event.key == vv.KEY_DOWN:
        # hide nodes
        t1.visible = False
        t2.visible = False
        t3.visible = False
        for node_point in node_points:
            node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes
        for node_point in node_points:
            node_point.visible = True
    if event.key == vv.KEY_ENTER:
        # restore edge
        assert len(selected_nodes) == 2
        select1 = selected_nodes[0].node
        select2 = selected_nodes[1].node
        c = sd._nodes2.edge[select1][select2]['cost']
        ct = sd._nodes2.edge[select1][select2]['ctvalue']
        p = sd._nodes2.edge[select1][select2]['path']
        sd._nodes3.add_edge(select1,select2, cost = c, ctvalue = ct, path = p)
        l = _edge_length(sd._nodes3, select1, select2)
        # Visualize restored edge and deselect nodes
        selected_nodes[1].faceColor = 'b'
        selected_nodes[0].faceColor = 'b'
        selected_nodes.clear()
        t1.text = 'Edge ctvalue: \b{%1.2f HU}' % ct
        t2.text = 'Edge cost: \b{%1.7f }' % c
        t3.text = 'Edge length: \b{%1.2f mm}' % l
        t1.visible = True
        t2.visible = True
        t3.visible = True
        view = a3.GetView()
        pp = Pointset(p)  # visvis meshes do not work with PointSet
        line = vv.solidLine(pp, radius = 0.2)
        line.faceColor = 'g'
        a3.SetView(view)
    if event.key == vv.KEY_DELETE:
        # remove edge
        assert len(selected_nodes) == 2
        select1 = selected_nodes[0].node
        select2 = selected_nodes[1].node
        c = sd._nodes3.edge[select1][select2]['cost']
        ct = sd._nodes3.edge[select1][select2]['ctvalue']
        p = sd._nodes3.edge[select1][select2]['path']
        l = _edge_length(sd._nodes3, select1, select2)
        sd._nodes3.remove_edge(select1, select2)
        # Visualize removed edge, show keys and deselect nodes
        selected_nodes[1].faceColor = 'b'
        selected_nodes[0].faceColor = 'b'
        selected_nodes.clear()
        t1.text = 'Edge ctvalue: \b{%1.2f HU}' % ct
        t2.text = 'Edge cost: \b{%1.7f }' % c
        t3.text = 'Edge length: \b{%1.2f mm}' % l
        t1.visible = True
        t2.visible = True
        t3.visible = True
        view = a3.GetView()
        pp = Pointset(p)
        line = vv.solidLine(pp, radius = 0.2)
        line.faceColor = 'r'
        a3.SetView(view)
    elif event.key == vv.KEY_ESCAPE:
        # ESCAPE will finish model
        if cleanNodes == False: # clean here and not in base.py; use None to not clean?
            stentgraph.pop_nodes(sd._nodes3)
            stentgraph.add_nodes_at_crossings(sd._nodes3)
            stentgraph.pop_nodes(sd._nodes3)  # because adding nodes can leave other redundant
            if stentType == 'anacondaRing': # because adding nodes can leave other redundant
                stentgraph_anacondaRing.prune_redundant(sd._nodes3, sd._params.graph_strongThreshold,
                                                    sd._params.graph_min_strutlength,
                                                    sd._params.graph_max_strutlength)
            else:
                stentgraph.prune_redundant(sd._nodes3, sd._params.graph_strongThreshold)
            stentgraph.pop_nodes(sd._nodes3) # because removing edges/add nodes can create degree 2 nodes
            stentgraph.add_corner_nodes(sd._nodes3)
        stentgraph.prune_clusters(sd._nodes3, sd._params.graph_minimumClusterSize) #remove residual nodes/clusters 
        #todo: minclustersize problem when analyzing anaconda stent? single ring 1 node?)
        stentgraph.prune_tails(sd._nodes3, sd._params.graph_trimLength)
        stentgraph.smooth_paths(sd._nodes3)
        # Create mesh and visualize
        view = a3.GetView()
        bm = create_mesh(sd._nodes3, 0.6)
        a3.Clear()
        t = vv.volshow(vol)
        t.clim = 0, 3000
        sd._nodes3.Draw(mc='b', mw = 8, lc = 'g', lw = 0.2)
        vv.xlabel('x')
        vv.ylabel('y')
        vv.zlabel('z')
        m = vv.mesh(bm)
        m.faceColor = 'g'
        a3.SetView(view)

selected_nodes = list()
def select_node(event):
    """ select and deselect nodes by Double Click
    """
    if event.owner not in selected_nodes:
        event.owner.faceColor = 'r'
        selected_nodes.append(event.owner)
    elif event.owner in selected_nodes:
        event.owner.faceColor = 'b'
        selected_nodes.remove(event.owner)
    
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventDoubleClick.Bind(select_node)


# Use same camera
a2.camera = a3.camera

switch = False
a2.axis.visible = switch
a3.axis.visible = switch

a3.SetView(viewringcrop)

switch = False
a2.axis.visible = switch
a3.axis.visible = switch

## Store segmentation to disk

# Get graph model
model = sd._nodes3

# Build struct
s2 = vv.ssdf.new()
# We do not need croprange, but keep for reference
s2.sampling = s.sampling
s2.origin = s.origin
s2.stenttype = s.stenttype
s2.croprange = s.croprange
for key in dir(s):
        if key.startswith('meta'):
            suffix = key[4:]
            s2['meta'+suffix] = s['meta'+suffix]
s2.what = what
s2.params = p
s2.stentType = stentType
# Store model
s2.model = model.pack()
#s2.mesh = ssdf.new()

# Save
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
ssdf.save(os.path.join(basedir, ptcode, filename), s2)


## Make model dynamic (and store/overwrite to disk)

import pirt
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges  

# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
paramsreg = s.params

# Load model
s = loadmodel(basedir, ptcode, ctcode, cropname, 'model'+what)
model = s.model

# Combine ...
incorporate_motion_nodes(model, deforms, s.origin)
incorporate_motion_edges(model, deforms, s.origin)

# Save back
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
s.model = model.pack()
s.paramsreg = paramsreg
ssdf.save(os.path.join(basedir, ptcode, filename), s)
