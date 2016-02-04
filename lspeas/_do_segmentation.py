""" Script to do the segmentation and store the result.

Do not run file but execute cells (overwrites!)
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.utils import PointSet
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import StentDirect, getDefaultParams, AnacondaDirect, EndurantDirect, NellixDirect
from stentseg.utils.picker import pick3d
from stentseg.apps.graph_manualprune import interactiveClusterRemovalGraph
import _utils_GUI # run as script

# Select the ssdf basedir
# basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
#                      r'D:\LSPEAS\LSPEAS_ssdf',
#                      r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
basedir = r'D:\LSPEAS\Nellix_dyna\DYNA_SSDF'

# Select dataset to register
ptcode = 'dyna_nellix_01'
ctcode = 'POSTOP'
cropname = 'prox'
what = 'phases' # avgreg

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
# vol = s.vol
vol = s.vol40
what = 'vol40'

# f = vv.figure(2)
# vv.hist(vol, bins = 1000)

## Initialize segmentation parameters
stentType = 'nellix'  # 'anacondaRing' runs modified pruning algorithm in Step3

p = getDefaultParams(stentType)
p.seed_threshold = [3070]        # step 1 [lower th] or [lower th, higher th]
p.mcp_speedFactor = 120                 # step 2, costToCtValue; lower less cost for lower HU; higher more cost for lower HU
p.mcp_maxCoverageFronts = 0.002         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 2995             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
p.graph_trimLength =  1                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 3         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 10000          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
# p.graph_min_strutlength = 6             # step 3, stent_anaconda prune_redundant
# p.graph_max_strutlength = 12            # step 3, stent_anaconda prune_redundant
p.graph_angleVector = 3                 # step 3, corner detect
p.graph_angleTh = 10                    # step 3, corner detect
p.seedSampleRate = 7                  # step 1, nellix

## Perform segmentation
cleanNodes = True  # True when NOT using GUI with restore option
guiRemove = False # option to remove nodes/edges but takes longer

# Instantiate stentdirect segmenter object
if stentType == 'anacondaRing':
        sd = AnacondaDirect(vol, p) # inherit _Step3_iter from AnacondaDirect class
        #runtime warning using anacondadirect due to mesh creation, ignore
elif stentType == 'endurant':
        sd = EndurantDirect(vol, p)
elif stentType == 'nellix':
        sd = NellixDirect(vol, p)
else:
        sd = StentDirect(vol, p) 

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
sd.Step3(cleanNodes)

# Create a mesh object for visualization (argument is strut tickness)
bm = create_mesh(sd._nodes3, 0.6) # new


# Visualize
fig = vv.figure(3); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00
clim = (0,3000)
#viewringcrop = 

# Show volume and model as graph
a1 = vv.subplot(131)
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes1.Draw(mc='b', mw = 6)       # draw seeded nodes
# sd._nodes2.Draw(mc='b', lc = 'g')    # draw seeded and MCP connected nodes

# Show volume and cleaned up graph
a2 = vv.subplot(132)
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes2.Draw(mc='b', lc='g')
# sd._nodes3.Draw(mc='b', lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes3.Draw(mc='b', lc='g')
# m = vv.mesh(bm)
# m.faceColor = 'g'
_utils_GUI.vis_spared_edges(sd._nodes3)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Use same camera
a1.camera = a2.camera = a3.camera

#a1.SetView(viewringcrop)

switch = False
a1.axis.visible = switch
a2.axis.visible = switch
a3.axis.visible = switch

## GUI to remove
from visvis import Pointset
from stentseg.stentdirect import stentgraph
from stentseg.stentdirect.stent_anaconda import _edge_length

# initialize labels
t1 = vv.Label(a3, 'Edge ctvalue: ', fontSize=11, color='c')
t1.position = 0.1, 25, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a3, 'Edge cost: ', fontSize=11, color='c')
t2.position = 0.1, 45, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a3, 'Edge length: ', fontSize=11, color='c')
t3.position = 0.1, 65, 0.5, 20
t3.bgcolor = None
t3.visible = False

def on_key(event):
    """KEY commands for user interaction
    UP/DOWN = show/hide nodes
    DELETE  = remove edge [select 2 nodes] or pop node [select 1 node]
    ALT     = SHOW RESULT: remove residual clusters, refine, smooth
    """
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
    if event.key == vv.KEY_DELETE:
        # remove edge
        if len(selected_nodes) == 2:
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
        if len(selected_nodes) == 1:
            # pop node
            select1 = selected_nodes[0].node
            stentgraph._pop_node(sd._nodes3, select1) # asserts degree == 2
            selected_nodes[0].faceColor = 'w'
            selected_nodes.clear()
    elif event.key == vv.KEY_ALT:
        # ALT will FINISH model
        stentgraph.prune_clusters(sd._nodes3, 3) #remove residual nodes/clusters
        # Create mesh and visualize
        view = a3.GetView()
        bm = create_mesh(sd._nodes3, 0.6)
        a3.Clear()
        t = vv.volshow(vol, clim=clim)
        pick3d(vv.gca(), vol)
        sd._nodes3.Draw(mc='b', mw = 8, lc = 'w', lw = 0.2)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        m = vv.mesh(bm)
        m.faceColor = 'g'
        a3.SetView(view)
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; EXECUTE NEXT CELL----')

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

#Add clickable nodes
if guiRemove==True:
        node_points = _utils_GUI.create_node_points(sd._nodes3, scale=0.6)
        
        # Bind event handlers
        fig.eventKeyDown.Bind(on_key)
        for node_point in node_points:
                node_point.eventDoubleClick.Bind(select_node)
        print('')
        print('UP/DOWN = show/hide nodes')
        print('DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node]')
        print('ALT  = SHOW RESULT: remove residual clusters, mesh')
        print('')


## Prevent save when 'run as script'
print('Model not yet saved to disk, run next cells')
1/0

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
print('saved to disk as {}.'.format(filename) )


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
print('saved to disk as {}.'.format(filename) )
