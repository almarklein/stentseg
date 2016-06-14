""" Script to do the segmentation and store the result.

Store graph in _save_segmentation_nellix
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.utils import PointSet, _utils_GUI
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import stentgraph, StentDirect, getDefaultParams, AnacondaDirect, EndurantDirect, NellixDirect
from stentseg.utils.picker import pick3d, label2worldcoordinates, label2volindices
from stentseg.apps.graph_manualprune import interactiveClusterRemovalGraph

# Select the ssdf basedir
basedir = select_dir(r'F:\Nellix_chevas\CHEVAS_SSDF', r'D:\LSPEAS\Nellix_chevas\CHEVAS_SSDF')

# Select dataset to register
ptcode = 'chevas_01'
ctcode = '12months'
cropname = 'prox'
what = '2avgreg' # avgreg

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol
# vol = s.vol40
# what = 'vol40'

# f = vv.figure(1)
# vv.hist(vol, bins = 1000)
# vv.surf(vol[:,:,150])
t0 = vv.volshow(vol, clim=(0,4000))
pick3d(vv.gca(), vol)
vv.gca().daspect = 1,1,-1

## Initialize segmentation parameters
stentType = 'nellix'  # 'anacondaRing' runs modified pruning algorithm in Step3

p = getDefaultParams(stentType)
p.seed_threshold = [3000]        # step 1 [lower th] or [lower th, higher th]
p.mcp_speedFactor = 100                 # step 2, costToCtValue; lower less cost for lower HU; higher more cost for lower HU
p.mcp_maxCoverageFronts = 0.008         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 50             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
p.graph_trimLength =  2                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 10         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 20000          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_angleVector = 5                 # step 3, corner detect
p.graph_angleTh = 45                    # step 3, corner detect

## Perform segmentation
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

#todo: compare different datasets. can we apply one range of params when we normalize?
# # Normalize vol to certain limit
# sd.Step0(5000)
# t0 = vv.volshow(sd._vol, clim=(0,1500))
# pick3d(vv.gca(), sd._vol)
# vv.gca().daspect = 1,1,-1

# Perform the three steps of stentDirect
sd.Step1()
##
sd.Step2()
try:
    sd.Step3(cleanNodes=True) # True when NOT using GUI with restore option
except AssertionError:
    print('--------------')
    print('Step3 failed: error with subpixel due to edges at borders?')
    print('--------------')

# Create a mesh object for visualization (argument is strut tickness)
bm = create_mesh(sd._nodes3, 0.6) # new


## Visualize

guiRemove = False # option to remove nodes/edges but takes longer
addSeeds = True # click to add seeds to sd._nodes1
#todo: depending on the speedFactor fronts do not propagate from manually added seeds. how does mcp work exactly? can we prioritize manually added seeds?

fig = vv.figure(2); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00
clim = (0,2000)
viewsaggital = {'azimuth': 90}

# Show volume and model as graph
a1 = vv.subplot(131)
a1.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
label = pick3d(vv.gca(), vol)
sd._nodes1.Draw(mc='b', mw = 7)       # draw seeded nodes
# sd._nodes2.Draw(mc='b', lc = 'g')    # draw seeded and MCP connected nodes
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show volume and cleaned up graph
a2 = vv.subplot(132)
a2.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
if not sd._nodes2 is None:
    sd._nodes2.Draw(mc='b', lc='g')
    # sd._nodes3.Draw(mc='b', lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
if not sd._nodes3 is None:
    sd._nodes3.Draw(mc='b', lc='g')
    # m = vv.mesh(bm)
    # m.faceColor = 'g'
    _utils_GUI.vis_spared_edges(sd._nodes3)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Use same camera
a1.camera = a2.camera = a3.camera
a1.SetView(viewsaggital)

switch = True
a1.axis.visible = switch
a2.axis.visible = switch
a3.axis.visible = switch

# GUI to remove and/or to add seeds
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
    DELETE  = remove edge [select 2 nodes] or pop node [select 1 node] or remove part of graph [pick a point]
    ALT     = SHOW RESULT after remove residual clusters, pop, corner
    CTRL    = add selected point (SHIFT+Rclick) as seed in sd._nodes1')
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
        if len(selected_nodes) == 0:
            # remove false seeds in spine using the point selected
            _utils_GUI.remove_nodes_by_selected_point(sd._nodes3, vol, a3, 133, label, clim)
            _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a2, 132, label, clim)
            _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, 131, label, clim)
        if len(selected_nodes) == 2:
            # remove edge
            select1 = selected_nodes[0].node
            select2 = selected_nodes[1].node
            c = sd._nodes3.edge[select1][select2]['cost']
            ct = sd._nodes3.edge[select1][select2]['ctvalue']
            p = sd._nodes3.edge[select1][select2]['path']
            l = _edge_length(sd._nodes3, select1, select2)
            sd._nodes3.remove_edge(select1, select2)
            stentgraph.pop_nodes(sd._nodes3) # pop residual nodes
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
    if event.key == vv.KEY_ALT:
        # ALT will FINISH model
        stentgraph.prune_clusters(sd._nodes3, 3) #remove residual nodes/clusters
        stentgraph.pop_nodes(sd._nodes3)
        stentgraph.add_corner_nodes(sd._nodes3, th=sd._params.graph_angleVector, angTh=sd._params.graph_angleTh)
        # Create mesh and visualize
        view = a3.GetView()
        bm = create_mesh(sd._nodes3, 0.6)
        a3.Clear()
        t = vv.volshow(vol, clim=clim)
        pick3d(vv.gca(), vol)
        sd._nodes3.Draw(mc='b', mw = 8, lc = 'g', lw = 0.2)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
#         m = vv.mesh(bm)
#         m.faceColor = 'g'
        _utils_GUI.vis_spared_edges(sd._nodes3)
        a3.SetView(view)
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; RUN _SAVE_SEGMENTATION----')
    elif event.key == vv.KEY_CONTROL:
        coord2 = get_picked_seed(vol, label)
        sd._nodes1.add_node(tuple(coord2))
        view = a1.GetView()
        point = vv.plot(coord2[0], coord2[1], coord2[2], mc= 'g', ms = 's', mw= 12)
        a1.SetView(view)

def get_picked_seed(data, label):
    coord = label2volindices(label) # [x,y,z]
    p = PointSet(coord, dtype=np.float32)
    # Correct for anisotropy and offset
    if hasattr(data, 'sampling'):
        p *= PointSet( list(reversed(data.sampling)) ) 
    if hasattr(data, 'origin'):
        p += PointSet( list(reversed(data.origin)) )
    return list(p.flat)
    

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
        print('DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node] or remove part of graph [pick a point]')
        print('ALT  = SHOW RESULT after remove residual clusters, pop, corner')
        print('CTRL = add selected point (SHIFT+Rclick) as seed')
        print('')
        
elif addSeeds==True:
    # Bind event handlers but do not make node_points
    fig.eventKeyDown.Bind(on_key)
    print('')
    print('CTRL = add selected point (SHIFT+Rclick) as seed')
    print('DELETE = remove part of graph in spine (separation by y of selected point)')
    print('')

