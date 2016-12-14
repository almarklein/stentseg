""" Script to do the segmentation and store the result.

Store graph in _save_segmentation
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf
from visvis import Pointset

from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect import stentgraph, getDefaultParams, initStentDirect
from stentseg.utils.picker import pick3d, get_picked_seed
from stentseg.utils.visualization import DrawModelAxes

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_025'
ctcode = '12months'
cropname = 'ring'
what = 'phases' # avgreg

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol60

# h = vv.hist(vol, bins = 1000)
# h.color = 'y'
# vv.surf(vol[:,:,150])
# f = vv.figure()
# t0 = vv.volshow(vol, clim=(0,2500))
# pick3d(vv.gca(), vol)
# vv.gca().daspect = 1,1,-1

## Initialize segmentation parameters
stentType = 'anacondaRing'  # 'anacondaRing' runs modified pruning algorithm in Step3

p = getDefaultParams(stentType)
p.seed_threshold = [1000,3000]        # step 1 [lower th] or [lower th, higher th]
p.mcp_speedFactor = 150                 # step 2, costToCtValue; lower-> longer paths -- higher-> short paths
p.mcp_maxCoverageFronts = 0.003         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 650             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 3       # step 3, stentgraph.prune_weak
p.graph_trimLength =  0                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 10         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 3000          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_min_strutlength = 5             # step 3, stent_anaconda prune_redundant
p.graph_max_strutlength = 13            # step 3, stent_anaconda prune_redundant
p.graph_angleVector = 5                 # step 3, corner detect
p.graph_angleTh = 180                    # step 3, corner detect

## Perform segmentation

# Instantiate stentdirect segmenter object
sd = initStentDirect(stentType, vol, p)
cleanNodes = True

#todo: compare different datasets. can we apply one range of params when we normalize?
# Normalize vol to certain limit
sd.Step0(3071)
vol = sd._vol

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
try:
    sd.Step3(cleanNodes=cleanNodes) # True when NOT using GUI with restore option
except AssertionError:
    print('--------------')
    print('Step3 failed: error with subpixel due to edges at borders? Change params')
    print('--------------')

## Visualize
#todo: depending on the speedFactor fronts do not propagate from manually added seeds. how does mcp work exactly? can we prioritize manually added seeds?

guiRemove = False # option to remove nodes/edges but takes longer
clim = (0,2500)
showVol = 'MIP'
meshColor = None # or give FaceColor
viewLR = {'azimuth': 90, 'roll': 0}

fig = vv.figure(3); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00

# Show model Step 1
a1 = vv.subplot(131)
label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol) # lc, mc

# Show model Step 2
a2 = vv.subplot(132)
DrawModelAxes(vol, sd._nodes2, a2, clim=clim, showVol=showVol)

# Show model Step 3
a3 = vv.subplot(133)
DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol)
_utils_GUI.vis_spared_edges(sd._nodes3)

# Use same camera
a1.camera = a2.camera = a3.camera
a1.SetView(viewLR)

# GUI to remove and/or to add seeds

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
        'UP/DOWN  = show/hide nodes'
        'DELETE   = remove edge [select 2 nodes] or pop node [select 1 node] '
                   'or remove seed in nodes1 closest to [picked point]'
        'PageDown = remove graph posterior (y-axis) to [picked point] (use for spine seeds)'
        'ALT      = clean graph: remove residual clusters, pop, corner'
        'CTRL+SHIFT = add [picked point] (SHIFT+R-click) as seed'
    """
    global label
    global node_points
    global sd
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
            # remove node closest to picked point
            node = _utils_GUI.snap_picked_point_to_graph(sd._nodes1, vol, label)
            sd._nodes1.remove_node(node)
            view = a1.GetView()
            a1.Clear()
            label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol)
            a1.SetView(view)    
        if len(selected_nodes) == 2:
            # remove edge
            select1 = selected_nodes[0].node
            select2 = selected_nodes[1].node
            c = sd._nodes3.edge[select1][select2]['cost']
            ct = sd._nodes3.edge[select1][select2]['ctvalue']
            path = sd._nodes3.edge[select1][select2]['path']
            l = stentgraph._edge_length(sd._nodes3, select1, select2)
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
            pp = Pointset(path)
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
        a3.Clear()
        DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol, lc='b', mw=8, lw=0.2)
        _utils_GUI.vis_spared_edges(sd._nodes3)
        a3.SetView(view)
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; RUN _SAVE_SEGMENTATION----')
    if event.key == vv.KEY_CONTROL and event.key == vv.KEY_SHIFT:
        # add picked seed to nodes_1
        coord2 = get_picked_seed(vol, label)
        sd._nodes1.add_node(tuple(coord2))
        view = a1.GetView()
        point = vv.plot(coord2[0], coord2[1], coord2[2], mc= 'b', ms = 'o', mw= 8, alpha=0.5)
        a1.SetView(view)
    if event.key == vv.KEY_PAGEDOWN:
        # remove false seeds posterior to picked point, e.g. for spine
        try:
            _utils_GUI.remove_nodes_by_selected_point(sd._nodes3, vol, a3, label, clim, showVol=showVol)
        except ValueError: # false nodes already cleaned by Step3
            pass
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a2, label, clim, showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, clim, showVol=showVol)
    if event.text == '1':
        # redo step1
        view = a1.GetView()
        a1.Clear(); a2.Clear(); a3.Clear()
        sd._params = p
        sd.Step1()
        label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol) # lc, mc
        a1.SetView(view)
    if event.text == '2':
        # redo step2
        view = a2.GetView()
        a2.Clear(); a3.Clear()
        sd._params = p
        sd.Step2()
        sd.Step3(cleanNodes=cleanNodes)
        DrawModelAxes(vol, sd._nodes2, a2, clim=clim, showVol=showVol)
        DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol)
        a2.SetView(view)
    if event.text == '3':
        view = a3.GetView()
        a3.Clear()
        sd._params = p
        sd.Step3(cleanNodes=cleanNodes)
        DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol)
        node_points = _utils_GUI.interactive_node_points(sd._nodes3, scale=0.6)
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
        a3.SetView(view)
    if event.text == 'z':
        # axes not visible
        _utils_GUI.AxesVis((a1,a2,a3))
    if event.text == 'x':
        # axes visible
        _utils_GUI.AxesVis((a1,a2,a3), axVis=True)

# Init list for nodes
selected_nodes = list()
# Bind event handlers
fig.eventKeyDown.Bind(on_key)

# Print user instructions
print('')
print('CTRL+SHIFT = add [picked point] (SHIFT+R-click) as seed')
print('PageDown = remove graph posterior (y-axis) to [picked point] (spine seeds)')
print('1 = redo step 1; 2 = redo step 2; 3 = redo step 3')
print('z/x = axis invisible/visible')

if guiRemove==True:
        # Add clickable nodes to remove edges
        node_points = _utils_GUI.interactive_node_points(sd._nodes3, scale=0.6)
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
        print('UP/DOWN = show/hide nodes')
        print('DELETE  = remove edge [select 2 nodes] or pop node [select 1 node] '
               'or remove seed in nodes1 closest to [picked point]')
        print('ALT  = SHOW RESULT after remove clusters, pop, corner')
        print('')
else:
    # Bind event handlers but do not make node_points
    print('DELETE = remove seed in nodes1 closest to [picked point]')
    print('')

