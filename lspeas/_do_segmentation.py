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
from stentseg.apps.crop import cropvol

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
# basedir = select_dir(r'D:\LSPEAS_F\LSPEASF_ssdf', r'F:\LSPEASF_ssdf_backup')

# Select dataset to register
ptcode = 'lspeas_002'
ctcode =  '24months'
cropname = 'ring'
what = 'avgreg' # avgreg
normalize = False
crop = False

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol

# Show volume to explore intesity values
# f = vv.figure()
# t0 = vv.volshow(vol, clim=(0,2500))
# label = pick3d(vv.gca(), vol)
# vv.gca().daspect = 1,1,-1
# clim = (-127,457)
# showVol = '2D'
# label1 = DrawModelAxes(vol, graph=None, clim=clim, showVol=showVol, climEditor=True)

# Explore histogram
# f = vv.figure()
# h = vv.hist(vol, bins=50)
# h.color = 'r'
# h.GetAxes().SetLimits(rangeX=(-1500,30000), rangeY=(0,10000000))
# vv.surf(vol[:,:,150])

# Optionally crop volume for segmentation
if crop:
    vol2 = cropvol(vol)
else:
    vol2 = vol


## Initialize segmentation parameters
stentType = 'anacondaRing'
# 'anacondaRing' runs AnacondaDirect with modified Step3 and _Step3_iter
# 'branch' or 'nellix' runs Nellixdirect with modified seeding

p = getDefaultParams(stentType)
p.seed_threshold = [600,2000]        # step 1 [lower th] or [lower th, higher th]
p.mcp_speedFactor = 750                 # step 2, costToCtValue; 
                                        # lower-> longer paths (costs low) -- higher-> short paths (costs high)
p.mcp_maxCoverageFronts = 0.003         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 500             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
p.graph_trimLength =  0                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 1         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 3500          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.graph_min_strutlength = 5             # step 3, stent_anaconda prune_redundant
p.graph_max_strutlength = 13            # step 3, stent_anaconda prune_redundant
p.graph_angleVector = 5                 # step 3, corner detect
p.graph_angleTh = 180                    # step 3, corner detect

## Perform segmentation

# Instantiate stentdirect segmenter object
sd = initStentDirect(stentType, vol2, p)

#todo: compare different datasets. can we apply one range of params when we normalize?
# Normalize vol to certain limit
if normalize:
    sd.Step0(3071)
    vol2 = sd._vol

# Perform the first (seeding) step out of 3 steps of stentDirect
sd.Step1()

## Visualization and interactive segmentation steps
#todo: depending on the speedFactor fronts do not propagate from manually added seeds. 
# see costfunction, from speedfactor > 750 it works for lspeas data

guiRemove = False # True for option to remove nodes/edges but takes longer
clim = (0,3000)
showVol = 'MIP'
meshColor = None # or give FaceColor
viewLR = {'azimuth': 90, 'roll': 0}

fig = vv.figure(3); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00

# Show model Step 1
a1 = vv.subplot(131)
label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, climEditor=True, removeStent=False) # lc, mc

# Create axis for Step 2
a2 = vv.subplot(132)
DrawModelAxes(vol, ax=a2, clim=clim, showVol=showVol, climEditor=False)

# Create axis for Step 3
a3 = vv.subplot(133)
DrawModelAxes(vol, ax=a3, clim=clim, showVol=showVol, climEditor=False)
# _utils_GUI.vis_spared_edges(sd._nodes3)

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
        'p = remove seeds posterior (y-axis) to [picked point] (use for spine seeds)'
        'o = remove seeds anterior (y-axis) to [picked point]'
        'i = remove seeds proximal (z-axis) to [picked point]'
        'k = remove seeds distal (z-axis) to [picked point]'
        'l = remove seeds left (x-axis) to [picked point]'
        'j = remove seeds right (x-axis) to [picked point]'
        'ALT   = clean graph: remove residual clusters, pop, corner'
        'PageUp= protect node closest to picked point in nodes1 axes, no pop
        'n = add [picked point] (SHIFT+R-click) as seed'
        '1 = redo step 1; 2 = redo step 2; 3 = redo step 3'
        'z/x/a/d = axis invisible/visible/rotate'
    """
    global label
    global node_points
    global sd
    if event.key == vv.KEY_DOWN:
        # hide nodes
        t1.visible = False
        t2.visible = False
        t3.visible = False
        if 'node_points' in globals():
            for node_point in node_points:
                node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes
        if 'node_points' in globals():
            for node_point in node_points:
                node_point.visible = True
    if event.key == vv.KEY_DELETE:
        if len(selected_nodes) == 0:
            # remove node closest to picked point
            node = _utils_GUI.snap_picked_point_to_graph(sd._nodes1, vol, label, nodesOnly=True)
            sd._nodes1.remove_node(node)
            view = a1.GetView()
            a1.Clear()
            label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, removeStent=False)
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
        DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol, lc='g', mw=8, lw=0.2)
        # _utils_GUI.vis_spared_edges(sd._nodes3)
        a3.SetView(view)
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; RUN _SAVE_SEGMENTATION----')
    if event.text == 'n':
        # add picked seed to nodes_1
        coord2 = get_picked_seed(vol, label)
        sd._nodes1.add_node(tuple(coord2))
        view = a1.GetView()
        point = vv.plot(coord2[0], coord2[1], coord2[2], mc= 'b', ms = 'o', mw= 8, alpha=0.5, axes=a1)
        a1.SetView(view)
    if event.key == vv.KEY_PAGEUP:
        # protect node from pop
        pickedNode = _utils_GUI.snap_picked_point_to_graph(sd._nodes1, vol, label, nodesOnly=True) 
        sd._nodes1.add_node(pickedNode, nopop = True)
        sd._nodes2.add_node(pickedNode, nopop = True)
        view = a1.GetView()
        point = vv.plot(pickedNode[0], pickedNode[1], pickedNode[2], mc= 'y', ms = 'o', mw= 8, alpha=0.5, axes=a1)
        a1.SetView(view)
        # now rerun step 3
    if event.text == 'p':
        # remove false seeds posterior to picked point, e.g. for spine
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a2, label, 
            clim, location='posterior', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='posterior', showVol=showVol)
    if event.text == 'o':
        # remove seeds prox to selected point
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a1, label, 
            clim, location='anterior', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='anterior', showVol=showVol)
    if event.text == 'i':
        # remove seeds prox to selected point
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a1, label, 
            clim, location='proximal', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='proximal', showVol=showVol)
    if event.text == 'k':
        # remove seeds dist to selected point
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a1, label, 
            clim, location='distal', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='distal', showVol=showVol)
    if event.text == 'l':
        # remove seeds left to selected point
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a1, label, 
            clim, location='left', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='left', showVol=showVol)
    if event.text == 'j':
        # remove seeds right to selected point
        _utils_GUI.remove_nodes_by_selected_point(sd._nodes2, vol, a1, label, 
            clim, location='right', showVol=showVol)
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, 
            clim, location='right', showVol=showVol)
    if event.text == '1':
        # redo step1
        view = a1.GetView()
        a1.Clear(); a2.Clear(); a3.Clear()
        sd._params = p
        sd.Step1()
        label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, removeStent=False) # lc, mc
        a1.SetView(view)
    if event.text == '2':
        # redo step2
        view = a2.GetView()
        a2.Clear(); a3.Clear()
        sd._params = p
        sd.Step2()
        DrawModelAxes(vol, sd._nodes2, a2, clim=clim, showVol=showVol,removeStent=False)
        a2.SetView(view)
    if event.text == '3':
        # redo step3
        view = a3.GetView()
        a3.Clear()
        sd._params = p
        sd.Step3(cleanNodes=True)
        DrawModelAxes(vol, sd._nodes3, a3, meshColor=meshColor, clim=clim, showVol=showVol,removeStent=False)
        node_points = _utils_GUI.interactive_node_points(sd._nodes3, scale=0.6)
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
        a3.SetView(view)


# Init list for nodes
selected_nodes = list()
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
fig.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a1, a2, a3], keyboard=['6', '7', '8', '9', '0']) )
fig.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1, a2, a3]) )

# Print user instructions
print('')
print('n = add [picked point] (SHIFT+R-click) as seed')
print('PageUp = protect node closest to picked point in nodes1 axes, no pop')
print('p = remove seeds posterior (y-axis) to [picked point] (use for spine seeds)')
print('o = remove seeds anterior (y-axis) to [picked point]')
print('i = remove seeds proximal (z-axis) to [picked point]')
print('k = remove seeds distal (z-axis) to [picked point]')
print('l = remove seeds left (x-axis) to [picked point]')
print('j = remove seeds right (x-axis) to [picked point]')
print('1 = redo step 1; 2 = redo step 2; 3 = redo step 3')
print('x and a/d = axis invisible/visible and rotate')

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

