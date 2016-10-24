""" Script to proceed segmentation with user interaction
A Graphical User Interface allows to restore and remove edges
Store graph in _save_segmentation
"""
from stentseg.stentdirect.stentgraph import create_mesh

## Rerun step 3 without removing nodes

sd.Step3(cleanNodes=False) # False when using GUI with restore: clean nodes and smooth after correct/restore


## Visualize with GUI

from visvis import Pointset
from stentseg.stentdirect import stentgraph
from stentseg.stentdirect.stent_anaconda import prune_redundant

fig = vv.figure(4); vv.clf()
fig.position = 8.00, 30.00,  1267.00, 1002.00

# Show volume and graph
a2 = vv.subplot(121)
DrawModelAxes(sd._nodes2, vol, a2, clim=clim, showVol=showVol)

# Show cleaned up graph
a3 = vv.subplot(122)
DrawModelAxes(sd._nodes3, vol, a3, meshColor=meshColor, clim=clim, showVol=showVol, mc='b')

# Use same camera
a2.camera = a3.camera

# Start GUI
def on_key(event):
    """KEY commands for user interaction
    UP/DOWN = show/hide nodes
    ENTER   = restore edge [select 2 nodes]
    DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node]'
    ALT     = clean graph: pop, crossings, corner
    ESCAPE  = FINISH: refine, smooth
    """
    global node_points
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
        l = stentgraph._edge_length(sd._nodes3, select1, select2)
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
        if len(selected_nodes) == 2:
            # remove edge
            select1 = selected_nodes[0].node
            select2 = selected_nodes[1].node
            c = sd._nodes3.edge[select1][select2]['cost']
            ct = sd._nodes3.edge[select1][select2]['ctvalue']
            p = sd._nodes3.edge[select1][select2]['path']
            l = stentgraph._edge_length(sd._nodes3, select1, select2)
            sd._nodes3.remove_edge(select1, select2)
            # visualize removed edge, show keys and deselect nodes
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
        # clean nodes
        if stentType == 'anacondaRing':
            stentgraph.add_nodes_at_crossings(sd._nodes3)
#             prune_redundant(sd._nodes3, sd._params.graph_strongThreshold,
#                                             sd._params.graph_min_strutlength,
#                                             sd._params.graph_max_strutlength)
        stentgraph.pop_nodes(sd._nodes3) # pop before corner detect or angles can not be found
        stentgraph.add_corner_nodes(sd._nodes3, th=sd._params.graph_angleVector, angTh=sd._params.graph_angleTh)
        stentgraph.pop_nodes(sd._nodes3) # because removing edges/add nodes can create degree 2 nodes
        stentgraph.prune_tails(sd._nodes3, sd._params.graph_trimLength)
        stentgraph.prune_clusters(sd._nodes3, 3) #remove residual nodes/clusters
        # visualize result
        view = a3.GetView()
        a3.Clear()
        DrawModelAxes(sd._nodes3, vol, a3, clim=clim, showVol=showVol, mw=8, lw=0.2)
        node_points = _utils_GUI.interactive_node_points(sd._nodes3, scale=0.6)
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
        a3.SetView(view)
        print('----Press ESCAPE to FINISH model----')
    if event.key == vv.KEY_ESCAPE:
        # ESCAPE will FINISH model
        stentgraph.pop_nodes(sd._nodes3)
        sd._nodes3 = sd._RefinePositions(sd._nodes3) # subpixel locations 
        stentgraph.smooth_paths(sd._nodes3, 3)
        # Create mesh and visualize
        view = a3.GetView()
        a3.Clear()
        DrawModelAxes(sd._nodes3, vol, a3, meshColor='g', clim=clim, showVol=showVol, lc='w', mw=8, lw=0.2)
        a3.SetView(view)
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; RUN _SAVE_SEGMENTATION----')
    if event.text == 'q':
        view = a3.GetView()
        _utils_GUI.interactiveClusterRemoval(sd._nodes3)
        a3.SetView(view)
    if event.text == 'z':
        # axes not visible
        _utils_GUI.AxesVis((a1,a2,a3))
    if event.text == 'x':
        # exes visible
        _utils_GUI.AxesVis((a1,a2,a3), axVis=True)

#Add clickable nodes
node_points = _utils_GUI.interactive_node_points(sd._nodes3, scale=0.6)
    
selected_nodes = list()
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
_utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
print('')
print('UP/DOWN = show/hide nodes')
print('ENTER   = restore edge [select 2 nodes]')
print('DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node]')
print('ALT     = clean nodes: crossings, pop, corner, tails, clusters<3')
print('ESCAPE  = FINISH: refine, smooth')
print('')

