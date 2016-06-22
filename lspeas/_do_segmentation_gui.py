""" Script to proceed segmentation with user interaction
A Graphical User Interface allows to restore and remove edges
Store graph in _save_segmentation
"""


## Rerun step 3 without removing nodes

sd.Step3(cleanNodes=False) # False when using GUI with restore: clean nodes and smooth after correct/restore


## Visualize with GUI

from visvis import Pointset
from stentseg.stentdirect import stentgraph
from stentseg.stentdirect.stent_anaconda import _edge_length, prune_redundant

fig = vv.figure(4); vv.clf()
fig.position = 8.00, 30.00,  1267.00, 1002.00
clim = (0,2000)
viewsaggital = {'azimuth': 90}

# Show volume and graph
a2 = vv.subplot(121)
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes2.Draw(mc='b', lc='g') # draw seeded and MCP connected nodes
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show cleaned up graph
a3 = vv.subplot(122)
a3.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
label=pick3d(vv.gca(), vol)
sd._nodes3.Draw(mc='b', lc = 'b')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')


# Start GUI
# initialize labels
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
    """KEY commands for user interaction
    UP/DOWN = show/hide nodes
    ENTER   = restore edge [select 2 nodes]
    DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node] or remove graph posterior to picked point'
    CTRL    = clean nodes: pop, crossings, corner
    ESCAPE  = FINISH: refine, smooth
    """
    global label
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
        if len(selected_nodes) == 0:
            # remove false seeds posterior to picked point, e.g. for spine
            try:
                _utils_GUI.remove_nodes_by_selected_point(sd._nodes3, vol, a3, 133, label, clim)
            except ValueError: # false nodes already cleaned in Step3
                pass
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
    if event.key == vv.KEY_CONTROL:
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
        t = vv.volshow(vol, clim=clim)
        label=pick3d(vv.gca(), vol)
        sd._nodes3.Draw(mc='b', mw = 8, lc = 'g', lw = 0.2)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        a3.SetView(view)
        print('----Press ESCAPE to FINISH model----')
        print('----Close FIG and EXECUTE CELL again to REMOVE more edges/nodes----')
    elif event.key == vv.KEY_ESCAPE:
        # ESCAPE will FINISH model
        sd._nodes3 = sd._RefinePositions(sd._nodes3) # subpixel locations 
        stentgraph.smooth_paths(sd._nodes3, 2)
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
        print('----DO NOT FORGET TO SAVE THE MODEL TO DISK; RUN _SAVE_SEGMENTATION----')


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
node_points = _utils_GUI.create_node_points(sd._nodes3)
    
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventDoubleClick.Bind(select_node)
print('')
print('UP/DOWN = show/hide nodes')
print('ENTER   = restore edge [select 2 nodes]')
print('DELETE  = remove edge [select 2 ndoes] or pop node [select 1 node] or remove graph posterior to picked point')
print('CTRL    = clean nodes: crossings, pop, corner, tails, clusters<3')
print('ESCAPE  = FINISH: refine, smooth')
print('')

# Use same camera
a2.camera = a3.camera

# switch = False
# a2.axis.visible = switch
# a3.axis.visible = switch

