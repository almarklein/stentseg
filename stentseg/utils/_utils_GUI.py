# Copyright (C) M.A Koenrades
""" Functionality for segmentation and graph modeling using GUI

"""

import visvis as vv
import numpy as np

alpha = 0.8

#todo: maak ButtonSegmentation class
# open vv.wibjects.buttons
# open vv.wibjects.ColormapEditor
# button = vv.buttons.PushButton(vv.gca())
# from visvis.wibjects.buttons import PushButton
# def buttonFoo(*args, **kwargs):
#     print("Pressed button, arguments: {}, {}".format(args, kwargs))
# button.eventPress.Bind(buttonFoo)

# class ButtonSegmentation:
#     """ 
#     """
#     def __init__(self, parent=None, sd=None, params=None, step=1, *args):
#         
#         # init size
#         # self.position.w = 300
#         # self.position.h = 50
#         
#         # params and segmenter object
#         self._params = params
#         self._sd = sd
#         self._parent = parent
#         
#         # create buttons
#         self._text = {0:None, 1:'STEP 1', 2:'STEP 2', 3:'STEP 3'}[step]
#         self._But1 = vv.PushButton(parent, self._text)
#         self._But1.position =  5,15, 12,14
#         
#         # bind
#         self._step = step
#         self._But1.eventPress.Bind(buttonFoo)
#     
#     def buttonFoo(*args, **kwargs):
#         from stentseg.utils.visualization import DrawModelAxes
#         # print("Pressed button, arguments: {}, {}".format(args, kwargs))
#         if self._step == 1:
#             a1 = parent
#             view = a1.GetView()
#             vv.clf()
#             # a1.Clear(); a2.Clear(); a3.Clear()
#             sd._params = self._params
#             sd.Step1()
#             label = DrawModelAxes(sd._nodes1, vol, a1, clim=clim, showVol=showVol) # lc, mc
#             a1.SetView(view)

def ViewPresets(event, axis=None):
    """ View presets for anatomic views: anterior, posterior, left, right, top,
    bottom. 1, 2.. keys
    Can be used with f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a1, a2]) )
    """
    if axis is None:
        ax = vv.gca()
        axis = [ax]
    for ax in axis:
        view = ax.GetView()
        view['roll'] = 0
        # sagital LR
        if event.text == '1':
            view['azimuth'] = 90
            view['elevation'] = 0
        # posterior
        elif event.text == '2':
            view['azimuth'] = 0
            view['elevation'] = 0
        # sagital RL
        elif event.text == '3':
            view['azimuth'] = -90
            view['elevation'] = 0
        elif event.text == '4':
            view['azimuth'] = -180
            view['elevation'] = 90
        elif event.text == '5':
            view['azimuth'] = 0
            view['elevation'] = -90
        elif event.text == 'v':
            view['zoom'] = 0.0198
        ax.SetView(view)
    if event.text == 'x':
        # axes visible or hide
        AxesVis(axis)


def ShowHideSlider(event, c):
    """ View/hide the slider (clim/isoth editor)
    c: single clim object or dict with multiple
    print('Use "s" to show/hide slider')
    """
    if event.text == 's':
        if isinstance(c, dict): # with c object from multiple axes
            for key in c:
                switch = c[key].visible
                if switch == True:
                    c[key].visible = False
                else:
                    c[key].visible = True
        else:
            switch = c.visible
            if switch == True:
                c.visible = False
            else:
                c.visible = True

def AxesVis(axis):
    """ Axis input list with axes
    show or hide axes
    """
    switch = axis[0].axis.visible
    for ax in axis:
        if switch == True:
            ax.axis.visible = False
        else:
            ax.axis.visible = True

def RotateView(event, axis=None):
    """ Rotate view in axes given in list axis. Use a and d keys 
    If axis is not given, current axes is used
    Can be used with f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1, a2]) )
    """
    if axis is None:
        ax = vv.gca()
        axis = [ax]
    for ax in axis:
        view = ax.GetView()
        if event.text == 'a':
            plus90 = view['azimuth'] + 90
            view['azimuth'] = plus90
        elif event.text == 'd':
            min90 = view['azimuth'] - 90
            view['azimuth'] = min90
        elif event.key == vv.KEY_UP and vv.KEY_CONTROL in event.modifiers:
            v = view['elevation'] - 5 # tilt
            view['elevation'] = v
        elif event.key == vv.KEY_DOWN and vv.KEY_CONTROL in event.modifiers:
            v = view['elevation'] + 5 # tilt
            view['elevation'] = v
        #todo: rotate with left and right keys, how to update smoothly?
        elif event.key == vv.KEY_LEFT and vv.KEY_CONTROL in event.modifiers:
            min1 = view['azimuth'] - 0.5
            view['azimuth'] = min1
        elif event.key == vv.KEY_RIGHT and vv.KEY_CONTROL in event.modifiers:
            plus1 = view['azimuth'] + 0.5
            view['azimuth'] = plus1
        #todo: how to move perpendicular to screen L and R?
        elif event.key == vv.KEY_LEFT:
            min1 = view['loc'][1] + 5 # shift in y
            view['loc'] = (view['loc'][0],min1,view['loc'][2])
        elif event.key == vv.KEY_RIGHT:
            min1 = view['loc'][1] - 5 # shift in y
            view['loc'] = (view['loc'][0],min1,view['loc'][2])
        elif event.key == vv.KEY_UP:
            min1 = view['loc'][-1] + 5 # shift in z
            view['loc'] = (view['loc'][0],view['loc'][1],min1)
        elif event.key == vv.KEY_DOWN:
            plus1 = view['loc'][-1] - 5 # shift in z
            view['loc'] = (view['loc'][0],view['loc'][1],plus1)
        ax.SetView(view)
    if event.text == 'x':
        # axes visible or hide
        AxesVis(axis)


def node_points_callbacks(node_points, selected_nodes, pick=True, t0=None):
    """ Bind callback functions to node points
    t0 = Label t0
    """ 
    for node_point in node_points:
        node_point.eventDoubleClick.Bind(lambda event: select_node(event, selected_nodes) )
        if pick == True:
            node_point.eventEnter.Bind(lambda event: pick_node(event, t0) )
            node_point.eventLeave.Bind(lambda event: unpick_node(event, t0) )

def select_node(event, selected_nodes):
    """ select and deselect nodes by Double Click
    """
    if event.owner not in selected_nodes:
        event.owner.faceColor = (1,0,0,alpha) # 'r' but with alpha 
        selected_nodes.append(event.owner)
    elif event.owner in selected_nodes:
        event.owner.faceColor = (0,0,1,alpha) # 'b' but with alpha
        selected_nodes.remove(event.owner)

def pick_node(event, t0):
    nodenr = event.owner.nr
    node = event.owner.node
    t0.text = '\b{Node nr|location}: %i | x=%1.3f y=%1.3f z=%1.3f' % (nodenr,node[0],node[1],node[2])

def unpick_node(event, t0):
    t0.text = '\b{Node nr|location}: '

def remove_nodes_by_selected_point(graph, vol, axes, label, clim, dim=1, **kwargs):
    """ removes nodes and edges in graph. Graph is separated by coord of selected point
    use y (dim=1) to remove graph in spine
    Input : graph, axes, label of selected point, 
            dimension how to separate graph
    Output: sd._nodes1,2,3  are modified and visualized in current view
    """
    from stentseg.utils.picker import pick3d, label2worldcoordinates
    from stentseg.utils.visualization import DrawModelAxes
    
    if graph is None:
        print('No nodes removed, graph is NoneType')
        return
    coord1 = np.asarray(label2worldcoordinates(label), dtype=np.float32) # x,y,z
    seeds = np.asarray(sorted(graph.nodes(), key=lambda x: x[dim])) # sort y
    falseindices = np.where(seeds[:,1]>coord1[1]) # indices with values higher than coord y
    falseseeds = seeds[min(falseindices[0]):]
    graph.remove_nodes_from(tuple(map(tuple, falseseeds)) ) # use map to convert to tuples
    view = axes.GetView()
    axes.Clear()
    DrawModelAxes(vol, graph, ax=axes, clim=clim, **kwargs)
    axes.SetView(view)
    if graph.number_of_edges() == 0: # get label from picked seeds sd._nodes1 
        label = pick3d(vv.gca(), vol)
        return label

def get_edge_attributes(model, n1, n2):
    """
    returns cost, ctvalue, path, edge length (mm)
    """ 
    from stentseg.stentdirect.stentgraph import _edge_length
    c = model.edge[n1][n2]['cost']
    ct = model.edge[n1][n2]['ctvalue']
    p = model.edge[n1][n2]['path']
    l = _edge_length(model, n1, n2)
    return c, ct, p, l
    
def set_edge_labels(t1,t2,t3,ct,c,l):
    t1.text = 'Edge ctvalue: \b{%1.2f HU}' % ct
    t2.text = 'Edge cost: \b{%1.7f }' % c
    t3.text = 'Edge length: \b{%1.2f mm}' % l
    t1.visible = True
    t2.visible = True
    t3.visible = True

def interactive_node_points(graph, scale=0.4, **kwargs):
    """ create node objects for gui
    """
    node_points = []
    for i, node in enumerate(sorted(graph.nodes())):
        node_point = vv.solidSphere(translation = (node), scaling = (scale,scale,scale))
        node_point.faceColor = (0,0,1,alpha) # 'b' but with alpha
        node_point.visible = False
        node_point.node = node
        node_point.nr = i
        node_points.append(node_point)
    return node_points

def create_node_points_with_amplitude(graph, scale =0.4, **kwargs):
    """ create node objects for gui and calculate motion amplitude for each node
    """
    from stentseg.motion.displacement import _calculateAmplitude
    pointsDeforms = []
    node_points = []
    for i, node in enumerate(sorted(graph.nodes())):
        node_point = vv.solidSphere(translation = (node), scaling = (scale,scale,scale))
        node_point.faceColor = (0,0,1,alpha) # 'b' but with alpha
        node_point.visible = False
        node_point.node = node
        node_point.nr = i
        nodeDeforms = graph.node[node]['deforms']
        dmax_xyz = _calculateAmplitude(nodeDeforms, dim='xyz') # [dmax, p1, p2]
        dmax_z = _calculateAmplitude(nodeDeforms, dim='z')
        dmax_y = _calculateAmplitude(nodeDeforms, dim='y')
        dmax_x = _calculateAmplitude(nodeDeforms, dim='x')
        pointsDeforms.append(nodeDeforms)
        node_point.amplXYZ = dmax_xyz # amplitude xyz = [0]
        node_point.amplZ = dmax_z 
        node_point.amplY = dmax_y  
        node_point.amplX = dmax_x 
        node_points.append(node_point)
    return node_points, pointsDeforms

def vis_spared_edges(graph, radius = 0.6, axes=None):
    """ in step 3 with anacondaRing, prune_redundant spares strong triangle edges.
        visualize with a model
    """ 
    from visvis import Pointset
    
    if graph is None:
        return
    if axes is None:
        a = vv.gca()
    else:
        axes.MakeCurrent()
        a = vv.gca()
    for (n1, n2) in graph.edges():
        if graph.node[n1].get('spared', False) and \
        graph.node[n2].get('spared', False):
            p = graph.edge[n1][n2]['path']
            pp = Pointset(p)
            line = vv.solidLine(pp, radius = radius)
            line.faceColor = 'y'

def snap_picked_point_to_graph(graph, vol, label, nodesOnly=False):
    """ Snap picked point to graph and return point on graph as tuple
    Also return edge of point and its index on this edge
    """
    from stentseg.utils.picker import get_picked_seed
    
    coord = get_picked_seed(vol, label)
    dist = 10000.0
    if nodesOnly == True: # no edges, get node closest to picked point
        for n in sorted(graph.nodes()):
            vec = np.asarray(n) - np.asarray(coord)
            d = (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
            dist = min(dist, d)
            if dist == d:
                point = n
        return point
        
    for n1, n2 in sorted(graph.edges()):
        path = graph.edge[n1][n2]['path']
        vectors = path - np.asarray(coord)
        dists = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5
        dist = min(dist, min(dists))
        if dist == min(dists):
            i = np.where(dists==dist)
            point = path[i]
            edgenodes = n1, n2
    
    return tuple(point.flat), edgenodes, np.asarray(i)

def Event_pick_graph_point(graph, vol, label, nodesOnly=False, axes=None, vis=True):
    """ As snap_picked_point_to_graph but bound to axes with key control
    Use p key to get picked point
    """
    if axes is None:
        axes = vv.gca()
    point = dict()
    # Define callback function
    @axes.eventKeyDown.Bind
    def pointPicked(event):
        if event.text == 'p':
            point['point'] = snap_picked_point_to_graph(graph, vol, label, nodesOnly=nodesOnly)
            p = point['point']
            print('picked cll point:{}'.format(p))
            if vis:
                view = axes.GetView()
                vv.plot(p[0], p[1], p[2], mc= 'r', ms = 'o', mw= 10, alpha=0.5, axes=axes)
                axes.SetView(view)
            return True  # prevent default mouse action
    
    return point
            

def interactiveClusterRemoval(graph, radius=0.7, axVis=False, 
        faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) ):
    """ showGraphAsMesh(graph, radius=0.7, 
                faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) )
    
    Manual delete clusters in the graph. Show the given graph as a mesh, or to 
    be more precize as a set of meshes representing the clusters of the graph. 
    By holding the mouse over a mesh, it can be selected, after which it can be 
    deleted by pressing delete. Use sd._nodes3 for graph when in segmentation.
    
    Returns the axes in which the meshes are drawn.
    
    """
    import visvis as vv
    import networkx as nx
    from stentseg.stentdirect import stentgraph
    from stentseg.stentdirect.stentgraph import create_mesh
    
    # Get clusters of nodes
    clusters = list(nx.connected_components(graph))
    
    # Build meshes 
    meshes = []
    for cluster in clusters:
        g = graph.copy()
        for c in clusters:
            if not c == cluster: 
                g.remove_nodes_from(c)
        
        # Convert to mesh (this takes a while)
        bm = create_mesh(g, radius = radius)
        
        # Store
        meshes.append(bm)
    
    # Define callback functions
    def meshEnterEvent(event):
        event.owner.faceColor = selectColor
    def meshLeaveEvent(event):
        event.owner.faceColor = faceColor
    def figureKeyEvent(event):
        if event.key == vv.KEY_DELETE:
            m = event.owner.underMouse
            if hasattr(m, 'faceColor'):
                m.Destroy()
                graph.remove_nodes_from(clusters[m.index])
     
    # Visualize
    a = vv.gca()
    fig = a.GetFigure()
    for i, bm in enumerate(meshes):
        m = vv.mesh(bm)
        m.faceColor = faceColor
        m.eventEnter.Bind(meshEnterEvent)
        m.eventLeave.Bind(meshLeaveEvent)
        m.hitTest = True
        m.index = i
    # Bind event handlers to figure
    fig.eventKeyDown.Bind(figureKeyEvent)
    a.SetLimits()
    a.bgcolor = 'k'
    a.axis.axisColor = 'w'
    a.axis.visible = axVis
    a.daspect = 1, 1, -1
    
    # Prevent the callback functions from going out of scope
    a._callbacks = meshEnterEvent, meshLeaveEvent, figureKeyEvent
    
    # Done return axes
    return a


from visvis import BaseMapableEditor
from visvis.wibjects.sliders import RangeSlider

class IsoThEditor(BaseMapableEditor):
    """ IsoThEditor(parent, *args) (Adapted from CLimEditor(parent, *args))
    
    A wibject to edit the isothreshold property of an iso rendering
    
    During initialization, SetMapables(*args) is called. The easiest way 
    to use this wibject is to attach it to an axes or figure instance. 
    The wibject then controls the colormaps of all mapable objects in them.
    
    """
    
    def __init__(self, parent, *args):
        BaseMapableEditor.__init__(self, parent)
        
        # init size
        self.position.w = 300
        self.position.h = 50
        
        # Init mappables
        self._mapables = []
        
        # Create slider widget
        self._slider = RangeSlider(self, fullRange=(0,1000))
        self._slider.range = 0, 310 # init
        self._slider.position = 15,5,-30,-30 # 80 = 55+25
        self._slider.showTicks = True
        
        # Bind events
        self._slider.eventSliderChanged.Bind(self._UpdateFull)
        
        # Set mappables
        self.SetMapables(*args)
        # self._InitFromMapables()
    
    
    @property
    def slider(self):
        """ Get the slider instance of this tool.
        """
        return self._slider
    
    def _UpdateFull(self, event):
        for mappable in self.GetMapables():
            ra = self._slider.range
            if hasattr(mappable, 'isoThreshold'):
                mappable.isoThreshold=ra.max
            else:
                print('mappable has no attribute isoThreshold')


