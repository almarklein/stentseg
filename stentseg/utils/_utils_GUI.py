""" Functionality for analysis GUI model


"""

import visvis as vv

def get_edge_attributes(model, n1, n2):
    """
    returns cost, ctvalue, path, edge length (mm)
    """ 
    from stentseg.stentdirect.stent_anaconda import _edge_length
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

def create_node_points(graph, scale=0.4):
    """ create node objects for gui
    """
    node_points = []
    for i, node in enumerate(sorted(graph.nodes())):
        node_point = vv.solidSphere(translation = (node), scaling = (scale,scale,scale))
        node_point.faceColor = 'b'
        node_point.visible = False
        node_point.node = node
        node_point.nr = i
        node_points.append(node_point)
    return node_points

def create_node_points_with_amplitude(graph, scale =0.4):
    """ create node objects for gui and calculate motion amplitude for each node
    """
    from stentseg.motion.displacement import _calculateAmplitude
    pointsDeforms = []
    node_points = []
    for i, node in enumerate(sorted(graph.nodes())):
        node_point = vv.solidSphere(translation = (node), scaling = (scale,scale,scale))
        node_point.faceColor = 'b'
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

def vis_spared_edges(graph, radius = 0.6):
    """ in step 3 with anacondaRing, prune_redundant spares strong triangle edges.
        visualize with a model
    """ 
    from visvis import Pointset
    
    a = vv.gca()
    for (n1, n2) in graph.edges():
        if graph.node[n1].get('spared', False) and \
        graph.node[n2].get('spared', False):
            p = graph.edge[n1][n2]['path']
            pp = Pointset(p)
            line = vv.solidLine(pp, radius = radius)
            line.faceColor = 'y'


