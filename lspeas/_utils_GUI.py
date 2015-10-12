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

def create_node_points(graph):
    """ create node objects for gui
    """
    node_points = []
    for i, node in enumerate(sorted(graph.nodes())):
        node_point = vv.solidSphere(translation = (node), scaling = (0.4,0.4,0.4))
        node_point.faceColor = 'b'
        node_point.visible = False
        node_point.node = node
        node_point.nr = i
        node_points.append(node_point)
    return node_points
