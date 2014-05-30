""" Module to make the stent module dynamic and then do some further
calculations on the model. 
"""

import os, sys, time
import gc

import numpy as np
import visvis as vv

import pirt

from stentseg.stentdirect import stentGraph
from stentseg.utils import PointSet


def incorporate_motion(g, deforms, origin):
    """ Incorporate motion in the graph stent module. An attribute
    _deforms is added to each node the graph.
    
    An attribute _deforms is added to the graph, which contains the 
    deformation at the nodes, but organized by deform.
    
    """
    
    # Get a list of nodes (nodes in fixed order)
    nodes = g.nodes()  
    
    # Turn this into a pointset
    g_nodes = PointSet(3, dtype='float32')
    [g_nodes.append(*p) for p in g.nodes()]
    
    # Create deformation for all nodes in the graph
    # todo: perhaps pirt *should* be aware of the origin!
    # it isn't now, so we need to take it into account here
    g_deforms = []
    samplePoints = g_nodes - PointSet([o for o in reversed(origin)], dtype='float32')
    for deform in deforms:
        delta_z = deform.get_field_in_points(samplePoints, 0).reshape(-1, 1)
        delta_y = deform.get_field_in_points(samplePoints, 1).reshape(-1, 1)
        delta_x = deform.get_field_in_points(samplePoints, 2).reshape(-1, 1)
        delta = PointSet( np.concatenate((delta_x, delta_y, delta_z), axis=1) )
        g_deforms.append(delta)
    
    # Attach deformation to each node.
    # Needed because the nodes do not know their own index
    for i, node in enumerate(nodes):
        deforms = PointSet(3, dtype='float32')
        for j in range(len(g_deforms)):
            deforms.append(g_deforms[j][i])
        g.add_node(node, deforms=deforms)
    
#     # Attach list that is sorted by deformation to the graph
#     # Convenient in other situations
#     # When storing, we can only store deforms per node, and calculate
#     # the other representation when loading. 
#     g._deforms = g_deforms


def calculate_angle_changes(g):
    """ Given a graph (which should have incorporated motion), calculate
    the angle changes per deformation. That is, the angle change with
    respect to the reference frame.
    
    This function attaches to each node in the graph two atrributes:
      * _angleChanges: the angle change for each deformation
      * _angleChange: the maximum value in the above list
    
    """
    
    # Calculate angular change for all nodes ...
    for node1 in g:
        
        # Init angle changes for each deform
        angleChanges = [0.0 for i in range(len(node1._deforms))]
        
        # For each combination of two edges ...
        for edge2 in node1._edges:
            node2 = edge2.GetOtherEnd(node1)
            for edge3 in node1._edges:
                if edge2 is edge3:
                    continue
                node3 = edge3.GetOtherEnd(node1)
                
                # Calculate angle
                angle = (node1-node2).angle(node1-node3) # order does not matter
                
                angles = []
                for i in range(len(node1._deforms)):
                    p1 = node1 + node1._deforms[i]
                    p2 = node2 + node2._deforms[i]
                    p3 = node3 + node3._deforms[i]
                    newAngle = (p2-p1).angle(p3-p1)
                    angles.append( newAngle )
                
                # Calculate angular change and get max 
                # (the max with respect to different node combinations)
                angleChanges2 = [abs(float(angle-newAngle)) for newAngle in angles]              
                angleChanges = [max(a1, a2) for a1, a2 in 
                                            zip(angleChanges,angleChanges2)]
        
        # Store found value, convert to degrees
        node1._angleChanges = [a * 180 / np.pi for a in angleChanges]
        # Also store max value
        node1._angleChange = max(node1._angleChanges)


def get_deform_in_nodes_at_sub_index(f, g):
    """ Given a factor and a graph (with motion incorporated), calculate
    the deformation at each node, at the index f, where f may be
    non-integer.
    
    Returns an Nx3 array with the deforms for each node.
    
    """
    
    g_deforms = g._deforms
    
    nphases = len(g_deforms)
    def correctIndex(i):
        if i >= nphases:
            i -= nphases
        if i < 0:
            i += nphases
        return i
    
    ii = int(f)
    t = f - ii
    i0 = correctIndex(ii-1)
    i1 = correctIndex(ii-0)
    i2 = correctIndex(ii+1)
    i3 = correctIndex(ii+2)
    #print f, t, i0, i1, i2, i3
    
    # Get coefficients
    w0, w1, w2, w3 = pirt.get_cubic_spline_coefs(t, 'C')
    
    # Calculate delta vectors for this still
    delta = None
    for i, w in zip([i0, i1, i2, i3], [w0, w1, w2, w3]):
        if delta is None:
            delta = w * g_deforms[i]
        else:
            delta = delta + w * g_deforms[i]
    
    return delta

