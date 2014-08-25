""" LSPEAS proximal ring pulsatility module

Module for obtaining node tot node pulsatility in ECG gated CT of Anaconda

"""

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet
import pirt
from stentseg.stentdirect import stentgraph


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_001'
ctcode = '1month'
cropname = 'ring'

# Load the stent model and mesh
s2 = loadmodel(basedir, ptcode, ctcode, cropname)
model = s2.model

## Start visualization and GUI

fig = vv.figure(1); vv.clf()
fig.position = 0.00, 22.00,  1366.00, 706.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
model.Draw(mc='b', mw = 10, lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
viewringcrop = {'azimuth': -69.53988383521587,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 13.60706860706862,
 'fov': 0.0,
 'loc': (101.44986220914902, 83.5231281276915, 55.32349467185847),
 'roll': 0.0,
 'zoom': 0.04051242023573491}

# Add clickable nodes
node_points = []
for i, node in enumerate(model.nodes()):
    node_point = vv.solidSphere(translation = (node), scaling = (0.4,0.4,0.4))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    node_points.append(node_point)

# Initialize labels
t0 = vv.Label(a, '\b{Node nr}: ', fontSize=11, color='b')
t0.position = 0.1, 5, 0.5, 20  # x (frac w), y, w (frac), h
t0.bgcolor = None
t0.visible = True
t1 = vv.Label(a, '\b{Nodepair}: ', fontSize=11, color='b')
t1.position = 0.1, 25, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a, 'Node-to-node Min: ', fontSize=11, color='b')
t2.position = 0.1, 45, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a, 'Node-to-node Max: ', fontSize=11, color='b')
t3.position = 0.1, 65, 0.5, 20
t3.bgcolor = None
t3.visible = False
t4 = vv.Label(a, 'Node-to-node Mean|Std: ', fontSize=11, color='b')
t4.position = 0.1, 85, 0.5, 20
t4.bgcolor = None
t4.visible = False
t5 = vv.Label(a, 'Node-to-node Pulsatility: ', fontSize=11, color='b')
t5.position = 0.1, 105, 0.5, 20
t5.bgcolor = None
t5.visible = False
#todo: perhaps use plot table instead: http://stackoverflow.com/questions/8524401/how-can-i-place-a-table-on-a-plot-in-matplotlib

# Initialize output variable to store pulsatility analysis
storeOutput = list()

def on_key(event): 
    if event.key == vv.KEY_DOWN:
        # hide nodes
        t1.visible = False
        t2.visible = False
        t3.visible = False
        t4.visible = False
        t5.visible = False
        for node_point in node_points:
            node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes
        for node_point in node_points:
            node_point.visible = True
    if event.key == vv.KEY_ENTER:
        assert len(selected_nodes) == 2 or 3 or 4
        # Node_to_node analysis
        if len(selected_nodes) == 2:
            # get nodes
            selectn1 = selected_nodes[0].node
            selectn2 = selected_nodes[1].node
            # get index of nodes which are in fixed order
            n1index = selected_nodes[0].nr
            n2index = selected_nodes[1].nr
            nindex = [n1index, n2index]
            # get deforms of nodes
            n1Deforms = model.node[selectn1]['deforms']
            n2Deforms = model.node[selectn2]['deforms']
            # get pulsatility
            output = point_to_point_pulsatility(model, selectn1, 
                                n1Deforms, selectn2, n2Deforms)
            # update labels
            t1.text = '\b{Node pair}: %i - %i' % (nindex[0], nindex[1])
            t2.text = 'Node-to-node Min: %1.2f mm' % output[0][1]
            t3.text = 'Node-to-node Max: %1.2f mm' % output[1][1]
            t4.text = 'Node-to-node Mean|Std: %1.2f +/- %1.2f mm' % (output[2], output[3])
            t5.text = 'Node-to-node Pulsatility: %1.2f mm (%1.2f %%)' % (output[4][0], output[4][1] )
            t1.visible = True
            t2.visible = True
            t3.visible = True
            t4.visible = True
            t5.visible = True
            # Store output including index/nr of nodes
            output.insert(0, nindex) # at the start
            if output not in storeOutput:
                storeOutput.append(output)
        # Midpoint_to_midpoint or midpoint_to_node analysis
        if len(selected_nodes) == 3 or 4:
            # find the edge(s) selected to analyze
            selected_nodes2 = selected_nodes.copy()
            outputs = list()
            for node1 in selected_nodes:
                selected_nodes2.remove(node1)  # check combination once
                for node2 in selected_nodes2:
                    if model.has_edge(node1.node, node2.node):
                        # get midpoint of edge and its deforms
                        output = get_midpoint_deforms_edge(model, node1.node, node2.node)
                        midpoint = output[1]
                        # store for both edges
                        outputs.append(output)
                        # visualize midpoint
                        view = a.GetView()
                        point = vv.plot(midpoint[0], midpoint[1], midpoint[2], 
                                        mc = 'm', ms = '.', mw = 8)
                        a.SetView(view)
                        break  # edge found, start from first for loop 
            if len(selected_nodes) == 4:
                # Get pulsatility midpoint to midpoint
                # get midpoints and deforms
                nodepair1 = outputs[0][0]
                midpoint1 = outputs[0][1]
                midpoint1Deforms = outputs[0][2]
                nodepair2 = outputs[1][0]
                midpoint2 = outputs[1][1]
                midpoint2Deforms = outputs[1][2]
                # get pulsatility
                output2 = point_to_point_pulsatility(model, midpoint1, 
                                    midpoint1Deforms, midpoint2, midpoint2Deforms)
                # update labels
                t1.text = '\b{Node pairs}: (%i %i) - (%i %i)' % (nodepair1[0], nodepair1[1],
                                                                nodepair2[0], nodepair2[1])
                t2.text = 'Midpoint-to-midpoint Min: %1.2f mm' % output2[0][1]
                t3.text = 'Midpoint-to-midpoint Max: %1.2f mm' % output2[1][1]
                t4.text = 'Midpoint-to-midpoint Mean|Std: %1.2f +/- %1.2f mm' % (output2[2], output2[3])
                t5.text = 'Midpoint-to-midpoint Pulsatility: %1.2f mm (%1.2f %%)' % (output2[4][0], output2[4][1] )
                t1.visible = True
                t2.visible = True
                t3.visible = True
                t4.visible = True
                t5.visible = True
                # Store output including nodepairs of the midpoints
                output2.insert(0, [nodepair1, nodepair2]) # at the start
                if output2 not in storeOutput:
                    storeOutput.append(output2)
            if len(selected_nodes)== 3:
                # Get pulsatility from midpoint to node
                # get index of nodepair and midpoint and its deforms
                nodepair1 = outputs[0][0]
                midpoint1 = outputs[0][1]
                midpoint1Deforms = outputs[0][2]
                # get node
                for node in selected_nodes:
                    if node.nr not in nodepair1:
                        n3 = node
                # get deforms for node
                n3Deforms = model.node[n3.node]['deforms']
                # get pulsatility
                output2 = point_to_point_pulsatility(model, midpoint1, 
                                    midpoint1Deforms, n3.node, n3Deforms)
                # update labels
                t1.text = '\b{Node pairs}: (%i %i) - (%i)' % (nodepair1[0],nodepair1[1],n3.nr)
                t2.text = 'Midpoint-to-node Min: %1.2f mm' % output2[0][1]
                t3.text = 'Midpoint-to-node Max: %1.2f mm' % output2[1][1]
                t4.text = 'Midpoint-to-node Mean|Std: %1.2f +/- %1.2f mm' % (output2[2], output2[3])
                t5.text = 'Midpoint-to-node Pulsatility: %1.2f mm (%1.2f %%)' % (output2[4][0],output2[4][1] )
                t1.visible = True
                t2.visible = True
                t3.visible = True
                t4.visible = True
                t5.visible = True
                # Store output including index nodes
                output2.insert(0, [nodepair1, n3.nr]) # at the start
                if output2 not in storeOutput:
                    storeOutput.append(output2)
        # visualize analyzed nodes and deselect
        for node in selected_nodes:
            node.faceColor = 'g'  # make green when analyzed
        selected_nodes.clear()
                    
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

def pick_node(event):
    nodenr = event.owner.nr
    t0.text = '\b{Node nr}: %i' % nodenr

def unpick_node(event):
    t0.text = '\b{Node nr}: '

def get_midpoint_deforms_edge(model, n1, n2):
    """ Get midpoint of a given edge
    Returns output array with index of nodes, midpoint and its deforms
    """
    # get index of nodes which are in fixed order
    n1index = model.nodes().index(n1)
    n2index = model.nodes().index(n2)
    nindex = [n1index, n2index]
    # get path
    edge = model.edge[n1][n2]
    path = edge['path']
    # find median of 3d points on path in x,y,z
    mid = np.median(path, axis = 0)
    # define vector from points to mid
    v = path - mid
    dist_to_mid = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1)
    # get point on path closest to mid
    midpointIndex =  list(dist_to_mid).index(dist_to_mid.min() )
    midpoint = path[midpointIndex]
    # get deforms of midpoint
    midpointDeforms = model.edge[n1][n2]['pathdeforms'][midpointIndex]
    return [nindex, midpoint, midpointDeforms]

def point_to_point_pulsatility(model, point1, point1Deforms, 
                                     point2, point2Deforms):
    """ Analyze pulsatility peak_to_peak or valley_to_valley between
    2 given points. Point can be a node or a midpoint, found by
    get_midpoint_deforms_edge. 
    Returns output array with min, max and mean distance with std 
    and the change/pulsatility
    """
    n1Indices = point1 + point1Deforms
    n2Indices = point2 + point2Deforms
    # define vector between nodes
    v = n1Indices - n2Indices
    distance = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1)
    point_to_pointMax = distance.max()
    point_to_pointMin = distance.min()
    # add phase in cardiac cycle where min and max where found (5th = 50%)
    point_to_pointMax = [(list(distance).index(point_to_pointMax) )*10, point_to_pointMax,]
    point_to_pointMin = [(list(distance).index(point_to_pointMin) )*10, point_to_pointMin]
    point_to_pointMean = distance.mean()
    point_to_pointStd = distance.std()
    # Pulsatility min max distance point to point
    point_to_pointP = point_to_pointMax[1] - point_to_pointMin[1]
    # add % change to pulsatility
    point_to_pointP = [point_to_pointP, (point_to_pointP/point_to_pointMin[1])*100 ]
    return [point_to_pointMin, point_to_pointMax, point_to_pointMean,
           point_to_pointStd, point_to_pointP]
    
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventDoubleClick.Bind(select_node)
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)

# Set view
a.SetView(viewringcrop)


