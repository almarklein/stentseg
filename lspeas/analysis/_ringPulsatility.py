""" LSPEAS proximal ring pulsatility and expansion module

Module for obtaining node to node pulsatility in ECG gated CT of Anaconda
Button interaction:
ENTER: calculate distances and pulsatility between selected nodes
ESCAPE: FINISH MODEL, GET MAX PULSATILITY OVERALL, STORE TO EXCEL 

"""

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet, _utils_GUI
from stentseg.stentdirect import stentgraph
from visvis import Pointset # for meshes
from stentseg.stentdirect.stentgraph import create_mesh
from visvis.processing import lineToMesh, combineMeshes
from visvis import ssdf
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes


def on_key(event):
    global node_points 
    if event.key == vv.KEY_DOWN:
        # hide nodes and labels
        t1.visible, t2.visible, t3.visible = False, False, False
        t4.visible, t5.visible, t6.visible = False, False, False
        for node_point in node_points:
            node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes and labels
        t1.visible, t2.visible, t3.visible = True, True, True
        t4.visible, t5.visible, t6.visible = True, True, True
        for node_point in node_points:
            node_point.visible = True
    if event.text == 'n':
        # add clickable point: point on graph closest to picked point (SHIFT+R-click )
        view = a.GetView()
        for node_point in node_points:
            node_point.visible = False
        snapOut = _utils_GUI.snap_picked_point_to_graph(model, vol, label) # x,y,z
        pickedOnGraph = snapOut[0]
        n1, n2 = snapOut[1]
        pickedOnGraphIndex = snapOut[2]
        pickedOnGraphDeforms = model.edge[n1][n2]['pathdeforms'][pickedOnGraphIndex]
        model.add_node(pickedOnGraph, deforms=pickedOnGraphDeforms)
        node_points = _utils_GUI.interactive_node_points(model, scale=0.7)
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, t0=t0)
        # visualize
        # pickedOnGraph_sphere = vv.solidSphere(translation = (pickedOnGraph), scaling = (scale,scale,scale))
        point = vv.plot(pickedOnGraph[0], pickedOnGraph[1], pickedOnGraph[2], 
                        mc = 'y', ms = 'o', mw = 9, alpha=0.5)
        a.SetView(view)
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
            output = point_to_point_pulsatility(selectn1, n1Deforms, selectn2, n2Deforms)
            # update labels
            t1.text = '\b{Node pair}: %i - %i' % (nindex[0], nindex[1])
            t2.text = 'Node-to-node Min: %1.2f mm' % output[0][0]
            t3.text = 'Node-to-node Max: %1.2f mm' % output[4][0]
            t4.text = 'Node-to-node Median: %1.2f mm' % output[2]
            t5.text = 'Node-to-node Q1 and Q3: %1.2f | %1.2f mm' % (output[1], output[3])
            t6.text = '\b{Node-to-node Pulsatility: %1.2f mm}' % (output[5][0] )
            t1.visible, t2.visible, t3.visible = True, True, True
            t4.visible, t5.visible, t6.visible = True, True, True
            # Store output including index/nr of nodes
            output.insert(0, [n1index]) # at the start
            output.insert(1, [n2index])
            output[8].insert(0, [n1index])
            output[9].insert(0, [n2index])
            if output not in storeOutput:
                storeOutput.append(output)
        # Midpoint_to_node analysis
        if len(selected_nodes)== 3:
            # find the edge selected to get midpoint
            selected_nodes2 = selected_nodes.copy()
            for node1 in selected_nodes:
                selected_nodes2.remove(node1) # check combination once and not to self
                for node2 in selected_nodes2:
                    if model.has_edge(node1.node, node2.node):
                        # get midpoint of edge and its deforms
                        output = get_midpoint_deforms_edge(model, node1.node, node2.node)
                        break  # edge found, to first for loop
            # get index of nodepair and midpoint and its deforms
            nodepair1 = output[0]
            midpoint1IndexPath = output[1]
            midpoint1 = output[2]
            midpoint1Deforms = output[3]
            # get node
            for i, node in enumerate(selected_nodes):
                if node.nr not in nodepair1:
                    n3 = node
                    break
            # get deforms for node
            n3Deforms = model.node[n3.node]['deforms']
            # get pulsatility
            # first selected first in output
            if i > 0: # single node was not selected first
                output2 = point_to_point_pulsatility(midpoint1, 
                            midpoint1Deforms, n3.node, n3Deforms)
            else:
                output2 = point_to_point_pulsatility(n3.node, n3Deforms,
                            midpoint1, midpoint1Deforms)
            # visualize midpoint
            view = a.GetView()
            point = vv.plot(midpoint1[0], midpoint1[1], midpoint1[2], 
                            mc = 'm', ms = 'o', mw = 8, alpha=0.5)
            a.SetView(view)
            # update labels
            t1.text = '\b{Node pairs}: (%i %i) - (%i)' % (nodepair1[0],nodepair1[1],n3.nr)
            t2.text = 'Midpoint-to-node Min: %1.2f mm' % output2[0][0]
            t3.text = 'Midpoint-to-node Max: %1.2f mm' % output2[4][0]
            t4.text = 'Midpoint-to-node Median: %1.2f mm' % output2[2]
            t5.text = 'Midpoint-to-node Q1 and Q3: %1.2f | %1.2f mm' % (output2[1], output2[3])
            t6.text = '\b{Midpoint-to-node Pulsatility: %1.2f mm}' % (output2[5][0])
            t1.visible, t2.visible, t3.visible = True, True, True
            t4.visible, t5.visible, t6.visible = True, True, True
            # Store output including index nodes
            if i > 0:
                output2.insert(0, nodepair1) # at the start
                output2.insert(1, [n3.nr])
                output2[8].insert(0, midpoint1IndexPath)
                output2[9].insert(0, [n3.nr])
            else:
                output2.insert(0, [n3.nr]) # at the start
                output2.insert(1, nodepair1)
                output2[8].insert(0, [n3.nr])
                output2[9].insert(0, midpoint1IndexPath)
            if output2 not in storeOutput:
                storeOutput.append(output2)
        # Midpoint_to_midpoint analysis
        if len(selected_nodes) == 4:
            outputs = list()
            # get midpoints for the two edges
            # get nodepairs from order selected
            for i in (0,2):
                n1 = selected_nodes[i].node
                n2 = selected_nodes[i+1].node
                assert model.has_edge(n1, n2)
                # get midpoint of edge and its deforms
                output = get_midpoint_deforms_edge(model, n1, n2)
                midpoint = output[2]
                # store for both edges
                outputs.append(output)
                # visualize midpoint
                view = a.GetView()
                point = vv.plot(midpoint[0], midpoint[1], midpoint[2], 
                                mc = 'm', ms = 'o', mw = 8, alpha=0.5)
                a.SetView(view)
            assert len(outputs) == 2 # two midpoints should be found
            # get midpoints and deforms
            nodepair1 = outputs[0][0]
            midpoint1IndexPath = outputs[0][1]
            midpoint1 = outputs[0][2]
            midpoint1Deforms = outputs[0][3]
            nodepair2 = outputs[1][0]
            midpoint2IndexPath = outputs[1][1]
            midpoint2 = outputs[1][2]
            midpoint2Deforms = outputs[1][3]
            # get pulsatility midp to midp
            output2 = point_to_point_pulsatility(midpoint1, 
                                midpoint1Deforms, midpoint2, midpoint2Deforms)
            # # get max pulsatility between points on the paths
            # outputmaxP.append(edge_to_edge_max_pulsatility(model, nodepair1, nodepair2))
            # update labels
            t1.text = '\b{Node pairs}: (%i %i) - (%i %i)' % (nodepair1[0], nodepair1[1],
                                                            nodepair2[0], nodepair2[1])
            t2.text = 'Midpoint-to-midpoint Min: %1.2f mm' % output2[0][0]
            t3.text = 'Midpoint-to-midpoint Max: %1.2f mm' % output2[4][0]
            t4.text = 'Midpoint-to-midpoint Median: %1.2f mm' % output2[2]
            t5.text = 'Midpoint-to-midpoint Q1 and Q3: %1.2f | %1.2f mm' % (output2[1], output2[3])
            t6.text = '\b{Midpoint-to-midpoint Pulsatility: %1.2f mm}' % (output2[5][0])
            t1.visible, t2.visible, t3.visible = True, True, True
            t4.visible, t5.visible, t6.visible = True, True, True
            # Store output including nodepairs of the midpoints
            output2.insert(0, nodepair1) # indices at the start
            output2.insert(1, nodepair2)
            output2[8].insert(0, midpoint1IndexPath)
            output2[9].insert(0, midpoint2IndexPath)
            if output2 not in storeOutput:
                storeOutput.append(output2)
        # Visualize analyzed nodes and deselect
        for node in selected_nodes:
            node.faceColor = (0,1,0,0.8) #  # make green when analyzed
        selected_nodes.clear()
    if event.key == vv.KEY_ESCAPE:
        # FINISH, STORE TO EXCEL
        # visualize
        view = a.GetView()
        t = vv.volshow(vol, clim=clim, renderStyle='mip')
        # show mesh of model without deform coloring
        modelmesh = create_mesh(model, 0.4)  # Param is thickness
        m = vv.mesh(modelmesh)
        m.faceColor = (0,1,0,1) # green
        a.SetView(view)
        # Store to EXCEL
        storeOutputToExcel(storeOutput,exceldir)
        for node_point in node_points:
            node_point.visible = False # show that store is ready


def get_midpoint_deforms_edge(model, n1, n2):
    """ Get midpoint of a given edge
    Returns output array with index of nodes, index of midpoint on path and 
    midpoint with its deforms
    """
    # get index of nodes which are in fixed order
    n1index = sorted(model.nodes()).index(n1) # sort to match node_point numbering
    n2index = sorted(model.nodes()).index(n2) # sort to match node_point numbering
    nindex = [n1index, n2index]
    # get path
    edge = model.edge[n1][n2]
    path = edge['path']
    # find point closest to mid of line n1 to n2
    mid = (n1[0]+n2[0])/2, (n1[1]+n2[1])/2, (n1[2]+n2[2])/2
    # define vector from points to mid
    v = path - mid
    dist_to_mid = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1)
    # get point on path closest to mid
    midpointIndex =  list(dist_to_mid).index(dist_to_mid.min() ) # index on path
    midpoint = path[midpointIndex]
    # get deforms of midpoint
    midpointDeforms = model.edge[n1][n2]['pathdeforms'][midpointIndex]
    if (len(path) % 2 == 0): #even; middle of 2 closest pathpoints to get actual mid
        dist_to_mid[midpointIndex] = dist_to_mid.max()+100 # replace for distance > max
        midpointIndex2 =  list(dist_to_mid).index(dist_to_mid.min() ) # index on path
        midpoint2 = path[midpointIndex2]
        # if neighbors on this path, find mid point and deforms
        if abs(midpointIndex - midpointIndex2) == 1:
            midpoint = (midpoint + midpoint2) / 2
            midpointDeforms2 = model.edge[n1][n2]['pathdeforms'][midpointIndex2]
            midpointDeforms = (midpointDeforms + midpointDeforms2) / 2
            midpointIndex = [midpointIndex,midpointIndex2]
        else: # take only closest as midpoint
            midpointIndex = [midpointIndex] # return as array, similar to when even pathlength
    else: # odd, expected one pathpoint closest to mid of line
        midpointIndex = [midpointIndex] # return as array, similar to when even pathlength
    midpoint = np.asarray(tuple(midpoint.flat)) if isinstance(midpoint, PointSet) else midpoint # if PointSet make array    
    
    return [nindex, midpointIndex, midpoint, midpointDeforms]


def point_to_point_pulsatility(point1, point1Deforms, 
                                     point2, point2Deforms):
    """ Analyze pulsatility peak_to_peak or valley_to_valley between
    2 given points. Point can be a node or a midpoint, found by
    get_midpoint_deforms_edge. 
    Returns output array with min, Q1, median, Q3, max distance 
    and the change/pulsatility as well as points with deforms and all distances
    """
    n1Indices = point1 + point1Deforms
    n2Indices = point2 + point2Deforms
    # define vector between nodes
    v = n1Indices - n2Indices
    distances = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1)
    # get min and max distance
    point_to_pointMax = distances.max()
    point_to_pointMin = distances.min()
    # add phase in cardiac cycle where min and max where found (5th = 50%)
    point_to_pointMax = [point_to_pointMax, (list(distances).index(point_to_pointMax) )*10]
    point_to_pointMin = [point_to_pointMin, (list(distances).index(point_to_pointMin) )*10]
    # get median of distances
    point_to_pointMedian = np.percentile(distances, 50) # Q2
    # median of the lower half, Q1 and upper half, Q3
    point_to_pointQ1 = np.percentile(distances, 25)
    point_to_pointQ3 = np.percentile(distances, 75)
    # Pulsatility min max distance point to point
    point_to_pointP = point_to_pointMax[0] - point_to_pointMin[0]
    # add % change to pulsatility
    point_to_pointP = [point_to_pointP, (point_to_pointP/point_to_pointMin[0])*100 ]
    return [point_to_pointMin, point_to_pointQ1, point_to_pointMedian,
           point_to_pointQ3, point_to_pointMax, point_to_pointP, [point1,
           point1Deforms], [point2, point2Deforms], distances]


def edge_to_edge_max_pulsatility(model, nodepair1, nodepair2):
    """ Find the max pulsatility between all points on paths of two selected edges
    Input: nodepairs (index) of edges to analyze max pulsatility
    Rationale: get an overall measure of max pulsatility in model. It could be 
    that max pulsatility is not present in direction peak-to-peak, 
    valley-to-valley or mid-to-mid on ring. Good to visualize line of maxP
    """ 
    # get path
    n1_1 = sorted(model.nodes())[nodepair1[0]] # sort to match node_point numbering 
    n2_1 = sorted(model.nodes())[nodepair1[1]] # nodepair = [12,13] for example
    n1_2 = sorted(model.nodes())[nodepair2[0]]
    n2_2 = sorted(model.nodes())[nodepair2[1]]
    edge1 = model.edge[n1_1][n2_1]
    edge2 = model.edge[n1_2][n2_2]
    path1 = edge1['path']
    path2 = edge2['path']
    # Get puls for every combination of points on both edges
    out = list()
    for pointIndex1, point1 in enumerate(path1):
        # get deforms of pathpoint
        pathpoint1Deforms = edge1['pathdeforms'][pointIndex1]
        for pointIndex2, point2 in enumerate(path2):
            pathpoint2Deforms = edge2['pathdeforms'][pointIndex2]
            out.append(point_to_point_pulsatility(point1, pathpoint1Deforms, point2, pathpoint2Deforms))
    out_pulsatility = list()
    for outcome in out:
        out_pulsatility.append(outcome[5][0]) # [5][0] point_to_pointP
    maxpulsatilityIndex = out_pulsatility.index(max(out_pulsatility))
    maxpulsatility_out = out[maxpulsatilityIndex]
    # return output for found max pulsatility pathpoints
    # add nodepair indices and pathpoint indices
    maxpulsatility_out.insert(0, nodepair1) # indices at the start
    maxpulsatility_out.insert(1, nodepair2)
    for pathpoint1Index, pathpoint1 in enumerate(path1):
        if list(pathpoint1) == list(maxpulsatility_out[8][0]): # [8][0] pathpoint1
            maxpulsatility_out[8].insert(0, [pathpoint1Index])
    for pathpoint2Index, pathpoint2 in enumerate(path2):
        if list(pathpoint2) == list(maxpulsatility_out[9][0]): # [9][0] pathpoint2
            maxpulsatility_out[9].insert(0, [pathpoint2Index])
    maxpulsatility_out.append('max pulsatility (edge_to_edge)')
    return maxpulsatility_out


import xlsxwriter
def storeOutputToExcel(storeOutput, exceldir):
    """Create file and add a worksheet or overwrite existing
    """
    # https://pypi.python.org/pypi/XlsxWriter
    workbook = xlsxwriter.Workbook(os.path.join(exceldir,'storeOutputTemplate.xlsx'))
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('A:A', 15)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    # write title
    worksheet.write('A1', 'Pulsatility and expansion', bold)
    analysisID = '%s_%s_%s_%s' % (ptcode, ctcode, cropname, modelname)
    worksheet.write('B3', analysisID, bold)
    # write 'storeOutput'
    rowoffset = 5
    for i, Output in enumerate(storeOutput):
        rowstart = rowoffset # startrow for this Output
        for j, variable in enumerate(Output):
            if j < 8: # write the 1D arrays
                if j in (3,4,5):
                    variable = [variable] # floats can not be stored with write_row
                # write the first variables in list
                worksheet.write_row(rowoffset, 1, variable) # row, columnm, variable
                rowoffset += 1
            elif j in (8,9): # write the 3D arrays with [index; point; deforms]
                rowoffset += 1 # add empty row
                if j == 8:
                    rowoffset += 1 # add empty row for write distances
                worksheet.write_row(rowoffset, 1, variable[0])
                rowoffset += 1
                worksheet.write_row(rowoffset, 1, variable[1])
                rowoffset += 1
                for pointdeform in variable[2]: # deforms
                    pointdeform = tuple(pointdeform.flat) if isinstance(pointdeform, PointSet) else pointdeform
                    worksheet.write_row(rowoffset, 1, pointdeform)
                    rowoffset += 1
            elif j == 10: # write 1D array distances
                worksheet.write_column(rowstart, 3, variable)
            elif j == 11: # write comment for max pulsatility
                worksheet.write_string(rowoffset, 1, variable)
        rowoffset += 2 # add rowspace for next analysis Output

    # Store screenshot of stent
    #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
    workbook.close()


if __name__ == '__main__':
    
    # Select the ssdf basedir
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
    
    # Select location storeOutputTemplate EXCEL file
    exceldir = select_dir(r'C:\Users\Maaike\Desktop',
                r'D:\Profiles\koenradesma\Desktop')
    
    # Select dataset to register
    ptcode = 'LSPEAS_023'
    ctcode = '24months'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    # Load static CT image to add as reference
    s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
    vol = s.vol
    
    # Load the stent model and mesh
    s2 = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
    model = s2.model
    
    
    ## Start visualization and GUI
    
    clim = (0,2500)
    showVol = 'MIP'
    
    fig = vv.figure(); vv.clf()
    fig.position = 9.00, 30.00,  944.00, 1002.00
    a = vv.gca()
    label = DrawModelAxes(vol, model, a, getLabel=True, clim=clim, showVol=showVol, mw=10)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    viewAP = {'zoom': 0.006,'fov': 0.0,
    'daspect': (1.0, 1.0, -1.0),
    'azimuth': 0.0,
    'roll': 0.0}
    
    # Set view
    # a.SetView(viewAP)
    
    # Initialize labels
    t0 = vv.Label(a, '\b{Node nr|location}: ', fontSize=11, color='w')
    t0.position = 0.1, 25, 0.5, 20  # x (frac w), y, w (frac), h
    t0.bgcolor = None
    t0.visible = True
    t1 = vv.Label(a, '\b{Nodepair}: ', fontSize=11, color='w')
    t1.position = 0.1, 45, 0.5, 20  # x (frac w), y, w (frac), h
    t1.bgcolor = None
    t1.visible = False
    t2 = vv.Label(a, 'Node-to-node Min: ', fontSize=11, color='w')
    t2.position = 0.1, 65, 0.5, 20
    t2.bgcolor = None
    t2.visible = False
    t3 = vv.Label(a, 'Node-to-node Max: ', fontSize=11, color='w')
    t3.position = 0.1, 85, 0.5, 20
    t3.bgcolor = None
    t3.visible = False
    t4 = vv.Label(a, 'Node-to-node Median: ', fontSize=11, color='w')
    t4.position = 0.1, 105, 0.5, 20
    t4.bgcolor = None
    t4.visible = False
    t5 = vv.Label(a, 'Node-to-node Q1 and Q3: ', fontSize=11, color='w')
    t5.position = 0.1, 125, 0.5, 20
    t5.bgcolor = None
    t5.visible = False
    t6 = vv.Label(a, '\b{Node-to-node Pulsatility: }', fontSize=11, color='c')
    t6.position = 0.1, 145, 0.5, 20
    t6.bgcolor = None
    t6.visible = False
    
    # Initialize output variable to store pulsatility analysis
    storeOutput = list()
    # outputmaxP = list()
    selected_nodes = list()
    
    # Add clickable nodes
    scale = 0.7
    node_points = _utils_GUI.interactive_node_points(model, scale=0.7)
    
    
    # Bind event handlers
    fig.eventKeyDown.Bind(on_key)
    _utils_GUI.node_points_callbacks(node_points, selected_nodes, t0=t0) # bind callback functions to node points
    fig.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event) )
    
    # Print user instructions
    print('')
    print('n = add node to click from graph point closest to [picked point]')
    print('Enter = get distance between selected nodes and/or midpoints')
    print('Esc = finish analysis, STORE TO EXCEL desktop')
    print('x = axis invisible/visible')

