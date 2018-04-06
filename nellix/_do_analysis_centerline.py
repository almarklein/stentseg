""" Author: M.A. Koenrades
Created October 2017
Module to perform automated analysis of motion on the centerlines
"""
from stentseg.utils.centerline import points_from_nodes_in_graph
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume
from stentseg.utils import PointSet, _utils_GUI, visualization
import visvis as vv
import numpy as np
import copy
import math
import itertools

class _Do_Analysis_Centerline:
    def __init__(self,ptcode,ctcode,basedir,showVol='MIP',clim=(0,2500),**kwargs):
        """
        Init motion analysis on centerlines 
        """
        self.s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelavgreg_deforms_id')
        s = loadvol(basedir, ptcode, ctcode, 'prox', what='avgreg')
        # set sampling for cases where this was not stored correctly
        s.vol.sampling = [s.sampling[1], s.sampling[1], s.sampling[2]]
        vol = s.vol
        self.vol = s.vol
        # figure
        f = vv.figure(1); vv.clf()
        f.position = 0.00, 22.00,  1920.00, 1018.00
        a = vv.gca()
        a.axis.axisColor = 1,1,1
        a.axis.visible = False
        a.bgcolor = 0,0,0
        a.daspect = 1, 1, -1
        t = show_ctvolume(vol, showVol=showVol, clim=clim, **kwargs)
        self.label = pick3d(a, vol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        for key in self.s:
            if key.startswith('model'):
                self.s[key].Draw(mc='b', mw = 5, lc='b', alpha = 0.5)
        vv.title('Model for ChEvas %s  -  %s' % (ptcode[7:], ctcode))
        
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1,a2,a3]) )
        
        # Initialize output variables to store analysis
        self.storeOutput = list()
        self.output_cl = list()
        self.output_clmin_index = list()
        self.output_clmin = list()
        self.output_clmax = list()
        self.output_clmax_index = list()
        
        self.node_points = []
    
    def createNodePoint(self, node, color='g'):
        node_point = vv.solidSphere(translation = (node), scaling = (0.6,0.6,0.6))
        node_point.faceColor = color
        node_point.alpha = 0.5
        node_point.visible = True
        node_point.node = node
        # node_point.nr = i
        self.node_points.append(node_point)
    
    def motion_points(self, lenSegment=5, dim='xyz'):
        """ given a centerline, compute motion of points in centerline segment
        dim: x,y,z,xyz 
        """
        
    
    def distance_points(self):
        """ distances cardiac cycle between 2 points
        """
        
    
    def chimney_angle_change(self):
        """calc angle change for each chimney stent
        """
        s = self.s # ssdf with centerline models identified
        for key in dir(s):
            if key.startswith('modelChR'):
                nodesChR = sorted(s[key].nodes())
                outChR = self.centerline_angle_change(key,nodesChR, 'Ang_ChR')
            if key.startswith('modelChL'):
                nodesChL = sorted(s[key].nodes())
                outChL = self.centerline_angle_change(key,nodesChL, 'Ang_ChL')
            if key.startswith('modelSMA'):
                nodesSMA = sorted(s[key].nodes())
                outSMA = self.centerline_angle_change(key,nodesSMA, 'Ang_SMA')
    
    def centerline_angle_change(self, key, nodesCh, name_output):
        """ Calculate the angle change of the given centerline
        nodesCh is a list of sorted (centerline) nodes 
        """
        modelnodes = nodesCh
        # n1 = modelnodes[0] # does not work: sorted nodes is not same order as pp centerline
        ends = []
        for n in modelnodes:
            if self.s[key].degree(n) == 1:
                ends.append(n)
        if len(ends) > 2:
            raise RuntimeError('Centerline has more than 2 nodes with 1 neighbour')
        n1 = ends[0]
        n3 = ends[1]
        
        self.createNodePoint(n1, color='b')
        self.createNodePoint(n3)
        
        # find midpoint that makes greatest angle
        # todo: find midpoint that has greatest angle change or calc angle per node as in stentgraph _detect_corners
        angle = 0 # straigth
        for i, n2 in enumerate(modelnodes[10:-10]): # omit first 5 nodes (check stepsize cll)
            # calculate vectors          
            vec1 = PointSet(np.column_stack(n1))-PointSet(np.column_stack(n2))
            vec2 = PointSet(np.column_stack(n2))-PointSet(np.column_stack(n3))
            # calc angle
            phi = abs(vec1.angle(vec2))
            anglenew = phi*180.0/np.pi # direction vector in degrees
            if anglenew > angle:
                print(anglenew)
                midnode = copy.copy(n2)
            angle = max(angle, anglenew)
            
        # show midnode
        print(midnode)
        self.createNodePoint(midnode, color='m')
            
        nindex = [0, 0, 0] # not applicable so set 0, but leave for reuse of old output code
        # get deforms of nodes
        model = self.s[key]
        n1Deforms = model.node[n1]['deforms']
        n2Deforms = model.node[midnode]['deforms']
        n3Deforms = model.node[n3]['deforms']
        # get angulation
        output = line_line_angulation(n1, 
                            n1Deforms, midnode, n2Deforms, n3, n3Deforms)
        output['NodesIndex'] = nindex
        
        # Store output with name
        output['Name'] = name_output
        self.storeOutput.append(output)


def line_line_angulation(point1, point1Deforms, point2, point2Deforms, point3, point3Deforms):
    n1Indices = point1 + point1Deforms
    n2Indices = point2 + point2Deforms
    n3Indices = point3 + point3Deforms
    
    # get vectors
    v1 = n1Indices - n2Indices
    v2 = n3Indices - n2Indices
    
    # get angles
    angles = []
    for i in range(len(v1)):
        angles.append(math.degrees(math.acos((np.dot(v1[i],v2[i]))/
        (np.linalg.norm(v1[i])*np.linalg.norm(v2[i])))))
    angles = np.array(angles)
    
    # get all angle differences of all phases
    pos_combinations = list(itertools.combinations(range(len(v1)),2))
    angle_diff = []
    for i in pos_combinations:
        v = point1Deforms[i[0]] - point1Deforms[i[1]]
        angle_diff.append(abs(angles[i[0]] - angles[i[1]]))
    angle_diff = np.array(angle_diff)
    
    # get max angle differences
    point_angle_diff_max = angle_diff.max()
    point_angle_diff_max = [point_angle_diff_max, [x*10 for x in
    (pos_combinations[list(angle_diff).index(point_angle_diff_max)])]]
    
    # get min angle differences
    point_angle_diff_min = angle_diff.min()
    point_angle_diff_min = [point_angle_diff_min, [x*10 for x in 
    (pos_combinations[list(angle_diff).index(point_angle_diff_min)])]]
    
    return {'point_angle_diff_min':point_angle_diff_min,
    'point_angle_diff_max': point_angle_diff_max, 'angles': angles, 
    'Node1': [point1, point1Deforms], 'Node2': [point2, point2Deforms], 
    'Node3': [point3, point1Deforms]}

