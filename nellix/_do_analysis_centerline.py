""" Author: M.A. Koenrades
Created October 2017
Module to perform automated motion analysis of dynamic centerline models
"""
import sys, os
from stentseg.utils.centerline import points_from_nodes_in_graph, dist_over_centerline
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume
from stentseg.utils import PointSet, _utils_GUI, visualization
import visvis as vv
import numpy as np
import copy
import math
import itertools
from stentseg.motion.displacement import calculateMeanAmplitude
import xlsxwriter
from datetime import datetime


class _Do_Analysis_Centerline:
    """ Analyze motion of dynamic centerline models
    Functions for chimneys/branches and main stents
    """
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
        self.ptcode = ptcode
        self.ctcode = ctcode
        self.cropname = 'prox'
        # figure
        f = vv.figure(1); vv.clf()
        f.position = 0.00, 22.00,  1920.00, 1018.00
        self.a = vv.gca()
        self.a.axis.axisColor = 1,1,1
        self.a.axis.visible = False
        self.a.bgcolor = 0,0,0
        self.a.daspect = 1, 1, -1
        t = show_ctvolume(vol, showVol=showVol, clim=clim, removeStent=False, climEditor=True, **kwargs)
        self.label = pick3d(self.a, vol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        for key in self.s:
            if key.startswith('model'):
                self.s[key].Draw(mc='b', mw = 5, lc='b', alpha = 0.5)
        vv.title('Model for ChEvas %s  -  %s' % (ptcode[7:], 'follow-up')) # ctcode not correct
        
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [self.a]) )
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [self.a]) )
        self.fig = f
        
        # Initialize output variables to store analysis
        self.storeOutput = list()
        self.exceldirOutput = os.path.join(basedir,ptcode)
        
        self.node_points = []
        self.points_plotted = []
    
    def createNodePoint(self, node, color='g', scalingfactor=0.7):
        """ Plot visvis object solidSpehere for nodepoint
        Visibilty of these objects is limited in MIP and not visible in ISO
        """
        node_point = vv.solidSphere(translation = (node), scaling = (scalingfactor,scalingfactor,scalingfactor))
        node_point.faceColor = color
        node_point.alpha = 0.1 # 0.5
        node_point.visible = True
        node_point.node = node
        # node_point.nr = i
        self.node_points.append(node_point)
    
    def motion_centerlines_segments(self, lenSegment=5, dim='xyz'):
        """ given a centerline, compute motion of points in centerline segment
        dim: amplitude of motion in x,y,z, or xyz 
        self has ssdf with dynamic model of cll and ppCenterline
        stores mean displacement pattern of segment points and amplitude mean std min max 
        """
        s = self.s # ssdf with centerline pointsets pp identified
        for key in s:
            if key.startswith('ppCenterline'): # each branch or Nel
                ppCll = s[key]
                name_output = 'Motion_'+key[12:]
                model = s['model'+key[12:]] # skip 'ppCenterline' in key
                assert model.number_of_edges() == 1 # a centerline is one edge
                edge = model.edges()[0]
                ppCllDeforms = model.edge[edge[0]][edge[1]]['pathdeforms']
                ppCllDeforms = np.asarray(ppCllDeforms) # npoints x nphases x 3
                #ppCll == model.edge[edge[0]][edge[1]]['path']
                output = calculate_motion_points(ppCll, ppCllDeforms, lenSegment,dim=dim)
                output['Type'] = 'motion_centerlines_segments' # n1index is old code
                # Store output with name
                output['Name'] = name_output
                self.storeOutput.append(output)
                # visualize segment analyzed in avgreg
                pp = output['ppSegment'] # [positions nodes avgreg]
                a1 = self.a
                point = plot_points(pp, mc='g', ax=a1)
                self.points_plotted.append(point)
    
    def distance_change_nelnel_nelCh(self):
        """ Distances cardiac cycle between the proximal points for:
        Nellix to Nellix stent (NelL to NelR)
        Nellix left (NelL) to Chimney left (LRA)
        Nellix right (NelR) to Chimney right (RRA)
        Stores output in centerlines_prox_distance_change()
        """
        s = self.s # ssdf with centerline pointsets pp identified
        for key in s:
            ppNelR = s['ppCenterlineNelR']
            ppNelL = s['ppCenterlineNelL']
            if key.startswith('ppCenterlineRRA'):
                ppRRA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppRRA, ppNelR, ppNelL)
                # ppNelR = s['ppCenterlineNelR']
                # key2 = 'NelR'
                self.centerlines_prox_distance_change(key1, key2, ppRRA, ppNel, 'Dist_{}_{}'.format(key1,key2),color='y')
            if key.startswith('ppCenterlineLRA'):
                ppLRA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppLRA, ppNelR, ppNelL)
                # ppNelL = s['ppCenterlineNelL']
                # key2 = 'NelL'
                self.centerlines_prox_distance_change(key1, key2, ppLRA, ppNel, 'Dist_{}_{}'.format(key1,key2),color='c')
            if key.startswith('ppCenterlineSMA'):
                ppSMA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppSMA, ppNelR, ppNelL)
                self.centerlines_prox_distance_change(key1, key2, ppSMA, ppNel, 'Dist_{}_{}'.format(key1,key2),color='m')
            if key.startswith('ppCenterlineNelL'):
                ppNelL = s[key]
                key1 = key[12:]
                ppNelR = s['ppCenterlineNelR']
                key2 = 'NelR'
                self.centerlines_prox_distance_change(key1, key2, ppNelL, ppNelR, 'Dist_{}_{}'.format(key1,key2), 
                                                      mw=17,color='r',marker='^',alpha=0.7)
                
    def get_nellix_closest_to_chimney(self, ppCh, ppNelR, ppNelL):
        """ Return pp of nellix closest to chimney, proximal points
        """
        proxpCh    = get_prox_dist_points_cll(ppCh)[0] # [0] for prox point
        proxpNelR  = get_prox_dist_points_cll(ppNelR)[0]
        proxpNelL  = get_prox_dist_points_cll(ppNelL)[0]
        
        v = proxpCh - proxpNelR
        distNelR = (v[0]**2 + v[1]**2 + v[2]**2)**0.5 
        v = proxpCh - proxpNelL
        distNelL = (v[0]**2 + v[1]**2 + v[2]**2)**0.5 
        
        if distNelR < distNelL:
            return ppNelR, 'NelR'
        
        else:
            return ppNelL, 'NelL'
        
    
    def centerlines_prox_distance_change(self, key1, key2, ppCll1, ppCll2, name_output,color,mw=15,marker='o',**kwargs):
        """ Calculate distances during the cardiac cycle between proximal point 
        of two given centerlines
        """
        # Get prox point for ppCll1
        out = get_prox_dist_points_cll(ppCll1)
        proxp1 = out[0]
        # Get prox point for ppCll2
        out2 = get_prox_dist_points_cll(ppCll2)
        proxp2 = out2[0]
        
        # get deforms of these prox points
        model1 = self.s['model'+key1] # skip 'ppCenterline' in key
        assert model1.number_of_nodes() > 2 # all points were also added as nodes
        n1 = tuple(proxp1.flat)
        proxp1Deforms = model1.node[n1]['deforms'] # nphases x 3
        
        model2 = self.s['model'+key2] # skip 'ppCenterline' in key
        assert model2.number_of_nodes() > 2 # all points were also added as nodes
        n2 = tuple(proxp2.flat)
        proxp2Deforms = model2.node[n2]['deforms'] # nphases x 3
        
        # visualize prox points
        a1 = self.a
        pp = np.array([proxp1, proxp2])
        points = plot_points(pp, mc=color, mw=mw, ms=marker, ax=a1, **kwargs)
        self.points_plotted.append(points)
        # self.createNodePoint(n1, color=color)
        # self.createNodePoint(n2, color=color)
        
        # get distance change during cycle for these prox points
        output = point_to_point_distance_change(proxp1, proxp1Deforms, proxp2, proxp2Deforms)
        
        # add chimney to nel distance mid cardiac cycle
        v = proxp1 - proxp2
        distanceMidCycle = (v[0]**2 + v[1]**2 + v[2]**2)**0.5 
        output['distMidCycle'] = distanceMidCycle # in avgreg
        
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'centerlines_prox_distance_change'
        self.storeOutput.append(output)
        
    
    def chimneys_angle_change(self):
        """Calculate angle change for each chimney stent
        """
        s = self.s # ssdf with centerline pointsets pp identified
        for key in s:
            if key.startswith('ppCenterlineRRA'):
                ppRRA = s[key]
                key1 = key[12:]
                self.centerline_angle_change(key1, ppRRA, 'Ang_RRA')
                self.centerline_tortuosity_change(key1, ppRRA, 'Tort_RRA')
            if key.startswith('ppCenterlineLRA'):
                ppLRA = s[key]
                key1 = key[12:]
                self.centerline_angle_change(key1, ppLRA, 'Ang_LRA')
                self.centerline_tortuosity_change(key1, ppLRA, 'Tort_LRA')
            if key.startswith('ppCenterlineSMA'):
                ppSMA = s[key]
                key1 = key[12:]
                self.centerline_angle_change(key1, ppSMA, 'Ang_SMA')
                self.centerline_tortuosity_change(key1, ppSMA, 'Tort_SMA')
    
    def centerline_tortuosity_change(self, key, ppCh, name_output):
        """ Tortuosity of pp centerline during cardiac cycle phases
        """
        model = self.s['model'+key] # skip 'ppCenterline' in key
        assert model.number_of_edges() == 1 # a centerline is one edge
        edge = model.edges()[0]
        ppChDeforms = model.edge[edge[0]][edge[1]]['pathdeforms']
        ppChDeforms = np.asarray(ppChDeforms) # npoints x nphases x 3
        #ppCh == model.edge[edge[0]][edge[1]]['path']
        output = calculate_tortuosity_change(ppCh, ppChDeforms)
        
        # add chimney tort mid cardiac cycle
        output['tortMidCycle'] = calculate_tortuosity(ppCh) # in avgreg
        
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'centerline_tortuosity_change'
        self.storeOutput.append(output)

    def centerline_angle_change(self, key, ppCh, name_output):
        """ Calculate the angle change of the given centerline
        ppCh is a PointSet of the CLL, which is in correct order 
        """
        pends = np.array([ppCh[0], ppCh[-1]])  # pp centerline is in order of centerline points
        pends = pends[pends[:,-1].argsort() ] # sort with z, ascending
        proxp = pends[0]   # smallest z; origin is prox
        distp = pends[-1] 
        
        a1 = self.a
        mw =18
        point1 = plot_points(proxp, mc='b', mw=mw, ax=a1)
        self.points_plotted.append(point1)
        point2 = plot_points(distp, mc='g', mw=mw, ax=a1)
        self.points_plotted.append(point2)
        # self.createNodePoint(tuple(proxp), color='b')
        # self.createNodePoint(tuple(distp))
        
        # find midpoint that makes greatest angle = clinically relevant for kink/fracture
        # todo: find midpoint that has greatest angle change? but maybe not likely nor relevant angle
        # todo: or calc angle per node as in stentgraph _detect_corners?
        angle = 180 # straigth
        # todo: need to correct for left or right oriented, 180 or 0 = straight ?
        nNodesToSkip = 5
        for i, n2 in enumerate(ppCh[nNodesToSkip:-nNodesToSkip]): # omit first x nodes (check stepsize cll)
            # calculate vectors  
            vec1, vec2 = PointSet(3), PointSet(3)        
            vec1.append(proxp-n2)
            vec2.append(distp-n2)
            # calc angle
            phi = abs(vec1.angle(vec2))
            anglenew = phi*180.0/np.pi # direction vector in degrees
            print('Angle for points along cll= {}'.format(anglenew) )
            if anglenew < angle:
                # print(anglenew)
                midnode = [copy.copy(n2), copy.copy(i+nNodesToSkip)]
            angle = min(angle, anglenew)
            
        # show midnode
        print('Detected midnode: location {} and angle {}'.format(midnode, angle))
        n2 = midnode[0]
        point3 = plot_points(n2, mc='m', mw=mw, ax=a1)
        self.points_plotted.append(point3)
        # self.createNodePoint(tuple(n2), color='m')
            
        # nindex =  [0, 0, 0] # not applicable so set 0, but leave for reuse of old output code
        # get deforms of nodes during cardiac cycle
        model = self.s['model'+key] # skip 'ppCenterline' in key
        n1Deforms = model.node[tuple(proxp)]['deforms']
        n2Deforms = model.node[tuple(n2)]['deforms']
        n3Deforms = model.node[tuple(distp)]['deforms']
        # get angulation
        output = line_line_angulation(proxp, 
                            n1Deforms, n2, n2Deforms, distp, n3Deforms)
        
        # add chimney angle mid cardiac cycle
        output['angleMidCycle'] = angle
        
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'centerline_angle_change'
        self.storeOutput.append(output)


    def storeOutputToExcel(self):
        """Create file and add a worksheet or overwrite existing
        Output of x,y,z positions in one cell can be handled in python with
        np.asrray(a), where a is the cell read from excel with tuple x,y,z
        """
        exceldir = self.exceldirOutput
        # https://pypi.python.org/pypi/XlsxWriter
        workbook = xlsxwriter.Workbook(os.path.join(exceldir,'ChevasStoreOutput.xlsx'))
        worksheet = workbook.add_worksheet('General')
        # set column width
        worksheet.set_column('A:A', 35)
        worksheet.set_column('B:B', 30)
        # add a bold format to highlight cells
        bold = workbook.add_format({'bold': True})
        # write title and general tab
        worksheet.write('A1', 'Output ChEVAS dynamic CT post-op', bold)
        analysisID = '%s_%s_%s' % (self.ptcode, self.ctcode, self.cropname)
        worksheet.write('A2', 'Filename:', bold)
        worksheet.write('B2', analysisID)
        worksheet.write('A3', 'Date and Time:', bold)
        date_time = datetime.now() #strftime("%d-%m-%Y %H:%M")
        date_format_str = 'dd-mm-yyyy hh:mm'
        date_format = workbook.add_format({'num_format': date_format_str,
                                    'align': 'left'})
        worksheet.write_datetime('B3', date_time, date_format)
        # write 'storeOutput'
        storeOutput = self.storeOutput 
        for out in storeOutput: # each analysis that was appended to storeOutput
            worksheet = workbook.add_worksheet(out['Name'])
            worksheet.set_column('A:A', 66)
            worksheet.set_column('B:B', 25)
            worksheet.write('A1', 'Name:', bold)
            worksheet.write('B1', out['Name'], bold)
            if out['Type'] == 'motion_centerlines_segments':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Motion of proximal segment of nodes on 1 centerline',bold)
                
                worksheet.write('A3', 'Length of centerline segment (#nodes)',bold)
                worksheet.write('B3', out['lengthSegment'])
                
                worksheet.write('A4', 'Mean segment amplitude (mean_std_min_max)',bold)
                worksheet.write_row('B4', out['mean_segment_amplitude_mean_std_min_max']) # tuple 4 el
                
                worksheet.write('A5', 'Segment mean postions at each phase in cardiac cycle (x,y,z per phase)',bold)
                worksheet.write_row('B5', [str(tuple(x)) for x in out['meanSegmentPosCycle']] ) # nphases x 3
                
                worksheet.write('A6', 'Positions of points in segment at mid cardiac cycle (x,y,z per point) [avgreg]',bold)
                worksheet.write_row('B6', [str(tuple(x)) for x in out['ppSegment']] ) # npoints x 3
                
                worksheet.write('A7', 'Deforms of points in segment at each phase in cardiac cycle [rows=points; columns=phases',bold)
                row = 6 # 6 = row 7 in excel
                for pDeforms in out['ppDeformsSegment']: # npoints x phases x 3
                    worksheet.write_row(row, 1, [str(tuple(x)) for x in pDeforms] )
                    row += 1
            
            elif out['Type'] == 'centerlines_prox_distance_change':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Distance change between the two most proximal points of two centerlines',bold)
                
                worksheet.write('A3', 'Position [avgreg] and deforms of node 1 (x,y,z)',bold)
                worksheet.write('B3', str(tuple(out['Node1'][0])) )
                worksheet.write_row('C3', [str(tuple(x)) for x in out['Node1'][1]] ) # nphases x 3
                
                worksheet.write('A4', 'Position [avgreg] and deforms of node 2 (x,y,z)',bold)
                worksheet.write('B4', str(tuple(out['Node2'][0])) )
                worksheet.write_row('C4', [str(tuple(x)) for x in out['Node2'][1]] ) # nphases x 3
                
                worksheet.write('A5', 'Distance between points at mid cardiac cycle [avgreg]',bold)
                worksheet.write('B5', out['distMidCycle'])
                
                worksheet.write('A6', 'Maximum distance change between points during cardiac cycle',bold)
                worksheet.write('B6', out['distances_diffMax'][0])
                worksheet.write_row('C6', out['distances_diffMax'][1]) # phase min dist, phase max dist
                
                worksheet.write('A7', 'Minimum and maximum distance between points during cardiac cycle',bold)
                worksheet.write_row('B7', out['distances_minmax'])
                
                worksheet.write('A8', 'Mean and std of distances between points during cardiac cycle',bold)
                worksheet.write_row('B8', out['distances_meanstd'])
                
                worksheet.write('A9', 'Distance between points at each phase in cardiac cycle',bold)
                worksheet.write_row('B9', list(out['distances_phases']) )
                
            elif out['Type'] == 'centerline_angle_change':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Angle change of chimney centerline',bold)
                
                worksheet.write('A3', 'Position [avgreg] and deforms of node at proximal end (x,y,z)',bold)
                worksheet.write('B3', str(tuple(out['Node1'][0])) )
                worksheet.write_row('C3', [str(tuple(x)) for x in out['Node1'][1]] ) # nphases x 3
                
                worksheet.write('A4', 'Position [avgreg] and deforms of midnode with max angulation (x,y,z)',bold)
                worksheet.write('B4', str(tuple(out['Node2'][0])) )
                worksheet.write_row('C4', [str(tuple(x)) for x in out['Node2'][1]] ) # nphases x 3
                
                worksheet.write('A5', 'Position [avgreg] and deforms of node at distal end (x,y,z)',bold)
                worksheet.write('B5', str(tuple(out['Node3'][0])) )
                worksheet.write_row('C5', [str(tuple(x)) for x in out['Node3'][1]] ) # nphases x 3
                
                worksheet.write('A6', 'Angle between points at mid cardiac cycle [avgreg]',bold)
                worksheet.write('B6', out['angleMidCycle'])
                
                worksheet.write('A7', 'Maximum angle change between points during cardiac cycle',bold)
                worksheet.write('B7', out['point_angle_diff_max'][0])
                worksheet.write_row('C7', out['point_angle_diff_max'][1]) # phase min angle, phase max angle
                
                worksheet.write('A8', 'Minimum and maximum angle between points during cardiac cycle',bold)
                worksheet.write_row('B8', out['angles_minmax'])
                
                worksheet.write('A9', 'Mean and std of angles between points during cardiac cycle',bold)
                worksheet.write_row('B9', out['angles_meanstd'])
                
                worksheet.write('A10', 'Angle between points at each phase in cardiac cycle',bold)
                worksheet.write_row('B10', list(out['angles_phases']) )
            
            elif out['Type'] == 'centerline_tortuosity_change':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Tortuosity change of chimney centerline',bold)
                
                worksheet.write('A3', 'Tortuosity at mid cardiac cycle [avgreg]',bold)
                worksheet.write('B3', out['tortMidCycle'])
                
                worksheet.write('A4', 'Maximum tortuosity change during cardiac cycle',bold)
                worksheet.write('B4', out['tort_diff_max'][0])
                worksheet.write_row('C4', out['tort_diff_max'][1]) # phase min , phase max
                
                worksheet.write('A5', 'Minimum and maximum tortuosity during cardiac cycle',bold)
                worksheet.write_row('B5', out['tort_minmax'])
                
                worksheet.write('A6', 'Mean and std of tortuosities during cardiac cycle',bold)
                worksheet.write_row('B6', out['tort_meanstd'])
                
                worksheet.write('A7', 'Tortuosity at each phase in cardiac cycle',bold)
                worksheet.write_row('B7', list(out['tort_phases']) )

        
        workbook.close()




def plot_points(pp, mc='g', ms='o', mw=8, alpha=0.5, ls='', ax=None, **kwargs):
    """ Plot a point or set of points in current axis and restore current view 
    alpha 0.9 = solid; 0.1 transparant
    """
    if ax is None:
        ax = vv.gca()
    # check if pp is 1 point and not a PointSet
    if not isinstance(pp, PointSet):
        pp = np.asarray(pp)
        if pp.ndim == 1:
            p = PointSet(3)
            p.append(pp)
            pp = p
    # get view and plot
    view = ax.GetView()
    point = vv.plot(PointSet(pp), mc=mc, ms=ms, mw=mw, ls=ls, alpha=alpha, axes=ax, **kwargs)
    ax.SetView(view)
    
    return point

def point_to_point_distance_change(point1,point1Deforms,point2, point2Deforms):
    """ Calculate distance change between 2 points during cardiac cycle
    """
    n1Indices = point1 + point1Deforms
    n2Indices = point2 + point2Deforms
    # define vector between nodes
    v = n1Indices - n2Indices
    distances = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1) # for nphases
    # get min and max distance
    point_to_pointMax = distances.max()
    point_to_pointMin = distances.min()
    diff_max = point_to_pointMax - point_to_pointMin
    # add phase in cardiac cycle where min and max where found (5th = 50%)
    phasesminmax = [(list(distances).index(point_to_pointMin) )*10, 
                    (list(distances).index(point_to_pointMax) )*10]
    diff_max = [diff_max, phasesminmax ]
    
    meanDistancesCycle = [np.mean(distances), np.std(distances)] 
    minmaxDistancesCycle = [np.min(distances), np.max(distances)]
    
    output = {
    'distances_meanstd': meanDistancesCycle,
    'distances_minmax': minmaxDistancesCycle,
    'distances_phases': distances,
    'distances_diffMax': diff_max,
    'Node1': [point1, point1Deforms], 
    'Node2': [point2, point2Deforms] } 
    
    return output


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
    angles = np.array(angles) # 10 angles; for each phase
    
    # get all angle differences of all phases
    # todo: could be simplified by just getting diff of min and max of angles 
    pos_combinations = list(itertools.combinations(range(len(v1)),2))
    angle_diff = []
    for i in pos_combinations:
        angle_diff.append(abs(angles[i[0]] - angles[i[1]]))
    angle_diff = np.array(angle_diff) # angle differences between all phase combinations
    
    # get max angle differences
    point_angle_diff_max = angle_diff.max()
    point_angle_diff_max = [point_angle_diff_max, [x*10 for x in
    (pos_combinations[list(angle_diff).index(point_angle_diff_max)])]] # angle diff, [phase1, phase2]
    
    # todo: min diff redundant info?
    # get min angle differences
    point_angle_diff_min = angle_diff.min()
    point_angle_diff_min = [point_angle_diff_min, [x*10 for x in 
    (pos_combinations[list(angle_diff).index(point_angle_diff_min)])]]
    
    print('Angle phases= {}'.format(angles))
    
    meanAnglesCycle = [np.mean(angles), np.std(angles)] 
    minmaxAnglesCycle = [np.min(angles), np.max(angles)]
    
    output = {
    'point_angle_diff_min':point_angle_diff_min,
    'point_angle_diff_max': point_angle_diff_max, 
    'angles_phases': angles, 'angles_meanstd': meanAnglesCycle, 
    'angles_minmax': minmaxAnglesCycle,
    'Node1': [point1, point1Deforms], 'Node2': [point2, point2Deforms], 
    'Node3': [point3, point1Deforms]}
    
    return output


def calculate_tortuosity(ppCll):
    """ Ratio centerline length to length between its ends
    tortuosity: array
    """
    CllEnds = PointSet(3, dtype='float32')
    CllEnds.append(ppCll[0])
    CllEnds.append(ppCll[-1])
    distCllEnds = CllEnds[0].distance(CllEnds[-1])
    ppCllLength = dist_over_centerline(ppCll, type='euclidian')
    tortuosity = ppCllLength/distCllEnds
    
    return tortuosity
    

def calculate_tortuosity_change(ppCll, ppCllDeforms):
    """ Tortuosity of centerline in different phases of cycle
    """
    # get positions of points in segment during cycle
    posCllCycle = np.zeros_like(ppCllDeforms)
    for i, ppDeform in enumerate(ppCllDeforms): # for each point
        posCllCycle[i] = ppCll[i] + ppDeform
    # calc tortuosity and change
    tortuosityPhases = []
    for phasenr in range(ppCllDeforms.shape[1]):
        ppCll_phase = posCllCycle[:,phasenr,:]
        tortuosityPhases.append(calculate_tortuosity(ppCll_phase))
    meanTortCycle = [np.mean(tortuosityPhases), np.std(tortuosityPhases)] 
    minmaxTortCycle = [np.min(tortuosityPhases), np.max(tortuosityPhases)]
    # get diff between phases
    pos_combinations = list(itertools.combinations(range(len(tortuosityPhases)),2))
    tort_diff = []
    for i in pos_combinations:
        tort_diff.append(abs(tortuosityPhases[i[0]] - tortuosityPhases[i[1]]))
    tort_diff = np.array(tort_diff) # tort differences between all phase combinations
    # get max tort differences
    diff_max = tort_diff.max()
    diff_max = [diff_max, [x*10 for x in
    (pos_combinations[list(tort_diff).index(diff_max)])]] # tort diff, [phase1, phase2]
    
    print('Tortuosity phases= {}'.format(tortuosityPhases))
    
    output = {
    'tort_meanstd': meanTortCycle,
    'tort_minmax': minmaxTortCycle, 
    'tort_phases': tortuosityPhases,
    'tort_diff_max': diff_max  }
    
    return output


def calculate_motion_points(ppCll, ppCllDeforms, lenSegment, dim='xyz'):
    """ Mean motion pattern of segment of points on Cll and mean amplitude
    of motion. Proximal segment is analyzed.
    ppCll is centerline path (points)
    """
    pends = np.array([ppCll[0], ppCll[-1]])  # pp centerline is in order of centerline points
    if pends[0,-1] > pends[1,-1]: # start was distal
        pp = ppCll[-1*lenSegment:] # last points=prox segment
        ppDeforms = ppCllDeforms[-1*lenSegment:]
    else: # first points=prox segment
        pp = ppCll[:lenSegment]
        ppDeforms = ppCllDeforms[:lenSegment]
    # get positions of points in segment during cycle
    posCycleSegment = np.zeros_like(ppDeforms)
    for i, ppDeform in enumerate(ppDeforms): # for each point
        posCycleSegment[i] = pp[i] + ppDeform
    meanPosCycleSegment = np.mean(posCycleSegment, axis=0) # nphases x 3
    motionOut = calculateMeanAmplitude(pp,ppDeforms, dim=dim) # mean, std, min, max
    # store in dict
    output = {'mean_segment_amplitude_mean_std_min_max': motionOut,
                'lengthSegment': lenSegment, 
                'meanSegmentPosCycle': meanPosCycleSegment,
                'ppSegment': pp, # [positions nodes avgreg]
                'ppDeformsSegment': ppDeforms # [deforms nodes avgreg]
                }
    return output


def get_prox_dist_points_cll(ppCll1):
    # Get prox point for ppCll1
    pends1 = np.array([ppCll1[0], ppCll1[-1]])  # pp centerline is in order of centerline points
    pends1 = pends1[pends1[:,-1].argsort() ] # sort with z, ascending
    proxp1 = pends1[0]   # smallest z; origin is prox
    distp1 = pends1[-1]
    
    return proxp1, distp1

