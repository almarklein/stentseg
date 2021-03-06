""" Module to perform automated motion analysis of dynamic centerline models
Author: Maaike A. Koenrades. Created October 2017

"""
import sys, os
from stentseg.utils.centerline import points_from_nodes_in_graph, dist_over_centerline
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume, plot_points
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.utils.utils_graphs_pointsets import get_graph_in_phase
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
    def __init__(self,ptcode,ctcode,basedir,showVol='MIP',clim=(0,2500), color='b', mw=5, **kwargs):
        """
        Init motion analysis on centerlines 
        """
        self.s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelavgreg_deforms_id')
        #load model of vessel centerline, distal of stent, if available
        try:
            self.s_vessel = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelvesselavgreg_deforms_id')
        except FileNotFoundError:
            self.s_vessel = None
        #load vol for visualization
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
        self.a.axis.axisColor = 0,0,0#1,1,1
        self.a.axis.visible = False
        self.a.bgcolor = 1,1,1#0,0,0
        self.a.daspect = 1, 1, -1
        t = show_ctvolume(vol, showVol=showVol, clim=clim, removeStent=False, 
                        climEditor=True, isoTh=225, **kwargs)
        self.label = pick3d(self.a, vol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        for key in self.s:
            if key.startswith('model'):
                self.s[key].Draw(mc=color, mw = mw, lc=color, alpha = 0.5)
        if not self.s_vessel is None:
            for key in self.s_vessel:
                if key.startswith('model'):
                    self.s_vessel[key].Draw(mc=color, mw = mw, lc=color, alpha = 0.5)
        vv.title('Model for ChEvas %s  -  %s' % (ptcode[7:], 'follow-up')) # ctcode not correct
        
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [self.a], axishandling=False) )
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
    
    def motion_centerlines_segments(self, lenSegment=10, type='stents'):
        """ given a centerline, compute motion of points in centerline segment
        dim: amplitude of motion in x,y,z, or xyz 
        type: stents or vessels
        self has ssdf with dynamic model of cll and ppCenterline
        stores mean displacement pattern of segment points and amplitude mean std min max 
        """
        if type == 'stents':
            s = self.s # ssdf with centerline pointsets pp identified
        elif type == 'vessels':
            s = self.s_vessel
        for key in s:
            if key.startswith('ppCenterline'): # each branch or Nel or vessel
                ppCll = s[key]
                name_output = 'Motion_'+key[12:]+'prox'
                model = s['model'+key[12:]] # skip 'ppCenterline' in key
                assert model.number_of_edges() == 1 # a centerline is one edge
                edge = model.edges()[0]
                ppCllDeforms = model.edge[edge[0]][edge[1]]['pathdeforms']
                ppCllDeforms = np.asarray(ppCllDeforms) # npoints x nphases x 3
                #ppCll == model.edge[edge[0]][edge[1]]['path']
                output = calculate_motion_points(key, ppCll, ppCllDeforms, lenSegment, part='prox')
                output['Type'] = 'motion_centerlines_segments'
                # Store output with name
                output['Name'] = name_output
                self.storeOutput.append(output)
                pp = output['ppSegment'] # [positions nodes avgreg]
                # visualize segment analyzed in avgreg
                a1 = self.a
                point = plot_points(pp, mc='y', ax=a1)
                self.points_plotted.append(point)
                
                if type == 'stents':
                    # now obtain motion for distal segment
                    name_output = 'Motion_'+key[12:]+'dist'
                    output = calculate_motion_points(key, ppCll, ppCllDeforms, lenSegment, part='dist')
                    output['Type'] = 'motion_centerlines_segments'
                    # Store output with name
                    output['Name'] = name_output
                    self.storeOutput.append(output)
                    pp = output['ppSegment'] # [positions nodes avgreg]
                    # visualize segment analyzed in avgreg
                    if not key.startswith('ppCenterlineNel'): # do not visualize distal nellix
                        a1 = self.a
                        point = plot_points(pp, mc='r', ax=a1)
                        self.points_plotted.append(point)
                    
    
    def distance_change_nelnel_nelCh(self):
        """ Distances cardiac cycle between the proximal points for:
        Nellix to Nellix stent (NelL to NelR)
        Chimney left (LRA) to closest Nellix
        Chimney right (RRA) to closest Nellix
        Stores output in centerlines_prox_distance_change()
        """
        s = self.s # ssdf with centerline pointsets pp identified
        ppNelR = s['ppCenterlineNelR']
        ppNelL = s['ppCenterlineNelL']
        for key in s:
            if key.startswith('ppCenterlineRRA'):
                ppRRA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppRRA, ppNelR, ppNelL)
                self.centerlines_prox_distance_change(key1, key2, ppRRA, ppNel, 'Dist_{}_{}'.format(key1,key2),color='y')
            elif key.startswith('ppCenterlineLRA'):
                ppLRA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppLRA, ppNelR, ppNelL)
                self.centerlines_prox_distance_change(key1, key2, ppLRA, ppNel, 'Dist_{}_{}'.format(key1,key2),color='c')
            elif key.startswith('ppCenterlineSMA'):
                ppSMA = s[key]
                key1 = key[12:]
                ppNel, key2 = self.get_nellix_closest_to_chimney(ppSMA, ppNelR, ppNelL)
                self.centerlines_prox_distance_change(key1, key2, ppSMA, ppNel, 'Dist_{}_{}'.format(key1,key2),
                                                      mw=17,color='m',marker='v',alpha=0.7)
            elif key.startswith('ppCenterlineNelL'):
                # ppNelL = s[key]
                key1 = key[12:]
                # ppNelR = s['ppCenterlineNelR']
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
        
    
    def chimneys_angle_change(self, armlength=10):
        """Calculate angle change for each chimney stent:
        - of chimney itself
        - between chimney and nellix proximal segments
        """
        s = self.s # ssdf with centerline pointsets pp identified
        for key in s:
            if key.startswith('ppCenterlineRRA'):
                ppRRA = s[key]
                key1 = key[12:]
                self.chimney_angle_change(key1, ppRRA, 'Ang_RRA', armlength=armlength)
                self.centerline_tortuosity_change(key1, ppRRA, 'Tort_RRA')
                self.chimney_nel_angle_change(key1, ppRRA, 'Ang_RRA_Nel', armlength=armlength)
            if key.startswith('ppCenterlineLRA'):
                ppLRA = s[key]
                key1 = key[12:]
                self.chimney_angle_change(key1, ppLRA, 'Ang_LRA', armlength=armlength)
                self.centerline_tortuosity_change(key1, ppLRA, 'Tort_LRA')
                self.chimney_nel_angle_change(key1, ppLRA, 'Ang_LRA_Nel', armlength=armlength)
            if key.startswith('ppCenterlineSMA'):
                ppSMA = s[key]
                key1 = key[12:]
                self.chimney_angle_change(key1, ppSMA, 'Ang_SMA', armlength=armlength)
                self.centerline_tortuosity_change(key1, ppSMA, 'Tort_SMA')
                self.chimney_nel_angle_change(key1, ppSMA, 'Ang_SMA_Nel', armlength=armlength)
    
    def chimneys_vessel_angle_change(self, armlength=10):
        """ Calculate angle at stent-vessel transition, between vector dist 
        stent and prox end vessel centerline
        """ 
        s_vessel = self.s_vessel # ssdf with centerline pointsets pp identified
        
        for key in s_vessel:
            if key.startswith('ppCenterline'):
                ppVessel = s_vessel[key]
                key1 = key[12:] # e.g. vLRA
                name_output = 'Ang_'+key1[1:]+'_Vessel' # Ang_LRA_Vessel
                self.chimney_vessel_angle_change(key1, ppVessel, name_output, armlength=10)
        
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
        output['tortMidCycle'] = calculate_tortuosity(ppCh)[0] # in avgreg
        
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'centerline_tortuosity_change'
        self.storeOutput.append(output)
    
    def chimney_vessel_angle_change(self, key, ppVessel, name_output, armlength=10):
        """ Calculate angle between dist segment chimney and vessel after stent
        at each phase in the cardiac cycle
        """ 
        # Get vessel model
        model = self.s_vessel['model'+key] # skip 'ppCenterline' in key
        # get number of phases during cardiac cycle
        number_of_phases = len(model.node[tuple(ppVessel[0])]['deforms'])
        assert model.number_of_edges() == 1 # a centerline is one edge
        
        # Get chimney model
        key2 = key[1:] # from vLRA to LRA
        modelChim = self.s['model'+key2]
        ppCh = self.s['ppCenterline'+key2]
        
        # Calculate for each phase angle between dist chimney and prox vessel
        anglesCycle = []
        for phasenr in range(number_of_phases):
            # === For chimney vector ===
            model_phase_Ch = get_graph_in_phase(modelChim, phasenr)
            edge = model_phase_Ch.edges()[0]
            ppCh_phase = model_phase_Ch.edge[edge[0]][edge[1]]['path']
            # Get prox segment vectors
            proxendcll, distendcll = get_prox_dist_points_cll(ppCh_phase)
            
            #get point on cll at armlength from distal end
            point2, vectorCh = get_point_on_cll_at_armslength(ppCh_phase, 
                    distendcll, armlength=armlength, type='proximal')
            
            # === For VESSEL vector ===
            # get vector prox segment vessel
            model_phase_Vessel = get_graph_in_phase(model, phasenr)
            edge = model_phase_Vessel.edges()[0]
            ppNel_phase = model_phase_Vessel.edge[edge[0]][edge[1]]['path']
            # Get prox segment vectors
            proxendcll, distendcll = get_prox_dist_points_cll(ppNel_phase, key=key)
            #get point on cll at armlength from prox end
            point2_nel, vectorNel = get_point_on_cll_at_armslength(ppNel_phase, 
                    proxendcll, armlength=armlength, type='distal', key=key)
            
            # Obtain angle between vectors
            phi = abs(PointSet(vectorCh).angle(PointSet(vectorNel)))
            angle = phi*180.0/np.pi # direction vector in degrees
            anglesCycle.append(angle) 
        
        # For visualization get vectors and angle at mid cycle
        # == vessel ==
        proxendcll, distendcll = get_prox_dist_points_cll(ppVessel, key=key)
        point, vector = get_point_on_cll_at_armslength(ppVessel, proxendcll, 
                armlength=armlength, type='distal', key=key)
        # == chimney ==
        proxendcllNel, distendcllNel = get_prox_dist_points_cll(ppCh)
        pointNel, vectorNel = get_point_on_cll_at_armslength(ppCh, distendcllNel, 
                armlength=armlength, type='proximal')
        
        phi = abs(PointSet(vector).angle(PointSet(vectorNel)))
        angle_avgreg = phi*180.0/np.pi # direction vector in degrees
        
        # visualize vectors chimney and nellix
        a1 = self.a
        mw =15
        if True: # False=do not show arm points
            color = 'y'
            vectorpoints = np.asarray([proxendcll, point])
            plotted_points = plot_points(vectorpoints,mc=color,mw=mw,ls='-',lw=12,lc=color,alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_points)
            vectorpointsNel = np.asarray([distendcllNel, pointNel])
            plotted_pointsNel = plot_points(vectorpointsNel,mc=color,mw=mw,ls='-',lw=12,lc=color,alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_pointsNel)
        
        # ============
        # store output
        print('Angles during cardiac cycle= {}'.format(anglesCycle))
        print('')
        meanAnglesCycle = [np.mean(anglesCycle), np.std(anglesCycle)] 
        minmaxAnglesCycle = [np.min(anglesCycle), np.max(anglesCycle)]
        angleChange = max(anglesCycle) - min(anglesCycle)
        
        output = {
        'vectorAngle_diff_max': angleChange, 
        'vectorAngles_phases': anglesCycle, 
        'vectorAngles_meanstd': meanAnglesCycle, 
        'vectorAngles_minmax': minmaxAnglesCycle, # where min is sharpest angle
        'vectorAngleMidCycle': angle_avgreg,
        'vectorArmlength': armlength
        }
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'chimney_nel_angle_change' # vessel-Nel but same format output dict
        
        self.vectorAngleChange_output = output
        self.storeOutput.append(output)
        
    def chimney_nel_angle_change(self, key, ppCh, name_output, armlength=10):
        """ Calculate angle between prox segment of chimney and nellix (vectors) at
        each phase in cardiac cycle
        default armlength 10 point from top of cll
        """
        # Get chimney model
        model = self.s['model'+key] # skip 'ppCenterline' in key
        # get number of phases during cardiac cycle
        number_of_phases = len(model.node[tuple(ppCh[0])]['deforms'])
        assert model.number_of_edges() == 1 # a centerline is one edge
        
        # determine which nellix stent was closest to chimney
        ppNelR = self.s['ppCenterlineNelR']
        ppNelL = self.s['ppCenterlineNelL']
        ppNel, key2 = self.get_nellix_closest_to_chimney(ppCh, ppNelR, ppNelL)
        
        # if 'LRA' in key: # for visualization of both Nellix stents case 02 for figure methods
        #     ppNel, key2 = ppNelL, 'NelL'
        
        # Get nellix model
        modelNel = self.s['model'+key2]
        
        # Calculate for each phase angle between prox segment of chimney and nellix
        anglesCycle = []
        for phasenr in range(number_of_phases):
            # === For chimney vector ===
            model_phase_Ch = get_graph_in_phase(model, phasenr)
            edge = model_phase_Ch.edges()[0]
            ppCh_phase = model_phase_Ch.edge[edge[0]][edge[1]]['path']
            # Get prox segment vectors
            proxendcll, distendcll = get_prox_dist_points_cll(ppCh_phase)
            
            #get point on cll at armlength from prox end
            point2, vectorCh = get_point_on_cll_at_armslength(ppCh_phase, proxendcll, armlength=armlength, type='distal')
            
            # === For Nellix vector ===
            # get vector prox segment nellix
            model_phase_Nel = get_graph_in_phase(modelNel, phasenr)
            edge = model_phase_Nel.edges()[0]
            ppNel_phase = model_phase_Nel.edge[edge[0]][edge[1]]['path']
            # Get prox segment vectors
            proxendcll, distendcll = get_prox_dist_points_cll(ppNel_phase)
            #get point on cll at armlength from prox end
            point2_nel, vectorNel = get_point_on_cll_at_armslength(ppNel_phase, proxendcll, armlength=armlength, type='distal')
            
            # Obtain angle between vectors
            phi = abs(PointSet(vectorCh).angle(PointSet(vectorNel)))
            angle = phi*180.0/np.pi # direction vector in degrees
            anglesCycle.append(angle) 
        
        # For visualization get vectors and angle at mid cycle
        # == chimney ==
        proxendcll, distendcll = get_prox_dist_points_cll(ppCh)
        point, vector = get_point_on_cll_at_armslength(ppCh, proxendcll, armlength=armlength, type='distal')
        # == nellix ==
        proxendcllNel, distendcllNel = get_prox_dist_points_cll(ppNel)
        pointNel, vectorNel = get_point_on_cll_at_armslength(ppNel, proxendcllNel, armlength=armlength, type='distal')
        
        phi = abs(PointSet(vector).angle(PointSet(vectorNel)))
        angle_avgreg = phi*180.0/np.pi # direction vector in degrees
        
        # visualize vectors chimney and nellix
        a1 = self.a
        mw =15
        if True: # False=do not show arm points
            color = 'y'
            vectorpoints = np.asarray([proxendcll, point])
            plotted_points = plot_points(vectorpoints,mc=color,mw=mw,ls='-',lw=12,lc=color,alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_points)
            vectorpointsNel = np.asarray([proxendcllNel, pointNel])
            plotted_pointsNel = plot_points(vectorpointsNel,mc=color,mw=mw,ls='-',lw=12,lc=color,alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_pointsNel)
        
        # ============
        # store output
        print('Angles during cardiac cycle= {}'.format(anglesCycle))
        print('')
        meanAnglesCycle = [np.mean(anglesCycle), np.std(anglesCycle)] 
        minmaxAnglesCycle = [np.min(anglesCycle), np.max(anglesCycle)]
        angleChange = max(anglesCycle) - min(anglesCycle)
        
        output = {
        'vectorAngle_diff_max': angleChange, 
        'vectorAngles_phases': anglesCycle, 
        'vectorAngles_meanstd': meanAnglesCycle, 
        'vectorAngles_minmax': minmaxAnglesCycle, # where min is sharpest angle
        'vectorAngleMidCycle': angle_avgreg,
        'vectorArmlength': armlength
        }
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'chimney_nel_angle_change'
        
        self.vectorAngleChange_output = output
        self.storeOutput.append(output)
    
    
    def chimney_angle_change(self, key, ppCh, name_output, armlength=10):
        """ Calculate the chimney angle change during the cycle
        -> for each point on the given centerline to obtain the values for the point with the greatest change
        -> and peak angles during the phases, anywhere on the stent  
        --> or between prox and dist segments?
        ppCh is a PointSet of the CLL, which is in correct order 
        """
        
        # get number of phases during cardiac cycle
        model = self.s['model'+key] # skip 'ppCenterline' in key
        number_of_phases = len(model.node[tuple(ppCh[0])]['deforms'])
        assert model.number_of_edges() == 1 # a centerline is one edge
        
        # determine arms length by check length of chimney
        ppChLength = dist_over_centerline(ppCh, type='euclidian')
        if not ppChLength > 2*armlength:
            armlength = np.floor(ppChLength/2)-1 # Angle from ~mid of the stent or fix at 5 mm?
        
        # ===================
        # Calculate for each cll point the angles during phases; obtain point with max change
        
        # for each centerline point...  
        angleChange = 0
        
        for i, n in enumerate(ppCh):
            anglesCycle_i = []
            anglesCycleNodes_i = []
            # ...calculate angle for each phase
            for phasenr in range(number_of_phases):
                model_phase_Ch = get_graph_in_phase(model, phasenr)
                edge = model_phase_Ch.edges()[0]
                ppCh_phase = model_phase_Ch.edge[edge[0]][edge[1]]['path']
                n2 = ppCh_phase[i]
                n1, n3, anglenew = get_angle_at_fixed_arms(ppCh_phase, n2, armlength=armlength)
                if anglenew is None:
                    break # next point; cll point not far enough from cll ends to calculate angle
                # collect angles during the phases
                anglesCycle_i.append(anglenew)
                anglesCycleNodes_i.append([i, n, n2, n1, n3]) # index,n,n2  ,n1,n3
            if anglenew is None:
                continue # next point; cll point not far enough from cll ends to calculate angle
            
            # get angle change
            angleChangeNew = max(anglesCycle_i) - min(anglesCycle_i)
            if angleChangeNew > angleChange:
                angleChange = angleChangeNew
                # store angles and nodes for so far max angle change
                anglesCycle = anglesCycle_i
                anglesCycleNodes = anglesCycleNodes_i
        
        # visualize angle arms avgreg=mid cycle
        n = anglesCycleNodes[0][1]
        n1, n3, anglenew = get_angle_at_fixed_arms(ppCh, n, armlength=armlength)
        armpoints = np.asarray([n1, n, n3])
        a1 = self.a
        mw =15
        if False: # False=do not show arm points
            plotted_points = plot_points(armpoints,mc='r',mw=mw,ls='-',lw=12,lc='r',alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_points)
        
        # get location of point with max angle change
        proxendcll, distendcll = get_prox_dist_points_cll(ppCh)
        location_of_midpoint_n = dist_over_centerline(ppCh,cl_point1=proxendcll,
                                 cl_point2=n,type='euclidian') # distance from proximal end centerline
        total_length_cll = dist_over_centerline(ppCh, type='euclidian')
        
        # store output
        print('Angle phases of point with max angle change= {}'.format(anglesCycle))
        print('')
        meanAnglesCycle = [np.mean(anglesCycle), np.std(anglesCycle)] 
        minmaxAnglesCycle = [np.min(anglesCycle), np.max(anglesCycle)]
        
        output = {
        'point_angle_diff_max': angleChange, 
        'angles_phases': anglesCycle, 
        'angles_meanstd': meanAnglesCycle, 
        'angles_minmax': minmaxAnglesCycle, # where min is sharpest angle
        'anglenodes_positions_cycle': anglesCycleNodes, #list 10xlist index,n,n2  ,n1,n3 for arms
        'location_point_max_angle_change': location_of_midpoint_n, # dist from proximal end
        'total_length_chimney_midCycle': total_length_cll,
        'angleMidCycle': anglenew
        }
        
        
        # ============
        # Calculate for each phase peak angle (peak may be at different locations on centerline)
        peakAngle_phases = []
        peakAngle_phases_location = []
        for phasenr in range(number_of_phases):
            model_phase_Ch = get_graph_in_phase(model, phasenr)
            edge = model_phase_Ch.edges()[0]
            ppCh_phase = model_phase_Ch.edge[edge[0]][edge[1]]['path']
            cllAngles = []
            cllAngles_n = []
            cllAngles_i = []
            for i, n in enumerate(ppCh_phase):
                n1, n3, anglenew = get_angle_at_fixed_arms(ppCh_phase, n, armlength=armlength)
                if anglenew is None:
                    continue # next point; cll point not far enough from cll ends to calculate angle
                cllAngles.append(anglenew) # for each point on cll an angle
                cllAngles_n.append(n)
                cllAngles_i.append(i)
            # get greatest of cll angles but 180 degrees is straigth so min
            peakAngle = min(cllAngles)
            peakAngle_index = cllAngles.index(peakAngle) # index in list of obtained angles
            proxendcll, distendcll = get_prox_dist_points_cll(ppCh_phase)
            n = cllAngles_n[peakAngle_index]
            peakAngle_index_cll = cllAngles_i[peakAngle_index]
            peakAngle_location = dist_over_centerline(ppCh_phase,cl_point1=proxendcll,
                                 cl_point2=n,type='euclidian') # distance from proximal end centerline
            peakAngle_phases.append(peakAngle)
            peakAngle_phases_location.append([peakAngle_index_cll, peakAngle_location])
        
        # get peakAngle at mid cycle
        cllAngles = []
        cllAngles_n = []
        cllAngles_i = []
        cllAngles_n1 = []
        cllAngles_n3 = []
        for i, n in enumerate(ppCh):
            n1, n3, anglenew = get_angle_at_fixed_arms(ppCh, n, armlength=armlength)
            if anglenew is None:
                continue # next point; cll point not far enough from cll ends to calculate angle
            cllAngles.append(anglenew) # for each point on cll an angle
            cllAngles_n.append(n)
            cllAngles_i.append(i)
            cllAngles_n1.append(n1)
            cllAngles_n3.append(n3)
        # get greatest of cll angles but 180 degrees is straigth so min
        peakAngle_midcycle = min(cllAngles)
        peakAngle_index = cllAngles.index(peakAngle_midcycle) # index in list of obtained angles
        proxendcll, distendcll = get_prox_dist_points_cll(ppCh)
        n = cllAngles_n[peakAngle_index]
        peakAngle_index_cll_midcycle = cllAngles_i[peakAngle_index]
        peakAngle_location_midcycle = dist_over_centerline(ppCh,cl_point1=proxendcll,
                                cl_point2=n,type='euclidian') # distance from proximal end centerline
        peakAngle_midcycle_location = [peakAngle_index_cll_midcycle, peakAngle_location_midcycle]
        # visualize
        if True: # False=do not show arm points
            n1 = cllAngles_n1[peakAngle_index]
            n3 = cllAngles_n3[peakAngle_index]
            armpoints = np.asarray([n1, n, n3])
            plotted_points = plot_points(armpoints,mc='r',mw=mw,ls='-',lw=12,lc='r',alpha=0.7,ax=a1)
            self.points_plotted.append(plotted_points)
        
        # stats phases
        peakAngle_phase_max = max(peakAngle_phases)
        peakAngle_phase_min = min(peakAngle_phases)
        peakAngle_phase_max_phase = peakAngle_phases.index(peakAngle_phase_max) # which phase was max?
        peakAngle_phase_min_phase = peakAngle_phases.index(peakAngle_phase_min) # which phase was min?
        peakAngle_phases_diff = peakAngle_phase_max - peakAngle_phase_min
        peakAngle_phases_meanstd = [np.mean(peakAngle_phases), np.std(peakAngle_phases)]
        
        print('PeakAngle during cycle= {}'.format(peakAngle_phases))
        print('')
        
        #add to output dict
        output['peakAngle_per_phase'] = peakAngle_phases
        output['peakAngle_min_and_max'] = [peakAngle_phase_min, peakAngle_phase_max] # where min is greatest angle
        output['peakAngle_phases_diff'] = peakAngle_phases_diff
        output['peakAngle_phases_meanstd'] = peakAngle_phases_meanstd
        output['peakAngle_phases_location'] = peakAngle_phases_location # for each phase location of peakAngle
        
        output['peakAngle_midcycle'] = peakAngle_midcycle
        output['peakAngle_midcycle_location'] = peakAngle_midcycle_location
        
        # ============
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'chimney_angle_change'
        
        self.angleChange_output = output
        self.storeOutput.append(output)
        
        
        # # Calculate greatest angle along cll with fixed arms
        # angle = 180 # straigth
        # for i, n2 in enumerate(ppCh):
        #     n1, n3, anglenew = get_angle_at_fixed_arms(ppCh, n2, armlength=15)
        #     if anglenew is None:
        #         continue
        #     print('Angle for points along cll= {}'.format(anglenew) )
        #     if anglenew < angle:
        #         midnode = [copy.copy(n2), copy.copy(i)]
        #         point1 = copy.copy(n1)
        #         point3 = copy.copy(n3)
        #     angle = min(angle, anglenew)
        # 
        # # show points at which angle was calculated
        # a1 = self.a
        # mw =18
        # if False: # False=do not show arm points
        #     plot_point1 = plot_points(point1, mc='b', mw=mw, ax=a1)
        #     self.points_plotted.append(plot_point1)
        #     plot_point3 = plot_points(point3, mc='g', mw=mw, ax=a1)
        #     self.points_plotted.append(plot_point3)
        #     # self.createNodePoint(tuple(point1), color='b')
        #     # self.createNodePoint(tuple(point3))
        #     
        # # show midnode
        # print('Detected midnode: location {} and angle {}'.format(midnode, angle))
        # n2 = midnode[0]
        # plot_point2 = plot_points(n2, mc='m', mw=mw, ax=a1)
        # self.points_plotted.append(plot_point2)
        # # self.createNodePoint(tuple(n2), color='m')
        # 
        # # Calc angle change cycle
        # # use the endpoints: get prox and dist point for ppCh
        # point1, point3 = get_prox_dist_points_cll(ppCh)
        # # get deforms of nodes during cardiac cycle
        # model = self.s['model'+key] # skip 'ppCenterline' in key
        # n1Deforms = model.node[tuple(point1)]['deforms']
        # n2Deforms = model.node[tuple(n2)]['deforms']
        # n3Deforms = model.node[tuple(point3)]['deforms']
        # # get angulation
        # output = line_line_angulation(point1, 
        #                     n1Deforms, n2, n2Deforms, point3, n3Deforms)
        # 
        # # add chimney angle mid cardiac cycle
        # output['angleMidCycle'] = output['angles_meanstd'][0]
        # 
        # # Store output with name
        # output['Name'] = name_output
        # output['Type'] = 'chimney_angle_change'
        # self.storeOutput.append(output)


    def storeOutputToExcel(self):
        """Create file and add a worksheet or overwrite existing
        Output of x,y,z positions in one cell can be handled in python with
        np.asrray(a), where a is the cell read from excel with tuple x,y,z
        """
        exceldir = self.exceldirOutput
        # https://pypi.python.org/pypi/XlsxWriter
        workbook = xlsxwriter.Workbook(os.path.join(exceldir,'ChevasStoreOutput{}.xlsx'.format(self.ptcode[7:])))
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
            worksheet.set_column('A:A', 82)
            worksheet.set_column('B:C', 24)
            worksheet.write('A1', 'Name:', bold)
            worksheet.write('B1', out['Name'], bold)
            if out['Type'] == 'motion_centerlines_segments':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Motion of segment of nodes on 1 centerline',bold)
                
                worksheet.write('A3', 'Length of centerline segment (#nodes)',bold)
                worksheet.write('B3', out['lengthSegment'])
                
                worksheet.write('A4', 'XYZ: segment mean displacement amplitude (vector magnitude) (mean_std_min_max)',bold)
                worksheet.write_row('B4', out['mean_segment_amplitudexyz_mean_std_min_max']) # tuple 4 el
                
                worksheet.write('A5', 'X: segment mean displacement amplitude (vector magnitude) (mean_std_min_max)',bold)
                worksheet.write_row('B5', out['mean_segment_amplitudex_mean_std_min_max']) # tuple 4 el
                
                worksheet.write('A6', 'Y: segment mean displacement amplitude (vector magnitude) (mean_std_min_max)',bold)
                worksheet.write_row('B6', out['mean_segment_amplitudey_mean_std_min_max']) # tuple 4 el
                
                worksheet.write('A7', 'Z: segment mean displacement amplitude (vector magnitude) (mean_std_min_max)',bold)
                worksheet.write_row('B7', out['mean_segment_amplitudez_mean_std_min_max']) # tuple 4 el
                
                worksheet.write('A8', 'Segment mean position (CoM) at each phase in cardiac cycle (x,y,z per phase)',bold)
                worksheet.write_row('B8', [str(tuple(x)) for x in out['meanSegmentPosCycle']] ) # nphases x 3
                
                worksheet.write('A9', 'Segment mean position (CoM) at mid cardiac cycle (x,y,z) [avgreg]',bold)
                worksheet.write('B9', str(tuple(out['meanPosAvgSegment']))  ) # 1 x 3
                
                worksheet.write('A11', 'Positions of points in segment at mid cardiac cycle (x,y,z per point) [avgreg]',bold)
                worksheet.write_row('B11', [str(tuple(x)) for x in out['ppSegment']] ) # npoints x 3
                
                worksheet.write('A12', 'Relative displacement of points in segment at each phase in cardiac cycle [rows=phases; columns=points; x,y,z]',bold)
                row = 11 # 11 = row 12 in excel
                col = 1
                for pDeforms in out['ppDeformsSegment']: # npoints x phases x 3
                    worksheet.write_column(row, col, [str(tuple(x)) for x in pDeforms] )
                    #row += 1
                    col += 1
            
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
                
            elif out['Type'] == 'chimney_angle_change':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Angle change of chimney centerline: max change for a point and peak angles of chimney',bold)
                
                worksheet.write('A3', 'Max angle change for a point on chimney',bold)
                worksheet.write('B3', out['point_angle_diff_max'] )
                
                worksheet.write('A4', 'Min and max angle of this point',bold)
                worksheet.write_row('B4', out['angles_minmax'] )
                
                worksheet.write('A5', 'Index of this point on cll (from prox or dist end)',bold)
                worksheet.write_row('B5', [phase[0] for phase in out['anglenodes_positions_cycle']] ) # index i of midnode n in phases
                worksheet.write('A6', 'Position of arm node n2 during cycle (x,y,z)',bold)
                worksheet.write_row('B6', [str(tuple(x[2])) for x in out['anglenodes_positions_cycle']] ) # nphases x 3 (xyz position)
                worksheet.write('A7', 'Position of arm node n1 during cycle (x,y,z)',bold)
                worksheet.write_row('B7', [str(tuple(x[3])) for x in out['anglenodes_positions_cycle']] ) # nphases x 3 
                worksheet.write('A8', 'Position of arm node n3 during cycle (x,y,z)',bold)
                worksheet.write_row('B8', [str(tuple(x[4])) for x in out['anglenodes_positions_cycle']] ) # nphases x 3 
                
                worksheet.write('A9', 'Location of this point from prox end of chimney, mm',bold)
                worksheet.write('B9', out['location_point_max_angle_change'] )
                
                worksheet.write('A10', 'Length of chimney at mid cardiac cycle, mm',bold)
                worksheet.write('B10', out['total_length_chimney_midCycle'] )
                
                worksheet.write('A11', 'Angle for this point at mid cardiac cycle [avgreg]',bold)
                worksheet.write('B11', out['angleMidCycle'])
                
                worksheet.write('A12', 'Angle for this point at each phase in cardiac cycle',bold)
                worksheet.write_row('B12', list(out['angles_phases']) )
                
                worksheet.write('A13', 'Mean and std of angles of this point during cardiac cycle',bold)
                worksheet.write_row('B13', out['angles_meanstd'])
                
                # now write peakAngle output
                worksheet.write('A15', 'Peak angle of chimney during cycle (peakAngle output):')
                
                worksheet.write('A16', 'Max angle diff between peak angle min and max',bold)
                worksheet.write('B16', out['peakAngle_phases_diff'] )
                
                worksheet.write('A17', 'Min and max peakAngle',bold)
                worksheet.write_row('B17', out['peakAngle_min_and_max'] )
                
                worksheet.write('A18', 'Index of peakAngle on cll during cardiac cycle (from prox or dist end)',bold)
                worksheet.write_row('B18', [x[0] for x in out['peakAngle_phases_location'] ])
                
                worksheet.write('A19', 'Distance from prox end chimney of peakAngle during cardiac cycle, mm',bold)
                worksheet.write_row('B19', [x[1] for x in out['peakAngle_phases_location'] ])
                
                worksheet.write('A20', 'PeakAngle at each phase in cardiac cycle',bold)
                worksheet.write_row('B20', list(out['peakAngle_per_phase']) )
                
                worksheet.write('A21', 'Mean and std of peakAngles during cardiac cycle',bold)
                worksheet.write_row('B21', out['peakAngle_phases_meanstd'])
                
                worksheet.write('A22', 'PeakAngle mid cycle [avgreg]',bold)
                worksheet.write('B22', out['peakAngle_midcycle'] )
                
                worksheet.write('A23', 'Index of peakAngle on cll at mid cycle (from prox or dist end)',bold)
                worksheet.write('B23', out['peakAngle_midcycle_location'][0] )
                
                worksheet.write('A24', 'Distance from prox end chimney of peakAngle at mid cycle, mm',bold)
                worksheet.write('B24', out['peakAngle_midcycle_location'][1] )
                
                
                
                
            
            elif out['Type'] == 'chimney_nel_angle_change':
                worksheet.write('A2', 'Type:', bold)
                worksheet.write('B2', 'Angle between prox chimney and prox nellix during the cycle',bold)
                
                worksheet.write('A3', 'Armlength for vectors, mm',bold)
                worksheet.write('B3', out['vectorArmlength'])
                
                worksheet.write('A4', 'Max angle diff between vector angle min and max',bold)
                worksheet.write('B4', out['vectorAngle_diff_max'] )
                
                worksheet.write('A5', 'Min and max vectorAngle',bold)
                worksheet.write_row('B5', out['vectorAngles_minmax'] )
                
                worksheet.write('A6', 'VectorAngle at each phase in cardiac cycle',bold)
                worksheet.write_row('B6', list(out['vectorAngles_phases']) )
                
                worksheet.write('A7', 'Mean and std of vectorAngles during cardiac cycle',bold)
                worksheet.write_row('B7', out['vectorAngles_meanstd'])
                
                worksheet.write('A8', 'VectorAngle at mid cardiac cycle [avgreg]',bold)
                worksheet.write('B8', out['vectorAngleMidCycle'])
                
                
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
                
                worksheet.write('A8', 'Straigth distance between ends at each phase in cardiac cycle',bold)
                worksheet.write_row('B8', list(out['straigth_dists']) )
                
                worksheet.write('A9', 'CLL length at each phase in cardiac cycle',bold)
                worksheet.write_row('B9', list(out['cll_dists']) )

        
        workbook.close()


# =======================================
def get_point_on_cll_at_armslength(ppCll, point1, armlength=10, type='distal', key=[]):
    """ From point1 on cll get point distally on cll at armslength 
    type: distal or proximal; to get point at armlength in distal or proximal 
    direction, assuming that origin of ct volume is cranial
    """
    pends = np.array([ppCll[0], ppCll[-1]])
    
    if 'vLRA' in key:
        if pends[0,0] > pends[1,0]: # check x, start was distal
            if type == 'distal': 
                ppCll = ppCll[::-1] # reverse array so that first point is proximal
            elif type == 'proximal':
                pass
        else: #start point is proximal
            if type == 'proximal':
                ppCll = ppCll[::-1] # reverse array so that first point is distal
            elif type == 'distal':
                pass
    elif 'vRRA' in key:
        if pends[0,0] < pends[1,0]: # check x, start was distal    
            if type == 'distal': 
                ppCll = ppCll[::-1] # reverse array so that first point is proximal
            elif type == 'proximal':
                pass
        else: #start point is proximal
            if type == 'proximal':
                ppCll = ppCll[::-1] # reverse array so that first point is distal
            elif type == 'distal':
                pass
    elif 'vSMA' in key:
        if pends[0,1] < pends[1,1]: # check y, start was distal
            if type == 'distal': 
                ppCll = ppCll[::-1] # reverse array so that first point is proximal
            elif type == 'proximal':
                pass
        else: #start point is proximal
            if type == 'proximal':
                ppCll = ppCll[::-1] # reverse array so that first point is distal
            elif type == 'distal':
                pass
    else:
        if pends[0,-1] > pends[1,-1]: # check of z, start point is distal
            if type == 'distal': 
                ppCll = ppCll[::-1] # reverse array so that first point is proximal
            elif type == 'proximal':
                pass
        else: #start point is proximal
            if type == 'proximal':
                ppCll = ppCll[::-1] # reverse array so that first point is distal
            elif type == 'distal':
                pass
    
    # find index of point1
    i_point1 = int(np.where( np.all(ppCll == point1, axis=-1) )[0])
    # loop through centerline from point1 to find point at armlength
    for i in range(len(ppCll)):
        point2 = ppCll[i_point1+i] # next point
        # calculate vector  
        vec1 = PointSet(3)        
        vec1.append(point1-point2)
        # get dist =length of vector
        dist = vec1.norm() # or np.linalg.norm(v)
        if dist >= armlength:
            break # break loop; point at armslength found
    
    return point2, vec1

def get_angle_at_fixed_arms(ppCll, p, armlength=15):
    """ Get angle at fixed arms length from point p.
    Arms length 15 mm is standard in angle tool in 3Mensio but chimney stent may be too short
    """
    index_p = np.where(np.all(ppCll == p, axis=-1))[0][0]
    npoints = len(ppCll)
    
    # find first point while looking at points before p
    for i in range(npoints):
        j = index_p - i
        if j < 0:
            # reached end of cll, we can't calculate angle for point p
            return None, None, None
        p1 = ppCll[j]
        # calculate vector  
        vec1 = PointSet(3)        
        vec1.append(p1-p)
        # get dist (length of vector)
        dist1 = vec1.norm() # or np.linalg.norm(v)
        if dist1 >= armlength:
            # print(i,j, dist1)
            break # break loop; point at armslength found
            
    # find second point while looking at points after p
    for i in range(npoints):
        j = index_p + i
        if j == npoints:
            # reached end of cll, we can't calculate angle for point p
            return None, None, None
        p3 = ppCll[j]
        # calculate vector  
        vec3 = PointSet(3)        
        vec3.append(p3-p)
        # get dists
        dist3 = vec3.norm()
        if dist3 >= armlength:
            #print(i,j, dist3)
            break # break loop point found
    
    # calc angle
    phi = abs(vec1.angle(vec3))
    angle = phi*180.0/np.pi # direction vector in degrees
    
    #print (p1, p3, angle)
    return p1, p3, angle
        

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
    maxangle = max(angles)
    minangle = min(angles)
    maxanglephase = int(np.where(angles==max(angles))[0]) * 10 # 180 degrees is straigth, so greater is smaller angle
    minanglephase = int(np.where(angles==min(angles))[0]) * 10
    maxanglechange = maxangle-minangle
    
    point_angle_diff_max = [maxanglechange, [maxanglephase, minanglephase]] # phases dias, syst
    
    print('Angle phases= {}'.format(angles))
    
    meanAnglesCycle = [np.mean(angles), np.std(angles)] 
    minmaxAnglesCycle = [np.min(angles), np.max(angles)]
    
    output = {
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
    distCllEnds = CllEnds[0].distance(CllEnds[-1]) # straight line
    ppCllLength = dist_over_centerline(ppCll, type='euclidian')
    tortuosity = ppCllLength/distCllEnds
    
    return tortuosity, distCllEnds, ppCllLength
    

def calculate_tortuosity_change(ppCll, ppCllDeforms):
    """ Tortuosity of centerline in different phases of cycle
    """
    # get positions of points in segment during cycle
    posCllCycle = np.zeros_like(ppCllDeforms)
    for i, ppDeform in enumerate(ppCllDeforms): # for each point
        posCllCycle[i] = ppCll[i] + ppDeform
    # calc tortuosity and change
    tortuosityPhases = []
    distCllEndsPhases = []
    ppCllLengthPhases = []
    for phasenr in range(ppCllDeforms.shape[1]):
        ppCll_phase = posCllCycle[:,phasenr,:]
        tortuosity, distCllEnds, ppCllLength = calculate_tortuosity(ppCll_phase)
        tortuosityPhases.append(tortuosity)
        distCllEndsPhases.append(distCllEnds)
        ppCllLengthPhases.append(ppCllLength) 
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
    'tort_diff_max': diff_max,
    'straigth_dists': distCllEndsPhases,
    'cll_dists': ppCllLengthPhases   }
    
    return output


def calculate_motion_points(key, ppCll, ppCllDeforms, lenSegment, part='prox'):
    """ Mean motion pattern of segment of points on Cll and mean amplitude
    of motion. Proximal segment is analyzed.
    ppCll is centerline path (points)
    part = 'prox' or 'dist'
    """
    def segment_when_start_distal(ppCll, lenSegment, part):
        if part == 'prox':
            pp = ppCll[-1*lenSegment:] # last points=prox segment
            ppDeforms = ppCllDeforms[-1*lenSegment:]
        elif part =='dist':
            pp = ppCll[:lenSegment]
            ppDeforms = ppCllDeforms[:lenSegment]
        return pp, ppDeforms
        
    def segment_when_start_proximal(ppCll, lenSegment, part):
        if part == 'prox':
            pp = ppCll[:lenSegment]
            ppDeforms = ppCllDeforms[:lenSegment]
        elif part == 'dist':
            pp = ppCll[-1*lenSegment:] # last points=dist segment
            ppDeforms = ppCllDeforms[-1*lenSegment:]
        return pp, ppDeforms
    
    pends = np.array([ppCll[0], ppCll[-1]])  # pp centerline is in order of centerline points
    
    if 'vLRA' in key:
        if pends[0,0] > pends[1,0]: # check x, start was distal
            pp, ppDeforms = segment_when_start_distal(ppCll, lenSegment, part)
        else: # first point=prox
            pp, ppDeforms = segment_when_start_proximal(ppCll, lenSegment, part)
    elif 'vRRA' in key:
        if pends[0,0] < pends[1,0]: # check x, start was distal
            pp, ppDeforms = segment_when_start_distal(ppCll, lenSegment, part) 
        else: # first point=prox
            pp, ppDeforms = segment_when_start_proximal(ppCll, lenSegment, part)
    elif 'vSMA' in key:
        if pends[0,1] < pends[1,1]: # check y, start was distal
            pp, ppDeforms = segment_when_start_distal(ppCll, lenSegment, part)
        else: # first point=prox
            pp, ppDeforms = segment_when_start_proximal(ppCll, lenSegment, part)
    else:
        if pends[0,-1] > pends[1,-1]: # check z, start was distal
            pp, ppDeforms = segment_when_start_distal(ppCll, lenSegment, part)
        else: # first point=prox
            pp, ppDeforms = segment_when_start_proximal(ppCll, lenSegment, part)
    
    # get CoM of segment at avgreg
    meanPosAvgSegment = np.mean(pp, axis = 0)
    
    # get positions of points in segment during cycle
    posCycleSegment = np.zeros_like(ppDeforms)
    for i, ppDeform in enumerate(ppDeforms): # for each point
        posCycleSegment[i] = pp[i] + ppDeform
    meanPosCycleSegment = np.mean(posCycleSegment, axis=0) # nphases x 3
    
    motionOutxyz = calculateMeanAmplitude(pp,ppDeforms, dim='xyz') # mean, std, min, max
    motionOutx = calculateMeanAmplitude(pp,ppDeforms, dim='x')
    motionOuty = calculateMeanAmplitude(pp,ppDeforms, dim='y')
    motionOutz = calculateMeanAmplitude(pp,ppDeforms, dim='z')
    
    # store in dict
    output = {'mean_segment_amplitudexyz_mean_std_min_max': motionOutxyz,
                'mean_segment_amplitudex_mean_std_min_max': motionOutx,
                'mean_segment_amplitudey_mean_std_min_max': motionOuty,
                'mean_segment_amplitudez_mean_std_min_max': motionOutz,
                'lengthSegment': lenSegment, 
                'meanSegmentPosCycle': meanPosCycleSegment,
                'meanPosAvgSegment': meanPosAvgSegment,
                'ppSegment': pp, # [positions nodes avgreg]
                'ppDeformsSegment': ppDeforms # [deforms nodes avgreg]
                }
    return output


def get_prox_dist_points_cll(ppCll1, key=[]):
    """ get prox and dist ends of the centerline assuming that origin is cranial,
    right, anterior
    """
    # Get prox and dist point for ppCll1
    pends1 = np.array([ppCll1[0], ppCll1[-1]])  # pp centerline is in order of centerline points
    if 'vLRA' in key:
        pends1 = pends1[pends1[:,0].argsort() ] # sort with x, ascending
        proxp1 = pends1[0]   # smallest x; origin is cranial, right, anterior
        distp1 = pends1[-1]
    elif 'vRRA' in key:
        pends1 = pends1[pends1[:,0].argsort() ] # sort with x, ascending
        proxp1 = pends1[-1]   # largest x; origin is cranial, right, anterior
        distp1 = pends1[0]
    elif 'vSMA' in key:
        pends1 = pends1[pends1[:,1].argsort() ] # sort with y, ascending
        proxp1 = pends1[-1]   # smallest x; origin is cranial, right, anterior
        distp1 = pends1[0]
    else:
        pends1 = pends1[pends1[:,-1].argsort() ] # sort with z, ascending
        proxp1 = pends1[0]   # smallest z; origin is cranial
        distp1 = pends1[-1]
    
    return proxp1, distp1

