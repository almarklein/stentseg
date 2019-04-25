""" Module to analyze ring-stent dynamics during the heart cycle from (dynamic) models
* ring curvature
* distance between ring1 and ring2
* displacement ring parts/quartiles

Uses excel pulsatility_and_expansion to get location of peaks and valleys
M.A. Koenrades. Created April 2019
"""

import sys, os
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume, plot_points
from stentseg.utils.centerline import dist_over_centerline
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.motion.vis import get_graph_in_phase, create_mesh_with_values
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges
from stentseg.stentdirect import stentgraph
import visvis as vv
import numpy as np
import copy
import math
from stentseg.motion.displacement import calculateMeanAmplitude
import xlsxwriter
import openpyxl
from lspeas.utils.curvature import measure_curvature, get_curvatures,length_along_path
from lspeas.utils.get_anaconda_ringparts import get_model_struts,get_model_rings, _get_model_hooks
from lspeas.analysis._plot_ring_motion import readPosDeformsOverCycle, orderlocation
from lspeas.utils.storesegmentation import make_model_dynamic
import pirt

class _Do_Analysis_Rings:
    """ Analyze motion and curvature of dynamic ring models
    Functions for chimneys/branches and main stents
    """
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis')
    
    # select the ssdf basedir
    basedir = select_dir(
        r'D:\LSPEAS\LSPEAS_ssdf',
        r'F:\LSPEAS_ssdf_backup',
        r'F:\LSPEAS_ssdf_BACKUP')
    
    def __init__(self,ptcode,ctcode,cropname,cropvol='stent', nstruts=8,
            showVol='MIP',removeStent=False, clim=(0,2500), **kwargs):
        """
        Init motion analysis on ring models 
        """
        self.exceldir =  _Do_Analysis_Rings.exceldir
        self.basedir = _Do_Analysis_Rings.basedir
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v2.1.xlsx'
        
        self.ptcode = ptcode
        self.ctcode = ctcode
        self.cropname = cropname
        self.cropvol = cropvol
        self.clim = clim
        self.showVol = showVol
        self.removeStent = removeStent
        
        # Load CT image data for reference, and deform data to measure motion
        try:
            # If we run this script without restart, we can re-use volume and deforms
            self.vol
            self.deforms
        except AttributeError:
            self.vol = loadvol(self.basedir, self.ptcode, self.ctcode, self.cropvol, 'avgreg').vol
            s_deforms = loadvol(self.basedir, self.ptcode, self.ctcode, self.cropname, 'deforms')
            deforms = [s_deforms[key] for key in dir(s_deforms) if key.startswith('deform')]
            deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
            self.deforms = deforms
            self.origin = s_deforms.origin
            
        # Load ring model
        try:
            self.model
        except AttributeError:
            self.s_model = loadmodel(self.basedir, self.ptcode, self.ctcode, self.cropname, 'modelavgreg')
            self.model = self.s_model.model 
        
        # Figure and init
        self.t = {} # store volume vis
        self.label = {}
        self.axes = {}
        f = vv.figure(); vv.clf()
        f.position = 8.00, 30.00,  1567.00, 1002.00
        a = vv.subplot(131)
        self.drawModel(model=[self.model], akey='a', a=a, color=['b'], mw=10)
        
        # get 2 seperate rings
        modelsout = get_model_struts(self.model, nstruts=nstruts)
        model_R1R2 = modelsout[2]
        # remove remaining strut parts if any
        modelhookparts, model_R1R2 = _get_model_hooks(model_R1R2) 
        modelR1, modelR2  = get_model_rings(model_R1R2)
        self.modelR1, self.modelR2 = modelR1, modelR2
        self.s_model['modelR1'] = modelR1
        self.s_model['modelR2'] = modelR2
        
        # vis rings
        colors = ['g', 'c']
        a0 = vv.subplot(132)
        self.drawModel(model=[self.modelR1, self.modelR2], akey='a0', a=a0, color=colors, mw=10)
        
        # init axis for mesh rings
        a1 = vv.subplot(133)
        self.drawModel(model=[self.model], akey='a1', a=a1, removeStent=True, alpha=0)
        # self.t['a1'].visible = False
        
        a.camera = a0.camera = a1.camera
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a, a0, a1], axishandling=False) )
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a, a0, a1]) )
        self.fig = f
        self.colors = colors
        self.mesh = {}
        
        # Initialize output variables to store analysis
        self.storeOutput = list()
        self.points_plotted = PointSet(3)
        
        
    def drawModel(self, model=[], akey='a', a=None, showVol=None, removeStent=None, isoTh=225, 
                climEditor=True, color=['b'], mw=10, alpha=0.6, **kwargs):
        """ Draw model(s) with white background in given axis
        model = list with models
        """
        if a is None:
            a = vv.gca()
        if showVol is None:
            showVol = self.showVol
        if removeStent is None:
            removeStent = self.removeStent
        
        a.MakeCurrent()
        
        a.axis.axisColor = 0,0,0#1,1,1
        a.axis.visible = False
        a.bgcolor = 1,1,1#0,0,0
        a.daspect = 1, 1, -1
        self.t[akey] = show_ctvolume(self.vol, graph=self.model, showVol=showVol, clim=self.clim, removeStent=removeStent, 
                        climEditor=climEditor, isoTh=isoTh, **kwargs)
        self.label[akey] = pick3d(a, self.vol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        for i, m in enumerate(model):
            m.Draw(mc=color[i], mw = mw, lc=color[i], alpha = alpha)
        vv.title('Model for LSPEAS %s  -  %s' % (self.ptcode[7:], self.ctcode))
        self.axes[akey] = a
        
    def curvature_ring_models(self):
        """ Calculate curvature of ring models R1, R2 
        * entire ring
        * 4 segments
        * mean and max curvature per phase
        * point with max change in curvature
        
        """
        s = self.s_model # ssdf with ring models
        deforms=self.deforms
        origin=self.origin
        
        for key in s:
            if key.startswith('modelR'): #R1 and R2
                model = s[key]
                name_output = 'Curvature {}'.format(key)
                self.calc_curvature_ring(model, key, deforms, origin, name_output)
                
        
    def calc_curvature_ring(self, model, key, deforms, origin, name_output, meshwithcurvature=True):
        """ Get curvature of ring model and change during the cardiac cycle
        For given ring model and quadrants of ring
        """
        # pop nodes
        model = pop_nodes_ring_models(model, deforms, origin) #duplicates on path were removed
        n1,n2 = model.edges()[0] # model has 1 edge, connected to same node so n1==n2
        pp = model.edge[n1][n2]['path']
        ppdeforms = model.edge[n1][n2]['pathdeforms']
        # measure curvature
        # mean_per_phase, max_per_phase, max_per_phase_loc, max_change = measure_curvature(pp, deforms) 
        #todo: xyz and origin error in measure_curvature?
        
        # Read locations peaks vallleys
        posAnt, posPost, posLeft, posRight = readLocationPeaksValleys(self.exceldir, 
                self.workbook_stent, self.ptcode, self.ctcode, ring=key[-2:])
        
        # Calculate curvature each phase for all points
        pp = np.asarray(pp) # nx3
        ppdeforms = np.asarray(ppdeforms) # nx10x3
        curvatures_per_phase=[]
        for phase in range(len(ppdeforms[0])): # len=10
            cv = get_curvatures(pp + ppdeforms[:,phase,:])
            # convert mm-1 to cm-1
            cv *= 10
            curvatures_per_phase.append(cv)
        
        # Mean curvature per phase (1 value per phase)
        mean_per_phase = []
        for curvatures in curvatures_per_phase:
            mean_per_phase.append(float(curvatures.mean()))
    
        # Max curvature per phase and position (1 tuple per phase)
        max_per_phase = []
        max_per_phase_loc = []
        for curvatures in curvatures_per_phase:
            index = np.argmax(curvatures)
            max_per_phase.append((float(curvatures[index])))
            max_per_phase_loc.append(length_along_path(pp, index))
            #todo: define loc with respect to anterior or segments
            
        # Max change in curvature (index, indexmm, max-change)
        max_change_index, max_change, max_value, min_value = 0, 0, 0, 0
        curvature_change = [] # collect change for each point
        for index in range(len(pp)):
            curvature_per_phase = [float(curvatures_per_phase[phase][index]) for phase in range(len(deforms))] # 10 values for point
            change = max(curvature_per_phase) - min(curvature_per_phase)
            curvature_change.append(change)
            if change > max_change:
                max_change_index, max_change_point = index, pp[index]
                max_change = change
                max_changeP = 100 * (change / min(curvature_per_phase))
                max_value, min_value = max(curvature_per_phase), min(curvature_per_phase)
                max_curvature_per_phase = curvature_per_phase
        
        # max_change_location = length_along_path(pp, max_change_index) # in mm but which way from node?
        # todo: define position as in which segment and/or closest to A, LA, L, LP, P, RP, R, RA
        # location_of_max_change = dist_over_centerline(pp[:-1],cl_point1=pp[0],
        #                             cl_point2=max_change_point,type='euclidian') # distance from proximal end centerline
        
        # add curvature change to model edge for visualization
        model.add_edge(n1, n2, path_curvature_change = curvature_change) # add new key
        
        # visualize
        if True: # False=do not show
            ax = self.axes['a0']
            mw =15
            view = ax.GetView()
            ax.Clear()
            #draw both rings again, current being popped
            self.drawModel(model=[self.modelR1, self.modelR2], akey='a0', a=ax, color=self.colors, mw=10)
            # plot point of max curve change
            mc = 'r'
            self.points_plotted.append(max_change_point) # add to PointSet
            plotted_point = plot_points(self.points_plotted,mc=mc,mw=mw,ls='',alpha=0.7,ax=ax)
            ax.SetView(view)
            # vis mesh 
            if meshwithcurvature:
                modelmesh = create_mesh_with_values(model, valueskey='path_curvature_change', radius=0.6)
                a1 = self.axes['a1']
                m = vv.mesh(modelmesh, axes=a1)
                # m.clim = climcurv
                m.colormap = vv.CM_JET
                # check if colorbar already created
                if not self.mesh: # empty dict evaluates False
                    vv.colorbar(axes=a1)
                self.mesh[key] = m
        
        # store output
        print('Curvature during cycle of point with max curvature change= {}'.format(max_curvature_per_phase))
        print('')
        meanCurvatureCycle = [np.mean(max_curvature_per_phase), np.std(max_curvature_per_phase)] 
        minmaxCurvatureCycle = [min_value, max_value]
        # 1/0
        # output = {}
        # output['peakCurvature_per_phase'] = peakAngle_phases
        # output['point_angle_diff_max'] = angleChange 
        # output['angles_phases'] = anglesCycle 
        # output['angles_meanstd'] = meanAnglesCycle 
        # output['angles_minmax'] = minmaxAnglesCycle # where min is sharpest angle
        # output['anglenodes_positions_cycle'] = anglesCycleNodes #list 10xlist index,n,n2  ,n1,n3 for arms
        # output['location_point_max_angle_change'] = location_of_midpoint_n # dist from proximal end
        # output['total_length_chimney_midCycle'] = total_length_cll
        # output['angleMidCycle'] = anglenew
        # 
        # 
        # # ============
        # # Store output with name
        # output['Name'] = name_output
        # output['Type'] = 'chimney_angle_change'
        # 
        # self.angleChange_output = output
        # self.storeOutput.append(output)



def pop_nodes_ring_models(model, deforms, origin):
    """ For curvature calculations and distances between rings, remove nodes to
    define ring models as single edge
    origin is origin of volume crop used in registration to get deforms, stored in s_deforms
    * graph one edge, made dynamic
    """
    
    # remove 'nopop' tags from nodes to allow pop for all (if these where created)
    # remove deforms
    for n in model.nodes():
        d = model.node[n] # dict
        # use dictionary comprehension to delete key
        for key in [key for key in d if key == 'nopop']: del d[key]
        for key in [key for key in d if key == 'deforms']: del d[key]
    # pop
    stentgraph.pop_nodes(model)
    assert model.number_of_edges() == 1 # a ring model is now one edge
    
    # remove duplicates in middle of path (due to last pop?)
    n1,n2 = model.edges()[0]
    pp = model.edge[n1][n2]['path'] # PointSet
    duplicates = []
    for p in pp[1:-1]: # exclude begin/end node
        index = np.where( np.all(pp == p, axis=-1))
        if len(np.array(index)[0]) == 2: # occurred at positions
            duplicates.append(p) 
        elif len(np.array(index)[0]) > 2:
            print('A point on the ring model occurred at more than 2 locations')
    # remove 1 occurance (remove duplicates)
    duplicates = set(tuple(p.flat) for p in duplicates )
    # turn into a PointSet
    duplicates = PointSet(np.array(list(duplicates)))
    for p in duplicates:
        pp.remove(p) # remove first occurance 
    # now change the edge path
    model.add_edge(n1, n2, path = pp)
    
    # make dynamic again
    incorporate_motion_nodes(model, deforms, origin) # adds deforms PointSets
    incorporate_motion_edges(model, deforms, origin) # adds deforms PointSets
    
    return model
        
        
def readLocationPeaksValleys(exceldir, workbook_stent, ptcode, ctcode, ring='R1', deforms=False):
    """ Get location ant, post, left, right on R1 or R2
    reorder read peak valley locations based on anatomy: ant, post, left, right
    * coordinates of locations
    * deforms of locations could be option
    """
    # read workbook
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
    
    # read sheet
    sheetname = 'Output_'+ ptcode[-3:]
    sheet = wb.get_sheet_by_name(sheetname)  
    # read peaks valleys for each ring
    rowsStart = [18,31,55,68]
    # timepoints = ['discharge', '1M', '6M', '12M', '24M']
    timepoints = ['discharge', '1month', '6months', '12months', '24months']
    
    if ring == 'R1':
        colStart = 'B'
    elif ring == 'R2':
        colStart = 'V'
    
    time = timepoints.index(ctcode)
    # read locations, deforms
    positionAnt, deformsAnt, poscycleAnt = readPosDeformsOverCycle(
        sheet, time=time, rowStart=rowsStart[0], colStart=colStart, nphases=10 ) # 1x3, 10x3, and 10x3
    positionPost, deformsPost, poscyclePost = readPosDeformsOverCycle(
        sheet, time=time, rowStart=rowsStart[1], colStart=colStart, nphases=10 )
    positionLeft, deformsLeft, poscycleLeft = readPosDeformsOverCycle(
        sheet, time=time, rowStart=rowsStart[2], colStart=colStart, nphases=10 )
    positionRight, deformsRight, poscycleRight = readPosDeformsOverCycle(
        sheet, time=time, rowStart=rowsStart[3], colStart=colStart, nphases=10 )
    # check and reorder ant post left right
    P = [positionAnt, positionPost, positionLeft, positionRight]
    D = [deformsAnt,  deformsPost,  deformsLeft,  deformsRight]
    indexorder = orderlocation(positionAnt, positionPost, positionLeft, positionRight) # index A, P, L, R
    R1positionAnt, R1deformsAnt = P[indexorder[0]], D[indexorder[0]]
    R1positionPost, R1deformsPost = P[indexorder[1]], D[indexorder[1]]
    R1positionLeft, R1deformsLeft = P[indexorder[2]], D[indexorder[2]]
    R1positionRight, R1deformsRight = P[indexorder[3]], D[indexorder[3]]
    
    if deforms:
        return ([R1positionAnt,R1deformsAnt], [R1positionPost,R1deformsPost], 
               [R1positionLeft,R1deformsLeft], [R1positionRight,R1deformsRight])
    else:
        return R1positionAnt, R1positionPost, R1positionLeft, R1positionRight




if __name__ == '__main__':
    
    # Select dataset
    ptcode = 'LSPEAS_001'
    ctcode = '24months'
    cropname = 'ring'
    
    showVol  = 'ISO'  # MIP or ISO or 2D or None
    nstruts = 8 # needed to get seperate ring models for R1 R2; 8 normally
    
    foo = _Do_Analysis_Rings(ptcode,ctcode,cropname,nstruts=nstruts,showVol=showVol)
    
    foo.curvature_ring_models()
    