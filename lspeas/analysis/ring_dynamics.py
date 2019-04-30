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
from stentseg.utils.utils_graphs_pointsets import point_in_pointcloud_closest_to_p
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
        
        # Read locations peaks vallleys
        self.posAnt, self.posPost, self.posLeft, self.posRight = {}, {}, {}, {}
        self.posAnt['R1'], self.posPost['R1'], self.posLeft['R1'], self.posRight['R1'] = readLocationPeaksValleys(
            self.exceldir, self.workbook_stent, self.ptcode, self.ctcode, ring='R1')
        self.posAnt['R2'], self.posPost['R2'], self.posLeft['R2'], self.posRight['R2'] = readLocationPeaksValleys(
            self.exceldir, self.workbook_stent, self.ptcode, self.ctcode, ring='R2')
        
    def drawModel(self, model=[], akey='a', a=None, showVol=None, removeStent=None, isoTh=180, 
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
        * mean and max curvature per phase and change
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
                
        
    def calc_curvature_ring(self, model, key, deforms, origin, name_output, 
        posStart=None, smoothfactor=30, meshwithcurvature=True):
        """ Get curvature of ring model and change during the cardiac cycle
        For given ring model and quadrants of ring
        """
        if posStart is None:
            posStart = self.posPost[key[-2:]]
            
        # pop nodes, get posterior as start node
        model = pop_nodes_ring_models(model, deforms, origin, posStart=posStart, smoothfactor=smoothfactor) #duplicates on path were removed
        
        n1,n2 = model.edges()[0] # model has 1 edge, connected to same node so n1==n2
        pp = model.edge[n1][n2]['path']
        ppdeforms = model.edge[n1][n2]['pathdeforms']
        # measure curvature
        # mean_per_phase, max_per_phase, max_per_phase_loc, max_change = measure_curvature(pp, deforms) 
        #todo: xyz and origin error in measure_curvature?
        
        # Calculate curvature each phase for all points
        pp = np.asarray(pp) # nx3
        ppdeforms = np.asarray(ppdeforms) # nx10x3
        curvatures_per_phase=[] # list with 10 times nx1 values
        for phase in range(len(ppdeforms[0])): # len=10
            cv = get_curvatures(pp + ppdeforms[:,phase,:])
            # convert mm-1 to cm-1
            cv *= 10
            curvatures_per_phase.append(cv)
        
        # store for reuse with segments
        self.curvatures_per_phase = curvatures_per_phase
        
        output = {}
        
        mean_curvature_output = curvature_mean(curvatures_per_phase)
        output['mean_curvature_per_phase'] = mean_curvature_output[0]
        output['mean_curvature_phase_minmax'] = [mean_curvature_output[1], mean_curvature_output[2]]
        output['mean_curvature_phase_change'] = mean_curvature_output[3] 
        
        max_curvature_output = curvature_max(curvatures_per_phase, pp)
        output['max_per_phase'] = max_curvature_output[0]
        output['max_per_phase_index'] = max_curvature_output[1]
        output['max_per_phase_rel_index'] = max_curvature_output[2]
        output['max_per_phase_location'] = max_curvature_output[3]
        output['max_phase_minmax'] = [max_curvature_output[4], max_curvature_output[5]]
        output['max_phase_change'] = max_curvature_output[6]
        
        curvature_maxchange_output = curvature_maxchange(curvatures_per_phase, pp)
        output['curvature_change'] = curvature_maxchange_output[0] #value for each point in pp
        output['max_change'] = curvature_maxchange_output[1]
        output['max_changeP'] = curvature_maxchange_output[2]
        output['min_and_max_value'] = [curvature_maxchange_output[3], curvature_maxchange_output[4]]
        output['max_change_index'] = curvature_maxchange_output[5]
        output['max_change_point'] = curvature_maxchange_output[6] # x,y,z location of point on graph
        output['rel_index_max_change'] = curvature_maxchange_output[7] # percentage of total points
        output['max_change_location'] = curvature_maxchange_output[8] # in mm from startnode
        output['total_ring_perimeter'] = curvature_maxchange_output[9]
        output['max_point_curvature_per_phase'] = curvature_maxchange_output[10]
        output['curvature_midcycle'] = curvature_maxchange_output[11] #value for each point in pp
        
        # # Mean curvature per phase (1 value per phase)
        # mean_per_phase = []
        # for curvatures in curvatures_per_phase:
        #     mean_per_phase.append(float(curvatures.mean()))
    
   #    #   # Max curvature per phase and position (1 tuple per phase)
        # max_per_phase = []
        # max_per_phase_location = [] # mm
        # max_index = []
        # max_rel_index = [] # %
        # for curvatures in curvatures_per_phase:
        #     index = np.argmax(curvatures)
        #     max_per_phase.append((float(curvatures[index])))
        #     max_per_phase_location.append(length_along_path(pp, index)) # in mm from startnode
        #     max_index.append(index)
        #     max_rel_index.append(100* (index/(len(pp)-1) )) # percentage of total points; estimation of location relative to posStart
        #     
        # # Max change in curvature (index, indexmm, max-change)
        # max_change_index, max_change, max_value, min_value = 0, 0, 0, 0
        # curvature_change = [] # collect change for each point
        # for index in range(len(pp)):
        #     curvature_per_phase = [float(curvatures_per_phase[phase][index]) for phase in range(len(deforms))] # 10 values for point
        #     change = max(curvature_per_phase) - min(curvature_per_phase)
        #     curvature_change.append(change)
        #     if change > max_change:
        #         max_change_index, max_change_point = index, pp[index]
        #         max_change = change
        #         max_changeP = 100 * (change / min(curvature_per_phase))
        #         max_value, min_value = max(curvature_per_phase), min(curvature_per_phase)
        #         max_curvature_per_phase = curvature_per_phase
        # 
        # max_change_location = length_along_path(pp, max_change_index) # in mm from startnode
        # # location_of_max_change = dist_over_centerline(pp[:-1],cl_point1=pp[0],
        # #    cl_point2=max_change_point,type='euclidian') # distance from proximal end centerline; same output
        # rel_index_max_change = 100* (max_change_index/(len(pp)-1) ) # percentage of total points
        # total_ring_perimeter = length_along_path(pp, len(pp)-1)
        
        
        # todo: get curvature of segments
        lenring = len(pp)
        indicesQ025 = 0, int(0.25*lenring) # int floor rounding
        indicesQ2550 = int(0.25*lenring)+1, int(0.5*lenring)
        indicesQ5075 = int(0.5*lenring)+1,  int(0.75*lenring)
        indicesQ75100 = int(0.75*lenring)+1,  lenring-1
        
        
        
        # add curvature change to model edge for visualization
        model.add_edge(n1, n2, path_curvature_change = output['curvature_change'] ) # add new key
        model.add_edge(n1, n2, path_curvature_midcycle = output['curvature_midcycle'] )
        
        # visualize
        if True: # False=do not show
            ax = self.axes['a0']
            mw =15
            view = ax.GetView()
            ax.Clear()
            #draw both rings again, current being popped
            self.drawModel(model=[self.modelR1, self.modelR2], akey='a0', a=ax, color=self.colors, mw=10)
            # self.drawModel(model=[self.model], akey='a0', a=ax, color=['b'], mw=10) # to compare
            # plot point of max curve change
            mc = 'r'
            self.points_plotted.append(output['max_change_point']) # add to PointSet
            plotted_point = plot_points(self.points_plotted,mc=mc,mw=mw,ls='',alpha=0.7,ax=ax)
            ax.SetView(view)
            # vis mesh 
            if meshwithcurvature:
                modelmesh = create_mesh_with_values(model, valueskey='path_curvature_change', radius=0.6)
                # modelmesh = create_mesh_with_values(model, valueskey='curvature_midcycle', radius=0.6)
                a1 = self.axes['a1']
                m = vv.mesh(modelmesh, axes=a1)
                m.clim = 0, 0.1 # when 2 rings are plotted seperately, clim must be fixed
                m.colormap = vv.CM_JET
                # check if colorbar already created
                if not self.mesh: # empty dict evaluates False
                    vv.colorbar(axes=a1)
                self.mesh[key] = m
        
        # print
        print('Curvature during cycle of point with max curvature change= {}'.format(output['max_point_curvature_per_phase']))
        print('')
        print('Max curvature change= {}'.format(output['max_change']) )
        
        # ============
        # Store output with name
        output['Name'] = name_output
        output['Type'] = 'curvature_entire_ring'
        
        self.curvatureChange_output = output
        self.storeOutput.append(output)

    
    def calc_curvature_ring_segments(self, model, key, deforms, origin, name_output, 
        posStart=None):
        """
        """
    
    
def curvature_mean(curvatures_per_phase):
    """ Given the curvatures along pp for each phase, calculate mean curvature
    """
    # Mean curvature per phase (1 value per phase)
    mean_curvature_per_phase = []
    for curvatures in curvatures_per_phase:
        mean_curvature_per_phase.append(float(curvatures.mean()))
    
    # stats
    mean_curvature_phase_min = min(mean_curvature_per_phase)
    mean_curvature_phase_max = max(mean_curvature_per_phase)
    mean_curvature_phase_change = mean_curvature_phase_max - mean_curvature_phase_min
    
    return (mean_curvature_per_phase, mean_curvature_phase_min, mean_curvature_phase_max, 
            mean_curvature_phase_change )
    
def curvature_max(curvatures_per_phase, pp):
    """ Given the curvatures along pp for each phase, calculate max curvature
    """
    # Max curvature per phase and position (1 tuple per phase)
    max_per_phase = []
    max_per_phase_location = [] # mm
    max_per_phase_index = []
    max_per_phase_rel_index = [] # %
    for curvatures in curvatures_per_phase:
        index = np.argmax(curvatures)
        max_per_phase.append((float(curvatures[index])))
        max_per_phase_location.append(length_along_path(pp, index)) # in mm from startnode
        max_per_phase_index.append(index)
        max_per_phase_rel_index.append(100* (index/(len(pp)-1) )) # percentage of total points; estimation of location relative to posStart
    
    # stats
    max_phase_min = min(max_per_phase)
    max_phase_max = max(max_per_phase)
    max_phase_change = max_phase_max - max_phase_min
    
    return (max_per_phase, max_per_phase_index, max_per_phase_rel_index, max_per_phase_location,
            max_phase_min, max_phase_max, max_phase_change)
    
def curvature_maxchange(curvatures_per_phase, pp):
    """ Given the curvatures along pp for each phase, calculate point with max change in 
    curvature during phases (cardiac cycle)
    """
    # Max change in curvature (index, indexmm, max-change)
    max_change_index, max_change, max_value, min_value = 0, 0, 0, 0
    curvature_change = [] # collect change for each point
    curvature_midcycle = [] # collect mean value over phases as mid cycle
    for index in range(len(pp)):
        curvature_per_phase = [float(curvatures_per_phase[phase][index]) for phase in range(len(curvatures_per_phase))] # 10 values for point
        change = max(curvature_per_phase) - min(curvature_per_phase)
        curvature_change.append(change)
        curvature_midcycle.append(np.mean(curvature_per_phase))
        if change > max_change:
            max_change_index, max_change_point = index, pp[index]
            max_change = change
            max_changeP = 100 * (change / min(curvature_per_phase))
            max_value, min_value = max(curvature_per_phase), min(curvature_per_phase)
            max_point_curvature_per_phase = curvature_per_phase
    
    max_change_location = length_along_path(pp, max_change_index) # in mm from startnode
    # location_of_max_change = dist_over_centerline(pp[:-1],cl_point1=pp[0],
    #    cl_point2=max_change_point,type='euclidian') # distance from proximal end centerline; same output
    rel_index_max_change = 100* (max_change_index/(len(pp)-1) ) # percentage of total points
    total_perimeter = length_along_path(pp, len(pp)-1)
    
    return (curvature_change, max_change, max_changeP, min_value, max_value, 
            max_change_index, max_change_point, rel_index_max_change, max_change_location,
            total_perimeter, max_point_curvature_per_phase, curvature_midcycle)

def set_direction_of_pp(pp):
    """ order direction of path so that we always go the same way and can define locations
    order going clock wise from posterior so that left is ~25% and right is ~75% (top view)
    """
    pplength = len(pp)
    p25 = pp[int(0.25*pplength)]
    p75 = pp[int(0.75*pplength)]
    # check which is left and right = dim 0
    if not p25[0] > p75[0]:
        # reorder back to front
        pp = [pp[-i] for i in range(1,len(pp)+1)]
    
    return pp
    
def pop_nodes_ring_models(model, deforms, origin, posStart, smoothfactor=2):
    """ For curvature calculations and distances between rings, remove nodes to
    define ring models as single edge
    origin is origin of volume crop used in registration to get deforms, stored in s_deforms
    * graph with one edge, popped, smoothed, and made dynamic
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
    
    # to relate to location posStart, add node to make this start of pp, remove old edge
    # get point on graph closest to posStart (will not be exactly the same after pop smooth)
    n1,n2 = model.edges()[0]
    assert n1 == n2 # connected to self
    pp2 = model.edge[n1][n2]['path'] # PointSet
    pointongraph = point_in_pointcloud_closest_to_p(pp2, posStart)[0] # p in pp, point 
    pointongraph = tuple(pointongraph.flat) # PointSet to tuple
    if not pointongraph == tuple(pp2[0].flat): #already is the startnode
        # get path parts from pointongraph to current node
        pp3 = [tuple(p.flat) for p in pp2]
        index = pp3.index(pointongraph)
        path1 = pp3[1:index+1]
        path2 = pp3[index:]
        newpath = path2 + path1
        #remove old node and path
        model.remove_node(n1)
        #add newpath
        # set direction of path going clockwise (left) from posterior (0)
        newpath = set_direction_of_pp(newpath)
        # nodes.add_edge(n1, pointongraph, path = PointSet(np.asarray(path1))  )
        # model.add_edge(pointongraph, n2, path = PointSet(np.asarray(path2)) )
        model.add_edge(pointongraph, pointongraph, path = PointSet(np.asarray(newpath)) )
    else:
        print('start node of edge was the same as the reference start pos (posterior): extra smooth needed here before analysis?')
    
    # smooth path
    stentgraph.smooth_paths(model, ntimes=smoothfactor)
       
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
    