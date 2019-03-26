""" Measure stent migration relative to renals
Option to visualize 2 longitudinal scans
"""
import sys, os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.utils.visualization import show_ctvolume
from stentseg.utils import _utils_GUI, PointSet
from stentseg.utils.picker import pick3d
from stentseg.utils.centerline import find_centerline, points_from_mesh, smooth_centerline, dist_over_centerline
from lspeas.analysis.utils_analysis import ExcelAnalysis
from stentseg.utils.utils_graphs_pointsets import point_in_pointcloud_closest_to_p
#sys.path.insert(0, os.path.abspath('..')) # parent, 2 folders further in pythonPath
#import utils_analysis
#from utils_analysis import ExcelAnalysis
#import get_anaconda_ringparts
from lspeas.utils.get_anaconda_ringparts import _get_model_hooks,get_midpoints_peaksvalleys,identify_peaks_valleys

#todo: from outline to script:

## Initialize
# select the ssdf basedir
basedir = select_dir(r'F/LSPEAS\LSPEAS_ssdf',
                     r'F/LSPEAS_ssdf_backup')
                     
basedirstl = select_dir(r'D:\Profiles\koenradesma\Dropbox\UTdrive\MedDataMimics\LSPEAS_Mimics\Tests')

# select dataset
ptcode = 'LSPEAS_003'
ctcodes = ctcode1, ctcode2 = 'discharge', '12months' # ctcode2 = None if no second code
cropname = 'ring'
modelname = 'modelavgreg'
vesselname1 = 'LSPEAS_003_D_MK Smoothed_Wrapped1.0_edit-smart 4_copy_001.stl'
# LSPEAS_003_D_MK Smoothed_Wrapped1.0_edit-smart 4_copy_noRenals 7_001
vesselname2 = 'LSPEAS_003_12M_MK Smoothed_Wrapped1.0_smart 3_copy_001.stl'
sheet_renals_obs = 'renal locations obs1'

showAxis = True  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
ringpart = True # True; False
clim0  = (0,2500)
# clim0 = -550,500
isoTh = 250
meshradius = 0.7

# create class object for excel analysis
foo = ExcelAnalysis() # excel locations initialized in class

## Renal origin coordinates: input by user/read excel
# coordinates, left and right most caudal renal
# ctcode1
xrenal1, yrenal1, zrenal1 = 132.7, 89.2, 85.5
renal1 = PointSet(list((xrenal1, yrenal1, zrenal1)))
# ctcode2
if ctcode2:
    xrenal2, yrenal2, zrenal2 = 171, 165.1, 39.5
    renal2 = PointSet(list((xrenal2, yrenal2, zrenal2)))

# renal_left, renal_right = foo.readRenalsExcel(sheet_renals_obs, ptcode, ctcode1)
# renal1 = renal_left

## Load (dynamic) stent models, vessel, ct
# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode1, cropname, 'avgreg')
vol1 = s.vol
if ctcode2:
    s = loadvol(basedir, ptcode, ctcode2, cropname, 'avgreg')
    vol2 = s.vol

# load stent model
s2 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
model1 = s2.model
modelmesh1 = create_mesh(model1, meshradius)
if ctcode2:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
    model2 = s2.model
    modelmesh2 = create_mesh(model2, meshradius)

# Load vessel mesh (output Mimics)
vessel1 = loadmesh(basedirstl,ptcode,vesselname1) #inverts Z
if ctcode2:
    vessel2 = loadmesh(basedirstl,ptcode,vesselname2) #inverts Z
# get pointset from STL 
ppvessel1 = points_from_mesh(vessel1, invertZ = False) # removes duplicates
if ctcode2:
    ppvessel2 = points_from_mesh(vessel2, invertZ = False) # removes duplicates


## Create centerline: input start/end
# ctcode1
c1_start1 = (153, 86, 104.5) # distal end
c1_ends = [(142, 94, 64.5)] # either single point or multiple
centerline1 = find_centerline(ppvessel1, c1_start1, c1_ends, 0.5, ndist=20, regfactor=0.2, regsteps=10)
centerline1 = smooth_centerline(centerline1, 30) # 20 iterations for stepsize 0.5 is reasonable
# ctcode2
if ctcode2:
    c2_start1 = (190, 165, 60) # distal end
    c2_ends = [(179, 169, 17)] # either single point or multiple
    centerline2 = find_centerline(ppvessel2, c2_start1, c2_ends, 0.5, ndist=20, regfactor=0.2, regsteps=10)
    centerline2 = smooth_centerline(centerline2, 30)

# scipy.ndimage.interpolation.zoom
# scipy.interpolate.interpn

## Get peak and valley points

if False:
# ===== OPTION automated detection =====
    # get midpoints peaks valleys
    midpoints_peaks_valleys = get_midpoints_peaksvalleys(model1)
    # from peaks valley pointcloud identiy peaks and valleys
    R1_left,R2_left,R1_right,R2_right,R1_ant,R2_ant,R1_post,R2_post = identify_peaks_valleys(
    midpoints_peaks_valleys, model1, vol1,vis=True)

# ===== OPTION excel =====

R1 = foo.readRingExcel(ptcode, ctcode1, ring='R1')
R1_ant, R1_post, R1_left, R1_right = R1[0], R1[1], R1[2], R1[3]


##
#todo: orientatie aorta bepalen dmv 4 hooks -> gemiddelde hoek
# z distance hiermee corrigeren
R2 = foo.readRingExcel(ptcode, ctcode1, ring='R2')
R2_ant, R2_post, R2_left, R2_right = R2[0], R2[1], R2[2], R2[3]

def get_stent_orientation(R1, R2):
    R1, R2 = np.asarray(R1), np.asarray(R2)
    R1, R2 = PointSet(R1), PointSet(R2) # turn array ndim2 into PointSet
    R1_ant, R1_post, R1_left, R1_right = R1[0], R1[1], R1[2], R1[3]
    R2_ant, R2_post, R2_left, R2_right = R2[0], R2[1], R2[2], R2[3]
    refvector = [0,0,10] # z-axis
    angle = (R1_ant-R2_ant).angle(refvector) # order does not matter

## Calculate distance ring peaks and valleys to renal

# ===== in Z =====
# proximal to renal is positive; origin is proximal
z_dist_R1_ant = list(renal1.flat)[2]-R1_ant[2]
z_dist_R1_post = list(renal1.flat)[2]-R1_post[2]
z_dist_R1_left = list(renal1.flat)[2]-R1_left[2]
z_dist_R1_right = list(renal1.flat)[2]-R1_right[2]

# ===== along centerline =====
# point of centerline closest to renal
renal1_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, renal1)
if ctcode2:
    renal2_and_cl_point = point_in_pointcloud_closest_to_p(centerline2, renal2)
# point of centerline closest to peaks valleys
R1_left_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, R1_left)
R1_right_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, R1_right) 
R1_ant_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, R1_ant)
R1_post_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, R1_post)
# calculate distance over centerline
dist_for_R1_left = dist_over_centerline(centerline1, R1_left_and_cl_point[0], renal1_and_cl_point[0])
dist_for_R1_right = dist_over_centerline(centerline1, R1_right_and_cl_point[0], renal1_and_cl_point[0])
dist_for_R1_ant = dist_over_centerline(centerline1, R1_ant_and_cl_point[0], renal1_and_cl_point[0])
dist_for_R1_post = dist_over_centerline(centerline1, R1_post_and_cl_point[0], renal1_and_cl_point[0])

# Main outcome 1: distance 2nd ring valleys to renal
# Main outcome 2: migration 2nd ring valleys from discharge to 1, 6, 12 months

## Visualize
f = vv.figure(2); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
alpha = 0.5
if ctcode2:
    a1 = vv.subplot(121)
else:
    a1 = vv.gca()
show_ctvolume(vol1, model1, showVol=showVol, clim=clim0, isoTh=isoTh)
pick3d(vv.gca(), vol1)
model1.Draw(mc='b', mw = 10, lc='g')
vm = vv.mesh(modelmesh1)
vm.faceColor = 'g'
# m = vv.mesh(vessel1)
# m.faceColor = (1,0,0, alpha) # red

# vis vessel, centerline, renal origo, peaks valleys R1
vv.plot(ppvessel1, ms='.', ls='', mc= 'r', alpha=0.2, mw = 7, axes = a1) # vessel
vv.plot(PointSet(list(c1_start1)), ms='.', ls='', mc='g', mw=18, axes = a1) # start1
vv.plot([e[0] for e in c1_ends], [e[1] for e in c1_ends],  [e[2] for e in c1_ends],  ms='.', ls='', mc='b', mw=18, axes = a1) # ends
vv.plot(centerline1, ms='.', ls='', mw=8, mc='y', axes = a1)
vv.plot(renal1, ms='.', ls='', mc='m', mw=18, axes = a1)
vv.plot(renal1_and_cl_point, ms='.', ls='-', mc='m', mw=18, axes = a1)
# vv.plot(R1_left_and_cl_point, ms='.', ls='-', mc='c', mw=18, axes = a1)
# vv.plot(R1_right_and_cl_point, ms='.', ls='-', mc='c', mw=18, axes = a1)
# vv.plot(R1_ant_and_cl_point, ms='.', ls='-', mc='c', mw=18, axes = a1)
# vv.plot(R1_post_and_cl_point, ms='.', ls='-', mc='c', mw=18, axes = a1)

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
a1.axis.axisColor= 1,1,1
a1.bgcolor= 0,0,0
a1.daspect= 1, 1, -1  # z-axis flipped
a1.axis.visible = showAxis

if ctcode2:
    a2 = vv.subplot(122)
    show_ctvolume(vol2, model2, showVol=showVol, clim=clim0, isoTh=isoTh)
    pick3d(vv.gca(), vol2)
    model2.Draw(mc='b', mw = 10, lc='g')
    vm = vv.mesh(modelmesh2)
    vm.faceColor = 'g'
    # m = vv.mesh(vessel2)
    # m.faceColor = (1,0,0, alpha) # red
    
    # vis vessel, centerline, renal origo, peaks valleys R1
    vv.plot(ppvessel2, ms='.', ls='', mc= 'r', alpha=0.2, mw = 7, axes = a2) # vessel
    vv.plot(PointSet(list(c2_start1)), ms='.', ls='', mc='g', mw=18, axes = a2) # start1
    vv.plot([e[0] for e in c2_ends], [e[1] for e in c2_ends],  [e[2] for e in c2_ends],  ms='.', ls='', mc='b', mw=18, axes = a2) # ends
    vv.plot(centerline2, ms='.', ls='', mw=8, mc='y', axes = a2)
    vv.plot(renal2, ms='.', ls='', mc='m', mw=18, axes = a2)
    vv.plot(renal2_and_cl_point, ms='.', ls='-', mc='m', mw=18, axes = a2)
    
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a2.axis.axisColor= 1,1,1
    a2.bgcolor= 0,0,0
    a2.daspect= 1, 1, -1  # z-axis flipped
    a2.axis.visible = showAxis





