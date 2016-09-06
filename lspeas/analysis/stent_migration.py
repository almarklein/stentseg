""" Measure stent migration relative to renals

"""
import sys, os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import show_ctvolume
from stentseg.utils import _utils_GUI, PointSet
from stentseg.utils.picker import pick3d
from stentseg.utils.centerline import find_centerline, points_from_mesh, smooth_centerline
sys.path.insert(0, os.path.abspath('..')) # 2 folders deeper in pythonPath
import utils_analysis
from utils_analysis import point_in_pointcloud_closest_to_p 
import get_anaconda_ringparts
from get_anaconda_ringparts import _get_model_hooks,get_midpoints_peaksvalleys,identify_peaks_valleys

#todo: from outline to script:

## Load (dynamic) stent models
# discharge, 1 month, 6 months, and/or 12 months

# select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup', r'G:\LSPEAS_ssdf_backup')
                     
basedirMesh = select_dir(r'D:\LSPEAS\LSPEAS_vessel',
                            'F:\LSPEAS_vessel_backup')

# select dataset
ptcode = 'LSPEAS_003'
ctcodes = ctcode1, ctcode2 = 'discharge', '12months'
cropname = 'ring'
modelname = 'modelavgreg'
vesselname1 = 'LSPEAS_003_MK Smoothed_Wrapped1.0_edit-smart 4_copy_001.stl'
# LSPEAS_003_MK Smoothed_Wrapped1.0_edit-smart 4_copy_noRenals 7_001
vesselname2 = 'LSPEAS_003_MK Smoothed_Wrapped1.0_smart 3_copy_001.stl'

showAxis = True  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
ringpart = True # True; False
clim0  = (0,2500)
clim3 = -550,500
isoTh = 250
meshradius = 0.7

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode1, cropname, 'avgreg')
vol1 = s.vol
s = loadvol(basedir, ptcode, ctcode2, cropname, 'avgreg')
vol2 = s.vol
  
# load stent model
s2 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
model1 = s2.model
modelmesh1 = create_mesh(model1, meshradius)
s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
model2 = s2.model
modelmesh2 = create_mesh(model2, meshradius)

# Load vessel mesh (output Mimics)
vessel1 = loadmesh(basedirMesh,ptcode,ctcode1,vesselname1) #inverts Z
vessel2 = loadmesh(basedirMesh,ptcode,ctcode2,vesselname2) #inverts Z
# get pointset from STL 
ppvessel1 = points_from_mesh(vessel1, invertZ = False) # removes duplicates
ppvessel2 = points_from_mesh(vessel2, invertZ = False) # removes duplicates

## Renal origin coordinates: input by user/read excel
# coordinates, left or right most caudal renal
# ctcode1
xrenal1, yrenal1, zrenal1 = 132.7, 89.2, 85.5
renal1 = PointSet(list((xrenal1, yrenal1, zrenal1)))
# ctcode2
xrenal2, yrenal2, zrenal2 = 171, 165.1, 39.5
renal2 = PointSet(list((xrenal2, yrenal2, zrenal2)))

## Create centerline
# ctcode1
c1_start1 = (153, 86, 104.5) # distal end
c1_ends = [(142, 94, 64.5)] # either single point or multiple
centerline1 = find_centerline(ppvessel1, c1_start1, c1_ends, 0.5, ndist=20, regfactor=0.2, regsteps=10)
centerline1 = smooth_centerline(centerline1, 20) # 20 iterations for stepsize 0.5 is reasonable
# ctcode2
c2_start1 = (190, 165, 60) # distal end
c2_ends = [(179, 169, 17)] # either single point or multiple
centerline2 = find_centerline(ppvessel2, c2_start1, c2_ends, 0.5, ndist=20, regfactor=0.2, regsteps=10)
centerline2 = smooth_centerline(centerline2, 20)


## Get peak and valley points

# ----OPTION automated detection----
# get midpoints peaks valleys
midpoints_peaks_valleys = get_midpoints_peaksvalleys(model1)
# from peaks valley pointcloud identiy peaks and valleys
R1_left,R2_left,R1_right,R2_right,R1_ant,R2_ant,R1_post,R2_post = identify_peaks_valleys(
midpoints_peaks_valleys, model1, vol1,vis=True)

# ----OPTION excel----






## Calculate distance ring peaks and valleys to renal in Z
#todo get peaks valleys

# point of centerline closest to renal
renal1_and_cl_point = point_in_pointcloud_closest_to_p(centerline1, renal1)
renal2_and_cl_point = point_in_pointcloud_closest_to_p(centerline2, renal2)
# point of centerline closest to peaks valleys
#todo get peaks valleys


# Main outcome 1: distance 2nd ring valleys to renal
# Main outcome 2: migration 2nd ring valleys from discharge to 1, 6, 12 months

## Visualize
f = vv.figure(2); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
alpha = 0.5
a1 = vv.subplot(121)
show_ctvolume(vol1, model1, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
pick3d(vv.gca(), vol1)
model1.Draw(mc='b', mw = 10, lc='g')
vm = vv.mesh(modelmesh1)
vm.faceColor = 'g'
# m = vv.mesh(vessel1)
# m.faceColor = (1,0,0, alpha) # red

# vis centerline and renal origo
vv.plot(ppvessel1, ms='.', ls='', mc= 'r', alpha=0.2, mw = 7, axes = a1) # vessel
vv.plot(PointSet(list(c1_start1)), ms='.', ls='', mc='g', mw=18, axes = a1) # start1
vv.plot([e[0] for e in c1_ends], [e[1] for e in c1_ends],  [e[2] for e in c1_ends],  ms='.', ls='', mc='b', mw=18, axes = a1) # ends
vv.plot(centerline1, ms='.', ls='', mw=8, mc='y', axes = a1)
vv.plot(renal1, ms='.', ls='', mc='m', mw=18, axes = a1)
vv.plot(renal1_and_cl_point, ms='.', ls='-', mc='m', mw=18, axes = a1)

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
a1.axis.axisColor= 1,1,1
a1.bgcolor= 0,0,0
a1.daspect= 1, 1, -1  # z-axis flipped
a1.axis.visible = showAxis

a2 = vv.subplot(122)
show_ctvolume(vol2, model2, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
pick3d(vv.gca(), vol2)
model2.Draw(mc='b', mw = 10, lc='g')
vm = vv.mesh(modelmesh2)
vm.faceColor = 'g'
# m = vv.mesh(vessel2)
# m.faceColor = (1,0,0, alpha) # red

# vis centerline and renal origo
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





