""" Measure stent migration relative to renals

"""
import sys, os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import show_ctvolume
sys.path.insert(0, os.path.abspath('..'))
from get_anaconda_ringparts import get_model_struts,get_model_rings,add_nodes_edge_to_newmodel 
from stentseg.utils import _utils_GUI
from stentseg.utils.picker import pick3d

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
vesselname = 'LSPEAS_003_MK Smoothed_Wrapped1.0_smart 3_copy_001.stl'

showAxis = True  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
ringpart = True # True; False
nstruts = 8
clim0  = (0,2500)
clim2 = (0,4)
clim3 = -550,500
radius = 0.07
isoTh = 250

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode1, 'stent', 'avgreg')
vol1 = s.vol
s = loadvol(basedir, ptcode, ctcode2, cropname, 'avgreg')
vol2 = s.vol

# load stent model
s2 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
model1 = s2.model
s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
model2 = s2.model

modelmesh = create_mesh(model2, 1.0)  # Param is thickness

# visualize
f = vv.figure(); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
a = vv.subplot(121)
show_ctvolume(vol1, model1, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
pick3d(vv.gca(), vol1)
model1.Draw(mc='b', mw = 10, lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
a.axis.axisColor= 1,1,1
a.bgcolor= 0,0,0
a.daspect= 1, 1, -1  # z-axis flipped
a.axis.visible = showAxis

a = vv.subplot(122)
show_ctvolume(vol2, model2, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
pick3d(vv.gca(), vol2)
model2.Draw(mc='b', mw = 10, lc='g')
vm = vv.mesh(modelmesh)
vm.faceColor = 'g' 
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
a.axis.axisColor= 1,1,1
a.bgcolor= 0,0,0
a.daspect= 1, 1, -1  # z-axis flipped
a.axis.visible = showAxis



## Input by user/read excel: renal coordinates
# coordinates, left or right renal
xrenal, yrenal, zrenal = 171, 165.1, -39.5
rr = xrenal, yrenal, -zrenal
r1 = vv.solidSphere(translation = (rr), scaling = (3.4,3.4,3.4))
r1.faceColor = 'm'
r1.visible = True

## Load vessel mesh (output Mimics)
# can be stl or iges or dxf

def loadvessel(basedirMesh,ptcode,ctcode,vesselname):
    mesh = vv.meshRead(os.path.join(basedirMesh, ptcode, ctcode, vesselname))
    # z is negative, must be flipped to match dicom orientation
    for vertice in mesh._vertices:
        vertice[-1] = vertice[-1]*-1
    return mesh

vessel = loadvessel(basedirMesh,ptcode,ctcode2,vesselname)

# Create centerline (main branch and renals)

m = vv.mesh(vessel) # mesh transparant?
m.faceColor = 'r' 

## Distances
# Calculate distance ring peaks and valleys to renal in Z

# Calculate distance ring peaks and valleys to renal over centerline (main branch)
# project points perpendicular to centerline

# Main outcome 1: distance 2nd ring valleys to renal
# Main outcome 2: migration 2nd ring valleys from discharge to 1, 6, 12 months


