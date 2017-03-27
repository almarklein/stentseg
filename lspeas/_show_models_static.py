""" Script to show the stent model static during follow up
Compare models up to 6 months (2 or 3 volumetric images)

"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
import pirt
import numpy as np
from stentseg.utils import _utils_GUI
from stentseg.utils.picker import pick3d

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_021'
# codes = ctcode1, ctcode2, ctcode3 = '1month', '6months', '12months'
codes = ctcode1, ctcode2 = 'discharge', '12months'
# codes = ctcode1 = '12months'
cropname = 'ring'
modelname = 'modelavgreg'
cropvol = 'ring'

drawModelLines = False  # True or False
drawMesh = False
showAxis = False
dimensions = 'xyz'
showVol  = 'ISO'  # MIP or ISO or 2D or None
showvol2D = True

clim = (0,2500)
clim2D = -200,500
clim2 = (0,1.5)
isoTh = 250

# view1 = 
#  
# view2 = 
#  
# view3 = 


# Load the stent model, create mesh, load CT image for reference
# 1 model 
s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
# modelmesh1 = create_mesh(s1.model, 1.0)  # Param is thickness
modelmesh1 = create_mesh_with_abs_displacement(s1.model, radius = 0.7, dim=dimensions)
vol1 = loadvol(basedir, ptcode, ctcode1, cropvol, 'avgreg').vol

# 2 models
if len(codes) == 2 or len(codes) == 3:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
#     modelmesh2 = create_mesh(s2.model, 1.0)  # Param is thickness
    modelmesh2 = create_mesh_with_abs_displacement(s2.model, radius = 0.7, dim=dimensions)
    vol2 = loadvol(basedir, ptcode, ctcode2, cropvol, 'avgreg').vol

# 3 models
if len(codes) == 3:
    s3 = loadmodel(basedir, ptcode, ctcode3, cropname, modelname)
#     modelmesh3 = create_mesh(s3.model, 1.0)  # Param is thickness   
    modelmesh3 = create_mesh_with_abs_displacement(s3.model, radius = 0.7, dim=dimensions)
    vol3 = loadvol(basedir, ptcode, ctcode3, cropvol, 'avgreg').vol


## Visualize
f = vv.figure(1); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00

if drawMesh == True:
    lc = 'w'
    mw = 10
else:
    lc = 'g'
    mw = 7

# 1 model
if codes==ctcode1 :
    a = vv.subplot(121)
    t = show_ctvolume(vol1, s1.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol1)
    if showvol2D:
        t2 = vv.volshow2(vol1, clim=clim2D)
    if drawModelLines == True:
        s1.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh1)
        # m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a.axis.axisColor= 1,1,1
    a.bgcolor= 0,0,0
    a.daspect= 1, 1, -1  # z-axis flipped
    a.axis.visible = showAxis
    a2 = vv.subplot(122)
    a2.bgcolor= 0,0,0
    a2.daspect= 1, 1, -1
    a2.axis.visible = False

# 2 models
if len(codes) == 2:
    a1 = vv.subplot(121)
    t = show_ctvolume(vol1, s1.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol1)
    if drawModelLines == True:
        s1.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh1)
        # m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a2 = vv.subplot(122)
    t = show_ctvolume(vol2, s2.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol2)
    if drawModelLines == True:
        s2.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh2)
        #m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    if showvol2D:
        t2 = vv.volshow2(vol1, clim=clim2D, axes=a1)
        t2 = vv.volshow2(vol2, clim=clim2D, axes=a2 )
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a1.axis.axisColor= a2.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor = 0,0,0
    a1.daspect= a2.daspect = 1, 1, -1  # z-axis flipped
    a1.axis.visible= a2.axis.visible = showAxis
    
# 3 models
if len(codes) == 3:
    a1 = vv.subplot(131)
    t = show_ctvolume(vol1, s1.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol1)
    if drawModelLines == True:
        s1.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh1)
        #m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a2 = vv.subplot(132)
    t = show_ctvolume(vol2, s2.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol2)
    if drawModelLines == True:
        s2.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh2)
        #m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a3 = vv.subplot(133)
    t = show_ctvolume(vol3, s3.model, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
    label = pick3d(vv.gca(), vol3)
    if drawModelLines == True:
        s3.model.Draw(mc='b', mw = mw, lc=lc)
    if drawMesh == True:
        m = vv.mesh(modelmesh3)
        #m.faceColor = 'g' # OR
        m.clim = clim2
        m.colormap = vv.CM_JET
        vv.colorbar()
    if showvol2D:
        t2 = vv.volshow2(vol1, clim=clim2D, axes=a1)
        t2 = vv.volshow2(vol2, clim=clim2D, axes=a2)
        t2 = vv.volshow2(vol3, clim=clim2D, axes=a3)
        
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode3))
    a1.axis.axisColor= a2.axis.axisColor= a3.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor= a3.bgcolor = 0,0,0
    a1.daspect= a2.daspect= a3.daspect = 1, 1, -1  # z-axis flipped
    a1.axis.visible= a2.axis.visible= a3.axis.visible = showAxis


## Axis on or off

# showAxis = False
if len(codes) == 1:
    a.axis.visible = showAxis
    # bind rotate view
    f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a]) )
if len(codes) == 2:
    a1.axis.visible= a2.axis.visible = showAxis
    # bind rotate view
    f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1, a2]) )
if len(codes) == 3:
    a1.axis.visible= a2.axis.visible= a3.axis.visible = showAxis
    # bind rotate view
    f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1, a2, a3]) )

## Set view
# a1.SetView(view1)
# a2.SetView(view2)
# a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera


