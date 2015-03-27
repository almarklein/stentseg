""" Script to show the stent model static during follow up
Compare models up to 6 months (2 or 3 volumetric images)

"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
import pirt
import numpy as np

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_001'
# codes = ctcode1, ctcode2, ctcode3 = 'discharge', '1month', '6months'
codes = ctcode1, ctcode2 = 'discharge', '1month'
cropname = 'ring'
modelname = 'modelavgreg'

drawModelLines = False  # True or False

view1 = {'loc': (145.26484406424328, 87.22041481859763, 76.50081218689655),
 'roll': 0.0,
 'elevation': 14.8159509202454,
 'zoom': 0.02233385266889955,
 'azimuth': 23.749999999999993,
 'fov': 0.0,
 'daspect': (1.0, 1.0, -1.0)}
 
view2 = {'loc': (139.1060084543532, 107.17217544666808, 72.92421103383431),
 'roll': 0.0,
 'elevation': 12.883435582822086,
 'zoom': 0.021213504980157494,
 'azimuth': 27.903846153846157,
 'fov': 0.0,
 'daspect': (1.0, 1.0, -1.0)}
 
view3 = {'loc': (156.88667588201432, 135.540898262675, 67.17765959975242),
 'roll': 0.0,
 'elevation': 13.159509202453988,
 'zoom': 0.0198068585220672,
 'azimuth': 29.11538461538462,
 'fov': 0.0,
 'daspect': (1.0, 1.0, -1.0)}


# Load the stent model and mesh
s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
if len(codes) == 2:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
if len(codes) == 3:
    s3 = loadmodel(basedir, ptcode, ctcode3, cropname, modelname)

# Create mesh
# modelmesh1 = create_mesh(s1.model, 1.0)  # Param is thickness
# if len(codes) == 2:    
#     modelmesh2 = create_mesh(s2.model, 1.0)  # Param is thickness
# if len(codes) == 3:    
#     modelmesh3 = create_mesh(s3.model, 1.0)  # Param is thickness
modelmesh1 = create_mesh_with_abs_displacement(s1.model, radius = 1.0, dimensions = 'xyz')
if len(codes) == 2:
    modelmesh2 = create_mesh_with_abs_displacement(s2.model, radius = 1.0, dimensions = 'xyz')
if len(codes) == 3:
    modelmesh3 = create_mesh_with_abs_displacement(s3.model, radius = 1.0, dimensions = 'xyz')

# Load static CT image to add as reference
vol1 = loadvol(basedir, ptcode, ctcode1, cropname, 'avgreg').vol
if len(codes) == 2:
    vol2 = loadvol(basedir, ptcode, ctcode2, cropname, 'avgreg').vol
if len(codes) == 3:
    vol3 = loadvol(basedir, ptcode, ctcode3, cropname, 'avgreg').vol


## Visualize
f = vv.figure(1); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00

# model 1
if len(codes) == 2:
    a1 = vv.subplot(121)
elif len(codes) == 3:
    a1 = vv.subplot(131)
t = vv.volshow(vol1, clim=(0, 2500), renderStyle='mip')
if drawModelLines == True:
    s1.model.Draw(mc='b', mw = 10, lc='w')
m = vv.mesh(modelmesh1)
# m.faceColor = 'g' # OR
m.clim = 0, 5
m.colormap = vv.CM_JET
vv.colorbar()
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))

# model 2
if len(codes) == 2 or 3:
    if len(codes) == 2:
        a2 = vv.subplot(122)
    elif len(codes) == 3:
        a2 = vv.subplot(132)
    t = vv.volshow(vol2, clim=(0, 2500), renderStyle='mip')
    if drawModelLines == True:
        s2.model.Draw(mc='b', mw = 10, lc='w')
    m = vv.mesh(modelmesh2)
    #m.faceColor = 'g' # OR
    m.clim = 0, 5
    m.colormap = vv.CM_JET
    vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))

# model 3
if len(codes) == 3:
    a3 = vv.subplot(133)
    t = vv.volshow(vol3, clim=(0, 2500), renderStyle='mip')
    if drawModelLines == True:
        s3.model.Draw(mc='b', mw = 10, lc='w')
    m = vv.mesh(modelmesh3)
    # m.faceColor = 'g' # OR
    m.clim = 0, 5
    m.colormap = vv.CM_JET
    vv.colorbar()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode3))

# vv.ColormapEditor(vv.gcf())
# t = vv.volshow2(vol)
# t.clim = -500, 500

if len(codes) == 2:
    a1.axis.axisColor= a2.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor = 0,0,0
    a1.daspect= a2.daspect = 1, 1, -1  # z-axis flipped
if len(codes) == 3:
    a1.axis.axisColor= a2.axis.axisColor= a3.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor= a3.bgcolor = 0,0,0
    a1.daspect= a2.daspect= a3.daspect = 1, 1, -1  # z-axis flipped

# Axis on or off
if len(codes) == 2:
    a1.axis.visible= a2.axis.visible = False
if len(codes) == 3:
    a1.axis.visible= a2.axis.visible= a3.axis.visible = False

## Set view
a1.SetView(view1)
a2.SetView(view2)
a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera


