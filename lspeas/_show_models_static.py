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
from lspeas.utils.vis import showModelsStatic

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_002'
# codes = ctcode1, ctcode2, ctcode3 = '6months', '12months', '24months'
# codes = ctcode1, ctcode2 = '12months', '24months'
# codes = ctcode1 = '12months'
codes = ctcode1, ctcode2, ctcode3, ctcode4 = 'discharge', '6months', '12months', '24months'
cropname = 'ring'
modelname = 'modelavgreg'
cropvol = 'stent'

drawModelLines = False  # True or False
drawMesh = True
showAxis = False
dimensions = 'xyz'
showVol  = 'ISO'  # MIP or ISO or 2D or None
showvol2D = False

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
vols = [vol1]
ss = [s1]
mm = [modelmesh1]

# 2 models
if len(codes) == 2 or len(codes) == 3 or len(codes) == 4:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
#     modelmesh2 = create_mesh(s2.model, 1.0)  # Param is thickness
    modelmesh2 = create_mesh_with_abs_displacement(s2.model, radius = 0.7, dim=dimensions)
    vol2 = loadvol(basedir, ptcode, ctcode2, cropvol, 'avgreg').vol
    vols = [vol1, vol2]
    ss = [s1,s2]
    mm = [modelmesh1, modelmesh2]

# 3 models
if len(codes) == 3 or len(codes) == 4:
    s3 = loadmodel(basedir, ptcode, ctcode3, cropname, modelname)
#     modelmesh3 = create_mesh(s3.model, 1.0)  # Param is thickness   
    modelmesh3 = create_mesh_with_abs_displacement(s3.model, radius = 0.7, dim=dimensions)
    vol3 = loadvol(basedir, ptcode, ctcode3, cropvol, 'avgreg').vol
    vols = [vol1, vol2, vol3]
    ss = [s1,s2,s3]
    mm = [modelmesh1, modelmesh2, modelmesh3]

# 4 models
if len(codes) == 4:
    s4 = loadmodel(basedir, ptcode, ctcode4, cropname, modelname)
#     modelmesh4 = create_mesh(s4.model, 1.0)  # Param is thickness   
    modelmesh4 = create_mesh_with_abs_displacement(s4.model, radius = 0.7, dim=dimensions)
    vol4 = loadvol(basedir, ptcode, ctcode4, cropvol, 'avgreg').vol
    vols = [vol1, vol2, vol3, vol4]
    ss = [s1,s2,s3,s4]
    mm = [modelmesh1, modelmesh2, modelmesh3, modelmesh4]

## Visualize multipanel
axes, cbars = showModelsStatic(ptcode, codes, vols, ss, mm, showVol, clim, isoTh, 
        clim2, clim2D, drawMesh, drawModelLines, showvol2D, showAxis)

# bind rotate view
f = vv.gcf()
f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, axes) )
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, axes) )

## Set view
# a1.SetView(view1)
# a2.SetView(view2)
# a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera

## Set colorbar position
# for cbar in cbars:
#     p1 = cbar.position
#     cbar.position = (p1[0], 20, p1[2], 0.98) # x,y,w,h
