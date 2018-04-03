""" Script to show the stent model static during follow up
Compare models up to 6 months (2 or 3 volumetric images)
(used to create figs in paper ring deployment)
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmesh
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
import numpy as np
from stentseg.utils import _utils_GUI
from stentseg.utils.picker import pick3d
from lspeas.utils.vis import showModelsStatic

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')
                     
basedirMesh = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics',
    r'C:\Users\Maaike\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics')

# Select dataset to register
ptcode = 'LSPEAS_022'
codes = ctcode1, ctcode2, ctcode3 = 'discharge', '12months', '12months'
# codes = ctcode1, ctcode2 = 'discharge', '24months'
# codes = ctcode1 = '12months'
# codes = ctcode1, ctcode2, ctcode3, ctcode4 = 'discharge', '1month', '6months', '12months'
# codes = ctcode1, ctcode2, ctcode3, ctcode4, ctcode5 = 'discharge', '1month', '6months', '12months', '24months'
cropname = 'ring'
modelname = 'modelavgreg'
cropvol = 'stent'

drawModelLines = False  # True or False
meshDisplacement = True
drawMesh = True
dimensions = 'xyz'
showAxis = False
showVol  = 'ISO'  # MIP or ISO or 2D or None
removeStent = True
showvol2D = False
drawVessel = False

clim = (0,2500)
clim2D = -200,500
clim2 = (0,2)
isoTh = 250 # 250

# view1 = 
#  
# view2 = 
#  
# view3 = 

mm = []
vs = []
# Load the stent model, create mesh, load CT image for reference
# 1 model 
s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
vol1 = loadvol(basedir, ptcode, ctcode1, cropvol, 'avgreg').vol
vols = [vol1]
ss = [s1]
if drawMesh: # stentmodel
    if not meshDisplacement:
        modelmesh1 = create_mesh(s1.model, 0.7)  # Param is thickness
    else:
        modelmesh1 = create_mesh_with_abs_displacement(s1.model, radius = 0.7, dim=dimensions)
    mm = [modelmesh1]
# load vesselmesh
if drawVessel:
    try:
        vessel1 = loadmesh(basedirMesh,ptcode[-3:],'{}_{}_neck.stl'.format(ptcode,ctcode1)) #inverts Z
    except OSError:
        print('vessel mesh does not exist')
        vessel1 = None
    vs = [vessel1]

# 2 models
if len(codes) == 2 or len(codes) == 3 or len(codes) == 4 or len(codes) == 5:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
    vol2 = loadvol(basedir, ptcode, ctcode2, cropvol, 'avgreg').vol
    vols = [vol1, vol2]
    ss = [s1,s2]
    if drawMesh:
        if not meshDisplacement:
            modelmesh2 = create_mesh(s2.model, 0.7)  # Param is thickness
        else:
            modelmesh2 = create_mesh_with_abs_displacement(s2.model, radius = 0.7, dim=dimensions)
        mm = [modelmesh1, modelmesh2]
    # load vesselmesh
    if drawVessel:
        try:
            vessel2 = loadmesh(basedirMesh,ptcode[-3:],'{}_{}_neck.stl'.format(ptcode,ctcode2)) #inverts Z
        except OSError:
            print('vessel mesh does not exist')
            vessel2 = None
        vs = [vessel1, vessel2]
    
# 3 models
if len(codes) == 3 or len(codes) == 4 or len(codes) == 5:
    s3 = loadmodel(basedir, ptcode, ctcode3, cropname, modelname)
    vol3 = loadvol(basedir, ptcode, ctcode3, cropvol, 'avgreg').vol
    vols = [vol1, vol2, vol3]
    ss = [s1,s2,s3]
    if drawMesh:
        if not meshDisplacement:
            modelmesh3 = create_mesh(s3.model, 0.7)  # Param is thickness   
        else:
            modelmesh3 = create_mesh_with_abs_displacement(s3.model, radius = 0.7, dim=dimensions)
        mm = [modelmesh1, modelmesh2, modelmesh3]
    # load vesselmesh
    if drawVessel:
        try:
            vessel3 = loadmesh(basedirMesh,ptcode[-3:],'{}_{}_neck.stl'.format(ptcode,ctcode3)) #inverts Z
        except OSError:
            print('vessel mesh does not exist')
            vessel3 = None
        vs = [vessel1, vessel2, vessel3]
    
# 4 models
if len(codes) == 4 or len(codes) == 5:
    s4 = loadmodel(basedir, ptcode, ctcode4, cropname, modelname)
    vol4 = loadvol(basedir, ptcode, ctcode4, cropvol, 'avgreg').vol
    vols = [vol1, vol2, vol3, vol4]
    ss = [s1,s2,s3,s4]
    if drawMesh:
        if not meshDisplacement:
            modelmesh4 = create_mesh(s4.model, 0.7)  # Param is thickness   
        else:
            modelmesh4 = create_mesh_with_abs_displacement(s4.model, radius = 0.7, dim=dimensions)
        mm = [modelmesh1, modelmesh2, modelmesh3, modelmesh4]
    # load vesselmesh
    if drawVessel:
        try:
            vessel4 = loadmesh(basedirMesh,ptcode[-3:],'{}_{}_neck.stl'.format(ptcode,ctcode4)) #inverts Z
        except OSError:
            print('vessel mesh does not exist')
            vessel4 = None
        vs = [vessel1, vessel2, vessel3, vessel4]

# 5 models
if len(codes) == 5:
    s5 = loadmodel(basedir, ptcode, ctcode5, cropname, modelname)
    vol5 = loadvol(basedir, ptcode, ctcode5, cropvol, 'avgreg').vol
    vols = [vol1, vol2, vol3, vol4, vol5]
    ss = [s1,s2,s3,s4, s5]
    if drawMesh:
        if not meshDisplacement:
            modelmesh5 = create_mesh(s5.model, 0.7)  # Param is thickness   
        else:
            modelmesh5 = create_mesh_with_abs_displacement(s5.model, radius = 0.7, dim=dimensions)
        mm = [modelmesh1, modelmesh2, modelmesh3, modelmesh4, modelmesh5]
    # load vesselmesh
    if drawVessel:
        try:
            vessel5 = loadmesh(basedirMesh,ptcode[-3:],'{}_{}_neck.stl'.format(ptcode,ctcode5)) #inverts Z
        except OSError:
            print('vessel mesh does not exist')
            vessel5 = None
        vs = [vessel1, vessel2, vessel3, vessel4, vessel5]

## Visualize multipanel
axes, cbars = showModelsStatic(ptcode, codes, vols, ss, mm, vs, showVol, clim, 
        isoTh, clim2, clim2D, drawMesh, meshDisplacement, drawModelLines, 
        showvol2D, showAxis, drawVessel, climEditor=False, removeStent=removeStent,
        meshColor=(0,1,0,1))

# bind rotate view
f = vv.gcf()
f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, axes) )
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, axes) )

## option to adjust clim/isoth
slider = True
sliders = dict()
if slider:
    print('Use "s" to show/hide slider')
    for i, ax in enumerate(axes):
        if showVol=='MIP':
            sliders['c'+str(i)] = vv.ClimEditor(ax)
            sliders['c'+str(i)].position = (10, 50)
            f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, sliders) )
        if showVol=='ISO':
            sliders['c'+str(i)] = _utils_GUI.IsoThEditor(ax)
            sliders['c'+str(i)].position = (10, 50)
            f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, sliders) )
            # manual sliders[c0].visible = False

## Set view
# a1.SetView(view1)
# a2.SetView(view2)
# a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera

## Save figure
# vv.screenshot(r'C:\Users\Maaike\Desktop\003_vessel_D_24M_LR.jpg', vv.gcf(), sf=2)

## Set colorbar position
# for cbar in cbars:
#     p1 = cbar.position
#     cbar.position = (p1[0], 20, p1[2], 0.98) # x,y,w,h
