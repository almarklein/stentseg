""" Script to show the stent model static during follow up
Compare models up to 6 months (2 or 3 volumetric images)
Run as script (for import within lspeas folder)
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.visualization import show_ctvolume,DrawModelAxes
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
from lspeas.utils.get_anaconda_ringparts import get_model_struts, get_model_rings
from stentseg.motion.vis import get_graph_in_phase
from stentseg.utils import _utils_GUI
from stentseg.utils.picker import pick3d

# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf', 
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_020'
# codes = ctcode1, ctcode2, ctcode3 = 'discharge', '1month', '6months'
# codes = ctcode1, ctcode2 = 'discharge', '1month'
codes = ctcode1 = 'discharge'
cropname = 'ring'
modelname = 'modelavgreg'
ringnames = ['modelR1'] # ['modelR1', 'modelR2'] or ['model']

showVol  = 'ISO'  # MIP or ISO or 2D or None
phases =  range(10) # range(10) for all 10 phases; [3,9] for 30% and 90%
showmodelavgreg = True # show also model in avgreg at mid cycle?
showvol = True
removeStent = True; stripSizeZ = 500
meshWithColors = False
ringpart = False # R1=1 ; R2=2 ; False = complete model

# vol1 = loadvol(basedir, ptcode, ctcode1, cropname, 'phases').vol30
# vol1 = loadvol(basedir, ptcode, ctcode1, cropname, 'phases').vol90

def loadvolmodel(basedir, ptcode, ctcode1, cropname, modelname, nstruts=8, ringpart=False):
    """ load vol and model
    """
    s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
    model = s1.model
    models1 = []
    for ringname in ringnames:
        model1 = s1[ringname]
        if ringpart:
            models = get_model_struts(model1, nstruts = nstruts)
            modelRs = get_model_rings(models[2]) # model_R1R2
            model1 = modelRs[ringpart-1] # R1 or R2
        models1.append(model1) 
    vol1 = loadvol(basedir, ptcode, ctcode1, cropname, 'avgreg').vol
    
    return vol1, models1, model

# Load the stent model, create mesh, load CT image for reference
# 1 model 
vol1, model1, modelori1 = loadvolmodel(basedir, ptcode, ctcode1, cropname, modelname, ringpart=ringpart)

# 2 models
if len(codes) == 2 or len(codes) == 3 or len(codes) == 4:
    vol2, model2, modelori2 = loadvolmodel(basedir, ptcode, ctcode2, cropname, modelname, ringpart=ringpart)

# 3 models
if len(codes) == 3 or len(codes) == 4:
    vol3, model3, modelori3 = loadvolmodel(basedir, ptcode, ctcode3, cropname, modelname, ringpart=ringpart)

# 4 models
if len(codes) == 4:
    vol4, model4, modelori4 = loadvolmodel(basedir, ptcode, ctcode4, cropname, modelname, ringpart=ringpart)



## Visualize
f = vv.figure(); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
colors = 'cgmrcgywmb'  # r op 30%, b op 90%
clim0  = (0,2500)
# clim0 = -550,500
clim2 = (0,1.5)
radius = 0.07
dimensions = 'xyz'
isoTh = 250

def drawmodelphasescycles(vol1, model1, modelori1, showVol, isoTh=300, removeStent=False, 
        showmodelavgreg=False, showvol=True, phases=range(10), colors='cgmrcgywmb',
        meshWithColors=False, stripSizeZ=None, ax=None):
    """ draw model and volume (show optional) at different phases cycle
    """
    if ax is None:
        ax = vv.gca()
    ax.daspect = 1,1,-1
    ax.axis.axisColor = 0,0,0
    ax.bgcolor = 1,1,1
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)') 
    # draw
    t = show_ctvolume(vol1, modelori1, showVol=showVol, removeStent=removeStent, 
                        climEditor=True, isoTh=isoTh, clim=clim0, stripSizeZ=stripSizeZ)
    if showmodelavgreg:
        # show model and CT mid cycle
        mw = 5
        for model in model1:
            model.Draw(mc='b', mw = mw, lc='b', alpha = 0.5)
        label = pick3d(ax, vol1)
    if not showvol:
        t.visible = False
    # get models in different phases
    for model in model1:
        for phasenr in phases:
            model_phase = get_graph_in_phase(model, phasenr = phasenr)
            if meshWithColors:
                modelmesh1 = create_mesh_with_abs_displacement(model_phase, radius = radius, dim = dimensions)
                m = vv.mesh(modelmesh1, colormap = vv.CM_JET, clim = clim2)
                #todo: use colormap Viridis or Magma as JET is not linear (https://bids.github.io/colormap/)
            else:
                model_phase.Draw(mc=colors[phasenr], mw = 10, lc=colors[phasenr])
        #         modelmesh1 = create_mesh(model_phase, radius = radius)
        #         m = vv.mesh(modelmesh1); m.faceColor = colors[phasenr]
    if meshWithColors:
        vv.colorbar()
    return ax
    

axes = []
# 1 model
if codes==ctcode1:
    ax = drawmodelphasescycles(vol1, model1, modelori1, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors, stripSizeZ=stripSizeZ)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    axes.append(ax)

# 2 models
if len(codes) == 2:
    a1 = vv.subplot(121)
    ax = drawmodelphasescycles(vol1, model1, modelori1, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    axes.append(ax)
    
    a2 = vv.subplot(122)
    ax = drawmodelphasescycles(vol2, model2, modelori2, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    axes.append(ax)
    
# 3 models
if len(codes) == 3:
    a1 = vv.subplot(131)
    ax = drawmodelphasescycles(vol1, model1, modelori1, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    axes.append(ax)
    
    a2 = vv.subplot(132)
    ax = drawmodelphasescycles(vol2, model2, modelori2, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    axes.append(ax)
    
    a3 = vv.subplot(133)
    ax = drawmodelphasescycles(vol3, model3, modelori3, showVol, isoTh=isoTh, removeStent=removeStent, 
            showmodelavgreg=showmodelavgreg, showvol=showvol, phases=phases, colors=colors,
            meshWithColors=meshWithColors)
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode3))
    axes.append(ax)

#bind view control
f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, axes, axishandling=False) )
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, axes) )


## Set view
# a1.SetView(view1)
# a2.SetView(view2)
# a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera
if False:
    a = vv.gca()
    a.camera = ax.camera

## Save figure
if False:
    vv.screenshot(r'D:\Profiles\koenradesma\Desktop\ZA3_phase90.jpg', vv.gcf(), sf=2) #phantom validation manuscript

