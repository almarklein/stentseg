""" Script to show the stent model static during follow up
Compare models up to 6 months (2 or 3 volumetric images)
Run as script (for import within lspeas folder)
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import stentgraph
from stentseg.motion.vis import create_mesh_with_abs_displacement
from get_anaconda_ringparts import get_model_rings
from stentseg.motion.vis import remove_stent_from_volume
import pirt
import numpy as np

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_BACKUP\LSPEAS_ssdf')

# Select dataset to register
ptcode = 'LSPEAS_001'
codes = ctcode1, ctcode2, ctcode3, ctcode4 = 'discharge', '1month', '6months', '12months'
# codes = ctcode1, ctcode2, ctcode3 = 'discharge', '1month', '6months'
# codes = ctcode1, ctcode2 = 'discharge', '1month'
# codes = ctcode1 = 'discharge'
cropname = 'ring'
modelname = 'modelavgreg'

showAxis = False  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
ringpart = None # top=1 ; 2nd=2 ; None = both rings

# view1 = 
#  
# view2 = 
#  
# view3 = 


# Load the stent model, create mesh, load CT image for reference
# 1 model 
s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
model1 = s1.model
if ringpart:
    models = get_model_rings(model1)
    model1 = models[ringpart-1]
vol1 = loadvol(basedir, ptcode, ctcode1, cropname, 'avgreg').vol

# 2 models
if len(codes) == 2 or len(codes) == 3 or len(codes) == 4:
    s2 = loadmodel(basedir, ptcode, ctcode2, cropname, modelname)
    model2 = s2.model
    if ringpart:
        models = get_model_rings(model2)
        model2 = models[ringpart-1]
    vol2 = loadvol(basedir, ptcode, ctcode2, cropname, 'avgreg').vol

# 3 models
if len(codes) == 3 or len(codes) == 4:
    s3 = loadmodel(basedir, ptcode, ctcode3, cropname, modelname)
    model3 = s3.model
    if ringpart:
        models = get_model_rings(model3)
        model3 = models[ringpart-1]
    vol3 = loadvol(basedir, ptcode, ctcode3, cropname, 'avgreg').vol

# 4 models
if len(codes) == 4:
    s4 = loadmodel(basedir, ptcode, ctcode4, cropname, modelname)
    model4 = s4.model
    if ringpart:
        models = get_model_rings(model4)
        model4 = models[ringpart-1]
    vol4 = loadvol(basedir, ptcode, ctcode4, cropname, 'avgreg').vol


def get_graph_in_phase(graph, phasenr):
    """ Get position of model in a certain phase
    """
    # initialize
    model_phase = stentgraph.StentGraph()
    for n1, n2 in graph.edges():
        # obtain path and deforms of nodes and edge
        path = graph.edge[n1][n2]['path']
        pathDeforms = graph.edge[n1][n2]['pathdeforms']
        n1_phase = tuple((n1 + pathDeforms[0][phasenr]).flat)
        n2_phase = tuple((n2 + pathDeforms[-1][phasenr]).flat)
        # obtain path in phase
        path_phase = []
        for i, point in enumerate(path):
            pointposition = point + pathDeforms[i][phasenr]
            path_phase.append(pointposition) # points on path, one phase
        model_phase.add_edge(n1_phase, n2_phase, path = np.asarray(path_phase), pathdeforms = np.asarray(pathDeforms))
        #todo: too many nodes are added to the graph
#     g.add_node(node, deforms=deforms)
#     sd._nodes3.add_edge(select1,select2, cost = c, ctvalue = ct, path = p)
    return model_phase

def show_ctvolume(vol, model, isoTh=250):
    if showVol == 'MIP':
        t = vv.volshow(vol, clim=clim, renderStyle='mip')
    elif showVol == 'ISO':
        vol = remove_stent_from_volume(vol, model, stripSize=7) # rings are removed for vis.
        t = vv.volshow(vol,clim=clim, renderStyle='iso')
        t.isoThreshold = isoTh; t.colormap = colormap
    elif showVol == '2D':
        t = vv.volshow2(vol); t.clim = clim3

## Visualize
f = vv.figure(1); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
color = 'rgbmcrywgb'
clim  = (0,2500)
clim2 = (0,4)
clim3 = -550,500
radius = 0.07
dimensions = 'xyz'
isoTh = 240
colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}

# 1 model
if codes=='discharge' or codes=='1month' or codes=='6months' or codes=='12months':
    a = vv.gca()
    show_ctvolume(vol1, model1, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model1, phasenr = phasenr)
#         model_phase.Draw(mc=color[phasenr], mw = 10, lc=color[phasenr])
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
#         modelmesh1 = create_mesh(model_phase, radius = radius)
#         m = vv.mesh(modelmesh1); m.faceColor = color[phasenr]
        modelmesh1 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh1, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar()    
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a.axis.axisColor= 1,1,1
    a.bgcolor= 0,0,0
    a.daspect= 1, 1, -1  # z-axis flipped
    a.axis.visible = showAxis

# 2 models
if len(codes) == 2:
    a1 = vv.subplot(121)
    show_ctvolume(vol1, model1, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model1, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh1 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh1, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a2 = vv.subplot(122)
    show_ctvolume(vol2, model2, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model2, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh2 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh2, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a1.axis.axisColor= a2.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor = 0,0,0
    a1.daspect= a2.daspect = 1, 1, -1  # z-axis flipped
    a1.axis.visible= a2.axis.visible = showAxis
    
# 3 models
if len(codes) == 3:
    a1 = vv.subplot(131)
    show_ctvolume(vol1, model1, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model1, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh1 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh1, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a2 = vv.subplot(132)
    show_ctvolume(vol2, model2, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model2, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh2 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh2, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a3 = vv.subplot(133)
    show_ctvolume(vol3, model3, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model3, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh3 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh3, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode3))
    a1.axis.axisColor= a2.axis.axisColor= a3.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor= a3.bgcolor = 0,0,0
    a1.daspect= a2.daspect= a3.daspect = 1, 1, -1  # z-axis flipped
    a1.axis.visible= a2.axis.visible= a3.axis.visible = showAxis

# 4 models
if len(codes) == 4:
    a1 = vv.subplot(141)
    show_ctvolume(vol1, model1, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model1, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 3, lc=color[phasenr])
        modelmesh1 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh1, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode1))
    a2 = vv.subplot(142)
    show_ctvolume(vol2, model2, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model2, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh2 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh2, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode2))
    a3 = vv.subplot(143)
    show_ctvolume(vol3, model3, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model3, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh3 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh3, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode3))
    a4 = vv.subplot(144)
    show_ctvolume(vol4, model4, isoTh = isoTh)
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model4, phasenr = phasenr)
#         model_phase.Draw(mc='', lw = 6, lc=color[phasenr])
        modelmesh4 = create_mesh_with_abs_displacement(model_phase, radius = radius, dimensions = dimensions)
        m = vv.mesh(modelmesh4, colormap = vv.CM_JET, clim = clim2)
    vv.colorbar() 
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode4))
    a1.axis.axisColor= a2.axis.axisColor= a3.axis.axisColor= a4.axis.axisColor = 1,1,1
    a1.bgcolor= a2.bgcolor= a3.bgcolor= a4.bgcolor = 0,0,0
    a1.daspect= a2.daspect= a3.daspect= a4.daspect = 1, 1, -1  # z-axis flipped
    a1.axis.visible= a2.axis.visible= a3.axis.visible =a4.axis.visible = showAxis

## Axis on or off

# showAxis = True
if len(codes) == 1:
    a.axis.visible = showAxis
if len(codes) == 2:
    a1.axis.visible= a2.axis.visible = showAxis
if len(codes) == 3:
    a1.axis.visible= a2.axis.visible= a3.axis.visible = showAxis
if len(codes) == 4:
    a1.axis.visible= a2.axis.visible= a3.axis.visible= a4.axis.visible= showAxis

## Set view
# a1.SetView(view1)
# a2.SetView(view2)
# a3.SetView(view3)

## Use same camera
#a1.camera = a2.camera = a3.camera


