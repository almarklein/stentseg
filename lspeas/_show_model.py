"""
Script to show the stent model in motion.
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import stentgraph
from stentseg.motion.vis import create_mesh_with_abs_displacement, create_mesh_with_values
import pirt
import numpy as np
from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion
from stentseg.motion.displacement import calculateMeanAmplitude
from lspeas.utils.ecgslider import runEcgSlider
from stentseg.utils import _utils_GUI
from stentseg.apps.record_movie import recordMovie

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_BACKUP',r'G:\LSPEAS_ssdf_BACKUP')

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode, nr = '12months', 1
# ptcode = 'QRM_FANTOOM_20160121'
# ctcode, nr = 'ZA3-75-1.2', 1
cropname = 'ring'
modelname = 'modelavgreg'
motion = 'amplitude'  # amplitude or sum
dimension = 'xyz'
showVol  = 'MIP'  # MIP or ISO or 2D or None
clim0  = (-10,2500) 
clim2 = (0,1.5) # mm
clim3 = (0,0.1) # cm-1
isoTh = 250
motionPlay = 9, 1  # each x ms, a step of perc of T
staticref =  'avgreg'# 'avg7020'
meshWithColors = 'displacement' # False or displacement or curvature
ringnames = ['modelR1', 'modelR2'] # ['model'] or ['modelR1'] or ['modelR1', 'modelR2']  

# Load deformations (forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
# deforms = [s['deform%i'%(i*10)] for i in range(10)]
deformkeys = []
for key in dir(s):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s[key] for key in deformkeys]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# These deforms are forward mapping. Turn into DeformationFields.
# Also get the backwards mapping variants (i.e. the inverse deforms).
# The forward mapping deforms should be used to deform meshes (since
# the information is used to displace vertices). The backward mapping
# deforms should be used to deform textures (since they are used in
# interpolating the texture data).
deforms_f = [pirt.DeformationFieldForward(*f) for f in deforms]
deforms_b = [f.as_backward() for f in deforms_f]

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
if len(ringnames)==1: # show entire model or 1 ring
    model = s[ringnames[0]]
else:
    # merge ring models into one graph for dynamic visualization
    model = stentgraph.StentGraph()
    for key in ringnames:
        model.add_nodes_from(s[key].nodes(data=True)) # also attributes
        model.add_edges_from(s[key].edges(data=True))

if meshWithColors=='displacement':
    modelmesh = create_mesh_with_abs_displacement(model, radius = 0.7, dim = dimension, motion = motion)
elif meshWithColors=='curvature':
    modelmesh = create_mesh_with_values(model, valueskey='path_curvature_change', radius=0.7)
else:
    modelmesh = create_mesh(model, 1.0)  # Param is thickness

# Load static CT image to add as reference
try:
    s2 = loadvol(basedir, ptcode, ctcode, 'stent', staticref)
except FileNotFoundError:
    s2 = loadvol(basedir, ptcode, ctcode, 'ring', staticref)
vol = s2.vol


## Start vis
f = vv.figure(nr); vv.clf()
if nr == 1:
    f.position = 8.00, 30.00,  1216.00, 960.00
else:
    f.position = 968.00, 30.00,  1216.00, 960.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
t = show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
if meshWithColors=='displacement':
    vv.title('Dynamic model for patient %s at %s  (colorbar \b{%s} of motion in mm in %s)' % (ptcode[7:], ctcode, motion, dimension))
elif meshWithColors=='curvature':
    vv.title('Dynamic model for patient %s at %s  (colorbar \b{%s} in cm^{-1})' % (ptcode[7:], ctcode, 'curvature change'))
else:
    vv.title('Dynamic model for patient %s at %s ' % (ptcode[7:], ctcode))
    
# viewringcrop = {'daspect': (1.0, 1.0, -1.0), 'azimuth': 32.9516129032258, 
# 'elevation': 14.162658990412158, 'roll': 0.0, 
# 'loc': (166.79323747325407, 162.4692971514962, 52.28745470591859), 
# 'fov': 0.0, 'zoom': 0.026156241036960133}
# m = vv.mesh(modelmesh)
# # m.faceColor = 'g'
# m.clim = 0, 5
# m.colormap = vv.CM_JET

# Add motion
pointsDeforms = []
node_points = []
for i, node in enumerate(sorted(model.nodes())):
    node_point = vv.solidSphere(translation = (node), scaling = (0.8,0.8,0.8))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    if meshWithColors:
        nodeDeforms = model.node[node]['deforms']
        dmax_xyz = _calculateAmplitude(nodeDeforms, dim='xyz') # [dmax, p1, p2]
        dmax_z = _calculateAmplitude(nodeDeforms, dim='z')
        dmax_y = _calculateAmplitude(nodeDeforms, dim='y')
        dmax_x = _calculateAmplitude(nodeDeforms, dim='x')
        pointsDeforms.append(nodeDeforms)
        node_point.amplXYZ = dmax_xyz # amplitude xyz = [0]
        node_point.amplZ = dmax_z 
        node_point.amplY = dmax_y  
        node_point.amplX = dmax_x 
    node_points.append(node_point)

if meshWithColors:
    points = sorted(model.nodes())
    meanAmplitudeXYZ=calculateMeanAmplitude(points,pointsDeforms, dim='xyz')
    meanAmplitudeZ=calculateMeanAmplitude(points,pointsDeforms, dim='z')
    meanAmplitudeY=calculateMeanAmplitude(points,pointsDeforms, dim='y')
    meanAmplitudeX=calculateMeanAmplitude(points,pointsDeforms, dim='x')

# Create deformable mesh
dm = DeformableMesh(a, modelmesh) # in x,y,z
dm.SetDeforms(*[list(reversed(deform)) for deform in deforms_f]) # from z,y,x to x,y,z
if meshWithColors:
    if meshWithColors == 'displacement':
        dm.clim = clim2
    elif meshWithColors == 'curvature':
        dm.clim = clim3
    dm.colormap = vv.CM_JET #todo: use colormap Viridis or Magma as JET is not linear (https://bids.github.io/colormap/)
    vv.colorbar()
else:
    dm.faceColor = 'g'

# Run mesh
a.SetLimits()
# a.SetView(viewringcrop)
dm.MotionPlay(motionPlay[0], motionPlay[1])  # (10, 0.2) = each 10 ms do a step of 20% for a phase
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 0.5  # 1 or less. For a mesh we can (more) safely increase amplitude
#todo: dm.SetValues in loop for changing color?

# Add clickable nodes
t0 = vv.Label(a, 'Node nr|location: ', fontSize=11, color='w')
t0.position = 0.01, 5, 0.5, 20
t0.bgcolor = None
t0.visible = False
if meshWithColors:
    t1 = vv.Label(a, 'Node amplitude XYZ: ', fontSize=11, color='w')
    t1.position = 0.01, 25, 0.5, 20  # x (frac w), y, w (frac), h
    t1.bgcolor = None
    t1.visible = False
    t2 = vv.Label(a, 'Node amplitude Z: ', fontSize=11, color='w')
    t2.position = 0.01, 45, 0.5, 20
    t2.bgcolor = None
    t2.visible = False
    t3 = vv.Label(a, 'Node amplitude Y: ', fontSize=11, color='w')
    t3.position = 0.01, 65, 0.5, 20
    t3.bgcolor = None
    t3.visible = False
    t4 = vv.Label(a, 'Node amplitude X: ', fontSize=11, color='w')
    t4.position = 0.01, 85, 0.5, 20
    t4.bgcolor = None
    t4.visible = False
    t5 = vv.Label(a, 'MEAN AMPLITUDE NODES: ', fontSize=11, color='w')
    t5.position = 0.45, 25, 0.5, 20
    t5.bgcolor = None
    t5.visible = False
    t5.text = 'MEAN AMPLITUDE NODES XYZ: \b{%1.3f+/-%1.3fmm} (%1.3f-%1.3f)' % (
            meanAmplitudeXYZ[0], meanAmplitudeXYZ[1],meanAmplitudeXYZ[2],meanAmplitudeXYZ[3] )
    t6 = vv.Label(a, 'MEAN AMPLITUDE NODES: ', fontSize=11, color='w')
    t6.position = 0.45, 45, 0.5, 20
    t6.bgcolor = None
    t6.visible = False
    t6.text = 'MEAN AMPLITUDE NODES Z: \b{%1.3f+/-%1.3fmm} (%1.3f-%1.3f)' % (
            meanAmplitudeZ[0], meanAmplitudeZ[1], meanAmplitudeZ[2], meanAmplitudeZ[3])
    t7 = vv.Label(a, 'MEAN AMPLITUDE NODES: ', fontSize=11, color='w')
    t7.position = 0.45, 65, 0.5, 20
    t7.bgcolor = None
    t7.visible = False
    t7.text = 'MEAN AMPLITUDE NODES Y: \b{%1.3f+/-%1.3fmm} (%1.3f-%1.3f)' % (
            meanAmplitudeY[0], meanAmplitudeY[1],meanAmplitudeY[2], meanAmplitudeY[3])
    t8 = vv.Label(a, 'MEAN AMPLITUDE NODES: ', fontSize=11, color='w')
    t8.position = 0.45, 85, 0.5, 20
    t8.bgcolor = None
    t8.visible = False
    t8.text = 'MEAN AMPLITUDE NODES X: \b{%1.3f+/-%1.3fmm} (%1.3f-%1.3f)' % (
            meanAmplitudeX[0], meanAmplitudeX[1],meanAmplitudeX[2], meanAmplitudeX[3])
    
    # print mean amplitude output
    print((t5.text).replace('\x08', '')) # \b is printed as \x08
    print((t6.text).replace('\x08', ''))
    print((t7.text).replace('\x08', ''))
    print((t8.text).replace('\x08', ''))

def on_key(event): 
    if event.key == vv.KEY_DOWN:
        t0.visible = False
        if meshWithColors:
            t1.visible = False
            t2.visible = False
            t3.visible = False
            t4.visible = False
            t5.visible = False
            t6.visible = False
            t7.visible = False
            t8.visible = False
        for node_point in node_points:
            node_point.visible = False
    elif event.key == vv.KEY_UP:
        t0.visible = True
        if meshWithColors:
            t1.visible = True
            t2.visible = True
            t3.visible = True
            t4.visible = True
            t5.visible = True
            t6.visible = True
            t7.visible = True
            t8.visible = True
        for node_point in node_points:
            node_point.visible = True

def pick_node(event):
    nodenr = event.owner.nr
    node = event.owner.node
    t0.text = 'Node nr|location: \b{%i | x=%1.3f y=%1.3f z=%1.3f}' % (nodenr,node[0],node[1],node[2])
    if meshWithColors:
        amplXYZ = event.owner.amplXYZ
        amplZ = event.owner.amplZ
        amplY = event.owner.amplY
        amplX = event.owner.amplX
        t1.text = 'Node amplitude XYZ: \b{%1.3f mm} (%i%%,%i%%)' % (amplXYZ[0],amplXYZ[1]*10,amplXYZ[2]*10)
        t2.text = 'Node amplitude Z: \b{%1.3f mm} (%i%%,%i%%)' % (amplZ[0],amplZ[1]*10,amplZ[2]*10)
        t3.text = 'Node amplitude Y: \b{%1.3f mm} (%i%%,%i%%)' % (amplY[0],amplY[1]*10,amplY[2]*10)
        t4.text = 'Node amplitude X: \b{%1.3f mm} (%i%%,%i%%)' % (amplX[0],amplX[1]*10,amplX[2]*10)

def unpick_node(event):
    t0.text = 'Node nr|location: '
    if meshWithColors: 
        t1.text = 'Node amplitude XYZ: ' 
        t2.text = 'Node amplitude Z: ' 
        t3.text = 'Node amplitude Y: '
        t4.text = 'Node amplitude X: '


# Bind event handlers
f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)

# Bind rotate view
f = vv.gcf()
ax = vv.gca()
f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [ax], axishandling=False) ) # crtl+L/R/up/down
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [ax]) )

# ax.SetView({'loc': (106.0082950235823, 108.20059588594421, 59.10612742987558), 'elevation': 60.3586956521739, 'zoom': 0.015789282821813144, 'azimuth': 146.66058394160584, 'roll': 0.0, 'daspect': (1.0, 1.0, -1.0), 'fov': 0.0})

## run ecgslider
ecg = runEcgSlider(dm, f, a, motionPlay)


## Hide volume
# t.visible = False

## Turn on/off axis
# vv.figure(1); a1 = vv.gca()
# vv.figure(2); a2= vv.gca()
# 
# switch = True
# 
# a1.axis.visible = switch
# a2.axis.visible = switch

## Use same camera when 2 models are running
# a1.camera = a2.camera

## Turn on/off moving mesh

# dm.visible = False
# dm.visible = True
