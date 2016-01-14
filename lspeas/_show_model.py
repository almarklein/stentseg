"""
Script to show the stent model.
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import remove_stent_from_volume, show_ctvolume
from stentseg.motion.vis import create_mesh_with_abs_displacement
import pirt
import numpy as np
from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion
from stentseg.motion.displacement import calculateMeanAmplitude

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_BACKUP',r'G:\LSPEAS_ssdf_BACKUP')

# Select dataset to register
ptcode = 'FANTOOM_20151202'
# ctcode, nr = 'prof0', 1
ctcode, nr = 'prof3', 2
cropname = 'ring'
modelname = 'modelavgreg'
motion = 'amplitude'  # amplitude or sum
dimension = 'z'
showVol  = 'ISO'  # MIP or ISO or 2D or None
clim0  = (0,3000)
clim2 = (0,1.5)
clim3 = -550,500
isoTh = 250
motionPlay = 5, 0.6  # each x ms, a step of x %


# Load deformations (forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
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
model = s.model
# modelmesh = create_mesh(model, 1.0)  # Param is thickness
modelmesh = create_mesh_with_abs_displacement(model, radius = 1.0, dim = dimension, motion = motion)

# Load static CT image to add as reference
s2 = loadvol(basedir, ptcode, ctcode, 'stent', 'avg3090')
vol = s2.vol



# Start vis
f = vv.figure(nr); vv.clf()
if nr == 1:
    f.position = 8.00, 30.00,  944.00, 1002.00
else:
    f.position = 968.00, 30.00,  944.00, 1002.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = True
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s  (colorbar \b{%s} of motion in mm in %s)' % (ptcode[7:], ctcode, motion, dimension))
# viewringcrop = {'azimuth': -166.8860353130016,
#  'daspect': (1.0, 1.0, -1.0),
#  'elevation': 8.783783783783782,
#  'fov': 0.0,
#  'loc': (113.99322808141005, 161.58640433480713, 73.92662200285992),
#  'roll': 0.0,
#  'zoom': 0.01066818643565108}
# m = vv.mesh(modelmesh)
# # m.faceColor = 'g'
# m.clim = 0, 5
# m.colormap = vv.CM_JET

# Add motion
pointsDeforms = []
node_points = []
for i, node in enumerate(sorted(model.nodes())):
    node_point = vv.solidSphere(translation = (node), scaling = (1.1,1.1,1.1))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    nodeDeforms = model.node[node]['deforms']
    dmax_xyz = _calculateAmplitude(nodeDeforms, dim='xyz') # [dmax, p1, p2]
    dmax_z = _calculateAmplitude(nodeDeforms, dim='z')
    dmax_y = _calculateAmplitude(nodeDeforms, dim='y')
    dmax_x = _calculateAmplitude(nodeDeforms, dim='x')
    pointsDeforms.append(nodeDeforms)
    node_point.amplXYZ = dmax_xyz[0] # amplitude xyz
    node_point.amplZ = dmax_z[0] 
    node_point.amplY = dmax_y[0]  
    node_point.amplX = dmax_x[0] 
    node_points.append(node_point)

points = sorted(model.nodes())
meanAmplitude=calculateMeanAmplitude(points,pointsDeforms, dim=dimension)

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms_f)
dm.clim = clim2
dm.colormap = vv.CM_JET
vv.colorbar()

# Run mesh
a.SetLimits()
# a.SetView(viewringcrop)
dm.MotionPlay(motionPlay[0], motionPlay[1])  # (10, 0.2) = each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 2.0  # For a mesh we can (more) safely increase amplitude
#dm.faceColor = 'g'


# Add clickable nodes
t0 = vv.Label(a, 'Node nr|location: ', fontSize=11, color='w')
t0.position = 0.2, 5, 0.5, 20
t0.bgcolor = None
t0.visible = False
t1 = vv.Label(a, 'Node amplitude XYZ: ', fontSize=11, color='w')
t1.position = 0.2, 25, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a, 'Node amplitude Z: ', fontSize=11, color='w')
t2.position = 0.2, 45, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a, 'Node amplitude Y: ', fontSize=11, color='w')
t3.position = 0.2, 65, 0.5, 20
t3.bgcolor = None
t3.visible = False
t4 = vv.Label(a, 'Node amplitude X: ', fontSize=11, color='w')
t4.position = 0.2, 85, 0.5, 20
t4.bgcolor = None
t4.visible = False
t5 = vv.Label(a, 'MEAN AMPLITUDE NODES: ', fontSize=11, color='w')
t5.position = 0.58, 85, 0.5, 20
t5.bgcolor = None
t5.visible = False
t5.text = 'MEAN AMPLITUDE NODES: \b{%1.3f+/-%1.3fmm}' % (meanAmplitude[0], meanAmplitude[1])

def on_key(event): 
    if event.key == vv.KEY_DOWN:
        t0.visible = False
        t1.visible = False
        t2.visible = False
        t3.visible = False
        t4.visible = False
        t5.visible = False
        for node_point in node_points:
            node_point.visible = False
    elif event.key == vv.KEY_UP:
        t0.visible = True
        t1.visible = True
        t2.visible = True
        t3.visible = True
        t4.visible = True
        t5.visible = True
        for node_point in node_points:
            node_point.visible = True

def pick_node(event):
    amplXYZ = event.owner.amplXYZ
    amplZ = event.owner.amplZ
    amplY = event.owner.amplY
    amplX = event.owner.amplX
    nodenr = event.owner.nr
    node = event.owner.node
    t0.text = 'Node nr|location: \b{%i | x=%1.3f y=%1.3f z=%1.3f}' % (nodenr,node[0],node[1],node[2])
    t1.text = 'Node amplitude XYZ: \b{%1.3f mm}' % amplXYZ
    t2.text = 'Node amplitude Z: \b{%1.3f mm}' % amplZ
    t3.text = 'Node amplitude Y: \b{%1.3f mm}' % amplY
    t4.text = 'Node amplitude X: \b{%1.3f mm}' % amplX

def unpick_node(event):
    t0.text = 'Node nr|location: ' 
    t1.text = 'Node amplitude XYZ: ' 
    t2.text = 'Node amplitude Z: ' 
    t3.text = 'Node amplitude Y: '
    t4.text = 'Node amplitude X: '


# Bind event handlers
f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.

## Turn on/off axis
# vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca()
# 
# switch = False
# 
# a1.axis.visible = switch
# a2.axis.visible = switch

## Use same camera when 2 models are running
# a1.camera = a2.camera

## Turn on/off moving mesh

# dm.visible = False
# dm.visible = True
