"""
Script to show the stent model.
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import remove_stent_from_volume
from stentseg.motion.vis import create_mesh_with_abs_displacement
import pirt
import numpy as np
# import skimage.morphology
# from skimage.morphology import reconstruction

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode, nr = 'discharge', 1
# ctcode, nr = 'pre', 2
cropname = 'ring'
modelname = 'modelavgreg'

# Load deformations (forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
model = s.model
# modelmesh = create_mesh(model, 1.0)  # Param is thickness
modelmesh = create_mesh_with_abs_displacement(model, radius = 1.0, dimensions = 'xyz')

# Load static CT image to add as reference
s2 = loadvol(basedir, ptcode, ctcode, 'stent', 'avgreg')
vol = s2.vol

# Remove stent from vol for visualization
vol = remove_stent_from_volume(vol, model, stripSize=7)

# todo: also create a way to show static ring thinner/transparent as reference 
# vol2 = np.copy(vol)
# seed = vol2*(vol2<1400) # erosion starts from seed image minima
# # seed start moet original, deel te eroden moet max
# seed[np.where(seed == 0)] = vol2.max()
# mask = vol2
# vol = reconstruction(seed, mask, method='erosion')


# Start vis
f = vv.figure(nr); vv.clf()
if nr == 1:
    f.position = 8.00, 30.00,  944.00, 1002.00
else:
    f.position = 968.00, 30.00,  944.00, 1002.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
t = vv.volshow(vol, clim=(0, 3000), renderStyle='iso')
t.isoThreshold = 250
# vv.ColormapEditor(vv.gcf())
t.colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}
# t = vv.volshow2(vol)
# t.clim = -550, 500
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
viewringcrop = {'azimuth': -166.8860353130016,
 'daspect': (1.0, 1.0, -1.0),
 'elevation': 8.783783783783782,
 'fov': 0.0,
 'loc': (113.99322808141005, 161.58640433480713, 73.92662200285992),
 'roll': 0.0,
 'zoom': 0.01066818643565108}
# m = vv.mesh(modelmesh)
# # m.faceColor = 'g'
# m.clim = 0, 5
# m.colormap = vv.CM_JET

## Add motion
node_points = []
for i, node in enumerate(model.nodes()):
    node_point = vv.solidSphere(translation = (node), scaling = (1.1,1.1,1.1))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    nodeDeforms = model.node[node]['deforms']
    nodepositions = node + nodeDeforms
    # get displacement during cardiac cycle for node
    vectors = []
    npositions = len(nodepositions)
    for j in range(npositions):
        if j == npositions-1:  # -1 as range starts at 0
            # vector from point at 90% RR to 0%% RR
            vectors.append(nodepositions[j]-nodepositions[0])
        else:
            vectors.append(nodepositions[j]-nodepositions[j+1])
    vectors = np.vstack(vectors)
    dxyz = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5  # 3Dvector length in mm
    dxy = (vectors[:,0]**2 + vectors[:,1]**2 )**0.5  # 2Dvector length in mm
    dz = abs(vectors[:,2])  # 1Dvector length in mm
    node_point.displacementXYZ = dxyz.sum() # displacement of node xyz
    node_point.displacementXY = dxy.sum() # displacement of node xy
    node_point.displacementZ = dz.sum() # displacement of node z
    node_points.append(node_point)

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms)
dm.clim = 0, 4
dm.colormap = vv.CM_JET
vv.colorbar()

# Run mesh
a.SetLimits()
a.SetView(viewringcrop)
dm.MotionPlay(10, 0.2)  # (10, 0.2) = each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
#dm.faceColor = 'g'


# Add clickable nodes
t0 = vv.Label(a, 'Node nr: ', fontSize=11, color='w')
t0.position = 0.2, 5, 0.5, 20
t0.bgcolor = None
t0.visible = False
t1 = vv.Label(a, 'Node displacement XYZ: ', fontSize=11, color='w')
t1.position = 0.2, 25, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a, 'Node displacement XY: ', fontSize=11, color='w')
t2.position = 0.2, 45, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a, 'Node displacement Z: ', fontSize=11, color='w')
t3.position = 0.2, 65, 0.5, 20
t3.bgcolor = None
t3.visible = False


def on_key(event): 
    if event.key == vv.KEY_DOWN:
        t0.visible = False
        t1.visible = False
        t2.visible = False
        t3.visible = False
        for node_point in node_points:
            node_point.visible = False
    elif event.key == vv.KEY_UP:
        t0.visible = True
        t1.visible = True
        t2.visible = True
        t3.visible = True
        for node_point in node_points:
            node_point.visible = True

def pick_node(event):
    displacementXYZ = event.owner.displacementXYZ
    displacementXY = event.owner.displacementXY
    displacementZ = event.owner.displacementZ
    nodenr = event.owner.nr
    t0.text = 'Node nr: \b{%i}' % nodenr
    t1.text = 'Node displacement XYZ: \b{%1.1f mm}' % displacementXYZ
    t2.text = 'Node displacement XY: \b{%1.1f mm}' % displacementXY
    t3.text = 'Node displacement Z: \b{%1.1f mm}' % displacementZ

def unpick_node(event):
    t0.text = 'Node nr: ' 
    t1.text = 'Node displacement XYZ: ' 
    t2.text = 'Node displacement XY: ' 
    t3.text = 'Node displacement Z: '


# Bind event handlers
f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.

## Turn on/off axis
vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca()

switch = False

a1.axis.visible = switch
a2.axis.visible = switch

## Use same camera when 2 models are running
a1.camera = a2.camera

## Turn on/off moving mesh

dm.visible = False
dm.visible = True
