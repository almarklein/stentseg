"""
Script to show the stent model.
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_deforms,remove_stent_from_volume
import pirt 
import skimage.morphology
from skimage.morphology import reconstruction

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode, nr = 'discharge', 1
# ctcode, nr = '1month', 2
cropname = 'ring'
modelname = 'modelavgreg'

# Load deformations
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deformsMesh = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
model = s.model
# modelmesh = create_mesh(model, 0.9)  # Param is thickness
modelmesh = create_mesh_with_deforms(model, deformsMesh, s.origin, radius=0.9, fullPaths=True)

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, 'stent', 'avgreg')
vol = s.vol

# Remove stent from vol for visualization
# vol = remove_stent_from_volume(vol, model, stripSize=4)

# todo: also create a way to show static ring thinner/transparent as reference 
# vol2 = np.copy(vol)
# seed = vol2*(vol2<1400) # erosion starts from seed image minima
# # seed start moet original, deel te eroden moet max
# seed[np.where(seed == 0)] = vol2.max()
# mask = vol2
# vol = reconstruction(seed, mask, method='erosion')

# todo: the deforms are stored in backward mapping (I think)
# so we need to transform them to forward here.

# Start vis
f = vv.figure(nr); vv.clf()
f.position = 0, 22, 1366, 706
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
t = vv.volshow(vol, clim=(0, 4000), renderStyle='mip')
#vv.ColormapEditor(vv.gcf())
#t.colormap = (0,0,0,0), (1,1,1,1) # 0 , 1500 clim
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
viewringcrop = {'azimuth': -166.8860353130016,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 8.783783783783782,
 'fov': 0.0,
 'loc': (113.99322808141005, 161.58640433480713, 73.92662200285992),
 'roll': 0.0,
 'zoom': 0.01066818643565108}

node_points = []
for i, node in enumerate(model.nodes()):
    node_point = vv.solidSphere(translation = (node), scaling = (1.1,1.1,1.1))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
    node_point.nr = i
    mov = model.node[node]['deforms']
    d = (mov[:,0]**2 + mov[:,1]**2 + mov[:,2]**2)**0.5  # magnitude in mm
    node_point.maxDeform = d.max()  # a measure of max deformation for point
    node_point.sumDeform = d.sum()
    node_points.append(node_point)

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms)
dm.clim = 0, 5
dm.colormap = vv.CM_JET
vv.colorbar()

# Run
a.SetLimits()
a.SetView(viewringcrop)
dm.MotionPlay(0.1, 0.2)  # (10, 0.2) = each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
#dm.faceColor = 'g'


# Add clickable nodes
t1 = vv.Label(a, 'Max of deformation vectors: ', fontSize=11, color='b')
t1.position = 0.2, 5, 0.5, 20  # x (frac w), y, w (frac), h
t1.bgcolor = None
t1.visible = False
t2 = vv.Label(a, 'Sum of deformation vectors: ', fontSize=11, color='b')
t2.position = 0.2, 25, 0.5, 20
t2.bgcolor = None
t2.visible = False
t3 = vv.Label(a, 'Node nr: ', fontSize=11, color='b')
t3.position = 0.2, 45, 0.5, 20
t3.bgcolor = None
t3.visible = False


def on_key(event): 
    if event.key == vv.KEY_DOWN:
        t1.visible = False
        t2.visible = False
        t3.visible = False
        for node_point in node_points:
            node_point.visible = False
    elif event.key == vv.KEY_UP:
        t1.visible = True
        t2.visible = True
        t3.visible = True
        for node_point in node_points:
            node_point.visible = True

def pick_node(event):
    maxDeform = event.owner.maxDeform
    sumDeform = event.owner.sumDeform
    nodenr = event.owner.nr
    t1.text = 'Max of deformation vectors: \b{%1.1f mm}' % maxDeform
    t2.text = 'Sum of deformation vectors: \b{%1.1f mm}' % sumDeform
    t3.text = 'Node nr: \b{%i}' % nodenr

def unpick_node(event):
    t1.text = 'Max of deformation vectors:'
    t2.text = 'Sum of deformation vectors:'
    t3.text = 'Node nr: '

# Bind event handlers
f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)

# # Record video
# rec = vv.record(a)
# #...
# # Export
# rec.Stop()
# rec.Export('%s_%s_%s_%s.avi') % (ptcode, ctcode, cropname, 'model')

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
