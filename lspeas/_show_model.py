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
#import skimage.morphology
#from skimage.morphology import reconstruction

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode, nr = 'discharge', 1
#ctcode, nr = '1month', 2
cropname = 'ring'

# Load deformations
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deformsMesh = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname)
model = s.model
#modelmesh = create_mesh(model, 0.9)  # Param is thickness
modelmesh = create_mesh_with_deforms(model, deformsMesh, s.origin, radius=0.9, fullPaths=True)


# todo: the deforms are stored in backward mapping (I think)
# so we need to transform them to forward here.


# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

# Remove stent from vol for visualization
#vol = remove_stent_from_volume(vol, model, stripSize=4)

# todo: also create a way to show static ring thinner/transparent as reference 
# skimage.morphology.reconstruction(seed, mask, method='dilation', selem=None, offset=None)
# seed = vol # seed image is eroded
# mask = np.zeros_like(vol, np.uint8)
# mask[np.where(vol < 2500)] = 2500
# vol2 = reconstruction(seed, mask, method='erosion')

# Start vis
f = vv.figure(nr); vv.clf()
f.position = 0, 22, 1366, 706
a = vv.gca()
#a.axis.axisColor = 1,1,1
#a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
t = vv.volshow(vol, clim=(0, 3000), renderStyle='mip')
#vv.ColormapEditor(vv.gcf())
#t.colormap = (0,0,0,0), (1,1,1,1) # 0 , 1500 clim
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
viewringcrop = {'azimuth': 82.31139646869984,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 24.864864864864863,
 'fov': 0.0,
 'loc': (149.82577398633552, 86.60102001731882, 82.34515557761024),
 'roll': 0.0,
 'zoom': 0.008816683004670308}

node_points = []
for node in model.nodes():
    node_point = vv.solidSphere(translation = (node), scaling = (1.1,1.1,1.1))
    node_point.faceColor = 'b'
    node_point.visible = False
    node_point.node = node
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
dm.MotionPlay(10, 0.2)  # Each 10 ms do a step of 20%
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


def on_key(event): 
    if event.key == vv.KEY_DOWN:
        t1.visible = False
        t2.visible = False
        for node_point in node_points:
            node_point.visible = False  # node_point.faceColor = 0, 0, 0, 0
    elif event.key == vv.KEY_UP:
        t1.visible = True
        t2.visible = True
        for node_point in node_points:
            node_point.visible = True

def pick_node(event):
    maxDeform = event.owner.maxDeform
    sumDeform = event.owner.sumDeform
    t1.text = 'Max of deformation vectors: \b{%1.1f mm}' % maxDeform
    t2.text = 'Sum of deformation vectors: \b{%1.1f mm}' % sumDeform

def unpick_node(event):
    t1.text = 'Max of deformation vectors:'
    t2.text = 'Sum of deformation vectors:'

# Bind event handlers
f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)
    node_point.eventLeave.Bind(unpick_node)


#vv.record(f)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.

## Use same camera when 2 models are running

#vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca(); a1.camera = a2.camera

