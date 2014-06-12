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
from skimage.morphology import reconstruction

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode = 'discharge'
cropname = 'ring'

# Load deformations
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deformsMesh = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname)
model = s.model
modelmesh = create_mesh(model, 0.9)  # Param is thickness
#modelmesh = create_mesh_with_deforms(model, deformsMesh, s.origin, radius=0.7, fullPaths=True)
#todo: create mesh based on path and deforms -> which points deform most?



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
f = vv.figure(2); vv.clf()
a = vv.gca()
#a.axis.axisColor = 1,1,1
#a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
#vv.ColormapEditor(vv.gcf())
#t.colormap = (0,0,0,0), (1,1,1,1) # 0 , 1500 clim
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms)
# dm.clim = 0, 5
dm.colormap = vv.CM_JET
#vv.colorbar()

# Run
a.SetLimits()
dm.MotionPlay(10, 0.2)  # Each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
dm.faceColor = 'g'


# Add clickable nodes
textoffset = a.GetLimits()
textoffset = [0.5*(textoffset[0].min+textoffset[0].max), 0.5*(textoffset[1].min+textoffset[1].max), textoffset[2].min-10]  # x,y,z
t1 = vv.Text(a, text='Max deformation: ', x=textoffset[0], y=textoffset[1], z=textoffset[2], fontSize=9, color='b')

node_points = []
#node = model.nodes()[0]
for node in model.nodes():
    node_point = vv.solidSphere(translation = (node), scaling = (1.5,1.5,1.5))
    node_point.faceColor = 'c'
    node_point.node = node
    mov = model.node[node]['deforms']
    d = (mov[:,0]**2 + mov[:,1]**2 + mov[:,2]**2)**0.5  # magnitude in mm
    node_point.nodeDeform = d.max()  # a measure of max deformation for point
    node_points.append(node_point)
    
# todo: fix hide and show all nodes on key commands; fix reuse of t1 for text
def hide_node(event):
    if event.key  == vv.KEY_DOWN:
        for node_point in node_points:
            node_point.alpha = 0
    #event.owner.alpha = 0
def show_node(event):
    if event.key  == vv.KEY_UP:
        for node_point in node_points:
            node_point.alpha = 1
    #event.owner.alpha = 1
def pick_node(event):
    nodeDeform = event.owner.nodeDeform
#     if t1:
#         t1.Destroy()
    t1 = vv.Text(a, text='Max deformation: %1.2f mm' % nodeDeform, x=textoffset[0], y=textoffset[1], z=textoffset[2], fontSize=9, color='b')
    #t1.UpdatePosition(text='Max deformation: %1.2f mm' % nodeDeform)

f.eventKeyDown.Bind(hide_node)
#node_point.eventLeave.Bind(hide_node)

f.eventKeyUp.Bind(show_node)
#node_point.eventEnter.Bind(show_node)

#f.eventDoubleClick.Bind(text_remove)

for node_point in node_points:
    node_point.eventEnter.Bind(pick_node)


#vv.record(a)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.
