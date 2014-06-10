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
#modelmesh = create_mesh(model, 0.7)  # Param is thickness
modelmesh = create_mesh_with_deforms(model, deformsMesh, s.origin, radius=0.7, fullPaths=False)
#todo: create mesh based on path and deforms -> which points deform most?



# todo: the deforms are stored in backward mapping (I think)
# so we need to transform them to forward here.


# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

# Remove stent from vol for visualization
vol = remove_stent_from_volume(vol, model, stripSize=4)

# todo: also create a way to show static ring thinner/transparent as reference 
# skimage.morphology.reconstruction(seed, mask, method='dilation', selem=None, offset=None)
# seed = vol # seed image is eroded
# mask = np.zeros_like(vol, np.uint8)
# mask[np.where(vol < 2500)] = 2500
# vol2 = reconstruction(seed, mask, method='erosion')

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1
t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
vv.ColormapEditor(vv.gcf())

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms)

# Run
a.SetLimits()
dm.MotionPlay(10, 0.2)  # Each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
#dm.faceColor = 'g'

#vv.record(a)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.
