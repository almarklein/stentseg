"""
Script to show the stent model.
"""

import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.stentdirect.stentgraph import create_mesh


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode = 'discharge'
cropname = 'ring'

# Load the stent model and mesh
s = loadmodel(basedir, ptcode, ctcode, cropname)
model = s.model
modelmesh = create_mesh(model, 0.6)  # Param is thickness

# Load deformations
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

# todo: the deforms are stored in backward mapping (I think)
# so we need to transform them to forward here.

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1

# Create deformable mesh
dm = DeformableMesh(a, modelmesh)
dm.SetDeforms(*deforms)

# Run
a.SetLimits()
dm.MotionPlay(10, 0.2)  # Each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude


# todo: add static CT image for reference. Nice excersise for Maaike :)

# In stentseg.motion.vis are a few functions, but they need to be adjusted
# to work with the new stent model.
