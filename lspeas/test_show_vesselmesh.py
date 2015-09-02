""" Show the vessel in motion, based on segmentated 3D mesh model.
Input: stl of vessel, avgreg and deforms
"""

import os
import imageio
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
import pirt
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
import numpy as np


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_002'
# ctcode, nr = 'discharge', 1
ctcode, nr = 'pre', 2
cropname = 'stent'
meshfile = 'LSPEAS_002_Smoothed5x0.8_Wrapped5mm_plus r renalis2 9_001.stl'

# Load ssdf
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
volavg = s.vol
origin = volavg.origin # z y x

# Load deformations (apply forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]
deforms_f = [pirt.DeformationFieldForward(*f) for f in deforms]

# Load mesh
basedir2 = select_dir(r'C:\MedData\LSPEAS_Mimics',
                r'F:\LSPEAS_Mimics_backup',
                r'K:\LSPEAS_Mimics_backup')
mesh = vv.meshRead(os.path.join(basedir2, ptcode, meshfile))
# z is negative, must be flipped to match dicom orientation
for vertice in mesh._vertices:
    vertice[-1] = vertice[-1]*-1
#mesh = vv.meshRead(r'D:\Profiles\koenradesma\Desktop\001_preavgreg 20150522 test itk.stl')
# x and y values of vertices are negative (as in original dicom), flip to match local coordinates
#mesh._vertices = mesh._vertices*-1  # when stl from itksnap


## Start vis
f = vv.figure(1); vv.clf()
f.position = 8.00, 31.00,  944.00, 1001.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

t = vv.volshow2(volavg, clim=(-550, 500)) # -750, 1000

# m = vv.mesh(mesh)
# m.faceColor = 'g' 

# Create deformable mesh
dm = DeformableMesh(a, mesh)
dm.SetDeforms(*deforms_f)
dm.clim = 0, 4
dm.colormap = vv.CM_JET
vv.colorbar()

# Run mesh
a.SetLimits()
# a.SetView(viewringcrop)
dm.MotionPlay(5, 1)  # (10, 0.2) = each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
dm.faceColor = 'g'
