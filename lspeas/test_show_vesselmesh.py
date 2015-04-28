""" Show the volume in a dynamic way.
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
                     r'D:\LSPEAS\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_002'
# ctcode, nr = 'discharge', 1
ctcode, nr = 'pre', 2
cropname = 'stent'

# Basedir for dicom from ssdf2dicom
basedir2 = select_dir(r'C:\DICOMavgreg_backup\DICOMavgreg',
                     r'D:\LSPEAS\DICOMavgreg',)

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, 'phases') # from original dicom
vols = [s['vol%i'%(i*10)] for i in range(10)]

# Load ssdf
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
volavg = s.vol
origin = volavg.origin # z y x

# Load dicom 
vol = imageio.volread(os.path.join(basedir2, ptcode, ptcode+'_'+ctcode), 'dicom') # from ssdf2dicom
sampling = vol.meta.sampling # z y x
vol = vv.Aarray(vol, sampling , origin)  # give origin, vv.Aarray otherwise defines origin: 0,0,0

# Load deformations (apply forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]
deforms_f = [pirt.DeformationFieldForward(*f) for f in deforms]

# Load Mesh
mesh = vv.meshRead(r'D:\Profiles\koenradesma\SURFdrive\LSPEAS\Mimics\test\LSPEAS_002_Smoothed_Wrapped_Purple 5_001.stl')



## Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.axis.axisColor = 1,1,1
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))

# t = vv.volshow2(vols[0], clim=(-550, 500)) # -750, 1000
# t2 = vv.volshow2(vol, clim=(-550, 2000)) # -750, 1000
t3 = vv.volshow2(volavg, clim=(-550, 500)) # -750, 1000
# test: t1 t2 t3 equal in orientation, scale, origin

# ... in xyz order
sampling_xyz, origin_xyz =  np.flipud(sampling), np.flipud(origin)

# Transform from world coordinates to volume indices
# # pp1 = ((pp - origin_xyz) / sampling_xyz + 0.25).astype('int32')
# mesh = ((mesh - origin_xyz) / sampling_xyz + 0.25).astype('int32')

# m = vv.mesh(mesh)
# m.faceColor = 'g' 

#todo: origin scale z-orientation wrong; mimics export stl in world coordinates?


# Create deformable mesh
dm = DeformableMesh(a, mesh)
dm.SetDeforms(*deforms_f)
dm.clim = 0, 4
dm.colormap = vv.CM_JET
vv.colorbar()

# Run mesh
a.SetLimits()
# a.SetView(viewringcrop)
dm.MotionPlay(5, 0.8)  # (10, 0.2) = each 10 ms do a step of 20%
dm.motionSplineType = 'B-spline'
dm.motionAmplitude = 3.0  # For a mesh we can (more) safely increase amplitude
dm.faceColor = 'g'
