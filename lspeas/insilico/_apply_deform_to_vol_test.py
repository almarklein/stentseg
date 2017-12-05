import os

import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
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
ptcode = 'LSPEAS_002'
ctcode, nr = '12months', 1
# ptcode = 'QRM_FANTOOM_20160121'
# ctcode, nr = 'ZA3-75-1.2', 1
cropname = 'ring'
modelname = 'modelavgreg'
motion = 'amplitude'  # amplitude or sum
dimension = 'xyz'
showVol  = 'ISO'  # MIP or ISO or 2D or None
clim0  = (-10,2500) 
clim2 = (0,3)
isoTh = 250
motionPlay = 9, 1  # each x ms, a step of perc of T
staticref =  'avgreg'# 'avg7020'
meshWithColors = True


# Load deformations (forward for mesh)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
# deforms = [s['deform%i'%(i*10)] for i in range(10)]
deformkeys = []
for key in dir(s):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s[key] for key in deformkeys]
# deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]

try:
    s2 = loadvol(basedir, ptcode, ctcode, 'ring', staticref)
except FileNotFoundError:
    s2 = loadvol(basedir, ptcode, ctcode, 'ring', staticref)
vol = s2.vol

s3 = loadvol(basedir, ptcode, ctcode, cropname, 'phases')
vol0ori = s3.vol0

# todo: backward/forward based on how deforms were obtained??
# deforms was obtained as backward, from original phases to mean volume avgreg
deform = pirt.DeformationFieldBackward(deforms[0])
# vol2 = pirt.interp.deform_backward(vol, deforms[0]) # te low level, gebruikt awarp niet
vol2 = deform.inverse().as_backward().apply_deformation(vol0ori) # gebruikt pirt deformation.py

vv.figure(1); vv.clf()
a1 = vv.subplot(131); t1 = vv.volshow(vol)
a1.daspect = (1, 1, -1)
vv.title('vol average of cardiac cycle')
# vv.figure(2); vv.clf()
a2 = vv.subplot(132); t2 = vv.volshow(vol2)
a2.daspect = (1, 1, -1)
vv.title('vol 0 deformed to avg volume')
# vv.figure(3); vv.clf()
a3 = vv.subplot(133); t3 = vv.volshow2((vol2-vol), clim=(-500,500))
a3.daspect = (1, 1, -1)
vv.title('difference')

a1.camera = a2.camera = a3.camera
t1.clim = t2.clim = 0, 2000
t3.clim = -500, 500



