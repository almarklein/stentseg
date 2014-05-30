""" Show the volume in a dynamic way.
"""

## Show 3D movie, by alternating the 10 volumes

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Load volumes
s = loadvol(basedir, 'LSPEAS_003', 'discharge', 'ring', 'phases')
vols = [s['vol%i'%(i*10)] for i in range(10)]

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1

# Setup data container
container = vv.MotionDataContainer(a)
for vol in vols:
    #t = vv.volshow(vol, clim=(0, 1000), renderStyle='mip')
    t = vv.volshow(vol, clim=(-1000, 3000), renderStyle='iso')
    t.isoThreshold = 400
    t.parent = container



## Show 3D movie, by showing one volume that is moved by motion fields

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
from pirt.utils.deformvis import DeformableTexture3D

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Load volume
s = loadvol(basedir, 'LSPEAS_003', 'discharge', 'ring', 'avgreg')
vol = s.vol

# Load deformations
s = loadvol(basedir, 'LSPEAS_003', 'discharge', 'ring', 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[-field[::2,::2,::2] for field in fields] for fields in deforms]

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1

# Setup motion container
dt = DeformableTexture3D(a, vol)
dt.clim = 0, 1000
dt.SetDeforms(*deforms)

# Set limits and play!
a.SetLimits()
dt.MotionPlay(10, 0.2)

dt.motionSplineType = 'B-spline'
dt.motionAmplitude = 2.0
dt.MotionPlay
