""" Show the volume in a dynamic way.
"""

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
from pirt.utils.deformvis import DeformableTexture3D

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode = 'discharge'
cropname = 'ring'

## Show 3D movie, by alternating the 10 volumes

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, 'phases')
vols = [s['vol%i'%(i*10)] for i in range(10)]

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1

# Setup data container
container = vv.MotionDataContainer(a)
for vol in vols:
    #t = vv.volshow(vol, clim=(0, 1000), renderStyle='mip')
    t = vv.volshow(vol, clim=(0, 3000))
    t.isoThreshold = 400
    t.renderStyle = 'mip'  # iso or mip work well
    t.parent = container



## Show 3D movie, by showing one volume that is moved by motion fields

# Load volume
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

# Load deformations
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [[-field[::2,::2,::2] for field in fields] for fields in deforms]

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, -1, -1

# Setup motion container
dt = DeformableTexture3D(a, vol)
dt.clim = 0, 3000
dt.isoThreshold = 400
dt.renderStyle = 'mip'  # iso or mip work well
dt.SetDeforms(*deforms)

# Set limits and play!
a.SetLimits()
dt.MotionPlay(10, 0.2)  # (10, 0.2) = each 10 ms do a step of 20%
                        # With 85 bpm every beat 706 ms; 141 ms per 20%  

dt.motionSplineType = 'B-spline'
dt.motionAmplitude = 2.0
