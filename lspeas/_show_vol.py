""" Show the volume in a dynamic way.
"""

import os
import pirt
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
from pirt.utils.deformvis import DeformableTexture3D

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode, nr = 'discharge', 1
# ctcode, nr = 'pre', 2
cropname = 'stent'

## Show 3D movie, by alternating the 10 volumes

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, 'phases')
vols = [s['vol%i'%(i*10)] for i in range(10)]

# Start vis
f = vv.figure(1); vv.clf()
a = vv.gca()
a.daspect = 1, 1, -1

# Setup data container
container = vv.MotionDataContainer(a)
for vol in vols:
    t = vv.volshow2(vol, clim=(-550, 500)) # -750, 1000
#     t = vv.volshow(vol, clim=(0, 3000), renderStyle = 'mip')
    #t.isoThreshold = 400               # iso or mip work well 
    t.parent = container



## Show 3D movie, by showing one volume that is moved by motion fields

# Load volume
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

# Load deformations (use backward mapping to deform texture 3D volume)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms_forward = [s['deform%i'%(i*10)] for i in range(10)]
deforms_forward = [pirt.DeformationFieldForward(*fields) for fields in deforms_forward] # wrap fields
deforms_backward = [deform.as_backward() for deform in deforms_forward] # get backward mapping

# Start vis
f = vv.figure(nr); vv.clf()
if nr == 1:
    f.position = 8.00, 30.00,  667.00, 690.00
else:
    f.position = 691.00, 30.00,  667.00, 690.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
vv.ColormapEditor(vv.gcf())

# Setup motion container
dt = DeformableTexture3D(a, vol)
dt.clim = 0, 3000
dt.isoThreshold = 400
dt.renderStyle = 'iso'  # iso or mip work well
dt.SetDeforms(*deforms_backward)

# Set limits and play!
a.SetLimits()
dt.MotionPlay(10, 0.3)  # (10, 0.2) = each 10 ms do a step of 20% ;(0.1,0.2)
                        # With 85 bpm every beat 706 ms; 141 ms per 20%  

dt.motionSplineType = 'B-spline'
dt.motionAmplitude = 2.0

## Turn on/off axis
vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca()

switch = False

a1.axis.visible = switch
a2.axis.visible = switch

## Use same camera when 2 vols are running
a1.camera = a2.camera
