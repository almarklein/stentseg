""" Show the volume in a dynamic way.
"""

import os
import pirt
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
from pirt.utils.deformvis import DeformableTexture3D
from stentseg.apps.record_movie import recordMovie

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
# basedir = select_dir(r'D:\LSPEAS_F\LSPEASF_ssdf', 
#                      r'E:\LSPEASF_ssdf_backup\LSPEASF_C_01')

# Select dataset to register
ptcode = 'LSPEAS_004'
# ctcode, nr = 'ZA6-75-0', 1
ctcode, nr = 'discharge', 2
cropname = 'ring'

## Show 3D movie, by alternating the 10 volumes
from lspeas.utils.vis import showVolPhases

showVol='mip'
t = showVolPhases(basedir, None, ptcode, ctcode, cropname, showVol=showVol, 
        mipIsocolor=False, isoTh=310, clim=(60,3000), slider=True  )

foo = recordMovie(frameRate=6)

## Show 3D movie, by showing one volume that is moved by motion fields

# Load volume
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

# Load deformations (use backward mapping to deform texture 3D volume)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms_forward = [s['deform%i'%(i*10)] for i in range(10)]
deforms_forward = [[field[::2,::2,::2] for field in fields] for fields in deforms_forward]
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
dt.isoThreshold = 300
dt.renderStyle = 'iso'  # iso or mip work well
dt.SetDeforms(*[list(reversed(deform)) for deform in deforms_backward])
dt.colormap = {'g': [(0.0, 0.0), (0.33636364, 1.0)],
 'b': [(0.0, 0.0), (0.49545455, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)],
 'r': [(0.0, 0.0), (0.22272727, 1.0)]}

# Set limits and play!
a.SetLimits()
dt.MotionPlay(5, 0.6)  # (10, 0.2) = each 10 ms do a step of 20% ;(0.1,0.2)
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

# foo = recordMovie()