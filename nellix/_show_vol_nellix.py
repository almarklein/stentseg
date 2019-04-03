""" Show the volume in a dynamic way.
Maaike Koenrades (C) 2016
"""

import os
import pirt
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.visualization import show_ctvolume
from pirt.utils.deformvis import DeformableTexture3D
import scipy
from stentseg.utils import _utils_GUI
import copy
from stentseg.utils.picker import pick3d

# Select the ssdf basedir
basedir = select_dir(r'E:\Nellix_chevas\CT_SSDF\SSDF_automated',
                r'D:\Nellix_chevas_BACKUP\CT_SSDF\SSDF_automated')

# Select dataset to register
ptcode = 'chevas_09_thin'
ctcode, nr = '12months', 1
cropname = 'prox'

s0 = loadvol(basedir, ptcode, ctcode, cropname, 'phases')

# vol = s0.vol
vol = s0.vol20
key = 'vol20'
zscale = (s0[key].sampling[0] / s0[key].sampling[1]) # z / y
# resample vol using spline interpolation, 3rd order polynomial
vol_zoom = scipy.ndimage.interpolation.zoom(s0[key],[zscale,1,1],'float32') 
s0[key].sampling = [s0[key].sampling[1],s0[key].sampling[1],s0[key].sampling[2]]
# aanpassingen voor scale en origin
vol_zoom_type = vv.Aarray(vol_zoom, s0[key].sampling, s0[key].origin)
vol = vol_zoom_type

fig = vv.figure(); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00
clim = (0,2000)
# Show volume
a1 = vv.subplot(111)
a1.daspect = 1,1,-1
t = show_ctvolume(vol, axis=a1, showVol='MIP', clim =clim, isoTh=250, 
                removeStent=False, climEditor=True)
label = pick3d(vv.gca(), vol)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

## Show 3D movie, by alternating the 10 volumes

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, 'phases')
vols = []
for key in dir(s):
    if key.startswith('vol'):
        zscale = (s[key].sampling[0] / s[key].sampling[1]) # z / y
        # resample vol using spline interpolation, 3rd order polynomial
        vol_zoom = scipy.ndimage.interpolation.zoom(s[key],[zscale,1,1],'float32') 
        s[key].sampling = [s[key].sampling[1],s[key].sampling[1],s[key].sampling[2]]
        # aanpassingen voor scale en origin
        vol_zoom_type = vv.Aarray(vol_zoom, s[key].sampling, s[key].origin)
        vol = vol_zoom_type
        vols.append(vol)

# Start vis
f = vv.figure(3); vv.clf()
a = vv.gca()
a.daspect = 1, 1, -1
a.axis.axisColor = 1,1,1
a.axis.visible = True
a.bgcolor = 0,0,0
vv.title('ECG-gated CT scan Nellix %s  -  %s' % (ptcode[7:], ctcode))

# Setup data container
container = vv.MotionDataContainer(a)
showVol = 'mip'
for vol in vols:
    #     t = vv.volshow2(vol, clim=(-550, 500)) # -750, 1000
    t = vv.volshow(vol, clim=(-300, 2000), renderStyle = showVol)
    t.isoThreshold = 275               # iso or mip work well 
    t.parent = container
    if showVol == 'iso':
        t.colormap = {'g': [(0.0, 0.0), (0.33636364, 1.0)],
        'b': [(0.0, 0.0), (0.49545455, 1.0)],
        'a': [(0.0, 1.0), (1.0, 1.0)],
        'r': [(0.0, 0.0), (0.22272727, 1.0)]}

f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a]) )
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a]) )


## Show 3D movie, by showing one volume that is moved by motion fields

# Load volume
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol_org = copy.deepcopy(s.vol)
s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]] # z,y,x
vol = s.vol

# Load deformations (use backward mapping to deform texture 3D volume)
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
phases = []
for key in dir(s):
    if key.startswith('deform'):
        phases.append(key)
deforms_forward = [s[key] for key in phases]
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
vv.title('Model for Nellix %s  -  %s' % (ptcode[7:], ctcode))
vv.ColormapEditor(vv.gcf())

# Setup motion container
dt = DeformableTexture3D(a, vol)
dt.clim = -300, 2000
dt.isoThreshold = 300
dt.renderStyle = 'iso'  # iso or mip work well
dt.SetDeforms(*[list(reversed(deform)) for deform in deforms_backward])
dt.colormap = {'g': [(0.0, 0.0), (0.33636364, 1.0)],
 'b': [(0.0, 0.0), (0.49545455, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)],
 'r': [(0.0, 0.0), (0.22272727, 1.0)]}

# Set limits and play!
a.SetLimits()
dt.MotionPlay(5, 0.4)  # (10, 0.2) = each 10 ms do a step of 20% ;(0.1,0.2)
                        # With 85 bpm every beat 706 ms; 141 ms per 20%  

dt.motionSplineType = 'B-spline'
dt.motionAmplitude = 1.0

## Turn on/off axis
vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca()

switch = False

a1.axis.visible = switch
a2.axis.visible = switch

## Use same camera when 2 vols are running
a1.camera = a2.camera
