""" Test in silico CT deformation

"""
import os
from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes
import visvis as vv
from stentseg.utils import _utils_GUI

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_008'
ctcode = '1month'
cropname = 'stent'
what = 'phases' # avgreg

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol80

clim = (0,2500)
showVol = '2D'

fig = vv.figure(2); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00

label = DrawModelAxes(vol, clim=clim, showVol=showVol) # lc, mc
a = vv.gca()

# bind rotate view [a,d rotate; z,x axes]
fig.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a]) )

