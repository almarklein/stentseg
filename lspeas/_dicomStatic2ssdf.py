""" Load static CT data and save to ssdf format

"""

import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged

# Select base directory for LOADING DICOM data

# The stentseg datahandling module is agnostic about where the DICOM data is
dicom_basedir = select_dir(r'G:\LSPEAS_data', r'D:\LSPEAS\LSPEAS_data_BACKUP')

# Select base directory to SAVE SSDF
ssdf_basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                          r'F:\LSPEAS_ssdf_toPC', r'G:\LSPEAS_ssdf_toPC')

ptcode = 'LSPEAS_022_CTA_20160126' #
ctcode = 'S40'  # 'CTA_pre', 'static..'
stenttype = 'anaconda'   

# Set which crops to save
cropnames = ['ring'] # ['prox'] or ['ring','stent'] or ..

# Step A: read single volume
if ctcode == 'CTA_pre':
    vol = imageio.volread(os.path.join(dicom_basedir, ctcode, ptcode), 'dicom')
else:
    vol = imageio.volread(os.path.join(dicom_basedir, ptcode, ctcode), 'dicom')
print ()
print(vol.meta.SeriesDescription)
print(vol.meta.sampling)
print()

# Step B: Crop and Save SSDF
vols = [vol]
for cropname in cropnames:
    savecropvols(vols, ssdf_basedir, ptcode, ctcode, cropname, stenttype)


## Visualize result

s1 = loadvol(ssdf_basedir, ptcode, ctcode, cropnames[0], what ='phase')
vol = s1.vol

# Visualize and compare
colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}
 
import visvis as vv
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
a = vv.gca()
a.daspect = 1,1,-1
t1 = vv.volshow(vol, clim=(0, 2500), renderStyle='iso') # iso or mip
t1.isoThreshold = 400
t1.colormap = colormap
t2 = vv.volshow2(vol, clim=(-550, 500))
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
# vv.title('One volume at %i procent of cardiac cycle' % phase )
vv.title('Static CT Volume' )
