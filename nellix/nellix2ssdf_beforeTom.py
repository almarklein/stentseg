import os
import sys

import imageio

from stentseg.utils.datahandling import loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged


# Select base directory for LOADING DICOM data
dicom_basedir = r'D:\LSPEAS\Nellix_chevas\DICOM\chevas_01_12months\10phases'

ptcode = 'chevas_01' 
ctcode = '12months'  # 'pre', 'post_x', '12months'
stenttype = 'nellix'      

# Select base directory to SAVE SSDF
basedir = r'D:\LSPEAS\Nellix_chevas\CHEVAS_SSDF'

# Set which crops to save
cropnames = ['prox','stent'] # ['ring'] or ['ring','stent'] or ..

## Step A: read single volumes to get vols:
folder1 = '0000DA00'
folder2 = '00003E1A'
vol1 = imageio.volread(os.path.join(dicom_basedir, folder1), 'dicom')
vol2 = imageio.volread(os.path.join(dicom_basedir, folder2), 'dicom')
print(  )

if vol1.meta.SeriesDescription[:2] < vol2.meta.SeriesDescription[:2]:
    vols4078 = [vol1,vol2]
else:
    vols4078 = [vol2,vol1]
    
vols = vols4078.copy()

for vol in vols:
    vol.meta.PatientName = ptcode # anonimyze
    vol.meta.PatientID = 'anonymous'
    print(vol.meta.SeriesDescription,'-', vol.meta.sampling)

## Step A: read 10 volumes to get vols

vols2 = [vol2 for vol2 in imageio.get_reader(dicom_basedir, 'DICOM', 'V')]
vols = [None] * 10
for i, vol in enumerate(vols2):
    print(vol.meta.sampling)
    phase = int(vol.meta.SeriesDescription[:1])
    vols[phase] = vol
for j, vol in enumerate(vols):
    print(vol.meta.SeriesDescription)
    assert vol.shape == vols[0].shape
    assert str(j*10) in vol.meta.SeriesDescription # 0% , 10% etc. 

## Step B: Crop and Save SSDF
for cropname in cropnames:
    savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype)


## Visualize result

s1 = loadvol(basedir, ptcode, ctcode, cropnames[0], what ='10phases')
vol1 = s1.vol40
vol2 = s1.vol90

# Visualize and compare
colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}
 
import visvis as vv
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
a1 = vv.subplot(121)
t1 = vv.volshow(vol1, clim=(0, 3000), renderStyle='iso') # iso or mip
t1.isoThreshold = 400
t1.colormap = colormap
a1b = vv.volshow2(vol1, clim=(-550, 500))
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
# vv.title('One volume at %i procent of cardiac cycle' % phase )
vv.title('Vol40' )

a2 = vv.subplot(122)
a2.daspect = 1,1,-1
t2 = vv.volshow(vol2, clim=(0, 3000), renderStyle='iso') # iso or mip
t2.isoThreshold = 400
t2.colormap = colormap
a2b = vv.volshow2(vol2, clim=(-550, 500))
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
# vv.title('One volume at %i procent of cardiac cycle' % phase )
vv.title('Vol78' )  

a1.camera = a2.camera




