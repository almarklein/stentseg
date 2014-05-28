
import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged, cropaveraged


# Select base directory for DICOM data

# The stentseg datahandling module is agnostic about where the DICOM data is
dicom_basedir = select_dir(r'C:\LSPEAS_data\DICOM',
                           '/home/almar/data/dicom/stent_LPEAS',)

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)


# Params Step A, B, C
CT1, CT2, CT3, CT4 = 'pre', 'discharge', '1month', '6months'  # preset
ctcode = CT2
ptcode = 'LSPEAS_003'

# Params Step B, C
stent, ring = 'stent', 'ring'  # preset
cropnames = stent, ring  # save crops of stent and/or ring
stenttype = 'anaconda'   # or 'endurant'
# C: start and end phase in cardiac cycle to average (50,90 = 5 phases)
phases = 30, 90

# todo: use fixed imageio instead
def readdcm(dirname):
    """ Function to read volumes while imageio suffers from "too may
    open files" bug.
    """
    
    if not os.path.isdir(dirname):
        raise RuntimeError('Could not find data for given input %s' % ptcode, ctcode)
        
    while True:
        subfolder = os.listdir(dirname)
        if len(subfolder) == 1:
            dirname = os.path.join(dirname, subfolder[0])
        else:
            break
    
    if not subfolder:
        raise RuntimeError('Could not find any files for given input %s' % ptcode, ctcode)
    
    vols = []
    for volnr in range(0,10): # 0,10 for all phases from 0 up to 90%
        vol = imageio.volread(os.path.join(dirname,subfolder[volnr]), 'dicom')
        vols.append(vol)
    #todo: Dit gaat niet gegarandeerd op de goede volgorde! Bij Almar niet namelijk
    # Check order of phases
    for i, vol in enumerate(vols):
        perc = '%i%%' % (i*10)
        assert perc in vol.meta.SeriesDescription
        
    return vols  


## Perform the steps A,B,C

# Step A: Read DICOM
vols = readdcm(os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode))
#vols = imageio.mvolread(os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode), 'dicom')

# Step B: Crop and Save SSDF
for cropname in cropnames:
    savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype)

# Step C: Save average of a number of volumes (phases in cardiac cycle)
for cropname in cropnames:
    saveaveraged(basedir, ptcode, ctcode, cropname, phases)


## Crop from averaged volume
cropaveraged(basedir, ptcode, ctcode, crop_in='stent', what='avg3090', crop_out= 'body')


## Test load ssdf and visualize

# Load one volume/phase from ssdf with phases
phase = 60
avg = 'avg5090'

s1 = loadvol(basedir, ptcode, ctcode, 'stent', what ='phases')
s2 = loadvol(basedir, ptcode, ctcode, 'body', avg)
s3 = loadvol(basedir, ptcode, ctcode, 'stent', 'avg3090')


## Visualize and compare

import visvis as vv
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00
a1 = vv.subplot(121)
# t = vv.volshow(s1['vol%i'% phase])
# t.clim = 0, 2500
t2 = vv.volshow(s3.vol)
t2.clim = 0, 1500
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')
vv.title('One volume at %i procent of cardiac cycle' % phase )

a2 = vv.subplot(122)
a2.daspect = 1,1,-1
t = vv.volshow(s2.vol)
t.clim = 0, 1500
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')
vv.title('Averaged volume %s' % avg ) 

# # Use same camera
a1.camera = a2.camera


## Visualize one volume

fig = vv.figure(2); vv.clf()
fig.position = 0, 22, 1366, 706
a = vv.gca()
a.cameraType = '3d'
a.daspect = 1,1,-1
a.daspectAuto = False
t = vv.volshow(vol)
t.clim = 0, 2500

