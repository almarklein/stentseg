
import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged, cropaveraged


# Select base directory for LOADING DICOM data

# The stentseg datahandling module is agnostic about where the DICOM data is
dicom_basedir = select_dir(r'F:\LSPEAS_data\ECGgatedCT',
                            'D:\LSPEAS\LSPEAS_data_BACKUP\ECGgatedCT')

# Select the ssdf basedir for SAVING
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_toPC')

# Params Step A, B, C
ctcode = 'ZA2-100-1.35'  # 'pre', 'discharge', '1month', '6months', '12months', x_Profx_Water_
ptcode = 'QRM_FANTOOM_20160121'  # LSPEAS_00x or FANTOOM_xxx
stenttype = 'anaconda'         # or 'endurant' or 'excluder'
dicomStructure = 'imaFolders' # 'dcmFolders' or 'imaFolder' or 'imaFolders' - different output data structure

# Params Step B, C (to save)
cropnames = ['ring']    # save crops of stent and/or ring
# C: start and end phase in cardiac cycle to average (50,90=5 phases;60-20=7)
phases = 70, 20

# todo: use imageio.mvolread instead when fixed
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
        # check order of phases; gaat niet altijd goed namelijk, linux fout bv
        perc = '%i%%' % (volnr*10)
        assert perc in vol.meta.SeriesDescription
        
        vols.append(vol)
        assert vol.shape==vols[0].shape
    
    return vols  


## Perform the steps A,B,C

# Step A: Read DICOM
if dicomStructure == 'dcmFolders': #toshiba from workstation
    if 'FANTOOM' in ptcode:
        dicom_basedir = os.path.join(dicom_basedir.replace('ECGgatedCT', ''), ptcode)
        subfolder = os.listdir(dicom_basedir) # folder UID
        dicom_basedir = os.path.join(dicom_basedir, subfolder[0],ctcode)
        vols = readdcm(dicom_basedir)
    else:
        vols = readdcm(os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode))
#   vols = imageio.mvolread(os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode), 'dicom')

if dicomStructure == 'imaFolder': #siemens from workstation
    if 'FANTOOM' in ptcode:
        dicom_basedir = os.path.join(dicom_basedir.replace('ECGgatedCT', ''), ptcode)
        vols = imageio.mvolread(os.path.join(dicom_basedir,ctcode), 'dicom') # todo: error on memory >1GB of data after upgrade imageio
#         r = imageio.get_reader(os.path.join(dicom_basedir,ctcode,subfolder))
    else:
        dicom_basedir = os.path.join(dicom_basedir,ptcode,ptcode+'_'+ctcode)
        vols = imageio.mvolread(dicom_basedir, 'dicom')  # todo: error on memory >1GB of data  
    for i, vol in enumerate(vols):
        assert vol.shape == vols[0].shape
        assert str(i*10) in vol.meta.SeriesDescription # 0% , 10% etc.

if dicomStructure == 'imaFolders': #toshiba from synapse
    if 'FANTOOM' in ptcode:
        dicom_basedir = os.path.join(dicom_basedir.replace('ECGgatedCT', ''), ptcode)
    else:
        dicom_basedir = os.path.join(dicom_basedir,ptcode,ptcode+'_'+ctcode)    
    vols = []
    subfolders = os.listdir(os.path.join(dicom_basedir,ctcode))
    for i, subfolder in enumerate(subfolders): # read per folder to avoid PermissionError: [Errno 13] Permission denied with imageio.get_reader
        vol = imageio.volread(os.path.join(dicom_basedir,ctcode,subfolder), 'dicom') # error on memory>1GB of data with mvolread
        vols.append(vol)
        assert vol.shape == vols[0].shape
        assert str(i*10) in vol.meta.SeriesDescription # 0% , 10% etc.

# Step B: Crop and Save SSDF
for cropname in cropnames:
    savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype)

# Step C: Save average of a number of volumes (phases in cardiac cycle)
for cropname in cropnames:
    saveaveraged(basedir, ptcode, ctcode, cropname, phases)


## Additional: Crop from averaged volume
# cropaveraged(basedir, ptcode, ctcode, crop_in='stent', what='avg3090', crop_out= 'body')


## Test load ssdf and visualize

# Load one volume/phase from ssdf with phases
phase = 60
avg = 'avg7020'

# s1 = loadvol(basedir, ptcode, ctcode, 'ring', what ='phases')
s2 = loadvol(basedir, ptcode, ctcode, 'ring', avg)
s3 = loadvol(basedir, ptcode, ctcode, 'ring', avg)


# Visualize and compare

import visvis as vv
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00
a1 = vv.subplot(121)
# t = vv.volshow(s1['vol%i'% phase], clim=(0, 3000))
t2 = vv.volshow(s3.vol, clim=(0, 3000), renderStyle='iso')
t2.isoThreshold = 250
t2.colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}
s = vv.volshow2(s3.vol, clim=(-550, 500))
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')
# vv.title('One volume at %i procent of cardiac cycle' % phase )
vv.title('Averaged volume %s' % avg ) 


a2 = vv.subplot(122)
a2.daspect = 1,1,-1
# t = vv.volshow(s2.vol)
# t.clim = 0, 2500
t = vv.volshow(s2.vol, clim=(0, 3000), renderStyle='iso')
t.isoThreshold = 250
t.colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
 'g': [(0.0, 0.0), (0.27272728, 1.0)],
 'b': [(0.0, 0.0), (0.34545454, 1.0)],
 'a': [(0.0, 1.0), (1.0, 1.0)]}
s = vv.volshow2(s2.vol, clim=(-550, 500))
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')
vv.title('Averaged volume %s' % avg ) 

# # Use same camera
a1.camera = a2.camera


## Visualize one volume

# fig = vv.figure(2); vv.clf()
# fig.position = 0, 22, 1366, 706
# a = vv.gca()
# a.cameraType = '3d'
# a.daspect = 1,1,-1
# a.daspectAuto = False
# t = vv.volshow(s1.vol80)
# t.clim = 0, 3000

