import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged, cropaveraged

dicom_basedir = r'D:\Profiles\koenradesma\Desktop\A02d_arterieel_veneus\arterieel'
vols2 = [vol2 for vol2 in imageio.get_reader(dicom_basedir, 'DICOM', 'V')]

for i, vol in enumerate(vols2):
    print(vol.meta.ImagePositionPatient)
for i, vol in enumerate(vols2):
    print(vol.shape)
for i, vol in enumerate(vols2):
    print(vol.meta.AcquisitionTime)
    print(vol.meta.sampling)
    assert vol.shape == vols2[0].shape
    assert vol.meta.SeriesTime == vols2[0].meta.SeriesTime
    

## Test show vol
from lspeas.utils.vis import showVolPhases

showVol='mip'
t = showVolPhases(dicom_basedir, vols2, showVol=showVol, 
        mipIsocolor=True, isoTh=310, clim=(60,3000), slider=True  )