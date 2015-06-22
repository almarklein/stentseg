"""Writing DICOM from vol in ssdf using pydicom.

Module saves 'avgreg' volume to dicom (.dcm) files
"""

from __future__ import print_function

import sys
import os.path
import dicom
from dicom.dataset import Dataset, FileDataset
import dicom.UID
from stentseg.utils.datahandling import select_dir, loadvol

if __name__ == "__main__":
    print("---------------------------- ")
    print("Write ssdf to dicom")
    print("----------------------------")
    print("Based on PyDicom demo code")
    print("See http://pydicom.googlecode.com")
    print("and https://code.google.com/p/pydicom/wiki/GettingStarted")
    print("NOTE: module reuses own official UIDs")
    print("----------------------------")
    
    # Select directory to save dicom
    basedir_save = select_dir(r'D:\LSPEAS\DICOMavgreg',
                            r'F:\DICOMavgreg_toPC')
    
    # Select directory to load dicom
    # the stentseg datahandling module is agnostic about where the DICOM data is
    dicom_basedir = select_dir(r'F:\LSPEAS_data\DICOM',
                            r'D:\LSPEAS\LSPEAS_data_BACKUP\DICOM',
                            '/home/almar/data/dicom/stent_LPEAS',)    
    # Select dataset
    ptcode = 'LSPEAS_001'
    ctcode = 'pre'
    cropname = 'ring'
    
    # Select basedirectory to load ssdf
    basedir_load = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup')
    
    # Load ssdf
    s = loadvol(basedir_load, ptcode, ctcode, cropname, 'avgreg')
    vol = s.vol
    
    # Read dicom file to rewrite slices from s
    dirname = os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode)
    
    if not os.path.isdir(dirname):
        raise RuntimeError('Could not find data for given input %s' % ptcode, ctcode)
    
    while True:
        subfolder = os.listdir(dirname)
        if len(subfolder) == 1:  # data should only contain one main uid folder
            dirname = os.path.join(dirname, subfolder[0])
        else:
            break
    
    if not subfolder:
        raise RuntimeError('Could not find any files for given input %s' % ptcode, ctcode)
    # get dir of first subfolder (phase 0%)
    dirsubfolder = os.path.join(dirname,subfolder[0])
    # get first .dcm file
    for filename in os.listdir(dirsubfolder):
        if 'dcm' in filename:
            base_filename = os.path.join(dirsubfolder,filename)
            break # for loop
    
    ds = dicom.read_file(base_filename) # read original dicom file to get ds
#     assert ds.InstanceNumber == 1 # first slice
    instance = 0

    # Rewrite slices to ds
    for slice in s.vol:
        ds.PixelData = slice.astype('int16').tostring() # must be int, not float
        ds.Rows = s.vol.shape[1]
        ds.Cols = ds.Columns = s.vol.shape[2]
        ds.SeriesDescription = cropname+' avgreg'
        # adjust slice z-position
        # note: in world coordinates the z-position is decreasing to match the "patient orientation"
        ds.ImagePositionPatient[2] =  -(s.vol.origin[0] + (instance * s.vol.sampling[0])) # z-flipped
        ds.ImagePositionPatient[1] = s.vol.origin[1] # y
        ds.ImagePositionPatient[0] = s.vol.origin[2] # x
        ds.InstanceNumber = instance + 1 # start at 1
    
        # save ds
        filename = '%s_%s_%s_%s%i.dcm' % (ptcode, ctcode, cropname, 'avgreg', instance)
        print("Writing test file", filename)
        ds.save_as(os.path.join(basedir_save, ptcode, ptcode+'_'+ctcode, filename))
        print("Files saved to:")
        print(os.path.join(basedir_save, ptcode, ptcode+'_'+ctcode))
        
        instance += 1
