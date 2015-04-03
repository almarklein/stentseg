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
    print("NOTE: module reuses own official UIDs")
    print("----------------------------")
    
    # Select directory to save dicom
    basedir_save = r'D:\LSPEAS\DICOMavgreg'
    
    # Select directory to load dicom
    # the stentseg datahandling module is agnostic about where the DICOM data is
    dicom_basedir = select_dir(r'E:\LSPEAS_data\DICOM',
                            r'D:\LSPEAS\BACKUP CTdata\LSPEAS_data\DICOM',
                            '/home/almar/data/dicom/stent_LPEAS',)    
    # Select dataset
    ptcode = 'LSPEAS_003'
    ctcode = '1month'
    cropname = 'stent'
    
    # Select basedirectory to load ssdf
    basedir_load = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',)
    
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
    
#     base_filename1 = r'D:\LSPEAS\BACKUP CTdata\LSPEAS_data\DICOM\LSPEAS_002\LSPEAS_002_pre\1.2.392.200036.9116.2.6.1.48.1214833767.1398122835.442544\1.2.392.200036.9116.2.6.1.48.1214833767.1398124575.640714\1.2.392.200036.9116.2.6.1.48.1214833767.1398124590.20661.dcm'
    
#     base_filename2 = r'D:\LSPEAS\BACKUP CTdata\LSPEAS_data\DICOM\LSPEAS_002\LSPEAS_002_pre\1.2.392.200036.9116.2.6.1.48.1214833767.1398122835.442544\1.2.392.200036.9116.2.6.1.48.1214833767.1398124575.640714\1.2.392.200036.9116.2.6.1.48.1214833767.1398124902.95575.dcm'   
    
    ds = dicom.read_file(base_filename) # read original dicom file to get ds
    assert ds.InstanceNumber == 1 # first slice
    SliceLocationStart = ds.SliceLocation 
    instance = 0
    
#     ds2 = dicom.read_file(base_filename2)

#     print("Setting file meta information...")
#     # Populate required values for file meta information
#     file_meta = Dataset()
#     file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
#     file_meta.MediaStorageSOPInstanceUID = "1.2.3"  # !! Need valid UID here for real work
#     file_meta.ImplementationClassUID = "1.2.3.4"  # !!! Need valid UIDs here

    # rewrite slices to ds
    for slice in s.vol:
        ds.PixelData = slice.astype('int16').tostring() # must be int, not float
        ds.Rows = s.vol.shape[1]
        ds.Cols = ds.Columns = s.vol.shape[2]
        ds.SeriesDescription = cropname+' avgreg'
        # adjust slice z-position
        # note: originally the z-position is decreasing to match the "patient orientation"
        ds.ImagePositionPatient[2] = - s.vol.origin[0] - (instance * s.vol.sampling[0]) # z-flipped
        ds.ImagePositionPatient[1] = s.vol.origin[1]
        ds.ImagePositionPatient[0] = s.vol.origin[2]
        ds.InstanceNumber = instance
        ds.SliceLocation = SliceLocationStart + (instance * s.vol.sampling[0]) # todo: we do not need this?
        instance += 1
    
        # save ds
        filename = '%s_%s_%s_%s%i.dcm' % (ptcode, ctcode, cropname, 'avgreg', instance)
        print("Writing test file", filename)
        ds.save_as(os.path.join(basedir_save, ptcode, ptcode+'_'+ctcode, filename))
        print("File saved.")
    

#todo: did we correctly fix start position: relative to z position of first slice in cropped vol avgreg?




    
