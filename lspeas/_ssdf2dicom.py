"""Writing DICOM from vol in ssdf using pydicom.

Module saves 'avgreg' volume to dicom (.dcm) files
"""

from __future__ import print_function

import sys
import os.path
import dicom
from dicom.dataset import Dataset, FileDataset
import dicom.UID
from stentseg.utils.datahandling import select_dir,loadvol,normalize_soft_limit
import numpy as np

if __name__ == "__main__":
    print("---------------------------- ")
    print("Write ssdf to dicom")
    print("----------------------------")
    print("Based on PyDicom demo code")
    print("See http://pydicom.googlecode.com")
    print("and https://code.google.com/p/pydicom/wiki/GettingStarted")
    print("NOTE: module reuses own official UIDs")
    print("----------------------------")
    
    ## Select directory to save dicom
    basedir_save = select_dir(r'D:\LSPEAS\DICOMavgreg',
                            r'F:\DICOMavgreg_toPC')
    # basedir_save = select_dir(r'D:\LSPEAS_F\DICOMavgreg')
    
    ## Select directory to load dicom
    dicom_basedir = select_dir(r'F:\LSPEAS_data\ECGgatedCT',
                            r'D:\LSPEAS\LSPEAS_data_BACKUP\ECGgatedCT')
    # dicom_basedir = select_dir(r'D:\LSPEAS_F\CT_dicom_backup')
    
    # *** or set a slice location manual ***
    # dicomfolder = r''
    dicomfile = 'LSPEAS_FEVAR001_PO_1.CT.0004.0001.2017.02.08.10.07.57.428321.447272392.IMA'
    manualdicomslice = False
    
    ## Select basedirectory to load ssdf
    basedir_load = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup')
                        
    basedir_load = select_dir(r'D:\LSPEAS_F\LSPEASF_ssdf')
    
    ## Select dataset
    # ptcodes = ['LSPEAS_001','LSPEAS_002','LSPEAS_003','LSPEAS_005','LSPEAS_008',
    #         'LSPEAS_009','LSPEAS_011','LSPEAS_015','LSPEAS_017','LSPEAS_018',
    #         'LSPEAS_019','LSPEAS_020','LSPEAS_021','LSPEAS_022']
    ptcodes = ['LSPEAS_023']
    ctcode = 'pre'
    cropname = 'stent'
    what = 'avgreg' # what volume to save to dicom
    normalizeLim = 3071 # HU
    
    ## loop through ptcodes
    for ptcode in ptcodes:
        # Load ssdf
        s = loadvol(basedir_load, ptcode, ctcode, cropname, what)
        if s.vol.max() < 3500:
            vol = s.vol # no normalization
            print('no normalization')
        else:
            vol = normalize_soft_limit(s.vol,normalizeLim) # normalize
            print('we normalized')
        
        ## -------- Read dicom file to rewrite slices from s --------
        if manualdicomslice:
            ds = dicom.read_file(os.path.join(dicomfolder,dicomfile)) # read original dicom file to get ds
            mainuid = ds.StudyInstanceUID
        else:
            if 'FANTOOM' in ptcode:
                dirname = os.path.join(dicom_basedir.replace('ECGgatedCT', ''), ptcode)
                # subfolder = os.listdir(dirname)
                dirname = os.path.join(dirname, ctcode)
            else:
                dirname = os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode)
            
            if not os.path.isdir(dirname):
                raise RuntimeError('Could not find directory for given input %s' % ptcode, ctcode, cropname, what)
            
            while True:
                subfolder = os.listdir(dirname)
                if len(subfolder) == 1:  # data should only contain one main uid folder
                    dirname = os.path.join(dirname, subfolder[0])
                    mainuid = subfolder.copy()[0]
                else: # we are in the folder with folders for each phase
                    break
            
            # get dir of first subfolder (phase 0%)
            try:
                dirsubfolder = os.path.join(dirname,subfolder[0]) # error when not folder
                # get first .dcm file
                for filename in os.listdir(dirsubfolder):
                    if 'dcm' in filename:
                        base_filename = os.path.join(dirsubfolder,filename)
                        ds = dicom.read_file(base_filename) # read original dicom file to get ds
                        break # leave for loop, we have the first dicom file
            except NotADirectoryError: # when we have imafolders with dirfile (S10, S20..) or only ima files no folder
                for folder in subfolder:
                    if folder == 'DIRFILE':
                        continue
                    if folder.endswith('.IMA'): # IMA files are not in folder
                        base_filename = os.path.join(dirname,folder)
                        ds = dicom.read_file(base_filename) # read original dicom file to get ds
                        mainuid = ds.StudyInstanceUID
                        break # leave for loop, we have ima dicom file
                    else: # get right folder with phase
                        subdir = os.path.join(dirname,folder)
                        numOfFiles = len(os.listdir(subdir))
                        if numOfFiles > 250: # gated phase with 0.5 mm spacing at least 250 slices 
                            filename = os.listdir(subdir)[1]
                            base_filename = os.path.join(subdir,filename)
                            ds = dicom.read_file(base_filename) # read original dicom file to get ds
                            mainuid = ds.StudyInstanceUID
                            break # leave for loop, we have ima dicom file
        # ------------------------------------------------------
    
        ## --------------- Get ds keys for rewriting ----------
        initialUID = ds.SOPInstanceUID
        UIDtoReplace = ds.SOPInstanceUID.split('.')[-1]
        instance = 0
        
        # Rewrite slices to ds
        for slice in vol:
            # ds.BitsStored = 16
            # ds.HighBit = 15
            # ds.BitsAllocated = 16
            # ds.SmallestImagePixelValue = int(vol.min())
            # ds.LargestImagePixelValue = int(vol.max())
            # ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.pixel_array = slice.astype(np.uint16)
            # ds.PixelData = slice.astype(np.uint16).tostring() # must be int, not float
            ds.PixelData = slice.astype('int16').tostring() # must be int, not float
            ds.PixelSpacing = [dicom.valuerep.DSfloat(s.vol.sampling[1]), dicom.valuerep.DSfloat(s.vol.sampling[2])]
            ds.Rows = s.vol.shape[1]
            ds.Cols = ds.Columns = s.vol.shape[2]
            ds.SeriesDescription = cropname+' '+what+' '+ctcode
            # adjust slice z-position
            # note: in world coordinates the z-position is decreasing to match the "patient orientation"
            ds.ImagePositionPatient[2] =  -(s.vol.origin[0] + (instance * s.vol.sampling[0])) # z-flipped
            ds.ImagePositionPatient[1] = s.vol.origin[1] # y
            ds.ImagePositionPatient[0] = s.vol.origin[2] # x
            ds.InstanceNumber = instance + 1 # start at 1
            ds.SOPInstanceUID = initialUID.replace(UIDtoReplace, str(int(UIDtoReplace)+instance)) # unique for each slice
            ds.StudyDate, ds.SeriesDate, ds.AcquisitionDate = s.meta0.AcquisitionDate,s.meta0.AcquisitionDate,s.meta0.AcquisitionDate
            ds.SeriesTime = s.meta0.SeriesTime
            ds.AcquisitionTime = s.meta0.AcquisitionTime
            ds.Manufacturer = s.meta0.Manufacturer
            ds.ReferringPhysicianName = ''
            
            instance += 1
            
            # save ds
            filename = '%s_%s_%s_%s%i.dcm' % (ptcode, ctcode, cropname, what, instance)
            # print("Writing slice", filename)
            if 'FANTOOM' in ptcode:
                targetdir = os.path.join(basedir_save, cropname, ptcode, ctcode, mainuid)
            else:
                targetdir = os.path.join(basedir_save, cropname, ptcode, ptcode+'_'+ctcode, mainuid)
            try:
                ds.save_as(os.path.join(targetdir, filename))
            except FileNotFoundError: # if targetdir does not exist, create
                os.makedirs(targetdir)
                ds.save_as(os.path.join(targetdir, filename))
        
        print("Check: shape of {} vol=".format(what), vol.shape)
        if not vol.shape[0] == instance:
            raise RuntimeError('Number of slices in s.vol does not agree with written dicom')
        print("Max HU s.vol= {}; max HU of stored dicom volume= {}".format(s.vol.max(),vol.max()))
        print("Files saved to:")
        print(targetdir)
    
