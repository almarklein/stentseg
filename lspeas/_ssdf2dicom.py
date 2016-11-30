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
                            r'G:\DICOMavgreg_toPC')
    
    # Select directory to load dicom
    # the stentseg datahandling module is agnostic about where the DICOM data is
    dicom_basedir = select_dir(r'G:\LSPEAS_data\ECGgatedCT',
                            r'D:\LSPEAS\LSPEAS_data_BACKUP\ECGgatedCT')    
    # Select dataset
    # ptcodes = ['LSPEAS_001','LSPEAS_002','LSPEAS_003','LSPEAS_005','LSPEAS_008',
    #         'LSPEAS_009','LSPEAS_011','LSPEAS_015','LSPEAS_017','LSPEAS_018',
    #         'LSPEAS_019','LSPEAS_020','LSPEAS_021','LSPEAS_022','LSPEAS_025', 
    #         'LSPEAS_023']
    ptcodes = ['LSPEAS_024']
    ctcode = '1month'
    cropname = 'stent'
    what = 'avgreg' # what volume to save to dicom
    normalizeLim = 3071 # HU
    
    for ptcode in ptcodes:
        if 'FANTOOM' in ptcode:
            dirname = os.path.join(dicom_basedir.replace('ECGgatedCT', ''), ptcode)
            subfolder = os.listdir(dirname)
            dirname = os.path.join(dirname, subfolder[0],ctcode)
        else:
            dirname = os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode)
        
        # Select basedirectory to load ssdf
        basedir_load = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                            r'D:\LSPEAS\LSPEAS_ssdf',
                            r'G:\LSPEAS_ssdf_backup')
        
        # Load ssdf
        s = loadvol(basedir_load, ptcode, ctcode, cropname, what)
        if s.vol.max() < 3100:
            vol = s.vol # no normalization
        else:
            vol = normalize_soft_limit(s.vol,normalizeLim) # normalize
        # Read dicom file to rewrite slices from s
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
        
    #     assert ds.InstanceNumber == 1 # first slice
        initialUID = ds.SOPInstanceUID
        UIDtoReplace = ds.SOPInstanceUID.split('.')[-1]
        instance = 0
        
        # Rewrite slices to ds
        for slice in vol:
            ds.PixelData = slice.astype('int16').tostring() # must be int, not float
            ds.Rows = s.vol.shape[1]
            ds.Cols = ds.Columns = s.vol.shape[2]
            ds.SeriesDescription = cropname+' '+what
            # adjust slice z-position
            # note: in world coordinates the z-position is decreasing to match the "patient orientation"
            ds.ImagePositionPatient[2] =  -(s.vol.origin[0] + (instance * s.vol.sampling[0])) # z-flipped
            ds.ImagePositionPatient[1] = s.vol.origin[1] # y
            ds.ImagePositionPatient[0] = s.vol.origin[2] # x
            ds.InstanceNumber = instance + 1 # start at 1
            ds.SOPInstanceUID = initialUID.replace(UIDtoReplace, str(int(UIDtoReplace)+instance)) # unique for each slice
            
            instance += 1
            
            # save ds
            filename = '%s_%s_%s_%s%i.dcm' % (ptcode, ctcode, cropname, what, instance)
            print("Writing slice", filename)
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
    
