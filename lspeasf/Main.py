""" Main script to do motion analysis for LSPEAS-F study

* displacement centerlines (e.g. branches, aorta)
* angulation branches/end-stent angle
* distance change between branch proximally and aorta
*

"""

import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol


ptcode = 'LSPEASF_C_01'
ctcode = 'pre' # pre, dis, 6w, 12m

# Set basedir for saving and loading ssdf
basedir = select_dir(r'D:\LSPEAS_F\LSPEASF_ssdf', r'F:\LSPEASF_backup\LSPEASF_ssdf')

## LOAD CT
if True:
        from lspeasf.utils.dicom2ssdf import dicom2ssdf
        
        dicom_basedir = r'F:\LSPEASF_backup\LSPEASF_CT_dicom_backup\LSPEASF_C_01\LSPEASF_C_01_pre'
        print(dicom_basedir)
        
        dicom2ssdf(dicom_basedir,ptcode,ctcode,basedir)
        print('load, crop and convert to ssdf: done')


## REGISTRATION