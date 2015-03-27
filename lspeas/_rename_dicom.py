"""Change filenames 4D CT dicom files (.dcm)

"""

import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import renamedcm


# Select base directory for DICOM data

# The stentseg datahandling module is agnostic about where the DICOM data is
# dicom_basedir = select_dir(r'E:\LSPEAS_data\DICOM',
#                            '/home/almar/data/dicom/stent_LPEAS',)
dicom_basedir = r'D:\LSPEAS\test_rename_dicom'

# Select dataset
ptcode = 'LSPEAS_002'
ctcode = '1month'


# Rename dcm files for all 10 phases
renamedcm(dicom_basedir, ptcode, ctcode)



