""" Manual landmark placement in volume MIP's
Select landmarks by SHIFT+RIGHTCLICK on a high intensity point
ls is returned that contains structs with graphs of selected landmark points 
(nodes in graph)
"""
import os
import scipy.io
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.stentdirect import stentgraph
from visvis import ssdf
from stentseg.utils.picker import pick3d, label2worldcoordinates
from stentseg.stentdirect import StentDirect, getDefaultParams
from stentseg.utils.visualization import DrawModelAxes
from lspeas.landmarks.landmarkselection import (saveLandmarkModel, LandmarkSelector, makeLandmarkModelDynamic)

# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_BACKUP',r'G:\LSPEAS_ssdf_BACKUP', 
                     r'E:\Maaike-Freija\LSPEAS_ssdf_backup',
                     r'D:\Maaike-Freija\LSPEAS_ssdf_backup',
                        )

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
                      r'D:\Profiles\koenradesma\Desktop',
                      )
                      
dirsave = select_dir(r'G:\LSPEAS_ssdf_toPC\landmarks',
                     r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\ssdf',
                     r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\ssdf')

# Select location .mat file
matsave = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\mat',
                        r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\mat')

# Select dataset to select landmarks
ptcode = 'LSPEASF_C_01' # 002_6months, 008_12months, 011_discharge, 17_1month, 20_discharge, 25_12months
ctcode = 'd'
cropname = 'ring' # use stent crops
what = 'avgreg' # 'phases' or 'avgreg'

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, what)


## =================Select landmarks in selected vol(s)=============
clim = (0,2500)
showVol = 'MIP' # MIP or ISO or 2D
# run LandmarkSelector, obtain ssdf ls with landmarkmodels for each volume that was analyzed
ls = LandmarkSelector(dirsave,ptcode,ctcode,cropname, s, what=what, clim=clim, showVol=showVol, axVis=True)


## ==================Show contents of ls==================
s_landmarks = ls.s_landmarks
if what == 'phases': 
    landmarks0 = s_landmarks.landmarks0
    print(landmarks0.nodes) # show positions of selected point in phase 0
else:
    landmarksavgreg = s_landmarks.landmarksavgreg
    print(landmarksavgreg.nodes)

## ================= Store landmark model to disk ==============
#saveLandmarkModel(ls, dirsave, ptcode, ctcode, cropname, what)


# =========== ALGORITHM PHASES / AVGREG ===============

## Load deformation fields avgreg to make landmark model dynamic

makeLandmarkModelDynamic(basedir, ptcode, ctcode, cropname, what='landmarksavgreg', savedir=dirsave)
   
