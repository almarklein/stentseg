""" Manual landmark placement in volume MIP's
Select landmarks by SHIFT+RIGHTCLICK on a high intensity point
ls is returned that contains structs with graphs of selected landmark points 
(nodes in graph)
"""
import os
import scipy.io
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
import numpy as np
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.stentdirect import stentgraph
from visvis import ssdf
from stentseg.utils.picker import pick3d, label2worldcoordinates
from stentseg.stentdirect import StentDirect, getDefaultParams
from stentseg.utils.visualization import DrawModelAxes
from lspeas.landmarks.landmarkselection import (saveLandmarkModel, 
     LandmarkSelector, makeModelDynamic)

# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_BACKUP',r'G:\LSPEAS_ssdf_BACKUP', 
                     r'E:\Maaike-Freija\LSPEAS_ssdf_backup',
                     r'D:\Maaike-Freija\LSPEAS_ssdf_backup',
                     r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\ssdf',
                     r'C:\Users\Freija Geldof\Documents\M2.2 stage MST Vaatchirurgie\ssdf')

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
                      r'D:\Profiles\koenradesma\Desktop',
                      r'C:\Users\Freija Geldof\Documents\M2.2 stage MST Vaatchirurgie\Resultaten',
                      r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\Resultaten')

# Select save location ssdf file with landmark locations                     
dirsave = select_dir(r'G:\LSPEAS_ssdf_toPC\landmarks',
                     r'E:\Maaike-Freija\LSPEAS_landmarks\observer_maaike',
                     r'D:\Maaike-Freija\LSPEAS_landmarks\observer_maaike')

# Select location .mat file
matdir = select_dir(r'C:\Users\Freija Geldof\Documents\M2.2 stage MST Vaatchirurgie\Matlab\Observer_maaike',
                    r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\Matlab\Observer_maaike')

# Select dataset to register
ptcode = 'LSPEAS_020' # 002_6months, 008_12months, 011_discharge, 17_1month, 20_discharge, 25_12months
ctcode = 'discharge'
cropname = 'stent' # use stent crops
what = 'phases' # 'phases' or 'avgreg'

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, what)

## Select landmarks in selected vol(s)
clim = (0,2500)
showVol = 'MIP' # MIP or ISO or 2D
# run LandmarkSelector to obtain an ssdf called ls with landmarkmodels for each 
# volume that was analyzed
ls = LandmarkSelector(ptcode, s, what=what, clim=clim, showVol=showVol, axVis=True)

## Show contents of ls
s_landmarks = ls.s_landmarks
if what == 'phases': 
    landmarks0 = s_landmarks.landmarks0 # landmarks0.nodes returns selected points
    landmarks10 = s_landmarks.landmarks10
    landmarks20 = s_landmarks.landmarks20
else:
    landmarksavgreg = s_landmarks.landmarksavgreg
    print(landmarksavgreg.nodes)

## Store landmark model to disk
saveLandmarkModel(ls, dirsave, ptcode, ctcode, cropname, what)





# ALGORITHM / AVGREG

## Load deformation fields avgreg
# Load dynamic landmark model avgreg
#todo: change loadmodel() to work when there is not a folder per patient like now
fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarksavgreg')
s2 = ssdf.load(os.path.join(dirsave, fname))
# turn into graph model
landmarks = stentgraph.StentGraph()
landmarks.unpack(s2.landmarksavgreg)

# Make landmark model from avgreg dynamic with registration deformation fields
makeModelDynamic(basedir, ptcode, ctcode, cropname, what='landmarksavgreg', savedir=dirsave)

## Calculate real algorithm coordinates and export to Matlab   
# Load dynamic landmark model avgreg
#todo: change loadmodel() to work when there is not a folder per patient like now
fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarksavgreg')
s2 = ssdf.load(os.path.join(dirsave, fname))
# turn into graph model
landmarks = stentgraph.StentGraph()
landmarks.unpack(s2.landmarksavgreg)
landmarks.nodes()

# Calculate deforms of avgreg
deformslandmark = []
for i in range(0, 12):
    point = landmarks.nodes()[i]
    u = landmarks.node[point]
    deformslandmark.append(u)   
print(deformslandmark)

# Calculate algorithm coordinates with avgreg & deforms and get the ordering number
algorithmpoints = []
algorithmnumbers = []
for i in range(0, 12):
    pointavgreg = landmarks.nodes()[i]
    pointavgreg2 = np.tile(pointavgreg,(10,1))
    pointdeforms = landmarks.node[pointavgreg]
    pointdeforms2 = pointdeforms['deforms']
    q = np.add(pointavgreg2, pointdeforms2)
    algorithmpoints.append(q)
    pointnumber = pointdeforms['number']
    algorithmnumbers.append(pointnumber)
print(algorithmpoints)
print(algorithmnumbers)

# Store algorithm coordinates to mat-file for data analysis in Matlab
mname = 'algorithmpoints_%s_%s.mat' % (ptcode, ctcode)
scipy.io.savemat(os.path.join(matdir,mname), mdict={'algorithmpoints': algorithmpoints})
mname2 = 'algorithmnumbers_%s_%s.mat' % (ptcode, ctcode)
scipy.io.savemat(os.path.join(matdir,mname2), mdict={'algorithmnumbers': algorithmnumbers})
print('---algorithmpoints, algorithmnumbers stored to .mat {} ---')


## Store algorithm coordinates to excel
import xlsxwriter
def storeCoordinatesToExcel(algorithmpoints, exceldir, graph):
    """Create file and add a worksheet or overwrite existing
    mind that floats can not be stored with write_row.
    Input = pp1, sorted array of picked points (coordinates)
    """
    rowoffset=0
    # https://pypi.python.org/pypi/XlsxWriter
    fname2 = 'Output_%s_%s_%s.xlsx' % (ptcode, ctcode, what)
    workbook = xlsxwriter.Workbook(os.path.join(exceldir, fname2))
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('A:E', 20)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    # write title
    worksheet.write('A1', 'Point coordinates', bold)
    analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
    worksheet.write('B3', analysisID, bold)
    # write 'picked coordinates'
    for x in algorithmpoints:
        for y, Output in enumerate(x):
            worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
            #worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
            rowoffset += 1
        rowoffset += 1

        
    workbook.close()

storeCoordinatesToExcel(algorithmpoints,exceldir,landmarks)
print('---stored to excel {} ---'.format(exceldir) )
print('---model can be stored as ssdf in do_segmentation---')





# LANDMARKS / PHASES

## Export identified landmark coordinates to Matlab   
# Load dynamic landmark model phases
#todo: change loadmodel() to work when there is not a folder per patient like now
fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarksphases')
s2 = ssdf.load(os.path.join(dirsave, fname))
# turn into graph model
landmarkpoints = []
for key in dir(s2):
    if key.startswith('landmarks'):
        landmarks = stentgraph.StentGraph()
        landmarks.unpack(s2[key])
        p = landmarks.nodes()
        p.sort(key=lambda x: landmarks.node[x]['number']) 
        p = np.asarray(p)
        landmarkpoints.append(p)

# Store algorithm coordinates to mat-file for data analysis in Matlab
mname3 = 'landmarkpoints_%s_%s.mat' % (ptcode, ctcode)
scipy.io.savemat(os.path.join(matdir,mname3), mdict={'landmarkpoints': landmarkpoints})
print('---landmarkpoints stored to .mat {} ---')

## Store landmark coordinates to excel
import xlsxwriter
def storeCoordinatesToExcel(landmarkpoints, exceldir, graph):
    """Create file and add a worksheet or overwrite existing
    mind that floats can not be stored with write_row.
    Input = pp1, sorted array of picked points (coordinates)
    """
    rowoffset=0
    # https://pypi.python.org/pypi/XlsxWriter
    fname2 = 'Output_%s_%s_%s.xlsx' % (ptcode, ctcode, what)
    workbook = xlsxwriter.Workbook(os.path.join(exceldir, fname2))
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('A:E', 20)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    # write title
    worksheet.write('A1', 'Point coordinates', bold)
    analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
    worksheet.write('B3', analysisID, bold)
    # write 'picked coordinates'
    for x in landmarkpoints:
        for y, Output in enumerate(x):
            worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
            #worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
            rowoffset += 1
        rowoffset += 1

    workbook.close()


storeCoordinatesToExcel(landmarkpoints,exceldir,landmarks)
print('---stored to excel {} ---'.format(exceldir) )
print('---model can be stored as ssdf in do_segmentation---')