""" Export and or analyze extracted landmark trajectories

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

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
                      r'D:\Profiles\koenradesma\Desktop',
                      )
# Location of ssdf landmarks                      
dirsave = select_dir(r'G:\LSPEAS_ssdf_toPC\landmarks',
                     r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\ssdf',
                     r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\ssdf')

# Select location .mat file
matsave = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\mat',
            r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\phantom_article\mat')

# Select dataset to select landmarks
ptcode = 'LSPEAS_020' # 002_6months, 008_12months, 011_discharge, 17_1month, 20_discharge, 25_12months
ctcode = 'discharge'
cropname = 'stent' # use stent crops

observer = 'obs1'
# =========== ALGORITHM PHASES / AVGREG ===============

## Load dynamic landmark model and export to mat file    
s2 = loadmodel(os.path.join(dirsave,observer), ptcode, ctcode, cropname, 'landmarksavgreg')

landmarks = s2.landmarksavgreg

# # landmarks with deforms
# deformslandmark = []
# for i in range(len(landmarks.nodes())):
#     point = landmarks.nodes()[i]
#     u = landmarks.node[point]
#     deformslandmark.append(u)   
# print(deformslandmark)

# Calculate algorithm coordinates with avgreg & deforms
algorithmpoints = []
algorithmnumbers =[]
algtranslationx = []
algtranslationy = []
algtranslationz = []
for i in range(len(landmarks.nodes())):
    pointavgreg = landmarks.nodes()[i]
    pointavgreg2 = np.tile(pointavgreg,(10,1))
    pointdeforms = landmarks.node[pointavgreg]
    pointdeforms2 = pointdeforms['deforms']
    q = np.add(pointavgreg2, pointdeforms2)
    algorithmpoints.append(q)
    pointnumber = pointdeforms['number']
    algorithmnumbers.append(pointnumber)
    algtranslationx.append(max(pointdeforms2[:,0]) - min(pointdeforms2[:,0]) )
    algtranslationy.append(max(pointdeforms2[:,1]) - min(pointdeforms2[:,1]) )
    algtranslationz.append(max(pointdeforms2[:,2]) - min(pointdeforms2[:,2]) )
print(algorithmpoints)
print(algorithmnumbers)
print('')

algtranslationx = [algtranslationx[i] for i in algorithmnumbers] #todo: order?

print(algtranslationx)
print(algtranslationy)
print(algtranslationz)
print('')
print(np.mean(algtranslationx))
print(np.mean(algtranslationy))
print(np.mean(algtranslationz))

# todo: fix order so we don t need mean but can compare points individual

## Store algorithm coordinates to mat-file for data analysis in Matlab
if False:
    mname = 'algorithmpoints_%s_%s.mat' % (ptcode, ctcode)
    scipy.io.savemat(os.path.join(matsave,mname), mdict={'algorithmpoints': algorithmpoints})
    mname2 = 'algorithmnumbers_%s_%s.mat' % (ptcode, ctcode)
    scipy.io.savemat(os.path.join(matsave,mname2), mdict={'algorithmnumbers': algorithmnumbers})
    print('---algorithmpoints, algorithmnumbers stored to .mat {} ---')




# ========= OBSERVER PHASES =============

## Export identified landmark coordinates to Matlab   
# Load dynamic landmark model phases
s2 = loadmodel(os.path.join(dirsave,observer), ptcode, ctcode, cropname, 'landmarksphases')

# turn into graph model
landmarkpoints = []
for key in dir(s2):
    if key.startswith('landmarks'): # each phase
        landmarks = s2[key]
        p = landmarks.nodes()
        p.sort(key=lambda x: landmarks.node[x]['number']) 
        p = np.asarray(p)
        landmarkpoints.append(p)
        
# Store algorithm coordinates to mat-file for data analysis in Matlab
if False:
    mname3 = 'landmarkpoints_%s_%s.mat' % (ptcode, ctcode)
    scipy.io.savemat(os.path.join(matsave,mname3), mdict={'landmarkpoints': landmarkpoints})
    print('---landmarkpoints stored to .mat in {} ---'.format(os.path.join(matsave,mname3)))

landmarkpoints = np.asarray(landmarkpoints)
obstranslationx = []
obstranslationy = []
obstranslationz = []
for i in range(landmarkpoints.shape[1]): # for each point
    xpositionphases = landmarkpoints[:,i,0] # all phases get x for one point
    ypositionphases = landmarkpoints[:,i,1] # all phases get y for one point
    zpositionphases = landmarkpoints[:,i,2] # all phases get z for one point
    xtranslation = max(xpositionphases) - min(xpositionphases)
    ytranslation = max(ypositionphases) - min(ypositionphases)
    ztranslation = max(zpositionphases) - min(zpositionphases)
    obstranslationx.append(xtranslation)
    obstranslationy.append(ytranslation)
    obstranslationz.append(ztranslation)

print('Observer landmark trajectories')
print(obstranslationx)
print(obstranslationy)
print(obstranslationz)
print('')
print(np.mean(obstranslationx))
print(np.mean(obstranslationy))
print(np.mean(obstranslationz))





# ## Store algorithm coordinates to excel
# import xlsxwriter
# def storeCoordinatesToExcel(algorithmpoints, exceldir, graph):
#     """Create file and add a worksheet or overwrite existing
#     mind that floats can not be stored with write_row.
#     Input = pp1, sorted array of picked points (coordinates)
#     """
#     rowoffset=0
#     # https://pypi.python.org/pypi/XlsxWriter
#     fname2 = 'Output_%s_%s_%s.xlsx' % (ptcode, ctcode, what)
#     workbook = xlsxwriter.Workbook(os.path.join(exceldir, fname2))
#     worksheet = workbook.add_worksheet()
#     # set column width
#     worksheet.set_column('A:E', 20)
#     # add a bold format to highlight cells
#     bold = workbook.add_format({'bold': True})
#     # write title
#     worksheet.write('A1', 'Point coordinates', bold)
#     analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
#     worksheet.write('B3', analysisID, bold)
#     # write 'picked coordinates'
#     for x in algorithmpoints:
#         for y, Output in enumerate(x):
#             worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
#             #worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
#             rowoffset += 1
#         rowoffset += 1
# 
#         # if (rowoffset % 2 == 0): # odd
#         #     rowoffset += 4 # add rowspace for next points
# 
#     # Store screenshot
#     #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
#     workbook.close()
# 
# 
# storeCoordinatesToExcel(algorithmpoints,exceldir,landmarks)
# print('---stored to excel {} ---'.format(exceldir) )
# print('---model can be stored as ssdf in do_segmentation---')
# 
# ## Store landmark coordinates to excel
# import xlsxwriter
# def storeCoordinatesToExcel(landmarkpoints, exceldir, graph):
#     """Create file and add a worksheet or overwrite existing
#     mind that floats can not be stored with write_row.
#     Input = pp1, sorted array of picked points (coordinates)
#     """
#     rowoffset=0
#     # https://pypi.python.org/pypi/XlsxWriter
#     fname2 = 'Output_%s_%s_%s.xlsx' % (ptcode, ctcode, what)
#     workbook = xlsxwriter.Workbook(os.path.join(exceldir, fname2))
#     worksheet = workbook.add_worksheet()
#     # set column width
#     worksheet.set_column('A:E', 20)
#     # add a bold format to highlight cells
#     bold = workbook.add_format({'bold': True})
#     # write title
#     worksheet.write('A1', 'Point coordinates', bold)
#     analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
#     worksheet.write('B3', analysisID, bold)
#     # write 'picked coordinates'
#     for x in landmarkpoints:
#         for y, Output in enumerate(x):
#             worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
#             #worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
#             rowoffset += 1
#         rowoffset += 1
# 
#         # if (rowoffset % 2 == 0): # odd
#         #     rowoffset += 4 # add rowspace for next points
# 
#     # Store screenshot
#     #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
#     workbook.close()
# 
# 
# storeCoordinatesToExcel(landmarkpoints,exceldir,landmarks)
# print('---stored to excel {} ---'.format(exceldir) )
# print('---model can be stored as ssdf in do_segmentation---')
