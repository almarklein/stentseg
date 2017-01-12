""" Manual landmark placement in volume MIP's
Select landmarks by SHIFT+RIGHTCLICK on a high intensity point
ls is returned that contains structs with graphs of selected landmark points 
(nodes in graph)
"""
import os
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
                     r'D:\Maaike-Freija\LSPEAS_ssdf_backup',
                     r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\ssdf')

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
                      r'D:\Profiles\koenradesma\Desktop',
                      r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\Resultaten')
                      
dirsave = select_dir(r'G:\LSPEAS_ssdf_toPC\landmarks',
                     r'D:\Maaike-Freija\LSPEAS_landmarks\observer_freija')

# Select dataset to register
ptcode = 'LSPEAS_020'
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
#todo: create button?


## Make landmark model from avgreg dynamic with registration deformation fields
makeModelDynamic(basedir, ptcode, ctcode, cropname, what='landmarksavgreg', savedir=dirsave)
    
                 
## Load dynamic landmark model avgreg
#todo: change loadmodel() to work when there is not a folder per patient like now
fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarksavgreg')
s2 = ssdf.load(os.path.join(dirsave, fname))
# turn into graph model
landmarks = stentgraph.StentGraph()
landmarks.unpack(s2.landmarksavgreg)
landmarks.nodes()


## Calculate deforms of avgreg
deformslandmark = []
for i in range(0, 12):
    point = landmarks.nodes()[i]
    u = landmarks.node[point]
    deformslandmark.append(u)   
print(deformslandmark)


## Calculate algorithm coordinates with avgreg & deforms
#deformstest = deforms1landmark0['deforms']
#point0 = landmarks.nodes()[0]
#deforms1landmark0 = landmarks.node[point0]
#print(deforms1landmark0)
#deformstest0 = deformstest[0]
#point0real = point0 + deformstest0

# realalgorithmpoints = []
# for loop voor alle 12 punten: 
    # point = landmarks.nodes()[i]
    # point2 = point*repmat*
    # deformspoint = landmarks.node[point]
    # deformspoint2 = alleen juiste list uit deformspoint
    # q = point + deformspoint
    # realalgorithmpoints.append(q)
# print(realalgorithmpoints)


## Load dynamic landmark model phases
#todo: change loadmodel() to work when there is not a folder per patient like now
fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarksphases')
s2 = ssdf.load(os.path.join(dirsave, fname))
# turn into graph model
landmarkphases = []
for key in dir(s2):
    if key.startswith('landmarks'):
        landmarks = stentgraph.StentGraph()
        landmarks.unpack(s2[key])
        p = landmarks.nodes()
        p.sort(key=lambda x: landmarks.node[x]['number']) 
        p = np.asarray(p)
        landmarkphases.append(p)

import xlsxwriter
def storeCoordinatesToExcel(landmarkphases, exceldir, graph):
    """Create file and add a worksheet or overwrite existing
    mind that floats can not be stored with write_row.
    Input = pp1, sorted array of picked points (coordinates)
    """
    rowoffset=0
    # https://pypi.python.org/pypi/XlsxWriter
    workbook = xlsxwriter.Workbook(os.path.join(exceldir,'storeOutputTemplatePhases_020_discharge.xlsx'))
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
    for x in landmarkphases:
        for y, Output in enumerate(x):
            worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
            #worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
            rowoffset += 1
        rowoffset += 1

        # if (rowoffset % 2 == 0): # odd
        #     rowoffset += 4 # add rowspace for next points

    # Store screenshot
    #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
    workbook.close()


storeCoordinatesToExcel(landmarkphases,exceldir,landmarks)
print('---stored to excel {} ---'.format(exceldir) )
print('---model can be stored as ssdf in do_segmentation---')



## Export to excel als je in matlab wil plotten.. 
# voor plotten in python:
# matplotlib, visvis, of pyplot

# import matplotlib.pyplot as plt
# from analysis.utils_analysis import _initaxis
# f1 = plt.figure(figsize=(16,4.5), num=1); plt.clf()
# ax0 = f1.add_subplot(121)
# f1.savefig(os.path.join(dirsave, 'patternexampleproc.pdf'), papertype='a0', dpi=300)

#todo: a workflow to visualize/plot outcome


# van oude code:

# Store to EXCEL
pp1 = sd2._graphrefined.nodes()
pp1.sort(key=lambda x: sd2._graphrefined.node[x]['number']) 
# sort nodes by click number
# pp1 = sd2._graphrefined.nodes()
print('*** refined manual picked were stored ***')

pp1 = np.asarray(pp1)
storeCoordinatesToExcel(pp1,exceldir)
print('---stored to excel {} ---'.format(exceldir) )
print('---model can be stored as ssdf in do_segmentation---')


import xlsxwriter
def storeCoordinatesToExcel(pp1, exceldir):
    """Create file and add a worksheet or overwrite existing
    mind that floats can not be stored with write_row.
    Input = pp1, sorted array of picked points (coordinates)
    """
    # https://pypi.python.org/pypi/XlsxWriter
    workbook = xlsxwriter.Workbook(os.path.join(exceldir,'storeOutputTemplate.xlsx'))
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('A:A', 15)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    # write title
    worksheet.write('A1', 'Point coordinates', bold)
    analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
    worksheet.write('B3', analysisID, bold)
    # write 'picked coordinates'
    rowoffset = 4
    try:
        graph = sd2._graphrefined
    except AttributeError:
        graph = sd2._nodes1
    for i, Output in enumerate(pp1):
        worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
        worksheet.write_number(rowoffset, 0, graph.node[tuple(Output)]['number'])
        rowoffset += 1
        # if (rowoffset % 2 == 0): # odd
        #     rowoffset += 4 # add rowspace for next points

    # Store screenshot
    #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
    workbook.close()


