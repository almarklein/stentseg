'''
Created on 1 jun. 2016

@author: TomLoonen / MaaikeKoenrades
'''
import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol


ptcode = 'chevas_05'
ctcode = '12months'
dicom_basedir = r'F:\Nellix_chevas\CT_SSDF\no 6\STD00001\10phases'
basedir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF')
print(dicom_basedir)


## LOAD CT
from nellix.nellix2ssdf import Nellix2ssdf
foo = Nellix2ssdf(dicom_basedir,ptcode,basedir)
print('load, crop and convert to ssdf: done')

## REGISTRATION
from nellix._do_registration_nellix import _Do_Registration_Nellix
foo = _Do_Registration_Nellix(ptcode,ctcode,basedir)
#vol = foo.getVol()
print('registration: done')

## SEGEMENTATION
from nellix._do_segmentation_nellix import _Do_Segmentation
foo = _Do_Segmentation(ptcode,ctcode,basedir)
print('segmentation: done')

## SELECT POINT FOR CENTERLINE
from nellix._select_stent_endpoints import _Select_Stent_Endpoints
foo = _Select_Stent_Endpoints(ptcode,ctcode,basedir) # loads prox_avgreg
StartPoints = foo.StartPoints # proximal
EndPoints = foo.EndPoints # distal
print('Get Endpoints: done')

## CALCULATE CENTERLINE
# 7 -mirthe 2
# StartPoints = [(184.3, 111.4, 16.7), (188.6, 125.2, 19.1), (154.9, 113.9, 17.8)] # prox
# EndPoints = [(186.1, 90.0, 78.9), (188.3, 101.7, 78.9), (147.0, 107.8, 33.3)] # dist
# 5 -mirthe 2
StartPoints = [(122.4, 139.3, 31.3), (132.8, 137.2, 31.3), (126.9, 150.4, 33.5)] # prox
EndPoints = [(123.9, 122.1, 69.5),(136.3, 125.1, 69.0), (147,130.8, 53.3) ]  # dist

from nellix._get_centerline import _Get_Centerline
foo = _Get_Centerline(ptcode,ctcode,EndPoints,StartPoints,basedir) # MK: start=distal points --> endpoints
# loads prox_avgreg (reg) and prox_modelavgreg (segm)
# saves centerline_modelavgreg and centerline_total_modelavgreg; also with _deforms as dynamic centerline
# in centerline we have all centerlines but separate graphs; in centerline_total merged in one graph
allcenterlines = foo.allcenterlines # list with PointSet per centerline
print('centerline: done')


## IDENTIFY CENTERLINES FOR ANALYSIS
from stentseg.utils.centerline import points_from_nodes_in_graph
s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelavgreg_deforms')

from nellix.identify_centerlines import identifyCenterlines
modelname='centerline_modelavgreg_deforms'
a, s2 = identifyCenterlines(s, ptcode,ctcode,basedir,modelname,showVol='MIP')

## MOTION ANALYSIS CENTERLINES
import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol
from stentseg.utils.picker import pick3d

ptcode = 'chevas_05'
ctcode = '12months'
# basedir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF')
basedir = select_dir(r'D:\Profiles\koenradesma\SURFdrive\Nellix_chevas_data_for_testing',
                     r'C:\Users\Maaike\SURFdrive\Nellix_chevas_data_for_testing')

s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelavgreg_deforms_id')
# s4 = loadvol(basedir, ptcode, ctcode, 'prox', 'avgreg')
# t = show_ctvolume(s4.vol, showVol='MIP', clim =(0,2500), isoTh=250,removeStent=False, climEditor=True)
# label = pick3d(vv.gca(), s4.vol)

from nellix._do_analysis_centerline import _Do_Analysis_Centerline
AC = _Do_Analysis_Centerline(ptcode,ctcode,basedir,showVol='MIP') # loads centerline_modelavgreg_deforms_id
AC.chimneys_angle_change()
AC.motion_centerlines_segments(lenSegment=5, dim='xyz')
AC.distance_change_nelnel_nelCh()
AC.storeOutputToExcel()

## [OLD CODE] MOTION ANALYSIS CENTERLINES - CLICK NODES
s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelavgreg_deforms')
allcenterlines = []
for key in dir(s):
        if key.startswith('model'):
                model = points_from_nodes_in_graph(s[key])
                allcenterlines.append(model)

from nellix._select_centerline_points import _Select_Centerline_Points
foo = _Select_Centerline_Points(ptcode, ctcode, allcenterlines, basedir)
# figure loading may take > 3 min when about 400 centerline points

## SHOW CENTERLINE DYNAMIC
from nellix._show_model import _Show_Model
foo = _Show_Model(ptcode,ctcode,basedir, meshWithColors=True,motion='sum',clim2=(0,4))
print('chEVAS centerline is shown in motion')

