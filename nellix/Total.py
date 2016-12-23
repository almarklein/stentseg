'''
Created on 1 jun. 2016

@author: TomLoonen
'''
import os
ptcode = 'chevas_7'
dicom_basedir = 'E:/CT/no 7/STD00001/10phases'
basedir = 'E:/CT/SSDF/'
print(dicom_basedir)

##
from nellix2ssdf import Nellix2ssdf
foo = Nellix2ssdf(dicom_basedir,ptcode,basedir)
print('load, crop and convert to ssdf: done')

##
from _do_registration_nellix import _Do_Registration_Nellix
foo = _Do_Registration_Nellix(ptcode,basedir)
#vol = foo.getVol()
print('registration: done')

##
from _do_segmentation_nellix import _Do_Segmentation
foo = _Do_Segmentation(ptcode,basedir)
print('segmentation: done')

##
from _select_stent_endpoints import _Select_Stent_Endpoints
foo = _Select_Stent_Endpoints(ptcode,basedir)
StartPoints = foo.StartPoints
EndPoints = foo.EndPoints
print('Get Endpoints: done')

##
# # Points no 

StartPoints = [(126.7, 99.8, 64.3), (136.7, 97.8, 66.2), (120.4, 99.8, 60.7)]
EndPoints = [(119.8, 85.2, 113.4), (128.7, 77.2, 113.7), (104.7, 97.0, 66.6)]


from _test_centerline import _Test_Centerline
foo = _Test_Centerline(ptcode,StartPoints,EndPoints,basedir)
allcenterlines = foo.allcenterlines # PointSet
print('centerline: done')

##
# visualize centerlines and select points for motion comparison

from _select_centerline_points import _Select_Centerline_Points
foo = _Select_Centerline_Points(ptcode, allcenterlines, StartPoints,EndPoints,basedir)
#foo = _Select_Centerline_Points(ptcode,basedir)

##
from _show_model import _Show_Model
foo = _Show_Model(ptcode,basedir)
print('chEVAS centerline finished')

