""" Show centerlines of stents and vessel together with Start en Endpoints
For image in flowchart methods
"""
import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol
from nellix._do_analysis_centerline import _Do_Analysis_Centerline
from stentseg.utils.centerline import points_from_nodes_in_graph
import visvis as vv
from stentseg.utils import PointSet
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume

basedir = select_dir(r'E:\Nellix_chevas\CT_SSDF\SSDF_automated',
r'D:\Nellix_chevas_BACKUP\CT_SSDF\SSDF_automated')

ptcode = 'chevas_10'
ctcode = '12months'

AC = _Do_Analysis_Centerline(ptcode, ctcode,basedir,showVol='ISO') # loads centerline_modelavgreg_deforms_id

AC.a.axis.axisColor = 0,0,0
AC.a.axis.visible = True
AC.a.bgcolor = 1,1,1

# get start endpoints
stentsStartPoints = AC.s.StartPoints
stentsEndPoints = AC.s.EndPoints
vesselStartPoints = AC.s_vessel.StartPoints
vesselEndPoints = AC.s_vessel.EndPoints

# get ppcenterlines to plot with appropriate markers and colors
allcenterlines = AC.s.ppallCenterlines
allcenterlinesv = AC.s_vessel.ppallCenterlines

# get seedpoints segmentations
s_modelvessel = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='modelvesselavgreg')
ppmodelvessel = points_from_nodes_in_graph(s_modelvessel.model)

s_model = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='modelavgreg')
ppmodel = points_from_nodes_in_graph(s_model.model)

# Visualize
AC.a.MakeCurrent()
vv.cla() # clear vol otherwise plots not visible somehow

# seedpoints segmentations
vv.plot(ppmodelvessel, ms='.', ls='', alpha=0.6, mw=2)
vv.plot(ppmodel,       ms='.', ls='', alpha=0.6, mw=2)

# stents
for j in range(len(stentsStartPoints)):
    vv.plot(PointSet(list(stentsStartPoints[j])), ms='.', ls='', mc='g', mw=20) # startpoint green
    vv.plot(PointSet(list(stentsEndPoints[j])),  ms='.', ls='', mc='r', mw=20) # endpoint red
for j in range(len(allcenterlines)):
            vv.plot(allcenterlines[j], ms='.', ls='', mw=10, mc='y')
# vessels
for j in range(len(vesselStartPoints)):
    vv.plot(PointSet(list(vesselStartPoints[j])), ms='.', ls='', mc='g', mw=20) # startpoint green
    vv.plot(PointSet(list(vesselEndPoints[j])),  ms='.', ls='', mc='r', mw=20) # endpoint red
for j in range(len(allcenterlinesv)):
            vv.plot(allcenterlinesv[j], ms='.', ls='', mw=10, mc='y')
       
vv.title('Centerlines and seed points')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')