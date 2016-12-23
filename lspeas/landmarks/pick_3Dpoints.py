""" Manual marker placement in volume MIP
The voxel with maximum intentity under the cursor when doing a SHIFT+RIGHTCLICK
is returned
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

# Select the ssdf basedir
basedir = select_dir(r'C:\Users\Freija\Documents\M2.2 stage MST Vaatchirurgie\ssdf')

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Freija\Desktop')

# Select dataset to register
ptcode = 'LSPEAS_008'
ctcode = 'discharge'
cropname = 'stent'
what = 'phases' # 'phases'
phase = 0 # % or RR interval

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, what)
if what == 'phases':
    vol = s['vol%i'% phase]
else:
    vol = s.vol

vol2 = s.vol10

# # params for ssdf saving
# stentType = 'manual'
# what += '_manual'

## Visualize and activate picker

clim = (0,2500)
# clim = 250
showVol = 'MIP' # MIP or ISO or 2D

#fig = vv.figure(1); vv.clf()
#fig.position = 8.00, 30.00,  944.00, 1500.00
#a1 = vv.subplot(131) #
#label = DrawModelAxes(vol, clim=clim, showVol=showVol, axVis = True)

#vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
#vv.title('CT Volume %i%% for LSPEAS %s  -  %s' % (phase, ptcode[7:], ctcode))

#subplots
fig = vv.figure(1); vv.clf()
fig.position = 8.00, 30.00,  944.00, 1500.00
a1 = vv.subplot(131) 
label = DrawModelAxes(vol, clim=clim, showVol=showVol, axVis = True)

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('CT Volume %i%% for LSPEAS %s  -  %s' % (phase, ptcode[7:], ctcode))

a2 = vv.subplot(132) 
label = DrawModelAxes(vol2, clim=clim, showVol=showVol, axVis = True)

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('CT Volume %i%% for LSPEAS %s  -  %s' % (phase, ptcode[7:], ctcode))

a1.camera = a2.camera

# bind rotate view (a, d, z, x active keys)
fig.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event) )

# instantiate stentdirect segmenter object
p = getDefaultParams()
sd2 = StentDirect(vol, p)
# initialize
sd2._nodes1 = stentgraph.StentGraph()
nr = 0
def on_key(event): 
    if event.key == vv.KEY_CONTROL:
        global nr
        coordinates = np.asarray(label2worldcoordinates(label), dtype=np.float32) # x,y,z
        n2 = tuple(coordinates.flat)
        sd2._nodes1.add_node(n2, number=nr)
        print(nr)
        if nr > 0:
            for n in list(sd2._nodes1.nodes()):
                if sd2._nodes1.node[n]['number']== nr-1:
                    path = [n2,n]
                    sd2._nodes1.add_edge(n2, n, path = PointSet(np.row_stack(path)) )
        sd2._nodes1.Draw(mc='r', mw = 10, lc='y')
        nr += 1
    if event.key == vv.KEY_ENTER:
        sd2._graphrefined = sd2._RefinePositions(sd2._nodes1)
        sd2._graphrefined.Draw(mc='b', mw = 10, lc='g') 
    if event.key == vv.KEY_ESCAPE:
        # Store to EXCEL
        pp1 = []
        try:
            pp1 = sd2._graphrefined.nodes()
            pp1.sort(key=lambda x: sd2._graphrefined.node[x]['number']) # sort nodes by click number
            # pp1 = sd2._graphrefined.nodes()
            print('*** refined manual picked were stored ***')
        except AttributeError:
            pp1 = sd2._nodes1.nodes()
            pp1.sort(key=lambda x: sd2._nodes1.node[x]['number']) # sort nodes by click number
            print('*** manual picked were stored ***')
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

# Bind event handlers
fig.eventKeyDown.Bind(on_key)

