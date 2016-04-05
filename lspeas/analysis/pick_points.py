""" Manual marker placement in volume MIP
The voxel with maximum intentity under the cursor when doing a SHIFT+RIGHTCLICK
is returned
"""

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet
from stentseg.stentdirect import stentgraph
from visvis import Pointset # for meshes
from stentseg.stentdirect.stentgraph import create_mesh
from visvis.processing import lineToMesh, combineMeshes
from visvis import ssdf
from stentseg.utils.picker import pick3d, label2worldcoordinates

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
            r'D:\Profiles\koenradesma\Desktop')

# Select dataset to register
ptcode = 'QRM_FANTOOM_20160121'
ctcode = 'ZA3-75-1.2'
cropname = 'ring'
what = 'avgreg'

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol


# Visualize and activate picker
fig = vv.figure(1); vv.clf()
fig.position = 8.00, 30.00,  944.00, 1002.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = True
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
lim = 2000
t = vv.volshow(vol, clim=(0, lim), renderStyle='mip')
label = pick3d(vv.gca(), vol)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('CT Volume for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))

storeOutput = list()
def on_key(event): 
    if event.key == vv.KEY_ENTER:
        coordinates = label2coordinates(label)
        storeOutput.append(coordinates)
    if event.key == vv.KEY_ESCAPE:
        # Store to EXCEL
        storeCoordinatesToExcel(storeOutput,exceldir)
        print('stored to excel {}.'.format(exceldir) )


import xlsxwriter
def storeCoordinatesToExcel(coordinates, exceldir):
    """Create file and add a worksheet or overwrite existing
    mind that floats can not be stored with write_row
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
    # write 'storeOutput'
    rowoffset = 4
    for i, Output in enumerate(storeOutput):
        rowstart = rowoffset # startrow for this Output
        worksheet.write_row(rowoffset, 1, Output) # row, columnm, point
        rowoffset += 1
        if (rowoffset % 2 == 0): # odd
            rowoffset += 4 # add rowspace for next points

    # Store screenshot of stent
    #vv.screenshot(r'C:\Users\Maaike\Desktop\storeScreenshot.png', vv.gcf(), sf=2)
    workbook.close()

# Bind event handlers
fig.eventKeyDown.Bind(on_key)