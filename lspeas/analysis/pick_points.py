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
from stentseg.stentdirect import stentpoints3d, StentDirect, getDefaultParams

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select location storeOutputTemplate EXCEL file
exceldir = select_dir(r'C:\Users\Maaike\Desktop',
            r'D:\Profiles\koenradesma\Desktop')

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode = '1month'
cropname = 'ring'
what = 'avgreg'


# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol

# params for ssdf saving
stentType = 'manual'
p = 'manualSeedsMip'
what += '_manual'

# Visualize and activate picker
fig = vv.figure(1); vv.clf()
fig.position = 8.00, 30.00,  944.00, 1002.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
lim = 2000
t = vv.volshow(vol, clim=(0, lim), renderStyle='mip')
label = pick3d(vv.gca(), vol)
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('CT Volume for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))

# pp2 = stentpoints3d.get_subpixel_positions(self._vol, np.array(pp1))
# M = {}
# for i in range(pp2.shape[0]):
#     M[pp1[i]] = tuple(pp2[i].flat)

# instantiate stentdirect segmenter object
sd = StentDirect(vol, getDefaultParams() )
# initialize
sd._nodes1 = stentgraph.StentGraph()
nr = 0
def on_key(event): 
    if event.key == vv.KEY_CONTROL:
        global nr
        coordinates = np.asarray(label2worldcoordinates(label), dtype=np.float32) # x,y,z
        n2 = tuple(coordinates.flat)
        sd._nodes1.add_node(n2, number=nr)
        print(nr)
        if nr > 0:
            for n in list(sd._nodes1.nodes()):
                if sd._nodes1.node[n]['number']== nr-1:
                    path = [n2,n]
                    # node2 = PointSet(np.column_stack(n2))
                    # node = PointSet(np.column_stack(n))
                    sd._nodes1.add_edge(n2, n, path = PointSet(np.row_stack(path)) )
        sd._nodes1.Draw(mc='r', mw = 10, lc='y')
        nr += 1
    if event.key == vv.KEY_ENTER:
        sd._graphrefined = sd._RefinePositions(sd._nodes1)
        sd._graphrefined.Draw(mc='b', mw = 10, lc='g') 
        # vv.plot(pp1[2,:], pp1[1,:], pp1[0,:], mc= 'g', ms= '.', mw= 10)
    if event.key == vv.KEY_ESCAPE:
        # Store to EXCEL
        pp1 = []
        try:
            pp1 = sd._graphrefined.nodes()
            pp1.sort(key=lambda x: sd._graphrefined.node[x]['number']) # sort nodes by click number
            # pp1 = sd._graphrefined.nodes()
            print('*** refined manual picked were stored ***')
        except AttributeError:
            pp1 = sd._nodes1.nodes()
            pp1.sort(key=lambda x: sd._nodes1.node[x]['number']) # sort nodes by click number
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
        graph = sd._graphrefined
    except AttributeError:
        graph = sd._nodes1
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

