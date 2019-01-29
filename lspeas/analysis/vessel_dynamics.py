""" Module to analyze vessel pulsatility during the heart cycle in ecg-gated CT
radius change - area change - volume change

"""

import openpyxl
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmesh
import sys, os
import numpy as np
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
from lspeas.utils.vis import showModelsStatic
import visvis as vv
from stentseg.utils.centerline import points_from_mesh


#todo: move function to lspeas utils?
def load_excel_centerline(basedirCenterline, vol, ptcode, ctcode, filename=None):
    """ Load centerline data from excel
    Centerline exported from Terarecon; X Y Z coordinates in colums
    """
    if filename is None:
        filename = '{}_{}_centerline.xlsx'.format(ptcode,ctcode)
    excel_location = os.path.join(basedirCenterline,ptcode)
    #read sheet
    try:
        wb = openpyxl.load_workbook(os.path.join(excel_location,filename),read_only=True)
    except FileNotFoundError:
        wb = openpyxl.load_workbook(os.path.join(basedirCenterline,filename),read_only=True)
    sheet = wb.get_sheet_by_name(wb.sheetnames[0]) # data on first sheet
    colStart = 2 # col C
    rowStart = 1 # row 2 excel
    coordx = sheet.columns[colStart][rowStart:] 
    coordy = sheet.columns[colStart+1][rowStart:]
    coordz = sheet.columns[colStart+2][rowStart:]  
    #convert to values
    coordx = [obj.value for obj in coordx]
    coordy = [obj.value for obj in coordy]
    coordz = [obj.value for obj in coordz]
    #from list to array
    centerlineX = np.asarray(coordx, dtype=np.float32)
    centerlineY = np.asarray(coordy, dtype=np.float32)
    centerlineZ = np.asarray(coordz, dtype=np.float32)
    centerlineZ = np.flip(centerlineZ, axis=0) # z of volume is also flipped
    
    # convert centerline coordinates to world coordinates (mm)
    origin = vol1.origin # z,y,x
    sampling = vol1.sampling # z,y,x
    
    centerlineX = centerlineX*sampling[2] +origin[2]
    centerlineY = centerlineY*sampling[1] +origin[1]
    centerlineZ = (centerlineZ-0.5*vol1.shape[0])*sampling[0] + origin[0]
    
    return centerlineX, centerlineY, centerlineZ


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')
                     
basedirMesh = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics',
    r'C:\Users\Maaike\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics')

basedirCenterline = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS_centerlines_terarecon')

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode1 = 'discharge'
cropname = 'ring'
modelname = 'modelavgreg'
cropvol = 'stent'

drawModelLines = False  # True or False
drawRingMesh, ringMeshDisplacement = True, False
meshColor = [(1,1,0,1)]
removeStent = True # for iso visualization
dimensions = 'xyz'
showAxis = False
showVol  = 'ISO'  # MIP or ISO or 2D or None
showvol2D = False
drawVessel = True

clim = (0,2500)
clim2D = -200,500 # MPR
clim2 = (0,2)
isoTh = 180 # 250


# Load CT image data for reference
s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
vol1 = loadvol(basedir, ptcode, ctcode1, cropvol, 'avgreg').vol

# Load ring model
if drawRingMesh:
    if not ringMeshDisplacement:
        modelmesh1 = create_mesh(s1.model, 0.7)  # Param is thickness
    else:
        modelmesh1 = create_mesh_with_abs_displacement(s1.model, radius = 0.7, dim=dimensions)

# Load vesselmesh (mimics)
filename = '{}_{}_neck.stl'.format(ptcode,ctcode1)
vessel1 = loadmesh(basedirMesh,ptcode[-3:],filename) #inverts Z
# get PointSet from vessel mesh 
ppvessel = points_from_mesh(vessel1, invertZ=False) # removes duplicates

# Load vessel centerline (excel terarecon)
centerlineX, centerlineY, centerlineZ = load_excel_centerline(basedirCenterline, 
                                        vol1, ptcode, ctcode1, filename=None)

# Show ctvolume, vessel mesh, ring model
axes, cbars = showModelsStatic(ptcode, ctcode1, [vol1], [s1], [modelmesh1], 
              [vessel1], showVol, clim, isoTh, clim2, clim2D, drawRingMesh, 
              ringMeshDisplacement, drawModelLines, showvol2D, showAxis, 
              drawVessel, vesselType=2,
              climEditor=True, removeStent=removeStent, meshColor=meshColor)

# Show the centerline
vv.plot(centerlineX, centerlineY, centerlineZ, ms='.', ls='', mw=8, mc='b')



#todo: create class/workflow to obtain radius change minor/major axis, asymmetry, 
#todo: and area at used selected levels and volume change 

#todo: visualize mesh with colors, each vertice representing axis change from COM/centerline
