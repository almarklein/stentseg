'''
Created on 1 jun. 2016

@author: TomLoonen / MaaikeKoenrades
'''
import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol


ptcode = 'chevas_01'
ctcode = '12months' # is not the true follow-up time but use for all
# dicom_basedir = r'E:\Nellix_chevas\CT_SSDF\no 1\STD00001\10phases'
# print(dicom_basedir)
basedir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF_automated')

## LOAD CT
if False:
        from nellix.nellix2ssdf import Nellix2ssdf
        foo = Nellix2ssdf(dicom_basedir,ptcode,ctcode,basedir)
        print('load, crop and convert to ssdf: done')

## REGISTRATION
if False:
        from nellix._do_registration_nellix import _Do_Registration_Nellix
        foo = _Do_Registration_Nellix(ptcode,ctcode,basedir)
        #vol = foo.getVol()
        print('registration: done')

## SEGEMENTATION STENT
if False:
        from nellix._do_segmentation_nellix import _Do_Segmentation
        foo = _Do_Segmentation(ptcode,ctcode,basedir,seed_th=[600], show=True, normalize=True)
        print('segmentation: done')

## SEGMENTATION VASCULATURE
if False:
        from nellix._do_segmentation_vessel import _Do_Segmentation_Vessel
        foo = _Do_Segmentation_Vessel(ptcode,ctcode,basedir,threshold=275, show=True, normalize=False)
        print('segmentation vessel: done')

## SELECT POINT FOR CENTERLINE
if False:
        from nellix._select_stent_endpoints import _Select_Stent_Endpoints
        foo = _Select_Stent_Endpoints(ptcode,ctcode,basedir) # loads prox_avgreg
        StartPoints = foo.StartPoints # proximal
        EndPoints = foo.EndPoints # distal
        print('Get Endpoints: done')

## CALCULATE CENTERLINE
if False:
        StartPoints = [ (173.2, 127.0, 113.7), 
                        (174.1, 139.5, 114.1), 
                        (144.6, 113.9, 66.5), 
                        (118.9, 145.9, 76.9), 
                        (171.6, 162.6, 42.4),(166.7, 159.5, 43.8), (161.6, 156.4, 44.9)]
        EndPoints = [   (143.3, 160.9, 37.7), 
                        (144.3, 168.4, 44.4), 
                        (132.1, 154.2, 38.0), 
                        (138.7, 156.3, 28.9), 
                        (166.7, 159.5, 43.8), (161.6, 156.4, 44.9), (141.1, 153.0, 40.1)]
        
        
        StartPoints = [(113.6, 89.5, 34.3)] # pt 1 chim vessel
        EndPoints =   [(128.5, 94.1, 31.6)]

        from nellix._get_centerline import _Get_Centerline
        # modelname = 'modelavgreg' # default for stents
        modelname = 'modelvesselavgreg' 
        foo = _Get_Centerline(ptcode,ctcode,StartPoints,EndPoints, basedir, modelname=modelname) # MK: start=distal points
        # loads prox_avgreg (reg) and prox_modelavgreg (segm)
        # saves centerline_modelavgreg and centerline_total_modelavgreg; also with _deforms as dynamic centerline
        # in centerline we have all centerlines but separate graphs; in centerline_total merged in one graph
        allcenterlines = foo.allcenterlines # list with PointSet per centerline
        print('centerline: done')


## IDENTIFY CENTERLINES FOR ANALYSIS
if False:
        from stentseg.utils.centerline import points_from_nodes_in_graph
        # modelname = 'centerline_modelavgreg_deforms'
        modelname = 'centerline_modelvesselavgreg_deforms' 
        s = loadmodel(basedir, ptcode, ctcode, 'prox', modelname=modelname)
        
        from nellix.identify_centerlines import identifyCenterlines
        a, s2 = identifyCenterlines(s, ptcode,ctcode,basedir,modelname,showVol='MIP')

## MOTION ANALYSIS CENTERLINES
if True: 
        import os
        from stentseg.utils.datahandling import select_dir, loadmodel, loadvol
        from stentseg.utils.picker import pick3d
        
        ptcodes = [
                'chevas_01',
                # 'chevas_02',
                # 'chevas_03',
                # 'chevas_04',
                # 'chevas_05',
                # 'chevas_06',
                # 'chevas_07',
                # 'chevas_08',
                # 'chevas_09',
                # 'chevas_10',
                # 'chevas_11',
                # 'chevas_09_thin'
                                ]
        ctcode = '12months'
        basedir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF_automated')
        
        for ptcode in ptcodes:
                from nellix._do_analysis_centerline import _Do_Analysis_Centerline
                AC = _Do_Analysis_Centerline(ptcode,ctcode,basedir,showVol='ISO') # loads centerline_modelavgreg_deforms_id
                # chimney / chimney-nel angles
                AC.chimneys_angle_change(armlength=10)
                # chimney-vessel transition angle
                AC.chimneys_vessel_angle_change(armlength=10)
                # motion stents
                AC.motion_centerlines_segments(lenSegment=10) # stents
                # motion vessel after stent
                AC.motion_centerlines_segments(lenSegment=10, type='vessels')
                # distances nel-nel and chim-nel
                AC.distance_change_nelnel_nelCh()
                # save to excel
                AC.storeOutputToExcel()

## [OLD CODE] MOTION ANALYSIS CENTERLINES - CLICK NODES
if False:
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
if False:
        from nellix._show_model import _Show_Model
        foo = _Show_Model(ptcode,ctcode,basedir, meshWithColors=True,motion='sum',clim2=(0,4))
        print('chEVAS centerline is shown in motion')

