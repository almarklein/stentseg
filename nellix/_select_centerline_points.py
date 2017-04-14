
class _Select_Centerline_Points:
    def __init__(self,ptcode,allcenterlines,StartPoints,EndPoints,basedir):
        """
        Script to show the stent plus centerline model and select ponits on 
        centerlines for motion analysis 
        """
        import os
        import pirt
        import visvis as vv
        import numpy as np
        import math
        import itertools
        import xlsxwriter
        from datetime import datetime

                
        from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
        from stentseg.utils.new_pointset import PointSet
        from stentseg.stentdirect.stentgraph import create_mesh
        from stentseg.motion.vis import create_mesh_with_abs_displacement
        from stentseg.utils.visualization import show_ctvolume
        # from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion ,calculateMeanAmplitude
        from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
        from stentseg.utils import PointSet
        from stentseg.stentdirect import stentgraph
        from visvis import Pointset # for meshes
        from stentseg.stentdirect.stentgraph import create_mesh
        from visvis.processing import lineToMesh, combineMeshes
        from visvis import ssdf
        from stentseg.utils.picker import pick3d
        # from PyQt4 import QtCore, QtGui
        # from Tom_utils.mydialog import MyDialog
        from stentseg.utils.centerline import dist_over_centerline # added for Mirthe 
        import copy
        
        
        cropname = 'prox'
        ctcode = '12months'
     
        exceldir = os.path.join(basedir,ptcode)
        
        # Load deformations and avg ct (forward for mesh)
        m = loadmodel(basedir, ptcode, ctcode, cropname, modelname = 'centerline_total_modelavgreg_deforms') # centerlines combined in 1 model
        model = m.model
        m_sep = loadmodel(basedir, ptcode, ctcode, cropname, modelname = 'centerline_modelavgreg_deforms') # centerlines separated in a model for each centerline
        s = loadvol(basedir, ptcode, ctcode, cropname, what='avgreg')
        vol_org = copy.deepcopy(s.vol)
        s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]]
        s.sampling = s.vol.sampling
        vol = s.vol
        
        # Start visualization and GUI
        
        fig = vv.figure(30); vv.clf()
        
        fig.position = 0.00, 30.00,  944.00, 1002.00
        a = vv.gca()
        a.axis.axisColor = 1,1,1
        a.axis.visible = True
        a.bgcolor = 0,0,0
        a.daspect = 1, 1, -1
        lim = 2500
        t = vv.volshow(vol, clim=(0, lim), renderStyle='mip')
        pick3d(vv.gca(), vol)
        b = model.Draw(mc='b', mw = 0, lc='g', alpha = 0.5)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
        viewringcrop = {'zoom': 0.012834824098558318,
        'fov': 0.0,
        'daspect': (1.0, 1.0, -1.0),
        'loc': (139.818258268377, 170.0738625060885, 80.55734045456558),
        'elevation': 11.471611096074625,
        'azimuth': 25.71485900482051,
        'roll': 0.0}
        
        # Add clickable nodes
        node_points = []
        for i, node in enumerate(sorted(model.nodes())):
            node_point = vv.solidSphere(translation = (node), scaling = (0.6,0.6,0.6))
            node_point.faceColor = 'b'
            node_point.alpha = 0.5
            node_point.visible = True
            node_point.node = node
            node_point.nr = i
            node_points.append(node_point)
        
        # list of correctly clicked nodes
        selected_nodes_sum = set()
        
        # Initialize labels
        t0 = vv.Label(a, '\b{Node nr|location}: ', fontSize=11, color='w')
        t0.position = 0.1, 25, 0.5, 20  # x (frac w), y, w (frac), h
        t0.bgcolor = None
        t0.visible = True
        t1 = vv.Label(a, '\b{Nodepair}: ', fontSize=11, color='w')
        t1.position = 0.1, 45, 0.5, 20
        t1.bgcolor = None
        t1.visible = True
        
        # Initialize output variable to store pulsatility analysis
        storeOutput = list()
        output_cl = list()
        output_clmin_index = list()
        output_clmin = list()
        output_clmax = list()
        output_clmax_index = list()
        
        
        def on_key(event): 
            if event.key == vv.KEY_ENTER:
                
                # mogenlijkheden aantal nodes 
                    # 1 voor relative beweging vanuit avg punt
                    # 2 voor onderlinge beweging tussen twee punten
                    # 3 voor hoek in punt 2 van punt 1 naar punt 3
                    
                if len(selected_nodes) == 1:
                    selectn1 = selected_nodes[0].node
                    n1index = selected_nodes[0].nr
                    n1Deforms = model.node[selectn1]['deforms']
                    output = point_pulsatility(selectn1, n1Deforms)
                    output['NodesIndex'] = [n1index]
                    
                    # Store output with name
                    dialog_output = get_index_name()
                    output['Name'] = dialog_output
                    storeOutput.append(output)
                    
                    # update labels
                    t1.text = '\b{Node}: %i' % (n1index)
                    t1.visible = True
                    print('selection of 1 node stored')

                if len(selected_nodes) == 2:
                    # get nodes
                    selectn1 = selected_nodes[0].node
                    selectn2 = selected_nodes[1].node
                    # get index of nodes which are in fixed order
                    n1index = selected_nodes[0].nr
                    n2index = selected_nodes[1].nr
                    nindex = [n1index, n2index]
                    # get deforms of nodes
                    n1Deforms = model.node[selectn1]['deforms']
                    n2Deforms = model.node[selectn2]['deforms']
                    # get pulsatility
                    cl_merged = append_centerlines(allcenterlines) 
                    output = point_to_point_pulsatility(cl_merged, selectn1, 
                                        n1Deforms, selectn2, n2Deforms, type='euclidian')
                    output['NodesIndex'] = nindex
                    
                    # get distance_centerline
                   
                    #dist_cl = dist_over_centerline(cl_merged, selectn1, selectn2, type='euclidian') # toegevoegd Mirthe
                    
                    #dist_cl = dist_centerline_total(cl_merged, selectn1, 
                    #                    n1Deforms, selectn2, n2Deforms, type='euclidian')
                    
                    # Store output with name
                    dialog_output = get_index_name()
                    output['Name'] = dialog_output
                    storeOutput.append(output)  
                   
                    # update labels
                    t1.text = '\b{Node pair}: %i - %i' % (nindex[0], nindex[1])
                    t1.visible = True
                    print('selection of 2 nodes stored')
                        
                if len(selected_nodes) == 3:
                    # get nodes
                    selectn1 = selected_nodes[0].node
                    selectn2 = selected_nodes[1].node
                    selectn3 = selected_nodes[2].node
                    # get index of nodes which are in fixed order
                    n1index = selected_nodes[0].nr
                    n2index = selected_nodes[1].nr
                    n3index = selected_nodes[2].nr
                    nindex = [n1index, n2index, n3index]
                    # get deforms of nodes
                    n1Deforms = model.node[selectn1]['deforms']
                    n2Deforms = model.node[selectn2]['deforms']
                    n3Deforms = model.node[selectn3]['deforms']
                    # get angulation
                    output = line_line_angulation(selectn1, 
                                        n1Deforms, selectn2, n2Deforms, selectn3, n3Deforms)
                    output['NodesIndex'] = nindex
                    
                    # Store output with name
                    dialog_output = get_index_name()
                    output['Name'] = dialog_output
                    storeOutput.append(output)
                    
                    # update labels
                    t1.text = '\b{Nodes}: %i - %i - %i' % (nindex[0], nindex[1], nindex[2])
                    t1.visible = True
                    print('selection of 3 nodes stored')
                    
                if len(selected_nodes) > 3:
                    for node in selected_nodes:
                        node.faceColor = 'b'
                    selected_nodes.clear()
                    print('to many nodes selected, select 1,2 or 3 nodes')
                if len(selected_nodes) < 1:
                    for node in selected_nodes:
                        node.faceColor = 'b'
                    selected_nodes.clear()
                    print('to few nodes selected, select 1,2 or 3 nodes')                
                
                # Visualize analyzed nodes and deselect
                for node in selected_nodes:
                    selected_nodes_sum.add(node)

                for node in selected_nodes_sum:
                    node.faceColor = 'g'  # make green when analyzed
                selected_nodes.clear()
                
            if event.key == vv.KEY_ESCAPE:
                # FINISH MODEL, STORE TO EXCEL
                
                # Store to EXCEL
                storeOutputToExcel(storeOutput, exceldir)
                vv.close(fig)
     
        selected_nodes = list()
        def select_node(event):
            """ select and deselect nodes by Double Click
            """
            if event.owner not in selected_nodes:
                event.owner.faceColor = 'r'
                selected_nodes.append(event.owner)
            elif event.owner in selected_nodes:
                event.owner.faceColor = 'b'
                selected_nodes.remove(event.owner)
        
        def pick_node(event):
            nodenr = event.owner.nr
            node = event.owner.node
            t0.text = '\b{Node nr|location}: %i | x=%1.3f y=%1.3f z=%1.3f' % (nodenr,node[0],node[1],node[2])
        
        def unpick_node(event):
            t0.text = '\b{Node nr|location}: '
        
        def point_pulsatility(point1, point1Deforms):
            n1Indices = point1 + point1Deforms
            pos_combinations = list(itertools.combinations(range(len(point1Deforms)),2))
            distances = []
            for i in pos_combinations:
                v = point1Deforms[i[0]] - point1Deforms[i[1]]
                distances.append(((v[0]**2 + v[1]**2 + v[2]**2)**0.5 ))
            distances = np.array(distances)
            
            # get max distance between phases
            point_phase_max = distances.max()
            point_phase_max = [point_phase_max, [x*10 for x in (pos_combinations[list(distances).index(point_phase_max)])]]
            
            # get min distance between phases
            point_phase_min = distances.min()
            point_phase_min = [point_phase_min, [x*10 for x in (pos_combinations[list(distances).index(point_phase_min)])]]
            
            return {'point_phase_min':point_phase_min,'point_phase_max': point_phase_max, 'Node1': [point1, point1Deforms]}

        def point_to_point_pulsatility(cl, point1, point1Deforms, 
                                            point2, point2Deforms,type='euclidian'):
            
            import numpy as np
            
            n1Indices = point1 + point1Deforms
            n2Indices = point2 + point2Deforms
            # define vector between nodes
            v = n1Indices - n2Indices
            distances = ( (v[:,0]**2 + v[:,1]**2 + v[:,2]**2)**0.5 ).reshape(-1,1)
            # get min and max distance
            point_to_pointMax = distances.max()
            point_to_pointMin = distances.min()
            # add phase in cardiac cycle where min and max where found (5th = 50%)
            point_to_pointMax = [point_to_pointMax, (list(distances).index(point_to_pointMax) )*10]
            point_to_pointMin = [point_to_pointMin, (list(distances).index(point_to_pointMin) )*10]
            # get median of distances
            point_to_pointMedian = np.percentile(distances, 50) # Q2
            # median of the lower half, Q1 and upper half, Q3
            point_to_pointQ1 = np.percentile(distances, 25)
            point_to_pointQ3 = np.percentile(distances, 75)
            # Pulsatility min max distance point to point
            point_to_pointP = point_to_pointMax[0] - point_to_pointMin[0]
            # add % change to pulsatility
            point_to_pointP = [point_to_pointP, (point_to_pointP/point_to_pointMin[0])*100 ]
            
            
            if isinstance(cl, PointSet):
                cl = np.asarray(cl).reshape((len(cl),3))
                
            indpoint1 = np.where( np.all(cl == point1, axis=-1) )[0] # -1 counts from last to the first axis
            indpoint2 = np.where( np.all(cl == point2, axis=-1) )[0] # renal point
            n1Indices = point1 + point1Deforms
            n2Indices = point2 + point2Deforms
            
            clDeforms = []
            clpart_deformed = []
            vectors = []
            clpart = []
            d = []
            dist_cl = []   
            clpartDeforms = []
            clpart_deformed_test = []
            
            for i in range(len(cl)):
                clDeforms1 = model.node[cl[i,0], cl[i,1], cl[i,2]]['deforms']
                clDeforms.append(clDeforms1)
                clpart_deformed1 = cl[i] + clDeforms1
                clpart_deformed.append(clpart_deformed1)
            
            # clpart = cl[min(indpoint1[0], indpoint2[0]):max(indpoint1[0], indpoint2[0])+1]
            
            clpart = clpart_deformed[min(indpoint1[0], indpoint2[0]):max(indpoint1[0], indpoint2[0])+1]
            
            # for i in range(len(clpart)):
            #     clpartDeforms1 = model.node[clpart[i,0], clpart[i,1], clpart[i,2]]['deforms']
            #     clpartDeforms.append(clpartDeforms1)
            #     clpart_deformed1_test = cl[i] + clpartDeforms1
            #     clpart_deformed_test.append(clpart_deformed1_test)
                
            # for k in range(len(n1Indices)):
            #     vectors_phases = np.vstack([clpart_deformed_test[i+1][k]-clpart_deformed_test[i][k] for i in range(len(clpart)-1)])
            #     vectors.append(vectors_phases)
                
            for k in range(len(n1Indices)):
                vectors_phases = np.vstack([clpart[i+1][k]-clpart[i][k] for i in range(len(clpart)-1)])
                vectors.append(vectors_phases)
            
            for i in range(len(vectors)):
                if type == 'euclidian':
                    d1 = (vectors[i][:,0]**2 + vectors[i][:,1]**2 + vectors[i][:,2]**2)**0.5  # 3Dvector length in mm
                    d.append(d1)
                elif type == 'z':
                    d = abs(vectors[i][:,2])  # x,y,z ; 1Dvector length in mm
             
            for i in range(len(d)):
                dist = d[i].sum()
                dist_cl.append(dist)
               
            #if indpoint2 > indpoint1: # stent point proximal to renal on centerline: positive
                #dist_cl*=-1
                
            cl_min_index1 = np.argmin(dist_cl)   
            cl_min_index = cl_min_index1*10
            cl_min = min(dist_cl)
            cl_max_index1 = np.argmax(dist_cl)   
            cl_max_index = cl_max_index1*10
            cl_max = max(dist_cl)
            
            print ([dist_cl])
            print ([point1, point2])
            
            return {'point_to_pointMin': point_to_pointMin, 'point_to_pointQ1': point_to_pointQ1, 'point_to_pointMedian': point_to_pointMedian, 'point_to_pointQ3': point_to_pointQ3, 'point_to_pointMax': point_to_pointMax, 'point_to_pointP': point_to_pointP, 'Node1': [point1, point1Deforms], 'Node2': [point2, point2Deforms], 'distances': distances, 'dist_cl': dist_cl, 'cl_min_index': cl_min_index, 'cl_max_index': cl_max_index, 'cl_min': cl_min, 'cl_max': cl_max}
            
        def line_line_angulation(point1, point1Deforms, point2, point2Deforms, point3, point3Deforms):
            n1Indices = point1 + point1Deforms
            n2Indices = point2 + point2Deforms
            n3Indices = point3 + point3Deforms
            
            # get vectors
            v1 = n1Indices - n2Indices
            v2 = n3Indices - n2Indices
            
            # get angles
            angles = []
            for i in range(len(v1)):
                angles.append(math.degrees(math.acos((np.dot(v1[i],v2[i]))/(np.linalg.norm(v1[i])*np.linalg.norm(v2[i])))))
            angles = np.array(angles)
            
            # get all angle differences of all phases
            pos_combinations = list(itertools.combinations(range(len(v1)),2))
            angle_diff = []
            for i in pos_combinations:
                v = point1Deforms[i[0]] - point1Deforms[i[1]]
                angle_diff.append(abs(angles[i[0]] - angles[i[1]]))
            angle_diff = np.array(angle_diff)
            
            # get max angle differences
            point_angle_diff_max = angle_diff.max()
            point_angle_diff_max = [point_angle_diff_max, [x*10 for x in (pos_combinations[list(angle_diff).index(point_angle_diff_max)])]]
            
            # get min angle differences
            point_angle_diff_min = angle_diff.min()
            point_angle_diff_min = [point_angle_diff_min, [x*10 for x in (pos_combinations[list(angle_diff).index(point_angle_diff_min)])]]
            
            return {'point_angle_diff_min':point_angle_diff_min,'point_angle_diff_max': point_angle_diff_max, 'angles': angles, 'Node1': [point1, point1Deforms], 'Node2': [point2, point2Deforms], 'Node3': [point3, point1Deforms]}
            
        
        def append_centerlines(allcenterlines):
            """ Merge seperated PointSet centerlines into one PointSet
            """
            cl_merged = allcenterlines[0]
            #for i in (1,2):
            for i in range(1,len(allcenterlines)):
                for point in allcenterlines[i]:
                    cl_merged.append(point)
            return cl_merged
            
            
        def get_index_name():
                # Gui for input name
                app = QtGui.QApplication([])
                m = MyDialog()
                m.show()
                m.exec_()
                dialog_output = m.edit.text()
                return dialog_output  
            
       
        def storeOutputToExcel(storeOutput, exceldir):
            """Create file and add a worksheet or overwrite existing
            """
            # https://pypi.python.org/pypi/XlsxWriter
            workbook = xlsxwriter.Workbook(os.path.join(exceldir,'storeOutput.xlsx'))
            worksheet = workbook.add_worksheet('General')
            # set column width
            worksheet.set_column('A:A', 35)
            worksheet.set_column('B:B', 30)
            # add a bold format to highlight cells
            bold = workbook.add_format({'bold': True})
            # write title and general tab
            worksheet.write('A1', 'Output ChEVAS dynamic CT, 10 Phases', bold)
            analysisID = '%s_%s_%s' % (ptcode, ctcode, cropname)
            worksheet.write('A2', 'Filename:', bold)
            worksheet.write('B2', analysisID)
            worksheet.write('A3', 'Date and Time:', bold)
            date_time = datetime.now() #strftime("%d-%m-%Y %H:%M")
            date_format_str = 'dd-mm-yyyy hh:mm'
            date_format = workbook.add_format({'num_format': date_format_str,
                                      'align': 'left'})
            worksheet.write_datetime('B3', date_time, date_format)
            # write 'storeOutput'
            sort_index = []
            for i in range(len(storeOutput)):
                type = len(storeOutput[i]['NodesIndex'])
                sort_index.append([i, type])
            sort_index = np.array(sort_index)
            sort_index = sort_index[sort_index[:,1].argsort()]
            
            for i, n in sort_index:
                worksheet = workbook.add_worksheet(storeOutput[i]['Name'])
                worksheet.set_column('A:A', 35)
                worksheet.set_column('B:B', 20)
                worksheet.write('A1', 'Name:', bold)
                worksheet.write('B1', storeOutput[i]['Name'])
                if n == 1:
                    worksheet.write('A2', 'Type:', bold)
                    worksheet.write('B2', '1 Node')
                    
                    worksheet.write('A3', 'Minimum translation (mm, Phases)',bold)
                    worksheet.write('B3', storeOutput[i]['point_phase_min'][0])
                    worksheet.write_row('C3', list(storeOutput[i]['point_phase_min'][1]))
                    
                    worksheet.write('A4', 'Maximum translation (mm, Phases)',bold)
                    worksheet.write('B4', storeOutput[i]['point_phase_max'][0])
                    worksheet.write_row('C4', list(storeOutput[i]['point_phase_max'][1]))
                    
                    worksheet.write('A5', 'Avg node position and deformations', bold)
                    worksheet.write('B5', str(list(storeOutput[i]['Node1'][0])))
                    worksheet.write_row('C5', [str(x)for x in list(storeOutput[i]['Node1'][1])])
                    
                    worksheet.write('A6', 'Node Index Number', bold)
                    worksheet.write_row('B6', list(storeOutput[i]['NodesIndex'])) 
                                       
                elif n == 2:
                    worksheet.write('A2', 'Type:', bold)
                    worksheet.write('B2', '2 Nodes')
                    
                    worksheet.write('A3', 'Minimum distance (mm, Phases)',bold)
                    worksheet.write('B3', storeOutput[i]['point_to_pointMin'][0])
                    worksheet.write('C3', storeOutput[i]['point_to_pointMin'][1])
                    
                    worksheet.write('A4', 'Q1 distance (mm)',bold)
                    worksheet.write('B4', storeOutput[i]['point_to_pointQ1'])

                    worksheet.write('A5', 'Median distance (mm)',bold)
                    worksheet.write('B5', storeOutput[i]['point_to_pointMedian'])
                    
                    worksheet.write('A6', 'Q3 distance (mm)',bold)
                    worksheet.write('B6', storeOutput[i]['point_to_pointQ3'])
                    
                    worksheet.write('A7', 'Maximum distance (mm, phases)',bold)
                    worksheet.write('B7', storeOutput[i]['point_to_pointMax'][0])
                    worksheet.write('C7', storeOutput[i]['point_to_pointMax'][1])
                    
                    worksheet.write('A8', 'Maximum distance difference (mm)', bold)
                    worksheet.write('B8', storeOutput[i]['point_to_pointP'][0])
                    
                    worksheet.write('A9', 'Distances for each phase', bold)
                    worksheet.write_row('B9', [str(x) for x in list(storeOutput[i]['distances'])])
                    
                    worksheet.write('A10', 'Avg node1 position and deformations', bold)
                    worksheet.write('B10', str(list(storeOutput[i]['Node1'][0])))
                    worksheet.write_row('C10', [str(x) for x in list(storeOutput[i]['Node1'][1])])
                    
                    worksheet.write('A11', 'Avg node2 position and deformations', bold)
                    worksheet.write('B11', str(list(storeOutput[i]['Node2'][0])))
                    worksheet.write_row('C11', [str(x) for x in list(storeOutput[i]['Node2'][1])])
                    
                    worksheet.write('A12', 'Node Index Number', bold)
                    worksheet.write_row('B12', list(storeOutput[i]['NodesIndex'])) 
                    
                    worksheet.write('A13', 'Length centerline', bold) 
                    worksheet.write('B13', str(list(storeOutput[i]['dist_cl']))) 
                    
                    worksheet.write('A14', 'Minimum length centerline', bold) 
                    worksheet.write('B14', storeOutput[i]['cl_min']) 
                    worksheet.write('C14', storeOutput[i]['cl_min_index']) 
                    
                    worksheet.write('A15', 'Maximum length centerline', bold) 
                    worksheet.write('B15', storeOutput[i]['cl_max'])  
                    worksheet.write('C15', storeOutput[i]['cl_max_index']) 
                
                elif n == 3:
                    worksheet.write('A2', 'Type:', bold)
                    worksheet.write('B2', '3 Nodes')
                    
                    worksheet.write('A3', 'Minimum angle difference (degrees, Phases)',bold)
                    worksheet.write('B3', storeOutput[i]['point_angle_diff_min'][0])
                    worksheet.write_row('C3', list(storeOutput[i]['point_angle_diff_min'][1]))
                    
                    worksheet.write('A4', 'Maximum angle difference (degrees, Phases)',bold)
                    worksheet.write('B4', storeOutput[i]['point_angle_diff_max'][0])
                    worksheet.write_row('C4', list(storeOutput[i]['point_angle_diff_max'][1]))
                    
                    worksheet.write('A5', 'Angles for each phase (degrees)',bold)
                    worksheet.write_row('B5', list(storeOutput[i]['angles']))
                    
                    worksheet.write('A6', 'Avg node1 position and deformations', bold)
                    worksheet.write('B6', str(list(storeOutput[i]['Node1'][0])))
                    worksheet.write_row('C6', [str(x) for x in list(storeOutput[i]['Node1'][1])])
                    
                    worksheet.write('A7', 'Avg node2 position and deformations', bold)
                    worksheet.write('B7', str(list(storeOutput[i]['Node2'][0])))
                    worksheet.write_row('C7', [str(x) for x in list(storeOutput[i]['Node2'][1])])
                    
                    worksheet.write('A8', 'Avg node2 position and deformations', bold)
                    worksheet.write('B8', str(list(storeOutput[i]['Node3'][0])))
                    worksheet.write_row('C8', [str(x) for x in list(storeOutput[i]['Node3'][1])])
                    
                    worksheet.write('A9', 'Node Index Number', bold)
                    worksheet.write_row('B9', list(storeOutput[i]['NodesIndex']))   
                    
            workbook.close()
        
        # Bind event handlers
        fig.eventKeyDown.Bind(on_key)
        for node_point in node_points:
            node_point.eventDoubleClick.Bind(select_node)
            node_point.eventEnter.Bind(pick_node)
            node_point.eventLeave.Bind(unpick_node)