""" Script to measure ring dynamics

run as script

"""
import sys
import os
import visvis as vv
from stentseg.utils import _utils_GUI
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import get_graph_in_phase
from stentseg.stentdirect import stentgraph
import numpy as np
#sys.path.insert(0, os.path.abspath('..'))
from lspeas.utils.get_anaconda_ringparts import get_model_struts,get_model_rings,add_nodes_edge_to_newmodel 


def on_key(event):
    """KEY commands for user interaction
    UP/DOWN = show/hide nodes
    ENTER   = show edge and attribute values [select 2 nodes]
    DELETE  = remove edge [select 2 nodes]
    CTRL    = replace intially created ringparts
    ESCAPE  = FINISH: refine, smooth
    """
    if event.key == vv.KEY_DOWN:
        # hide nodes
        t1.visible = False
        t2.visible = False
        t3.visible = False
        for node_point in node_points:
            node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes
        for node_point in node_points:
            node_point.visible = True
    if event.key == vv.KEY_ENTER:
        select1 = selected_nodes[0].node
        select2 = selected_nodes[1].node
        c, ct, p, l = _utils_GUI.get_edge_attributes(model, select1, select2)
        # visualize edge and deselect nodes
        selected_nodes[1].faceColor = 'b'
        selected_nodes[0].faceColor = 'b'
        selected_nodes.clear()
        _utils_GUI.set_edge_labels(t1,t2,t3,ct,c,l)
        a = vv.gca()
        view = a.GetView()
        pp = Pointset(p)  # visvis meshes do not work with PointSet
        line = vv.solidLine(pp, radius = 0.2)
        line.faceColor = 'y'
        a.SetView(view)
    if event.key == vv.KEY_DELETE:
        # remove edge
        assert len(selected_nodes) == 2
        select1 = selected_nodes[0].node
        select2 = selected_nodes[1].node
        c, ct, p, l = _utils_GUI.get_edge_attributes(model, select1, select2)
        model.remove_edge(select1, select2)
        # visualize removed edge, show keys and deselect nodes
        selected_nodes[1].faceColor = 'b'
        selected_nodes[0].faceColor = 'b'
        selected_nodes.clear()
        _utils_GUI.set_edge_labels(t1,t2,t3,ct,c,l)
        a = vv.gca()
        view = a.GetView()
        pp = Pointset(p)
        line = vv.solidLine(pp, radius = 0.2)
        line.faceColor = 'r'
        a.SetView(view)
    if event.key == vv.KEY_CONTROL:
        # replace intially created ringparts
        ringparts(ringpart=ringpart)
        figparts() 
#     if event.key == vv.KEY_ALT:
#         # add edge to struts
#         assert len(selected_nodes) == 2
#         select1 = selected_nodes[0].node
#         select2 = selected_nodes[1].node
#         add_nodes_edge_to_newmodel(models[0][0], model, select1, select2)
#         add_nodes_edge_to_newmodel(models[0][3], model, select1, select2)
        #todo: does this make sense? add strut should be before get ringparts?


models, modelsR1R2 = [None], [None] #init to modify variable in on_key
def ringparts(ringpart = True):
    if ringpart:
        models[0] = get_model_struts(model, nstruts=nstruts) # [0] tuple in list
        modelsR1R2[0] = get_model_rings(models[0][2]) # [2]=model_R1R2
        

def figparts():
    """ Visualize ring parts
    """
    fig = vv.figure(4);
    fig.position = 8.00, 30.00,  944.00, 1002.00
    vv.clf()
    a0 = vv.subplot(121)
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh)
    modelR1, modelR2 = modelsR1R2[0][0], modelsR1R2[0][1]
    modelR1.Draw(mc='g', mw = 10, lc='g') # R1 = green
    modelR2.Draw(mc='c', mw = 10, lc='c') # R2 = cyan
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    a0.axis.axisColor= 1,1,1
    a0.bgcolor= 0,0,0
    a0.daspect= 1, 1, -1  # z-axis flipped
    a0.axis.visible = showAxis
    
    a1 = vv.subplot(122)
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh)
    models[0][0].Draw(mc='y', mw = 10, lc='y') # struts = yellow
    models[0][1].Draw(mc='r', mw = 10, lc='r') # hooks = red
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    a1.axis.axisColor= 1,1,1
    a1.bgcolor= 0,0,0
    a1.daspect= 1, 1, -1  # z-axis flipped
    a1.axis.visible = showAxis
    
    a0.camera = a1.camera


def _fit3D(model):
    from stentseg.utils import fitting
    from stentseg.utils.new_pointset import PointSet
    pp3 = PointSet(3)
    l, l2 = 0, 0
    for n in model.nodes():
        pp3.append(n)
    for n1, n2 in model.edges():
        path = model.edge[n1][n2]['path'][1:-1] # do not include nodes to prevent duplicates
        for p in path:
            pp3.append(p)
    plane = fitting.fit_plane(pp3) # todo: plane niet intuitief door saddle ring; maakt aantal punten uit?? toch loodrecht op centerline stent?
    pp3_2 = fitting.project_to_plane(pp3, plane)
#     c3 = fitting.fit_circle(pp3_2)
    e3 = fitting.fit_ellipse(pp3_2)
#     print('area circle 3D: % 1.2f' % fitting.area(c3))
#     print('area ellipse 3D: % 1.2f' % fitting.area(e3))
    
    return pp3, plane, pp3_2, e3


def vis3Dfit(fitted, vol, model, ptcode, ctcode, showAxis, **kwargs):
    """Visualize ellipse fit in 3D with CT volume in current axis
    input: _fit3D output
    """
    from stentseg.utils import fitting
    import numpy as np
    pp3,plane,pp3_2,e3 = fitted[0],fitted[1],fitted[2],fitted[3]
    a = vv.gca()
    # show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh)
    show_ctvolume(vol, model, **kwargs)
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Ellipse fit for model %s  -  %s' % (ptcode[7:], ctcode))
    a.axis.axisColor= 1,1,1
    a.bgcolor= 0,0,0
    a.daspect= 1, 1, -1  # z-axis flipped
    a.axis.visible = showAxis
    # For visualization, calculate 4 points on rectangle that lies on the plane
    x1, x2 = pp3.min(0)[0]-0.3, pp3.max(0)[0]+0.3
    y1, y2 = pp3.min(0)[1]-0.3, pp3.max(0)[1]+0.3
    p1 = x1, y1, -(x1*plane[0] + y1*plane[1] + plane[3]) / plane[2]
    p2 = x2, y1, -(x2*plane[0] + y1*plane[1] + plane[3]) / plane[2]
    p3 = x2, y2, -(x2*plane[0] + y2*plane[1] + plane[3]) / plane[2]
    p4 = x1, y2, -(x1*plane[0] + y2*plane[1] + plane[3]) / plane[2]
    
    vv.plot(pp3, ls='', ms='.', mc='y', mw = 10)
    vv.plot(fitting.project_from_plane(pp3_2, plane), lc='r', ls='', ms='.', mc='r', mw=9)
    #     vv.plot(fitting.project_from_plane(fitting.sample_circle(c3), plane), lc='r', lw=2)
    vv.plot(fitting.project_from_plane(fitting.sample_ellipse(e3), plane), lc='b', lw=2)
    vv.plot(np.array([p1, p2, p3, p4, p1]), lc='g', lw=2)
    #     vv.legend('3D points', 'Projected points', 'Circle fit', 'Ellipse fit', 'Plane fit')
    vv.legend('3D points', 'Projected points', 'Ellipse fit', 'Plane fit')

def vis2Dfit(fitted):
    """Visualize ellipse fit in 2D in current axis
    input: _fit3D output
    """
    from stentseg.utils import fitting
    import numpy as np
    plane,pp3_2,e3 = fitted[1],fitted[2],fitted[3]
    # endpoints axis
    x0,y0,res1,res2,phi = e3[0], e3[1], e3[2], e3[3], e3[4]
    dxax1 = np.cos(phi)*res1 
    dyax1 = np.sin(phi)*res1
    dxax2 = np.cos(phi+0.5*np.pi)*res2 
    dyax2 = np.sin(phi+0.5*np.pi)*res2
    p1ax1, p2ax1 = (x0+dxax1, y0+dyax1), (x0-dxax1, y0-dyax1)
    p1ax2, p2ax2 = (x0+dxax2, y0+dyax2), (x0-dxax2, y0-dyax2)
      
    a = vv.gca()
    vv.xlabel('x (mm)');vv.ylabel('y (mm)')
    vv.title('Ellipse fit for model %s  -  %s' % (ptcode[7:], ctcode))
    a.axis.axisColor= 0,0,0
    a.bgcolor= 0,0,0
    a.axis.visible = showAxis
    a.daspectAuto = False
    a.axis.showGrid = True
    vv.plot(pp3_2, ls='', ms='.', mc='r', mw=9)
    vv.plot(fitting.sample_ellipse(e3), lc='b', lw=2)
    vv.plot(np.array([p1ax1, p2ax1]), lc='w', lw=2) # major axis
    vv.plot(np.array([p1ax2, p2ax2]), lc='w', lw=2) # minor axis
    vv.legend('3D points projected to plane', 'Ellipse fit on projected points')

def ellipse2excel(exceldir, analysisID, e3top=None, e3bot=None):
    """ Create/overwrite excel and store ellipse output
    Input:  exceldir string file location
            analysisID string for in excel
            e3top tuple with 5 elements x0, y0, res1, res2, phi (optional)
            e3bot tuple with 5 elements x0, y0, res1, res2, phi (optional)
    """
    from stentseg.utils import fitting
    import xlsxwriter
    
    workbook = xlsxwriter.Workbook(os.path.join(exceldir,'storeOutputTemplate.xlsx'))
    worksheet = workbook.add_worksheet()
    # set column width
    worksheet.set_column('A:L', 15)
    # add a bold format to highlight cells
    bold = workbook.add_format({'bold': True})
    worksheet.write('B4', analysisID, bold)
    
    if e3top:
        res1 = e3top[2] # radius
        res2 = e3top[3]
        phi = e3top[4]*180.0/np.pi # direction vector in degrees
        area = fitting.area(e3top)
        # write titles
        worksheet.write('B6', 'minor axis (mm)', bold)
        worksheet.write('C6', 'major axis (mm)', bold)
        worksheet.write('D6', 'axis mean (mm)', bold)
        worksheet.write('E6', 'axis angle (degrees)', bold)
        worksheet.write('F6', 'area (mm2)', bold)
        # store
        rowstart = 6
        columnstart = 1
        worksheet.write_row(rowstart, columnstart, [res1*2]) # row, columnm, variable
        worksheet.write_row(rowstart, columnstart+1, [res2*2])
        worksheet.write_row(rowstart, columnstart+2, [(res2*2+res1*2)/2])
        worksheet.write_row(rowstart, columnstart+3, [phi])
        worksheet.write_row(rowstart, columnstart+4, [area])
    
    if e3bot:
        res1 = e3bot[2] # radius
        res2 = e3bot[3]
        phi = e3bot[4]*180.0/np.pi # direction vector in degrees
        area = fitting.area(e3bot)
        # write titles
        worksheet.write('G6', 'minor axis (mm)', bold)
        worksheet.write('H6', 'major axis (mm)', bold)
        worksheet.write('I6', 'axis mean (mm)', bold)
        worksheet.write('J6', 'axis angle (degrees)', bold)
        worksheet.write('K6', 'area (mm2)', bold)
        # store
        columnstart += 5
        worksheet.write_row(rowstart, columnstart, [res1*2]) # row, columnm, variable
        worksheet.write_row(rowstart, columnstart+1, [res2*2])
        worksheet.write_row(rowstart, columnstart+2, [(res2*2+res1*2)/2])
        worksheet.write_row(rowstart, columnstart+3, [phi])
        worksheet.write_row(rowstart, columnstart+4, [area])
        
    workbook.close()
    print('--- stored to excel: {}-- {}'.format(exceldir, analysisID) )
    return
    
    
def calculateAreaChange(model, mname):
    """
    """
    import numpy as np
    areas, res1s, res2s, phis = [], [], [], []
    #todo: change axis change definition 
    for phasenr in range(10):
        model_phase = get_graph_in_phase(model, phasenr = phasenr)
        fitted = _fit3D(model_phase)
        e3 = fitted[3] # [3] is e3 with 5 elements x0, y0, res1, res2, phi
        area = fitting.area(e3)
        areas.append(area)
        res1s.append(e3[2])
        res2s.append(e3[3])
        phis.append(e3[4])
        
    areaMax = max(areas)
    areaMin = min(areas)
    areaChangemm = areaMax-areaMin
    areaChangepr = 100*(areaMax-areaMin) / areaMin
    des1Min, des1Max = 2*min(res1s), 2*max(res1s)
    des1Change = des1Max - des1Min
    des2Min, des2Max = 2*min(res2s), 2*max(res2s)
    des2Change = des2Max - des2Min
    phiMin, phiMax = min(phis)*180.0/np.pi, max(phis)*180.0/np.pi # degrees
    phiChange = phiMax - phiMin
    print('   ')
    print('=== Ring measurements %s ===' % mname)
    print('Min area: %1.1f mm2' % areaMin)
    print('Max area: %1.1f mm2' % areaMax)
    print('Area change: %1.1f mm2 ; %1.1f%%' % (areaChangemm, areaChangepr))
    print('Minor axis: %1.1f-%1.1f (%1.2f) mm' % (des2Min, des2Max, des2Change) )
    print('Major axis: %1.1f-%1.1f (%1.2f) mm' % (des1Min, des1Max, des1Change) )
    print('Angle axis: %1.1f-%1.1f (%1.1f) degrees' % (phiMin, phiMax, abs(phiChange))    )
    
    return areaMax, areaMin, areaChangemm, areaChangepr

def getTopBottomNodesZring(model,nTop=5,nBot=5):
    """Get top and bottom nodes in endurant proximal ring, based on z
    return graphs
    """
    nSorted = np.asarray(sorted(model.nodes(), key=lambda x: x[2])) # sort by z ascending
    nTop = nSorted[:nTop]
    nBot = nSorted[nBot:]
    m_nTop = stentgraph.StentGraph()
    m_nBot = stentgraph.StentGraph()
    for p in nTop:
        m_nTop.add_node(tuple(p.flat))
    for p in nBot:
        m_nBot.add_node(tuple(p.flat))
    
    return m_nTop, m_nBot
    

##

if __name__ == '__main__':

    from stentseg.utils import fitting
    
    # Select the ssdf basedir
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup', r'G:\LSPEAS_ssdf_backup')
    
    # Select dataset to register
    ptcode = 'LSPEAS_021'
    ctcode = 'discharge'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    # Load static CT image to add as reference
    s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
    vol = s.vol
    
    # Load the stent model and mesh
    s2 = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
    model = s2.model
    modelmesh = create_mesh(model, 0.6)  # Param is thickness
    
    showAxis = True  # True or False
    showVol  = 'MIP'  # MIP or ISO or 2D or None
    ringpart = True # True; False
    nstruts = 8
    clim0  = (0,3000)
    # clim0 = -550,500
    clim2 = (0,4)
    radius = 0.07
    dimensions = 'xyz'
    isoTh = 250

    ## Visualize with GUI
    f = vv.figure(3); vv.clf()
    f.position = 968.00, 30.00,  944.00, 1002.00
    a = vv.gca()
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh)
    model.Draw(mc='b', mw = 10, lc='g')
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    a.axis.axisColor= 1,1,1
    a.bgcolor= 0,0,0
    a.daspect= 1, 1, -1  # z-axis flipped
    a.axis.visible = showAxis
    
    # Initialize labels GUI
    from visvis import Pointset
    from stentseg.stentdirect import stentgraph
    
    t1 = vv.Label(a, 'Edge ctvalue: ', fontSize=11, color='c')
    t1.position = 0.1, 5, 0.5, 20  # x (frac w), y, w (frac), h
    t1.bgcolor = None
    t1.visible = False
    t2 = vv.Label(a, 'Edge cost: ', fontSize=11, color='c')
    t2.position = 0.1, 25, 0.5, 20
    t2.bgcolor = None
    t2.visible = False
    t3 = vv.Label(a, 'Edge length: ', fontSize=11, color='c')
    t3.position = 0.1, 45, 0.5, 20
    t3.bgcolor = None
    t3.visible = False
    
    #Add clickable nodes
    node_points = _utils_GUI.interactive_node_points(model)
    
    selected_nodes = list()
    # Bind event handlers
    f.eventKeyDown.Bind(on_key)
    for node_point in node_points:
        node_point.eventDoubleClick.Bind(lambda event: _utils_GUI.select_node(event, selected_nodes) )
    print('')
    print('UP/DOWN = show/hide nodes')
    print('ENTER   = show edge and attribute values [select 2 nodes]')
    print('DELETE  = remove edge [select 2 nodes]')
    print('CTRL    = replace intially created ringparts')
    print('')

    # Get ring parts
    ringparts(ringpart=ringpart)
    
    # Visualize ring parts
    if ringpart:
        figparts()
        
    # Area and cyclic change -- ellipse fit
    if ringpart:
        # fit plane and ellipse
        R1 = modelsR1R2[0][0]
        R2 = modelsR1R2[0][1]
        fittedR1, fittedR2, fittedR1R2 = _fit3D(R1), _fit3D(R2), _fit3D(models[0][4]) # [4]=model_noHooks
        print("------------")
        f = vv.figure(); vv.clf()
        f.position = 258.00, 30.00,  1654.00, 1002.00
        a1 = vv.subplot(231)
        vis3Dfit(fittedR1,vol,model,ptcode,ctcode,showAxis,showVol=showVol, clim=clim0, isoTh=isoTh)
        a2 = vv.subplot(232)
        vis3Dfit(fittedR2,vol,model,ptcode,ctcode,showAxis,showVol=showVol, clim=clim0, isoTh=isoTh)
        a3 = vv.subplot(233)
        vis3Dfit(fittedR1R2,vol,model,ptcode,ctcode,showAxis,showVol=showVol, clim=clim0, isoTh=isoTh)
        a1.camera = a2.camera = a3.camera
        a1b = vv.subplot(2,3,4)
        vis2Dfit(fittedR1)
        a2b = vv.subplot(2,3,5)
        vis2Dfit(fittedR2)
        a3b = vv.subplot(2,3,6)
        vis2Dfit(fittedR1R2)    
        # cyclic change
        A_R1 = calculateAreaChange(R1, 'R1')
        A_R2 = calculateAreaChange(R2, 'R2')
        A_noHooks = calculateAreaChange(models[0][4],'R1R2struts')
    
#         f = vv.figure(); vv.clf()
#         f.position = 968.00, 30.00,  944.00, 1002.00
#         a = vv.gca()
#         show_ctvolume(vol[:,vol.shape[1]*0.65:,:], model, showVol=showVol, clim=clim0, isoTh=isoTh)
#         color = 'rgbmcrywgb'
#         for phasenr in range(10):
#             model_phase = get_graph_in_phase(R1, phasenr = phasenr)
#             model_phase.Draw(mc=color[phasenr], lc=color[phasenr])
#         a.axis.axisColor= 1,1,1
#         a.bgcolor= 0,0,0
#         a.daspect= 1, 1, -1  # z-axis flipped
#         a.axis.visible = True
    
    
#     # 3D, longitudinal and lateral motion
#     from stentseg.utils.new_pointset import PointSet
#     pp3 = PointSet(3)
#     for n1, n2 in R1.edges():
#         path = model.edge[n1][n2]['path']
#         for p in path:
#             if p not in path: 
#                 pp3.append(p)
    
#     
#     dmax = 0.0
#     for i in range()
    
    
#     print('Longitudinal motion: %1.1f mm' % zMotion)
#     print('Motion magnitude: %1.1f mm' % motionMag)
    
    # Angle hook-strut and cyclic change
    
    
    # Curvature rings 


