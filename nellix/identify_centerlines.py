""" Author: M.A. Koenrades
Created October 2017
Module to identify the anatomical location of the centerlines by the user
"""
import visvis as vv
import os

def get_index_name():
    try:
        from PyQt5 import QtCore, QtGui # PyQt5; conda install pyqt
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
    except ImportError:
        from PySide import QtCore, QtGui # PySide2
        from PySide.QtGui import QApplication
    from stentseg.apps.ui_dialog import MyDialog
    
    # Gui for input name
    m = MyDialog()
    m.show()
    m.exec_()
    dialog_output = m.edit.text()
    return dialog_output  


def interactiveCenterlineID(s,ptcode,ctcode,basedir,cropname,modelname, 
        radius=0.7, axVis=True, faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) ):
    """ showGraphAsMesh(graph, radius=0.7, 
                faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) )
    
    Manual identidy centerlines; s contains models. 
    Show the given graphs as a mesh, or to be more precize as a set of meshes 
    representing the centerlines in the struct. 
    By holding the mouse over a mesh, it can be selected, after which it can be 
    identified by pressing enter.
    
    Returns the axes in which the meshes are drawn and s2 with new named models.
    
    """
    import visvis as vv
    import networkx as nx
    from stentseg.stentdirect import stentgraph
    from stentseg.stentdirect.stentgraph import create_mesh
    # from nellix._select_centerline_points import _Select_Centerline_Points
    import copy
    
    print("Move mouse over centerlines and press ENTER to identify")
    print("Give either NelL, NelR, ChL, ChR, SMA")
    print("Press ESCAPE to save ssdf and finish")
    # Get clusters of nodes from each centerline
    clusters = []
    meshes = []
    s2 = copy.copy(s)
    for key in s:
        if key.startswith('model'):
            clusters.append(s[key])
            del s2[key]
            # Convert to mesh (this takes a while)
            bm = create_mesh(s[key], radius = radius, fullPaths=False)
            # Store
            meshes.append(bm)
    
    centerlines = [None]*len(clusters)
    # Define callback functions
    def meshEnterEvent(event):
        event.owner.faceColor = selectColor
    def meshLeaveEvent(event):
        if event.owner.hitTest: # True
            event.owner.faceColor = faceColor
        else:
            event.owner.faceColor = 'b'
    def figureKeyEvent(event):
        if event.key == vv.KEY_ENTER:
            m = event.owner.underMouse
            if hasattr(m, 'faceColor'):
                m.faceColor = 'y'
                dialog_output = get_index_name()
                name = dialog_output
                if name in ['NelL', 'NelR', 'ChL', 'ChR', 'SMA']:
                    s2['model'+name] = clusters[m.index]
                    m.hitTest = False
                else:
                    print("Name entered not known, give either NelL, NelR, ChL, ChR, SMA")
        if event.key == vv.KEY_ESCAPE:
            # Save ssdf
            filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname+'_id')
            s3 = copy.deepcopy(s2) # do not change s2
            for key in s3:
                if key.startswith('model'):
                    s3[key] = s3[key].pack()
            vv.ssdf.save(os.path.join(basedir, ptcode, filename), s3)
            print("Finished, ssdf {} saved to disk".format(filename) )
            # fig.Destroy() # warning?
     
    # Visualize
    a = vv.gca()
    fig = a.GetFigure()
    for i, bm in enumerate(meshes):
        m = vv.mesh(bm)
        m.faceColor = faceColor
        m.eventEnter.Bind(meshEnterEvent)
        m.eventLeave.Bind(meshLeaveEvent)
        m.hitTest = True
        m.index = i
    # Bind event handlers to figure
    fig.eventKeyDown.Bind(figureKeyEvent)
    a.SetLimits()
    a.bgcolor = 'k'
    a.axis.axisColor = 'w'
    a.axis.visible = axVis
    a.daspect = 1, 1, -1
    
    # Prevent the callback functions from going out of scope
    a._callbacks = meshEnterEvent, meshLeaveEvent, figureKeyEvent
    
    # Done return axes and s2 with new named centerlines
    return a, s2


def identifyCenterlines(s,ptcode,ctcode,basedir,modelname,showVol='MIP',**kwargs):
    """ s is struct with models
    """
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
    from stentseg.utils.picker import pick3d
    from stentseg.utils.visualization import show_ctvolume
    from stentseg.utils import _utils_GUI

    showAxis = False
    # showVol  = 'MIP'  # MIP or ISO or 2D or None
    clim = (0,2500)
    # clim = -200,500 # 2D
    isoTh = 250
    cropname = 'prox'
    
    # init fig
    f = vv.figure(); vv.clf()
    f.position = 0.00, 22.00,  1920.00, 1018.00
    # load vol
    svol = loadvol(basedir, ptcode, ctcode, cropname, what='avgreg')
    # set sampling for cases where this was not stored correctly
    svol.vol.sampling = [svol.sampling[1], svol.sampling[1], svol.sampling[2]]
    vol = svol.vol
    s.sampling = [svol.sampling[1], svol.sampling[1], svol.sampling[2]] # for model
    # show vol
    t = show_ctvolume(vol, showVol=showVol, clim=clim, isoTh=isoTh, **kwargs)
    label = pick3d(vv.gca(), vol)
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    
    if showVol=='MIP':
        c = vv.ClimEditor(vv.gca())
        c.position = (10, 50)
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, c) )
        print('Use "s" to show/hide slider')
    if showVol=='ISO':
        c = _utils_GUI.IsoThEditor(vv.gca())
        c.position = (10, 50)
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, c) )
        print('Use "s" to show/hide slider')
    
    f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [vv.gca()]) )
    print('------------------------')
    print('Use keys 1, 2, 3, 4 and 5 for preset anatomic views')
    print('Use v for a default zoomed view')
    print('Use x to show and hide axis')
    print('------------------------')
    
    ax, s2 = interactiveCenterlineID(s,ptcode,ctcode,basedir,cropname,modelname)

    return ax, s2

