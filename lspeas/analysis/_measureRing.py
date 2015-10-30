""" Script to measure ring dynamics

run as script

"""
import sys
import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import show_ctvolume, get_graph_in_phase

sys.path.insert(0, os.path.abspath('..'))
from get_anaconda_ringparts import get_model_struts,get_model_rings,add_nodes_edge_to_newmodel 
import _utils_GUI

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_003'
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


showAxis = False  # True or False
showVol  = 'MIP'  # MIP or ISO or 2D or None
ringpart = True # True; False
nstruts = 8
clim0  = (0,4000)
clim2 = (0,4)
clim3 = -550,500
radius = 0.07
dimensions = 'xyz'
isoTh = 250


## Visualize with GUI
f = vv.figure(3); vv.clf()
f.position = 968.00, 30.00,  944.00, 1002.00
a = vv.gca()
show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
model.Draw(mc='b', mw = 10, lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
a.axis.axisColor= 1,1,1
a.bgcolor= 0,0,0
a.daspect= 1, 1, -1  # z-axis flipped
a.axis.visible = showAxis

## Initialize labels GUI
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
node_points = _utils_GUI.create_node_points(model) 

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
        # Visualize edge and deselect nodes
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
        # Visualize removed edge, show keys and deselect nodes
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

f.eventKeyDown.Bind(on_key)
for node_point in node_points:
    node_point.eventDoubleClick.Bind(select_node)
print('')
print('UP/DOWN = show/hide nodes')
print('ENTER   = show edge and attribute values [select 2 nodes]')
print('DELETE  = remove edge [select 2 nodes]')
print('CTRL    = replace intially created ringparts')
print('')


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
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
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
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
    models[0][0].Draw(mc='y', mw = 10, lc='y') # struts = yellow
    models[0][1].Draw(mc='r', mw = 10, lc='r') # hooks = red
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    a1.axis.axisColor= 1,1,1
    a1.bgcolor= 0,0,0
    a1.daspect= 1, 1, -1  # z-axis flipped
    a1.axis.visible = showAxis
    
    a0.camera = a1.camera


def fit3D(model):
    from stentseg.utils import fitting
    from stentseg.utils.new_pointset import PointSet
    pp3 = PointSet(3)
    for n1, n2 in model.edges(): # mind that node point duplicates are added
        path = model.edge[n1][n2]['path']
        for p in path: 
            pp3.append(p)
    plane = fitting.fit_plane(pp3) # todo: plane niet intuitief door saddle ring; maakt aantal punten uit?? toch loodrecht op centerline stent?
    pp3_2 = fitting.project_to_plane(pp3, plane)
    c3 = fitting.fit_circle(pp3_2)
    e3 = fitting.fit_ellipse(pp3_2)
    print('area circle 3D: % 1.2f' % fitting.area(c3))
    print('area ellipse 3D: % 1.2f' % fitting.area(e3))
    
    return pp3, plane, pp3_2, e3


def fig3Dfit(fitted):
    from stentseg.utils import fitting
    import numpy as np
    pp3,plane,pp3_2,e3 = fitted[0],fitted[1],fitted[2],fitted[3]
    f = vv.figure(); vv.clf()
    f.position = 968.00, 30.00,  944.00, 1002.00
    a = vv.gca()
    show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Analysis for model LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
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
    
    vv.plot(pp3, ls='', ms='.', mc='y')
    vv.plot(fitting.project_from_plane(pp3_2, plane), lc='r', ls='', ms='.', mc='r', mw=4)
    #     vv.plot(fitting.project_from_plane(fitting.sample_circle(c3), plane), lc='r', lw=2)
    vv.plot(fitting.project_from_plane(fitting.sample_ellipse(e3), plane), lc='b', lw=2)
    vv.plot(np.array([p1, p2, p3, p4, p1]), lc='g', lw=2)
    #     vv.legend('3D points', 'Projected points', 'Circle fit', 'Ellipse fit', 'Plane fit')
    vv.legend('3D points', 'Projected points', 'Ellipse fit', 'Plane fit')








if __name__ == '__main__':

    # Get ring parts
    ringparts(ringpart=ringpart)
    
    # Visualize ring parts
    if ringpart:
        figparts()
        
    # Area and cyclic change
    if ringpart:
        # fit plane and ellipse
        R1 = modelsR1R2[0][0]
        R2 = modelsR1R2[0][1]
        modelsR1R2 = modelsR1R2[0]
        fittedR1, fittedR2, fittedR1R2 = fit3D(R1), fit3D(R2), fit3D(models[0][2]) # [2]=model_R1R2 
        fig3Dfit(fittedR1)
        fig3Dfit(fittedR2)
        fig3Dfit(fittedR1R2)    
        # cyclic change
        e3R1 = fittedR1[3] # 5 elements x0, y0, res1, res2, phi
        
        f = vv.figure(); vv.clf()
        f.position = 968.00, 30.00,  944.00, 1002.00
        a = vv.gca()
        show_ctvolume(vol, model, showVol=showVol, clim=clim0, isoTh=isoTh, clim3=clim3)
        color = 'rgbmcrywgb'
        for phasenr in range(10):
            model_phase = get_graph_in_phase(R1, phasenr = phasenr)
#             model_phase.Draw(mc=color[phasenr], mw = 10, lc=color[phasenr])
            model_phase.Draw(mc='', lc=color[phasenr])
        a.axis.axisColor= 1,1,1
        a.bgcolor= 0,0,0
        a.daspect= 1, 1, -1  # z-axis flipped
        a.axis.visible = showAxis
    
    
    
    # 3D, longitudinal and lateral motion
    
    
    # Angle hook-strut and cyclic change
    
    
    # Curvature rings 






    
    