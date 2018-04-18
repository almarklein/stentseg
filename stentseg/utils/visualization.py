# Author: M.A. Koenrades
""" Module with functionality for visualization of model
"""

import visvis as vv
from stentseg.utils.picker import pick3d
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.utils import _utils_GUI


def remove_stent_from_volume(vol, graph, stripSize=5):
    """ Give the high intensity voxels that belong to the stent a
    lower value, so that the stent appears to be "removed". This is for
    visualization purposes only. Makes use of known paths in graph model.
    """
    from visvis import Pointset

    vol2 = vol.copy()
    for n1,n2 in graph.edges():
        path = graph.edge[n1][n2]['path']
        path = Pointset(path)  # Make a visvis pointset
        stripSize2 = stripSize // 2 # floor division
        for point in path:
            z,y,x = vol2.point_to_index(point)
            vol2[z-stripSize:z+stripSize2+1, y-stripSize:y+stripSize+1, x-stripSize:x+stripSize+1] = 0
            # remove less in distal direction -> stripSize2
    return vol2


def show_ctvolume(vol, graph=None, axis=None, showVol='MIP', clim =(0,2500), isoTh=250, 
                  removeStent=True, climEditor=False):
    """ Different ways to visualize the CT volume as reference
    For '2D' clim (-550,500) often good range
    """
    import visvis as vv
    
    colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
                'g': [(0.0, 0.0), (0.27272728, 1.0)],
                'b': [(0.0, 0.0), (0.34545454, 1.0)],
                'a': [(0.0, 1.0), (1.0, 1.0)]}
    if axis is None:
        axis = vv.gca()
    axis.MakeCurrent()
    if showVol == 'MIP':
        t = vv.volshow(vol, clim=clim, renderStyle='mip')
    elif showVol == 'ISO':
        if removeStent == True:
            vol = remove_stent_from_volume(vol, graph, stripSize=6) # rings are removed for vis.
        t = vv.volshow(vol,clim=clim, renderStyle='iso')
        t.isoThreshold = isoTh; t.colormap = colormap
    elif showVol == '2D':
        t = vv.volshow2(vol); t.clim = clim
    # bind ClimEditor to figure
    if climEditor:
        if showVol=='ISO':
            c = _utils_GUI.IsoThEditor(vv.gcf())
        else:
            c = vv.ClimEditor(vv.gcf())
        c.position = (10, 50)
        # bind for show hide
        fig = vv.gcf()
        fig.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, c) )
        print('****')
        print('Use "s" to show/hide slider')
        print('****')
    
    return t


def DrawModelAxes(vol, graph=None, ax=None, axVis=False, meshColor=None, getLabel=False, 
                  mc='b', lc='g', mw=7, lw=0.6, **kwargs):
    """ Draw model with volume with axes set
    ax = axes to draw (a1 or a2 or a3); graph = sd._nodes1 or 2 or 3
    meshColor = None or faceColor e.g. 'g'
    """
    #todo: prevent TypeError: draw() got an unexpected keyword argument mc/lc when not given as required variable
    #todo: *args voor vol in drawModelAxes of **kwargs[key] in functies hieronder
    if ax is None:
        ax = vv.gca()
    ax.MakeCurrent()
    ax.daspect = 1,1,-1
    ax.axis.axisColor = 1,1,1
    ax.bgcolor = 0,0,0
    ax.axis.visible = axVis
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    if graph is None:
        show_ctvolume(vol, graph, axis=ax, removeStent=False, **kwargs) 
        label = pick3d(vv.gca(), vol)
        return label
    if hasattr(graph, 'number_of_edges'):
        if graph.number_of_edges() == 0: # get label from picked seeds sd._nodes1 
            show_ctvolume(vol, graph, axis=ax, **kwargs) 
            label = pick3d(vv.gca(), vol)
            graph.Draw(mc=mc, lc=lc)
            return label
    if not meshColor is None:
        bm = create_mesh(graph, 0.5) # (argument is strut tickness)
        m = vv.mesh(bm)
        m.faceColor = meshColor # 'g'
    show_ctvolume(vol, graph, axis=ax, **kwargs)
    graph.Draw(mc=mc, lc=lc)
    if getLabel == True:
        label = pick3d(vv.gca(), vol)
        return label
    else:
        pick3d(vv.gca(), vol)
        return


