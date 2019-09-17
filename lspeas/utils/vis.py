""" LSPEAS visualization module
Created April 18, 2017. Maaike Koenrades.
"""

import visvis as vv
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import show_ctvolume
import numpy as np
from stentseg.utils.centerline import points_from_mesh
from stentseg.utils import _utils_GUI
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

def showModel3d(basedir,ptcode, ctcode, cropname='ring', showVol='MIP',
    showmodel=True, graphname='model', **kwargs):
    """ show model and vol in 3d by mip iso or 2D
    graphname 'all' draws all graph models stored in s, if multiple
    """
    s = loadmodel(basedir, ptcode, ctcode, cropname, modelname='modelavgreg')
    vol = loadvol(basedir, ptcode, ctcode, cropname, what='avgreg').vol
    # figure
    f = vv.figure(); vv.clf()
    f.position = 0.00, 22.00,  1920.00, 1018.00
    a = vv.gca()
    a.axis.axisColor = 1,1,1
    a.axis.visible = False
    a.bgcolor = 0,0,0
    a.daspect = 1, 1, -1
    t = show_ctvolume(vol, axis=a, showVol=showVol, removeStent=False, 
                        climEditor=True, **kwargs)
    label = pick3d(a, vol)
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    if showmodel:
        if graphname == 'all':
            for key in dir(s):
                if key.startswith('model'):
                    s[key].Draw(mc='b', mw = 5, lc='g', alpha = 0.5)
        else:
            s[graphname].Draw(mc='b', mw = 5, lc='g', alpha = 0.5)
            
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[8:], ctcode))
    
    # f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a], axishandling=False ))
    f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a]) )
    
    return f, a, label, s

def showModelsStatic(ptcode,codes, vols, ss, mm, vs, showVol, clim, isoTh, clim2, 
    clim2D, drawMesh=True, meshDisplacement=True, drawModelLines=True, 
    showvol2D=False, showAxis=False, drawVessel=False, vesselType=1,
    meshColor=None, **kwargs):
    """ show one to four models in multipanel figure. 
    Input: arrays of codes, vols, ssdfs; params from show_models_static
    Output: axes, colorbars 
    """
    # init fig
    f = vv.figure(1); vv.clf()
    # f.position = 0.00, 22.00,  1920.00, 1018.00
    mw = 5
    if drawMesh == True:
        lc = 'w'
        meshColor = meshColor
    else:
        lc = 'g'
    # create subplots
    if isinstance(codes, str): # if 1 ctcode, otherwise tuple of strings
        a1 = vv.subplot(111)
        axes = [a1]
    elif codes == (codes[0],codes[1]):
        a1 = vv.subplot(121)
        a2 = vv.subplot(122)
        axes = [a1,a2]
    elif codes == (codes[0],codes[1], codes[2]):
        a1 = vv.subplot(131)
        a2 = vv.subplot(132)
        a3 = vv.subplot(133)
        axes = [a1,a2,a3]
    elif codes == (codes[0],codes[1], codes[2], codes[3]):
        a1 = vv.subplot(141)
        a2 = vv.subplot(142)
        a3 = vv.subplot(143)
        a4 = vv.subplot(144)
        axes = [a1,a2,a3,a4]
    elif codes == (codes[0],codes[1], codes[2], codes[3], codes[4]):
        a1 = vv.subplot(151)
        a2 = vv.subplot(152)
        a3 = vv.subplot(153)
        a4 = vv.subplot(154)
        a5 = vv.subplot(155)
        axes = [a1,a2,a3,a4,a5]
    else:
        a1 = vv.subplot(111)
        axes = [a1]
    for i, ax in enumerate(axes):
        ax.MakeCurrent()
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], codes[i]))
        t = show_ctvolume(vols[i], ss[i].model, axis=ax, showVol=showVol, clim=clim, isoTh=isoTh, **kwargs)
        label = pick3d(ax, vols[i])
        if drawModelLines == True:
            ss[i].model.Draw(mc='b', mw = mw, lc=lc)
    if showvol2D:
        for i, ax in enumerate(axes):
            t2 = vv.volshow2(vols[i], clim=clim2D, axes=ax)
    cbars = [] # colorbars
    if drawMesh:
        for i, ax in enumerate(axes):
            m = vv.mesh(mm[i], axes=ax)
            if meshDisplacement:
                m.clim = clim2
                m.colormap = vv.CM_JET #todo: use colormap Viridis or Magma as JET is not linear (https://bids.github.io/colormap/)
                cb = vv.colorbar(ax)
                cbars.append(cb)
            elif meshColor is not None:
                if len(meshColor) == 1:
                    m.faceColor = meshColor[0] # (0,1,0,1)
                else:
                    m.faceColor = meshColor[i]
            else:
                m.faceColor = 'g'
    if drawVessel:
        for i, ax in enumerate(axes):
            v = showVesselMesh(vs[i], ax, type=vesselType)
    for ax in axes:
        ax.axis.axisColor = 1,1,1
        ax.bgcolor = 25/255,25/255,112/255 # midnightblue
        # http://cloford.com/resources/colours/500col.htm
        ax.daspect = 1, 1, -1  # z-axis flipped
        ax.axis.visible = showAxis
    # set colorbar position
    for cbar in cbars:
        p1 = cbar.position
        cbar.position = (p1[0], 20, p1[2], 0.98) # x,y,w,h
    
    # bind rotate view and view presets [1,2,3,4,5]
    f = vv.gcf()
    f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event,axes,axishandling=False) )
    f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event,axes) )
    
    return axes, cbars


def showVesselMesh(vesselstl, ax=None, vesselMeshColor=(1,0,0,0.5),
    type = 1, **kwargs):
    """ plot pointcloud of mesh in ax
    type = 1 use vv.mesh; type = 2 use vv.plot
    """
    if ax is None:
        ax = vv.gca()
    if vesselstl is None:
        return
    if type == 1:
        vessel = vv.mesh(vesselstl, axes=ax, **kwargs)
        vessel.faceColor = vesselMeshColor # (1,0,0,0.5) or alpha 0.4-0.6
        return vessel
    elif type == 2:
        # get PointSet from STL 
        ppvessel = points_from_mesh(vesselstl, invertZ = False) # removes duplicates
        vv.plot(ppvessel, ms='.', ls='', mc= 'r', alpha=0.2, mw = 7, axes = ax)
        return ppvessel


def showVolPhases(basedir, vols=None, ptcode=None, ctcode=None, cropname=None, 
    showVol='iso', mipIsocolor=False, isoTh=310,
            slider=False, clim=(0,3000), clim2D=(-550, 500), fname=None):
    """ Show vol phases in motion container
    showVol= mip or iso or 2D; Provide either vols or location
    """
    if vols is None:
        # Load volumes
        s = loadvol(basedir, ptcode, ctcode, cropname, 'phases', fname=fname)
        vols = []
        for key in dir(s):
            if key.startswith('vol'):
                vols.append(s[key])
        
    # Start vis
    f = vv.figure(1); vv.clf()
    f.position = 9.00, 38.00,  992.00, 944.00
    a = vv.gca()
    a.daspect = 1, 1, -1
    a.axis.axisColor = 1,1,1
    a.axis.visible = False
    a.bgcolor = 0,0,0
    if showVol=='mip':
        if not ptcode is None and not ctcode is None:
            vv.title('Maximum intensity projection cine-loop of the original ECG-gated CT volumes of patient %s at %s ' % (ptcode[8:], ctcode))
        else:
            vv.title('Maximum intensity projection cine-loop of the original ECG-gated CT volumes ')
    else:
        if not ptcode is None and not ctcode is None:
            vv.title('ECG-gated CT scan cine-loop of the original ECG-gated CT volumes of patient %s at %s ' % (ptcode[8:], ctcode))
        else:
            vv.title('ECG-gated CT scan cine-loop of the original ECG-gated CT volumes ')
    
    # Setup data container
    container = vv.MotionDataContainer(a)
    for vol in vols:
        if showVol == '2D':
            t = vv.volshow2(vol, clim=clim2D) # -750, 1000
            t.parent = container
        else:
            t = vv.volshow(vol, clim=clim, renderStyle = showVol)
            t.parent = container
            if showVol == 'iso':
                t.isoThreshold = isoTh    # iso or mip work well 
                t.colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
                            'g': [(0.0, 0.0), (0.27272728, 1.0)],
                            'b': [(0.0, 0.0), (0.34545454, 1.0)],
                            'a': [(0.0, 1.0), (1.0, 1.0)]}
            if mipIsocolor:
                t.colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
                            'g': [(0.0, 0.0), (0.27272728, 1.0)],
                            'b': [(0.0, 0.0), (0.34545454, 1.0)],
                            'a': [(0.0, 1.0), (1.0, 1.0)]}
    # bind ClimEditor to figure
    if slider:
        if showVol=='mip':
            c = vv.ClimEditor(vv.gcf())
            c.position = (10, 50)
            f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, c) )
        if showVol=='iso':
            c = IsoThEditor(vv.gcf())
            c.position = (10, 50)
            f.eventKeyDown.Bind(lambda event: _utils_GUI.ShowHideSlider(event, c) )
    
    f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a]) )
    print('------------------------')
    print('Use keys 1, 2, 3, 4 and 5 for preset anatomic views')
    print('Use v for a default zoomed view')
    print('Use x to show and hide axis')
    print('------------------------')
    
    return t
