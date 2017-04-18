""" LSPEAS visualization module
Created April 18, 2017. Maaike Koenrades.
"""

import visvis as vv
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import show_ctvolume
import numpy as np

def showModelsStatic(ptcode,codes, vols, ss, mm, showVol, clim, isoTh, clim2, 
    clim2D, drawMesh=True, drawModelLines=True, showvol2D=False, showAxis=False):
    """ show one to four models in multipanel figure. 
    Input: arrays of codes, vols, ssdfs; params from show_models_static
    Output: axes, colorbars 
    """
    # init fig
    f = vv.figure(1); vv.clf()
    f.position = 0.00, 22.00,  1920.00, 1018.00
    if drawMesh == True:
        lc = 'w'
        mw = 10
    else:
        lc = 'g'
        mw = 7
    # create subplots
    if codes == (codes[0],codes[1]):
        a1 = vv.subplot(121)
        a2 = vv.subplot(122)
        axes = [a1,a2]
    elif codes == (codes[0],codes[1], codes[2]):
        a1 = vv.subplot(131)
        a2 = vv.subplot(132)
        a3 = vv.subplot(133)
        axes = [a1,a2,a3]
    elif codes == (codes[0],codes[1], codes[2], codes[3]):
        a1 = vv.subplot(221)
        a2 = vv.subplot(222)
        a3 = vv.subplot(223)
        a4 = vv.subplot(224)
        axes = [a1,a2,a3,a4]
    else:
        a1 = vv.subplot(111)
        axes = [a1]
    for i, ax in enumerate(axes):
        ax.MakeCurrent()
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], codes[i]))
        t = show_ctvolume(vols[i], ss[i].model, axis=ax, showVol=showVol, clim=clim, isoTh=isoTh, removeStent=True)
        label = pick3d(ax, vols[i])
        if drawModelLines == True:
            ss[i].model.Draw(mc='b', mw = mw, lc=lc)
    if showvol2D:
        for i, ax in enumerate(axes):
            t2 = vv.volshow2(vols[i], clim=clim2D, axes=ax)
    cbars = [] # colorbars
    if drawMesh == True:
        for i, ax in enumerate(axes):
            m = vv.mesh(mm[i], axes=ax)
            #m.faceColor = 'g' # OR
            m.clim = clim2
            m.colormap = vv.CM_JET
            cb = vv.colorbar(ax)
            cbars.append(cb)
    for ax in axes:
        ax.axis.axisColor = 1,1,1
        ax.bgcolor = 0,0,0
        ax.daspect = 1, 1, -1  # z-axis flipped
        ax.axis.visible = showAxis
    # set colorbar position
    for cbar in cbars:
        p1 = cbar.position
        cbar.position = (p1[0], 20, p1[2], 0.98) # x,y,w,h
    
    return axes, cbars
