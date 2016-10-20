import numpy as np
import visvis as vv

from stentseg.utils import PointSet
from stentseg.utils.centerline import find_centerline, points_from_mesh, smooth_centerline, pp_to_graph
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect import stentgraph

TEST = 14

if TEST == 1:
    import imageio
    im = imageio.imread('~/Desktop/test_centerline.png')[:,:,0]
    
    y, x = np.where(im < 200)
    pp = PointSet(np.column_stack([x, y]))

    start = (260, 60)
    ends = [(230, 510), (260, 510), (360, 510)]
    centerline = find_centerline(pp, start, ends, 8, ndist=20, regfactor=0.2, regsteps=10)
    
    vv.figure(1); vv.clf()
    a1 = vv.subplot(111)
    vv.plot(pp, ms='.', ls='')
    vv.plot(start[0], start[1],  ms='.', ls='', mc='g', mw=15)
    vv.plot([e[0] for e in ends], [e[1] for e in ends],  ms='.', ls='', mc='r', mw=15)
    
    vv.plot(centerline, ms='x', ls='', mw=15, mc='y')
    a1.daspectAuto = False

elif TEST > 10:
    
    # Select the ssdf basedir
    basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup', r'G:\LSPEAS_ssdf_backup')
    ptcode = 'LSPEAS_003'
    ctcode = 'discharge'
    cropname = 'stent'
    showAxis = False  # True or False
    showVol  = 'MIP'  # MIP or ISO or 2D or None
    clim0  = (0,2500)
    # clim0 = -550,500
    isoTh = 250
    s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg') 
    
    if TEST == 11:
        fname = r'D:\LSPEAS\LSPEAS_vessel\LSPEAS_002\12months\LSPEAS_002_MGK Smoothed_Wrapped_DRGseEditRG 2_001.stl'
        start1 = (110, 100, 70) # x,y,z ; dist
        start2 = (87, 106, 40)  # branch right
        ends = [(110, 120, 15)] # prox
    elif TEST == 12:
        fname = r'D:\LSPEAS\LSPEAS_vessel\LSPEAS_003\12months\LSPEAS_003_MGK Smoothed_Wrapped_DRGseEditRG 2_001.stl'
        start1 = (190, 165, 60)
        start2 = (207, 184, 34) 
        ends = [(179, 169, 17)] 
    elif TEST == 13:
        fname = r'D:\Profiles\koenradesma\Dropbox\UTdrive\MedDataMimics\LSPEAS_Mimics\LSPEAS_004\LSPEAS_004_D_stent-l-th500.stl'
        start1 = (146.1, 105.3, 69.3) # x,y,z
        ends = [(112.98, 100.08, 62.03)]
    elif TEST == 14:
        from stentseg.apps._3DPointSelector import select3dpoints
        points = select3dpoints(s.vol,nr_of_stents = 1)
        start1 = points[0][0]
        ends = points[1]
        print('Get Endpoints: done')
    else:
        raise RuntimeError('Invalid test')
    
    if TEST == 14:
        pp = points_from_nodes_in_graph(sd._nodes1)
    else:
        # Get pointset from STL, remove duplicates
        pp = points_from_mesh(fname, invertZ = True)
    
    # Find main centerline
    #regsteps = distance of centerline points from where the start/end point have no affect on centerline finding
    centerline1 = find_centerline(pp, start1, ends, 0.5, ndist=20, regfactor=0.2, regsteps=10, verbose=True)
    # centerline1 = find_centerline(pp, start1, ends, 2, ndist=100, regfactor=0.2, regsteps=10, verbose=True)
    
    # Find centerline of branch, using (last part of) main centerline as an end.
    centerline2 = find_centerline(pp, start2, centerline1[-100:], 0.5, ndist=20, regfactor=0.2, regsteps=10, verbose=True)
    # centerline2 = smooth_centerline(centerline2, 3)
    
    centerline_nodes = pp_to_graph(centerline1)
        
    f = vv.figure(1); vv.clf()
    f.position = 709.00, 30.00,  1203.00, 1008.00
    a1 = vv.subplot(121)
    vv.plot(pp, ms='.', ls='', alpha=0.2, mw = 7) # vessel
    vv.plot(PointSet(list(start1)), ms='.', ls='', mc='g', mw=18) # start1
    vv.plot(PointSet(list(start2)), ms='.', ls='', mc='m', mw=16) # start2
    vv.plot([e[0] for e in ends], [e[1] for e in ends],  [e[2] for e in ends],  ms='.', ls='', mc='r', mw=18) # ends
    vv.plot(centerline1, ms='.', ls='', mw=8, mc='y')
    vv.plot(centerline2, ms='.', ls='', mw=8, mc='c')
    a1.axis.visible = showAxis
    a1.daspect = 1,1,-1
    
    a2 = vv.subplot(122)
    show_ctvolume(s.vol, None, showVol=showVol, clim=clim0, isoTh=isoTh, removeStent=False)
    centerline_nodes.Draw(mc='y', mw=8, lc='g', lw=2)
    a2.axis.visible = showAxis
    a2.daspect = 1,1,-1
    
    a1.camera = a2.camera
    

